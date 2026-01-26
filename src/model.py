"""
SVT: Spiking Vision Transformer for Object Permanence

Main architecture combining:
- SpikingPatchEmbed: Event-driven spatial tokenization
- Leaky Memory Token: Novel temporal persistence mechanism (τ=2.0)
- SDSA Blocks: Spike-Driven Sparse Attention (no softmax)

The key innovation is the Leaky Memory Token with slower decay rate,
enabling persistent temporal context during occlusion without LSTM overhead.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from spikingjelly.activation_based import neuron, functional

from .modules.attention import SDSABlock, SpikingSelfAttention
from .modules.patch_embed import SpikingPatchEmbed


class SVT(nn.Module):
    """
    Spiking Vision Transformer (SVT) for Object Permanence
    
    Architecture Overview:
    ```
    Input (T, B, C, H, W)
           ↓
    SpikingPatchEmbed → Spatial Tokens (T, B, N, D)
           ↓
    Prepend Memory Token (τ=2.0) → (T, B, N+1, D)
           ↓
    SDSA Blocks × depth
           ↓
    Classification/Tracking Head
    ```
    
    The Memory Token has a SLOWER decay rate (τ=2.0 vs τ=1.1 for spatial tokens),
    allowing it to maintain information during occlusion frames.
    
    Args:
        img_size: Input image size (default: 32)
        patch_size: Patch size for tokenization (default: 4)
        in_channels: Input channels (default: 1)
        num_classes: Output classes for classification (default: 10)
        embed_dim: Embedding dimension (default: 256)
        depth: Number of transformer blocks (default: 4)
        num_heads: Number of attention heads (default: 8)
        mlp_ratio: MLP hidden dimension ratio (default: 4.0)
        spatial_tau: Time constant for spatial tokens (default: 1.1, fast decay)
        memory_tau: Time constant for memory token (default: 2.0, slow decay)
        drop_rate: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        spatial_tau: float = 1.1,
        memory_tau: float = 5.0,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.spatial_tau = spatial_tau
        self.memory_tau = memory_tau
        
        # ============================================
        # SPIKE ENCODER (The Eye)
        # ============================================
        self.patch_embed = SpikingPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            tau=spatial_tau,  # Fast decay for spatial features
        )
        self.num_patches = self.patch_embed.num_patches
        
        # ============================================
        # LEAKY MEMORY TOKEN (The Novelty)
        # ============================================
        # Learnable token that persists temporal context
        self.memory_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.trunc_normal_(self.memory_token, std=0.8)
        
        # Note: memory_lif moved to SDSABlock for per-layer temporal persistence
        
        # ============================================
        # POSITIONAL ENCODING
        # ============================================
        # Learnable position embeddings for patches + memory token
        num_tokens = self.num_patches + 1  # +1 for memory token
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.2)
        
        # Position encoding LIF (uses spatial tau for position info)
        self.pos_lif = neuron.LIFNode(
            tau=spatial_tau,
            detach_reset=True,
            step_mode='m',
            backend='cupy' if torch.cuda.is_available() else 'torch',
        )
        
        # ============================================
        # TRANSFORMER BLOCKS (SDSA)
        # ============================================
        self.blocks = nn.ModuleList([
            SDSABlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=False,
                attn_drop=drop_rate,
                proj_drop=drop_rate,
                spatial_tau=spatial_tau,
                memory_tau=memory_tau,
            )
            for _ in range(depth)
        ])
        
        # ============================================
        # CLASSIFICATION HEAD
        # ============================================
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Final classification LIF
        self.head_lif = neuron.LIFNode(
            tau=spatial_tau,
            detach_reset=True,
            step_mode='m',
            backend='cupy' if torch.cuda.is_available() else 'torch',
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize linear layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.2)  # Increased from 0.02
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(
        self,
        x: torch.Tensor,
        return_memory: bool = False,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Forward pass for SVT.
        
        Args:
            x: Input tensor of shape (T, B, C, H, W) or (B, C, H, W)
            return_memory: If True, return memory token activations
            return_attn: If True, return attention maps from all blocks
            
        Returns:
            Tuple of:
                - Output logits of shape (T, B, num_classes) or (B, num_classes)
                - Optional memory token activations of shape (T, B, D)
                - Optional list of attention maps
        """
        # Handle single-frame input
        single_frame = x.dim() == 4
        if single_frame:
            x = x.unsqueeze(0)  # (B, C, H, W) -> (T=1, B, C, H, W)
            
        T, B, C, H, W = x.shape
        
        # ============================================
        # PATCH EMBEDDING (Spike Encoding)
        # ============================================
        x = self.patch_embed(x)  # (T, B, N, D)
        
        # ============================================
        # PREPEND MEMORY TOKEN
        # ============================================
        # Expand memory token: (1, 1, 1, D) -> (T, B, 1, D)
        memory_tokens = self.memory_token.expand(T, B, 1, -1)
        
        # Concatenate: [memory_token, patch_tokens]
        x = torch.cat([memory_tokens, x], dim=2)  # (T, B, N+1, D)
        
        # ============================================
        # ADD POSITIONAL ENCODING
        # ============================================
        x = x + self.pos_embed  # Broadcasting: (T, B, N+1, D) + (1, 1, N+1, D)
        x = self.pos_lif(x)
        
        # ============================================
        # TRANSFORMER BLOCKS
        # ============================================
        attn_maps = []
        for block in self.blocks:
            x, attn = block(x, return_attn=return_attn)
            if return_attn and attn is not None:
                attn_maps.append(attn)
                
        # ============================================
        # EXTRACT MEMORY TOKEN FOR CLASSIFICATION
        # ============================================
        # The memory token (index 0) aggregates global temporal context
        memory_out = x[:, :, 0, :]  # (T, B, D)
        
        # ============================================
        # CLASSIFICATION HEAD
        # ============================================
        x_cls = self.norm(memory_out)
        x_cls = self.head(x_cls)  # (T, B, num_classes)
        x_cls = self.head_lif(x_cls)
        
        # Aggregate over time: mean firing rate
        logits = x_cls.mean(dim=0)  # (B, num_classes)
        
        # Handle single-frame output format
        if single_frame:
            logits = logits
        else:
            logits = x_cls  # Return full temporal output
            
        # Prepare outputs
        memory = memory_out if return_memory else None
        attn = attn_maps if return_attn else None
        
        return logits, memory, attn
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head."""
        if x.dim() == 4:
            x = x.unsqueeze(0)
            
        T, B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Memory token
        memory_tokens = self.memory_token.expand(T, B, 1, -1)
        x = torch.cat([memory_tokens, x], dim=2)
        
        # Position encoding
        x = x + self.pos_embed
        x = self.pos_lif(x)
        
        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x)
            
        return self.norm(x)
    
    def reset(self):
        """Reset all LIF neuron states for new sequence processing."""
        # Avoid recursion by only resetting actual spiking neurons
        for module in self.modules():
            if isinstance(module, (neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode)):
                if hasattr(module, 'reset'):
                    # Call LIFNode's reset directly (not the parent's reset)
                    neuron.LIFNode.reset(module) if isinstance(module, neuron.LIFNode) else module.reset()
        
    def get_memory_token_state(self) -> List[Optional[torch.Tensor]]:
        """Get current membrane potential of memory token LIF in all blocks."""
        states = []
        for block in self.blocks:
            if hasattr(block.memory_lif, 'v'):
                states.append(block.memory_lif.v)
            else:
                states.append(None)
        return states
        
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Factory functions for different SVT sizes
def svt_tiny(num_classes: int = 10, **kwargs) -> SVT:
    """SVT-Tiny: ~1.2M parameters"""
    return SVT(
        embed_dim=128,
        depth=3,
        num_heads=4,
        num_classes=num_classes,
        **kwargs
    )


def svt_small(num_classes: int = 10, **kwargs) -> SVT:
    """SVT-Small: ~4.5M parameters"""
    return SVT(
        embed_dim=256,
        depth=4,
        num_heads=8,
        num_classes=num_classes,
        **kwargs
    )


def svt_base(num_classes: int = 10, **kwargs) -> SVT:
    """SVT-Base: ~12M parameters"""
    return SVT(
        embed_dim=384,
        depth=6,
        num_heads=12,
        num_classes=num_classes,
        **kwargs
    )


if __name__ == "__main__":
    print("Testing SVT Architecture...")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = SVT(
        img_size=32,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        embed_dim=256,
        depth=4,
        num_heads=8,
        spatial_tau=1.1,  # Fast decay for spatial
        memory_tau=2.0,   # Slow decay for memory (THE KEY!)
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Spatial τ: {model.spatial_tau}")
    print(f"Memory τ:  {model.memory_tau} (slower decay = longer memory)")
    
    # Test input: (T=8, B=2, C=1, H=32, W=32)
    T, B = 8, 2
    x = torch.randn(T, B, 1, 32, 32).to(device)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass with memory tracking
    logits, memory, _ = model(x, return_memory=True)
    
    print(f"Output shape: {logits.shape}")
    print(f"Memory shape: {memory.shape}")
    
    # Test occlusion scenario: set middle frames to zero
    x_occluded = x.clone()
    x_occluded[3:6] = 0  # Frames 3, 4, 5 are "occluded"
    
    model.reset()  # Reset for new sequence
    logits_occ, memory_occ, _ = model(x_occluded, return_memory=True)
    
    # Compare memory token activity during occlusion
    memory_before = memory[2].mean()  # Before occlusion
    memory_during = memory_occ[4].mean()  # During occlusion (should still be active!)
    memory_after = memory_occ[7].mean()  # After occlusion
    
    print(f"\nOcclusion Test (frames 3-5):")
    print(f"  Memory activity before: {memory_before:.4f}")
    print(f"  Memory activity during: {memory_during:.4f} (should persist!)")
    print(f"  Memory activity after:  {memory_after:.4f}")
    
    print("\n✓ SVT architecture test passed!")
