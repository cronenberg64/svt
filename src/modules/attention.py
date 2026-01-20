"""
Spike-Driven Sparse Attention (SDSA) Module

This module implements the core attention mechanism for the SVT architecture.
Key innovation: NO SOFTMAX - uses binary spike events for energy efficiency.

Reference:
- Spikformer (Zhou et al., 2023)
- Spike-driven Transformer V2 (Yao et al., 2024, ICLR)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from spikingjelly.activation_based import neuron, layer, functional


class SpikingSelfAttention(nn.Module):
    """
    Spike-Driven Sparse Attention (SDSA) - The Core Innovation
    
    Unlike standard attention which uses softmax:
        Attn = softmax(Q @ K^T / sqrt(d)) @ V
    
    SDSA uses spike-based binary masking:
        Q_spike, K_spike, V_spike = LIF(Q), LIF(K), LIF(V)
        Attn = Q_spike @ K_spike^T  (accumulation, no softmax!)
        Output = Attn @ V_spike
    
    This enables ~10x energy reduction by avoiding floating-point multiplications.
    
    Args:
        dim: Input embedding dimension
        num_heads: Number of attention heads (default: 8)
        qkv_bias: Whether to include bias in QKV projections (default: False)
        attn_drop: Dropout rate for attention weights (default: 0.0)
        proj_drop: Dropout rate for output projection (default: 0.0)
        tau: LIF neuron time constant (default: 2.0)
        v_threshold: Spike threshold voltage (default: 1.0)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        tau: float = 2.0,
        v_threshold: float = 1.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(d_k)
        
        # Q, K, V projections using spiking linear layers
        # Using standard Linear followed by LIF for spike generation
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # LIF neurons for generating binary spikes from Q, K, V
        # These convert continuous values to binary spike events (0 or 1)
        self.q_lif = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            detach_reset=True,
            step_mode='m',  # Multi-step mode for temporal processing
            backend='cupy' if torch.cuda.is_available() else 'torch',
        )
        self.k_lif = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            detach_reset=True,
            step_mode='m',
            backend='cupy' if torch.cuda.is_available() else 'torch',
        )
        self.v_lif = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            detach_reset=True,
            step_mode='m',
            backend='cupy' if torch.cuda.is_available() else 'torch',
        )
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_lif = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            detach_reset=True,
            step_mode='m',
            backend='cupy' if torch.cuda.is_available() else 'torch',
        )
        
        # Dropout layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Spike-Driven Sparse Attention.
        
        Args:
            x: Input tensor of shape (T, B, N, D) where:
               - T: Number of time steps
               - B: Batch size
               - N: Number of tokens (patches + memory token)
               - D: Embedding dimension
            mask: Optional attention mask (not typically used in SDSA)
            
        Returns:
            Tuple of:
                - Output tensor of shape (T, B, N, D)
                - Attention map of shape (T, B, num_heads, N, N) for visualization
        """
        T, B, N, D = x.shape
        
        # Project to Q, K, V (still continuous values)
        q = self.q_proj(x)  # (T, B, N, D)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Generate binary spikes through LIF neurons
        # This is where energy saving happens - outputs are {0, 1}
        q_spike = self.q_lif(q)  # (T, B, N, D) with binary values
        k_spike = self.k_lif(k)
        v_spike = self.v_lif(v)
        
        # Reshape for multi-head attention
        # (T, B, N, D) -> (T, B, N, num_heads, head_dim) -> (T, B, num_heads, N, head_dim)
        q_spike = q_spike.reshape(T, B, N, self.num_heads, self.head_dim).transpose(2, 3)
        k_spike = k_spike.reshape(T, B, N, self.num_heads, self.head_dim).transpose(2, 3)
        v_spike = v_spike.reshape(T, B, N, self.num_heads, self.head_dim).transpose(2, 3)
        
        # SDSA: Spike-based attention computation
        # Key insight: Q and K are BINARY, so Q @ K^T becomes accumulation!
        # This replaces expensive float multiplication with simple addition
        attn = torch.matmul(q_spike, k_spike.transpose(-2, -1))  # (T, B, num_heads, N, N)
        
        # Scale by 1/sqrt(d_k) for stability
        # Note: We scale AFTER matmul, not before, to preserve integer accumulation
        attn = attn * self.scale
        
        # NO SOFTMAX! This is the key difference from standard attention.
        # Instead, we rely on the sparsity of binary spikes for regularization.
        # Optionally apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, 0.0)
        
        # Apply dropout (optional during training)
        attn = self.attn_drop(attn)
        
        # Store attention map for visualization / debugging
        attn_map = attn.clone().detach()
        
        # Apply attention to V (also binary spikes)
        # Result is sparse due to binary inputs
        out = torch.matmul(attn, v_spike)  # (T, B, num_heads, N, head_dim)
        
        # Reshape back: (T, B, num_heads, N, head_dim) -> (T, B, N, D)
        out = out.transpose(2, 3).reshape(T, B, N, D)
        
        # Output projection with final LIF spike generation
        out = self.proj(out)
        out = self.proj_lif(out)
        out = self.proj_drop(out)
        
        return out, attn_map
    
    def reset(self):
        """Reset LIF neuron states for new sequence processing."""
        functional.reset_net(self)


class SpikingMLP(nn.Module):
    """
    Spiking MLP (Feed-Forward Network) for transformer blocks.
    
    Standard MLP: Linear -> GELU -> Linear
    Spiking MLP:  Linear -> LIF -> Linear -> LIF
    
    Uses binary spikes between layers for energy efficiency.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
        tau: float = 2.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.lif1 = neuron.LIFNode(
            tau=tau,
            detach_reset=True,
            step_mode='m',
            backend='cupy' if torch.cuda.is_available() else 'torch',
        )
        
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.lif2 = neuron.LIFNode(
            tau=tau,
            detach_reset=True,
            step_mode='m',
            backend='cupy' if torch.cuda.is_available() else 'torch',
        )
        
        self.drop = nn.Dropout(drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (T, B, N, D)
        Returns:
            Output tensor of shape (T, B, N, D)
        """
        x = self.fc1(x)
        x = self.lif1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.lif2(x)
        x = self.drop(x)
        return x


class SDSABlock(nn.Module):
    """
    Complete SDSA Transformer Block.
    
    Architecture:
        x -> LayerNorm -> SDSA -> Add -> LayerNorm -> MLP -> Add -> out
        
    Uses pre-normalization (more stable for SNNs).
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        tau: float = 2.0,
    ):
        super().__init__()
        
        # Pre-normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Spike-Driven Sparse Attention
        self.attn = SpikingSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            tau=tau,
        )
        
        # Spiking MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SpikingMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=proj_drop,
            tau=tau,
        )
        
        # Optional: DropPath for stochastic depth (not implemented for simplicity)
        self.drop_path_rate = drop_path
        
    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (T, B, N, D)
            return_attn: Whether to return attention maps
            
        Returns:
            Tuple of output tensor and optional attention map
        """
        # SDSA with residual connection
        residual = x
        x = self.norm1(x)
        attn_out, attn_map = self.attn(x)
        x = residual + attn_out
        
        # MLP with residual connection
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        
        if return_attn:
            return x, attn_map
        return x, None
    
    def reset(self):
        """Reset all LIF neuron states."""
        functional.reset_net(self)


# Utility function to create attention layers with different tau values
def create_dual_tau_attention(
    dim: int,
    num_heads: int = 8,
    spatial_tau: float = 1.1,
    memory_tau: float = 2.0,
) -> Tuple[SpikingSelfAttention, SpikingSelfAttention]:
    """
    Create two attention modules with different tau values.
    
    Args:
        dim: Embedding dimension
        num_heads: Number of attention heads
        spatial_tau: Fast decay for spatial tokens (default: 1.1)
        memory_tau: Slow decay for memory token (default: 2.0)
        
    Returns:
        Tuple of (spatial_attn, memory_attn) modules
    """
    spatial_attn = SpikingSelfAttention(dim=dim, num_heads=num_heads, tau=spatial_tau)
    memory_attn = SpikingSelfAttention(dim=dim, num_heads=num_heads, tau=memory_tau)
    return spatial_attn, memory_attn


if __name__ == "__main__":
    # Quick test
    print("Testing SpikingSelfAttention...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create attention module
    attn = SpikingSelfAttention(dim=256, num_heads=8, tau=2.0).to(device)
    
    # Test input: (T=4, B=2, N=16, D=256)
    x = torch.randn(4, 2, 16, 256).to(device)
    
    # Forward pass
    out, attn_map = attn(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Attn shape:   {attn_map.shape}")
    print(f"Output sparsity: {(out == 0).float().mean():.2%}")
    print("\n✓ SDSA test passed!")
