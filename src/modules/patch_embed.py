"""
Spiking Patch Embedding Module

Converts input images/event streams into spike-encoded patch tokens.
Uses Conv2D + LIF for efficient spatial tokenization.
"""

import torch
import torch.nn as nn
from typing import Tuple

from spikingjelly.activation_based import neuron, functional


class SpikingPatchEmbed(nn.Module):
    """
    Spiking Patch Embedding Layer (The Eye)
    
    Tokenizes input images into a sequence of spiking patches.
    
    Standard PatchEmbed: Conv2D -> Flatten
    Spiking PatchEmbed:  Conv2D -> LIF -> Flatten (binary spikes)
    
    Args:
        img_size: Input image size (default: 32)
        patch_size: Patch size (default: 4)
        in_channels: Number of input channels (default: 1 for grayscale)
        embed_dim: Output embedding dimension (default: 256)
        tau: LIF time constant (default: 2.0)
    """
    
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 256,
        tau: float = 2.0,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
        # Convolutional projection: (C, H, W) -> (D, H/P, W/P)
        # Uses kernel_size = stride = patch_size for non-overlapping patches
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        
        # LIF neuron for spike encoding
        self.lif = neuron.LIFNode(
            tau=tau,
            detach_reset=True,
            step_mode='m',  # Multi-step for temporal processing
            backend='cupy' if torch.cuda.is_available() else 'torch',
        )
        
        # Layer normalization (optional, applied after flattening)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spiking patch embedding.
        
        Args:
            x: Input tensor of shape:
               - (T, B, C, H, W) for temporal input
               - (B, C, H, W) for single frame (will add T=1 dimension)
               
        Returns:
            Spike-encoded patches of shape (T, B, N, D) where:
            - N = num_patches (grid_size^2)
            - D = embed_dim
        """
        # Handle both temporal and single-frame inputs
        if x.dim() == 4:
            # Single frame: (B, C, H, W) -> (T=1, B, C, H, W)
            x = x.unsqueeze(0)
            
        T, B, C, H, W = x.shape
        
        # Reshape for batch processing: (T, B, C, H, W) -> (T*B, C, H, W)
        x = x.reshape(T * B, C, H, W)
        
        # Convolutional projection: (T*B, C, H, W) -> (T*B, D, H/P, W/P)
        x = self.proj(x)
        
        # Flatten spatial dimensions: (T*B, D, H/P, W/P) -> (T*B, D, N)
        x = x.flatten(2)  # (T*B, D, N)
        
        # Transpose to (T*B, N, D) for sequence processing
        x = x.transpose(1, 2)  # (T*B, N, D)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Reshape back to temporal format: (T*B, N, D) -> (T, B, N, D)
        x = x.reshape(T, B, self.num_patches, self.embed_dim)
        
        # Generate spikes through LIF neuron
        x = self.lif(x)
        
        return x
    
    def get_grid_size(self) -> Tuple[int, int]:
        """Return the spatial grid dimensions of patches."""
        return (self.grid_size, self.grid_size)


class SpikingConvStem(nn.Module):
    """
    Alternative spike encoder using multiple conv layers.
    
    Provides a more gradual spatial reduction for better feature extraction.
    Architecture: Conv -> LIF -> Conv -> LIF -> Flatten
    """
    
    def __init__(
        self,
        img_size: int = 32,
        in_channels: int = 1,
        embed_dim: int = 256,
        tau: float = 2.0,
    ):
        super().__init__()
        
        self.img_size = img_size
        hidden_dim = embed_dim // 2
        
        # Stage 1: 32x32 -> 16x16
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.lif1 = neuron.LIFNode(tau=tau, detach_reset=True, step_mode='m')
        
        # Stage 2: 16x16 -> 8x8
        self.conv2 = nn.Conv2d(hidden_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.lif2 = neuron.LIFNode(tau=tau, detach_reset=True, step_mode='m')
        
        # Flatten for sequence
        self.num_patches = (img_size // 4) ** 2  # 8x8 = 64 patches for 32x32 input
        self.grid_size = img_size // 4
        self.embed_dim = embed_dim
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (T, B, C, H, W)
        Returns:
            Output of shape (T, B, N, D)
        """
        if x.dim() == 4:
            x = x.unsqueeze(0)
            
        T, B, C, H, W = x.shape
        x = x.reshape(T * B, C, H, W)
        
        # Stage 1
        x = self.conv1(x)
        x = x.reshape(T, B, *x.shape[1:])  # (T, B, C, H, W)
        x = self.lif1(x)
        x = x.reshape(T * B, *x.shape[2:])  # (T*B, C, H, W)
        
        # Stage 2
        x = self.conv2(x)
        x = x.reshape(T, B, *x.shape[1:])
        x = self.lif2(x)
        x = x.reshape(T * B, *x.shape[2:])
        
        # Flatten and normalize
        x = x.flatten(2).transpose(1, 2)  # (T*B, N, D)
        x = self.norm(x)
        x = x.reshape(T, B, self.num_patches, self.embed_dim)
        
        return x
    

if __name__ == "__main__":
    print("Testing SpikingPatchEmbed...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create patch embedding
    patch_embed = SpikingPatchEmbed(
        img_size=32,
        patch_size=4,
        in_channels=1,
        embed_dim=256,
    ).to(device)
    
    # Test input: (T=4, B=2, C=1, H=32, W=32)
    x = torch.randn(4, 2, 1, 32, 32).to(device)
    
    # Forward pass
    out = patch_embed(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Num patches:  {patch_embed.num_patches}")
    print(f"Grid size:    {patch_embed.get_grid_size()}")
    print(f"Output sparsity: {(out == 0).float().mean():.2%}")
    print("\n✓ Patch embedding test passed!")
