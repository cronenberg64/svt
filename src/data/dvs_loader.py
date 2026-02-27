"""
DVS Data Loader for SVT
Loads DVS128 Gesture dataset and processes events into frames.
"""

import torch
import os
from torch.utils.data import DataLoader, Dataset, random_split
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import pad_sequence_collate
from tqdm import tqdm
from typing import Tuple, Optional
from torchvision.transforms import v2


class RAMDVS128(Dataset):
    """
    RAM-cached DVS128 dataset loader.
    Loads ALL pre-processed .pt files into RAM at startup.
    Zero disk I/O during training = blazing fast iteration.
    """
    def __init__(self, root: str, train: bool = True):
        self.train = train  # Store for Event Drop
        split_dir = os.path.join(root, 'train' if train else 'test')
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"Processed data not found at {split_dir}\n"
                "Run preprocessing first: python -m src.utils.preprocess_dvs128"
            )
        
        files = sorted([f for f in os.listdir(split_dir) if f.endswith('.pt')])
        
        # Load everything into RAM
        self.data = []
        total_bytes = 0
        
        print(f"Loading {'train' if train else 'test'} data into RAM...")
        for f in tqdm(files, desc="RAM Cache"):
            frames, label = torch.load(os.path.join(split_dir, f), weights_only=False)
            self.data.append((frames.float(), label))
            total_bytes += frames.numel() * 4  # float32 = 4 bytes
        
        gb = total_bytes / (1024 ** 3)
        print(f"✅ Loaded {len(self.data)} samples into RAM (~{gb:.2f} GB)")

        # Spatial Augmentation (Run 14)
        # Apply to (T, C, H, W) directly using v2
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=0, translate=(0.0625, 0.0625)), # +/- 8 pixels (8/128)
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frames, label = self.data[idx]
        
        # Event Drop: 10% random frame dropout (Training Only)
        if self.train:
            # Generate mask [T] -> [T, 1, 1, 1] for broadcasting
            # p=0.1 drop probability (keep p=0.9)
            T = frames.shape[0]
            mask = (torch.rand(T) > 0.1).float().view(-1, 1, 1, 1)
            mask = (torch.rand(T) > 0.1).float().view(-1, 1, 1, 1)
            frames = frames * mask
            
            # Spatial Augmentation (Run 14 - Fixed)
            # v2 transforms on (T, C, H, W) treat T as batch size (independent transforms).
            # We must flatten T into C to force SAME transform across time.
            T, C, H, W = frames.shape
            frames = frames.view(T * C, H, W)  # Treat time as extra channels
            frames = self.transforms(frames)
            frames = frames.view(T, C, H, W)   # Restore shape
            
        return frames, label


def split_to_frames(events: dict, T: int, H: int = 128, W: int = 128) -> torch.Tensor:
    """
    Integrates DVS events into T frames using vectorized operations.
    
    Args:
        events: Dictionary containing 't', 'x', 'y', 'p' arrays
        T: Number of time steps
        H, W: Height and Width of the sensor
        
    Returns:
        Tensor of shape (T, C=2, H, W)
    """
    import numpy as np
    
    t, x, y, p = events['t'], events['x'], events['y'], events['p']
    
    # Normalize time to [0, T-1]
    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_norm = ((t - t_min) / (t_max - t_min) * (T - 1)).astype(np.int64)
    else:
        t_norm = np.zeros_like(t, dtype=np.int64)
    
    # Clip coordinates to valid range
    t_norm = np.clip(t_norm, 0, T - 1)
    x = np.clip(x.astype(np.int64), 0, W - 1)
    y = np.clip(y.astype(np.int64), 0, H - 1)
    p = p.astype(np.int64)
    
    # Convert to flat indices for scatter_add: index = t*2*H*W + p*H*W + y*W + x
    flat_indices = t_norm * (2 * H * W) + p * (H * W) + y * W + x
    
    # Use bincount to accumulate (much faster than for-loop)
    counts = np.bincount(flat_indices, minlength=T * 2 * H * W)
    frames = torch.from_numpy(counts.reshape(T, 2, H, W).astype(np.float32))
    
    # Clip to binary spikes (0 or 1) for SVT compatibility
    return (frames > 0).float()


    def __getitem__(self, idx):
        return self.data[idx]

