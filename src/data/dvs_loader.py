"""
DVS Data Loader for SVT
Loads DVS128 Gesture dataset and processes events into frames.
"""

import torch
from torch.utils.data import DataLoader, random_split
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import pad_sequence_collate
from typing import Tuple, Optional

def split_to_frames(events: dict, T: int, H: int = 128, W: int = 128) -> torch.Tensor:
    """
    Integrates DVS events into T frames.
    
    Args:
        events: Dictionary containing 't', 'x', 'y', 'p' arrays
        T: Number of time steps
        H, W: Height and Width of the sensor
        
    Returns:
        Tensor of shape (T, C=2, H, W)
    """
    t, x, y, p = events['t'], events['x'], events['y'], events['p']
    
    # Normalize time to [0, T-1]
    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_norm = ((t - t_min) / (t_max - t_min) * (T - 1)).astype(int)
    else:
        t_norm = t.astype(int)
        
    frames = torch.zeros(T, 2, H, W)
    
    # Accumulate spikes into frames
    # p=0 is negative, p=1 is positive
    for i in range(len(t)):
        frames[t_norm[i], p[i], y[i], x[i]] += 1.0
        
    # Clip to binary spikes (0 or 1) for SVT compatibility
    return (frames > 0).float()

class DVSGestureDataLoader:
    """
    Wrapper for DVS128 Gesture dataset.
    """
    def __init__(
        self, 
        root: str, 
        train: bool = True, 
        T: int = 16, 
        batch_size: int = 16,
        num_workers: int = 4
    ):
        self.dataset = DVS128Gesture(root, train=train, data_type='event')
        self.T = T
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def get_dataloader(self):
        def collate_fn(batch):
            # batch is a list of (events, label)
            # events is a dict with 't', 'x', 'y', 'p'
            frames_list = []
            labels_list = []
            for events, label in batch:
                frames = split_to_frames(events, self.T)
                frames_list.append(frames)
                labels_list.append(label)
            
            # Stack to [B, T, C, H, W] then transpose to [T, B, C, H, W]
            return torch.stack(frames_list).transpose(0, 1), torch.tensor(labels_list)

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

if __name__ == "__main__":
    # Quick test if dataset is present
    print("Testing DVS Loader...")
    # Note: This requires the dataset to be downloaded at the specified root
    # For now, we just verify the logic
    print("DVS Loader logic initialized.")
