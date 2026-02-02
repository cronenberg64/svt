"""
One-time preprocessing script for DVS128 Gesture dataset.
Converts raw events to pre-computed frame tensors for 10-15x faster training.

Usage:
    python -m src.utils.preprocess_dvs128
"""
import torch
import numpy as np
import os
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from tqdm import tqdm


def split_to_frames(events: dict, T: int, H: int = 128, W: int = 128) -> torch.Tensor:
    """
    Vectorized event-to-frame conversion using np.bincount.
    Much faster than on-the-fly integration.
    """
    t, x, y, p = events['t'], events['x'], events['y'], events['p']
    
    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        t_norm = ((t - t_min) / (t_max - t_min) * (T - 1)).astype(np.int64)
    else:
        t_norm = np.zeros_like(t, dtype=np.int64)
    
    t_norm = np.clip(t_norm, 0, T - 1)
    x = np.clip(x.astype(np.int64), 0, W - 1)
    y = np.clip(y.astype(np.int64), 0, H - 1)
    p = p.astype(np.int64)
    
    flat_indices = t_norm * (2 * H * W) + p * (H * W) + y * W + x
    counts = np.bincount(flat_indices, minlength=T * 2 * H * W)
    frames = torch.from_numpy(counts.reshape(T, 2, H, W).astype(np.float32))
    
    return (frames > 0).float()


def preprocess_and_save(origin_dir: str, output_dir: str, T: int = 16, train: bool = True):
    """Process raw events and save as .pt tensors."""
    split_name = 'train' if train else 'test'
    print(f"\nProcessing {split_name} set...")
    
    # Load raw event dataset
    dataset = DVS128Gesture(root=origin_dir, train=train, data_type='event')
    
    save_path = os.path.join(output_dir, split_name)
    os.makedirs(save_path, exist_ok=True)
    
    for i, (events, label) in enumerate(tqdm(dataset, desc=f"{split_name}")):
        # Convert events to frames
        frames = split_to_frames(events, T=T)
        
        # Save as .pt tensor
        torch.save((frames, label), os.path.join(save_path, f"sample_{i:04d}.pt"))
    
    print(f"  Saved {len(dataset)} samples to {save_path}")


def main():
    # Config
    ORIGIN_DIR = './data/DVS128Gesture'
    OUTPUT_DIR = './data/DVS128Gesture_Processed'
    TIME_STEPS = 16
    
    print("=" * 60)
    print("DVS128 GESTURE PREPROCESSING")
    print("=" * 60)
    print(f"Source: {ORIGIN_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Time Steps: {TIME_STEPS}")
    
    if not os.path.exists(ORIGIN_DIR):
        print(f"\n❌ Error: Could not find dataset at {ORIGIN_DIR}")
        print("Please download the dataset first by running the training script once.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process both splits
    preprocess_and_save(ORIGIN_DIR, OUTPUT_DIR, T=TIME_STEPS, train=True)
    preprocess_and_save(ORIGIN_DIR, OUTPUT_DIR, T=TIME_STEPS, train=False)
    
    print("\n" + "=" * 60)
    print(f"✅ Done! Processed data saved to {OUTPUT_DIR}")
    print("=" * 60)
    print("\nYou can now run training with the fast loader:")
    print("  python -m src.train_dvs")


if __name__ == "__main__":
    main()
