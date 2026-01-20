"""
SVT Training Script

Skeleton training loop with a Toy Occlusion Dataset to validate
the Leaky Memory Token maintains object permanence during occlusion.
"""

import argparse
import random
import time
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Project imports
from src.model import SVT, svt_tiny, svt_small
from src.utils.energy_meter import estimate_energy, print_energy_report


# ============================================
# TOY OCCLUSION DATASET
# ============================================

class ToyOcclusionDataset(Dataset):
    """
    Synthetic dataset with moving shapes that get occluded.
    
    Generates sequences where:
    1. A shape appears and moves for T1 frames
    2. Shape is completely occluded for T2 frames (blank frames)
    3. Shape reappears at a predictable location for T3 frames
    
    The task: Classify the shape type (circle, square, triangle)
    after it reappears post-occlusion.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        img_size: int = 32,
        num_classes: int = 3,
        time_steps: int = 16,
        occlusion_start: int = 5,
        occlusion_duration: int = 6,
        noise_level: float = 0.1,
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.time_steps = time_steps
        self.occlusion_start = occlusion_start
        self.occlusion_duration = occlusion_duration
        self.noise_level = noise_level
        
        # Pre-generate all samples
        self.samples = []
        self.labels = []
        
        for _ in range(num_samples):
            sample, label = self._generate_sample()
            self.samples.append(sample)
            self.labels.append(label)
            
    def _generate_sample(self) -> Tuple[torch.Tensor, int]:
        """Generate a single sequence with occlusion."""
        # Random class: 0=circle, 1=square, 2=triangle
        label = random.randint(0, self.num_classes - 1)
        
        # Random starting position
        start_x = random.randint(4, self.img_size - 8)
        start_y = random.randint(4, self.img_size - 8)
        
        # Random velocity
        vx = random.choice([-1, 0, 1])
        vy = random.choice([-1, 0, 1])
        
        # Shape size
        size = random.randint(3, 6)
        
        # Generate frames
        frames = torch.zeros(self.time_steps, 1, self.img_size, self.img_size)
        
        occlusion_end = self.occlusion_start + self.occlusion_duration
        
        for t in range(self.time_steps):
            # Check if occluded
            if self.occlusion_start <= t < occlusion_end:
                # Blank frame during occlusion
                continue
                
            # Calculate position
            x = start_x + vx * t
            y = start_y + vy * t
            
            # Clamp to image bounds
            x = max(size, min(self.img_size - size - 1, x))
            y = max(size, min(self.img_size - size - 1, y))
            
            # Draw shape
            frame = torch.zeros(self.img_size, self.img_size)
            
            if label == 0:  # Circle
                for i in range(-size, size + 1):
                    for j in range(-size, size + 1):
                        if i * i + j * j <= size * size:
                            px, py = x + i, y + j
                            if 0 <= px < self.img_size and 0 <= py < self.img_size:
                                frame[py, px] = 1.0
                                
            elif label == 1:  # Square
                for i in range(-size, size + 1):
                    for j in range(-size, size + 1):
                        px, py = x + i, y + j
                        if 0 <= px < self.img_size and 0 <= py < self.img_size:
                            frame[py, px] = 1.0
                            
            elif label == 2:  # Triangle
                for i in range(-size, size + 1):
                    for j in range(abs(i) - size, size + 1):
                        px, py = x + i, y + j
                        if 0 <= px < self.img_size and 0 <= py < self.img_size:
                            frame[py, px] = 1.0
                            
            frames[t, 0] = frame
            
        # Add noise
        frames = frames + torch.randn_like(frames) * self.noise_level
        frames = frames.clamp(0, 1)
        
        return frames, label
        
    def __len__(self) -> int:
        return self.num_samples
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.samples[idx], self.labels[idx]


# ============================================
# TRAINING UTILITIES
# ============================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device)  # (B, T, C, H, W) or (T, B, C, H, W)
        target = target.to(device)
        
        # Transpose if needed: (B, T, C, H, W) -> (T, B, C, H, W)
        if data.dim() == 5 and data.shape[0] != data.shape[1]:
            data = data.transpose(0, 1)
            
        # Reset neuron states for new sequence
        model.reset()
        
        # Forward pass
        optimizer.zero_grad()
        output, _, _ = model(data)
        
        # Compute loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.1f}%'
        })
        
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)
            
            if data.dim() == 5 and data.shape[0] != data.shape[1]:
                data = data.transpose(0, 1)
                
            model.reset()
            output, _, _ = model(data)
            
            loss = criterion(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def test_occlusion_recovery(
    model: nn.Module,
    device: torch.device,
    num_tests: int = 50,
    occlusion_frames: int = 30,
) -> float:
    """
    Test the Permanence Score: accuracy after long occlusion.
    
    This specifically tests if the Memory Token can maintain
    object information during extended occlusion periods.
    """
    model.eval()
    
    # Create challenging dataset with long occlusion
    dataset = ToyOcclusionDataset(
        num_samples=num_tests,
        time_steps=occlusion_frames + 10,  # Extra frames before/after
        occlusion_start=5,
        occlusion_duration=occlusion_frames,
    )
    
    correct = 0
    
    with torch.no_grad():
        for i in range(num_tests):
            data, target = dataset[i]
            data = data.unsqueeze(1).to(device)  # (T, 1, C, H, W)
            
            model.reset()
            output, memory, _ = model(data, return_memory=True)
            
            pred = output.argmax(dim=-1)
            if pred.item() == target:
                correct += 1
                
    permanence_score = correct / num_tests
    return permanence_score


# ============================================
# MAIN TRAINING LOOP
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Train SVT on Toy Occlusion Dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--model', type=str, default='tiny', choices=['tiny', 'small', 'base'])
    parser.add_argument('--time-steps', type=int, default=16, help='Temporal steps')
    parser.add_argument('--occlusion-duration', type=int, default=6, help='Occlusion frame count')
    parser.add_argument('--train-samples', type=int, default=2000, help='Training samples')
    parser.add_argument('--test-samples', type=int, default=500, help='Test samples')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode (1 batch)')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    
    # Device setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model_factories = {
        'tiny': svt_tiny,
        'small': svt_small,
        'base': lambda **kw: SVT(**kw),
    }
    model = model_factories[args.model](
        img_size=32,
        patch_size=4,
        in_channels=1,
        num_classes=3,  # circle, square, triangle
    ).to(device)
    print("DEBUG: Model created")
    
    # print(f"\nModel: SVT-{args.model.capitalize()}")
    # print(f"Parameters: {model.count_parameters():,}")
    # print(f"Spatial τ: {model.spatial_tau}, Memory τ: {model.memory_tau}")
    
    # Create datasets
    print("\nGenerating toy occlusion datasets...")
    train_dataset = ToyOcclusionDataset(
        num_samples=args.train_samples if not args.debug else 64,
        time_steps=args.time_steps,
        occlusion_duration=args.occlusion_duration,
    )
    test_dataset = ToyOcclusionDataset(
        num_samples=args.test_samples if not args.debug else 32,
        time_steps=args.time_steps,
        occlusion_duration=args.occlusion_duration,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # TensorBoard
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(save_dir / 'logs')
    
    # Training loop
    best_acc = 0.0
    print("\n" + "=" * 60)
    print("TRAINING SVT")
    print("=" * 60)
    
    epochs = 1 if args.debug else args.epochs
    print("DEBUG: Starting training loop")
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f"DEBUG: Epoch {epoch} train complete")
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.1f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, save_dir / 'best_model.pt')
            print(f"  ✓ New best model saved (acc: {test_acc:.1f}%)")
            
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    # Test Permanence Score (30+ frame occlusion)
    permanence_score = test_occlusion_recovery(
        model, device, num_tests=50, occlusion_frames=30
    )
    print(f"\nPermanence Score (30-frame occlusion): {permanence_score:.1%}")
    
    if permanence_score >= 0.85:
        print("✓ GOAL MET: ≥85% recovery after 30+ frame occlusion!")
    else:
        print(f"✗ Goal not met: {permanence_score:.1%} < 85% target")
        
    # Energy report
    print("\n" + "-" * 60)
    sample_input = torch.randn(args.time_steps, 1, 1, 32, 32).to(device)
    model.reset()
    report = estimate_energy(model, sample_input)
    print_energy_report(report)
    
    writer.close()
    print(f"\nTraining complete! Best accuracy: {best_acc:.1f}%")
    print(f"Checkpoints saved to: {save_dir.absolute()}")


if __name__ == "__main__":
    main()
