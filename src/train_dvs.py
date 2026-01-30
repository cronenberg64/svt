"""
Full Training Script for DVS128 Gesture Dataset
Supports pause/resume functionality via checkpoints.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.model import SVT
from src.data.dvs_loader import DVSGestureDataLoader
from src.utils.energy_meter import EnergyMeter
from spikingjelly.activation_based import functional
from tqdm import tqdm
import argparse
import time
import os
import logging

# Suppress verbose SpikingJelly warnings
logging.getLogger().setLevel(logging.ERROR)


class MixLoss(nn.Module):
    """CrossEntropy + Rate Regularization to penalize high firing rates."""
    def __init__(self, model, reg_weight=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.reg_weight = reg_weight
        self.firing_rates = []
        self.hooks = []
        
        from spikingjelly.activation_based import neuron
        for m in model.modules():
            if isinstance(m, (neuron.LIFNode, neuron.IFNode)):
                def hook(module, input, output):
                    self.firing_rates.append(output.mean())
                self.hooks.append(m.register_forward_hook(hook))
        
    def forward(self, outputs, target):
        loss_ce = self.ce(outputs.mean(0), target)
        if self.firing_rates:
            loss_reg = torch.stack(self.firing_rates).mean()
        else:
            loss_reg = torch.tensor(0.0).to(outputs.device)
        self.firing_rates = []
        return loss_ce + self.reg_weight * loss_reg, loss_ce, loss_reg

    def __del__(self):
        for h in self.hooks:
            h.remove()


def train_dvs(resume: bool = False):
    print("=" * 60)
    print("NEURO-SVT DVS128 GESTURE FULL TRAINING")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    T = 16
    batch_size = 16
    epochs = 50
    lr = 1e-3
    reg_weight = 0.1
    
    # Data Loaders
    print("Initializing DVS Loaders...")
    data_root = './data/DVS128Gesture'
    os.makedirs(data_root, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    train_wrapper = DVSGestureDataLoader(root=data_root, train=True, T=T, batch_size=batch_size, num_workers=0)
    test_wrapper = DVSGestureDataLoader(root=data_root, train=False, T=T, batch_size=batch_size, num_workers=0)
    train_loader = train_wrapper.get_dataloader()
    test_loader = test_wrapper.get_dataloader()
    
    print(f"Train samples: {len(train_wrapper.dataset)}")
    print(f"Test samples: {len(test_wrapper.dataset)}")
    
    # Model
    model = SVT(
        img_size=128,
        patch_size=4,
        in_channels=2,
        num_classes=11,
        embed_dim=192,
        depth=4,
        num_heads=3,
        spatial_tau=1.1,
        memory_tau=5.0
    ).to(device)
    
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = MixLoss(model, reg_weight=reg_weight)
    
    start_epoch = 0
    best_acc = 0.0
    
    # Resume from checkpoint if requested
    checkpoint_path = 'checkpoints/dvs_latest.pt'
    if resume and os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resuming from epoch {start_epoch}, best_acc={best_acc:.2f}%")
    elif resume:
        print("No checkpoint found, starting fresh.")
    
    print("\nStarting Training...")
    print("=" * 80)
    print("(Checkpoints saved after each epoch - you can resume anytime with --resume)")
    print("=" * 80)
    
    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            functional.reset_net(model)
            outputs, _, _ = model(x)
            loss, loss_ce, loss_reg = criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = outputs.mean(0).argmax(dim=1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * train_correct / train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", leave=False)
        with torch.no_grad():
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                
                functional.reset_net(model)
                outputs, _, _ = model(x)
                loss, _, _ = criterion(outputs, y)
                
                test_loss += loss.item()
                pred = outputs.mean(0).argmax(dim=1)
                test_correct += (pred == y).sum().item()
                test_total += y.size(0)
        
        test_loss /= len(test_loader)
        test_acc = 100.0 * test_correct / test_total
        
        # Save best model
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, 'checkpoints/dvs_best.pt')
        
        # Always save latest checkpoint (for resume)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
        }, 'checkpoints/dvs_latest.pt')
        
        best_marker = " *** BEST ***" if is_best else ""
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} / {train_acc:.2f}% | Test: {test_loss:.4f} / {test_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}{best_marker}")
        
        scheduler.step()
    
    print("=" * 80)
    print(f"Training complete! Best Test Accuracy: {best_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), 'checkpoints/dvs_final.pt')
    print("Models saved: dvs_best.pt, dvs_latest.pt, dvs_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SVT on DVS128 Gesture')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    args = parser.parse_args()
    
    train_dvs(resume=args.resume)
