"""
Optimized Training Script for DVS128 Gesture Dataset
- RAM-cached data loading (zero disk I/O during training)
- CuPy backend for fused CUDA kernels
- Mixed precision training (AMP)
- Large batch size (64) for better GPU utilization
- Supports pause/resume

Usage:
    # First, run preprocessing (one-time, ~2 min):
    python -m src.utils.preprocess_dvs128
    
    # Then train (~10+ it/s expected):
    python -m src.train_dvs
    
    # Resume training:
    python -m src.train_dvs --resume
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from src.model import SVT
from src.data.dvs_loader import RAMDVS128
from spikingjelly.activation_based import functional
from tqdm import tqdm
import argparse
import os
import logging

# Suppress verbose warnings
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


def collate_fn(batch):
    """Collate function for RAMDVS128 - stack and transpose to [T, B, C, H, W]."""
    frames_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    # Stack to [B, T, C, H, W] then transpose to [T, B, C, H, W]
    frames = torch.stack(frames_list).transpose(0, 1)
    labels = torch.tensor(labels_list, dtype=torch.long)
    return frames, labels


def train_dvs(resume: bool = False, use_cupy: bool = True, use_amp: bool = True):
    print("=" * 60)
    print("NEURO-SVT DVS128 GESTURE TRAINING")
    print("RAM-CACHED + CUPY + AMP")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters - tuned for RTX A4000 (16GB VRAM) with BPTT
    # Note: T=16 timesteps means batch_size effectively uses 16x more memory
    T = 16
    batch_size = 16  # Reduced for BPTT memory requirements
    epochs = 50
    lr = 1e-3
    reg_weight = 0.1
    
    # RAM-cached data loading
    data_root = './data/DVS128Gesture_Processed'
    os.makedirs('checkpoints', exist_ok=True)
    
    try:
        train_set = RAMDVS128(root=data_root, train=True)
        test_set = RAMDVS128(root=data_root, train=False)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return
    
    # DataLoader: num_workers=0 (data already in RAM), pin_memory=False
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=collate_fn
    )
    
    print(f"\nBatch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
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
    
    # ===== BACKEND VERIFICATION & WARMUP =====
    from spikingjelly.activation_based import neuron
    
    # 1. Force Multi-Step Mode (Crucial for CuPy speed)
    for m in model.modules():
        if isinstance(m, neuron.BaseNode):
            m.step_mode = 'm'  # Force multi-step
    
    # 2. Set Backend Explicitly
    if use_cupy and device.type == 'cuda':
        try:
            functional.set_backend(model, 'cupy', instance=neuron.BaseNode)
            print("Attempted to set CuPy backend.")
        except Exception as e:
            print(f"CuPy backend setup failed: {e}")
            use_cupy = False
    
    # 3. VERIFY Backend
    print("\n[INFO] Backend Verification:")
    layer_found = False
    for name, m in model.named_modules():
        if isinstance(m, neuron.LIFNode):
            print(f"  Layer: {name} | Backend: {m.backend} | Step Mode: {m.step_mode}")
            layer_found = True
            break
    if not layer_found:
        print("  Warning: No LIF layers found for verification.")
    
    print(f"\nDevice: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Mixed Precision (AMP): {'ENABLED' if use_amp else 'DISABLED'}")
    
    # 4. JIT Warmup
    print("\n[INFO] Warming up CUDA kernels (this might take 30s)...")
    dummy_input = torch.randn(16, 16, 2, 128, 128).to(device)  # [T, B, C, H, W]
    with torch.no_grad():
        _ = model(dummy_input)
    del dummy_input
    torch.cuda.empty_cache()
    print("[INFO] Warmup complete. Kernels compiled.")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = MixLoss(model, reg_weight=reg_weight)
    scaler = GradScaler('cuda', enabled=use_amp)
    
    start_epoch = 0
    best_acc = 0.0
    
    # Resume from checkpoint
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
    
    print("\n" + "=" * 80)
    print("Training started! (Ctrl+C to pause, --resume to continue)")
    print("=" * 80)
    
    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (x, y) in enumerate(pbar):
            try:
                x, y = x.to(device), y.to(device)
                
                functional.reset_net(model)
                
                # Mixed precision forward pass
                with autocast('cuda', enabled=use_amp):
                    outputs, _, _ = model(x)
                    loss, loss_ce, loss_reg = criterion(outputs, y)
                
                # Scaled backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pred = outputs.mean(0).argmax(dim=1)
                train_correct += (pred == y).sum().item()
                train_total += y.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * train_correct / train_total:.2f}%'
                })
            except Exception as e:
                print(f"\n[ERROR] Batch {batch_idx} failed: {e}")
                raise
        
        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                functional.reset_net(model)
                with autocast('cuda', enabled=use_amp):
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
        
        # Always save latest (for resume)
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
    
    torch.save(model.state_dict(), 'checkpoints/dvs_final.pt')
    print("Models saved: dvs_best.pt, dvs_latest.pt, dvs_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SVT on DVS128 Gesture (Optimized)')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--no-cupy', action='store_true', help='Disable CuPy backend')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    args = parser.parse_args()
    
    train_dvs(
        resume=args.resume,
        use_cupy=not args.no_cupy,
        use_amp=not args.no_amp
    )
