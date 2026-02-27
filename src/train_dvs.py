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
from src.modules.model import SVT
from src.data.dvs_loader import RAMDVS128
from spikingjelly.activation_based import functional
from tqdm import tqdm
import argparse
import os
import math
import logging
import matplotlib.pyplot as plt
import numpy as np

# Suppress verbose warnings
logging.getLogger().setLevel(logging.ERROR)


class MixLoss(nn.Module):
    """CrossEntropy + Rate Regularization to penalize high firing rates."""
    def __init__(self, model, reg_weight=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.reg_weight = reg_weight
        self.firing_rates = []
        self.last_rates = []
        self.hooks = []
        
        from spikingjelly.activation_based import neuron
        for m in model.modules():
            if isinstance(m, (neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode)):
                def hook(module, input, output):
                    self.firing_rates.append(output.mean())
                self.hooks.append(m.register_forward_hook(hook))
        
    def forward(self, outputs, target):
        # Spikformer output is already [B, num_classes] (logits)
        # Note: self.ce is standard CrossEntropyLoss
        loss_ce = self.ce(outputs, target)
        if self.firing_rates:
            loss_reg = torch.stack(self.firing_rates).mean()
        else:
            loss_reg = torch.tensor(0.0).to(outputs.device)
            
        # Store for visualization (detach to save memory)
        self.last_rates = [fr.detach().cpu().item() for fr in self.firing_rates]
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


def log_vital_signs(model, epoch, mix_loss):
    """Log Gradient Norms and Firing Rate Histogram."""
    print(f"\n[VITAL SIGNS] Logging for Epoch {epoch}...")
    
    # 1. Gradient Norms
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            norm = p.grad.norm().item()
            norms.append(norm)
    
    # 2. Firing Rates
    rates = mix_loss.last_rates
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ax1: Gradient Norms (Boxplot or Scatter)
    ax1.plot(norms, alpha=0.6)
    ax1.set_title("Gradient Norms per Parameter")
    ax1.set_xlabel("Parameter Index")
    ax1.set_ylabel("L2 Norm")
    ax1.set_yscale('log')
    
    # Ax2: Firing Rates Histogram
    if rates:
        ax2.hist(rates, bins=20, range=(0, 1), alpha=0.7, color='green')
        ax2.set_title("Firing Rate Distribution (All Layers)")
        ax2.set_xlabel("Mean Firing Rate")
        ax2.set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(f'results/vital_signs_epoch_{epoch}.png')
    plt.close()


def visualize_permanence(model, test_loader, epoch, device):
    """Visualize Memory Token Evolution over Time."""
    # print(f"\n[VIZ] Generating Object Permanence Check for Epoch {epoch}...")
    model.eval()
    
    # 1. Enable Sequence Recording for Memory Nodes
    for block in model.blocks:
        block.memory_lif.store_v_seq = True
        
    # 2. Get one sample
    try:
        x, _ = next(iter(test_loader))
        x = x.to(device)
        # 3. Forward pass (on full batch, but we analyze first sample)
        with torch.no_grad():
            functional.reset_net(model)
            model(x)
            
        # 4. Extract Memory Traces
        traces = []
        for i, block in enumerate(model.blocks):
            # v is [T, B, D] or [B, D] depending on store_v_seq
            v = block.memory_lif.v
            if v.dim() == 3: # [T, B, D]
                 # Take first sample in batch: [T, D]
                 traces.append(v[:, 0, :].cpu().numpy())
            
        # 5. Plot
        if traces:
            fig, axes = plt.subplots(len(traces), 1, figsize=(10, 2*len(traces)), sharex=True)
            if len(traces) == 1: axes = [axes]
            
            for i, ax in enumerate(axes):
                # Plot Heatmap: Time (X) vs Dimensions (Y)
                # Transpose trace to [D, T]
                im = ax.imshow(traces[i].T, aspect='auto', cmap='magma', interpolation='nearest', vmin=0, vmax=1.0)
                ax.set_ylabel(f'Layer {i}')
                if i == 0:
                    ax.set_title(f"Memory Token Potentials (Epoch {epoch}) - Sample 0")
                if i == len(traces)-1:
                    ax.set_xlabel('Time Step')
            
            plt.tight_layout()
            plt.savefig(f'results/live_permanence_check_epoch_{epoch}.png')
            plt.close()
            
    except Exception as e:
        print(f"[VIZ WARN] Failed to visualize permanence: {e}")
    finally:
        # 6. Cleanup (Disable Seq Recording)
        for block in model.blocks:
            block.memory_lif.store_v_seq = False


def train_dvs(resume: bool = False, use_cupy: bool = True, use_amp: bool = True, debug: bool = False):
    print("=" * 60)
    print("NEURO-SVT DVS128 GESTURE TRAINING")
    print("RAM-CACHED + CUPY + AMP")
    if debug:
        print(" DEBUG MODE ENABLED: No Reg, Data Stats, Firing Rates")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters - tuned for RTX A4000 (16GB VRAM) with BPTT
    # Note: T=16 timesteps means batch_size effectively uses 16x more memory
    T = 16
    batch_size = 16  # Reduced for BPTT memory requirements
    batch_size = 16  # Reduced for BPTT memory requirements
    epochs = 100 if not debug else 5  # Run 14 "Final Surge" - 100 Epochs
    
    # Production Strategy: Warmup + Cosine Annealing
    warmup_epochs = 5
    peak_lr = 1e-3  # "High Velocity"
    min_lr = 1e-6
    weight_decay = 0.01
    reg_weight = 0.0  # NO regularization for Golden Standard

    
    # RAM-cached data loading
    data_root = './data/DVS128Gesture_Processed'
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    try:
        train_set = RAMDVS128(root=data_root, train=True)
        test_set = RAMDVS128(root=data_root, train=False)
    except FileNotFoundError as e:
        print(f"\n {e}")
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
        spatial_tau=2.0,  # Increased from 1.1 for better memory (decay ~0.5 vs ~0.09)
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
            functional.set_backend(model, 'cupy', instance=(neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode))
            print("Attempted to set CuPy backend.")
        except Exception as e:
            print(f"CuPy backend setup failed: {e}")
            use_cupy = False
    
    # 3. VERIFY Backend
    print("\n[INFO] Backend Verification:")
    layer_found = False
    for name, m in model.named_modules():
        if isinstance(m, (neuron.LIFNode, neuron.ParametricLIFNode)):
            print(f"  Layer: {name} | Backend: {m.backend} | Step Mode: {m.step_mode}")
            layer_found = True
            break
    if not layer_found:
        print("  Warning: No LIF/PLIF layers found for verification.")
    
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
    
    # Production Strategy: AdamW + Linear Warmup + Cosine Annealing
    optimizer = optim.AdamW(model.parameters(), lr=min_lr, weight_decay=weight_decay)
    
    def lr_lambda(epoch):
        """Linear warmup for warmup_epochs, then cosine annealing."""
        if epoch < warmup_epochs:
            # Linear warmup: 0 -> 1 over warmup_epochs
            return (peak_lr / min_lr) * (epoch / warmup_epochs)
        else:
            # Cosine annealing from peak_lr to min_lr
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return (peak_lr / min_lr) * (1 + math.cos(math.pi * progress)) / 2
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
        # For Run 13: RESET SCHEDULER to force Warm Start
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # start_epoch = checkpoint['epoch'] # Optional: Resume epoch count or reset? 
        # Better to resume epoch count so logs are continuous (101-200)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resuming from epoch {start_epoch}, best_acc={best_acc:.2f}%")
        print(" Scheduler Reset: Learning Rate spiked to initial value.")
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
                    
                    if debug and batch_idx == 0:
                        print(f"\n[DEBUG] Batch 0 Stats:")
                        print(f"  Input: shape={x.shape}, mean={x.mean():.4f}, max={x.max():.4f}, unique={len(torch.unique(x))}")
                        print(f"  Output: shape={outputs.shape}, mean={outputs.mean():.4f}, max={outputs.max():.4f}")
                        
                        # Check firing rates manually
                        print("  Firing Rates (Frame 0):")
                        for name, m in model.named_modules():
                            from spikingjelly.activation_based import neuron
                            # Check backend for LIF neurons
                            if isinstance(m, (neuron.LIFNode, neuron.ParametricLIFNode)):
                                # Note: This gets the last stored 'v' if available, otherwise just check if output had spikes
                                # Ideally we check the hook data if we had it, but for now let's just inspect MixLoss if possible
                                # Or better, just rely on the explicit stats loop below if needed.
                                pass 
                                
                    loss, loss_ce, loss_reg = criterion(outputs, y)
                    
                    if debug and batch_idx == 0:
                         print(f"  Loss Components: CE={loss_ce.item():.4f}, Reg={loss_reg.item():.4f}")

                # Scaled backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                # Model now strictly returns [B, num_classes] (logits)
                pred = outputs.argmax(dim=1)
                
                # Batch stats
                batch_correct = (pred == y).sum().item()
                batch_total = y.size(0)
                batch_acc = 100.0 * batch_correct / batch_total
                
                # Cumulative stats
                train_correct += batch_correct
                train_total += batch_total
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_acc': f'{100.0 * train_correct / train_total:.2f}%',
                    'batch_acc': f'{batch_acc:.2f}%'
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
                # Model now strictly returns [B, num_classes] (logits)
                pred = outputs.argmax(dim=1)
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
        
        # Vital Signs Visualization (Every 5 epochs)
        if (epoch + 1) % 5 == 0:
            try:
                log_vital_signs(model, epoch + 1, criterion)
            except Exception as e:
                print(f"[WARN] Vital Signs failed: {e}")
                
        # Object Permanence Visualization (Every 10 epochs)
        if (epoch + 1) % 10 == 0:
            try:
                visualize_permanence(model, test_loader, epoch + 1, device)
            except Exception as e:
                print(f"[WARN] Permanence Viz failed: {e}")
        
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
    parser.add_argument('--debug', action='store_true', help='Run short debug mode')
    args = parser.parse_args()
    
    train_dvs(
        resume=args.resume,
        use_cupy=not args.no_cupy,
        use_amp=not args.no_amp,
        debug=args.debug
    )
