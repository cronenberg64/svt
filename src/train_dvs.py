import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import SVT
from src.data.dvs_loader import DVSGestureDataLoader
from src.utils.energy_meter import EnergyMeter
from spikingjelly.activation_based import functional
import time
import os

class MixLoss(nn.Module):
    """
    CrossEntropy + Rate Regularization to penalize high firing rates.
    """
    def __init__(self, model, reg_weight=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.reg_weight = reg_weight
        self.firing_rates = []
        self.hooks = []
        
        # Register hooks to collect firing rates
        from spikingjelly.activation_based import neuron
        for m in model.modules():
            if isinstance(m, (neuron.LIFNode, neuron.IFNode)):
                def hook(module, input, output):
                    # output is [T, B, ...]
                    self.firing_rates.append(output.mean())
                self.hooks.append(m.register_forward_hook(hook))
        
    def forward(self, outputs, target):
        # outputs: [T, B, num_classes]
        # target: [B]
        
        # 1. Classification Loss (Average over time)
        loss_ce = self.ce(outputs.mean(0), target)
        
        # 2. Rate Regularization
        if self.firing_rates:
            loss_reg = torch.stack(self.firing_rates).mean()
        else:
            loss_reg = torch.tensor(0.0).to(outputs.device)
            
        # Clear firing rates for next batch
        self.firing_rates = []
            
        return loss_ce + self.reg_weight * loss_reg, loss_ce, loss_reg

    def __del__(self):
        for h in self.hooks:
            h.remove()

def train_dvs():
    print("="*60)
    print("NEURO-SVT DVS128 GESTURE TRAINING")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    T = 16
    batch_size = 16
    epochs = 1 # Sanity run
    lr = 1e-3
    reg_weight = 0.1
    
    # Data Loader
    print("Initializing DVS Loader (this may download the dataset)...")
    data_root = './data/DVS128Gesture'
    os.makedirs(data_root, exist_ok=True)
    loader_wrapper = DVSGestureDataLoader(root=data_root, train=True, T=T, batch_size=batch_size)
    train_loader = loader_wrapper.get_dataloader()
    
    # Model
    model = SVT(
        img_size=128,
        patch_size=4,
        in_channels=2, # DVS has 2 polarities
        num_classes=11,
        embed_dim=192,
        depth=4,
        num_heads=3,
        spatial_tau=1.1,
        memory_tau=5.0
    ).to(device)
    
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = MixLoss(model, reg_weight=reg_weight)
    
    # Energy Meter for SOP tracking
    meter = EnergyMeter(model)
    
    print("\nStarting Sanity Training (1 Epoch):")
    print("-" * 80)
    print(f"{'Batch':<10} {'Loss':<10} {'CE_Loss':<10} {'Reg_Loss':<10} {'Avg SOPs':<15}")
    print("-" * 80)
    
    model.train()
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            if i >= 50: # Only run 50 batches for the deliverable
                break
                
            x, y = x.to(device), y.to(device)
            # x is already [T, B, C, H, W] from our loader
            
            functional.reset_net(model)
            
            # Forward pass
            outputs, _, _ = model(x)
            
            # Compute Loss
            loss, loss_ce, loss_reg = criterion(outputs, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track Energy (SOPs)
            # We use the meter to compute SOPs for this batch
            total_sops, _ = meter.counter.compute_sops(model)
            avg_sops_per_sample = total_sops / batch_size
            
            if i % 1 == 0:
                print(f"{i:<10} {loss.item():<10.4f} {loss_ce.item():<10.4f} {loss_reg.item():<10.4f} {avg_sops_per_sample:<15.0f}")
                
    print("-" * 80)
    print("Sanity training complete!")
    
    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/dvs_sanity.pt')
    print("Model saved to checkpoints/dvs_sanity.pt")

if __name__ == "__main__":
    train_dvs()
