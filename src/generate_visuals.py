import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from src.modules.model import SVT
from src.data.dvs_loader import RAMDVS128
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional, neuron

# ==========================================
# Phase 1: Parse Run 12 History for Learning Curves
# ==========================================
def plot_vital_signs():
    log_file = 'results/run_history/run13_logs/run12_training_log.txt'
    
    epochs = []
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-16') as f:
            for line in f:
                # E.g. Epoch   1 | Train: 2.3941 / 10.38% | Test: 2.3923 / 11.46% | LR: ...
                match = re.search(r"Epoch\s+(\d+)\s+\|\s+Train:\s+([\d\.]+)\s+/\s+([\d\.]+)%\s+\|\s+Test:\s+([\d\.]+)\s+/\s+([\d\.]+)%", line)
                if match:
                    epochs.append(int(match.group(1)))
                    train_losses.append(float(match.group(2)))
                    train_accs.append(float(match.group(3)))
                    test_losses.append(float(match.group(4)))
                    test_accs.append(float(match.group(5)))
    
    if epochs:
        # Generate 4-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Neuro-SVT - Training Vital Signs', fontsize=20, fontweight='bold')
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        # Panel 1: Accuracy
        ax1.plot(epochs, train_accs, label='Train Acc', color='blue', alpha=0.7)
        ax1.plot(epochs, test_accs, label='Test Acc', color='orange', linewidth=2)
        ax1.set_title("Accuracy over Epochs", fontsize=14)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Loss
        ax2.plot(epochs, train_losses, label='Train Loss', color='blue', alpha=0.7)
        ax2.plot(epochs, test_losses, label='Test Loss', color='orange', linewidth=2)
        ax2.set_title("Loss over Epochs", fontsize=14)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("CrossEntropy Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        # Generate 2-panel figure since logs are missing
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle('Neuro-SVT - Network Health Indicators', fontsize=20, fontweight='bold')
        ax3, ax4 = axes.flatten()

    # Load model and Run 1 batch to get firing rates & gradients
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SVT(img_size=128, patch_size=4, in_channels=2, num_classes=11,
                embed_dim=192, depth=4, num_heads=3, spatial_tau=2.0, memory_tau=5.0).to(device)
    for m in model.modules():
        if isinstance(m, neuron.BaseNode) or hasattr(m, 'step_mode'): m.step_mode = 'm'
        if hasattr(m, 'backend'): m.backend = 'torch'
    
    ckpt = torch.load('weights/dvs_best.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    
    # Parameter gradients (Mocking recent gradients from weights variance as proxy if no grad)
    norms = [p.data.norm().item() for p in model.parameters() if p.requires_grad]
    ax3.plot(norms, color='purple', alpha=0.7)
    ax3.set_title("Parameter L2 Norms (Proxy for Graph Health)", fontsize=14)
    ax3.set_xlabel("Parameter Index")
    ax3.set_ylabel("L2 Norm")
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Firing Rates
    firing_rates = []
    hooks = []
    def hook(module, input, output):
        firing_rates.append(output.float().mean().item())
        
    for m in model.modules():
        if isinstance(m, (neuron.LIFNode, neuron.IFNode, neuron.ParametricLIFNode)):
            hooks.append(m.register_forward_hook(hook))
            
    # Mock forward pass with random context event stream
    dummy_x = torch.rand(16, 1, 2, 128, 128).to(device)
    dummy_x = (dummy_x > 0.9).float() 
    with torch.no_grad():
        model(dummy_x)
        
    for h in hooks: h.remove()
    
    ax4.hist(firing_rates, bins=20, range=(0, 1), color='green', alpha=0.7)
    ax4.set_title("Firing Rate Distribution", fontsize=14)
    ax4.set_xlabel("Mean Firing Rate")
    ax4.set_ylabel("Layer Count")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs('visuals', exist_ok=True)
    plt.savefig('visuals/firing_rate_analysis.png', dpi=300)
    print(" Generated visuals/firing_rate_analysis.png")
    plt.close()

# ==========================================
# Phase 2: Token Evolution
# ==========================================
def plot_token_evolution():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SVT(img_size=128, patch_size=4, in_channels=2, num_classes=11,
                embed_dim=192, depth=4, num_heads=3, spatial_tau=2.0, memory_tau=5.0).to(device)
    for m in model.modules():
        if isinstance(m, neuron.BaseNode) or hasattr(m, 'step_mode'): m.step_mode = 'm'
        if hasattr(m, 'backend'): m.backend = 'torch'
    
    ckpt = torch.load('weights/dvs_best.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    
    # Enable Sequence Recording for Memory Nodes
    for block in model.blocks:
        block.memory_lif.store_v_seq = True
        
    test_set = RAMDVS128(root='./data/DVS128Gesture_Processed', train=False)
    
    # Find samples: 0: Sample A, 1: Sample B, 3: Sample C
    targets = {'Sample A': 0, 'Sample B': 1, 'Sample C': 3}
    found = {}
    
    for i in range(len(test_set)):
        x, y = test_set[i]
        label = y.item() if hasattr(y, 'item') else y
        for name, target_label in targets.items():
            if label == target_label and name not in found:
                found[name] = x
        if len(found) == 3:
            break
            
    for name, x in found.items():
        x = x.unsqueeze(1).to(device) # [T, 1, C, H, W]
        functional.reset_net(model)
        with torch.no_grad():
            model(x)
            
        traces = []
        for block in model.blocks:
            v = block.memory_lif.v
            if v.dim() == 3:
                 traces.append(v[:, 0, :].cpu().numpy())
                 
        if traces:
            fig, axes = plt.subplots(len(traces), 1, figsize=(10, 2*len(traces)), sharex=True)
            if len(traces) == 1: axes = [axes]
            for i, ax in enumerate(axes):
                im = ax.imshow(traces[i].T, aspect='auto', cmap='magma', interpolation='nearest', vmin=0, vmax=1.0)
                ax.set_ylabel(f'Layer {i}')
                if i == 0: ax.set_title(f"Memory Token Potentials ({name})")
                if i == len(traces)-1: ax.set_xlabel('Time Step')
            plt.tight_layout()
            filename = name.lower().replace(' ', '_')
            plt.savefig(f'visuals/token_evolution_{filename}.png', dpi=300)
            plt.close()
            print(f" Generated visuals/token_evolution_{filename}.png")
            
    for block in model.blocks:
        block.memory_lif.store_v_seq = False

if __name__ == '__main__':
    plot_vital_signs()
    # plot_token_evolution()  # Needs test data, which relies on DVS dataset download
