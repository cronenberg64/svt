import torch
import os
import time
from spikingjelly.activation_based import neuron
from src.modules.model import SVT
from src.utils.energy_meter import SOPCounter

def evaluate_metrics():
    print("=" * 60)
    print("Neuro-SVT Metric Evaluation Script")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Initialize Model
    print("\n[1/3] Initializing Model...")
    model = SVT(img_size=128, patch_size=4, in_channels=2, num_classes=11,
                embed_dim=192, depth=4, num_heads=3, spatial_tau=2.0, memory_tau=5.0).to(device)
                
    for m in model.modules():
        if isinstance(m, neuron.BaseNode) or hasattr(m, 'step_mode'):
            m.step_mode = 'm'
        if hasattr(m, 'backend'):
            m.backend = 'torch' # Ensure compatibility without CuPy compilation

    # Load Weights if available
    weight_path = 'weights/dvs_best.pt'
    if os.path.exists(weight_path):
        print(f"Loading weights from {weight_path}")
        ckpt = torch.load(weight_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    else:
        print(f"Warning: {weight_path} not found. Running metrics on initialized weights.")

    model.eval()

    # 2. Model Parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[2/3] Model Parameters: {num_params / 1e6:.2f}M")

    # 3. Energy Consumption and Firing Rate
    print("\n[3/3] Simulating Forward Pass for Energy Metrics...")
    
    # Try to load real data, otherwise use dummy data
    try:
        from src.data.dvs_loader import RAMDVS128
        dataset = RAMDVS128(root='./data/DVS128Gesture_Processed', train=False)
        x, _ = dataset[0] # [T, C, H, W]
        x = x.unsqueeze(1).to(device) # [T, B, C, H, W]
        print(" Using sample from DVS128 Gesture Dataset")
    except Exception as e:
        print(" DVS128 Gesture Dataset not found or not preprocessed.")
        print(" Using randomly generated spike events (dummy data).")
        # Simulate an event stream [T=16, B=1, C=2, H=128, W=128]
        x = torch.rand(16, 1, 2, 128, 128).to(device)
        x = (x > 0.95).float() # Sparse events

    # Register energy hooks
    counter = SOPCounter()
    counter.register_hooks(model)

    start_time = time.time()
    with torch.no_grad():
        model(x)
    inference_time = time.time() - start_time

    # Compute SOPs and Firing Rates
    total_sops, layer_stats = counter.compute_sops(model)
    counter.remove_hooks()

    if layer_stats:
        avg_firing_rate = sum(s.firing_rate for s in layer_stats) / len(layer_stats)
    else:
        avg_firing_rate = 0.0

    print("\n" + "=" * 60)
    print("METRICS REPORT")
    print("=" * 60)
    print(f"Model Parameters:       {num_params / 1e6:.2f}M")
    print(f"Average Firing Rate:    {avg_firing_rate:.2%}")
    print(f"Energy Consumption:     {total_sops / 1e6:.2f} M-SOPs (per sample)")
    print(f"Inference Time:         {inference_time * 1000:.2f} ms")
    print("=" * 60)
    print("\nNote: 'Recall @ 10-Frame Drop' requires a full dataset evaluation run.")
    print("To evaluate full dataset accuracy, use: python -m src.train_dvs --eval_only")

if __name__ == "__main__":
    evaluate_metrics()
