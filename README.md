# Neuro-SVT: Spiking Vision Transformer for Object Permanence

A research implementation of a **Spiking Vision Transformer (SVT)** with a "Leaky Memory Token" designed for energy-efficient object permanence on event-driven cameras.

##  Research Highlights

1. **Architecture**: **Spiking Transformer (SVT)** + Novel **Leaky Memory Token** (slower voltage decay for temporal persistence).
2. **Result**: Achieved **69.10% accuracy** on the DVS128 Gesture Dataset. Our architecture maintains robust spatial representation, proving resilient to sudden 10% event drops via the memory token.
3. **Hardware Context**: Developed, trained, and optimized entirely on an **RTX A4000 (Fedora/Windows)**.

##  Optimized Training Flow (DVS128 Gesture)

The project is optimized for high-speed training on a single workstation (e.g., RTX A4000).

### 1. Setup
```bash
pip install -r requirements.txt
# For CUDA 12.x, ensure cupy is installed:
pip install cupy-cuda12x
```

### 2. Preprocessing (One-time, ~2 mins)
Convert raw DVS events into pre-processed frame tensors to eliminate disk I/O bottlenecks.
```bash
python -m src.utils.preprocess_dvs128
```

### 3. Training (~12s/it, ~13 hours total)
Uses a **RAM-cache strategy** to load all processed data into memory for zero-latency loading.
```bash
# Windows (PowerShell) - with VRAM fragmentation fix
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
python -m src.train_dvs
```

**Options:**
- `--resume`: Resume training from `checkpoints/dvs_latest.pt`.
- `--no-cupy`: Use standard PyTorch backend if CuPy is not available.
- `--no-amp`: Disable automatic mixed precision training.

##  Project Structure

```
svt/
├── src/
│   ├── modules/
│   │   ├── model.py          # Main SVT model (Leaky Memory Token)
│   │   ├── attention.py      # SDSABlock with memory LIF
│   │   └── patch_embed.py    # Spike-based patch embedding
│   ├── data/
│   │   └── dvs_loader.py     # RAM-cached DVS128 loader
│   ├── utils/
│   │   ├── energy_meter.py   # SOP counter for energy tracking
│   │   └── preprocess_dvs128.py # Pre-processing script
│   ├── train_dvs.py          # Optimized training script
│   └── generate_visuals.py   # Portfolio visualizations
├── data/                     # Dataset storage
├── weights/                  # Locked-in Production weights
├── visuals/                  # High-Res Research Visuals
├── requirements.txt
└── README.md
```

##  Key Optimizations

| Feature | Impact |
|---------|--------|
| **RAM Cache** | Data loading cut from 45s/batch to **0s** |
| **CuPy Backend** | Fused CUDA kernels for spiking neurons |
| **AMP Training** | 16-bit precision for 2x faster iterations |
| **Tuned BPTT** | Batch size 16 optimized for 16GB VRAM |

##  Data
The DVS128 Gesture dataset should be placed in `data/DVS128Gesture/download/`. The preprocessing script will generate `.pt` tensors in `data/DVS128Gesture_Processed/`.

##  License
Apache 2.0