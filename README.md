# Neuro-SVT: Spiking Vision Transformer for Object Permanence

A research implementation of a **Spiking Vision Transformer (SVT)** with a "Leaky Memory Token" for energy-efficient object permanence on event-driven cameras.

## 🎯 Research Goals

1. **Energy Efficiency**: Achieve >40% energy reduction vs standard ViT through spike-based computation
2. **Memory Persistence**: Maintain object representations during occlusion via slow-decay LIF neurons

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/[your-repo]/svt.git
cd svt
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Validate the Model
```bash
python validate.py
```
This runs the energy baseline test and generates `results/memory_demonstration_v2.png`.

### 3. Train on DVS128 Gesture (Optional)
```bash
# Download dataset from IBM Box (see Data section below)
python -m src.train_dvs
```

## 📁 Project Structure

```
svt/
├── src/
│   ├── model.py              # Main SVT model
│   ├── modules/
│   │   ├── attention.py      # SDSABlock with memory LIF
│   │   └── patch_embed.py    # Spike-based patch embedding
│   ├── data/
│   │   └── dvs_loader.py     # DVS128 Gesture data loader
│   ├── utils/
│   │   └── energy_meter.py   # SOP counter for energy tracking
│   └── train_dvs.py          # DVS128 training script
├── validate.py               # Validation & visualization
├── quick_train.py            # Quick smoke test
├── requirements.txt
└── results/                  # Generated plots
```

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Energy Reduction | **99.8%** vs ViT baseline |
| Average Firing Rate | ~28-33% |
| Memory Token Persistence | ~0.16 during occlusion |

## 📦 Data

### DVS128 Gesture Dataset
The DVS128 Gesture dataset must be downloaded manually from IBM Box:

1. Go to: [IBM Box - DVS128 Gesture](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794)
2. Download these files:
   - `DvsGesture.tar.gz` (~2.9 GB)
   - `gesture_mapping.csv`
   - `LICENSE.txt`
   - `README.txt`
3. Place them in: `data/DVS128Gesture/download/`

The loader will automatically extract and preprocess the data on first run.

## 🔧 Configuration

### Model Parameters (SVT-Tiny)
- `embed_dim`: 192
- `depth`: 4 transformer blocks
- `num_heads`: 3
- `spatial_tau`: 1.1 (fast decay for spatial tokens)
- `memory_tau`: 5.0 (slow decay for memory persistence)

### Training
- Optimizer: AdamW (lr=1e-3)
- Loss: MixLoss (CrossEntropy + Rate Regularization)
- Batch size: 16
- Time steps (T): 16

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 (6GB) | RTX 4070+ (8GB+) |
| RAM | 16GB | 32GB |
| CUDA | 11.8+ | 12.x |

## 📝 Notes

- SpikingJelly's Cupy backend requires CUDA. Install `cupy-cuda12x` for CUDA 12.
- On Windows, multiprocessing can be slow. Set `num_workers=0` in the DataLoader if you encounter issues.
- The DVS128 preprocessing only runs once - subsequent runs load from `events_np/`.

## 📄 License

Apache 2.0