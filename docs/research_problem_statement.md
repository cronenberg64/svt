# Project Title: Neuro-SVT
## Energy-Efficient Object Permanence via Event-Driven Spiking Transformers

---

## 1. The Specific Gap

Standard Vision Transformers (ViTs) process empty frames (occlusion) with the **same compute cost** as full frames. True embodied AGI requires "Always-On" memory that is **metabolically cheap**—active when objects move, silent when they don't.

### Current Limitations

| Approach | Limitation |
|----------|------------|
| **Standard ViT** | $O(N^2)$ complexity; processes blank frames with full compute |
| **LSTM/ConvLSTM** | Heavy recurrent memory; high parameter count for temporal modeling |
| **Spiking-YOLO** | Lacks temporal attention mechanisms for belief state maintenance |
| **ANN-based tracking** | Power-hungry; no event-driven sparsity |

### The Core Problem

During **object occlusion** (e.g., a robot's gripper blocks view of target object):
1. Standard networks waste energy processing empty/occluded frames
2. Temporal context is lost without explicit memory mechanisms
3. Object recovery after occlusion requires expensive re-detection

---

## 2. The Methodology

### 2.1 Spike-Driven Sparse Attention (SDSA)

Replaces dense matrix multiplications with **binary, event-driven masking**:

$$\text{Standard Attention: } \text{Attn} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) \cdot V$$

$$\text{SDSA: } \text{Attn} = Q_{\text{spike}} \cdot K_{\text{spike}}^\top \cdot V_{\text{spike}}$$

Where $Q_{\text{spike}}, K_{\text{spike}}, V_{\text{spike}} \in \{0, 1\}$ are binary spike outputs from LIF neurons.

**Key Benefits:**
- **No Softmax**: Eliminates expensive exponential computations
- **Binary Operations**: Replaces float multiplication with accumulation
- **Sparse Computation**: Only process when spikes occur

### 2.2 Leaky Membrane Memory Token

A specialized learnable token that "leaks" information slowly, allowing the system to **"remember"** object locations during occlusion without explicit buffers.

**Implementation:**
```
Memory Token τ = 2.0  (slow decay)
Spatial Tokens τ = 1.1 (fast decay)
```

The higher time constant ($\tau$) in the Leaky Integrate-and-Fire (LIF) neuron causes:
- **Slower membrane potential decay** during zero-input frames
- **Persistent activation** that outlasts the occlusion period
- **Seamless recovery** when the object reappears

### 2.3 Architecture Overview

```
Input Event Stream (T, B, C, H, W)
          │
          ▼
┌─────────────────────────┐
│   SpikingPatchEmbed     │  Conv2D → LIF → Tokenize
└─────────────────────────┘
          │
          ▼
┌─────────────────────────┐
│   + Memory Token        │  Learnable param with τ=2.0
│     (slow decay LIF)    │
└─────────────────────────┘
          │
          ▼
┌─────────────────────────┐
│   SDSA Block × N        │  Spike-driven attention (no softmax)
│   + Spiking MLP         │
└─────────────────────────┘
          │
          ▼
┌─────────────────────────┐
│   Classification Head   │  Extract from Memory Token
└─────────────────────────┘
```

---

## 3. Metrics for Success

### 3.1 Permanence Score

**Definition:** Accuracy of object localization/classification after $>30$ frames of complete occlusion.

$$\text{Permanence Score} = \frac{\text{Correct recoveries after occlusion}}{\text{Total occlusion events}}$$

**Target:** $\geq 85\%$ recovery accuracy after 30+ frame occlusions.

### 3.2 Energy Efficiency

**Metric:** Synaptic Operations (SOPs)

$$\text{SOP} = \sum_{l} \text{FiringRate}_l \times \text{FanOut}_l$$

**Target:** $>40\%$ reduction in SOPs compared to ViT-Tiny baseline.

**Comparison:**

| Model | MACs/SOPs | Relative Energy |
|-------|-----------|-----------------|
| ViT-Tiny | ~1.3G MACs | 100% (baseline) |
| **SVT** | ~0.5G SOPs × 0.1 | <50% |

*Note: 1 SOP ≈ 0.1 MAC energy due to binary accumulation vs multiply-add.*

### 3.3 Latency

**Target:** $\geq 30$ FPS inference on NVIDIA RTX A4000.

**Optimizations:**
- CuPy backend for LIF neurons
- Multi-step mode (`step_mode='m'`) for parallel temporal processing
- Sparse attention avoids full $O(N^2)$ computation

---

## 4. Expected Contributions

1. **Novel Architecture:** First Spiking Vision Transformer with dedicated memory mechanism for object permanence

2. **Energy Efficiency:** Demonstrate $>40\%$ energy reduction while maintaining accuracy

3. **Temporal Reasoning:** Leaky Memory Token enables belief state maintenance without LSTM overhead

4. **Neuromorphic-Ready:** Architecture compatible with event cameras and neuromorphic hardware (Loihi, SpiNNaker)

---

## 5. References

1. Zhou et al., "Spikformer: When Spiking Neural Network Meets Transformer," ICLR 2023
2. Yao et al., "Spike-driven Transformer V2," ICLR 2024
3. Deng et al., "Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting," 2022
4. SpikingJelly Documentation, 2024
