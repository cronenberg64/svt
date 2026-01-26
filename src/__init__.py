"""
Neuro-SVT: Spiking Vision Transformer for Object Permanence

A novel SNN architecture using Spike-Driven Sparse Attention (SDSA)
and Leaky Memory Tokens for energy-efficient temporal reasoning.
"""

__version__ = "0.1.0"
__author__ = "SVT Research Team"

# Monkeypatch numpy.int for SpikingJelly compatibility
# This is required because SpikingJelly's Cupy backend uses np.int which was removed in NumPy 1.20+
import numpy as np
if not hasattr(np, 'int'):
    np.int = int
