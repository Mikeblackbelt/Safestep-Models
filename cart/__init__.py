"""
CART: AI-Generated Text Detector
==================================

Hybrid classifier combining a DistilBERT-based neural model with a
heuristic scoring layer (perplexity/burstiness, repetition, stylistic
markers), fused via a lightweight meta-classifier for adversarial
robustness. See /projects/.../cart-architecture.md for the full design.
"""

from cart.config import CARTConfig
from cart.heuristics import HeuristicExtractor
from cart.model import CARTNeuralClassifier
from cart.fusion import CARTFusionModel
from cart.detector import CARTDetector

__all__ = [
    "CARTConfig",
    "HeuristicExtractor",
    "CARTNeuralClassifier",
    "CARTFusionModel",
    "CARTDetector",
]