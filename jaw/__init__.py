"""
JAW: AI-Generated Image Classifier
=====================================

CNN-based classifier combining an EfficientNet-B3 spatial backbone
with hand-engineered frequency-domain features (FFT spectral energy,
Gabor texture responses) to catch generation artifacts invisible in
the spatial domain alone. See /projects/.../jaw-architecture.md for
the full design.
"""

from jaw.config import JAWConfig
from jaw.frequency_features import FrequencyFeatureExtractor
from jaw.model import JAWModel
from jaw.detector import JAWDetector

__all__ = [
    "JAWConfig",
    "FrequencyFeatureExtractor",
    "JAWModel",
    "JAWDetector",
]