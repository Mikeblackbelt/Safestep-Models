"""
SafeStep AI Utilities

Utility package for the SafeStep AI detection model suite.

Currently provides:
- util.datasets: dataset loading, deduplication, and splitting
  for CART, JAW, and the Attack/Beast/Warhammer phishing pipeline.
"""

from util.datasets.dataset_config import DatasetConfig
from util.manager import SafeStepDataManager
from util.loaders.batch import BatchDataLoader
from util.datasets.text import TextDataset
from util.datasets.image import ImageDataset
from util.loaders.cart import CARTDatasetLoader
from util.loaders.jaw import JAWDatasetLoader
from util.loaders.phishing import PhishingDatasetLoader

__all__ = [
    "DatasetConfig",
    "SafeStepDataManager",
    "BatchDataLoader",
    "TextDataset",
    "ImageDataset",
    "CARTDatasetLoader",
    "JAWDatasetLoader",
    "PhishingDatasetLoader",
]