"""
SafeStepDataManager: single entry point for loading, deduplicating,
and splitting data for all 5 SafeStep models.
"""

from typing import Dict, List, Optional, Tuple

from util.datasets.dataset_config import DatasetConfig
from util.loaders.cart import CARTDatasetLoader
from util.loaders.jaw import JAWDatasetLoader
from util.loaders.phishing import PhishingDatasetLoader
from util.loaders.batch import BatchDataLoader
from util.logging_config import get_logger

logger = get_logger(__name__)


class SafeStepDataManager:
    """Main data manager for all SafeStep models."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        logger.info(f"Data root: {self.config.data_root}")

    def get_cart_data(self) -> Dict[str, Tuple[List[str], List[str]]]:
        """Get CART (text detection) train/val/test splits."""
        loader = CARTDatasetLoader(self.config)
        loader.load_all()
        return loader.get_splits()

    def get_jaw_data(self) -> Dict[str, Tuple[List[str], List[str]]]:
        """Get JAW (image detection) train/val/test splits."""
        loader = JAWDatasetLoader(self.config)
        loader.load_all()
        return loader.get_splits()

    def get_phishing_data(self) -> Dict[str, Tuple[List[str], List[str]]]:
        """Get phishing detection (Attack/Beast/Warhammer) train/val/test splits."""
        loader = PhishingDatasetLoader(self.config)
        loader.load_all()
        return loader.get_splits()

    def get_batch_loader(
        self,
        texts: List[str],
        labels: List[str],
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> BatchDataLoader:
        """Get a batch loader for training."""
        return BatchDataLoader(texts, labels, batch_size=batch_size, shuffle=shuffle)