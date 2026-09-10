"""
Dataset configuration for SafeStep AI.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DatasetConfig:
    """Configuration for dataset locations and parameters."""

    # Base paths
    data_root: Path = field(default_factory=lambda: Path.home() / ".safestep_data")
    cache_dir: Optional[Path] = None
    raw_data_dir: Optional[Path] = None
    processed_data_dir: Optional[Path] = None

    # Dataset sizes (target)
    cart_target_size: int = 500_000
    jaw_target_size: int = 300_000
    phishing_target_size: int = 1_000_000

    # Deduplication
    text_similarity_threshold: float = 0.95   # Jaccard similarity for text
    image_similarity_threshold: float = 0.90  # Perceptual hash for images

    # Train/val/test splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self):
        """Initialize derived paths and create directories."""
        self.data_root = Path(self.data_root)

        if self.cache_dir is None:
            self.cache_dir = self.data_root / "cache"
        if self.raw_data_dir is None:
            self.raw_data_dir = self.data_root / "raw"
        if self.processed_data_dir is None:
            self.processed_data_dir = self.data_root / "processed"

        self.cache_dir = Path(self.cache_dir)
        self.raw_data_dir = Path(self.raw_data_dir)
        self.processed_data_dir = Path(self.processed_data_dir)

        for path in (self.cache_dir, self.raw_data_dir, self.processed_data_dir):
            path.mkdir(parents=True, exist_ok=True)

        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(ratio_sum - 1.0) < 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {ratio_sum}"
            )