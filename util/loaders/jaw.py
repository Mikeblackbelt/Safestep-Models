"""
JAWDatasetLoader: loads AI-generated image detection datasets
(CIFAKE, plus stubs for GenImage/ArtiFact).
"""

from typing import Dict, List, Tuple

from util.datasets.dataset_config import DatasetConfig
from util.datasets.image import ImageDataset
from util.logging_config import get_logger

logger = get_logger(__name__)


class JAWDatasetLoader:
    """Loader for JAW (AI-generated image detection) datasets."""

    DATASETS = {
        "cifake": {
            "url": "https://github.com/Yuheng-Li/CIFAKE",
            "description": "CIFAKE dataset",
        },
        "genimage": {
            "url": "https://github.com/GenImage-Dataset/GenImage",
            "description": "GenImage dataset",
        },
        "artifact": {
            "url": "https://github.com/ArtiFact-AI/ArtiFact",
            "description": "ArtiFact dataset",
        },
    }

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.image_dataset = ImageDataset(config)

    def load_cifake(self) -> ImageDataset:
        """Load CIFAKE dataset. Expects cache/cifake/{real,fake}/ directories."""
        logger.info("Loading CIFAKE dataset...")

        cache_path = self.config.cache_dir / "cifake"
        if not cache_path.exists():
            logger.warning(
                f"CIFAKE not found at {cache_path}. "
                f"Download from {self.DATASETS['cifake']['url']} and place it there "
                f"with real/ and fake/ subfolders."
            )
            return self.image_dataset

        for split_type, label in [("real", "authentic"), ("fake", "ai_generated")]:
            split_dir = cache_path / split_type
            if split_dir.exists():
                images = list(split_dir.glob("**/*.png")) + list(split_dir.glob("**/*.jpg"))
                self.image_dataset.add_images(images, label=label)
                logger.info(f"Loaded {len(images)} {label} images from CIFAKE")

        return self.image_dataset

    def load_all(self) -> ImageDataset:
        """Load all available JAW datasets."""
        logger.info("Loading all JAW datasets...")
        self.load_cifake()
        # GenImage / ArtiFact follow the same pattern; add load_genimage()/
        # load_artifact() once their cache layout is finalized.

        logger.info(f"Total JAW samples: {len(self.image_dataset.image_paths)}")
        return self.image_dataset

    def get_splits(self) -> Dict[str, Tuple[List[str], List[str]]]:
        """Get deduplicated, split JAW data."""
        logger.info("Deduplicating JAW data...")
        unique_paths, unique_labels = self.image_dataset.deduplicate(method="exact")

        logger.info(f"JAW data after deduplication: {len(unique_paths)}")
        stats = self.image_dataset.get_statistics(unique_paths, unique_labels)
        logger.info(f"JAW statistics: {stats}")

        return self.image_dataset.train_val_test_split(unique_paths, unique_labels)