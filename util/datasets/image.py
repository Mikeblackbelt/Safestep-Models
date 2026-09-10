"""
ImageDataset: container + operations for image-based datasets
(used by JAW).
"""

import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split

from safestep_utils.datasets.config import DatasetConfig
from safestep_utils.logging_config import get_logger

logger = get_logger(__name__)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class ImageDataset:
    """Handler for image-based datasets (JAW)."""

    def __init__(self, config: DatasetConfig):
        if not HAS_PIL:
            raise ImportError(
                "PIL (Pillow) is required for ImageDataset. Install with: pip install Pillow"
            )
        self.config = config
        self.image_paths: List[str] = []
        self.labels: List[str] = []

    def add_images(self, image_paths: List[Union[str, Path]], label: str) -> None:
        """
        Add images to dataset.

        Args:
            image_paths: List of image file paths.
            label: Label (e.g. 'ai_generated', 'authentic').
        """
        for raw_path in image_paths:
            path = Path(raw_path)
            if path.exists() and path.suffix.lower() in VALID_IMAGE_SUFFIXES:
                self.image_paths.append(str(path))
                self.labels.append(label)

    def deduplicate(self, method: str = "perceptual") -> Tuple[List[str], List[str]]:
        """
        Remove duplicate/similar images.

        Args:
            method: 'exact' (file hash) or 'perceptual' (image content hash).

        Returns:
            (unique_paths, unique_labels)
        """
        if method == "exact":
            return self._dedupe_exact()
        if method == "perceptual":
            return self._dedupe_perceptual()
        raise ValueError(f"Unknown deduplication method: {method}")

    def _dedupe_exact(self) -> Tuple[List[str], List[str]]:
        seen = set()
        unique_paths, unique_labels = [], []

        for path, label in zip(self.image_paths, self.labels):
            try:
                with open(path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash not in seen:
                    seen.add(file_hash)
                    unique_paths.append(path)
                    unique_labels.append(label)
            except OSError as e:
                logger.warning(f"Error hashing {path}: {e}")

        removed = len(self.image_paths) - len(unique_paths)
        logger.info(f"Removed {removed} exact duplicate images (file hash method)")
        return unique_paths, unique_labels

    def _dedupe_perceptual(self) -> Tuple[List[str], List[str]]:
        try:
            import imagehash
        except ImportError:
            logger.warning("imagehash not installed; falling back to exact deduplication")
            return self._dedupe_exact()

        seen_hashes: List[str] = []
        unique_paths, unique_labels = [], []

        for path, label in zip(self.image_paths, self.labels):
            try:
                img = Image.open(path)
                phash = str(imagehash.phash(img))

                is_duplicate = any(
                    bin(int(phash, 16) ^ int(seen_hash, 16)).count("1") < 5
                    for seen_hash in seen_hashes
                )
                if not is_duplicate:
                    seen_hashes.append(phash)
                    unique_paths.append(path)
                    unique_labels.append(label)
            except Exception as e:
                logger.warning(f"Error processing {path}: {e}")

        removed = len(self.image_paths) - len(unique_paths)
        logger.info(f"Removed {removed} perceptually similar images")
        return unique_paths, unique_labels

    def train_val_test_split(
        self,
        paths: List[str],
        labels: List[str],
        random_state: int = 42,
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """Split image paths into train/val/test."""
        val_test_size = self.config.val_ratio + self.config.test_ratio

        train_paths, val_test_paths, train_labels, val_test_labels = train_test_split(
            paths,
            labels,
            test_size=val_test_size,
            stratify=labels,
            random_state=random_state,
        )

        val_ratio_within_holdout = self.config.val_ratio / val_test_size
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            val_test_paths,
            val_test_labels,
            test_size=(1 - val_ratio_within_holdout),
            stratify=val_test_labels,
            random_state=random_state,
        )

        splits = {
            "train": (train_paths, train_labels),
            "val": (val_paths, val_labels),
            "test": (test_paths, test_labels),
        }
        logger.info(
            f"Train/val/test split: {len(train_paths)}/{len(val_paths)}/{len(test_paths)}"
        )
        return splits

    def get_statistics(self, paths: List[str], labels: List[str]) -> Dict:
        """Compute dataset statistics."""
        label_counts = Counter(labels)
        image_sizes = []

        for path in paths:
            try:
                img = Image.open(path)
                image_sizes.append(img.size)
            except Exception as e:
                logger.warning(f"Could not read {path}: {e}")

        return {
            "total_images": len(paths),
            "label_distribution": dict(label_counts),
            "avg_size": tuple(np.mean(image_sizes, axis=0)) if image_sizes else None,
            "sample_count_per_label": dict(label_counts),
        }