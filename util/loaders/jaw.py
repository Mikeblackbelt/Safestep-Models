"""
JAWDatasetLoader: loads AI-generated image detection datasets
(CIFAKE, GenImage, ArtiFact, custom folders, and synthetic sample data).
"""

import argparse
import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

from util.datasets.dataset_config import DatasetConfig
from util.datasets.image import ImageDataset, VALID_IMAGE_SUFFIXES
from util.logging_config import get_logger

logger = get_logger(__name__)

# Label normalization sets
AUTHENTIC_ALIASES = {
    "authentic", "real", "human", "nature", "genuine",
    "original", "photo", "live", "0",
}
AI_GENERATED_ALIASES = {
    "ai_generated", "fake", "synthetic", "ai", "generated",
    "deepfake", "midjourney", "dalle", "stable_diffusion", "1",
}


def normalize_image_label(label: str) -> str:
    """Normalize common folder/class names into standard JAW labels ('authentic' or 'ai_generated')."""
    cleaned = str(label).strip().lower()
    if cleaned in AUTHENTIC_ALIASES:
        return "authentic"
    if cleaned in AI_GENERATED_ALIASES:
        return "ai_generated"
    # Fuzzy heuristic matching
    if any(k in cleaned for k in ("fake", "synth", "ai", "gen", "deepfake")):
        return "ai_generated"
    if any(k in cleaned for k in ("real", "auth", "nature", "human", "orig")):
        return "authentic"
    return cleaned


class JAWDatasetLoader:
    """Loader for JAW (AI-generated image detection) datasets."""

    DATASETS = {
        "cifake": {
            "url": "https://github.com/Yuheng-Li/CIFAKE",
            "kaggle": "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images",
            "description": "CIFAKE dataset: 60,000 real & 60,000 AI-generated 32x32 images",
        },
        "genimage": {
            "url": "https://github.com/GenImage-Dataset/GenImage",
            "description": "GenImage benchmark with 8 generative models (SD, Midjourney, etc.)",
        },
        "artifact": {
            "url": "https://github.com/ArtiFact-AI/ArtiFact",
            "description": "ArtiFact multi-generator synthetic image dataset",
        },
    }

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.image_dataset = ImageDataset(self.config)

    def _resolve_candidate_dirs(self, name: str, custom_path: Optional[Union[str, Path]] = None) -> List[Path]:
        """Find candidate search paths for a given dataset name."""
        candidates = []
        if custom_path:
            candidates.append(Path(custom_path))
        candidates.extend([
            self.config.cache_dir / name,
            self.config.raw_data_dir / name,
            self.config.data_root / name,
            Path("data") / name,
            Path("data") / "jaw" / name,
            Path("data") / "jaw",
            Path("data"),
        ])
        # Return unique existing paths
        seen = set()
        existing = []
        for p in candidates:
            resolved = p.resolve() if p.exists() else p
            if resolved not in seen and p.exists():
                seen.add(resolved)
                existing.append(p)
        return existing

    def load_cifake(self, path: Optional[Union[str, Path]] = None) -> ImageDataset:
        """
        Load CIFAKE dataset. Supports Kaggle (train/REAL, test/FAKE) and flat (real, fake) structures.
        Automatically normalizes 'real' -> 'authentic' and 'fake' -> 'ai_generated'.
        """
        logger.info("Loading CIFAKE dataset...")
        candidate_dirs = self._resolve_candidate_dirs("cifake", path)

        target_dir = None
        for candidate in candidate_dirs:
            # Check if directory has real/fake or REAL/FAKE or train/REAL
            sub_names = {sub.name.lower() for sub in candidate.glob("*") if sub.is_dir()}
            if {"real", "fake"}.issubset(sub_names) or {"train", "test"}.issubset(sub_names):
                target_dir = candidate
                break
            if candidate.name.lower() == "cifake":
                target_dir = candidate
                break

        if not target_dir or not target_dir.exists():
            logger.warning(
                f"CIFAKE dataset not found. Looked in: {[str(p) for p in candidate_dirs] if candidate_dirs else path}.\n"
                f"Download from {self.DATASETS['cifake']['url']} or {self.DATASETS['cifake']['kaggle']} "
                f"and extract to data/cifake or {self.config.cache_dir}/cifake."
            )
            return self.image_dataset

        return self.load_directory(target_dir)

    def load_genimage(self, path: Optional[Union[str, Path]] = None) -> ImageDataset:
        """
        Load GenImage dataset. GenImage contains subfolders per generator (e.g. midjourney,
        stable_diffusion_v_1_4) with 'nature' (authentic) and 'ai' (ai_generated) subdirectories.
        """
        logger.info("Loading GenImage dataset...")
        candidate_dirs = self._resolve_candidate_dirs("genimage", path)
        target_dir = candidate_dirs[0] if candidate_dirs else None

        if not target_dir or not target_dir.exists():
            logger.warning(
                f"GenImage dataset not found at {path or self.config.cache_dir / 'genimage'}. "
                f"Download from {self.DATASETS['genimage']['url']} and extract to data/genimage."
            )
            return self.image_dataset

        return self.load_directory(target_dir)

    def load_artifact(self, path: Optional[Union[str, Path]] = None) -> ImageDataset:
        """Load ArtiFact dataset with generator subfolders and real/fake splits."""
        logger.info("Loading ArtiFact dataset...")
        candidate_dirs = self._resolve_candidate_dirs("artifact", path)
        target_dir = candidate_dirs[0] if candidate_dirs else None

        if not target_dir or not target_dir.exists():
            logger.warning(
                f"ArtiFact dataset not found at {path or self.config.cache_dir / 'artifact'}. "
                f"Download from {self.DATASETS['artifact']['url']} and extract to data/artifact."
            )
            return self.image_dataset

        return self.load_directory(target_dir)

    def load_directory(
        self,
        directory: Union[str, Path],
        label_mapping: Optional[Dict[str, str]] = None,
    ) -> ImageDataset:
        """
        Load images from any directory structure.
        Recursively discovers image files and normalizes labels based on parent folder names.
        
        Supported structures:
          - Class folders: dir/{authentic,ai_generated}/*.jpg or dir/{real,fake}/*.jpg
          - Split folders: dir/{train,val,test}/{real,fake}/*.jpg
          - Generator folders: dir/{generator_name}/{ai,nature}/*.jpg
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory {dir_path} does not exist.")
            return self.image_dataset

        logger.info(f"Scanning directory for images: {dir_path}")
        label_mapping = label_mapping or {}
        counts = {"authentic": 0, "ai_generated": 0, "other": 0}

        # Walk through directory
        for root, dirs, files in os.walk(dir_path):
            current_path = Path(root)
            folder_name = current_path.name
            parent_name = current_path.parent.name if current_path.parent else ""

            # Check if current folder or parent indicates label
            raw_label = label_mapping.get(folder_name) or normalize_image_label(folder_name)
            if raw_label not in ("authentic", "ai_generated"):
                raw_label = label_mapping.get(parent_name) or normalize_image_label(parent_name)

            if raw_label not in ("authentic", "ai_generated"):
                continue

            images_in_folder = [
                current_path / f for f in files
                if Path(f).suffix.lower() in VALID_IMAGE_SUFFIXES
            ]
            if images_in_folder:
                self.image_dataset.add_images(images_in_folder, label=raw_label)
                counts[raw_label] += len(images_in_folder)

        logger.info(
            f"Loaded {counts['authentic']} authentic and {counts['ai_generated']} "
            f"ai_generated images from {dir_path}"
        )
        return self.image_dataset

    def load_all(self) -> ImageDataset:
        """Load all available JAW datasets found on disk."""
        logger.info("Loading all available JAW datasets...")
        self.load_cifake()
        self.load_genimage()
        self.load_artifact()

        # Check default data/jaw directory as well
        local_jaw = Path("data") / "jaw"
        if local_jaw.exists():
            self.load_directory(local_jaw)

        logger.info(f"Total JAW samples loaded: {len(self.image_dataset.image_paths)}")
        return self.image_dataset

    def get_splits(self) -> Dict[str, Tuple[List[str], List[str]]]:
        """Get deduplicated, train/val/test split JAW data."""
        logger.info("Deduplicating JAW data...")
        unique_paths, unique_labels = self.image_dataset.deduplicate(method="exact")

        logger.info(f"JAW data after deduplication: {len(unique_paths)}")
        stats = self.image_dataset.get_statistics(unique_paths, unique_labels)
        logger.info(f"JAW statistics: {stats}")

        return self.image_dataset.train_val_test_split(unique_paths, unique_labels)

    @staticmethod
    def create_sample_dataset(
        output_dir: Optional[Union[str, Path]] = None,
        samples_per_class: int = 50,
    ) -> Path:
        """
        Create a lightweight, ready-to-train sample dataset with authentic
        and AI-generated images in train/ and val/ splits.
        Useful for quick smoke testing, debugging, and offline demos.
        """
        out_path = Path(output_dir or Path("data") / "jaw_sample")
        train_auth = out_path / "train" / "authentic"
        train_ai = out_path / "train" / "ai_generated"
        val_auth = out_path / "val" / "authentic"
        val_ai = out_path / "val" / "ai_generated"

        for p in (train_auth, train_ai, val_auth, val_ai):
            p.mkdir(parents=True, exist_ok=True)

        val_count = max(5, int(samples_per_class * 0.2))
        train_count = samples_per_class

        def _make_auth_img(idx: int) -> Image.Image:
            # Natural gradients and smooth shapes
            img = Image.new("RGB", (224, 224), color=(120 + (idx * 3) % 100, 150, 200))
            draw = ImageDraw.Draw(img)
            draw.rectangle([20, 60, 180, 180], fill=(40, 120 + (idx * 5) % 80, 60))
            draw.ellipse([60, 30, 120, 90], fill=(240, 220, 80))
            return img

        def _make_ai_img(idx: int) -> Image.Image:
            # High-frequency checkerboard / synthetic spectral patterns
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            freq = 8 + (idx % 8)
            for y in range(224):
                for x in range(224):
                    if (x // freq + y // freq) % 2 == 0:
                        arr[y, x] = [180 + (idx * 2) % 60, 40, 140]
                    else:
                        arr[y, x] = [30, 180 + (idx * 4) % 60, 160]
            return Image.fromarray(arr)

        # Generate train
        for i in range(train_count):
            _make_auth_img(i).save(train_auth / f"auth_train_{i:04d}.png")
            _make_ai_img(i).save(train_ai / f"ai_train_{i:04d}.png")

        # Generate val
        for i in range(val_count):
            _make_auth_img(1000 + i).save(val_auth / f"auth_val_{i:04d}.png")
            _make_ai_img(1000 + i).save(val_ai / f"ai_val_{i:04d}.png")

        logger.info(
            f"Created sample JAW dataset at {out_path} "
            f"({train_count * 2} train, {val_count * 2} val images)"
        )
        return out_path

    def export_to_train_val(
        self,
        output_dir: Union[str, Path],
        copy_files: bool = True,
    ) -> Dict[str, int]:
        """
        Export currently loaded data into a standard train/val directory format
        suitable for direct use with `python -m train --model jaw --data-dir <output_dir>`.
        """
        splits = self.get_splits()
        out_path = Path(output_dir)
        counts = {"train": 0, "val": 0, "test": 0}

        for split_name in ("train", "val", "test"):
            if split_name not in splits:
                continue
            paths, labels = splits[split_name]
            for src_path_str, label in zip(paths, labels):
                norm_label = normalize_image_label(label)
                dest_dir = out_path / split_name / norm_label
                dest_dir.mkdir(parents=True, exist_ok=True)
                src = Path(src_path_str)
                dest = dest_dir / src.name
                if copy_files:
                    if not dest.exists():
                        shutil.copy2(src, dest)
                else:
                    # Symlink if permitted
                    if not dest.exists():
                        try:
                            os.symlink(src.resolve(), dest)
                        except OSError:
                            shutil.copy2(src, dest)
                counts[split_name] += 1

        logger.info(f"Exported dataset to {out_path}: {counts}")
        return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAW Dataset Loader Utility")
    parser.add_argument("--sample", nargs="?", const="data/jaw_sample", help="Generate sample dataset for quick testing")
    parser.add_argument("--cifake", type=str, help="Load CIFAKE dataset from path")
    parser.add_argument("--dir", type=str, help="Load custom image dataset from directory")
    parser.add_argument("--export", type=str, help="Export loaded dataset into standard train/val directory")
    args = parser.parse_args()

    if args.sample:
        target = JAWDatasetLoader.create_sample_dataset(args.sample)
        print(f"Sample JAW dataset generated at: {target}")
    else:
        cfg = DatasetConfig()
        loader = JAWDatasetLoader(cfg)
        if args.cifake:
            loader.load_cifake(args.cifake)
        elif args.dir:
            loader.load_directory(args.dir)
        else:
            loader.load_all()

        print(f"Loaded {len(loader.image_dataset.image_paths)} images.")
        if args.export:
            loader.export_to_train_val(args.export)
            print(f"Exported to: {args.export}")