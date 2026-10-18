"""
SafeStepDataManager: single entry point for loading, deduplicating,
splitting, and preparing data for all SafeStep models (CART, JAW, Phishing).
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from util.datasets.dataset_config import DatasetConfig
from util.loaders.batch import BatchDataLoader
from util.loaders.cart import CARTDatasetLoader, normalize_text_label
from util.loaders.jaw import JAWDatasetLoader, normalize_image_label
from util.loaders.phishing import PhishingDatasetLoader
from util.logging_config import get_logger

logger = get_logger(__name__)


class SafeStepDataManager:
    """Main data manager for all SafeStep models."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        logger.info(f"Data root: {self.config.data_root}")

    def get_cart_data(self, path: Optional[Union[str, Path]] = None) -> Dict[str, Tuple[List[str], List[str]]]:
        """Get CART (text detection) train/val/test splits."""
        loader = CARTDatasetLoader(self.config)
        if path:
            p = Path(path)
            if p.is_file():
                loader.load_file(p)
            elif p.is_dir():
                loader.load_directory(p)
        else:
            loader.load_all()
        return loader.get_splits()

    def get_jaw_data(self, path: Optional[Union[str, Path]] = None) -> Dict[str, Tuple[List[str], List[str]]]:
        """Get JAW (image detection) train/val/test splits."""
        loader = JAWDatasetLoader(self.config)
        if path:
            p = Path(path)
            if p.is_dir():
                loader.load_directory(p)
        else:
            loader.load_all()
        return loader.get_splits()

    def get_phishing_data(self, path: Optional[Union[str, Path]] = None) -> Dict[str, Tuple[List[str], List[str]]]:
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

    @staticmethod
    def prepare_sample_data(
        model: str = "all",
        target_dir: Union[str, Path] = "data",
    ) -> Dict[str, Path]:
        """
        Generate lightweight, ready-to-train sample datasets for CART, JAW, or both.
        Sets up the exact train/ and val/ split layout expected by train.py.
        """
        base = Path(target_dir)
        base.mkdir(parents=True, exist_ok=True)
        results = {}

        if model in ("jaw", "all"):
            jaw_dir = base if model == "jaw" else base / "jaw"
            results["jaw"] = JAWDatasetLoader.create_sample_dataset(jaw_dir)

        if model in ("cart", "all"):
            cart_dir = base if model == "cart" else base / "cart"
            results["cart"] = CARTDatasetLoader.create_sample_dataset(cart_dir)

        if model in ("phishing", "all"):
            phish_dir = base / "phishing"
            results["phishing"] = PhishingDatasetLoader.create_sample_dataset(phish_dir)

        return results

    @staticmethod
    def inspect_directory(directory: Union[str, Path]) -> Dict[str, int]:
        """Inspect a directory to report discovered images, texts, and classes."""
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Directory {dir_path} does not exist.")
            return {}

        image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        images = [p for p in dir_path.glob("**/*") if p.suffix.lower() in image_suffixes]
        texts = [p for p in dir_path.glob("**/*.txt") if p.is_file()]
        csvs = [p for p in dir_path.glob("**/*.csv") if p.is_file()]
        jsonls = [p for p in dir_path.glob("**/*.jsonl") if p.is_file()]

        summary = {
            "total_images": len(images),
            "total_text_files": len(texts),
            "csv_files": len(csvs),
            "jsonl_files": len(jsonls),
        }

        print(f"--- Inspection report for {dir_path} ---")
        print(f"Images found:     {summary['total_images']:,}")
        print(f"Text files found: {summary['total_text_files']:,}")
        print(f"CSV files found:  {summary['csv_files']:,}")
        print(f"JSONL files found:{summary['jsonl_files']:,}")

        # Check subfolder distribution
        subfolders = [p for p in dir_path.iterdir() if p.is_dir()]
        if subfolders:
            print("\nSubdirectories:")
            for sub in sorted(subfolders):
                sub_imgs = len([p for p in sub.glob("**/*") if p.suffix.lower() in image_suffixes])
                sub_txts = len([p for p in sub.glob("**/*.txt")])
                info = []
                if sub_imgs:
                    info.append(f"{sub_imgs:,} images")
                if sub_txts:
                    info.append(f"{sub_txts:,} text files")
                details = ", ".join(info) if info else "empty or other files"
                print(f"  - {sub.name}/: {details}")

        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeStep Unified Data Manager")
    parser.add_argument(
        "--sample",
        choices=["cart", "jaw", "phishing", "all"],
        help="Generate sample datasets in data/ for instant training test",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Target data directory (default: data)",
    )
    parser.add_argument(
        "--inspect",
        type=str,
        help="Inspect a dataset directory and print file and label statistics",
    )
    parser.add_argument(
        "--cifake",
        type=str,
        help="Path to downloaded CIFAKE directory to import",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Target directory to export normalized train/val splits",
    )
    args = parser.parse_args()

    if args.sample:
        created = SafeStepDataManager.prepare_sample_data(args.sample, args.data_dir)
        print("\n[SUCCESS] Sample datasets ready:")
        for k, p in created.items():
            print(f"  - {k.upper()}: {p}")
        print(f"\nYou can now run:\n  python -m train --model {args.sample} --data-dir {args.data_dir}")
    elif args.inspect:
        SafeStepDataManager.inspect_directory(args.inspect)
    elif args.cifake:
        cfg = DatasetConfig()
        loader = JAWDatasetLoader(cfg)
        loader.load_cifake(args.cifake)
        out = args.export or f"{args.data_dir}/jaw"
        loader.export_to_train_val(out)
        print(f"\n[SUCCESS] Imported CIFAKE from {args.cifake} and exported to {out}")
    else:
        parser.print_help()