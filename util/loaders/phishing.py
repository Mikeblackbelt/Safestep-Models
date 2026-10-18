"""
PhishingDatasetLoader: loads phishing detection datasets used by
the Attack / Beast / Warhammer escalation pipeline.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from util.datasets.dataset_config import DatasetConfig
from util.datasets.text import TextDataset
from util.logging_config import get_logger

logger = get_logger(__name__)


class PhishingDatasetLoader:
    """Loader for phishing detection datasets (Attack/Beast/Warhammer)."""

    DATASETS = {
        "enron_spam": {
            "url": "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz",
            "description": "Enron spam corpus",
        },
        "phishtank": {
            "url": "http://phishtank.com/developer_info.php",
            "description": "PhishTank URLs",
        },
        "uci_phishing": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/phishing.data",
            "description": "UCI Phishing Websites Dataset",
        },
    }

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.text_dataset = TextDataset(self.config)

    def _resolve_candidate_dirs(self, name: str, custom_path: Optional[Union[str, Path]] = None) -> List[Path]:
        candidates = []
        if custom_path:
            candidates.append(Path(custom_path))
        candidates.extend([
            self.config.cache_dir / name,
            self.config.raw_data_dir / name,
            self.config.data_root / name,
            Path("data") / "phishing" / name,
            Path("data") / name,
        ])
        existing = []
        seen = set()
        for p in candidates:
            if p.exists() and p.resolve() not in seen:
                seen.add(p.resolve())
                existing.append(p)
        return existing

    def load_enron_spam(self, path: Optional[Union[str, Path]] = None) -> TextDataset:
        """Load Enron spam dataset. Expects cache/enron_spam/{spam,ham}/*.txt."""
        logger.info("Loading Enron spam corpus...")
        candidates = self._resolve_candidate_dirs("enron_spam", path)
        if not candidates:
            logger.warning(
                f"Enron corpus not found. Download from {self.DATASETS['enron_spam']['url']} "
                f"and extract to data/phishing/enron_spam or {self.config.cache_dir}/enron_spam."
            )
            return self.text_dataset

        target_dir = candidates[0]
        for split_type, label in [("spam", "phishing"), ("ham", "legitimate")]:
            split_dir = target_dir / split_type
            if not split_dir.exists():
                continue

            email_count = 0
            for email_file in split_dir.glob("**/*.txt"):
                try:
                    with open(email_file, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    self.text_dataset.add_records([{"text": text}], label=label)
                    email_count += 1
                except OSError as e:
                    logger.debug(f"Error reading {email_file}: {e}")

            logger.info(f"Loaded {email_count} {label} emails from Enron")

        return self.text_dataset

    def load_uci_phishing(self, path: Optional[Union[str, Path]] = None) -> TextDataset:
        """Load UCI Phishing Websites dataset. Expects cache/uci_phishing.data."""
        logger.info("Loading UCI Phishing Websites dataset...")
        candidates = []
        if path:
            candidates.append(Path(path))
        candidates.extend([
            self.config.cache_dir / "uci_phishing.data",
            self.config.cache_dir / "uci_phishing.csv",
            Path("data") / "phishing" / "uci_phishing.data",
            Path("data") / "uci_phishing.data",
        ])
        target_path = None
        for c in candidates:
            if c.exists() and c.is_file():
                target_path = c
                break

        if not target_path:
            logger.warning(
                f"UCI phishing dataset not found. Download from {self.DATASETS['uci_phishing']['url']}."
            )
            return self.text_dataset

        try:
            df = pd.read_csv(target_path, header=None)
            for _, row in df.iterrows():
                label = "phishing" if row.iloc[-1] == -1 else "legitimate"
                text = " ".join(str(v) for v in row.iloc[:-1])
                self.text_dataset.add_records([{"text": text}], label=label)
            logger.info(f"Loaded {len(df)} records from UCI phishing dataset")
        except Exception as e:
            logger.error(f"Error loading UCI phishing dataset: {e}")

        return self.text_dataset

    def load_phishtank(self, path: Optional[Union[str, Path]] = None) -> TextDataset:
        """Load PhishTank URL dataset."""
        logger.info("Loading PhishTank dataset...")
        candidates = []
        if path:
            candidates.append(Path(path))
        candidates.extend([
            self.config.cache_dir / "phishtank.csv",
            self.config.cache_dir / "phishtank.json",
            Path("data") / "phishing" / "phishtank.csv",
        ])
        target = next((c for c in candidates if c.exists()), None)
        if not target:
            logger.warning("PhishTank data not found.")
            return self.text_dataset

        try:
            if target.suffix == ".csv":
                df = pd.read_csv(target)
                url_col = next((c for c in ("url", "URL", "phish_detail_url") if c in df.columns), None)
                if url_col:
                    for url in df[url_col].dropna():
                        self.text_dataset.add_records([{"text": str(url)}], label="phishing")
            logger.info(f"Loaded records from {target}")
        except Exception as e:
            logger.error(f"Error loading PhishTank: {e}")
        return self.text_dataset

    def load_all(self) -> TextDataset:
        """Load all available phishing datasets."""
        logger.info("Loading all phishing datasets...")
        self.load_enron_spam()
        self.load_uci_phishing()
        self.load_phishtank()

        logger.info(f"Total phishing samples: {len(self.text_dataset.data)}")
        return self.text_dataset

    def get_splits(self) -> Dict[str, Tuple[List[str], List[str]]]:
        """Get deduplicated, split phishing data."""
        logger.info("Deduplicating phishing data...")
        unique_data, unique_labels = self.text_dataset.deduplicate(method="exact")

        logger.info(f"Phishing data after deduplication: {len(unique_data)}")
        stats = self.text_dataset.get_statistics(unique_data, unique_labels)
        logger.info(f"Phishing statistics: {stats}")

        return self.text_dataset.train_val_test_split(unique_data, unique_labels)

    @staticmethod
    def create_sample_dataset(
        output_dir: Optional[Union[str, Path]] = None,
        samples_per_class: int = 50,
    ) -> Path:
        """Create sample phishing vs legitimate emails for testing."""
        out_path = Path(output_dir or Path("data") / "phishing_sample")
        spam_dir = out_path / "enron_spam" / "spam"
        ham_dir = out_path / "enron_spam" / "ham"
        spam_dir.mkdir(parents=True, exist_ok=True)
        ham_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            (spam_dir / f"phish_{i:04d}.txt").write_text(
                f"Subject: URGENT: Verify your account immediately!\nPlease click http://fake-login-{i}.com to update credentials.",
                encoding="utf-8",
            )
            (ham_dir / f"ham_{i:04d}.txt").write_text(
                f"Subject: Project sync meeting notes #{i}\nHi Team, please find attached the meeting agenda for tomorrow.",
                encoding="utf-8",
            )
        logger.info(f"Created sample phishing dataset at {out_path}")
        return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing Dataset Loader Utility")
    parser.add_argument("--sample", nargs="?", const="data/phishing_sample", help="Generate sample phishing dataset")
    args = parser.parse_args()

    if args.sample:
        target = PhishingDatasetLoader.create_sample_dataset(args.sample)
        print(f"Sample phishing dataset generated at: {target}")
    else:
        loader = PhishingDatasetLoader()
        loader.load_all()
        print(f"Loaded {len(loader.text_dataset.data)} samples.")