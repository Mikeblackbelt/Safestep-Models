"""
PhishingDatasetLoader: loads phishing detection datasets used by
the Attack / Beast / Warhammer escalation pipeline.
"""

from typing import Dict, List, Tuple

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

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.text_dataset = TextDataset(config)

    def load_enron_spam(self) -> TextDataset:
        """Load Enron spam dataset. Expects cache/enron_spam/{spam,ham}/*.txt."""
        logger.info("Loading Enron spam corpus...")

        cache_path = self.config.cache_dir / "enron_spam"
        if not cache_path.exists():
            logger.warning(
                f"Enron corpus not found at {cache_path}. "
                f"Download from {self.DATASETS['enron_spam']['url']} and extract there."
            )
            return self.text_dataset

        for split_type, label in [("spam", "phishing"), ("ham", "legitimate")]:
            split_dir = cache_path / split_type
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

    def load_uci_phishing(self) -> TextDataset:
        """Load UCI Phishing Websites dataset. Expects cache/uci_phishing.data."""
        logger.info("Loading UCI Phishing Websites dataset...")

        cache_path = self.config.cache_dir / "uci_phishing.data"
        if not cache_path.exists():
            logger.warning(
                f"UCI phishing dataset not found at {cache_path}. "
                f"Download from {self.DATASETS['uci_phishing']['url']}."
            )
            return self.text_dataset

        try:
            df = pd.read_csv(cache_path, header=None)
        except Exception as e:
            logger.error(f"Error loading UCI phishing dataset: {e}")
            return self.text_dataset

        for _, row in df.iterrows():
            # UCI convention: -1 = phishing, 1 = legitimate (last column)
            label = "phishing" if row.iloc[-1] == -1 else "legitimate"
            text = " ".join(str(v) for v in row.iloc[:-1])
            self.text_dataset.add_records([{"text": text}], label=label)

        logger.info(f"Loaded {len(df)} records from UCI phishing dataset")
        return self.text_dataset

    def load_all(self) -> TextDataset:
        """Load all available phishing datasets."""
        logger.info("Loading all phishing datasets...")
        self.load_enron_spam()
        self.load_uci_phishing()
        # PhishTank / CEAS 2008 / IWSPA-AP follow the same pattern; add
        # load_phishtank()/load_ceas()/load_iwspa() once their cache
        # format is finalized (see DATASETS dict above for source URLs).

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