"""
PhishingDatasetLoader: loads phishing detection datasets used by
the Attack / Beast / Warhammer escalation pipeline.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from util.datasets.dataset_config import DatasetConfig
from util.datasets.text import TextDataset
from util.logging_config import get_logger

logger = get_logger(__name__)

PHISHING_ALIASES = {
    "phishing", "phish", "spam", "malicious", "bad", "fraud", "scam", "1", "1.0", "true",
    "yes", "-1", "phishing email", "phishing_email", "phishing url", "unsafe",
}
LEGITIMATE_ALIASES = {
    "legitimate", "legit", "ham", "benign", "good", "safe", "normal", "0", "0.0", "false",
    "no", "safe email", "safe_email", "not phishing", "clean",
}
TEXT_COLUMNS = ("text", "body", "email", "email text", "email_text", "content", "message", "url", "urls")
LABEL_COLUMNS = ("label", "labels", "class", "type", "email type", "email_type", "target", "is_phishing", "status", "category")
GENERIC_SUFFIXES = {".csv", ".tsv", ".jsonl", ".json"}
# Files and folders handled by the dedicated loaders, never re-read as generic files.
SPECIAL_NAMES = {"phishtank.csv", "phishtank.json", "uci_phishing.csv", "raw", "enron_spam", "cart", "jaw"}


def normalize_phishing_label(label) -> Optional[str]:
    """Map common label spellings to 'phishing' / 'legitimate'; None if unrecognized."""
    cleaned = str(label).strip().lower()
    if cleaned in PHISHING_ALIASES:
        return "phishing"
    if cleaned in LEGITIMATE_ALIASES:
        return "legitimate"
    if any(k in cleaned for k in ("phish", "spam", "malic", "fraud", "scam")):
        return "phishing"
    if any(k in cleaned for k in ("legit", "ham", "benign", "safe")):
        return "legitimate"
    return None


def _pick_column(columns, preferred) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in columns}
    return next((lowered[name] for name in preferred if name in lowered), None)


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
            logger.debug(
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
            logger.debug(
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
            logger.debug("PhishTank data not found.")
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

    def load_file(self, path: Union[str, Path]) -> int:
        """
        Load a generic CSV/TSV/JSONL/JSON file with a text column and a label column.

        Text columns tried: text, body, email text, content, message, url.
        Label columns tried: label, class, type, email type, target, is_phishing.
        Labels such as spam/phishing/1 and ham/legitimate/safe/0 are normalized.
        Returns the number of records added.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        try:
            if suffix == ".jsonl":
                df = pd.read_json(path, lines=True)
            elif suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    payload = next((v for v in payload.values() if isinstance(v, list)), [])
                df = pd.DataFrame(payload)
            else:
                df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", on_bad_lines="skip",
                                 encoding_errors="ignore", engine="python")
        except Exception as e:
            logger.warning(f"Skipping {path}: could not parse ({e})")
            return 0

        text_col = _pick_column(df.columns, TEXT_COLUMNS)
        label_col = _pick_column(df.columns, LABEL_COLUMNS)
        if text_col is None or label_col is None:
            logger.warning(
                f"Skipping {path}: need a text column ({', '.join(TEXT_COLUMNS)}) and a label column "
                f"({', '.join(LABEL_COLUMNS)}); found {list(df.columns)}"
            )
            return 0

        added, unknown = 0, set()
        for text, raw_label in zip(df[text_col], df[label_col]):
            if pd.isna(text) or pd.isna(raw_label):
                continue
            label = normalize_phishing_label(raw_label)
            if label is None:
                unknown.add(str(raw_label))
                continue
            before = len(self.text_dataset.data)
            self.text_dataset.add_records([{"text": str(text)}], label=label)
            added += len(self.text_dataset.data) - before
        if unknown:
            logger.warning(f"{path}: ignored rows with unrecognized labels {sorted(unknown)[:10]}")
        logger.info(f"Loaded {added} records from {path}")
        return added

    def load_directory(self, root: Union[str, Path]) -> int:
        """Load every generic CSV/TSV/JSONL/JSON file under root (recursively)."""
        root = Path(root)
        if not root.is_dir():
            return 0
        total = 0
        for path in sorted(root.rglob("*")):
            rel_parts = {p.lower() for p in path.relative_to(root).parts}
            if (path.is_file() and path.suffix.lower() in GENERIC_SUFFIXES
                    and not rel_parts & SPECIAL_NAMES and not path.name.startswith(".")):
                total += self.load_file(path)
        return total

    def load_all(self) -> TextDataset:
        """Load all available phishing datasets."""
        logger.info("Loading all phishing datasets...")
        self.load_enron_spam()
        self.load_uci_phishing()
        self.load_phishtank()
        self.load_directory(Path("data") / "phishing")

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