"""
TextDataset: container + operations for text-based datasets
(used by CART and the phishing text sources).
"""

import hashlib
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from safestep_utils.datasets.config import DatasetConfig
from safestep_utils.logging_config import get_logger

logger = get_logger(__name__)


class TextDataset:
    """Handler for text-based datasets (CART, phishing email/website text)."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data: List[str] = []
        self.labels: List[str] = []

    def add_records(self, records: List[Dict[str, str]], label: str) -> None:
        """
        Add records to the dataset.

        Args:
            records: List of dicts with a 'text' key (or plain strings).
            label: Label for all records in this call (e.g. 'ai_generated', 'phishing').
        """
        for record in records:
            text = record.get("text", "") if isinstance(record, dict) else str(record)
            if text.strip():
                self.data.append(text)
                self.labels.append(label)

    def deduplicate(self, method: str = "exact") -> Tuple[List[str], List[str]]:
        """
        Remove duplicate texts.

        Args:
            method: 'exact' (hash match) or 'jaccard' (token-overlap similarity).

        Returns:
            (unique_texts, unique_labels)
        """
        if method == "exact":
            return self._dedupe_exact()
        if method == "jaccard":
            return self._dedupe_jaccard()
        raise ValueError(f"Unknown deduplication method: {method}")

    def _dedupe_exact(self) -> Tuple[List[str], List[str]]:
        seen = set()
        unique_data, unique_labels = [], []
        for text, label in zip(self.data, self.labels):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash not in seen:
                seen.add(text_hash)
                unique_data.append(text)
                unique_labels.append(label)

        removed = len(self.data) - len(unique_data)
        logger.info(f"Removed {removed} exact duplicates (exact method)")
        return unique_data, unique_labels

    def _dedupe_jaccard(self) -> Tuple[List[str], List[str]]:
        def tokenize(text: str):
            return set(text.lower().split())

        def jaccard_sim(set1, set2) -> float:
            if not set1 and not set2:
                return 1.0
            union = len(set1 | set2)
            return len(set1 & set2) / union if union > 0 else 0.0

        unique_data, unique_labels = [], []
        unique_token_sets = []
        threshold = self.config.text_similarity_threshold

        for text, label in zip(self.data, self.labels):
            tokens = tokenize(text)
            is_duplicate = any(
                jaccard_sim(tokens, existing) >= threshold for existing in unique_token_sets
            )
            if not is_duplicate:
                unique_data.append(text)
                unique_labels.append(label)
                unique_token_sets.append(tokens)

        removed = len(self.data) - len(unique_data)
        logger.info(f"Removed {removed} near-duplicates (Jaccard similarity >= {threshold})")
        return unique_data, unique_labels

    def train_val_test_split(
        self,
        data: List[str],
        labels: List[str],
        stratify_by: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Split data into train/val/test with stratification.

        Args:
            data: Text samples.
            labels: Labels.
            stratify_by: Optional column for stratification (defaults to labels).
            random_state: Random seed.

        Returns:
            {'train': (texts, labels), 'val': (texts, labels), 'test': (texts, labels)}
        """
        if stratify_by is None:
            stratify_by = labels

        val_test_size = self.config.val_ratio + self.config.test_ratio

        train_data, val_test_data, train_labels, val_test_labels, _, val_test_strat = (
            train_test_split(
                data,
                labels,
                stratify_by,
                test_size=val_test_size,
                stratify=stratify_by,
                random_state=random_state,
            )
        )

        val_ratio_within_holdout = self.config.val_ratio / val_test_size
        val_data, test_data, val_labels, test_labels = train_test_split(
            val_test_data,
            val_test_labels,
            test_size=(1 - val_ratio_within_holdout),
            stratify=val_test_strat,
            random_state=random_state,
        )

        splits = {
            "train": (train_data, train_labels),
            "val": (val_data, val_labels),
            "test": (test_data, test_labels),
        }
        logger.info(
            f"Train/val/test split: {len(train_data)}/{len(val_data)}/{len(test_data)}"
        )
        return splits

    def get_statistics(self, texts: List[str], labels: List[str]) -> Dict:
        """Compute dataset statistics."""
        label_counts = Counter(labels)
        text_lengths = [len(text.split()) for text in texts]

        return {
            "total_samples": len(texts),
            "label_distribution": dict(label_counts),
            "avg_text_length": float(np.mean(text_lengths)) if text_lengths else 0.0,
            "min_text_length": int(np.min(text_lengths)) if text_lengths else 0,
            "max_text_length": int(np.max(text_lengths)) if text_lengths else 0,
            "std_text_length": float(np.std(text_lengths)) if text_lengths else 0.0,
        }