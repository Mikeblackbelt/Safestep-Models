"""
External Dataset Ingestion and Validation Module for SafeStep.

Provides robust dataset loading, schema auto-inference, label normalization,
corrupted line resilience, and evaluation execution for CART, JAW, and Phishing.
"""

import csv
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from training.evaluation import compute_evaluation_metrics, export_roc_curve
from util.logging_config import get_logger

logger = get_logger(__name__)


class DatasetRecord(dict):
    """Dictionary subclass supporting attribute-style property access."""

    @property
    def label(self) -> int:
        return self["label"]

    @property
    def text(self) -> str:
        return self.get("text", "")

    @property
    def image(self) -> str:
        return self.get("image", "")


def normalize_label(val: Any, model_type: str = "cart") -> Optional[int]:
    """
    Normalizes label values (integers, floats, strings) to binary 0 or 1.

    Args:
        val: Raw label value from dataset.
        model_type: 'cart', 'jaw', or 'phishing'.

    Returns:
        0 (negative / clean / authentic), 1 (positive / malicious / synthetic),
        or None if the label cannot be recognized.
    """
    if val is None:
        return None

    # Handle numeric inputs directly
    if isinstance(val, (int, np.integer)):
        if val in (0, 1):
            return int(val)
    elif isinstance(val, (float, np.floating)):
        if val == 0.0:
            return 0
        elif val == 1.0:
            return 1

    s = str(val).strip().lower()
    m = model_type.lower().strip()

    if m == "cart":
        if s in ("0", "0.0", "human", "clean", "ham", "authentic", "real", "original") or "human" in s:
            return 0
        if s in ("1", "1.0", "ai", "chatgpt", "synthetic", "generated", "fake", "gpt", "bot", "llm", "ai_generated") or "ai" in s or "synthetic" in s or "generated" in s:
            return 1
    elif m == "jaw":
        if s in ("0", "0.0", "authentic", "real", "photo", "original", "clean") or "real" in s or "authentic" in s:
            return 0
        if s in ("1", "1.0", "fake", "midjourney", "deepfake", "synthetic", "ai", "generated", "dall-e", "stablediffusion") or "fake" in s or "deepfake" in s:
            return 1
    elif m in ("phishing", "phish"):
        if s in ("0", "0.0", "legitimate", "legit", "ham", "safe", "clean", "normal") or "legit" in s or "ham" in s:
            return 0
        if s in ("1", "1.0", "phishing", "malicious", "fraud", "spam", "phish", "attack", "bad") or "phish" in s or "malicious" in s:
            return 1

    return None


class ExternalDatasetIngester:
    """
    External Benchmark Dataset Ingestion Engine.

    Handles CSV, JSON, JSONL, and directory datasets with schema auto-inference,
    robust label mapping, and graceful skipping of corrupted rows.
    """

    CONTENT_COLUMNS = [
        "text", "content", "message_body", "body", "message", "sentence",
        "email", "url", "image", "image_path", "filepath", "file", "path"
    ]

    LABEL_COLUMNS = [
        "label", "is_bot", "target", "class", "category", "is_phish",
        "is_ai", "is_synthetic", "ground_truth", "status"
    ]

    @classmethod
    def normalize_label(cls, val: Any, model_type: str = "cart") -> Optional[int]:
        return normalize_label(val, model_type=model_type)

    @classmethod
    def load_external_dataset(
        cls,
        path_or_id: str,
        model_type: str = "cart",
    ) -> List[DatasetRecord]:
        """
        Loads an external benchmark dataset from a file path or benchmark ID.

        Args:
            path_or_id: Path to CSV, JSONL, JSON, or directory.
            model_type: 'cart', 'jaw', or 'phishing'.

        Returns:
            List of DatasetRecord items with 'text'/'image' and normalized 'label'.
        """
        path = Path(path_or_id)
        if not path.exists():
            raise FileNotFoundError(f"External dataset path not found: {path_or_id}")

        if path.is_file() and path.stat().st_size == 0:
            raise ValueError(f"External dataset file is empty (0 bytes): {path_or_id}")

        suffix = path.suffix.lower()
        records: List[DatasetRecord] = []

        if suffix == ".csv":
            records = cls._load_csv(path, model_type)
        elif suffix in (".jsonl", ".ndjson"):
            records = cls._load_jsonl(path, model_type)
        elif suffix == ".json":
            records = cls._load_json(path, model_type)
        elif path.is_dir():
            records = cls._load_directory(path, model_type)
        else:
            # Attempt CSV first, then JSONL
            try:
                records = cls._load_csv(path, model_type)
            except Exception:
                records = cls._load_jsonl(path, model_type)

        if len(records) == 0:
            raise ValueError(f"No usable valid samples found in dataset '{path_or_id}'.")

        return records

    @classmethod
    def _load_csv(cls, path: Path, model_type: str) -> List[DatasetRecord]:
        records: List[DatasetRecord] = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            try:
                raw_headers = next(reader)
            except StopIteration:
                raise ValueError(f"CSV file is empty: {path}")

            # Strip and clean headers
            headers = [h.strip().lower() for h in raw_headers]

            # Detect content column
            content_idx = None
            content_key = "text" if model_type != "jaw" else "image"
            for col in cls.CONTENT_COLUMNS:
                if col in headers:
                    content_idx = headers.index(col)
                    if "image" in col or "file" in col or "path" in col:
                        content_key = "image"
                    break

            # Detect label column
            label_idx = None
            for col in cls.LABEL_COLUMNS:
                if col in headers:
                    label_idx = headers.index(col)
                    break

            if label_idx is None:
                raise ValueError(
                    f"No identifiable label column found in CSV headers: {raw_headers}"
                )
            if content_idx is None:
                # If only 2 columns, infer the other one as content
                non_label = [i for i in range(len(headers)) if i != label_idx]
                if non_label:
                    content_idx = non_label[0]
                else:
                    raise ValueError(
                        f"No identifiable content column found in CSV headers: {raw_headers}"
                    )

            for line_no, row in enumerate(reader, start=2):
                if not row or len(row) <= max(content_idx, label_idx):
                    continue
                raw_content = row[content_idx].strip()
                raw_label = row[label_idx].strip()

                norm = normalize_label(raw_label, model_type)
                if norm is None:
                    warnings.warn(
                        f"Skipping row {line_no} with unrecognized label '{raw_label}'.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

                records.append(
                    DatasetRecord(
                        {
                            content_key: raw_content,
                            "label": norm,
                            "raw_label": raw_label,
                        }
                    )
                )

        return records

    @classmethod
    def _load_jsonl(cls, path: Path, model_type: str) -> List[DatasetRecord]:
        records: List[DatasetRecord] = []
        content_key = "text" if model_type != "jaw" else "image"

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    obj = json.loads(clean_line)
                except Exception as exc:
                    warnings.warn(
                        f"Skipping corrupted JSONL line {line_no}: {exc}",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

                if not isinstance(obj, dict):
                    continue

                # Detect content
                c_val = None
                for col in cls.CONTENT_COLUMNS:
                    if col in obj:
                        c_val = str(obj[col]).strip()
                        if "image" in col or "file" in col or "path" in col:
                            content_key = "image"
                        break

                # Detect label
                l_val = None
                for col in cls.LABEL_COLUMNS:
                    if col in obj:
                        l_val = obj[col]
                        break

                if l_val is None or c_val is None:
                    continue

                norm = normalize_label(l_val, model_type)
                if norm is None:
                    continue

                records.append(
                    DatasetRecord(
                        {
                            content_key: c_val,
                            "label": norm,
                            "raw_label": str(l_val),
                        }
                    )
                )

        return records

    @classmethod
    def _load_json(cls, path: Path, model_type: str) -> List[DatasetRecord]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for k in ["data", "samples", "rows", "items"]:
                if k in data and isinstance(data[k], list):
                    items = data[k]
                    break
            else:
                items = [data]
        else:
            raise ValueError(f"Unexpected JSON format in {path}")

        records: List[DatasetRecord] = []
        content_key = "text" if model_type != "jaw" else "image"
        for obj in items:
            if not isinstance(obj, dict):
                continue
            c_val = None
            for col in cls.CONTENT_COLUMNS:
                if col in obj:
                    c_val = str(obj[col]).strip()
                    break
            l_val = None
            for col in cls.LABEL_COLUMNS:
                if col in obj:
                    l_val = obj[col]
                    break
            if c_val is None or l_val is None:
                continue
            norm = normalize_label(l_val, model_type)
            if norm is None:
                continue
            records.append(
                DatasetRecord({content_key: c_val, "label": norm, "raw_label": str(l_val)})
            )
        return records

    @classmethod
    def _load_directory(cls, path: Path, model_type: str) -> List[DatasetRecord]:
        records: List[DatasetRecord] = []
        content_key = "image" if model_type == "jaw" else "text"

        for sub in path.iterdir():
            if sub.is_dir():
                norm = normalize_label(sub.name, model_type)
                if norm is not None:
                    for item in sub.iterdir():
                        if item.is_file():
                            val = str(item) if model_type == "jaw" else item.read_text(errors="replace")
                            records.append(
                                DatasetRecord({content_key: val, "label": norm, "source": str(item)})
                            )

        return records


def evaluate_external_dataset(
    model: Any,
    dataset_path: str,
    model_type: str = "cart",
    output_roc: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates a model instance against an external benchmark dataset.

    Args:
        model: Trained detector instance (implementing predict_proba or predict).
        dataset_path: Path to benchmark dataset file.
        model_type: 'cart', 'jaw', or 'phishing'.
        output_roc: Optional path to export JSON/CSV ROC curve coordinates.

    Returns:
        Evaluation metrics dictionary including accuracy, precision, recall, f1, auroc,
        and roc_curve.
    """
    samples = ExternalDatasetIngester.load_external_dataset(
        dataset_path, model_type=model_type
    )

    y_true = np.array([s["label"] for s in samples], dtype=int)
    content_key = "image" if model_type == "jaw" else "text"
    inputs = [s.get(content_key, "") for s in samples]

    if hasattr(model, "predict_proba"):
        y_pred = model.predict_proba(inputs)
    elif hasattr(model, "predict"):
        y_pred = model.predict(inputs)
    elif callable(model):
        y_pred = model(inputs)
    else:
        raise AttributeError("Model does not expose predict_proba, predict, or __call__")

    metrics = compute_evaluation_metrics(y_true, y_pred)

    if output_roc and "roc_curve" in metrics:
        fmt = "csv" if output_roc.endswith(".csv") else "json"
        export_roc_curve(metrics["roc_curve"], output_roc, format=fmt)

    return metrics
