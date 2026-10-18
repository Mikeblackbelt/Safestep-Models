"""
CARTDatasetLoader: loads AI-generated text detection datasets
(HC3, GPT-2 Output, RAID, MAGE, CSV/JSONL files, custom folders, and synthetic sample data).
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from util.datasets.dataset_config import DatasetConfig
from util.datasets.text import TextDataset
from util.logging_config import get_logger

logger = get_logger(__name__)

# Label normalization sets for text
HUMAN_ALIASES = {"human", "real", "authentic", "genuine", "original", "0"}
AI_GENERATED_ALIASES = {"ai_generated", "ai", "fake", "synthetic", "chatgpt", "gpt", "generated", "1"}


def normalize_text_label(label: Union[str, int]) -> str:
    """Normalize common text classification labels into standard CART labels ('human' or 'ai_generated')."""
    cleaned = str(label).strip().lower()
    if cleaned in HUMAN_ALIASES:
        return "human"
    if cleaned in AI_GENERATED_ALIASES:
        return "ai_generated"
    if any(k in cleaned for k in ("fake", "ai", "synth", "gpt", "gen")):
        return "ai_generated"
    if any(k in cleaned for k in ("human", "real", "auth", "orig")):
        return "human"
    return cleaned


class CARTDatasetLoader:
    """Loader for CART (AI-generated text detection) datasets."""

    DATASETS = {
        "hc3": {
            "url": "https://github.com/Hello-SimpleAI/ChatGPT-Comparison-Corpus/raw/main/HC3/all_data.json",
            "description": "Human ChatGPT Comparison Corpus",
        },
        "raid": {
            "url": "https://huggingface.co/datasets/RAID-Institute/RAID",
            "description": "RAID benchmark dataset for AI text detection",
        },
        "mage": {
            "url": "https://huggingface.co/datasets/MAGE-Institute/MAGE",
            "description": "MAGE dataset across multiple LLMs",
        },
        "gpt2_output": {
            "url": "https://github.com/openai/gpt-2-output-dataset",
            "description": "GPT-2 Output Dataset (human WebText vs GPT-2 generated)",
        },
    }

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.text_dataset = TextDataset(self.config)

    def _resolve_candidate_files(self, name: str, custom_path: Optional[Union[str, Path]] = None, extensions: Tuple[str, ...] = (".json", ".jsonl", ".csv", ".txt")) -> List[Path]:
        """Find candidate files for a dataset name."""
        candidates = []
        if custom_path:
            p = Path(custom_path)
            if p.is_file():
                candidates.append(p)
            elif p.is_dir():
                for ext in extensions:
                    candidates.extend(list(p.glob(f"*{ext}")))
        search_dirs = [
            self.config.cache_dir,
            self.config.raw_data_dir,
            self.config.data_root,
            Path("data") / "cart",
            Path("data"),
        ]
        for d in search_dirs:
            if not d.exists():
                continue
            for ext in extensions:
                candidates.append(d / f"{name}{ext}")
                candidates.extend(list(d.glob(f"{name}*{ext}")))

        existing = []
        seen = set()
        for c in candidates:
            if c.exists() and c.is_file() and c.resolve() not in seen:
                seen.add(c.resolve())
                existing.append(c)
        return existing

    def load_hc3(self, path: Optional[Union[str, Path]] = None) -> TextDataset:
        """Load HC3 dataset (JSON, JSONL, or folder)."""
        logger.info("Loading HC3 (Human ChatGPT Comparison Corpus)...")
        candidates = self._resolve_candidate_files("hc3", path)

        if not candidates:
            logger.warning(
                f"HC3 dataset not found. Download from {self.DATASETS['hc3']['url']} "
                f"and place it at data/cart/hc3.json or {self.config.cache_dir}/hc3.json."
            )
            return self.text_dataset

        target_file = candidates[0]
        logger.info(f"Loading HC3 from {target_file}")

        try:
            if target_file.suffix.lower() == ".jsonl":
                with open(target_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if "human_answers" in row:
                            for ans in row["human_answers"]:
                                self.text_dataset.add_records([{"text": ans}], label="human")
                        if "chatgpt_answers" in row:
                            for ans in row["chatgpt_answers"]:
                                self.text_dataset.add_records([{"text": ans}], label="ai_generated")
            else:
                with open(target_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, dict):
                    if "conversations" in data:
                        for conv in data["conversations"]:
                            if "human" in conv:
                                self.text_dataset.add_records([{"text": conv["human"]}], label="human")
                            if "chatgpt" in conv:
                                self.text_dataset.add_records([{"text": conv["chatgpt"]}], label="ai_generated")
                    else:
                        for k, v in data.items():
                            if isinstance(v, list):
                                for item in v:
                                    if isinstance(item, dict):
                                        if "human" in item:
                                            self.text_dataset.add_records([{"text": item["human"]}], label="human")
                                        if "chatgpt" in item:
                                            self.text_dataset.add_records([{"text": item["chatgpt"]}], label="ai_generated")
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            if "human_answers" in item:
                                for ans in item["human_answers"]:
                                    self.text_dataset.add_records([{"text": ans}], label="human")
                            if "chatgpt_answers" in item:
                                for ans in item["chatgpt_answers"]:
                                    self.text_dataset.add_records([{"text": ans}], label="ai_generated")
        except Exception as exc:
            logger.error(f"Error loading HC3 from {target_file}: {exc}")

        return self.text_dataset

    def load_gpt2_output(self, path: Optional[Union[str, Path]] = None) -> TextDataset:
        """Load GPT-2 Output Dataset."""
        logger.info("Loading GPT-2 Output Dataset...")
        candidates = self._resolve_candidate_files("gpt2_output", path)

        if not candidates:
            logger.warning(
                f"GPT-2 Output dataset not found. Download from {self.DATASETS['gpt2_output']['url']} "
                f"and place it at data/cart/gpt2_output.json or {self.config.cache_dir}/gpt2_output.json."
            )
            return self.text_dataset

        target_file = candidates[0]
        logger.info(f"Loading GPT-2 Output from {target_file}")

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            samples = data.get("samples", []) if isinstance(data, dict) else data
            for record in samples:
                if isinstance(record, dict):
                    text = record.get("text", "")
                    raw_label = record.get("label", 0)
                    label = normalize_text_label(raw_label)
                    self.text_dataset.add_records([{"text": text}], label=label)
        except Exception as exc:
            logger.error(f"Error loading GPT-2 output from {target_file}: {exc}")

        return self.text_dataset

    def load_raid(self, path: Optional[Union[str, Path]] = None) -> TextDataset:
        """Load RAID benchmark dataset."""
        logger.info("Loading RAID dataset...")
        candidates = self._resolve_candidate_files("raid", path)
        if not candidates:
            logger.warning("RAID dataset not found.")
            return self.text_dataset
        return self.load_file(candidates[0])

    def load_mage(self, path: Optional[Union[str, Path]] = None) -> TextDataset:
        """Load MAGE dataset."""
        logger.info("Loading MAGE dataset...")
        candidates = self._resolve_candidate_files("mage", path)
        if not candidates:
            logger.warning("MAGE dataset not found.")
            return self.text_dataset
        return self.load_file(candidates[0])

    def load_file(
        self,
        file_path: Union[str, Path],
        text_column: str = "text",
        label_column: str = "label",
    ) -> TextDataset:
        """Load from a CSV, JSONL, or JSON file."""
        path = Path(file_path)
        if not path.is_file():
            logger.warning(f"File {path} does not exist.")
            return self.text_dataset

        ext = path.suffix.lower()
        if ext == ".csv":
            return self.load_csv(path, text_column=text_column, label_column=label_column)
        if ext in (".jsonl", ".ndjson"):
            return self.load_jsonl(path, text_key=text_column, label_key=label_column)
        if ext == ".json":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            t = item.get(text_column, "")
                            l = normalize_text_label(item.get(label_column, "human"))
                            self.text_dataset.add_records([{"text": t}], label=l)
            except Exception as exc:
                logger.error(f"Error loading {path}: {exc}")
        return self.text_dataset

    def load_csv(
        self,
        csv_path: Union[str, Path],
        text_column: str = "text",
        label_column: str = "label",
    ) -> TextDataset:
        """Load text and labels from a CSV file."""
        path = Path(csv_path)
        logger.info(f"Loading CSV from {path}")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    text = row.get(text_column)
                    # Try common column fallbacks if missing
                    if text is None:
                        for col in ("content", "body", "sentence", "document"):
                            if col in row:
                                text = row[col]
                                break
                    if not text:
                        continue
                    label = row.get(label_column)
                    if label is None:
                        for col in ("target", "class", "y", "is_ai", "generated"):
                            if col in row:
                                label = row[col]
                                break
                    norm_label = normalize_text_label(label or "human")
                    self.text_dataset.add_records([{"text": text}], label=norm_label)
                    count += 1
            logger.info(f"Loaded {count} rows from {path}")
        except Exception as exc:
            logger.error(f"Failed to load CSV {path}: {exc}")
        return self.text_dataset

    def load_jsonl(
        self,
        jsonl_path: Union[str, Path],
        text_key: str = "text",
        label_key: str = "label",
    ) -> TextDataset:
        """Load text and labels from a JSONL file."""
        path = Path(jsonl_path)
        logger.info(f"Loading JSONL from {path}")
        try:
            count = 0
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = row.get(text_key) or row.get("content") or ""
                    label = row.get(label_key) or row.get("target") or "human"
                    if text:
                        self.text_dataset.add_records([{"text": text}], label=normalize_text_label(label))
                        count += 1
            logger.info(f"Loaded {count} items from {path}")
        except Exception as exc:
            logger.error(f"Failed to load JSONL {path}: {exc}")
        return self.text_dataset

    def load_directory(self, directory: Union[str, Path]) -> TextDataset:
        """
        Load text from a directory.
        Supports:
          - texts.txt and labels.txt
          - Subfolders named by label: dir/{human,ai_generated}/*.txt
          - Any .csv or .jsonl files in the directory
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Directory {dir_path} does not exist.")
            return self.text_dataset

        txt_file = dir_path / "texts.txt"
        lbl_file = dir_path / "labels.txt"
        if txt_file.exists() and lbl_file.exists():
            texts = txt_file.read_text(encoding="utf-8").splitlines()
            labels = lbl_file.read_text(encoding="utf-8").splitlines()
            for t, l in zip(texts, labels):
                self.text_dataset.add_records([{"text": t}], label=normalize_text_label(l))
            return self.text_dataset

        for sub in dir_path.glob("*"):
            if sub.is_dir():
                norm_label = normalize_text_label(sub.name)
                if norm_label in ("human", "ai_generated"):
                    for txt_p in sub.glob("**/*.txt"):
                        try:
                            content = txt_p.read_text(encoding="utf-8", errors="ignore").strip()
                            if content:
                                self.text_dataset.add_records([{"text": content}], label=norm_label)
                        except OSError:
                            pass

        for csv_file in dir_path.glob("*.csv"):
            self.load_csv(csv_file)
        for jsonl_file in dir_path.glob("*.jsonl"):
            self.load_jsonl(jsonl_file)

        return self.text_dataset

    def load_all(self) -> TextDataset:
        """Load all available CART datasets."""
        logger.info("Loading all available CART datasets...")
        self.load_hc3()
        self.load_gpt2_output()
        self.load_raid()
        self.load_mage()

        local_cart = Path("data") / "cart"
        if local_cart.exists():
            self.load_directory(local_cart)

        logger.info(f"Total CART samples loaded: {len(self.text_dataset.data)}")
        return self.text_dataset

    def get_splits(self) -> Dict[str, Tuple[List[str], List[str]]]:
        """Get deduplicated, split CART data."""
        logger.info("Deduplicating CART data...")
        unique_data, unique_labels = self.text_dataset.deduplicate(method="exact")

        logger.info(f"CART data after deduplication: {len(unique_data)}")
        stats = self.text_dataset.get_statistics(unique_data, unique_labels)
        logger.info(f"CART statistics: {stats}")

        return self.text_dataset.train_val_test_split(unique_data, unique_labels)

    @staticmethod
    def create_sample_dataset(
        output_dir: Optional[Union[str, Path]] = None,
        samples_per_class: int = 100,
    ) -> Path:
        """
        Create a ready-to-train sample dataset with human and AI-generated texts
        in standard texts.txt / labels.txt format under train/ and val/ splits.
        """
        out_path = Path(output_dir or Path("data") / "cart_sample")
        train_dir = out_path / "train"
        val_dir = out_path / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        val_count = max(10, int(samples_per_class * 0.2))
        train_count = samples_per_class

        human_templates = [
            "I recently visited the local farmer's market and purchased some delicious fresh berries.",
            "In my experience, debugging concurrency issues requires patience and careful logging.",
            "The committee reviewed the quarterly fiscal reports and approved the expansion plans.",
            "Yesterday afternoon our team completed the hardware upgrade on our primary cluster.",
            "She decided to take the scenic trail along the coast despite the light rain.",
        ]
        ai_templates = [
            "As an artificial intelligence language model, I strive to provide balanced perspectives.",
            "Furthermore, it is important to consider the multifaceted implications of this decision.",
            "In conclusion, modern technological advancement offers both significant opportunities and hurdles.",
            "Consequently, leveraging deep neural network architectures provides robust representations.",
            "To delve deeper into this phenomenon, we must examine several key dimensions.",
        ]

        def _generate_texts(count: int) -> Tuple[List[str], List[str]]:
            texts, labels = [], []
            for i in range(count):
                ht = human_templates[i % len(human_templates)]
                texts.append(f"{ht} Variation index {i}.")
                labels.append("human")

                at = ai_templates[i % len(ai_templates)]
                texts.append(f"{at} Variation index {i}.")
                labels.append("ai_generated")
            return texts, labels

        tr_texts, tr_labels = _generate_texts(train_count)
        va_texts, va_labels = _generate_texts(val_count)

        (train_dir / "texts.txt").write_text("\n".join(tr_texts), encoding="utf-8")
        (train_dir / "labels.txt").write_text("\n".join(tr_labels), encoding="utf-8")
        (val_dir / "texts.txt").write_text("\n".join(va_texts), encoding="utf-8")
        (val_dir / "labels.txt").write_text("\n".join(va_labels), encoding="utf-8")

        logger.info(
            f"Created sample CART dataset at {out_path} "
            f"({len(tr_texts)} train, {len(va_texts)} val samples)"
        )
        return out_path

    def export_to_train_val(self, output_dir: Union[str, Path]) -> Dict[str, int]:
        """Export currently loaded data to standard texts.txt and labels.txt splits."""
        splits = self.get_splits()
        out_path = Path(output_dir)
        counts = {"train": 0, "val": 0, "test": 0}

        for split_name in ("train", "val", "test"):
            if split_name not in splits:
                continue
            texts, labels = splits[split_name]
            split_dir = out_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            norm_labels = [normalize_text_label(l) for l in labels]
            (split_dir / "texts.txt").write_text("\n".join(texts), encoding="utf-8")
            (split_dir / "labels.txt").write_text("\n".join(norm_labels), encoding="utf-8")
            counts[split_name] = len(texts)

        logger.info(f"Exported dataset to {out_path}: {counts}")
        return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CART Dataset Loader Utility")
    parser.add_argument("--sample", nargs="?", const="data/cart_sample", help="Generate sample dataset for quick testing")
    parser.add_argument("--hc3", type=str, help="Load HC3 dataset from path")
    parser.add_argument("--csv", type=str, help="Load dataset from CSV file")
    parser.add_argument("--dir", type=str, help="Load custom text dataset from directory")
    parser.add_argument("--export", type=str, help="Export loaded dataset into standard train/val directory")
    args = parser.parse_args()

    if args.sample:
        target = CARTDatasetLoader.create_sample_dataset(args.sample)
        print(f"Sample CART dataset generated at: {target}")
    else:
        cfg = DatasetConfig()
        loader = CARTDatasetLoader(cfg)
        if args.hc3:
            loader.load_hc3(args.hc3)
        elif args.csv:
            loader.load_csv(args.csv)
        elif args.dir:
            loader.load_directory(args.dir)
        else:
            loader.load_all()

        print(f"Loaded {len(loader.text_dataset.data)} text samples.")
        if args.export:
            loader.export_to_train_val(args.export)
            print(f"Exported to: {args.export}")