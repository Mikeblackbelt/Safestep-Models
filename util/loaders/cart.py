"""
CARTDatasetLoader: loads AI-generated text detection datasets
(HC3, GPT-2 Output Dataset, plus stubs for RAID/MAGE).
"""

import json
from typing import Dict, List, Tuple

from util.datasets.dataset_config import DatasetConfig
from util.datasets.text import TextDataset
from util.logging_config import get_logger

logger = get_logger(__name__)


class CARTDatasetLoader:
    """Loader for CART (AI-generated text detection) datasets."""

    DATASETS = {
        "hc3": {
            "url": "https://github.com/Hello-SimpleAI/ChatGPT-Comparison-Corpus/raw/main/HC3/all_data.json",
            "description": "Human ChatGPT Comparison Corpus",
        },
        "raid": {
            "url": "https://huggingface.co/datasets/RAID-Institute/RAID/raw/main/data.json",
            "description": "RAID benchmark",
        },
        "mage": {
            "url": "https://huggingface.co/datasets/MAGE-Institute/MAGE/raw/main/mage_data.json",
            "description": "MAGE dataset",
        },
        "gpt2_output": {
            "url": "https://github.com/openai/gpt-2-output-dataset/raw/master/data.json",
            "description": "GPT-2 Output Dataset",
        },
    }

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.text_dataset = TextDataset(config)

    def load_hc3(self) -> TextDataset:
        """Load HC3 dataset (expects a pre-downloaded cache file)."""
        logger.info("Loading HC3 (Human ChatGPT Comparison Corpus)...")

        cache_path = self.config.cache_dir / "hc3.json"
        if not cache_path.exists():
            logger.warning(
                f"HC3 not found at {cache_path}. "
                f"Download from {self.DATASETS['hc3']['url']} and place it there."
            )
            return self.text_dataset

        with open(cache_path) as f:
            data = json.load(f)

        if isinstance(data, dict) and "conversations" in data:
            for conv in data.get("conversations", []):
                if "human" in conv:
                    self.text_dataset.add_records([{"text": conv["human"]}], label="human")
                if "chatgpt" in conv:
                    self.text_dataset.add_records(
                        [{"text": conv["chatgpt"]}], label="ai_generated"
                    )

        return self.text_dataset

    def load_gpt2_output(self) -> TextDataset:
        """Load GPT-2 Output Dataset (expects a pre-downloaded cache file)."""
        logger.info("Loading GPT-2 Output Dataset...")

        cache_path = self.config.cache_dir / "gpt2_output.json"
        if not cache_path.exists():
            logger.warning(
                f"GPT-2 Output not found at {cache_path}. "
                f"Download from {self.DATASETS['gpt2_output']['url']} and place it there."
            )
            return self.text_dataset

        with open(cache_path) as f:
            data = json.load(f)

        for record in data.get("samples", []):
            text = record.get("text", "")
            label = "ai_generated" if record.get("label", 0) == 1 else "human"
            self.text_dataset.add_records([{"text": text}], label=label)

        return self.text_dataset

    def load_all(self) -> TextDataset:
        """Load all available CART datasets."""
        logger.info("Loading all CART datasets...")
        self.load_hc3()
        self.load_gpt2_output()
        # RAID / MAGE follow the same pattern; add load_raid()/load_mage()
        # once their cache format is finalized.

        logger.info(f"Total CART samples: {len(self.text_dataset.data)}")
        return self.text_dataset

    def get_splits(self) -> Dict[str, Tuple[List[str], List[str]]]:
        """Get deduplicated, split CART data."""
        logger.info("Deduplicating CART data...")
        unique_data, unique_labels = self.text_dataset.deduplicate(method="exact")

        logger.info(f"CART data after deduplication: {len(unique_data)}")
        stats = self.text_dataset.get_statistics(unique_data, unique_labels)
        logger.info(f"CART statistics: {stats}")

        return self.text_dataset.train_val_test_split(unique_data, unique_labels)