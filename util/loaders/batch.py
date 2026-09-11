"""
BatchDataLoader: simple, memory-efficient batch iterator for
text/label pairs (framework-agnostic — no torch/tf dependency).
"""

from typing import List, Tuple

import numpy as np


class BatchDataLoader:
    """Memory-efficient batch loader for training."""

    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_idx = 0

        if shuffle:
            indices = np.random.permutation(len(texts))
            self.texts = [texts[i] for i in indices]
            self.labels = [labels[i] for i in indices]
        else:
            self.texts = list(texts)
            self.labels = list(labels)

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> Tuple[List[str], List[str]]:
        if self.current_idx >= len(self.texts):
            raise StopIteration

        end_idx = min(self.current_idx + self.batch_size, len(self.texts))
        batch_texts = self.texts[self.current_idx:end_idx]
        batch_labels = self.labels[self.current_idx:end_idx]

        self.current_idx = end_idx
        return batch_texts, batch_labels

    def __len__(self) -> int:
        return (len(self.texts) + self.batch_size - 1) // self.batch_size