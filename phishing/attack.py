"""
Attack: stage 1 of the phishing pipeline.

A character n-gram TF-IDF + logistic regression classifier. Character
n-grams catch obfuscated spellings ("paypa1", "rn" for "m") and URL
fragments without any tokenizer, and the whole model scores thousands
of samples per second on CPU, so it runs on everything and only hands
off the samples it is unsure about.
"""

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phishing.config import PhishingConfig


class AttackModel:
    """Fast lexical first-pass classifier."""

    def __init__(self, config: PhishingConfig):
        self.config = config
        self.model: Optional[Pipeline] = None

    def _build_model(self) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(self.config.attack_ngram_min, self.config.attack_ngram_max),
                max_features=self.config.attack_max_features,
                sublinear_tf=True,
                lowercase=True,
            )),
            ("clf", LogisticRegression(
                C=self.config.attack_C,
                max_iter=1000,
                class_weight="balanced",
            )),
        ])

    def fit(self, texts: List[str], labels: np.ndarray) -> None:
        """Fit on raw texts with binary labels (0=legitimate, 1=phishing)."""
        self.model = self._build_model()
        self.model.fit(texts, labels)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return (N, 2) array of [p_legitimate, p_phishing]."""
        if self.model is None:
            raise RuntimeError("AttackModel has not been fit yet.")
        return self.model.predict_proba(texts)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfit AttackModel.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
