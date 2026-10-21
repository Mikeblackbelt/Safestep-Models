"""
Attack: stage 1 of the phishing pipeline.

A character n-gram TF-IDF + logistic regression classifier. Character
n-grams catch obfuscated spellings ("paypa1", "rn" for "m") and URL
fragments without any tokenizer, and the whole model scores thousands
of samples per second on CPU, so it runs on everything and only hands
off the samples it is unsure about.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phishing.config import PhishingConfig

logger = logging.getLogger("safestep.phishing.attack")


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
        """Safely serialize AttackModel parameters into JSON (no pickle/joblib)."""
        if self.model is None:
            raise RuntimeError("Cannot save an unfit AttackModel.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tfidf = self.model.named_steps["tfidf"]
        clf = self.model.named_steps["clf"]
        payload = {
            "format": "safestep_attack_v1",
            "vocabulary": {k: int(v) for k, v in tfidf.vocabulary_.items()},
            "idf": tfidf.idf_.tolist(),
            "coef": clf.coef_.tolist(),
            "intercept": clf.intercept_.tolist(),
            "classes": clf.classes_.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load(self, path: Path) -> None:
        """Load AttackModel parameters safely from JSON, with fallback for legacy files."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Attempt safe JSON loading
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "vocabulary" in data and "coef" in data:
                self.model = self._build_model()
                tfidf = self.model.named_steps["tfidf"]
                clf = self.model.named_steps["clf"]

                tfidf.vocabulary_ = {k: int(v) for k, v in data["vocabulary"].items()}
                tfidf.fixed_vocabulary_ = True
                tfidf._tfidf = TfidfTransformer(
                    norm=tfidf.norm,
                    use_idf=tfidf.use_idf,
                    smooth_idf=tfidf.smooth_idf,
                    sublinear_tf=tfidf.sublinear_tf,
                )
                tfidf._tfidf.idf_ = np.array(data["idf"], dtype=np.float64)

                clf.coef_ = np.array(data["coef"], dtype=np.float64)
                clf.intercept_ = np.array(data["intercept"], dtype=np.float64)
                clf.classes_ = np.array(data["classes"])
                return
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        # Legacy fallback if existing checkpoint is joblib binary
        import joblib
        logger.warning(
            f"Loading legacy joblib checkpoint from {path}. "
            "Consider re-saving in safe JSON format to avoid pickle deserialization."
        )
        self.model = joblib.load(path)
