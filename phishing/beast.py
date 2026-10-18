"""
Beast: stage 2 of the phishing pipeline.

A gradient-boosted classifier (XGBoost) over Attack's probabilities
plus the structural indicators from phishing/indicators.py. It only
sees samples Attack escalated, and is fit on a split held out from
Attack's training data so it learns how far to trust Attack's score
rather than memorising Attack's training-set confidence.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb

from phishing.config import PhishingConfig
from phishing.indicators import FEATURE_NAMES


class BeastModel:
    """Gradient-boosted classifier over Attack's score + phishing indicators."""

    def __init__(self, config: PhishingConfig):
        self.config = config
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names = ["p_legitimate_attack", "p_phishing_attack"] + FEATURE_NAMES

    def _build_model(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=self.config.beast_n_estimators,
            max_depth=self.config.beast_max_depth,
            learning_rate=self.config.beast_learning_rate,
            subsample=0.8,
            eval_metric="logloss",
        )

    def build_features(self, attack_probs: np.ndarray, indicator_features: np.ndarray) -> np.ndarray:
        """
        Concatenate Attack's probabilities with indicator features.

        Args:
            attack_probs: (N, 2) array of [p_legitimate, p_phishing].
            indicator_features: (N, num_indicators) array from IndicatorExtractor.

        Returns:
            (N, 2 + num_indicators) feature matrix.
        """
        return np.concatenate([attack_probs, indicator_features], axis=1)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit on build_features() output with binary labels (0=legitimate, 1=phishing)."""
        self.model = self._build_model()
        self.model.fit(features, labels)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return (N, 2) array of [p_legitimate, p_phishing]."""
        if self.model is None:
            raise RuntimeError("BeastModel has not been fit yet.")
        return self.model.predict_proba(features)

    def feature_importance(self) -> dict:
        """Return {feature_name: importance} for interpretability."""
        if self.model is None:
            raise RuntimeError("BeastModel has not been fit yet.")
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names, importances.tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        ))

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfit BeastModel.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: Path) -> None:
        self.model = self._build_model()
        self.model.load_model(str(path))
