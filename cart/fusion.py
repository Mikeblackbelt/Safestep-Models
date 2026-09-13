"""
Fusion layer for CART: combines the neural model's softmax output
with the hand-engineered heuristic features via a lightweight
gradient-boosted meta-classifier (XGBoost).

This is trained on top of a held-out split (not the same data used
to fine-tune the neural backbone) to avoid the meta-classifier
overfitting to the neural model's own training distribution.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb

from cart.config import CARTConfig
from cart.heuristics import FEATURE_NAMES


class CARTFusionModel:
    """Gradient-boosted fusion of neural logits + heuristic features."""

    def __init__(self, config: CARTConfig):
        self.config = config
        self.model: Optional[xgb.XGBClassifier] = None
        # neural: p_human, p_ai + heuristic feature block
        self.feature_names = ["p_human_neural", "p_ai_neural"] + FEATURE_NAMES

    def _build_model(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=self.config.fusion_n_estimators,
            max_depth=self.config.fusion_max_depth,
            learning_rate=self.config.fusion_learning_rate,
            subsample=0.8,
            eval_metric="logloss",
        )

    def build_features(
        self,
        neural_probs: np.ndarray,
        heuristic_features: np.ndarray,
    ) -> np.ndarray:
        """
        Concatenate neural softmax probabilities with heuristic features.

        Args:
            neural_probs: (N, 2) array of [p_human, p_ai].
            heuristic_features: (N, num_heuristics) array from HeuristicExtractor.

        Returns:
            (N, 2 + num_heuristics) fused feature matrix.
        """
        return np.concatenate([neural_probs, heuristic_features], axis=1)

    def fit(self, fused_features: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit the meta-classifier.

        Args:
            fused_features: output of build_features().
            labels: binary labels (0=human, 1=ai_generated).
        """
        self.model = self._build_model()
        self.model.fit(fused_features, labels)

    def predict_proba(self, fused_features: np.ndarray) -> np.ndarray:
        """Return (N, 2) array of [p_human, p_ai]."""
        if self.model is None:
            raise RuntimeError("CARTFusionModel has not been fit yet.")
        return self.model.predict_proba(fused_features)

    def predict(self, fused_features: np.ndarray) -> np.ndarray:
        """Return binary predictions using config.decision_threshold."""
        probs = self.predict_proba(fused_features)[:, 1]
        return (probs >= self.config.decision_threshold).astype(int)

    def feature_importance(self) -> dict:
        """Return {feature_name: importance} for interpretability."""
        if self.model is None:
            raise RuntimeError("CARTFusionModel has not been fit yet.")
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names, importances.tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        ))

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfit CARTFusionModel.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: Path) -> None:
        self.model = self._build_model()
        self.model.load_model(str(path))