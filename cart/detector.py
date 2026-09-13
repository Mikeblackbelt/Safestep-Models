"""
CARTDetector: end-to-end interface for the CART pipeline.

Wraps tokenization -> neural inference -> heuristic extraction ->
fusion into a single predict() call, and exposes save/load for the
trained components. Training itself lives in cart/train.py, which
uses this class's components directly.
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from cart.config import CARTConfig
from cart.heuristics import HeuristicExtractor
from cart.model import CARTNeuralClassifier, CARTTokenizerWrapper
from cart.fusion import CARTFusionModel


class CARTDetector:
    """High-level interface: raw text in, p(ai_generated) + decision out."""

    def __init__(self, config: CARTConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)

        self.tokenizer = CARTTokenizerWrapper(config)
        self.neural_model = CARTNeuralClassifier(config).to(self.device)
        self.heuristics = HeuristicExtractor(config)
        self.fusion = CARTFusionModel(config)

    @torch.no_grad()
    def _neural_probs(self, texts: List[str]) -> np.ndarray:
        self.neural_model.eval()
        encoded = self.tokenizer.encode(texts)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        outputs = self.neural_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs["logits"], dim=-1)
        return probs.cpu().numpy()

    def predict(self, texts: List[str]) -> List[dict]:
        """
        Run the full hybrid pipeline on a batch of texts.

        Returns:
            List of dicts: {text, p_human, p_ai, label, decision_threshold}
        """
        neural_probs = self._neural_probs(texts)
        heuristic_feats = self.heuristics.extract_batch(texts)
        fused = self.fusion.build_features(neural_probs, heuristic_feats)

        final_probs = self.fusion.predict_proba(fused)
        results = []
        for text, probs in zip(texts, final_probs):
            p_human, p_ai = float(probs[0]), float(probs[1])
            label = "ai_generated" if p_ai >= self.config.decision_threshold else "human"
            results.append({
                "text": text,
                "p_human": p_human,
                "p_ai": p_ai,
                "label": label,
                "decision_threshold": self.config.decision_threshold,
            })
        return results

    def save(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.neural_model.state_dict(), save_dir / "neural_model.pt")
        self.fusion.save(save_dir / "fusion_model.json")

    def load(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        self.neural_model.load_state_dict(
            torch.load(save_dir / "neural_model.pt", map_location=self.device)
        )
        self.neural_model.to(self.device)
        self.fusion.load(save_dir / "fusion_model.json")