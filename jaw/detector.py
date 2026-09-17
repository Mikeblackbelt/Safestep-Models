"""
JAWDetector: end-to-end interface for the JAW pipeline.

Wraps image loading -> preprocessing -> frequency feature extraction ->
CNN forward pass into a single predict() call, with save/load for the
trained model weights.
"""

from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F

from jaw.config import JAWConfig
from jaw.frequency_features import FrequencyFeatureExtractor
from jaw.model import JAWModel
from jaw.preprocessing import JAWPreprocessor


class JAWDetector:
    """High-level interface: image paths in, p(ai_generated) + decision out."""

    def __init__(self, config: JAWConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)

        self.preprocessor = JAWPreprocessor(config)
        self.freq_extractor = FrequencyFeatureExtractor(config)
        self.model = JAWModel(config, freq_feature_dim=self.freq_extractor.feature_dim).to(self.device)

    @torch.no_grad()
    def _predict_from_tensors(
        self,
        image_tensor: torch.Tensor,
        raw_arrays: List[np.ndarray],
    ) -> np.ndarray:
        self.model.eval()
        freq_feats = self.freq_extractor.extract_batch(raw_arrays)
        freq_tensor = torch.from_numpy(freq_feats).to(self.device)
        image_tensor = image_tensor.to(self.device)

        outputs = self.model(images=image_tensor, freq_features=freq_tensor)
        probs = F.softmax(outputs["logits"], dim=-1)
        return probs.cpu().numpy()

    def predict(self, image_paths: List[Union[str, Path]]) -> List[dict]:
        """
        Run the full pipeline on a batch of image paths.

        Returns:
            List of dicts: {path, p_authentic, p_ai, label, decision_threshold}
        """
        image_tensor, raw_arrays = self.preprocessor.load_and_prepare(image_paths)
        probs = self._predict_from_tensors(image_tensor, raw_arrays)

        results = []
        for path, p in zip(image_paths, probs):
            p_authentic, p_ai = float(p[0]), float(p[1])
            label = "ai_generated" if p_ai >= self.config.decision_threshold else "authentic"
            results.append({
                "path": str(path),
                "p_authentic": p_authentic,
                "p_ai": p_ai,
                "label": label,
                "decision_threshold": self.config.decision_threshold,
            })
        return results

    def save(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / "jaw_model.pt")

    def load(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        state_dict = torch.load(save_dir / "jaw_model.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)