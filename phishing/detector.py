"""
PhishingDetector: end-to-end interface for the Attack -> Beast -> Warhammer
escalation pipeline.

Every sample is scored by Attack. Samples whose p(phishing) lands inside
Attack's uncertainty band escalate to Beast; those still inside Beast's
band escalate to Warhammer. Each result records which stage made the
final call. Training lives in phishing/train.py, which uses this class's
components directly.
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from phishing.attack import AttackModel
from phishing.beast import BeastModel
from phishing.config import PhishingConfig
from phishing.indicators import IndicatorExtractor
from phishing.warhammer import WarhammerClassifier, WarhammerTokenizerWrapper

STAGES = ("attack", "beast", "warhammer")


class PhishingDetector:
    """High-level interface: raw email/URL text in, p(phishing) + decision out."""

    def __init__(self, config: PhishingConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)

        self.attack = AttackModel(config)
        self.indicators = IndicatorExtractor()
        self.beast = BeastModel(config)

        self.tokenizer = None
        self.warhammer = None
        if config.use_warhammer:
            self.tokenizer = WarhammerTokenizerWrapper(config)
            self.warhammer = WarhammerClassifier(config).to(self.device)
        # An untrained head is worse than Beast's verdict, so Warhammer only
        # joins the pipeline once it has been trained or loaded.
        self.warhammer_ready = False

    @torch.no_grad()
    def _warhammer_probs(self, texts: List[str]) -> np.ndarray:
        self.warhammer.eval()
        chunks = []
        batch_size = self.config.eval_batch_size
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer.encode(texts[start:start + batch_size])
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            outputs = self.warhammer(input_ids=input_ids, attention_mask=attention_mask)
            chunks.append(F.softmax(outputs["logits"], dim=-1).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    @staticmethod
    def _uncertain(p: np.ndarray, low: float, high: float) -> np.ndarray:
        return (p > low) & (p < high)

    def predict(self, texts: List[str], max_stage: str = "warhammer") -> List[dict]:
        """
        Run the escalation pipeline on a batch of texts.

        Args:
            texts: Raw email bodies, URLs, or page text.
            max_stage: Stop escalating after this stage ("attack", "beast", or "warhammer").

        Returns:
            List of dicts: {text, p_legitimate, p_phishing, label, stage, decision_threshold}
        """
        if max_stage not in STAGES:
            raise ValueError(f"max_stage must be one of {STAGES}, got {max_stage!r}")
        if not texts:
            return []

        attack_probs = self.attack.predict_proba(texts)
        p_phishing = attack_probs[:, 1].copy()
        stages = np.array(["attack"] * len(texts), dtype=object)

        escalate = np.flatnonzero(self._uncertain(
            p_phishing, self.config.attack_escalate_low, self.config.attack_escalate_high
        ))
        if escalate.size and max_stage != "attack":
            sub_texts = [texts[i] for i in escalate]
            features = self.beast.build_features(
                attack_probs[escalate], self.indicators.extract_batch(sub_texts)
            )
            p_phishing[escalate] = self.beast.predict_proba(features)[:, 1]
            stages[escalate] = "beast"

            escalate = escalate[self._uncertain(
                p_phishing[escalate], self.config.beast_escalate_low, self.config.beast_escalate_high
            )]
            if escalate.size and max_stage == "warhammer" and self.warhammer_ready:
                p_phishing[escalate] = self._warhammer_probs([texts[i] for i in escalate])[:, 1]
                stages[escalate] = "warhammer"

        results = []
        for text, p, stage in zip(texts, p_phishing, stages):
            p = float(p)
            results.append({
                "text": text,
                "p_legitimate": 1.0 - p,
                "p_phishing": p,
                "label": "phishing" if p >= self.config.decision_threshold else "legitimate",
                "stage": stage,
                "decision_threshold": self.config.decision_threshold,
            })
        return results

    def predict_proba(self, texts: List[str], max_stage: str = "warhammer") -> np.ndarray:
        """
        Return (N, 2) array of [p_legitimate, p_phishing] across the escalation pipeline.
        """
        results = self.predict(texts, max_stage=max_stage)
        if not results:
            return np.empty((0, 2), dtype=float)
        return np.array([[r["p_legitimate"], r["p_phishing"]] for r in results], dtype=float)

    def evaluate(self, texts: List[str], labels: List[str], max_stage: str = "warhammer") -> dict:
        """
        Evaluate the detector on raw texts and labels, returning comprehensive evaluation metrics.
        """
        from training.evaluation import compute_evaluation_metrics

        results = self.predict(texts, max_stage=max_stage)
        if not results:
            return {}
        y_true = np.array([self.config.label2id[l] if isinstance(l, str) else int(l) for l in labels], dtype=np.int64)
        p_phishing = np.array([r["p_phishing"] for r in results], dtype=float)
        metrics = compute_evaluation_metrics(y_true, p_phishing)
        metrics["stage_counts"] = {s: sum(1 for r in results if r["stage"] == s) for s in ("attack", "beast", "warhammer")}
        return metrics

    def save(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.attack.save(save_dir / "attack.joblib")
        self.beast.save(save_dir / "beast.json")
        if self.warhammer is not None and self.warhammer_ready:
            torch.save(self.warhammer.state_dict(), save_dir / "warhammer.pt")

    def load(self, save_dir: Path) -> None:
        save_dir = Path(save_dir)
        self.attack.load(save_dir / "attack.joblib")
        self.beast.load(save_dir / "beast.json")
        warhammer_path = save_dir / "warhammer.pt"
        if self.warhammer is not None and warhammer_path.exists():
            self.warhammer.load_state_dict(torch.load(warhammer_path, map_location=self.device, weights_only=True))
            self.warhammer.to(self.device)
            self.warhammer_ready = True
