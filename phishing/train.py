"""
Training pipeline for the phishing escalation pipeline.

Stage 1 (attack):    fit the TF-IDF + logistic regression model on the train split.
Stage 2 (beast):     fit XGBoost on Attack's probabilities + indicators, computed on
                     the validation split (held out from Attack's training).
Stage 3 (warhammer): frozen-head then full fine-tune of DistilBERT on the train
                     split, early-stopped on validation accuracy. Skipped when
                     config.use_warhammer is False.

Usage:
    from phishing.config import PhishingConfig
    from phishing.train import PhishingTrainer

    trainer = PhishingTrainer(PhishingConfig())
    trainer.fit(train_texts, train_labels, val_texts, val_labels)
    trainer.detector.save("checkpoints/phishing")
"""

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from phishing.config import PhishingConfig
from phishing.detector import PhishingDetector
from training.evaluation import compute_evaluation_metrics
from util.logging_config import get_logger

logger = get_logger(__name__)

# Receives progress events: {"type": "stage"|"batch"|"epoch"|"early_stop"|"attack"|"beast"|"pipeline", ...}
ProgressCallback = Callable[[dict], None]


def _linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return LambdaLR(optimizer, lr_lambda)


class PhishingTrainer:
    """Orchestrates Attack, Beast and Warhammer training."""

    def __init__(
        self,
        config: PhishingConfig,
        device: str = "cpu",
        callback: Optional[ProgressCallback] = None,
    ):
        self.config = config
        self.device = device
        self.detector = PhishingDetector(config, device=device)
        self.callback = callback
        self._stage: Optional[str] = None
        self._epoch = 0

    def _emit(self, event_type: str, **payload) -> None:
        if self.callback is not None:
            self.callback({"type": event_type, "stage": self._stage, "epoch": self._epoch, **payload})

    def _label_ids(self, labels: List[str]) -> np.ndarray:
        return np.array([self.config.label2id[l] for l in labels], dtype=np.int64)

    def _iter_batches(self, texts: List[str], labels: List[str], batch_size: int, shuffle: bool):
        n = len(texts)
        indices = np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield [texts[i] for i in batch_idx], [labels[i] for i in batch_idx]

    # -- Stage 1: Attack -------------------------------------------------

    def _fit_attack(self, train_texts, train_labels, val_texts, val_labels) -> Dict[str, Any]:
        logger.info("Phishing Stage 1: fitting Attack (char n-gram TF-IDF + logistic regression)")
        self._stage = "attack"
        self._epoch = 0
        self._emit("stage", epochs=0)
        self.detector.attack.fit(train_texts, self._label_ids(train_labels))

        probs = self.detector.attack.predict_proba(val_texts)[:, 1]
        y_true = self._label_ids(val_labels)
        metrics = compute_evaluation_metrics(y_true, probs)
        accuracy = metrics["accuracy"]
        auroc = metrics.get("auroc", float("nan"))
        auroc_str = f"  auroc: {auroc:.4f}" if auroc == auroc else ""
        logger.info(f"  Attack val accuracy: {accuracy:.4f}{auroc_str}")
        self._emit("attack", val_acc=accuracy, val_auroc=auroc, metrics=metrics)
        return metrics

    # -- Stage 2: Beast --------------------------------------------------

    def _fit_beast(self, val_texts: List[str], val_labels: List[str]) -> Dict[str, Any]:
        """
        Fit Beast on the validation split — held out from Attack's training,
        so Attack's probabilities here reflect how it behaves on unseen data.
        """
        logger.info("Phishing Stage 2: fitting Beast (XGBoost on Attack score + indicators)")
        self._stage = "beast"
        self._epoch = 0
        self._emit("stage", epochs=0)
        attack_probs = self.detector.attack.predict_proba(val_texts)
        indicators = self.detector.indicators.extract_batch(val_texts)
        features = self.detector.beast.build_features(attack_probs, indicators)
        label_ids = self._label_ids(val_labels)

        self.detector.beast.fit(features, label_ids)

        probs = self.detector.beast.predict_proba(features)[:, 1]
        metrics = compute_evaluation_metrics(label_ids, probs)
        accuracy = metrics["accuracy"]
        auroc = metrics.get("auroc", float("nan"))
        auroc_str = f"  auroc: {auroc:.4f}" if auroc == auroc else ""
        logger.info(f"  Beast val accuracy (fit split): {accuracy:.4f}{auroc_str}")
        self._emit("beast", val_acc=accuracy, val_auroc=auroc, metrics=metrics)
        logger.info(f"  Top Beast features: {list(self.detector.beast.feature_importance().items())[:5]}")
        return metrics

    # -- Stage 3: Warhammer ----------------------------------------------

    def _run_warhammer_epoch(self, texts, labels, optimizer, scheduler, criterion) -> float:
        model = self.detector.warhammer
        model.train()
        total_loss = 0.0
        num_batches = 0
        total_batches = math.ceil(len(texts) / self.config.train_batch_size)

        for batch_texts, batch_labels in self._iter_batches(
            texts, labels, self.config.train_batch_size, shuffle=True
        ):
            encoded = self.detector.tokenizer.encode(batch_texts)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            label_ids = torch.from_numpy(self._label_ids(batch_labels)).to(self.device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs["logits"], label_ids)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self._emit(
                "batch", phase="train", batch=num_batches, n_batches=total_batches,
                samples=len(batch_texts), loss=loss.item(),
            )

        return total_loss / max(1, num_batches)

    def _evaluate_warhammer(self, texts: List[str], labels: List[str]) -> dict:
        probs = self.detector._warhammer_probs(texts)
        phish_probs = probs[:, 1]
        label_ids = self._label_ids(labels)
        metrics = compute_evaluation_metrics(label_ids, phish_probs)
        self._emit("batch", phase="eval", samples=len(texts))
        return metrics

    def _train_warhammer(self, train_texts, train_labels, val_texts, val_labels) -> None:
        model = self.detector.warhammer
        criterion = nn.CrossEntropyLoss()

        # Stage 3a: frozen backbone, train classifier head only
        logger.info("Phishing Stage 3a: training Warhammer head (backbone frozen)")
        self._stage = "frozen"
        self._emit("stage", epochs=self.config.num_epochs_frozen)
        model.freeze_encoder()
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
            weight_decay=self.config.weight_decay,
        )
        for epoch in range(self.config.num_epochs_frozen):
            self._epoch = epoch + 1
            loss = self._run_warhammer_epoch(train_texts, train_labels, optimizer, None, criterion)
            metrics = self._evaluate_warhammer(val_texts, val_labels)
            val_acc = metrics["accuracy"]
            val_auroc = metrics.get("auroc", float("nan"))
            auroc_str = f" val_auroc={val_auroc:.4f}" if val_auroc == val_auroc else ""
            logger.info(
                f"  [frozen] epoch {epoch + 1}/{self.config.num_epochs_frozen} "
                f"loss={loss:.4f} val_acc={val_acc:.4f}{auroc_str}"
            )
            self._emit(
                "epoch",
                epochs=self.config.num_epochs_frozen,
                loss=loss,
                val_acc=val_acc,
                val_auroc=val_auroc,
                metrics=metrics,
            )

        # Stage 3b: unfreeze, fine-tune end-to-end at lower LR with warmup
        logger.info("Phishing Stage 3b: fine-tuning full Warhammer backbone")
        self._stage = "finetune"
        self._epoch = 0
        self._emit("stage", epochs=self.config.num_epochs_finetune)
        model.unfreeze_encoder()
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        steps_per_epoch = max(1, math.ceil(len(train_texts) / self.config.train_batch_size))
        total_steps = steps_per_epoch * self.config.num_epochs_finetune
        scheduler = _linear_warmup_scheduler(optimizer, self.config.warmup_steps, total_steps)

        best_val_acc = -1.0
        patience_counter = 0
        for epoch in range(self.config.num_epochs_finetune):
            self._epoch = epoch + 1
            loss = self._run_warhammer_epoch(train_texts, train_labels, optimizer, scheduler, criterion)
            metrics = self._evaluate_warhammer(val_texts, val_labels)
            val_acc = metrics["accuracy"]
            val_auroc = metrics.get("auroc", float("nan"))
            auroc_str = f" val_auroc={val_auroc:.4f}" if val_auroc == val_auroc else ""
            logger.info(
                f"  [finetune] epoch {epoch + 1}/{self.config.num_epochs_finetune} "
                f"loss={loss:.4f} val_acc={val_acc:.4f}{auroc_str}"
            )
            self._emit(
                "epoch",
                epochs=self.config.num_epochs_finetune,
                loss=loss,
                val_acc=val_acc,
                val_auroc=val_auroc,
                metrics=metrics,
            )

            if metrics["accuracy"] > best_val_acc:
                best_val_acc = metrics["accuracy"]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"  Early stopping at epoch {epoch + 1} (patience exceeded)")
                    self._emit("early_stop")
                    break

        self.detector.warhammer_ready = True

    # -- Full pipeline ---------------------------------------------------

    def _evaluate_pipeline(self, val_texts: List[str], val_labels: List[str]) -> Dict[str, Any]:
        self._stage = "pipeline"
        self._epoch = 0
        results = self.detector.predict(val_texts)
        y_true = self._label_ids(val_labels)
        p_phishing = np.array([r["p_phishing"] for r in results], dtype=float)
        metrics = compute_evaluation_metrics(y_true, p_phishing)
        stage_counts = {s: sum(1 for r in results if r["stage"] == s) for s in ("attack", "beast", "warhammer")}
        metrics["stage_counts"] = stage_counts

        # Beast was fit on this split, so this is optimistic; use a test split for a real number.
        auroc = metrics.get("auroc", float("nan"))
        auroc_str = f"  auroc: {auroc:.4f}" if auroc == auroc else ""
        logger.info(
            f"  Pipeline val accuracy: {metrics['accuracy']:.4f}{auroc_str}  "
            f"decided by stage: {stage_counts}"
        )
        self._emit(
            "pipeline",
            val_acc=metrics["accuracy"],
            val_auroc=auroc,
            stage_counts=stage_counts,
            metrics=metrics,
        )
        return metrics

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        """Evaluate the full escalation pipeline on arbitrary texts and labels."""
        return self.detector.evaluate(texts, labels)

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: List[str],
        val_labels: List[str],
    ) -> PhishingDetector:
        """Run Attack -> Beast -> Warhammer training and return the trained detector."""
        self._fit_attack(train_texts, train_labels, val_texts, val_labels)
        self._fit_beast(val_texts, val_labels)
        if self.detector.warhammer is not None:
            self._train_warhammer(train_texts, train_labels, val_texts, val_labels)
        self.last_val_metrics = self._evaluate_pipeline(val_texts, val_labels)
        logger.info("Phishing training complete.")
        return self.detector
