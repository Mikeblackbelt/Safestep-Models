"""
Training pipeline for CART.

Stage 1: freeze DistilBERT, train only the classification head.
Stage 2: unfreeze and fine-tune the full backbone at a lower LR.
Stage 3: fit the XGBoost fusion model on neural probs + heuristics,
         computed on the validation split (held out from backbone training).

Usage:
    from cart.config import CARTConfig
    from cart.train import CARTTrainer

    trainer = CARTTrainer(CARTConfig())
    trainer.fit(train_texts, train_labels, val_texts, val_labels)
    trainer.detector.save("checkpoints/cart")
"""

import math
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from cart.config import CARTConfig
from cart.detector import CARTDetector
from util.logging_config import get_logger

logger = get_logger(__name__)

# Receives progress events: {"type": "stage"|"batch"|"epoch"|"early_stop"|"fusion", "stage", "epoch", ...}
ProgressCallback = Callable[[dict], None]


def _linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return LambdaLR(optimizer, lr_lambda)


class CARTTrainer:
    """Orchestrates multi-stage CART training."""

    def __init__(
        self,
        config: CARTConfig,
        device: str = "cpu",
        callback: Optional[ProgressCallback] = None,
    ):
        self.config = config
        self.device = device
        self.detector = CARTDetector(config, device=device)
        self.callback = callback
        self._stage: Optional[str] = None
        self._epoch = 0

    def _emit(self, event_type: str, **payload) -> None:
        if self.callback is not None:
            self.callback({"type": event_type, "stage": self._stage, "epoch": self._epoch, **payload})

    def _labels_to_ids(self, labels: List[str]) -> torch.Tensor:
        return torch.tensor([self.config.label2id[l] for l in labels], dtype=torch.long)

    def _iter_batches(self, texts: List[str], labels: List[str], batch_size: int, shuffle: bool):
        n = len(texts)
        indices = np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield [texts[i] for i in batch_idx], [labels[i] for i in batch_idx]

    def _run_neural_epoch(
        self,
        texts: List[str],
        labels: List[str],
        optimizer,
        scheduler,
        criterion,
    ) -> float:
        model = self.detector.neural_model
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
            label_ids = self._labels_to_ids(batch_labels).to(self.device)

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

    @torch.no_grad()
    def _evaluate_neural(self, texts: List[str], labels: List[str]) -> dict:
        from training.evaluation import compute_evaluation_metrics

        model = self.detector.neural_model
        model.eval()
        label_ids = self._labels_to_ids(labels).numpy()

        all_probs = []
        for batch_texts, _ in self._iter_batches(
            texts, labels, self.config.eval_batch_size, shuffle=False
        ):
            encoded = self.detector.tokenizer.encode(batch_texts)
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs["logits"], dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            self._emit("batch", phase="eval", samples=len(batch_texts))

        all_probs = np.array(all_probs)
        return compute_evaluation_metrics(y_true=label_ids, y_pred=all_probs, pos_label=1)

    def _train_neural_backbone(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: List[str],
        val_labels: List[str],
    ) -> None:
        model = self.detector.neural_model
        criterion = nn.CrossEntropyLoss()

        # Stage 1: frozen backbone, train classifier head only
        logger.info("CART Stage 1: training classifier head (backbone frozen)")
        self._stage = "frozen"
        self._emit("stage", epochs=self.config.num_epochs_frozen)
        model.freeze_encoder()
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3,
            weight_decay=self.config.weight_decay,
        )
        best_val_acc = -1.0
        patience_counter = 0

        for epoch in range(self.config.num_epochs_frozen):
            self._epoch = epoch + 1
            loss = self._run_neural_epoch(train_texts, train_labels, optimizer, None, criterion)
            metrics = self._evaluate_neural(val_texts, val_labels)
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

        # Stage 2: unfreeze, fine-tune end-to-end at lower LR with warmup
        logger.info("CART Stage 2: fine-tuning full backbone")
        self._stage = "finetune"
        self._epoch = 0
        self._emit("stage", epochs=self.config.num_epochs_finetune)
        model.unfreeze_encoder()
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        steps_per_epoch = max(1, len(train_texts) // self.config.train_batch_size)
        total_steps = steps_per_epoch * self.config.num_epochs_finetune
        scheduler = _linear_warmup_scheduler(optimizer, self.config.warmup_steps, total_steps)

        for epoch in range(self.config.num_epochs_finetune):
            self._epoch = epoch + 1
            loss = self._run_neural_epoch(train_texts, train_labels, optimizer, scheduler, criterion)
            metrics = self._evaluate_neural(val_texts, val_labels)
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

    def _fit_fusion(self, val_texts: List[str], val_labels: List[str]) -> None:
        """
        Fit the XGBoost fusion model on the validation split — held out
        from backbone training to avoid the meta-classifier overfitting
        to the neural model's own training distribution.
        """
        logger.info("CART Stage 3: fitting fusion (heuristic + neural) meta-classifier")
        self._stage = "fusion"
        self._epoch = 0
        self._emit("stage", epochs=0)
        neural_probs = self.detector._neural_probs(val_texts)
        heuristic_feats = self.detector.heuristics.extract_batch(val_texts)
        fused = self.detector.fusion.build_features(neural_probs, heuristic_feats)
        label_ids = np.array([self.config.label2id[l] for l in val_labels])

        self.detector.fusion.fit(fused, label_ids)

        from training.evaluation import compute_evaluation_metrics

        fusion_probs = self.detector.fusion.predict_proba(fused)[:, 1]
        metrics = compute_evaluation_metrics(y_true=label_ids, y_pred=fusion_probs, pos_label=1)
        accuracy = metrics["accuracy"]
        auroc = metrics.get("auroc", float("nan"))
        auroc_str = f" val_auroc: {auroc:.4f}" if auroc == auroc else ""
        logger.info(f"  Fusion model val accuracy: {accuracy:.4f}{auroc_str}")
        self._emit("fusion", val_acc=accuracy, val_auroc=auroc, metrics=metrics)
        logger.info(f"  Top fusion features: {list(self.detector.fusion.feature_importance().items())[:5]}")
        self.last_val_metrics = metrics

    def evaluate(self, texts: List[str], labels: List[str]) -> dict:
        """Evaluate CART model on texts and labels using full pipeline or neural backbone."""
        if self.detector.fusion.model is not None:
            return self.detector.evaluate(texts, labels)
        return self._evaluate_neural(texts, labels)

    def _evaluate(self, texts: List[str], labels: List[str]) -> dict:
        return self.evaluate(texts, labels)

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[str],
        val_texts: List[str],
        val_labels: List[str],
    ) -> CARTDetector:
        """Run the full 3-stage training pipeline and return the trained detector."""
        self._train_neural_backbone(train_texts, train_labels, val_texts, val_labels)
        self._fit_fusion(val_texts, val_labels)
        logger.info("CART training complete.")
        return self.detector