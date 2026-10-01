"""
Training pipeline for JAW.

Stage 1: freeze the EfficientNet-B3 backbone, train the frequency
         branch + classification head only.
Stage 2: unfreeze and fine-tune the full model end-to-end at a
         lower learning rate with cosine decay.

Usage:
    from jaw.config import JAWConfig
    from jaw.train import JAWTrainer

    trainer = JAWTrainer(JAWConfig())
    trainer.fit(train_paths, train_labels, val_paths, val_labels)
    trainer.detector.save("checkpoints/jaw")
"""

import math
from typing import Callable, List, Optional, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from jaw.config import JAWConfig
from jaw.detector import JAWDetector
from util.logging_config import get_logger

logger = get_logger(__name__)

# Receives progress events: {"type": "stage"|"batch"|"epoch"|"early_stop", "stage", "epoch", ...}
ProgressCallback = Callable[[dict], None]


class JAWTrainer:
    """Orchestrates 2-stage JAW training."""

    def __init__(
        self,
        config: JAWConfig,
        device: str = "cpu",
        callback: Optional[ProgressCallback] = None,
    ):
        self.config = config
        self.device = device
        self.detector = JAWDetector(config, device=device)
        self.callback = callback
        self._stage: Optional[str] = None
        self._epoch = 0

    def _emit(self, event_type: str, **payload) -> None:
        if self.callback is not None:
            self.callback({"type": event_type, "stage": self._stage, "epoch": self._epoch, **payload})

    def _labels_to_ids(self, labels: List[str]) -> torch.Tensor:
        return torch.tensor([self.config.label2id[l] for l in labels], dtype=torch.long)

    def _iter_batches(
        self,
        paths: List[Union[str, Path]],
        labels: List[str],
        batch_size: int,
        shuffle: bool,
    ):
        n = len(paths)
        indices = np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            yield [paths[i] for i in batch_idx], [labels[i] for i in batch_idx]

    def _run_epoch(
        self,
        paths: List[Union[str, Path]],
        labels: List[str],
        optimizer,
        scheduler,
        criterion,
    ) -> float:
        model = self.detector.model
        model.train()
        total_loss = 0.0
        num_batches = 0
        total_batches = math.ceil(len(paths) / self.config.train_batch_size)

        for batch_paths, batch_labels in self._iter_batches(
            paths, labels, self.config.train_batch_size, shuffle=True
        ):
            image_tensor, raw_arrays = self.detector.preprocessor.load_and_prepare(batch_paths)
            freq_feats = self.detector.freq_extractor.extract_batch(raw_arrays)

            image_tensor = image_tensor.to(self.device)
            freq_tensor = torch.from_numpy(freq_feats).to(self.device)
            label_ids = self._labels_to_ids(batch_labels).to(self.device)

            optimizer.zero_grad()
            outputs = model(images=image_tensor, freq_features=freq_tensor)
            loss = criterion(outputs["logits"], label_ids)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self._emit(
                "batch", phase="train", batch=num_batches, n_batches=total_batches,
                samples=len(batch_paths), loss=loss.item(),
            )

        if scheduler is not None:
            scheduler.step()

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def _evaluate(self, paths: List[Union[str, Path]], labels: List[str]) -> dict:
        model = self.detector.model
        model.eval()
        label_ids = self._labels_to_ids(labels).numpy()

        all_preds = []
        for batch_paths, _ in self._iter_batches(
            paths, labels, self.config.eval_batch_size, shuffle=False
        ):
            image_tensor, raw_arrays = self.detector.preprocessor.load_and_prepare(batch_paths)
            freq_feats = self.detector.freq_extractor.extract_batch(raw_arrays)

            image_tensor = image_tensor.to(self.device)
            freq_tensor = torch.from_numpy(freq_feats).to(self.device)

            outputs = model(images=image_tensor, freq_features=freq_tensor)
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            self._emit("batch", phase="eval", samples=len(batch_paths))

        all_preds = np.array(all_preds)
        accuracy = float((all_preds == label_ids).mean())
        return {"accuracy": accuracy}

    def fit(
        self,
        train_paths: List[Union[str, Path]],
        train_labels: List[str],
        val_paths: List[Union[str, Path]],
        val_labels: List[str],
    ) -> JAWDetector:
        """Run the full 2-stage training pipeline and return the trained detector."""
        model = self.detector.model
        criterion = nn.CrossEntropyLoss()
        best_val_acc = -1.0
        patience_counter = 0

        # Stage 1: frozen backbone
        logger.info("JAW Stage 1: training frequency branch + head (backbone frozen)")
        self._stage = "frozen"
        self._emit("stage", epochs=self.config.num_epochs_frozen)
        model.freeze_backbone()
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.config.lr_frozen,
            weight_decay=self.config.weight_decay,
        )

        for epoch in range(self.config.num_epochs_frozen):
            self._epoch = epoch + 1
            loss = self._run_epoch(train_paths, train_labels, optimizer, None, criterion)
            metrics = self._evaluate(val_paths, val_labels)
            logger.info(
                f"  [frozen] epoch {epoch + 1}/{self.config.num_epochs_frozen} "
                f"loss={loss:.4f} val_acc={metrics['accuracy']:.4f}"
            )
            self._emit("epoch", epochs=self.config.num_epochs_frozen, loss=loss, val_acc=metrics["accuracy"])

        # Stage 2: unfreeze, fine-tune end-to-end
        logger.info("JAW Stage 2: fine-tuning full model")
        self._stage = "finetune"
        self._epoch = 0
        self._emit("stage", epochs=self.config.num_epochs_finetune)
        model.unfreeze_backbone()
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.lr_finetune,
            weight_decay=self.config.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.num_epochs_finetune)

        for epoch in range(self.config.num_epochs_finetune):
            self._epoch = epoch + 1
            loss = self._run_epoch(train_paths, train_labels, optimizer, scheduler, criterion)
            metrics = self._evaluate(val_paths, val_labels)
            logger.info(
                f"  [finetune] epoch {epoch + 1}/{self.config.num_epochs_finetune} "
                f"loss={loss:.4f} val_acc={metrics['accuracy']:.4f}"
            )
            self._emit("epoch", epochs=self.config.num_epochs_finetune, loss=loss, val_acc=metrics["accuracy"])

            if metrics["accuracy"] > best_val_acc:
                best_val_acc = metrics["accuracy"]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"  Early stopping at epoch {epoch + 1} (patience exceeded)")
                    self._emit("early_stop")
                    break

        logger.info("JAW training complete.")
        return self.detector