"""
Neural backbone for CART: a DistilBERT encoder with a small
classification head, producing p(ai_generated) plus the pooled
[CLS] embedding (useful as a fusion feature).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from cart.config import CARTConfig


class CARTNeuralClassifier(nn.Module):
    """DistilBERT encoder + dense classification head."""

    def __init__(self, config: CARTConfig):
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.backbone_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, config.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.classifier_hidden_dim, 2),
        )

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT has no pooler; use the [CLS] token (position 0) of the last hidden state.
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return {"logits": logits, "cls_embedding": cls_embedding}


class CARTTokenizerWrapper:
    """Thin wrapper so callers don't need to know the tokenizer's exact API."""

    def __init__(self, config: CARTConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)

    def encode(self, texts, return_tensors: str = "pt"):
        return self.tokenizer(
            texts,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
        )