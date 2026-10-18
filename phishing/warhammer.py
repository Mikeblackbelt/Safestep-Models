"""
Warhammer: stage 3 of the phishing pipeline.

A DistilBERT encoder with a small classification head. It reads the
message the way a person would — pretexts, impersonation, social
engineering that has no suspicious URL or keyword — and is the most
expensive stage, so it only runs on what Attack and Beast escalate.
"""

from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from phishing.config import PhishingConfig


class WarhammerClassifier(nn.Module):
    """DistilBERT encoder + dense classification head."""

    def __init__(self, config: PhishingConfig):
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


class WarhammerTokenizerWrapper:
    """Thin wrapper so callers don't need to know the tokenizer's exact API."""

    def __init__(self, config: PhishingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)

    def encode(self, texts, return_tensors: str = "pt"):
        return self.tokenizer(
            texts,
            max_length=self.config.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors=return_tensors,
        )
