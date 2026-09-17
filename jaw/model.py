"""
Model architecture for JAW: EfficientNet-B3 spatial backbone +
frequency-domain feature branch, fused via a dense classification head.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from jaw.config import JAWConfig
from jaw.frequency_features import FrequencyFeatureExtractor


class JAWModel(nn.Module):
    """EfficientNet-B3 spatial features + frequency features -> binary classifier."""

    def __init__(self, config: JAWConfig, freq_feature_dim: int):
        super().__init__()
        self.config = config
        self.freq_feature_dim = freq_feature_dim

        self.backbone = self._build_backbone()
        fused_dim = config.spatial_feature_dim + freq_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, config.fusion_hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim_1, config.fusion_hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dim_2, 2),
        )

    def _build_backbone(self) -> nn.Module:
        weights = tv_models.EfficientNet_B3_Weights.DEFAULT if self.config.pretrained else None
        net = tv_models.efficientnet_b3(weights=weights)
        # Drop the original classifier; keep the feature extractor + pooling.
        net.classifier = nn.Identity()
        return net

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor, freq_features: torch.Tensor) -> dict:
        """
        Args:
            images: (B, 3, H, W) normalized image tensor.
            freq_features: (B, freq_feature_dim) precomputed frequency features.

        Returns:
            {"logits": (B, 2), "spatial_features": (B, spatial_feature_dim)}
        """
        spatial_features = self.backbone(images)  # (B, spatial_feature_dim)
        fused = torch.cat([spatial_features, freq_features], dim=1)
        logits = self.classifier(fused)
        return {"logits": logits, "spatial_features": spatial_features}