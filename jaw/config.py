"""
Configuration for JAW (AI-Generated Image Classifier).
"""

from dataclasses import dataclass


@dataclass
class JAWConfig:
    """Configuration for the JAW spatial + frequency-domain image detector."""

    # Backbone
    backbone_name: str = "efficientnet_b3"
    pretrained: bool = True
    image_size: int = 224
    spatial_feature_dim: int = 1536  # EfficientNet-B3 output dim

    # Frequency-domain branch
    num_radial_bands: int = 8       # spectral energy histogram bins
    num_gabor_orientations: int = 4
    num_gabor_scales: int = 4
    # frequency_feature_dim is derived: radial bands + peak/flatness + gabor grid
    # (see FrequencyFeatureExtractor.FEATURE_DIM)

    # Classification head
    fusion_hidden_dim_1: int = 512
    fusion_hidden_dim_2: int = 128
    dropout: float = 0.2

    # Decision
    decision_threshold: float = 0.5

    # Training
    train_batch_size: int = 16
    eval_batch_size: int = 32
    lr_frozen: float = 1e-3
    lr_finetune: float = 1e-5
    weight_decay: float = 0.01
    num_epochs_frozen: int = 3
    num_epochs_finetune: int = 15
    early_stopping_patience: int = 3
    gradient_clip_norm: float = 1.0

    # ImageNet normalization (matches pretrained EfficientNet)
    norm_mean: tuple = (0.485, 0.456, 0.406)
    norm_std: tuple = (0.229, 0.224, 0.225)

    # Labels
    label2id: dict = None
    id2label: dict = None

    def __post_init__(self):
        if self.label2id is None:
            self.label2id = {"authentic": 0, "ai_generated": 1}
        if self.id2label is None:
            self.id2label = {v: k for k, v in self.label2id.items()}