"""
Configuration for CART (AI-Generated Text Detector).
"""

from dataclasses import dataclass


@dataclass
class CARTConfig:
    """Configuration for the CART hybrid text detector."""

    # Neural backbone
    backbone_name: str = "distilbert-base-uncased"
    max_seq_length: int = 512
    hidden_dropout: float = 0.2
    classifier_hidden_dim: int = 256

    # Heuristic layer
    ngram_min: int = 2
    ngram_max: int = 5
    repetition_threshold: int = 2  # flag n-grams repeated more than this

    # Fusion (meta-classifier)
    fusion_n_estimators: int = 100
    fusion_max_depth: int = 4
    fusion_learning_rate: float = 0.1

    # Decision
    decision_threshold: float = 0.5

    # Training
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    num_epochs_frozen: int = 2   # classifier head only
    num_epochs_finetune: int = 10  # full backbone unfrozen
    weight_decay: float = 0.01
    early_stopping_patience: int = 3
    gradient_clip_norm: float = 1.0

    # Labels
    label2id: dict = None
    id2label: dict = None

    def __post_init__(self):
        if self.label2id is None:
            self.label2id = {"human": 0, "ai_generated": 1}
        if self.id2label is None:
            self.id2label = {v: k for k, v in self.label2id.items()}