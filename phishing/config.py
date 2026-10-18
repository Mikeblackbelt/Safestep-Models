"""
Configuration for the Phishing escalation pipeline (Attack -> Beast -> Warhammer).
"""

from dataclasses import dataclass


@dataclass
class PhishingConfig:
    """Configuration for the three-stage phishing detector."""

    # Attack (stage 1): character n-gram TF-IDF + logistic regression.
    # Cheap enough to run on every email/URL.
    attack_ngram_min: int = 2
    attack_ngram_max: int = 5
    attack_max_features: int = 50_000
    attack_C: float = 4.0

    # Beast (stage 2): XGBoost over hand-engineered phishing indicators
    # (URLs, urgency/credential language, HTML forms) + Attack's score.
    beast_n_estimators: int = 200
    beast_max_depth: int = 5
    beast_learning_rate: float = 0.1

    # Warhammer (stage 3): fine-tuned DistilBERT, only for samples the
    # first two stages are unsure about.
    use_warhammer: bool = True
    backbone_name: str = "distilbert-base-uncased"
    max_seq_length: int = 256
    hidden_dropout: float = 0.2
    classifier_hidden_dim: int = 256

    # Escalation: a stage's verdict is final unless p(phishing) falls
    # strictly inside its (low, high) uncertainty band.
    attack_escalate_low: float = 0.1
    attack_escalate_high: float = 0.9
    beast_escalate_low: float = 0.2
    beast_escalate_high: float = 0.8

    # Decision
    decision_threshold: float = 0.5

    # Warhammer training
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_steps: int = 200
    num_epochs_frozen: int = 1   # classifier head only
    num_epochs_finetune: int = 3  # full backbone unfrozen
    weight_decay: float = 0.01
    early_stopping_patience: int = 2
    gradient_clip_norm: float = 1.0

    # Labels
    label2id: dict = None
    id2label: dict = None

    def __post_init__(self):
        if self.label2id is None:
            self.label2id = {"legitimate": 0, "phishing": 1}
        if self.id2label is None:
            self.id2label = {v: k for k, v in self.label2id.items()}
