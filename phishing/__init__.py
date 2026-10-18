"""
Phishing: Attack -> Beast -> Warhammer escalation pipeline
===========================================================

Three detectors of increasing cost for phishing emails and URLs:

- Attack:    character n-gram TF-IDF + logistic regression; scores everything.
- Beast:     XGBoost over Attack's score + structural phishing indicators
             (URLs, urgency/credential language, HTML forms).
- Warhammer: fine-tuned DistilBERT for the samples both earlier stages
             are unsure about.

PhishingDetector routes each sample through the stages and records which
one made the final call.
"""

from phishing.config import PhishingConfig
from phishing.indicators import IndicatorExtractor
from phishing.attack import AttackModel
from phishing.beast import BeastModel
from phishing.warhammer import WarhammerClassifier
from phishing.detector import PhishingDetector

__all__ = [
    "PhishingConfig",
    "IndicatorExtractor",
    "AttackModel",
    "BeastModel",
    "WarhammerClassifier",
    "PhishingDetector",
]
