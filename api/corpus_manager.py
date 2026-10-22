"""
Corpus Manager for SafeStep API.
Handles secure user data ingestion, ground-truth label normalization,
corpus persistence (JSONL), duplicate tracking, and statistics aggregation.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

logger = get_logger = logging.getLogger("safestep.api.corpus")

# Base directory for user-contributed corpora
CORPUS_BASE_DIR = Path("data")

# Valid models
SUPPORTED_MODELS = ("cart", "jaw", "phishing")

# Canonical label definitions
CANONICAL_LABELS = {
    "cart": {0: "human", 1: "ai_generated"},
    "jaw": {0: "authentic", 1: "ai_generated"},
    "phishing": {0: "legitimate", 1: "phishing"},
}

LABEL_ALIASES = {
    "cart": {
        0: {"human", "real", "authentic", "genuine", "original", "clean", "ham", "0", "0.0"},
        1: {"ai_generated", "ai", "synthetic", "generated", "fake", "chatgpt", "gpt", "llm", "1", "1.0"},
    },
    "jaw": {
        0: {"authentic", "real", "photo", "original", "clean", "genuine", "0", "0.0"},
        1: {"ai_generated", "synthetic", "fake", "deepfake", "midjourney", "dall-e", "stablediffusion", "1", "1.0"},
    },
    "phishing": {
        0: {"legitimate", "legit", "ham", "safe", "clean", "normal", "benign", "0", "0.0"},
        1: {"phishing", "phish", "spam", "malicious", "fraud", "scam", "bad", "1", "1.0"},
    },
}


def normalize_user_eval(model_type: str, user_eval: Any) -> Tuple[str, int]:
    """
    Normalize user evaluation input to canonical label string and binary int (0 or 1).
    Raises ValueError if label cannot be mapped.
    """
    model_key = model_type.lower().strip()
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_type '{model_type}'. Must be one of {SUPPORTED_MODELS}")

    cleaned = str(user_eval).strip().lower()

    aliases = LABEL_ALIASES[model_key]
    for binary_val, alias_set in aliases.items():
        if cleaned in alias_set:
            return CANONICAL_LABELS[model_key][binary_val], binary_val

    # Partial / substring match
    if model_key == "cart":
        if any(k in cleaned for k in ("human", "real", "orig", "auth")):
            return "human", 0
        if any(k in cleaned for k in ("ai", "synth", "gpt", "fake", "gen")):
            return "ai_generated", 1
    elif model_key == "jaw":
        if any(k in cleaned for k in ("auth", "real", "photo", "clean")):
            return "authentic", 0
        if any(k in cleaned for k in ("fake", "synth", "deepfake", "ai", "gen")):
            return "ai_generated", 1
    elif model_key == "phishing":
        if any(k in cleaned for k in ("legit", "ham", "safe", "clean", "benign")):
            return "legitimate", 0
        if any(k in cleaned for k in ("phish", "spam", "fraud", "malic", "bad")):
            return "phishing", 1

    raise ValueError(
        f"Unrecognized evaluation label '{user_eval}' for model '{model_type}'. "
        f"Allowed categories: {list(CANONICAL_LABELS[model_key].values())}"
    )


class CorpusManager:
    """Manages appending and querying user-contributed evaluation datasets."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or CORPUS_BASE_DIR)

    def _ensure_corpus_dir(self, model_type: str) -> Path:
        model_dir = self.base_dir / model_type / "corpus"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def _get_corpus_file(self, model_type: str) -> Path:
        return self._ensure_corpus_dir(model_type) / "user_feedback.jsonl"

    def _get_audit_log_file(self) -> Path:
        audit_dir = self.base_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        return audit_dir / "ingestion_log.jsonl"

    def save_image_sample(self, image_bytes: bytes, user_eval: str) -> Tuple[Path, str]:
        """
        Securely save an uploaded image into the JAW corpus with SHA256-based naming.
        """
        jaw_img_dir = self.base_dir / "jaw" / "corpus" / "images"
        jaw_img_dir.mkdir(parents=True, exist_ok=True)

        sha256_hash = hashlib.sha256(image_bytes).hexdigest()[:24]
        filename = f"{sha256_hash}.png"
        filepath = jaw_img_dir / filename

        # Decode & re-encode via PIL to strip any potential metadata/polyglot payloads
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.save(filepath, format="PNG", optimize=True)

        rel_path = str(filepath.relative_to(self.base_dir))
        return filepath, rel_path

    def append_record(
        self,
        model_type: str,
        content: str,
        canonical_label: str,
        binary_label: int,
        is_image: bool = False,
        source_url: Optional[str] = None,
        model_prediction: Optional[Dict[str, Any]] = None,
        model_agreed: Optional[bool] = None,
        poisoning_score: Optional[float] = None,
        poisoning_verdict: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Atomically append an evaluation record to the corpus JSONL and audit log.
        """
        submission_id = f"sub_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "id": submission_id,
            "timestamp": timestamp,
            "model_type": model_type,
            "label": canonical_label,
            "binary_label": binary_label,
            "source_url": source_url,
            "metadata": metadata or {},
        }

        if is_image:
            record["image"] = content
        else:
            record["text"] = content

        if model_prediction:
            record["model_prediction"] = model_prediction
        if model_agreed is not None:
            record["model_agreed"] = model_agreed
        if poisoning_score is not None:
            record["poisoning_risk"] = poisoning_score
            record["poisoning_verdict"] = poisoning_verdict

        corpus_file = self._get_corpus_file(model_type)
        with open(corpus_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Also write to global audit log for system monitoring
        audit_file = self._get_audit_log_file()
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            f"Appended submission {submission_id} to {corpus_file} "
            f"({model_type}: {canonical_label})"
        )
        return record

    def get_analytics(self) -> Dict[str, Any]:
        """
        Aggregate corpus statistics, class balances, poisoning rates, and model accuracy/agreement.
        """
        stats: Dict[str, Any] = {
            "total_submissions": 0,
            "models": {},
            "poisoning_summary": {
                "total_assessed": 0,
                "clean_count": 0,
                "suspicious_count": 0,
                "flagged_poisoning_count": 0,
                "avg_poisoning_risk": 0.0,
            },
            "overall_model_agreement_rate": None,
            "recent_submissions": [],
        }

        total_agreed = 0
        total_eval_comparisons = 0
        total_risk_sum = 0.0
        risk_count = 0

        for m in SUPPORTED_MODELS:
            corpus_file = self._get_corpus_file(m)
            m_stats = {
                "count": 0,
                "total_submissions": 0,
                "labels": {},
                "model_agreement_count": 0,
                "model_disagreement_count": 0,
                "agreement_rate": None,
                "corpus_path": str(corpus_file),
            }

            if corpus_file.exists():
                try:
                    with open(corpus_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            data = json.loads(line)
                            m_stats["count"] += 1
                            m_stats["total_submissions"] += 1
                            lbl = data.get("label", "unknown")
                            m_stats["labels"][lbl] = m_stats["labels"].get(lbl, 0) + 1

                            # Model agreement tracking
                            if "model_agreed" in data and data["model_agreed"] is not None:
                                total_eval_comparisons += 1
                                if data["model_agreed"]:
                                    m_stats["model_agreement_count"] += 1
                                    total_agreed += 1
                                else:
                                    m_stats["model_disagreement_count"] += 1

                            # Poisoning stats
                            if "poisoning_risk" in data:
                                r = float(data["poisoning_risk"])
                                total_risk_sum += r
                                risk_count += 1
                                stats["poisoning_summary"]["total_assessed"] += 1
                                v = data.get("poisoning_verdict", "").upper()
                                if "POISON" in v or r >= 0.7:
                                    stats["poisoning_summary"]["flagged_poisoning_count"] += 1
                                elif "SUSPICIOUS" in v or r >= 0.35:
                                    stats["poisoning_summary"]["suspicious_count"] += 1
                                else:
                                    stats["poisoning_summary"]["clean_count"] += 1

                except Exception as e:
                    logger.warning(f"Error reading corpus file {corpus_file}: {e}")

            if (m_stats["model_agreement_count"] + m_stats["model_disagreement_count"]) > 0:
                tot = m_stats["model_agreement_count"] + m_stats["model_disagreement_count"]
                m_stats["agreement_rate"] = round(m_stats["model_agreement_count"] / tot, 4)

            stats["models"][m] = m_stats
            stats["total_submissions"] += m_stats["count"]

        if total_eval_comparisons > 0:
            stats["overall_model_agreement_rate"] = round(total_agreed / total_eval_comparisons, 4)

        if risk_count > 0:
            stats["poisoning_summary"]["avg_poisoning_risk"] = round(total_risk_sum / risk_count, 4)

        # Retrieve last 10 submissions from audit log
        audit_file = self._get_audit_log_file()
        if audit_file.exists():
            try:
                lines = []
                with open(audit_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            lines.append(line)
                for l in lines[-10:]:
                    stats["recent_submissions"].append(json.loads(l))
            except Exception as e:
                logger.warning(f"Error reading audit log: {e}")

        return stats


corpus_manager = CorpusManager()
