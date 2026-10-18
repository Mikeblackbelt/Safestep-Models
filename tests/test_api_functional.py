"""
Unit and integration tests for SafeStep Python Functional API.
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

import api
from api.schemas import (
    ImageDetectionResult,
    ModelStatus,
    TextDetectionResult,
    UnifiedDetectionResponse,
)
from tests.generate_test_assets import generate_test_assets


@pytest.fixture(scope="module")
def sample_assets(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("samples")
    auth_path, syn_path = generate_test_assets(tmp_dir)
    return {"auth": auth_path, "synthetic": syn_path}


def test_model_status():
    """Test get_model_status returns cart and jaw statuses."""
    statuses = api.get_model_status()
    assert len(statuses) >= 2
    model_ids = {s.model_id for s in statuses}
    assert "cart" in model_ids
    assert "jaw" in model_ids
    assert "phishing" in model_ids


def test_predict_text_single():
    """Test predict_text with a single string."""
    text = "The quick brown fox jumps over the lazy dog."
    result = api.predict_text(text)
    assert isinstance(result, TextDetectionResult)
    assert result.text == text
    assert 0.0 <= result.p_ai <= 1.0
    assert 0.0 <= result.p_human <= 1.0
    assert result.label in ("human", "ai_generated")
    assert result.decision_threshold == 0.5


def test_predict_text_batch():
    """Test predict_text with a list of strings."""
    texts = [
        "First human written paragraph with rich vocabulary.",
        "As an artificial intelligence model, I am trained to output text.",
    ]
    results = api.predict_text(texts)
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, TextDetectionResult)
        assert 0.0 <= r.p_ai <= 1.0


def test_predict_text_custom_threshold():
    """Test predict_text with a custom decision threshold."""
    text = "Machine learning models analyze patterns in input data."
    result = api.predict_text(text, threshold=0.8)
    assert result.decision_threshold == 0.8
    expected_label = "ai_generated" if result.p_ai >= 0.8 else "human"
    assert result.label == expected_label


def test_predict_text_include_features():
    """Test predict_text with heuristic features extraction."""
    text = "However, the results furthermore indicate high repetition repetition repetition."
    result = api.predict_text(text, include_features=True)
    assert result.features is not None
    assert "repetition_ratio" in result.features
    assert "burstiness" in result.features


def test_predict_image_path(sample_assets):
    """Test predict_image using a file path."""
    img_path = sample_assets["auth"]
    result = api.predict_image(img_path)
    assert isinstance(result, ImageDetectionResult)
    assert 0.0 <= result.p_ai <= 1.0
    assert 0.0 <= result.p_authentic <= 1.0
    assert result.label in ("authentic", "ai_generated")


def test_predict_image_pil(sample_assets):
    """Test predict_image using a PIL Image object."""
    img = Image.open(sample_assets["synthetic"])
    result = api.predict_image(img)
    assert isinstance(result, ImageDetectionResult)
    assert 0.0 <= result.p_ai <= 1.0


def test_predict_image_bytes(sample_assets):
    """Test predict_image using raw bytes."""
    with open(sample_assets["auth"], "rb") as f:
        img_bytes = f.read()
    result = api.predict_image(img_bytes)
    assert isinstance(result, ImageDetectionResult)
    assert 0.0 <= result.p_ai <= 1.0


def test_predict_multimodal(sample_assets):
    """Test predict_multimodal with both text and image."""
    result = api.predict_multimodal(
        text="Suspicious security notification with attached image.",
        image=sample_assets["synthetic"],
    )
    assert isinstance(result, UnifiedDetectionResponse)
    assert result.text_result is not None
    assert result.image_result is not None
    assert result.overall_risk in ("HIGH", "MEDIUM", "LOW")
    assert 0.0 <= result.risk_score <= 1.0
    assert len(result.summary) > 0


def test_reload_model():
    """Test reloading model from default/specified path."""
    reloaded = api.reload_model("cart")
    assert isinstance(reloaded, bool)


def test_predict_phishing_single_and_batch():
    from api.schemas import PhishingDetectionResult
    res = api.predict_phishing("Urgent: confirm your login at http://paypa1.example", threshold=0.7)
    assert isinstance(res, PhishingDetectionResult)
    assert res.decision_threshold == 0.7
    assert 0.0 <= res.p_phishing <= 1.0
    assert abs(res.p_phishing + res.p_legitimate - 1.0) < 1e-3
    if res.is_mock:
        assert res.stage == "none" and res.p_phishing == 0.5
    else:
        assert res.label == ("phishing" if res.p_phishing >= 0.7 else "legitimate")
    batch = api.predict_phishing(["hi team", "verify now"], max_stage="attack")
    assert len(batch) == 2
    assert all(r.stage in ("attack", "none") for r in batch)


def test_phishing_real_model_roundtrip(tmp_path, monkeypatch):
    """Fit a tiny Attack/Beast checkpoint, then serve it through the registry."""
    from api.config import APIConfig
    from api.model_registry import SafeStepModelRegistry
    from api.service import SafeStepService
    from phishing.config import PhishingConfig
    from phishing.detector import PhishingDetector

    texts = [f"Meeting notes for project {i}, see you tomorrow." for i in range(20)] + [
        f"URGENT verify your account password at http://login{i}.example.ru/secure now" for i in range(20)
    ]
    labels = np.array([0] * 20 + [1] * 20)
    det = PhishingDetector(PhishingConfig(use_warhammer=False))
    det.attack.fit(texts, labels)
    feats = det.beast.build_features(det.attack.predict_proba(texts), det.indicators.extract_batch(texts))
    det.beast.fit(feats, labels)
    det.save(tmp_path)

    cfg = APIConfig(phishing_checkpoint_dir=tmp_path, enable_mock_fallback=False)
    monkeypatch.delenv("SAFESTEP_PHISHING_CHECKPOINT", raising=False)
    cfg.phishing_checkpoint_dir = tmp_path
    reg = SafeStepModelRegistry(cfg)
    svc = SafeStepService(reg)
    out = svc.predict_phishing(texts[:1] + texts[-1:])
    assert all(not r.is_mock for r in out)
    assert reg._phishing_detector.warhammer is None
    assert out[0].label == "legitimate" and out[1].label == "phishing"
