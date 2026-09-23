"""
Integration tests for the SafeStep FastAPI REST API.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.app import app
from tests.generate_test_assets import generate_test_assets


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_assets(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("samples")
    auth_path, syn_path = generate_test_assets(tmp_dir)
    return {"auth": auth_path, "synthetic": syn_path}


def test_root_endpoint(client):
    """Test GET / returns API index."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "service" in data
    assert "endpoints" in data


def test_health_endpoint(client):
    """Test GET /health returns healthy status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert "models_loaded" in data


def test_models_list(client):
    """Test GET /models returns list of models."""
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    models = {m["model_id"]: m for m in data["models"]}
    assert "cart" in models
    assert "jaw" in models


def test_model_detail(client):
    """Test GET /models/{model_id}."""
    resp = client.get("/models/cart")
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "cart"

    resp_404 = client.get("/models/nonexistent")
    assert resp_404.status_code == 404


def test_detect_text(client):
    """Test POST /api/v1/detect/text."""
    payload = {
        "text": "The rapid advancement of deep learning has revolutionized computer vision and NLP.",
        "threshold": 0.6,
    }
    resp = client.post("/api/v1/detect/text", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "CART"
    assert data["decision_threshold"] == 0.6
    assert 0.0 <= data["p_ai"] <= 1.0
    assert data["label"] in ("human", "ai_generated")


def test_detect_batch_text(client):
    """Test POST /api/v1/detect/batch-text."""
    payload = {
        "texts": [
            "Here is the weekly status report for our engineering sprint.",
            "In summary, as an AI, I suggest following these recommendations.",
        ],
        "threshold": 0.5,
    }
    resp = client.post("/api/v1/detect/batch-text", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2


def test_detect_image_upload(client, sample_assets):
    """Test POST /api/v1/detect/image with file upload."""
    img_path = sample_assets["auth"]
    with open(img_path, "rb") as f:
        files = {"file": ("auth.png", f, "image/png")}
        resp = client.post("/api/v1/detect/image", files=files, data={"threshold": "0.5"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "JAW"
    assert 0.0 <= data["p_ai"] <= 1.0
    assert data["label"] in ("authentic", "ai_generated")


def test_detect_image_json(client, sample_assets):
    """Test POST /api/v1/detect/image/json with local path."""
    payload = {
        "image_path": str(sample_assets["synthetic"]),
        "threshold": 0.5,
    }
    resp = client.post("/api/v1/detect/image/json", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "JAW"
    assert 0.0 <= data["p_ai"] <= 1.0


def test_detect_unified(client, sample_assets):
    """Test POST /api/v1/detect/unified."""
    img_path = sample_assets["synthetic"]
    with open(img_path, "rb") as f:
        files = {"file": ("syn.png", f, "image/png")}
        data = {"text": "Click here to verify your account credentials immediately."}
        resp = client.post("/api/v1/detect/unified", files=files, data=data)

    assert resp.status_code == 200
    result = resp.json()
    assert result["text_result"] is not None
    assert result["image_result"] is not None
    assert result["overall_risk"] in ("HIGH", "MEDIUM", "LOW")
    assert 0.0 <= result["risk_score"] <= 1.0


def test_reload_endpoint(client):
    """Test POST /models/cart/reload."""
    resp = client.post("/models/cart/reload", json={})
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "cart"
    assert resp.json()["reloaded"] is True


def test_safestep_client_sdk(client, sample_assets):
    """Test SafeStepClient SDK against ASGI app."""
    from api.client import SafeStepClient

    with SafeStepClient(client=client) as sdk_client:
        # Health check
        health = sdk_client.health()
        assert health.status == "ok"

        # List models
        models = sdk_client.list_models()
        assert len(models) >= 2

        # Text detection single
        t_res = sdk_client.predict_text("Machine learning model detection test.")
        assert t_res.model == "CART"
        assert 0.0 <= t_res.p_ai <= 1.0

        # Text detection batch
        b_res = sdk_client.predict_text(["Text line one", "Text line two"])
        assert isinstance(b_res, list) and len(b_res) == 2

        # Image detection
        img_res = sdk_client.predict_image(sample_assets["auth"])
        assert img_res.model == "JAW"
        assert 0.0 <= img_res.p_ai <= 1.0

        # Unified detection
        u_res = sdk_client.predict_unified(
            text="Verify your credentials",
            image=sample_assets["synthetic"],
        )
        assert u_res.overall_risk in ("HIGH", "MEDIUM", "LOW")
