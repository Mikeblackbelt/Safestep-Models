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
    assert "phishing" in models


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


def test_detect_image_json_invalid_url_scheme(client):
    """Test POST /api/v1/detect/image/json rejects non-http schemes (e.g. file://)."""
    resp = client.post("/api/v1/detect/image/json", json={"image_url": "file:///etc/passwd"})
    assert resp.status_code == 400
    assert "http or https" in resp.json()["detail"]


def test_detect_image_json_rejects_ssrf_private_ips(client):
    """Test POST /api/v1/detect/image/json blocks SSRF attempts against loopback/private IPs."""
    # Loopback IP
    resp = client.post("/api/v1/detect/image/json", json={"image_url": "http://127.0.0.1/test.png"})
    assert resp.status_code == 400
    assert "prohibited" in resp.json()["detail"].lower() or "forbidden" in resp.json()["detail"].lower()

    # Link-local / Cloud metadata IP
    resp_meta = client.post("/api/v1/detect/image/json", json={"image_url": "http://169.254.169.254/latest/meta-data"})
    assert resp_meta.status_code == 400
    assert "prohibited" in resp_meta.json()["detail"].lower() or "forbidden" in resp_meta.json()["detail"].lower()


def test_detect_image_json_rejects_path_outside_project(client):
    """Test POST /api/v1/detect/image/json rejects paths attempting traversal or probing system directories."""
    resp = client.post("/api/v1/detect/image/json", json={"image_path": "/etc/passwd"})
    assert resp.status_code == 400
    assert "allowed project directory" in resp.json()["detail"]


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


def test_reload_requires_admin_token(client, monkeypatch):
    """With SAFESTEP_ADMIN_TOKEN configured, reload rejects missing/wrong tokens."""
    from api.config import default_api_config
    monkeypatch.setattr(default_api_config, "admin_token", "s3cret")
    assert client.post("/models/cart/reload", json={}).status_code == 401
    assert client.post("/models/cart/reload", json={}, headers={"X-Admin-Token": "nope"}).status_code == 401
    assert client.post("/models/cart/reload", json={}, headers={"X-Admin-Token": "s3cret"}).status_code == 200


def test_reload_forbidden_in_prod_without_token(client, monkeypatch):
    """In production mode (mock fallback disabled), unauthenticated reload is strictly forbidden."""
    from api.config import default_api_config
    monkeypatch.setattr(default_api_config, "admin_token", None)
    monkeypatch.setattr(default_api_config, "enable_mock_fallback", False)
    resp = client.post("/models/cart/reload", json={})
    assert resp.status_code == 403
    assert "SAFESTEP_ADMIN_TOKEN" in resp.json()["detail"]


def test_reload_rejects_path_outside_checkpoints(client):
    resp = client.post("/models/cart/reload", json={"checkpoint_path": "/etc"})
    assert resp.status_code == 400


def test_missing_weights_return_503_without_mock_fallback(client, monkeypatch):
    from api.model_registry import registry
    monkeypatch.setattr(registry.config, "enable_mock_fallback", False)
    monkeypatch.setattr(registry, "_cart_weights_loaded", False)
    monkeypatch.setattr(registry, "_cart_detector", object())
    resp = client.post("/api/v1/detect/text", json={"text": "hello world"})
    assert resp.status_code == 503


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


@pytest.fixture
def phishing_client():
    # Fresh client per test: the SDK test above closes the shared module client.
    with TestClient(app) as c:
        yield c


def test_detect_phishing(phishing_client):
    """Test POST /api/v1/detect/phishing (mock or real weights)."""
    payload = {"text": "Your account is locked, verify at http://bad.example/login", "threshold": 0.6, "max_stage": "beast"}
    resp = phishing_client.post("/api/v1/detect/phishing", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "Phishing"
    assert data["decision_threshold"] == 0.6
    assert 0.0 <= data["p_phishing"] <= 1.0
    assert data["label"] in ("phishing", "legitimate")
    assert data["stage"] in ("attack", "beast", "warhammer", "none")
    assert isinstance(data["is_mock"], bool)


def test_detect_phishing_batch_and_validation(phishing_client):
    resp = phishing_client.post("/api/v1/detect/phishing/batch", json={"texts": ["hello", "click http://x.example"]})
    assert resp.status_code == 200
    assert resp.json()["count"] == 2
    bad = phishing_client.post("/api/v1/detect/phishing", json={"text": "hi", "max_stage": "nuke"})
    assert bad.status_code == 422


def test_phishing_model_detail_and_reload(phishing_client):
    resp = phishing_client.get("/models/phishing")
    assert resp.status_code == 200
    assert "warhammer_ready" in resp.json()["details"]
    r = phishing_client.post("/models/phishing/reload", json={})
    assert r.status_code == 200
    assert r.json()["model_id"] == "phishing"
    assert phishing_client.post("/models/phishing/reload", json={"checkpoint_path": "/etc"}).status_code == 400


def test_phishing_503_without_mock_fallback(phishing_client, monkeypatch):
    from api.model_registry import registry
    monkeypatch.setattr(registry.config, "enable_mock_fallback", False)
    monkeypatch.setattr(registry, "_phishing_weights_loaded", False)
    monkeypatch.setattr(registry, "_phishing_detector", object())
    resp = phishing_client.post("/api/v1/detect/phishing", json={"text": "hello world"})
    assert resp.status_code == 503


def test_unified_includes_phishing(phishing_client):
    resp = phishing_client.post("/api/v1/detect/unified", data={"text": "Verify your password now"})
    assert resp.status_code == 200
    assert resp.json()["phishing_result"] is not None
    assert resp.json()["phishing_error"] is None


def test_unified_reports_phishing_error(phishing_client, monkeypatch):
    """A phishing failure doesn't fail the unified request but is surfaced, not swallowed."""
    from api.service import service

    def boom(*args, **kwargs):
        raise RuntimeError("phishing model unavailable")

    monkeypatch.setattr(service, "predict_phishing", boom)
    resp = phishing_client.post("/api/v1/detect/unified", data={"text": "Verify your password now"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["phishing_result"] is None
    assert "phishing model unavailable" in body["phishing_error"]
    assert "Phishing: skipped" in body["summary"]


def test_sdk_predict_phishing(phishing_client):
    from api.client import SafeStepClient
    with SafeStepClient(client=phishing_client) as sdk_client:
        single = sdk_client.predict_phishing("Reset your bank password here")
        assert single.model == "Phishing"
        batch = sdk_client.predict_phishing(["a", "b"], max_stage="attack")
        assert len(batch) == 2


def test_detect_text_rejects_ssrf_urls(client):
    """Test POST /api/v1/detect/text rejects SSRF attempts with private or link-local URLs."""
    # Loopback
    resp = client.post("/api/v1/detect/text", json={"url": "http://127.0.0.1:8000/health"})
    assert resp.status_code == 400
    assert "prohibited" in resp.json()["detail"].lower()

    # Cloud metadata
    resp_meta = client.post("/api/v1/detect/text", json={"url": "http://169.254.169.254/latest/meta-data"})
    assert resp_meta.status_code == 400
    assert "prohibited" in resp_meta.json()["detail"].lower()


def test_detect_phishing_rejects_ssrf_urls(client):
    """Test POST /api/v1/detect/phishing rejects SSRF attempts."""
    resp = client.post("/api/v1/detect/phishing", json={"url": "http://10.0.0.1/intranet"})
    assert resp.status_code == 400
    assert "prohibited" in resp.json()["detail"].lower()


def test_detect_text_with_mocked_external_url(client, monkeypatch):
    """Test POST /api/v1/detect/text successfully prompts CART with external URL text."""
    import api.security

    def fake_fetch_text(url, max_chars=100000, max_bytes=5000000):
        return "This is a comprehensive article about artificial intelligence and generative models."

    monkeypatch.setattr(api.security, "fetch_external_text", fake_fetch_text)
    resp = client.post("/api/v1/detect/text", json={"url": "https://example.com/ai-article"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "CART"
    assert "artificial intelligence" in data["text"]
    assert 0.0 <= data["p_ai"] <= 1.0


def test_detect_phishing_with_mocked_external_url(phishing_client, monkeypatch):
    """Test POST /api/v1/detect/phishing successfully prompts phishing pipeline with external URL."""
    import api.security

    def fake_fetch_text(url, max_chars=100000, max_bytes=5000000):
        return "Please enter your password and banking PIN to continue."

    monkeypatch.setattr(api.security, "fetch_external_text", fake_fetch_text)
    resp = phishing_client.post("/api/v1/detect/phishing", json={"url": "https://example.com/login"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "Phishing"
    assert "password" in data["text"]


def test_detect_unified_json_endpoint(client, sample_assets, monkeypatch):
    """Test POST /api/v1/detect/unified/json with external data and local image."""
    import api.security

    def fake_fetch_text(url, max_chars=100000, max_bytes=5000000):
        return "Breaking news: groundbreaking scientific experiment proves new physics theory."

    monkeypatch.setattr(api.security, "fetch_external_text", fake_fetch_text)
    payload = {
        "text_url": "https://example.com/news",
        "image_path": str(sample_assets["synthetic"]),
    }
    resp = client.post("/api/v1/detect/unified/json", json=payload)
    assert resp.status_code == 200
    result = resp.json()
    assert result["text_result"] is not None
    assert result["image_result"] is not None
    assert result["overall_risk"] in ("HIGH", "MEDIUM", "LOW")
