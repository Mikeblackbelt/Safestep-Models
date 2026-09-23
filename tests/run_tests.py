"""
Direct test runner for SafeStep API without needing external pytest.
Runs functional checks and REST API checks.
"""

import io
import sys
import time
from pathlib import Path
from PIL import Image

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.generate_test_assets import generate_test_assets


def run_all_tests():
    print("=" * 60)
    print("SAFESTEP FUNCTIONAL API TEST SUITE")
    print("=" * 60)

    # 1. Setup sample assets
    sample_dir = Path("tests/samples")
    auth_img, syn_img = generate_test_assets(sample_dir)

    import api
    from api.app import app
    from fastapi.testclient import TestClient

    passed = 0
    failed = 0

    def assert_test(name, condition, msg=""):
        nonlocal passed, failed
        if condition:
            print(f" [PASS] {name}")
            passed += 1
        else:
            print(f" [FAIL] {name}: {msg}")
            failed += 1

    # --- Test Model Status ---
    print("\n[1] Testing Model Registry and Status...")
    statuses = api.get_model_status()
    assert_test("Get Model Status returns list", isinstance(statuses, list) and len(statuses) >= 2)
    model_ids = {s.model_id for s in statuses}
    assert_test("CART and JAW present in status", "cart" in model_ids and "jaw" in model_ids)

    # --- Test Text Detection (Functional) ---
    print("\n[2] Testing Text Detection (CART) Functional API...")
    single_res = api.predict_text("Machine learning models are increasingly capable.")
    assert_test("predict_text single returns TextDetectionResult", hasattr(single_res, "p_ai") and hasattr(single_res, "label"))
    assert_test("p_ai within valid range [0, 1]", 0.0 <= single_res.p_ai <= 1.0)
    assert_test("label is valid", single_res.label in ("human", "ai_generated"))

    batch_res = api.predict_text([
        "The committee convened to review the quarterly financial statements.",
        "As an AI, I am unable to perform physical tasks.",
    ])
    assert_test("predict_text batch returns list of 2", isinstance(batch_res, list) and len(batch_res) == 2)

    feat_res = api.predict_text("Testing heuristic feature extraction.", include_features=True)
    assert_test("predict_text with features=True contains features dict", feat_res.features is not None and "repetition_ratio" in feat_res.features)

    # --- Test Image Detection (Functional) ---
    print("\n[3] Testing Image Detection (JAW) Functional API...")
    img_res = api.predict_image(auth_img)
    assert_test("predict_image from path returns ImageDetectionResult", hasattr(img_res, "p_ai") and hasattr(img_res, "p_authentic"))
    assert_test("Image p_ai within valid range [0, 1]", 0.0 <= img_res.p_ai <= 1.0)
    assert_test("Image label is valid", img_res.label in ("authentic", "ai_generated"))

    pil_img = Image.open(syn_img)
    img_pil_res = api.predict_image(pil_img)
    assert_test("predict_image from PIL Image succeeds", 0.0 <= img_pil_res.p_ai <= 1.0)

    with open(auth_img, "rb") as f:
        bytes_data = f.read()
    img_bytes_res = api.predict_image(bytes_data)
    assert_test("predict_image from raw bytes succeeds", 0.0 <= img_bytes_res.p_ai <= 1.0)

    # --- Test Unified Multimodal (Functional) ---
    print("\n[4] Testing Unified Multimodal Functional API...")
    mm_res = api.predict_multimodal(
        text="Please review the invoice attached in the image below.",
        image=syn_img,
    )
    assert_test("predict_multimodal returns UnifiedDetectionResponse", hasattr(mm_res, "overall_risk"))
    assert_test("Multimodal overall_risk valid", mm_res.overall_risk in ("HIGH", "MEDIUM", "LOW"))
    assert_test("Multimodal has both text and image result", mm_res.text_result is not None and mm_res.image_result is not None)

    # --- Test REST API Endpoints ---
    print("\n[5] Testing FastAPI REST Endpoints via TestClient...")
    with TestClient(app) as client:
        # GET /health
        h_resp = client.get("/health")
        assert_test("GET /health status 200", h_resp.status_code == 200)
        assert_test("GET /health contains status ok", h_resp.json().get("status") == "ok")

        # GET /models
        m_resp = client.get("/models")
        assert_test("GET /models status 200", m_resp.status_code == 200)

        # POST /api/v1/detect/text
        t_resp = client.post("/api/v1/detect/text", json={"text": "Synthesizing deep neural networks."})
        assert_test("POST /api/v1/detect/text status 200", t_resp.status_code == 200)
        assert_test("POST /api/v1/detect/text p_ai present", "p_ai" in t_resp.json())

        # POST /api/v1/detect/batch-text
        bt_resp = client.post("/api/v1/detect/batch-text", json={"texts": ["Sample 1", "Sample 2"]})
        assert_test("POST /api/v1/detect/batch-text status 200", bt_resp.status_code == 200)
        assert_test("POST /api/v1/detect/batch-text returns 2 results", len(bt_resp.json().get("results", [])) == 2)

        # POST /api/v1/detect/image (multipart file upload)
        with open(auth_img, "rb") as f:
            files = {"file": ("auth.png", f, "image/png")}
            i_resp = client.post("/api/v1/detect/image", files=files)
        assert_test("POST /api/v1/detect/image status 200", i_resp.status_code == 200)
        assert_test("POST /api/v1/detect/image returns JAW result", i_resp.json().get("model") == "JAW")

        # POST /api/v1/detect/image/json
        ij_resp = client.post("/api/v1/detect/image/json", json={"image_path": str(syn_img)})
        assert_test("POST /api/v1/detect/image/json status 200", ij_resp.status_code == 200)

        # POST /api/v1/detect/unified
        with open(syn_img, "rb") as f:
            files = {"file": ("syn.png", f, "image/png")}
            data = {"text": "Urgent security update required"}
            u_resp = client.post("/api/v1/detect/unified", files=files, data=data)
        assert_test("POST /api/v1/detect/unified status 200", u_resp.status_code == 200)
        assert_test("POST /api/v1/detect/unified has overall_risk", "overall_risk" in u_resp.json())

        # POST /models/cart/reload
        r_resp = client.post("/models/cart/reload")
        assert_test("POST /models/cart/reload status 200", r_resp.status_code == 200)

    # --- Test Python Client SDK ---
    print("\n[6] Testing SafeStepClient Python SDK...")
    from api.client import SafeStepClient
    with SafeStepClient(client=client) as sdk_client:
        sdk_health = sdk_client.health()
        assert_test("SafeStepClient health() returns ok", sdk_health.status == "ok")

        sdk_text = sdk_client.predict_text("Machine learning validation test.")
        assert_test("SafeStepClient predict_text() succeeds", sdk_text.model == "CART")

        sdk_img = sdk_client.predict_image(auth_img)
        assert_test("SafeStepClient predict_image() succeeds", sdk_img.model == "JAW")

        sdk_unified = sdk_client.predict_unified(text="Check invoice", image=syn_img)
        assert_test("SafeStepClient predict_unified() succeeds", sdk_unified.overall_risk in ("HIGH", "MEDIUM", "LOW"))

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
