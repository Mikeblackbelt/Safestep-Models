# SafeStep AI Detection API

A unified, functional API for running inference on the SafeStep suite of AI detection models:
- **CART**: Context-Aware Representation for Text (DistilBERT + Stylistic Heuristics + XGBoost Meta-Fusion)
- **JAW**: Joint Artifact Wavelet Image Classifier (EfficientNet-B3 + FFT Radial Spectral Energy + Gabor Wavelets)
- **Unified / Multimodal**: Synthetic content risk assessor combining text and visual analyses.

The API is accessible in two ways:
1. **Direct Python Functional API** (`import api`): Seamless in-memory inference in Python scripts, pipelines, and notebooks.
2. **REST API Service** (FastAPI): Production-grade HTTP microservice with Swagger docs, batching, image upload, and dynamic model reloading.

---

## 1. Direct Python Functional API

Call the models directly with simple Python functions:

```python
import api

# --- Text Detection (CART) ---
# Single string
result = api.predict_text("Artificial intelligence has seen rapid advancements.")
print(f"Label: {result.label} (p_ai: {result.p_ai:.4f}, p_human: {result.p_human:.4f})")

# Batch strings
batch_results = api.predict_text([
    "Human written paragraph with varied rhythm.",
    "As an AI language model, I recommend following best practices."
])
for r in batch_results:
    print(r.text[:30], "->", r.label, r.p_ai)

# Include stylistic heuristic features (repetition, burstiness, punctuation, etc.)
res_with_feats = api.predict_text("Sample text to analyze.", include_features=True)
print(res_with_feats.features)


# --- Image Detection (JAW) ---
# From local file path
img_result = api.predict_image("path/to/image.png")
print(f"Image Label: {img_result.label} (p_ai: {img_result.p_ai:.4f})")

# From PIL Image object or raw bytes
from PIL import Image
pil_img = Image.open("path/to/image.png")
img_result = api.predict_image(pil_img)

with open("path/to/image.png", "rb") as f:
    raw_bytes = f.read()
img_result = api.predict_image(raw_bytes)


# --- Unified Multimodal Detection ---
mm_result = api.predict_multimodal(
    text="Urgent: Your account has been suspended. Review attached badge.",
    image="path/to/badge.png"
)
print("Risk Level:", mm_result.overall_risk)  # HIGH, MEDIUM, or LOW
print("Risk Score:", mm_result.risk_score)
print("Summary:", mm_result.summary)


# --- Inspect or Reload Models ---
statuses = api.get_model_status()
for s in statuses:
    print(s.model_id, "Loaded:", s.is_loaded, "Weights:", s.weights_found)

# Hot reload model checkpoint after a training run
api.reload_model("cart", checkpoint_path="checkpoints/cart")
```

---

## 2. Running the REST API Server

### Start with CLI
```bash
# Start server on default port 8000
python -m api.cli serve

# Start on custom host/port with auto-reload
python -m api.cli serve --host 0.0.0.0 --port 8080 --reload
```

### Or Start with Uvicorn
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Once running, interactive OpenAPI / Swagger documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 3. REST API Endpoints

### Text Detection
- `POST /api/v1/detect/text`
  ```bash
  curl -X POST http://localhost:8000/api/v1/detect/text \
    -H "Content-Type: application/json" \
    -d '{"text": "The rapid advancement of deep learning has revolutionized NLP.", "threshold": 0.5}'
  ```

- `POST /api/v1/detect/batch-text`
  ```bash
  curl -X POST http://localhost:8000/api/v1/detect/batch-text \
    -H "Content-Type: application/json" \
    -d '{"texts": ["Text sample one", "Text sample two"]}'
  ```

### Image Detection
- `POST /api/v1/detect/image` (Multipart Form File Upload)
  ```bash
  curl -X POST http://localhost:8000/api/v1/detect/image \
    -F "file=@/path/to/sample.png" \
    -F "threshold=0.5"
  ```

- `POST /api/v1/detect/image/json` (Path, URL, or Base64)
  ```bash
  curl -X POST http://localhost:8000/api/v1/detect/image/json \
    -H "Content-Type: application/json" \
    -d '{"image_path": "tests/samples/sample_authentic.png"}'
  ```

- `POST /api/v1/detect/batch-images` (Multiple File Upload)
  ```bash
  curl -X POST http://localhost:8000/api/v1/detect/batch-images \
    -F "files=@image1.png" \
    -F "files=@image2.png"
  ```

### Unified Multimodal Detection
- `POST /api/v1/detect/unified`
  ```bash
  curl -X POST http://localhost:8000/api/v1/detect/unified \
    -F "text=Verify your banking details" \
    -F "file=@suspicious_receipt.png"
  ```

### System & Model Lifecycle Management
- `GET /health`: Uptime, device (CPU/CUDA/MPS), and loaded model flags.
- `GET /models`: List model architectures, checkpoint paths, and thresholds.
- `GET /models/{model_id}`: Status for `cart` or `jaw`.
- `POST /models/{model_id}/reload`: Dynamically reload model weights after training without restarting server:
  ```bash
  curl -X POST http://localhost:8000/models/cart/reload \
    -H "Content-Type: application/json" \
    -d '{"checkpoint_path": "checkpoints/cart"}'
  ```

---

## 4. Python Client SDK

Call the remote API from another machine or script using `SafeStepClient`:

```python
from api.client import SafeStepClient

client = SafeStepClient("http://localhost:8000")

# Check health
health = client.health()
print(health.status, health.device)

# Predict text
result = client.predict_text("Sample input text to classify.")
print(result.label, result.p_ai)

# Predict image
img_result = client.predict_image("/path/to/image.png")
print(img_result.label, img_result.p_ai)

# Unified analysis
unified = client.predict_unified(
    text="Click here to claim your reward.",
    image="/path/to/promo.png"
)
print(unified.overall_risk, unified.summary)
```

---

## 5. Configuration & Environment Variables

| Variable | Description | Default |
|---|---|---|
| `SAFESTEP_API_HOST` | Server host to bind | `0.0.0.0` |
| `SAFESTEP_API_PORT` | Server port | `8000` |
| `SAFESTEP_DEVICE` | Execution device (`auto`, `cpu`, `cuda`) | `auto` |
| `SAFESTEP_CART_CHECKPOINT` | Directory containing `neural_model.pt` & `fusion_model.json` | `checkpoints/cart` |
| `SAFESTEP_JAW_CHECKPOINT` | Directory containing `jaw_model.pt` | `checkpoints/jaw` |
| `SAFESTEP_CART_THRESHOLD` | Default decision threshold override for CART | `0.5` |
| `SAFESTEP_JAW_THRESHOLD` | Default decision threshold override for JAW | `0.5` |
