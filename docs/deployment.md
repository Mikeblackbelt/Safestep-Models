# SafeStep Deployment: Production Architecture, API Reference & Operations

SafeStep ships a production-ready, asynchronous HTTP microservice built on **FastAPI** (`api/app.py`) and backed by an in-memory **Model Registry** (`api/model_registry.py`). It serves real-time inferences for **CART** (AI text), **JAW** (AI images), and the **Phishing** escalation cascade (Attack $\rightarrow$ Beast $\rightarrow$ Warhammer), as well as a unified multimodal threat evaluator.

---

## Table of Contents
- [1. Service Architecture & Component Overview](#1-service-architecture--component-overview)
- [2. Running the API](#2-running-the-api)
  - [2.1 CLI Server Runner](#21-cli-server-runner)
  - [2.2 Direct Uvicorn Execution](#22-direct-uvicorn-execution)
- [3. Complete REST API Endpoint Reference](#3-complete-rest-api-endpoint-reference)
  - [3.1 System & Model Management](#31-system--model-management)
  - [3.2 Text Detection Endpoints (CART)](#32-text-detection-endpoints-cart)
  - [3.3 Image Detection Endpoints (JAW)](#33-image-detection-endpoints-jaw)
  - [3.4 Phishing Detection Endpoints](#34-phishing-detection-endpoints)
  - [3.5 Unified Multimodal Detection Endpoint](#35-unified-multimodal-detection-endpoint)
- [4. Configuration & Environment Variables](#4-configuration--environment-variables)
- [5. Zero-Downtime Checkpoint Hot-Reloading](#5-zero-downtime-checkpoint-hot-reloading)
- [6. Docker Containerization & Orchestration](#6-docker-containerization--orchestration)
- [7. Production Reverse Proxy & Security Hardening](#7-production-reverse-proxy--security-hardening)

---

## 1. Service Architecture & Component Overview

```
                                  Client Request (HTTP/REST)
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     Uvicorn ASGI HTTP Server     │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     FastAPI Application Router   │
                              │       • CORS Middleware          │
                              │       • Process Timing (X-Header)│
                              │       • Request Size Validator   │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     DetectionService Layer       │
                              │       • Pydantic Serialization   │
                              │       • Batch Array Slicing      │
                              │       • Error Code Mapping       │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │       SafeStep Model Registry    │
                              │       • Lazy Model Instantiation │
                              │       • Checkpoint State Cache   │
                              │       • Atomic Weight Hot-Swap   │
                              └────────────────┬─────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               ▼                               ▼                               ▼
       ┌───────────────┐               ┌───────────────┐               ┌───────────────┐
       │ CART Detector │               │ JAW Detector  │               │ Phishing Pipe │
       │ (Neural+XGB)  │               │ (CNN+Wavelet) │               │ (3-Tier Gate) │
       └───────────────┘               └───────────────┘               └───────────────┘
```

The service is decoupled into distinct architectural layers:
- **`api/app.py`**: Initializes the FastAPI app, attaches CORS middleware, and measures total execution time via the `X-Process-Time-Ms` response header.
- **`api/service.py`**: Business logic orchestration. Decodes image payloads (multipart/form-data, base64 strings, or direct file paths) and maps detector outputs to Pydantic schemas.
- **`api/model_registry.py`**: Manages singleton instances of `CARTDetector`, `JAWDetector`, and `PhishingDetector`. Implements thread-safe lazy loading and atomic hot-reloading.
- **`api/schemas.py`**: Strongly typed data contracts enforcing validation on thresholds, batch lengths, and response structures.

---

## 2. Running the API

### 2.1 CLI Server Runner
The easiest way to launch the server is through the `api.cli` module:

```bash
# Start server on default host (0.0.0.0) and port (8000)
python -m api.cli serve

# Custom host and port
python -m api.cli serve --host 127.0.0.1 --port 9000

# Development mode with auto-reload
python -m api.cli serve --reload
```

Interactive OpenAPI documentation is generated automatically:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### 2.2 Direct Uvicorn Execution
For multi-worker deployment in production:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4 --loop uvloop --http httptools
```

---

## 3. Complete REST API Endpoint Reference

Every response includes the `X-Process-Time-Ms` header indicating processing latency in milliseconds.

### 3.1 System & Model Management

#### Health Check
```http
GET /health
```
**Response (200 OK):**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2026-10-21T11:00:00Z"
}
```

#### List Model Statuses
```http
GET /models
```
**Response (200 OK):**
```json
[
  {
    "model_id": "cart",
    "is_loaded": true,
    "weights_found": true,
    "checkpoint_path": "checkpoints/cart",
    "details": {
      "neural_backbone": "distilbert-base-uncased",
      "fusion_model_loaded": true
    }
  },
  {
    "model_id": "jaw",
    "is_loaded": true,
    "weights_found": true,
    "checkpoint_path": "checkpoints/jaw",
    "details": {
      "spatial_backbone": "efficientnet_b3",
      "frequency_features": 26
    }
  },
  {
    "model_id": "phishing",
    "is_loaded": true,
    "weights_found": true,
    "checkpoint_path": "checkpoints/phishing",
    "details": {
      "attack_ready": true,
      "beast_ready": true,
      "warhammer_ready": true
    }
  }
]
```

---

### 3.2 Text Detection Endpoints (CART)

#### Single Text Sample
```http
POST /api/v1/detect/text
Content-Type: application/json

{
  "text": "Furthermore, it is essential to consider the ramifications of this policy.",
  "threshold": 0.50,
  "include_features": true
}
```

**Response (200 OK):**
```json
{
  "p_human": 0.0821,
  "p_ai": 0.9179,
  "label": "ai_generated",
  "decision_threshold": 0.50,
  "features": {
    "perplexity_proxy": 14.28,
    "burstiness": 0.312,
    "repetition_ratio": 0.450,
    "unique_ngram_ratio": 0.620,
    "max_ngram_repeat": 4,
    "punctuation_regularity": 0.125,
    "sentence_length_mean": 21.4,
    "sentence_length_std": 2.1,
    "capitalization_consistency": 1.0,
    "rare_word_ratio": 0.35,
    "transition_word_density": 0.082,
    "avg_word_length": 5.4
  }
}
```

#### Batch Text Detection
```http
POST /api/v1/detect/batch-text
Content-Type: application/json

{
  "texts": [
    "Human written paragraph with organic vocabulary flow.",
    "As an artificial intelligence language model, I recommend..."
  ],
  "threshold": 0.50
}
```

---

### 3.3 Image Detection Endpoints (JAW)

#### Multipart File Upload
```http
POST /api/v1/detect/image
Content-Type: multipart/form-data

file: [binary image file (.png, .jpg, .webp)]
threshold: 0.50
```

**Response (200 OK):**
```json
{
  "p_authentic": 0.0345,
  "p_ai": 0.9655,
  "label": "ai_generated",
  "decision_threshold": 0.50
}
```

#### JSON Payload (Base64 or Server Path)
```http
POST /api/v1/detect/image/json
Content-Type: application/json

{
  "image_path": "path/to/image.png",
  "threshold": 0.60
}
```

---

### 3.4 Phishing Detection Endpoints

```http
POST /api/v1/detect/phishing
Content-Type: application/json

{
  "text": "URGENT: Your bank account has been locked. Verify immediately at http://192.168.1.1/login.php",
  "threshold": 0.50,
  "max_stage": "warhammer"
}
```

**Response (200 OK):**
```json
{
  "p_legitimate": 0.0120,
  "p_phishing": 0.9880,
  "label": "phishing",
  "decision_threshold": 0.50,
  "stage": "beast",
  "is_mock": false
}
```

---

### 3.5 Unified Multimodal Detection Endpoint

```http
POST /api/v1/detect/unified
Content-Type: multipart/form-data

text: "Urgent: Please review the updated account access badge attached."
file: [binary image file]
include_phishing: true
text_threshold: 0.50
image_threshold: 0.50
phishing_threshold: 0.50
```

**Response (200 OK):**
```json
{
  "overall_risk": "HIGH",
  "risk_score": 0.9420,
  "summary": "AI image detected (p=0.94); text authentic (p=0.12); phishing suspected (p=0.88). Overall HIGH risk.",
  "text_result": {
    "p_human": 0.8800,
    "p_ai": 0.1200,
    "label": "human",
    "decision_threshold": 0.50
  },
  "image_result": {
    "p_authentic": 0.0580,
    "p_ai": 0.9420,
    "label": "ai_generated",
    "decision_threshold": 0.50
  },
  "phishing_result": {
    "p_legitimate": 0.1200,
    "p_phishing": 0.8800,
    "label": "phishing",
    "decision_threshold": 0.50,
    "stage": "attack",
    "is_mock": false
  },
  "phishing_error": null
}
```

---

## 4. Configuration & Environment Variables

SafeStep reads configuration parameters from environment variables (`api/config.py`):

| Variable | Default | Operational Effect |
|---|---|---|
| `SAFESTEP_API_HOST` | `0.0.0.0` | Binding network interface. |
| `SAFESTEP_API_PORT` | `8000` | Binding TCP port. |
| `SAFESTEP_DEVICE` | `auto` | Inference device (`auto`, `cpu`, `cuda`, `mps`). |
| `SAFESTEP_CART_CHECKPOINT` | `checkpoints/cart` | Path to CART model directory (`neural_model.pt`, `fusion_model.json`). |
| `SAFESTEP_JAW_CHECKPOINT` | `checkpoints/jaw` | Path to JAW model directory (`jaw_model.pt`). |
| `SAFESTEP_PHISHING_CHECKPOINT`| `checkpoints/phishing` | Path to Phishing directory (`attack.joblib`, `beast.json`, `warhammer.pt`). |
| `SAFESTEP_CART_THRESHOLD` | `0.50` | Default classification threshold for CART. |
| `SAFESTEP_JAW_THRESHOLD` | `0.50` | Default classification threshold for JAW. |
| `SAFESTEP_PHISHING_THRESHOLD`| `0.50` | Default classification threshold for Phishing. |
| `SAFESTEP_ADMIN_TOKEN` | unset | Secret token required in `X-Admin-Token` header for `/reload` operations. If unset, reloading is open (dev only). |
| `SAFESTEP_CORS_ORIGINS` | unset ($\rightarrow$ `*`)| Comma-separated list of allowed browser origins. Set to empty string `""` to deny all cross-origin requests. |
| `SAFESTEP_MOCK_FALLBACK` | `true` (dev) / `false` (Docker)| When `false`, missing checkpoints cause requests to fail with HTTP `503 Service Unavailable`. When `true`, random weights are served. |

---

## 5. Zero-Downtime Checkpoint Hot-Reloading

When training finishes, models can be reloaded in-place without restarting the service or dropping requests:

```bash
# Reload CART
curl -X POST http://localhost:8000/models/cart/reload \
     -H "X-Admin-Token: $SAFESTEP_ADMIN_TOKEN"

# Reload JAW with a custom path
curl -X POST http://localhost:8000/models/jaw/reload \
     -H "X-Admin-Token: $SAFESTEP_ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"checkpoint_path": "checkpoints/jaw_run_42"}'
```

Path traversal attacks are blocked: custom `checkpoint_path` targets must reside inside the configured checkpoint directory root.

---

## 6. Docker Containerization & Orchestration

The root `Dockerfile` provides a production container based on `python:3.12-slim`:
- Installs CPU-optimized PyTorch wheels.
- Pre-caches DistilBERT tokenizers and weights.
- Runs as an unprivileged non-root user (`safestep`).
- Hardens defaults: sets `SAFESTEP_MOCK_FALLBACK=false` and disables CORS by default.

### Build and Run:
```bash
# Build the container image
docker build -t safestep:latest .

# Run container with mounted checkpoint volume
docker run -d \
  --name safestep-api \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  -e SAFESTEP_ADMIN_TOKEN="your-secure-secret-token" \
  -e SAFESTEP_CORS_ORIGINS="https://dashboard.example.com" \
  safestep:latest
```

---

## 7. Production Reverse Proxy & Security Hardening

In production, place the SafeStep container behind **Nginx** or **Cloudflare** for TLS termination, rate limiting, and request buffering:

```nginx
# Sample Nginx upstream configuration
server {
    listen 443 ssl http2;
    server_name api.safestep.internal;

    # SSL certificates
    ssl_certificate /etc/ssl/certs/safestep.crt;
    ssl_certificate_key /etc/ssl/private/safestep.key;

    # Support up to 25 MB image uploads
    client_max_body_size 25M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (if enabled)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Restrict reload endpoint to internal subnets
    location ~* ^/models/[^/]+/reload$ {
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://127.0.0.1:8000;
    }
}
```
