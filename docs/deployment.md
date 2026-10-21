# SafeStep Deployment: Production Architecture, API Reference & Cloud Operations

SafeStep ships a production-ready, asynchronous HTTP microservice built on **FastAPI** (`api/app.py`) and backed by an in-memory **Model Registry** (`api/model_registry.py`). It serves real-time inferences for **CART** (AI text), **JAW** (AI images), and the **Phishing** escalation cascade (Attack $\rightarrow$ Beast $\rightarrow$ Warhammer), as well as a unified multimodal threat evaluator.

---

## Table of Contents
- [1. Service Architecture & Component Overview](#1-service-architecture--component-overview)
- [2. Running the API Locally](#2-running-the-api-locally)
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
- [8. Hardware Resource Requirements & Sizing Guide](#8-hardware-resource-requirements--sizing-guide)
- [9. Cloud Services & Pricing Matrix (Free, Cheap, High Performance)](#9-cloud-services--pricing-matrix-free-cheap-high-performance)
- [10. Step-by-Step Cloud Deployment Walkthroughs](#10-step-by-step-cloud-deployment-walkthroughs)
  - [10.1 Free Tier: Hugging Face Spaces (Docker Space)](#101-free-tier-hugging-face-spaces-docker-space)
  - [10.2 Serverless Tier: Google Cloud Run (Free / Pay-Per-Request)](#102-serverless-tier-google-cloud-run-free--pay-per-request)
  - [10.3 Budget VPS Tier: Hetzner / DigitalOcean (Docker Compose + SSL)](#103-budget-vps-tier-hetzner--digitalocean-docker-compose--ssl)
  - [10.4 High-Performance GPU Tier: RunPod / Lambda / AWS G5 (CUDA)](#104-high-performance-gpu-tier-runpod--lambda--aws-g5-cuda)
- [11. Production Operational Best Practices & Optimization](#11-production-operational-best-practices--optimization)

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

## 2. Running the API Locally

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

---

## 8. Hardware Resource Requirements & Sizing Guide

SafeStep detectors are engineered to be lightweight and operational across diverse environments—from micro virtual machines to multi-GPU clusters.

### Resource Footprint per Detector

| Model | Component Breakdown | Active RAM | VRAM (if GPU) | Latency (CPU) | Latency (GPU) | Throughput (per vCPU) |
|---|---|---|---|---|---|---|
| **CART** | DistilBERT (66M params) + MLP + XGBoost | $\sim 450$ MB | $\sim 800$ MB | $15\text{--}35$ ms | $3\text{--}8$ ms | $30\text{--}50$ req/s |
| **JAW** | EfficientNet-B3 (10.7M) + 26-d FFT/Gabor + MLP | $\sim 600$ MB | $\sim 1.1$ GB | $40\text{--}90$ ms | $8\text{--}15$ ms | $15\text{--}25$ req/s |
| **Phishing** | Attack ($<0.5$ ms) $\rightarrow$ Beast ($2$ ms) $\rightarrow$ Warhammer | $\sim 400$ MB | $\sim 800$ MB | $2\text{--}30$ ms | $1\text{--}6$ ms | $200\text{--}500$ req/s |
| **Unified** | All 3 models loaded simultaneously | $\sim 1.4\text{--}1.8$ GB | $\sim 2.2$ GB | $50\text{--}110$ ms | $10\text{--}20$ ms | $10\text{--}20$ req/s |

### Hardware Sizing Recommendations

| Target Workload | Minimum CPU | Minimum RAM | Recommended Storage | Recommended Device |
|---|---|---|---|---|
| **Development / Testing** | 2 vCPU | 2 GB | 10 GB SSD | CPU |
| **Light Production ($<10$ req/sec)** | 2 vCPU | 4 GB | 20 GB SSD | CPU |
| **Standard Production ($10\text{--}80$ req/sec)** | 4 vCPU | 8 GB | 40 GB NVMe | CPU (or budget GPU) |
| **High Throughput ($>100$ req/sec)** | 8+ vCPU | 16 GB | 80 GB NVMe | Dedicated GPU (A10G / L4 / RTX 4090) |

---

## 9. Cloud Services & Pricing Matrix (Free, Cheap, High Performance)

Depending on your budget, latency SLA, and anticipated traffic volume, SafeStep can be hosted on a variety of battle-tested cloud providers across three pricing tiers:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 SafeStep Cloud Spectrum                                │
├────────────────────────┬───────────────────────────────┬───────────────────────────────┤
│    Tier 1: Free Tier   │     Tier 2: Budget VPS        │    Tier 3: High Performance   │
│       ($0 / month)     │     ($5 – $35 / month)        │     ($50 – $300+ / month)     │
├────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ • Hugging Face Spaces  │ • Hetzner Cloud (CPX21/31)    │ • RunPod Secure Cloud (A10G)  │
│ • Oracle Always Free   │ • DigitalOcean / Linode       │ • AWS ECS / EKS (g5.xlarge)   │
│ • Google Cloud Run     │ • AWS Lightsail / EC2 Graviton│ • GCP Cloud Run with L4 GPU   │
│ • Render Free Service  │ • RunPod Community Cloud (GPU)│ • Dedicated Bare Metal Server │
└────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

### Comprehensive Pricing & Provider Comparison

| Provider & Service | Pricing Model | Included Specs | Strengths | Limitations | Best Suited For |
|---|---|---|---|---|---|
| **Hugging Face Spaces** | **Free ($0/mo)** | 2 vCPU, 16 GB RAM, 50 GB disk | 16 GB RAM at zero cost, free HTTPS & Git-push deploy | Sleeps after 48h idle, CPU only | Public demo, portfolio, staging testing |
| **Oracle Cloud Always Free** | **Free ($0/mo)** | Up to 4 ARM OCPUs, 24 GB RAM, 200 GB disk | 24/7 uptime without sleep, massive 24GB RAM | Strict sign-up verification, requires ARM Docker build | Non-profit 24/7 production, student labs |
| **Google Cloud Run (Free Tier)** | **Free ($0/mo)** | 2M requests/mo, 360k GB-sec free | Pay-only-for-traffic, scales to 0, Google edge routing | Cold-starts on zero traffic ($\sim 5\text{--}8$s) | Sporadic or bursty API traffic |
| **Render Free Tier** | **Free ($0/mo)** | 0.5 CPU, 512 MB RAM | Dead simple setup | Low RAM (only suitable for single model CART) | Quick toy testing |
| **Hetzner Cloud (CPX21/CPX31)** | **Cheap (~$6 – $16/mo)** | 3 to 4 AMD vCPU, 4 to 8 GB RAM, 80-160 GB NVMe | **Unbeatable cost-to-performance**, high CPU clock speed, 20TB egress | Self-managed Linux VM (requires basic DevOps) | **Production recommendation for budget startups** |
| **DigitalOcean Droplets** | **Cheap ($6 – $24/mo)** | 1 to 2 vCPU, 1 to 4 GB RAM, 25-80 GB SSD | 1-click Docker marketplace, global datacenters | Moderate bandwidth pricing | Rapid prototyping and standard web apps |
| **AWS Lightsail / t4g.medium** | **Cheap ($10 – $25/mo)** | 2 vCPU (AWS Graviton ARM), 4 GB RAM | Native AWS VPC integration, highly reliable | AWS ecosystem lock-in | Teams running within existing AWS infrastructures |
| **RunPod Community GPU** | **Cheap ($0.20 – $0.40/hr)** | 1x RTX 3090 / 4090 (24 GB VRAM), 8 vCPU, 32 GB RAM | Blazing fast GPU inference at consumer prices | Community peer-to-peer hosting (variable reliability) | High-volume batch evaluation & fine-tuning |
| **RunPod Secure / Lambda Labs** | **High Perf ($0.70 – $1.20/hr)** | 1x NVIDIA A10G (24 GB VRAM), 8 vCPU, 30 GB RAM | Dedicated datacenter GPU, 99.99% SLA, ultra-low latency | Requires on-demand or reserved commitment | Real-time security gateway ($>100$ req/s) |
| **AWS ECS on EC2 (`g5.xlarge`)** | **High Perf (~$1.00/hr, ~$450–$720/mo)** | 1x NVIDIA A10G (24 GB VRAM), 4 vCPU, 16 GB RAM | Enterprise compliance (SOC2/HIPAA), IAM, CloudWatch | High idle cost if unreserved | High-scale enterprise infrastructure |
| **GCP Cloud Run with L4 GPU** | **High Perf (Pay-per-second, ~$0.70/hr active)** | 1x NVIDIA L4 (24 GB VRAM), 4 vCPU, 16 GB RAM | Serverless GPU that scales to zero | Higher cost during continuous sustained traffic | Real-time GPU inference with variable traffic |
| **Hetzner Dedicated Server** | **High Perf ($40 – $75/mo flat)** | 8–16 CPU cores (Intel i7/AMD Ryzen), 64 GB DDR4 RAM | **Flat monthly price**, unmetered gigabit network, raw performance | Setup fee, European locations | Massive CPU throughput (millions of daily calls) |

---

## 10. Step-by-Step Cloud Deployment Walkthroughs

### 10.1 Free Tier: Hugging Face Spaces (Docker Space)
Hugging Face provides **16 GB RAM** and **2 vCPUs** completely free in their basic Docker Spaces.

1. **Create Space:**
   - Navigate to [huggingface.co/new-space](https://huggingface.co/new-space).
   - Name your Space: `safestep-detector`.
   - Select **Docker** as the Space SDK.
   - Choose the free **CPU basic (2 vCPU, 16 GB RAM)** hardware.

2. **Add `README.md` Frontmatter:**
   Create a `README.md` in your Space root with the following YAML header:
   ```yaml
   ---
   title: SafeStep Detection API
   emoji: 🛡️
   colorFrom: blue
   colorTo: green
   sdk: docker
   app_port: 8000
   ---
   ```

3. **Deploy Repository:**
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/safestep-detector
   git push space main
   ```
   Hugging Face will automatically build your `Dockerfile`, download model weights, and expose the FastAPI app on a public SSL endpoint: `https://YOUR_USERNAME-safestep-detector.hf.space`.

---

### 10.2 Serverless Tier: Google Cloud Run (Free / Pay-Per-Request)
Google Cloud Run scales containers from 0 to 1,000+ instances automatically, incurring **$0 cost** when no requests arrive.

1. **Authenticate and Set Project:**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_GCP_PROJECT_ID
   ```

2. **Build and Submit Container Image to Google Artifact Registry:**
   ```bash
   gcloud builds submit --tag gcr.io/YOUR_GCP_PROJECT_ID/safestep:latest .
   ```

3. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy safestep-api \
     --image gcr.io/YOUR_GCP_PROJECT_ID/safestep:latest \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 4Gi \
     --cpu 2 \
     --concurrency 40 \
     --min-instances 0 \
     --max-instances 10 \
     --set-env-vars "SAFESTEP_DEVICE=cpu,SAFESTEP_MOCK_FALLBACK=false"
   ```
   Cloud Run outputs a production URL: `https://safestep-api-xyz-uc.a.run.app`.

---

### 10.3 Budget VPS Tier: Hetzner / DigitalOcean (Docker Compose + SSL)
For under **$15/month**, a dedicated virtual server from Hetzner (CPX21) or DigitalOcean runs SafeStep with high availability and no cold starts.

1. **Provision Server:**
   - OS: Ubuntu 24.04 LTS.
   - Hardware: 3 vCPU, 4 GB RAM.

2. **Install Docker & Docker Compose:**
   ```bash
   sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin certbot python3-certbot-nginx
   sudo usermod -aG docker $USER
   ```

3. **Create `docker-compose.yml`:**
   ```yaml
   version: "3.8"
   services:
     safestep:
       image: safestep:latest
       build: .
       restart: always
       ports:
         - "127.0.0.1:8000:8000"
       environment:
         - SAFESTEP_DEVICE=cpu
         - SAFESTEP_ADMIN_TOKEN=your-random-production-token
         - SAFESTEP_CORS_ORIGINS=https://yourdomain.com
       volumes:
         - ./checkpoints:/app/checkpoints:ro
   ```

4. **Launch Application:**
   ```bash
   docker compose up -d --build
   ```

5. **Configure Nginx & Free Let's Encrypt SSL:**
   ```bash
   # Obtain automated SSL certificate
   sudo certbot --nginx -d api.yourdomain.com
   ```

6. **Create Systemd Auto-Start Service (`/etc/systemd/system/safestep.service`):**
   ```ini
   [Unit]
   Description=SafeStep Detection Microservice
   After=docker.service
   Requires=docker.service

   [Service]
   WorkingDirectory=/opt/Safestep-Models
   ExecStart=/usr/bin/docker compose up
   ExecStop=/usr/bin/docker compose down
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```
   ```bash
   sudo systemctl daemon-reload && sudo systemctl enable --now safestep
   ```

---

### 10.4 High-Performance GPU Tier: RunPod / Lambda / AWS G5 (CUDA)
For enterprise workloads demanding $<15$ ms response times across all detectors:

1. **Provision GPU Instance:**
   - 1x NVIDIA A10G (AWS EC2 `g5.xlarge`) or 1x RTX 4090 (RunPod).
   - Ensure NVIDIA Container Toolkit (`nvidia-docker2`) is installed.

2. **Run Container with Full GPU Acceleration:**
   ```bash
   docker run -d \
     --name safestep-gpu \
     --gpus all \
     -p 8000:8000 \
     -v $(pwd)/checkpoints:/app/checkpoints:ro \
     -e SAFESTEP_DEVICE=cuda \
     -e SAFESTEP_MOCK_FALLBACK=false \
     -e SAFESTEP_ADMIN_TOKEN="super-secret-token" \
     safestep:latest
   ```

3. **Verify CUDA Device Initialization:**
   ```bash
   curl http://localhost:8000/models
   ```
   The response will confirm `"device": "cuda"` and load models onto GPU memory.

---

## 11. Production Operational Best Practices & Optimization

### Worker Sizing & Concurrency
- **CPU Deployments:** Set Uvicorn workers to:
  $$\text{Workers} = 2 \times \text{vCPU Cores} + 1$$
  *(For a 4-core server, run 9 Uvicorn workers).*
- **GPU Deployments:** Keep Uvicorn workers low ($\le 2$ workers per GPU) to avoid CUDA out-of-memory context thrashing. Let FastAPI's asynchronous event loop queue incoming requests.

### Cold-Start Mitigation
On serverless platforms (Cloud Run, AWS Lambda):
- Keep `--min-instances 1` if sub-second first-request latency is required.
- Set up a lightweight cron ping hitting `GET /health` every 4 minutes to keep container warm.

### Monitoring & Telemetry
- Inspect the `X-Process-Time-Ms` response header in client applications or reverse proxies to track 95th/99th percentile latencies.
- In Docker, monitor CPU and memory utilization live:
  ```bash
  docker stats safestep-api
  ```
