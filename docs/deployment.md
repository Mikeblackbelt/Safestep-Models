# Deployment

The FastAPI service in `api/` serves **CART** (AI text), **JAW** (AI images) and the **phishing** pipeline (Attack -> Beast -> Warhammer).

---

## Running the API

```bash
python -m api.cli serve                     # 0.0.0.0:8000
python -m api.cli serve --host 127.0.0.1 --port 9000
python -m api.cli serve --reload            # auto-reload for development
```

Swagger UI is at `/docs` and ReDoc at `/redoc`.

`api.cli` has a few other subcommands for local checks that don't start a server:

```bash
python -m api.cli status                                   # registered models + checkpoint status (JSON)
python -m api.cli predict-text "Some text" [--threshold 0.6] [--features]
python -m api.cli predict-phishing "Verify your account at http://..." [--threshold 0.6] [--max-stage attack|beast|warhammer]
python -m api.cli predict-image path/to/img.png [--threshold 0.6]
python -m api.cli predict-unified --text "..." --image img.png [--text-threshold T] [--image-threshold T]
```

### Endpoints

| Method | Path |
|---|---|
| GET | `/`, `/health`, `/models`, `/models/{model_id}` |
| POST | `/models/{model_id}/reload` (optional body `{"checkpoint_path": "..."}`) |
| POST | `/api/v1/detect/text`, `/api/v1/detect/batch-text` |
| POST | `/api/v1/detect/image` (upload), `/api/v1/detect/image/json`, `/api/v1/detect/batch-images` |
| POST | `/api/v1/detect/phishing` (`{"text", "threshold"?, "max_stage"?}`), `/api/v1/detect/phishing/batch` (`{"texts": [...], ...}`) |
| POST | `/api/v1/detect/unified` (form fields `text`, `file`, `text_threshold`, `image_threshold`, `phishing_threshold`, `include_phishing`) |

Phishing results carry `p_phishing`, `label` (`phishing`/`legitimate`), `stage` (which stage decided) and `is_mock`. The unified endpoint also runs phishing on `text` and returns `phishing_result`; it only feeds `risk_score` when real weights are loaded.

`model_id` is `cart`, `jaw` or `phishing`. Every response carries an `X-Process-Time-Ms` header.

---

## Configuration

`api/config.py` (`APIConfig`) reads these environment variables:

| Variable | Default | Effect |
|---|---|---|
| `SAFESTEP_API_HOST` | `0.0.0.0` | Default for `serve --host` |
| `SAFESTEP_API_PORT` | `8000` | Default for `serve --port` |
| `SAFESTEP_DEVICE` | `auto` | Inference device (`auto`, `cpu`, `cuda`, `mps`) |
| `SAFESTEP_CART_CHECKPOINT` | `checkpoints/cart` | CART checkpoint directory |
| `SAFESTEP_JAW_CHECKPOINT` | `checkpoints/jaw` | JAW checkpoint directory |
| `SAFESTEP_PHISHING_CHECKPOINT` | `checkpoints/phishing` | Phishing checkpoint directory (also an allowed reload root) |
| `SAFESTEP_CART_THRESHOLD` | model default (`0.5`) | CART decision threshold |
| `SAFESTEP_JAW_THRESHOLD` | model default (`0.5`) | JAW decision threshold |
| `SAFESTEP_PHISHING_THRESHOLD` | model default (`0.5`) | Phishing decision threshold on `p_phishing` |
| `SAFESTEP_ADMIN_TOKEN` | unset | Required in the `X-Admin-Token` header for `POST /models/{id}/reload`. Unset leaves reload open (dev only) |
| `SAFESTEP_CORS_ORIGINS` | unset → `*` | Comma-separated allowed browser origins; empty string allows none |
| `SAFESTEP_MOCK_FALLBACK` | `true` (image: `false`) | If `false`, requests to a model without trained weights return `503` instead of untrained output |

Things to know:

- **The API doesn't load `.env` itself.** Only `train.py` does. Export the variables in your shell, or use `docker run --env-file .env`.
- Relative checkpoint paths resolve against the working directory the server starts in.
- Other `APIConfig` fields (`lazy_load`, `max_batch_size=128`, `max_upload_size_bytes=25 MB`, `workers`) can only be changed in code.

---

## Checkpoint layout

```
checkpoints/
├── cart/
│   ├── neural_model.pt
│   └── fusion_model.json
├── jaw/
│   └── jaw_model.pt
└── phishing/
    ├── attack.joblib
    ├── beast.json
    └── warhammer.pt     # optional
```

Models load lazily on first use. If the checkpoint files are missing, the server still starts. With `SAFESTEP_MOCK_FALLBACK=false` (the Docker default) detection requests for that model return `503`; with `true` it serves an untrained model (random CART head/fusion, or JAW with a random head). Phishing in mock mode returns a neutral `p_phishing=0.5`, `label=legitimate`, `stage=none`, `is_mock=true`. Phishing weights count as loaded once `attack.joblib` and `beast.json` load; Warhammer (DistilBERT) is only constructed when `warhammer.pt` exists (`details.warhammer_ready`). Check `GET /models` (or `python -m api.cli status`) and confirm `weights_found` is `true` before trusting any output.

After retraining, hot-swap the weights without a restart:

```bash
curl -X POST http://localhost:8000/models/cart/reload -H "X-Admin-Token: $SAFESTEP_ADMIN_TOKEN"
curl -X POST http://localhost:8000/models/jaw/reload -H "X-Admin-Token: $SAFESTEP_ADMIN_TOKEN" -H "Content-Type: application/json" \
     -d '{"checkpoint_path": "checkpoints/jaw"}'
```

A custom `checkpoint_path` must sit inside the configured checkpoint directory (e.g. `checkpoints/`), otherwise the request is rejected with `400`.

---

## Docker

The `Dockerfile` at the repo root builds from `python:3.12-slim`, installs CPU-only PyTorch and `requirements.txt`, copies the code into `/app`, exposes port 8000, and runs:

```
python -m api.cli serve --host 0.0.0.0 --port 8000
```

Checkpoints aren't baked into the image. Mount them at `/app/checkpoints`:

```bash
docker build -t safestep .
docker run -p 8000:8000 -v ./checkpoints:/app/checkpoints --env-file .env safestep
```

- Inside the container, the default `checkpoints/cart` and `checkpoints/jaw` resolve to `/app/checkpoints/...`. Keep `SAFESTEP_*_CHECKPOINT` relative, or set it to `/app/checkpoints/...`.
- The command passes `--host` and `--port` explicitly, so `SAFESTEP_API_HOST` and `SAFESTEP_API_PORT` have no effect in the container. Change the published port with `-p <host>:8000`.
- The image is CPU-only. `SAFESTEP_DEVICE=auto` resolves to CPU.
- DistilBERT is baked into the image at build time. The EfficientNet-B3 base weights are still downloaded on first JAW load, so that first request needs network access.
- The image sets `SAFESTEP_MOCK_FALLBACK=false` and `SAFESTEP_CORS_ORIGINS=""` and runs as a non-root user.

---

## Production checklist

- [ ] **Set `SAFESTEP_ADMIN_TOKEN`** to a long random secret (`python -c "import secrets; print(secrets.token_hex(32))"`); without it `/models/{id}/reload` is unauthenticated.
- [ ] **Restrict CORS** with `SAFESTEP_CORS_ORIGINS=https://your-frontend.example`.
- [ ] Keep `SAFESTEP_MOCK_FALLBACK=false` so missing weights fail with `503` instead of returning junk.
- [ ] **Mount checkpoints** at `/app/checkpoints` and confirm `GET /models` reports `weights_found: true` for `cart`, `jaw` and `phishing`.
- [ ] Never commit `.env`.
- [ ] Put the service behind a reverse proxy / TLS terminator, and size request limits to match `max_upload_size_bytes` (25 MB).
