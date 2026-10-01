# SafeStep Models

Detection models for AI-generated and malicious content.

| Model | Package | Detects | Approach |
|-------|---------|---------|----------|
| **CART** | `cart/` | AI-generated text | DistilBERT + stylistic heuristics, fused with XGBoost |
| **JAW** | `jaw/` | AI-generated images | EfficientNet-B3 + FFT spectral / Gabor texture features |
| Phishing (Attack / Beast / Warhammer) | `util/loaders/phishing.py` | Phishing emails & sites | Planned — data loaders only |

## Layout

```
cart/       CART text detector (config, model, heuristics, fusion, detector, trainer)
jaw/        JAW image detector (config, model, frequency features, preprocessing, detector, trainer)
training/   Training dashboard: device metrics, load planner, parallel sessions, run history, charts
util/       Shared dataset handling, loaders, and logging
api/        Python functional API + FastAPI REST service (see api/README.md)
tests/      pytest suite for the API
train.py    Training CLI for CART and JAW
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then edit values, especially SAFESTEP_ADMIN_TOKEN
```

## Training

```bash
# Estimate time only
python -m train --model cart --data-dir data/ --estimate-only

# Train one or both models
python -m train --model jaw --data-dir data/ --device cuda
python -m train --model all --data-dir data/ --workers 8
```

Checkpoints are written to `checkpoints/cart` and `checkpoints/jaw` by default. Model weights and
datasets are not committed (`checkpoints/` and `data/` are gitignored).

### Training dashboard

```bash
# Live full-screen dashboard
python -m train --model all --data-dir data/ --dashboard

# See the plan (memory, throughput, parallelism, ETA) without training
python -m train --model all --data-dir data/ --dashboard --estimate-only

# Hyper-parameter sweep; sessions run in parallel where the hardware allows
python -m train --model cart --sweep lr=1e-5,3e-5 --sweep batch_size=8,16

# Previous and partial runs, with accuracy / throughput charts
python -m train --history
```

Before launching, the planner measures free RAM/VRAM and cores, estimates each session's peak memory
and throughput, and simulates every feasible level of parallelism to pick the shortest schedule
(capped with `--max-parallel`). Throughput estimates start from conservative baselines and are
calibrated from your own previous runs on the same device type, so ETAs get more accurate over time.

While training, the dashboard shows per-session progress and live ETA, per-process CPU/RAM, system
utilisation, and validation-accuracy and loss charts overlaid with your best previous run. Each
session runs in its own process and logs to `runs/<run_id>/<session>.log`.

Every run (dashboard or not) is recorded under `runs/` after each epoch. If training is interrupted
or crashes, the metrics up to the last finished epoch are kept and show up as *partial* in `--history`.

## Serving

```bash
python -m api.cli serve --port 8000
```

Interactive docs are then available at `http://localhost:8000/docs`. See [api/README.md](api/README.md)
for the Python API, REST endpoints, and client usage.

## Tests

```bash
pytest
```
