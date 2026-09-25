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
