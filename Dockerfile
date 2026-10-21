FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    # Production defaults: fail loudly on missing weights, no cross-origin browser access.
    SAFESTEP_MOCK_FALLBACK=false \
    SAFESTEP_CORS_ORIGINS=""

WORKDIR /app

# CPU-only torch keeps the image ~1.5 GB smaller; swap the index URL for a CUDA build if needed.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt

COPY . .

# Bake the DistilBERT weights in so the container starts without network access.
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
AutoModel.from_pretrained('distilbert-base-uncased'); AutoTokenizer.from_pretrained('distilbert-base-uncased')"

RUN useradd --create-home --uid 1000 safestep && chown -R safestep /app
USER safestep

# Trained checkpoints are mounted here at runtime: -v ./checkpoints:/app/checkpoints
VOLUME ["/app/checkpoints"]
EXPOSE 8000

# Container health monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["python", "-m", "api.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
