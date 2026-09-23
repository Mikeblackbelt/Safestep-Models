"""
FastAPI application for SafeStep AI detection models.
Provides endpoints for single and batch text detection, image detection, and model lifecycle control.
"""

from contextlib import asynccontextmanager
import io
import logging
import time
from typing import List, Optional

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import default_api_config
from api.model_registry import registry
from api.schemas import (
    BatchImageDetectionResponse,
    BatchTextDetectionRequest,
    BatchTextDetectionResponse,
    HealthResponse,
    ImageDetectionJsonRequest,
    ImageDetectionResult,
    ModelListResponse,
    ModelStatus,
    ReloadModelRequest,
    TextDetectionRequest,
    TextDetectionResult,
    UnifiedDetectionResponse,
)
from api.service import service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("safestep.api")

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: optionally pre-load models at startup."""
    logger.info("SafeStep API server starting up...")
    if not default_api_config.lazy_load:
        logger.info("Eager loading models as configured...")
        try:
            registry.get_cart_detector()
            registry.get_jaw_detector()
        except Exception as e:
            logger.warning(f"Could not eager-load all models on startup: {e}")
    yield
    logger.info("SafeStep API server shutting down...")


app = FastAPI(
    title="SafeStep AI Detection API",
    description=(
        "Production-grade functional API for calling trained SafeStep AI detection models:\n"
        "- **CART**: AI-Generated Text Detector (DistilBERT + Stylistic Heuristics + XGBoost Fusion)\n"
        "- **JAW**: AI-Generated Image Detector (EfficientNet-B3 + Spectral/Gabor Wavelet Frequency Analysis)\n"
        "- **Unified**: Multi-modal synthetic content risk analyzer"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=default_api_config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    return response


# ---------------------------------------------------------------------------
# Health and Info Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    summary="Root index",
    tags=["General"],
)
def root():
    return {
        "service": "SafeStep AI Detection API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "text_detection": "/api/v1/detect/text",
            "batch_text_detection": "/api/v1/detect/batch-text",
            "image_detection": "/api/v1/detect/image",
            "image_json_detection": "/api/v1/detect/image/json",
            "batch_image_detection": "/api/v1/detect/batch-images",
            "unified_detection": "/api/v1/detect/unified",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health_check():
    """Returns system status, device info, and loaded model flags."""
    statuses = registry.get_status()
    models_loaded = {s.model_id: s.is_loaded for s in statuses}
    return HealthResponse(
        status="ok",
        version="1.0.0",
        uptime_seconds=round(time.time() - START_TIME, 2),
        device=registry.device,
        models_loaded=models_loaded,
    )


@app.get(
    "/models",
    response_model=ModelListResponse,
    summary="List available models",
    tags=["Models"],
)
def list_models():
    """List all registered models, their configurations, and checkpoint status."""
    return ModelListResponse(
        models=registry.get_status(),
        system_device=registry.device,
    )


@app.get(
    "/models/{model_id}",
    response_model=ModelStatus,
    summary="Get model details",
    tags=["Models"],
)
def get_model(model_id: str):
    """Retrieve detailed status for a specific model ('cart' or 'jaw')."""
    for status_item in registry.get_status():
        if status_item.model_id.lower() == model_id.lower():
            return status_item
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Model '{model_id}' not found. Available models: 'cart', 'jaw'",
    )


@app.post(
    "/models/{model_id}/reload",
    summary="Reload model checkpoint",
    tags=["Models"],
)
def reload_model(model_id: str, request: Optional[ReloadModelRequest] = None):
    """
    Force reload model weights from a specified checkpoint path or default path.
    Allows hot-reloading weights after a training run completes without restarting the server.
    """
    path = request.checkpoint_path if request else None
    mid = model_id.lower()
    if mid == "cart":
        loaded = registry.reload_cart(path)
        return {
            "model_id": "cart",
            "reloaded": True,
            "weights_loaded": loaded,
            "checkpoint_path": str(registry._cart_checkpoint_path),
        }
    elif mid == "jaw":
        loaded = registry.reload_jaw(path)
        return {
            "model_id": "jaw",
            "reloaded": True,
            "weights_loaded": loaded,
            "checkpoint_path": str(registry._jaw_checkpoint_path),
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown model '{model_id}'. Available: 'cart', 'jaw'",
        )


# ---------------------------------------------------------------------------
# Text Detection Endpoints (CART)
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/detect/text",
    response_model=TextDetectionResult,
    summary="Analyze text for AI generation signals",
    tags=["Text Detection (CART)"],
)
def detect_text(
    payload: TextDetectionRequest,
    include_features: bool = Query(False, description="Include extracted heuristic feature values"),
):
    """
    Run CART pipeline on a single text string:
    - DistilBERT neural classification
    - Stylistic heuristic extraction (repetition, burstiness, punctuation)
    - XGBoost meta-classifier fusion
    """
    try:
        return service.predict_text(
            texts=payload.text,
            threshold=payload.threshold,
            include_features=include_features,
        )
    except Exception as e:
        logger.error(f"Error in detect_text endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post(
    "/api/v1/detect/batch-text",
    response_model=BatchTextDetectionResponse,
    summary="Batch analyze multiple texts",
    tags=["Text Detection (CART)"],
)
def detect_batch_text(payload: BatchTextDetectionRequest):
    """Process a batch of texts concurrently through the CART pipeline."""
    if len(payload.texts) > default_api_config.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(payload.texts)} exceeds limit of {default_api_config.max_batch_size}",
        )
    start_time = time.perf_counter()
    try:
        results = service.predict_text(texts=payload.texts, threshold=payload.threshold)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return BatchTextDetectionResponse(
            results=results,
            count=len(results),
            latency_ms=duration_ms,
        )
    except Exception as e:
        logger.error(f"Error in detect_batch_text endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ---------------------------------------------------------------------------
# Image Detection Endpoints (JAW)
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/detect/image",
    response_model=ImageDetectionResult,
    summary="Analyze uploaded image for AI generation artifacts",
    tags=["Image Detection (JAW)"],
)
async def detect_image(
    file: UploadFile = File(..., description="Image file to analyze (PNG, JPEG, WebP)"),
    threshold: Optional[float] = Form(None, description="Optional decision threshold override"),
):
    """
    Run JAW pipeline on an uploaded image file:
    - EfficientNet-B3 spatial feature backbone
    - 2D FFT spectral radial energy distribution
    - Multi-scale Gabor filter texture response
    - Dense fusion classification
    """
    try:
        content = await file.read()
        if len(content) > default_api_config.max_upload_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image size exceeds limit of {default_api_config.max_upload_size_bytes // (1024 * 1024)}MB",
            )
        result = service.predict_image(
            images=content,
            threshold=threshold,
            source_identifiers=[file.filename or "uploaded_image"],
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_image endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image: {e}")


@app.post(
    "/api/v1/detect/image/json",
    response_model=ImageDetectionResult,
    summary="Analyze image via JSON (file path, URL, or base64)",
    tags=["Image Detection (JAW)"],
)
def detect_image_json(payload: ImageDetectionJsonRequest):
    """Analyze an image provided via local path, base64 data URI, or URL."""
    img_target = None
    identifier = "json_image"

    if payload.image_path:
        img_target = payload.image_path
        identifier = payload.image_path
    elif payload.image_base64:
        img_target = payload.image_base64
        identifier = "base64_payload"
    elif payload.image_url:
        import urllib.request
        try:
            req = urllib.request.Request(payload.image_url, headers={"User-Agent": "SafeStep-API/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                img_target = resp.read()
            identifier = payload.image_url
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to fetch image from URL: {e}",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide one of 'image_path', 'image_base64', or 'image_url'",
        )

    try:
        return service.predict_image(
            images=img_target,
            threshold=payload.threshold,
            source_identifiers=[identifier],
        )
    except Exception as e:
        logger.error(f"Error in detect_image_json endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post(
    "/api/v1/detect/batch-images",
    response_model=BatchImageDetectionResponse,
    summary="Batch analyze multiple uploaded images",
    tags=["Image Detection (JAW)"],
)
async def detect_batch_images(
    files: List[UploadFile] = File(..., description="List of image files to analyze"),
    threshold: Optional[float] = Form(None),
):
    """Process multiple images in a single batch forward pass."""
    if len(files) > default_api_config.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Number of files {len(files)} exceeds batch limit {default_api_config.max_batch_size}",
        )

    start_time = time.perf_counter()
    image_bytes_list = []
    names = []

    for f in files:
        b = await f.read()
        image_bytes_list.append(b)
        names.append(f.filename or "image")

    try:
        results = service.predict_image(
            images=image_bytes_list,
            threshold=threshold,
            source_identifiers=names,
        )
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return BatchImageDetectionResponse(
            results=results,
            count=len(results),
            latency_ms=duration_ms,
        )
    except Exception as e:
        logger.error(f"Error in batch image detection: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ---------------------------------------------------------------------------
# Unified Multimodal Detection Endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/detect/unified",
    response_model=UnifiedDetectionResponse,
    summary="Unified multimodal analysis (text + image)",
    tags=["Multimodal Unified Detection"],
)
async def detect_unified(
    text: Optional[str] = Form(None, description="Optional text content to analyze"),
    file: Optional[UploadFile] = File(None, description="Optional image file to analyze"),
    text_threshold: Optional[float] = Form(None),
    image_threshold: Optional[float] = Form(None),
):
    """
    Combined inspection for content containing both text and visual media.
    Produces individual verdicts and a synthesized overall synthetic content risk rating.
    """
    if not text and file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide at least one of 'text' or 'file' for unified analysis.",
        )

    image_content = None
    if file is not None:
        image_content = await file.read()

    try:
        return service.predict_unified(
            text=text,
            image=image_content,
            text_threshold=text_threshold,
            image_threshold=image_threshold,
        )
    except Exception as e:
        logger.error(f"Error in detect_unified endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
