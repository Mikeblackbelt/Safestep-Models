"""
Pydantic schemas for the SafeStep API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Text Detection Schemas (CART)
# ---------------------------------------------------------------------------

class TextDetectionRequest(BaseModel):
    """Input payload for a single text detection query."""
    text: str = Field(..., min_length=1, description="Raw text to analyze for AI generation signals")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional override for decision threshold")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "The rapid advancement of deep neural networks has led to profound breakthroughs in AI.",
                "threshold": 0.5,
            }
        }
    )


class BatchTextDetectionRequest(BaseModel):
    """Input payload for batch text detection."""
    texts: List[str] = Field(..., min_length=1, description="List of raw texts to analyze")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional override for decision threshold")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "Dear team, please find the attached notes from yesterday's sync.",
                    "As an AI language model, I strive to provide balanced, comprehensive answers.",
                ],
                "threshold": 0.5,
            }
        }
    )


class TextDetectionResult(BaseModel):
    """Inference output for a single text item."""
    text: str
    p_human: float = Field(..., description="Estimated probability of human authorship")
    p_ai: float = Field(..., description="Estimated probability of AI generation")
    label: str = Field(..., description="'human' or 'ai_generated'")
    decision_threshold: float = Field(..., description="Threshold applied to p_ai")
    model: str = "CART"
    is_mock: bool = Field(False, description="True if inference ran via stub/fallback mode")
    features: Optional[Dict[str, float]] = Field(None, description="Extracted heuristic features if requested")


class BatchTextDetectionResponse(BaseModel):
    """Response containing batch text predictions."""
    results: List[TextDetectionResult]
    count: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Image Detection Schemas (JAW)
# ---------------------------------------------------------------------------

class ImageDetectionJsonRequest(BaseModel):
    """Input payload for image detection via JSON (path, base64, or URL)."""
    image_path: Optional[str] = Field(None, description="Local filesystem path to an image file")
    image_base64: Optional[str] = Field(None, description="Base64-encoded image bytes")
    image_url: Optional[str] = Field(None, description="URL pointing to an image to fetch and analyze")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional override for decision threshold")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_path": "tests/samples/sample_face.png",
                "threshold": 0.5,
            }
        }
    )


class ImageDetectionResult(BaseModel):
    """Inference output for a single image item."""
    source: str = Field(..., description="Source path, filename, or URL identifier")
    p_authentic: float = Field(..., description="Estimated probability that image is authentic / camera-captured")
    p_ai: float = Field(..., description="Estimated probability that image is AI-generated / synthetic")
    label: str = Field(..., description="'authentic' or 'ai_generated'")
    decision_threshold: float = Field(..., description="Threshold applied to p_ai")
    model: str = "JAW"
    is_mock: bool = Field(False, description="True if inference ran via stub/fallback mode")


class BatchImageDetectionResponse(BaseModel):
    """Response containing batch image predictions."""
    results: List[ImageDetectionResult]
    count: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Unified Multimodal Detection Schemas
# ---------------------------------------------------------------------------

class UnifiedDetectionResponse(BaseModel):
    """Combined multimodal analysis for text and image artifacts."""
    text_result: Optional[TextDetectionResult] = None
    image_result: Optional[ImageDetectionResult] = None
    overall_risk: str = Field(..., description="'HIGH', 'MEDIUM', 'LOW', or 'UNKNOWN'")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Composite synthetic content risk score")
    summary: str = Field(..., description="Human-readable assessment summary")
    latency_ms: float


# ---------------------------------------------------------------------------
# Model Management & Health Schemas
# ---------------------------------------------------------------------------

class ModelStatus(BaseModel):
    """Status and configuration metadata for a registered model."""
    model_id: str
    name: str
    type: str  # "text_detector" | "image_detector"
    is_loaded: bool
    weights_found: bool
    checkpoint_path: Optional[str] = None
    device: str
    decision_threshold: float
    details: Dict[str, Any] = Field(default_factory=dict)


class ModelListResponse(BaseModel):
    """List of all available models in the registry."""
    models: List[ModelStatus]
    system_device: str


class ReloadModelRequest(BaseModel):
    """Request to reload a model from an optional new checkpoint directory."""
    checkpoint_path: Optional[str] = Field(None, description="Optional new directory path containing model weights")


class HealthResponse(BaseModel):
    """Server health and operational status."""
    status: str = "ok"
    version: str = "1.0.0"
    uptime_seconds: float
    device: str
    models_loaded: Dict[str, bool]
