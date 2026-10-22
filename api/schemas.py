"""
Pydantic schemas for the SafeStep API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Text Detection Schemas (CART)
# ---------------------------------------------------------------------------

class TextDetectionRequest(BaseModel):
    """Input payload for a single text detection query (direct text or external URL)."""
    text: Optional[str] = Field(None, min_length=1, max_length=100000, description="Raw text to analyze for AI generation signals")
    url: Optional[str] = Field(None, max_length=2048, description="External URL to fetch text content from securely")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional override for decision threshold")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "The rapid advancement of deep neural networks has led to profound breakthroughs in AI.",
                "url": "https://example.com/sample-article",
                "threshold": 0.5,
            }
        }
    )


class BatchTextDetectionRequest(BaseModel):
    """Input payload for batch text detection."""
    texts: List[str] = Field(..., min_length=1, max_length=128, description="List of raw texts to analyze")
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
    image_path: Optional[str] = Field(None, max_length=2048, description="Local filesystem path to an image file")
    image_base64: Optional[str] = Field(None, max_length=35000000, description="Base64-encoded image bytes")
    image_url: Optional[str] = Field(None, max_length=2048, description="URL pointing to an image to fetch and analyze")
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
# Phishing Detection Schemas (Attack -> Beast -> Warhammer)
# ---------------------------------------------------------------------------

class PhishingDetectionRequest(BaseModel):
    """Input payload for a single phishing detection query (direct text or external URL)."""
    text: Optional[str] = Field(None, min_length=1, max_length=100000, description="Email body, URL, or page text")
    url: Optional[str] = Field(None, max_length=2048, description="External URL whose live content will be fetched securely and analyzed for phishing")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional override for decision threshold")
    max_stage: str = Field("warhammer", pattern="^(attack|beast|warhammer)$", description="Stop escalating after this stage")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Your account is locked. Verify now at http://secure-login.example.ru/verify",
                "url": "https://example.com/login",
                "threshold": 0.5,
                "max_stage": "warhammer",
            }
        }
    )


class BatchPhishingDetectionRequest(BaseModel):
    """Input payload for batch phishing detection."""
    texts: List[str] = Field(..., min_length=1, max_length=128, description="List of emails / URLs / page texts")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional override for decision threshold")
    max_stage: str = Field("warhammer", pattern="^(attack|beast|warhammer)$", description="Stop escalating after this stage")


class PhishingDetectionResult(BaseModel):
    """Inference output for a single phishing query."""
    text: str
    p_legitimate: float = Field(..., description="Estimated probability the message is legitimate")
    p_phishing: float = Field(..., description="Estimated probability the message is phishing")
    label: str = Field(..., description="'legitimate' or 'phishing'")
    stage: str = Field(..., description="Stage that made the call: 'attack', 'beast', 'warhammer', or 'none' (mock)")
    decision_threshold: float = Field(..., description="Threshold applied to p_phishing")
    model: str = "Phishing"
    is_mock: bool = Field(False, description="True if no trained checkpoint is loaded (neutral 0.5 result)")


class BatchPhishingDetectionResponse(BaseModel):
    """Response containing batch phishing predictions."""
    results: List[PhishingDetectionResult]
    count: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Unified Multimodal Detection Schemas
# ---------------------------------------------------------------------------

class UnifiedDetectionResponse(BaseModel):
    """Combined multimodal analysis for text and image artifacts."""
    text_result: Optional[TextDetectionResult] = None
    image_result: Optional[ImageDetectionResult] = None
    phishing_result: Optional[PhishingDetectionResult] = None
    phishing_error: Optional[str] = Field(
        None, description="Why phishing analysis was skipped (e.g. model unavailable); null when it ran"
    )
    overall_risk: str = Field(..., description="'HIGH', 'MEDIUM', 'LOW', or 'UNKNOWN'")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Composite synthetic content risk score")
    summary: str = Field(..., description="Human-readable assessment summary")
    latency_ms: float


class UnifiedDetectionJsonRequest(BaseModel):
    """Input payload for multimodal analysis via JSON (allowing external data URLs)."""
    text: Optional[str] = Field(None, max_length=100000, description="Optional raw text content")
    text_url: Optional[str] = Field(None, max_length=2048, description="Optional external URL to fetch text from")
    image_path: Optional[str] = Field(None, max_length=2048, description="Optional local filesystem image path")
    image_url: Optional[str] = Field(None, max_length=2048, description="Optional external image URL")
    image_base64: Optional[str] = Field(None, max_length=35000000, description="Optional base64 image data")
    text_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    image_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    phishing_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    include_phishing: bool = Field(True, description="Also run the phishing pipeline on the text")


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


# ---------------------------------------------------------------------------
# Corpus Ingestion & Groq Poisoning Assessment Schemas
# ---------------------------------------------------------------------------

class PoisoningReportSchema(BaseModel):
    """Result of Groq LLM-as-a-Judge poisoning & accuracy analysis."""
    poisoning_risk: float = Field(..., ge=0.0, le=1.0, description="Estimated risk score from 0.0 (Clean) to 1.0 (Poisoning)")
    verdict: str = Field(..., description="'CLEAN', 'SUSPICIOUS', or 'POISONING_ATTEMPT'")
    is_poisoning_suspected: bool
    flags: List[str] = Field(default_factory=list)
    groq_independent_label: Optional[str] = None
    groq_user_agreement: Optional[bool] = None
    groq_model_agreement: Optional[bool] = None
    model_user_agreement: Optional[bool] = None
    confidence: float
    analysis_reasoning: str
    evaluator_provider: str


class CorpusInputRequest(BaseModel):
    """Input payload for submitting user data with evaluation to the training corpus."""
    model_type: str = Field(..., pattern="^(cart|jaw|phishing)$", description="Target model: 'cart', 'jaw', or 'phishing'")
    user_eval: str = Field(..., min_length=1, max_length=50, description="User's ground-truth evaluation label (e.g. 'human', 'ai_generated', 'legitimate', 'phishing')")
    text: Optional[str] = Field(None, max_length=100000, description="Raw text sample (CART or Phishing)")
    url: Optional[str] = Field(None, max_length=2048, description="External URL to fetch text from or analyze")
    image_path: Optional[str] = Field(None, max_length=2048, description="Local path to image (JAW)")
    image_url: Optional[str] = Field(None, max_length=2048, description="External URL to fetch image from (JAW)")
    image_base64: Optional[str] = Field(None, max_length=35000000, description="Base64 image data (JAW)")
    check_poisoning: bool = Field(True, description="Whether to dynamically assess sample for data poisoning")
    reject_poisoning: bool = Field(True, description="If True, rejects ingestion into corpus if high-risk poisoning is detected")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata (e.g. evaluator ID, notes, tags)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_type": "cart",
                "text": "This essay was written entirely by me for my college assignment.",
                "user_eval": "human",
                "check_poisoning": True,
                "metadata": {"user_id": "auditor_42", "notes": "Verified authentic text"},
            }
        }
    )


class CorpusInputResponse(BaseModel):
    """Response returned upon submitting data to the corpus."""
    status: str = Field(..., description="'success' or 'rejected_poisoning'")
    submission_id: str
    model_type: str
    canonical_label: str
    binary_label: int
    model_prediction: Optional[Dict[str, Any]] = None
    model_user_agreement: Optional[bool] = None
    poisoning_assessment: Optional[PoisoningReportSchema] = None
    added_to_corpus: bool
    corpus_file: Optional[str] = None
    timestamp: str


class GroqAssessRequest(BaseModel):
    """Request payload for direct Groq data poisoning & model accuracy assessment."""
    text: Optional[str] = Field(None, max_length=100000, description="Text to assess")
    url: Optional[str] = Field(None, max_length=2048, description="External URL to fetch text from")
    user_eval: str = Field(..., max_length=50, description="Evaluation label assigned to the text")
    model_type: str = Field("cart", pattern="^(cart|jaw|phishing)$", description="Target model")
    groq_api_key: Optional[str] = Field(None, max_length=256, description="Optional Groq API key override")


class CorpusAnalyticsResponse(BaseModel):
    """Comprehensive analytics and metrics for user-contributed corpora."""
    total_submissions: int
    models: Dict[str, Any]
    poisoning_summary: Dict[str, Any]
    overall_model_agreement_rate: Optional[float] = None
    recent_submissions: List[Dict[str, Any]] = Field(default_factory=list)
