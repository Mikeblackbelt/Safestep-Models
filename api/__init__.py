"""
SafeStep Functional API.

Usage as a Python functional module:
    >>> from api import predict_text, predict_image, predict_multimodal
    >>> # Text detection (CART)
    >>> result = predict_text("Here is a sample sentence.")
    >>> print(result.label, result.p_ai)
    >>>
    >>> # Image detection (JAW)
    >>> img_result = predict_image("path/to/image.png")
    >>> print(img_result.label, img_result.p_ai)
    >>>
    >>> # Multimodal analysis
    >>> mm_result = predict_multimodal(text="Click this link", image="path/to/badge.png")
    >>> print(mm_result.overall_risk)

Usage as an HTTP REST Service:
    $ python -m api.cli serve --port 8000
    Or:
    $ uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

from typing import Any, List, Optional, Union
from pathlib import Path
from PIL import Image

from api.config import APIConfig, default_api_config
from api.model_registry import SafeStepModelRegistry, registry
from api.service import SafeStepService, service
from api.client import SafeStepClient
from api.schemas import (
    TextDetectionResult,
    ImageDetectionResult,
    UnifiedDetectionResponse,
    ModelStatus,
)


def predict_text(
    texts: Union[str, List[str]],
    threshold: Optional[float] = None,
    include_features: bool = False,
) -> Union[TextDetectionResult, List[TextDetectionResult]]:
    """
    Functional entrypoint: Run CART AI-text detection on a text string or list of strings.
    """
    return service.predict_text(
        texts=texts,
        threshold=threshold,
        include_features=include_features,
    )


def predict_image(
    images: Union[str, Path, bytes, Image.Image, List[Union[str, Path, bytes, Image.Image]]],
    threshold: Optional[float] = None,
    source_identifiers: Optional[List[str]] = None,
) -> Union[ImageDetectionResult, List[ImageDetectionResult]]:
    """
    Functional entrypoint: Run JAW AI-image detection on an image path, raw bytes, PIL Image, or list.
    """
    return service.predict_image(
        images=images,
        threshold=threshold,
        source_identifiers=source_identifiers,
    )


def predict_multimodal(
    text: Optional[str] = None,
    image: Optional[Union[str, Path, bytes, Image.Image]] = None,
    text_threshold: Optional[float] = None,
    image_threshold: Optional[float] = None,
) -> UnifiedDetectionResponse:
    """
    Functional entrypoint: Run simultaneous text and image synthetic content detection with unified risk score.
    """
    return service.predict_unified(
        text=text,
        image=image,
        text_threshold=text_threshold,
        image_threshold=image_threshold,
    )


def get_model_status() -> List[ModelStatus]:
    """Return status of all registered models (CART, JAW)."""
    return registry.get_status()


def reload_model(model_id: str, checkpoint_path: Optional[Union[str, Path]] = None) -> bool:
    """
    Reload model weights dynamically from disk.
    model_id: 'cart' or 'jaw'.
    """
    mid = model_id.lower()
    if mid == "cart":
        return registry.reload_cart(checkpoint_path)
    elif mid == "jaw":
        return registry.reload_jaw(checkpoint_path)
    raise ValueError(f"Unknown model_id '{model_id}'. Expected 'cart' or 'jaw'.")


__all__ = [
    "predict_text",
    "predict_image",
    "predict_multimodal",
    "get_model_status",
    "reload_model",
    "registry",
    "service",
    "SafeStepClient",
    "APIConfig",
    "default_api_config",
    "TextDetectionResult",
    "ImageDetectionResult",
    "UnifiedDetectionResponse",
    "ModelStatus",
]
