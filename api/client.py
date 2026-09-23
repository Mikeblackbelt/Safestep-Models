"""
Python client SDK for the SafeStep AI Detection API.
Allows client applications, scripts, or pipelines to query the API with Pythonic methods.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

from api.schemas import (
    BatchImageDetectionResponse,
    BatchTextDetectionResponse,
    HealthResponse,
    ImageDetectionResult,
    ModelListResponse,
    ModelStatus,
    TextDetectionResult,
    UnifiedDetectionResponse,
)


class SafeStepClient:
    """HTTP Client for communicating with a running SafeStep API server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        client: Optional[httpx.Client] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if client is not None:
            self._client = client
        else:
            self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def health(self) -> HealthResponse:
        """Check API server health."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return HealthResponse(**resp.json())

    def list_models(self) -> List[ModelStatus]:
        """List all models registered on the server."""
        resp = self._client.get("/models")
        resp.raise_for_status()
        return ModelListResponse(**resp.json()).models

    def reload_model(self, model_id: str, checkpoint_path: Optional[str] = None) -> dict:
        """Trigger a model checkpoint reload on the server."""
        payload = {"checkpoint_path": checkpoint_path} if checkpoint_path else {}
        resp = self._client.post(f"/models/{model_id}/reload", json=payload)
        resp.raise_for_status()
        return resp.json()

    def predict_text(
        self,
        text: Union[str, List[str]],
        threshold: Optional[float] = None,
        include_features: bool = False,
    ) -> Union[TextDetectionResult, List[TextDetectionResult]]:
        """
        Detect whether text is AI-generated (single string or list).
        """
        if isinstance(text, list):
            payload = {"texts": text, "threshold": threshold}
            resp = self._client.post("/api/v1/detect/batch-text", json=payload)
            resp.raise_for_status()
            batch_data = BatchTextDetectionResponse(**resp.json())
            return batch_data.results
        else:
            payload = {"text": text, "threshold": threshold}
            params = {"include_features": include_features} if include_features else {}
            resp = self._client.post("/api/v1/detect/text", json=payload, params=params)
            resp.raise_for_status()
            return TextDetectionResult(**resp.json())

    def predict_image(
        self,
        image: Union[str, Path, bytes],
        threshold: Optional[float] = None,
    ) -> ImageDetectionResult:
        """
        Detect whether an image is AI-generated (from local path or raw bytes).
        """
        data = {}
        if threshold is not None:
            data["threshold"] = str(threshold)

        if isinstance(image, (str, Path)):
            p = Path(image)
            with open(p, "rb") as f:
                content = f.read()
            files = {"file": (p.name, content, "image/octet-stream")}
            resp = self._client.post("/api/v1/detect/image", data=data, files=files)
        elif isinstance(image, bytes):
            files = {"file": ("uploaded.png", image, "image/png")}
            resp = self._client.post("/api/v1/detect/image", data=data, files=files)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        resp.raise_for_status()
        return ImageDetectionResult(**resp.json())

    def predict_unified(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Path, bytes]] = None,
        text_threshold: Optional[float] = None,
        image_threshold: Optional[float] = None,
    ) -> UnifiedDetectionResponse:
        """
        Analyze both text and image simultaneously.
        """
        data = {}
        if text:
            data["text"] = text
        if text_threshold is not None:
            data["text_threshold"] = str(text_threshold)
        if image_threshold is not None:
            data["image_threshold"] = str(image_threshold)

        files = None
        if image is not None:
            if isinstance(image, (str, Path)):
                p = Path(image)
                with open(p, "rb") as f:
                    content = f.read()
                files = {"file": (p.name, content, "image/octet-stream")}
            elif isinstance(image, bytes):
                files = {"file": ("image.png", image, "image/png")}

        resp = self._client.post("/api/v1/detect/unified", data=data, files=files)
        resp.raise_for_status()
        return UnifiedDetectionResponse(**resp.json())
