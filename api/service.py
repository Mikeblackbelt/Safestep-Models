"""
Functional Service Layer for SafeStep Models.
Provides high-level, robust prediction functions for text, image, and multimodal inputs.
"""

import base64
import io
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from api.model_registry import SafeStepModelRegistry, registry
from api.schemas import (
    ImageDetectionResult,
    TextDetectionResult,
    UnifiedDetectionResponse,
)

logger = logging.getLogger("safestep.api.service")


class SafeStepService:
    """Core functional service managing model inference pipelines."""

    def __init__(self, model_registry: Optional[SafeStepModelRegistry] = None):
        self.registry = model_registry or registry

    # -----------------------------------------------------------------------
    # Text Detection (CART)
    # -----------------------------------------------------------------------

    def predict_text(
        self,
        texts: Union[str, List[str]],
        threshold: Optional[float] = None,
        include_features: bool = False,
    ) -> Union[TextDetectionResult, List[TextDetectionResult]]:
        """
        Run the CART AI-text detection pipeline on a single text or batch of texts.

        Args:
            texts: Single string or list of text strings.
            threshold: Optional threshold override (default: model config threshold, typically 0.5).
            include_features: Whether to include raw heuristic features in response.

        Returns:
            TextDetectionResult if input was a string, or List[TextDetectionResult] if a list.
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts)

        if not text_list:
            return [] if not is_single else None

        detector = self.registry.get_cart_detector()
        effective_threshold = (
            threshold if threshold is not None else detector.config.decision_threshold
        )

        results: List[TextDetectionResult] = []

        try:
            # 1. Neural probabilities
            neural_probs = detector._neural_probs(text_list)

            # 2. Heuristic extraction
            heuristic_dict_list = [detector.heuristics.extract(t) for t in text_list]
            heuristic_feats = detector.heuristics.extract_batch(text_list)

            # 3. Fusion prediction (if XGBoost is fit) or neural fallback
            if detector.fusion.model is not None:
                fused = detector.fusion.build_features(neural_probs, heuristic_feats)
                final_probs = detector.fusion.predict_proba(fused)
                is_mock = False
            else:
                # If XGBoost fusion model hasn't been fit yet, use DistilBERT neural head
                logger.debug("CART fusion model not fit; using neural classifier head.")
                final_probs = neural_probs
                is_mock = not self.registry._cart_weights_loaded

            for idx, (txt, probs) in enumerate(zip(text_list, final_probs)):
                p_human = float(probs[0])
                p_ai = float(probs[1])
                label = "ai_generated" if p_ai >= effective_threshold else "human"

                results.append(
                    TextDetectionResult(
                        text=txt,
                        p_human=round(p_human, 4),
                        p_ai=round(p_ai, 4),
                        label=label,
                        decision_threshold=effective_threshold,
                        model="CART",
                        is_mock=is_mock,
                        features=heuristic_dict_list[idx] if include_features else None,
                    )
                )

        except Exception as e:
            logger.error(f"Error during CART prediction: {e}", exc_info=True)
            # Robust fallback: use heuristic scoring
            for txt in text_list:
                feats = detector.heuristics.extract(txt)
                # Repetition and burstiness heuristic approximation
                heuristic_score = min(
                    1.0,
                    max(0.0, float(feats.get("repetition_ratio", 0.0) * 0.5 + (1.0 / max(feats.get("burstiness", 1.0), 0.1)) * 0.3)),
                )
                p_ai = heuristic_score
                p_human = 1.0 - p_ai
                label = "ai_generated" if p_ai >= effective_threshold else "human"
                results.append(
                    TextDetectionResult(
                        text=txt,
                        p_human=round(p_human, 4),
                        p_ai=round(p_ai, 4),
                        label=label,
                        decision_threshold=effective_threshold,
                        model="CART",
                        is_mock=True,
                        features=feats if include_features else None,
                    )
                )

        return results[0] if is_single else results

    # -----------------------------------------------------------------------
    # Image Detection (JAW)
    # -----------------------------------------------------------------------

    def _coerce_to_pil_and_identifier(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO, Image.Image],
    ) -> Tuple[Image.Image, str]:
        """Convert various image input representations into a PIL Image and source string."""
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            # Check if base64 encoded string
            if path_str.startswith("data:image") or len(path_str) > 1000:
                try:
                    header, data = path_str.split(",", 1) if "," in path_str else ("", path_str)
                    decoded = base64.b64decode(data)
                    pil_img = Image.open(io.BytesIO(decoded)).convert("RGB")
                    return pil_img, "base64_upload"
                except Exception:
                    pass
            # Regular file path
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Image file not found: {p}")
            pil_img = Image.open(p).convert("RGB")
            return pil_img, p.name

        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
            return pil_img, "bytes_upload"

        elif isinstance(image_input, io.BytesIO):
            pil_img = Image.open(image_input).convert("RGB")
            return pil_img, "stream_upload"

        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB"), "pil_image"

        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def predict_image(
        self,
        images: Union[
            str,
            Path,
            bytes,
            Image.Image,
            List[Union[str, Path, bytes, Image.Image]],
        ],
        threshold: Optional[float] = None,
        source_identifiers: Optional[List[str]] = None,
    ) -> Union[ImageDetectionResult, List[ImageDetectionResult]]:
        """
        Run the JAW AI-image detection pipeline on a single image or batch.

        Args:
            images: Image path, bytes, PIL Image, or list thereof.
            threshold: Optional threshold override.
            source_identifiers: Optional custom names/paths for output reporting.

        Returns:
            ImageDetectionResult if input was a single item, else List[ImageDetectionResult].
        """
        is_single = not isinstance(images, list)
        img_items = [images] if is_single else list(images)

        if not img_items:
            return [] if not is_single else None

        detector = self.registry.get_jaw_detector()
        effective_threshold = (
            threshold if threshold is not None else detector.config.decision_threshold
        )

        pil_images: List[Image.Image] = []
        sources: List[str] = []

        for i, item in enumerate(img_items):
            pil_img, default_src = self._coerce_to_pil_and_identifier(item)
            pil_images.append(pil_img)
            if source_identifiers and i < len(source_identifiers):
                sources.append(source_identifiers[i])
            else:
                sources.append(default_src)

        results: List[ImageDetectionResult] = []

        try:
            # Prepare tensors and raw uint8 arrays
            image_tensor, raw_arrays = detector.preprocessor.to_model_inputs(pil_images)
            probs = detector._predict_from_tensors(image_tensor, raw_arrays)

            for src, p in zip(sources, probs):
                p_authentic = float(p[0])
                p_ai = float(p[1])
                label = "ai_generated" if p_ai >= effective_threshold else "authentic"
                results.append(
                    ImageDetectionResult(
                        source=src,
                        p_authentic=round(p_authentic, 4),
                        p_ai=round(p_ai, 4),
                        label=label,
                        decision_threshold=effective_threshold,
                        model="JAW",
                        is_mock=not self.registry._jaw_weights_loaded,
                    )
                )

        except Exception as e:
            logger.error(f"Error during JAW prediction: {e}", exc_info=True)
            # Fallback stub
            for src in sources:
                results.append(
                    ImageDetectionResult(
                        source=src,
                        p_authentic=0.5,
                        p_ai=0.5,
                        label="authentic",
                        decision_threshold=effective_threshold,
                        model="JAW",
                        is_mock=True,
                    )
                )

        return results[0] if is_single else results

    # -----------------------------------------------------------------------
    # Unified Multimodal Detection
    # -----------------------------------------------------------------------

    def predict_unified(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Path, bytes, Image.Image]] = None,
        text_threshold: Optional[float] = None,
        image_threshold: Optional[float] = None,
    ) -> UnifiedDetectionResponse:
        """
        Analyze both text and image simultaneously and provide an aggregated risk rating.
        """
        start_time = time.perf_counter()

        text_res: Optional[TextDetectionResult] = None
        image_res: Optional[ImageDetectionResult] = None

        if text and text.strip():
            text_res = self.predict_text(text, threshold=text_threshold)

        if image is not None:
            image_res = self.predict_image(image, threshold=image_threshold)

        if text_res is None and image_res is None:
            raise ValueError("At least one of 'text' or 'image' must be provided for unified analysis.")

        # Determine overall risk
        scores = []
        summaries = []

        if text_res is not None:
            scores.append(text_res.p_ai)
            summaries.append(f"Text: {text_res.label} ({text_res.p_ai * 100:.1f}% AI)")

        if image_res is not None:
            scores.append(image_res.p_ai)
            summaries.append(f"Image: {image_res.label} ({image_res.p_ai * 100:.1f}% AI)")

        risk_score = round(float(max(scores)), 4)

        if risk_score >= 0.70:
            overall_risk = "HIGH"
        elif risk_score >= 0.40:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        summary = f"Risk {overall_risk} ({risk_score:.2f}). " + " | ".join(summaries)

        return UnifiedDetectionResponse(
            text_result=text_res,
            image_result=image_res,
            overall_risk=overall_risk,
            risk_score=risk_score,
            summary=summary,
            latency_ms=latency_ms,
        )


# Global service instance
service = SafeStepService()
