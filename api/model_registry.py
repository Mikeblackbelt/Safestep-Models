"""
Model registry and lifecycle manager for SafeStep detection models.
Provides thread-safe lazy loading, device resolution, and checkpoint reloading.
"""

from dataclasses import asdict
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from api.config import APIConfig, default_api_config
from api.schemas import ModelStatus
from cart.config import CARTConfig
from cart.detector import CARTDetector
from jaw.config import JAWConfig
from jaw.detector import JAWDetector

logger = logging.getLogger("safestep.api.registry")


class SafeStepModelRegistry:
    """Central registry and manager for SafeStep models."""

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or default_api_config
        self._lock = threading.RLock()

        # Model instances
        self._cart_detector: Optional[CARTDetector] = None
        self._jaw_detector: Optional[JAWDetector] = None

        # Checkpoint statuses
        self._cart_weights_loaded = False
        self._jaw_weights_loaded = False
        self._cart_checkpoint_path: Optional[Path] = None
        self._jaw_checkpoint_path: Optional[Path] = None

        # Device determination
        self._device = self._resolve_device(self.config.device)
        logger.info(f"SafeStepModelRegistry initialized on device: {self._device}")

    def _resolve_device(self, requested: str) -> str:
        """Resolve requested device string to an available torch device."""
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        elif requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return requested

    @property
    def device(self) -> str:
        return self._device

    # -----------------------------------------------------------------------
    # CART Detector Management
    # -----------------------------------------------------------------------

    def get_cart_detector(self) -> CARTDetector:
        """Return the initialized CART detector (lazy loaded)."""
        with self._lock:
            if self._cart_detector is None:
                self._load_cart(self.config.cart_checkpoint_dir)
            return self._cart_detector

    def _load_cart(self, checkpoint_dir: Optional[Union[str, Path]]) -> None:
        """Initialize and optionally load checkpoint weights for CART."""
        cart_config = CARTConfig()
        if self.config.cart_decision_threshold is not None:
            cart_config.decision_threshold = self.config.cart_decision_threshold

        logger.info(f"Initializing CARTDetector on {self._device}...")
        detector = CARTDetector(cart_config, device=self._device)

        weights_loaded = False
        loaded_path = None

        if checkpoint_dir is not None:
            ckpt = Path(checkpoint_dir)
            neural_file = ckpt / "neural_model.pt"
            fusion_file = ckpt / "fusion_model.json"

            if neural_file.exists() and fusion_file.exists():
                logger.info(f"Found CART checkpoints at {ckpt}. Loading weights...")
                try:
                    detector.load(ckpt)
                    weights_loaded = True
                    loaded_path = ckpt
                    logger.info("CART weights loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load CART weights from {ckpt}: {e}")
            else:
                logger.info(
                    f"No complete CART checkpoint found at {ckpt} "
                    f"(needs {neural_file.name} and {fusion_file.name}). "
                    "Running with initialized architecture."
                )
                loaded_path = ckpt

        self._cart_detector = detector
        self._cart_weights_loaded = weights_loaded
        self._cart_checkpoint_path = loaded_path

    def reload_cart(self, checkpoint_dir: Optional[Union[str, Path]] = None) -> bool:
        """Force reload of the CART model from a given or default directory."""
        with self._lock:
            path = checkpoint_dir or self.config.cart_checkpoint_dir
            self._load_cart(path)
            return self._cart_weights_loaded

    # -----------------------------------------------------------------------
    # JAW Detector Management
    # -----------------------------------------------------------------------

    def get_jaw_detector(self) -> JAWDetector:
        """Return the initialized JAW detector (lazy loaded)."""
        with self._lock:
            if self._jaw_detector is None:
                self._load_jaw(self.config.jaw_checkpoint_dir)
            return self._jaw_detector

    def _load_jaw(self, checkpoint_dir: Optional[Union[str, Path]]) -> None:
        """Initialize and optionally load checkpoint weights for JAW."""
        jaw_config = JAWConfig()
        if self.config.jaw_decision_threshold is not None:
            jaw_config.decision_threshold = self.config.jaw_decision_threshold

        logger.info(f"Initializing JAWDetector on {self._device}...")
        detector = JAWDetector(jaw_config, device=self._device)

        weights_loaded = False
        loaded_path = None

        if checkpoint_dir is not None:
            ckpt = Path(checkpoint_dir)
            model_file = ckpt / "jaw_model.pt"

            if model_file.exists():
                logger.info(f"Found JAW checkpoint at {model_file}. Loading weights...")
                try:
                    detector.load(ckpt)
                    weights_loaded = True
                    loaded_path = ckpt
                    logger.info("JAW weights loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load JAW weights from {ckpt}: {e}")
            else:
                logger.info(
                    f"No JAW checkpoint found at {model_file}. "
                    "Running with initialized EfficientNet-B3 backbone."
                )
                loaded_path = ckpt

        self._jaw_detector = detector
        self._jaw_weights_loaded = weights_loaded
        self._jaw_checkpoint_path = loaded_path

    def reload_jaw(self, checkpoint_dir: Optional[Union[str, Path]] = None) -> bool:
        """Force reload of the JAW model from a given or default directory."""
        with self._lock:
            path = checkpoint_dir or self.config.jaw_checkpoint_dir
            self._load_jaw(path)
            return self._jaw_weights_loaded

    # -----------------------------------------------------------------------
    # Model Status and Inspection
    # -----------------------------------------------------------------------

    def get_status(self) -> List[ModelStatus]:
        """Return detailed status for all registered models."""
        cart_status = ModelStatus(
            model_id="cart",
            name="CART (Context-Aware Representation for Text)",
            type="text_detector",
            is_loaded=self._cart_detector is not None,
            weights_found=self._cart_weights_loaded,
            checkpoint_path=str(self._cart_checkpoint_path) if self._cart_checkpoint_path else None,
            device=self._device,
            decision_threshold=(
                self._cart_detector.config.decision_threshold
                if self._cart_detector
                else (self.config.cart_decision_threshold or 0.5)
            ),
            details={
                "backbone": "distilbert-base-uncased",
                "meta_classifier": "xgboost",
                "max_seq_length": 512,
            },
        )

        jaw_status = ModelStatus(
            model_id="jaw",
            name="JAW (Joint Artifact Wavelet Image Classifier)",
            type="image_detector",
            is_loaded=self._jaw_detector is not None,
            weights_found=self._jaw_weights_loaded,
            checkpoint_path=str(self._jaw_checkpoint_path) if self._jaw_checkpoint_path else None,
            device=self._device,
            decision_threshold=(
                self._jaw_detector.config.decision_threshold
                if self._jaw_detector
                else (self.config.jaw_decision_threshold or 0.5)
            ),
            details={
                "backbone": "efficientnet_b3",
                "frequency_branches": ["radial_spectral_energy", "gabor_wavelets"],
                "image_size": 224,
            },
        )

        return [cart_status, jaw_status]


# Global singleton instance
registry = SafeStepModelRegistry()
