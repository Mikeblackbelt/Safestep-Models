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
from phishing.config import PhishingConfig
from phishing.detector import PhishingDetector

logger = logging.getLogger("safestep.api.registry")


class ModelUnavailableError(RuntimeError):
    """Raised when a model has no trained weights and mock fallback is disabled."""


class SafeStepModelRegistry:
    """Central registry and manager for SafeStep models."""

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or default_api_config
        self._lock = threading.RLock()

        # Model instances
        self._cart_detector: Optional[CARTDetector] = None
        self._jaw_detector: Optional[JAWDetector] = None
        self._phishing_detector: Optional[PhishingDetector] = None

        # Checkpoint statuses
        self._cart_weights_loaded = False
        self._jaw_weights_loaded = False
        self._cart_checkpoint_path: Optional[Path] = None
        self._jaw_checkpoint_path: Optional[Path] = None
        self._phishing_weights_loaded = False
        self._phishing_checkpoint_path: Optional[Path] = None

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

    def _require_weights(self, model_id: str, loaded: bool, path) -> None:
        """Refuse to serve untrained weights unless mock fallback is enabled."""
        if not loaded and not self.config.enable_mock_fallback:
            raise ModelUnavailableError(
                f"{model_id} checkpoint not loaded from {path}; "
                "mount trained weights or set SAFESTEP_MOCK_FALLBACK=true"
            )

    # -----------------------------------------------------------------------
    # CART Detector Management
    # -----------------------------------------------------------------------

    def get_cart_detector(self) -> CARTDetector:
        """Return the initialized CART detector (lazy loaded)."""
        with self._lock:
            if self._cart_detector is None:
                self._load_cart(self.config.cart_checkpoint_dir)
            self._require_weights("cart", self._cart_weights_loaded, self._cart_checkpoint_path)
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
            self._require_weights("jaw", self._jaw_weights_loaded, self._jaw_checkpoint_path)
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
    # Phishing Detector Management (Attack -> Beast -> Warhammer)
    # -----------------------------------------------------------------------

    def get_phishing_detector(self) -> PhishingDetector:
        """Return the phishing detector (lazy loaded).

        Attack/Beast are unfit until a checkpoint is loaded; in mock-fallback mode
        the service layer returns neutral is_mock results instead of calling them.
        """
        with self._lock:
            if self._phishing_detector is None:
                self._load_phishing(self.config.phishing_checkpoint_dir)
            self._require_weights(
                "phishing", self._phishing_weights_loaded, self._phishing_checkpoint_path
            )
            return self._phishing_detector

    def _load_phishing(self, checkpoint_dir: Optional[Union[str, Path]]) -> None:
        """Initialize the phishing pipeline; Warhammer is built only if warhammer.pt exists."""
        ckpt = Path(checkpoint_dir) if checkpoint_dir is not None else None
        has_warhammer = ckpt is not None and (ckpt / "warhammer.pt").exists()

        phishing_config = PhishingConfig(use_warhammer=has_warhammer)
        if self.config.phishing_decision_threshold is not None:
            phishing_config.decision_threshold = self.config.phishing_decision_threshold

        logger.info(f"Initializing PhishingDetector (warhammer={has_warhammer}) on {self._device}...")
        detector = PhishingDetector(phishing_config, device=self._device)

        weights_loaded = False
        if ckpt is not None:
            if (ckpt / "attack.joblib").exists() and (ckpt / "beast.json").exists():
                try:
                    detector.load(ckpt)
                    weights_loaded = True
                    logger.info("Phishing weights loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load phishing weights from {ckpt}: {e}")
            else:
                logger.info(
                    f"No complete phishing checkpoint at {ckpt} "
                    "(needs attack.joblib and beast.json)."
                )

        self._phishing_detector = detector
        self._phishing_weights_loaded = weights_loaded
        self._phishing_checkpoint_path = ckpt

    def reload_phishing(self, checkpoint_dir: Optional[Union[str, Path]] = None) -> bool:
        """Force reload of the phishing pipeline from a given or default directory."""
        with self._lock:
            path = checkpoint_dir or self.config.phishing_checkpoint_dir
            self._load_phishing(path)
            return self._phishing_weights_loaded

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

        ph = self._phishing_detector
        phishing_status = ModelStatus(
            model_id="phishing",
            name="Phishing (Attack -> Beast -> Warhammer escalation pipeline)",
            type="text_detector",
            is_loaded=ph is not None,
            weights_found=self._phishing_weights_loaded,
            checkpoint_path=str(self._phishing_checkpoint_path) if self._phishing_checkpoint_path else None,
            device=self._device,
            decision_threshold=(
                ph.config.decision_threshold
                if ph
                else (self.config.phishing_decision_threshold or 0.5)
            ),
            details={
                "stages": ["attack", "beast", "warhammer"],
                "attack": "char n-gram TF-IDF + logistic regression",
                "beast": "xgboost over attack score + indicators",
                "warhammer": "distilbert (only if warhammer.pt present)",
                "warhammer_ready": bool(ph.warhammer_ready) if ph else False,
            },
        )

        return [cart_status, jaw_status, phishing_status]


# Global singleton instance
registry = SafeStepModelRegistry()
