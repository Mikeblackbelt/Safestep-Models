"""
Configuration for SafeStep Model API.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Optional


@dataclass
class APIConfig:
    """Server and model loading configuration."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # Device configuration: 'auto', 'cpu', 'cuda', etc.
    device: str = "auto"

    # Default checkpoint search directories
    cart_checkpoint_dir: Optional[Path] = None
    jaw_checkpoint_dir: Optional[Path] = None

    # Threshold overrides (None means use model config defaults, typically 0.5)
    cart_decision_threshold: Optional[float] = None
    jaw_decision_threshold: Optional[float] = None

    # Execution behavior
    lazy_load: bool = True
    enable_mock_fallback: bool = True
    max_batch_size: int = 128
    max_upload_size_bytes: int = 25 * 1024 * 1024  # 25 MB

    # Shared secret required in the X-Admin-Token header for model reloads
    admin_token: Optional[str] = None

    def __post_init__(self):
        # Resolve environment variable overrides
        if os.getenv("SAFESTEP_API_HOST"):
            self.host = os.environ["SAFESTEP_API_HOST"]
        if os.getenv("SAFESTEP_API_PORT"):
            self.port = int(os.environ["SAFESTEP_API_PORT"])
        if os.getenv("SAFESTEP_DEVICE"):
            self.device = os.environ["SAFESTEP_DEVICE"]

        cart_env = os.getenv("SAFESTEP_CART_CHECKPOINT")
        if cart_env:
            self.cart_checkpoint_dir = Path(cart_env)
        elif self.cart_checkpoint_dir is None:
            # Check default locations
            default_path = Path("checkpoints/cart")
            self.cart_checkpoint_dir = default_path

        jaw_env = os.getenv("SAFESTEP_JAW_CHECKPOINT")
        if jaw_env:
            self.jaw_checkpoint_dir = Path(jaw_env)
        elif self.jaw_checkpoint_dir is None:
            default_path = Path("checkpoints/jaw")
            self.jaw_checkpoint_dir = default_path

        cors_env = os.getenv("SAFESTEP_CORS_ORIGINS")
        if cors_env is not None:
            self.cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
        mock_env = os.getenv("SAFESTEP_MOCK_FALLBACK")
        if mock_env is not None:
            self.enable_mock_fallback = mock_env.strip().lower() in ("1", "true", "yes")
        # Gates POST /models/{id}/reload; unset means reload is open (dev only).
        self.admin_token = os.getenv("SAFESTEP_ADMIN_TOKEN") or None

        if os.getenv("SAFESTEP_CART_THRESHOLD"):
            self.cart_decision_threshold = float(os.environ["SAFESTEP_CART_THRESHOLD"])
        if os.getenv("SAFESTEP_JAW_THRESHOLD"):
            self.jaw_decision_threshold = float(os.environ["SAFESTEP_JAW_THRESHOLD"])


default_api_config = APIConfig()
