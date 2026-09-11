"""
Shared logging configuration for the util package.

Import get_logger(__name__) from any module instead of calling
logging.basicConfig() repeatedly.
"""

import logging

_CONFIGURED = False


def _configure_once() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with shared formatting."""
    _configure_once()
    return logging.getLogger(name)