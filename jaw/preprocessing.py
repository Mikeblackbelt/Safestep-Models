"""
Image preprocessing for JAW: load -> resize -> normalize, producing
both the CNN-ready tensor and the raw array used for frequency
feature extraction.
"""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image

from jaw.config import JAWConfig

# Defense against pixel flood / decompression bomb attacks
MAX_IMAGE_PIXELS = 25_000_000  # 25 Megapixels


class JAWPreprocessor:
    """Loads images and prepares both spatial (tensor) and raw (array) forms."""

    def __init__(self, config: JAWConfig):
        self.config = config
        self.mean = torch.tensor(config.norm_mean).view(3, 1, 1)
        self.std = torch.tensor(config.norm_std).view(3, 1, 1)

    def load(self, path: Union[str, Path]) -> Image.Image:
        img = Image.open(path)
        if img.width * img.height > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Image dimensions ({img.width}x{img.height}) exceed maximum allowed pixels ({MAX_IMAGE_PIXELS})."
            )
        return img.convert("RGB")

    def to_model_inputs(self, images: List[Image.Image]) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """
        Args:
            images: list of PIL images (any size).

        Returns:
            (image_tensor, raw_arrays)
                image_tensor: (B, 3, H, W) normalized, resized to config.image_size.
                raw_arrays: list of (H, W, 3) uint8 arrays at the resized resolution,
                            used for frequency-domain feature extraction.
        """
        size = self.config.image_size
        tensors = []
        raw_arrays = []

        for img in images:
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image dimensions ({img.width}x{img.height}) exceed maximum allowed pixels ({MAX_IMAGE_PIXELS})."
                )
            resized = img.resize((size, size))
            raw_arrays.append(np.array(resized))

            tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255.0
            tensor = (tensor - self.mean) / self.std
            tensors.append(tensor)

        return torch.stack(tensors, dim=0), raw_arrays

    def load_and_prepare(self, paths: List[Union[str, Path]]) -> Tuple[torch.Tensor, List[np.ndarray]]:
        images = [self.load(p) for p in paths]
        return self.to_model_inputs(images)