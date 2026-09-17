"""
Frequency-domain feature extraction for JAW.

AI image generators (GANs, diffusion models) leave characteristic
fingerprints in the frequency domain — spectral energy concentrated
in specific radial bands, unusual spectral flatness, and distinct
texture responses — that are often invisible or subtle in the raw
pixel (spatial) domain. This module computes those features so they
can be fused with the CNN's spatial features.
"""

from typing import List

import numpy as np
from scipy.fft import fft2, fftshift
from skimage.filters import gabor_kernel
from scipy.signal import convolve2d

from jaw.config import JAWConfig


class FrequencyFeatureExtractor:
    """Computes FFT spectral + Gabor texture features from a grayscale image."""

    def __init__(self, config: JAWConfig):
        self.config = config
        self._gabor_kernels = self._build_gabor_kernels()

    @property
    def feature_dim(self) -> int:
        # radial energy bands + peak frequency location + spectral flatness
        # + one mean response per gabor kernel
        return self.config.num_radial_bands + 2 + len(self._gabor_kernels)

    def _build_gabor_kernels(self) -> List[np.ndarray]:
        kernels = []
        for scale in range(self.config.num_gabor_scales):
            frequency = 0.1 + 0.15 * scale  # spans low -> high frequency
            for o in range(self.config.num_gabor_orientations):
                theta = o * np.pi / self.config.num_gabor_orientations
                kernel = np.real(gabor_kernel(frequency, theta=theta))
                kernels.append(kernel)
        return kernels

    def extract(self, image_array: np.ndarray) -> np.ndarray:
        """
        Compute frequency-domain features for a single image.

        Args:
            image_array: (H, W) or (H, W, 3) array, values in [0, 255] or [0, 1].

        Returns:
            1D float32 array of length self.feature_dim.
        """
        gray = self._to_grayscale(image_array)

        radial_energy = self._radial_spectral_energy(gray)
        peak_freq, flatness = self._spectral_shape(gray)
        gabor_responses = self._gabor_responses(gray)

        return np.concatenate([
            radial_energy,
            [peak_freq, flatness],
            gabor_responses,
        ]).astype(np.float32)

    def extract_batch(self, image_arrays: List[np.ndarray]) -> np.ndarray:
        """Vectorize a batch of images into a (N, feature_dim) array."""
        return np.stack([self.extract(img) for img in image_arrays], axis=0)

    # -- helpers ----------------------------------------------------

    def _to_grayscale(self, image_array: np.ndarray) -> np.ndarray:
        arr = image_array.astype(np.float64)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 3:
            # standard luminance weighting
            arr = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
        return arr

    def _radial_spectral_energy(self, gray: np.ndarray) -> np.ndarray:
        """Energy distribution across radial frequency bands (low -> high freq)."""
        spectrum = np.abs(fftshift(fft2(gray)))
        h, w = spectrum.shape
        cy, cx = h // 2, w // 2

        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        max_radius = radius.max()

        n_bands = self.config.num_radial_bands
        band_edges = np.linspace(0, max_radius, n_bands + 1)

        energies = np.zeros(n_bands, dtype=np.float64)
        for i in range(n_bands):
            mask = (radius >= band_edges[i]) & (radius < band_edges[i + 1])
            energies[i] = spectrum[mask].sum() if mask.any() else 0.0

        total = energies.sum()
        if total > 0:
            energies = energies / total
        return energies

    def _spectral_shape(self, gray: np.ndarray):
        """Peak frequency location (normalized radius) and spectral flatness (entropy-based)."""
        spectrum = np.abs(fftshift(fft2(gray)))
        h, w = spectrum.shape
        cy, cx = h // 2, w // 2

        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        max_radius = radius.max()

        flat_spectrum = spectrum.flatten()
        flat_radius = radius.flatten()

        if flat_spectrum.sum() == 0:
            return 0.0, 0.0

        peak_idx = np.argmax(flat_spectrum)
        peak_freq = float(flat_radius[peak_idx] / max_radius) if max_radius > 0 else 0.0

        # Spectral flatness: geometric mean / arithmetic mean of the power spectrum
        power = flat_spectrum ** 2 + 1e-12
        geo_mean = np.exp(np.mean(np.log(power)))
        arith_mean = np.mean(power)
        flatness = float(geo_mean / arith_mean) if arith_mean > 0 else 0.0

        return peak_freq, flatness

    def _gabor_responses(self, gray: np.ndarray) -> np.ndarray:
        """Mean absolute response for each Gabor kernel in the bank."""
        # Downsample for speed if the image is large — texture response
        # doesn't need full resolution.
        if gray.shape[0] > 128:
            step = gray.shape[0] // 128
            gray = gray[::step, ::step]

        responses = []
        for kernel in self._gabor_kernels:
            filtered = convolve2d(gray, kernel, mode="same", boundary="symm")
            responses.append(float(np.mean(np.abs(filtered))))
        return np.array(responses, dtype=np.float64)