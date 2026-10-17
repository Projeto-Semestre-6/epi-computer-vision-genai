"""Enhancement techniques compared in the A/B evaluation.

Every technique is a pure function: BGR uint8 (H, W, 3) in, a new BGR uint8 array of the same
shape out; the input is never modified. Contrast and sharpness techniques act on the L channel of
CIELAB so colors (a cue for vests) are not shifted. Neighborhood parameters (sigmas, frequency
cutoffs) are defined at model scale, i.e. after the downstream model resizes the longest side to
640 px, and are rescaled to the native resolution, so a 4000 px photo gets the same effect the
A/B measured on smaller validation images. None of these is applied unless a client asks for it
or the A/B decision rule promotes it to default.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from epi_imgproc import frequency


def _check_image(image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected BGR uint8 image with shape (H, W, 3), got {image.dtype} {image.shape}")


def _apply_to_lightness(image: np.ndarray, operation: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Run ``operation`` on CIELAB L (float, 0..100) and convert back to BGR uint8."""
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    lab[..., 0] = np.clip(operation(lab[..., 0]), 0.0, 100.0)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.clip(np.rint(bgr * 255.0), 0, 255).astype(np.uint8)


def _lightness_to_uint8(lightness: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(lightness * 2.55), 0, 255).astype(np.uint8)


def none(image: np.ndarray) -> np.ndarray:
    _check_image(image)
    return image.copy()


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Global histogram equalization of the lightness channel."""
    _check_image(image)
    return _apply_to_lightness(image, lambda lightness: cv2.equalizeHist(_lightness_to_uint8(lightness)) / 2.55)


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on lightness (tile grid is relative to image size)."""
    _check_image(image)
    equalizer = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return _apply_to_lightness(image, lambda lightness: equalizer.apply(_lightness_to_uint8(lightness)) / 2.55)


def gamma_correction(image: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """Power-law transform s = 255 * (r / 255) ** gamma on each channel; gamma < 1 brightens shadows."""
    _check_image(image)
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    table = np.clip(np.rint(255.0 * (np.arange(256) / 255.0) ** gamma), 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Spatial Gaussian smoothing; ``sigma`` in model-scale pixels."""
    _check_image(image)
    native_sigma = sigma * frequency.model_scale(image)
    return cv2.GaussianBlur(image, (0, 0), sigmaX=native_sigma, sigmaY=native_sigma)


def unsharp_mask(image: np.ndarray, sigma: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """g = f + amount * (f - blur(f)) on lightness; ``sigma`` in model-scale pixels."""
    _check_image(image)
    native_sigma = sigma * frequency.model_scale(image)

    def sharpen(lightness: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(lightness, (0, 0), sigmaX=native_sigma, sigmaY=native_sigma)
        return lightness + amount * (lightness - blurred)

    return _apply_to_lightness(image, sharpen)


def fft_lowpass(image: np.ndarray, cutoff: float = 0.25) -> np.ndarray:
    """Gaussian low-pass in the frequency domain on lightness; ``cutoff`` in cycles/pixel at model scale."""
    _check_image(image)
    native_cutoff = cutoff / frequency.model_scale(image)

    def smooth(lightness: np.ndarray) -> np.ndarray:
        return frequency.filter_channel(
            lightness,
            lambda distance: frequency.gaussian_lowpass(distance, native_cutoff),
            pad=frequency.gaussian_pad(native_cutoff),
        )

    return _apply_to_lightness(image, smooth)


def fft_high_emphasis(image: np.ndarray, cutoff: float = 0.1, k1: float = 1.0, k2: float = 0.5) -> np.ndarray:
    """High-frequency emphasis H = k1 + k2 * (1 - Gaussian low-pass) on lightness; frequency-domain sharpening."""
    _check_image(image)
    native_cutoff = cutoff / frequency.model_scale(image)

    def emphasize(lightness: np.ndarray) -> np.ndarray:
        return frequency.filter_channel(
            lightness,
            lambda distance: frequency.high_frequency_emphasis(distance, native_cutoff, k1, k2),
            pad=frequency.gaussian_pad(native_cutoff),
        )

    return _apply_to_lightness(image, emphasize)


@dataclass(frozen=True)
class Technique:
    name: str
    function: Callable[..., np.ndarray]
    params: dict = field(default_factory=dict)
    domain: str = "spatial"

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.function(image, **self.params)


TECHNIQUES: dict[str, Technique] = {
    t.name: t
    for t in (
        Technique("none", none),
        Technique("histogram_equalization", histogram_equalization),
        Technique("clahe", clahe, {"clip_limit": 2.0, "tile_grid": (8, 8)}),
        Technique("gamma", gamma_correction, {"gamma": 0.8}),
        Technique("gaussian_blur", gaussian_blur, {"sigma": 1.0}),
        Technique("unsharp_mask", unsharp_mask, {"sigma": 1.0, "amount": 1.0}),
        Technique("fft_lowpass", fft_lowpass, {"cutoff": 0.25}, domain="frequency"),
        Technique("fft_high_emphasis", fft_high_emphasis, {"cutoff": 0.1, "k1": 1.0, "k2": 0.5}, domain="frequency"),
    )
}

DEFAULT_TECHNIQUE = "none"


def apply_technique(name: str, image: np.ndarray) -> np.ndarray:
    try:
        technique = TECHNIQUES[name]
    except KeyError:
        raise ValueError(f"unknown technique: {name!r}") from None
    return technique(image)
