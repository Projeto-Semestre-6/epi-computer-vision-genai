"""Frequency-domain tools: spectrum display, Gaussian transfer functions, and a sharpness metric.

Method follows the course material (ignored/FFT.pdf): transform with a 2D FFT, multiply by a
transfer function H(u, v), and return to the spatial domain with the inverse FFT. Low
frequencies are smooth regions and illumination; high frequencies are edges, fine texture, and
noise. Frequencies here are in cycles per pixel (Nyquist = 0.5).
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

# The downstream YOLO model sees images letterboxed to this size (train_args.imgsz of best.pt).
MODEL_INPUT_SIZE = 640
SHARPNESS_CUTOFF = 0.25  # cycles/pixel at model scale: the upper half of the band up to Nyquist


def to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float64)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)


def radial_frequency(shape: tuple[int, int], real_input: bool = False) -> np.ndarray:
    """Distance D(u, v) from the zero frequency, in cycles/pixel, in unshifted FFT layout.

    With ``real_input`` the grid matches ``np.fft.rfft2`` output (half spectrum along columns).
    """
    rows, cols = shape
    fy = np.fft.fftfreq(rows)[:, None]
    fx = (np.fft.rfftfreq(cols) if real_input else np.fft.fftfreq(cols))[None, :]
    return np.sqrt(fx * fx + fy * fy)


def gaussian_lowpass(distance: np.ndarray, cutoff: float) -> np.ndarray:
    """H(u, v) = exp(-D^2 / (2 * D0^2)); D0 is the cutoff in cycles/pixel."""
    return np.exp(-(distance**2) / (2.0 * cutoff**2))


def high_frequency_emphasis(distance: np.ndarray, cutoff: float, k1: float, k2: float) -> np.ndarray:
    """H(u, v) = k1 + k2 * H_highpass(u, v), with a Gaussian high-pass (Gonzalez & Woods, 4.9)."""
    return k1 + k2 * (1.0 - gaussian_lowpass(distance, cutoff))


def _axis_padding(size: int, pad: int) -> tuple[int, int]:
    """Mirror padding (before, after) for one axis.

    When ``pad`` would reach past the opposite border, the axis is extended to exactly one period
    of its reflect-101 extension, 2 * (size - 1) samples. Circular convolution over one period
    equals linear convolution with the infinitely mirrored signal, and the axis at most doubles
    instead of growing with the long side of the image.
    """
    if pad < size - 1:
        return pad, pad
    return 0, max(size - 2, 0)


def filter_channel(
    channel: np.ndarray,
    transfer: Callable[[np.ndarray], np.ndarray],
    pad: int,
) -> np.ndarray:
    """Filter one 2D channel in the frequency domain: pad, FFT, multiply by H, inverse FFT, crop.

    Mirror padding (same rule as OpenCV's default BORDER_REFLECT_101) keeps the circular
    convolution implied by the DFT from mixing opposite borders (wraparound).
    """
    rows, cols = channel.shape
    top, bottom = _axis_padding(rows, pad)
    left, right = _axis_padding(cols, pad)
    data = channel.astype(np.float64)
    if top or bottom or left or right:
        data = cv2.copyMakeBorder(data, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    spectrum = np.fft.rfft2(data)
    spectrum *= transfer(radial_frequency(data.shape, real_input=True))
    filtered = np.fft.irfft2(spectrum, s=data.shape)
    return filtered[top : top + rows, left : left + cols]


def gaussian_pad(cutoff: float) -> int:
    """Padding that covers the spatial Gaussian equivalent to a frequency cutoff (sigma = 1/(2*pi*D0))."""
    sigma = 1.0 / (2.0 * np.pi * cutoff)
    return int(np.ceil(4.0 * sigma)) + 1


def model_scale(image: np.ndarray, reference_size: int = MODEL_INPUT_SIZE) -> float:
    """Native pixels per model pixel: the downstream model resizes the longest side to ``reference_size``."""
    return max(image.shape[:2]) / float(reference_size)


def resize_to_model_scale(gray: np.ndarray, reference_size: int = MODEL_INPUT_SIZE) -> np.ndarray:
    height, width = gray.shape[:2]
    scale = reference_size / float(max(height, width))
    if scale == 1.0:
        return gray
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(gray, size, interpolation=interpolation)


def _hann(size: int) -> np.ndarray:
    """1D Hann window; an axis of 4 samples or fewer keeps unit weights.

    With so few samples the window zeroes the image (np.hanning(2) is [0, 0]) or its own spectrum
    puts half or more of the energy at 0.25 cycle/pixel and above (np.hanning(3) is [0, 1, 0],
    np.hanning(4) is [0, 0.75, 0.75, 0]), so the ratio would no longer follow the image content.
    """
    return np.hanning(size) if size > 4 else np.ones(size)


def high_frequency_energy_ratio(
    image: np.ndarray,
    cutoff: float = SHARPNESS_CUTOFF,
    reference_size: int = MODEL_INPUT_SIZE,
) -> float:
    """Fraction of the spectral energy above ``cutoff`` (cycles/pixel), measured at model scale.

    Lower values mean less fine detail, i.e. a blurrier image as the model will see it. The
    image is resized so its longest side equals ``reference_size``; the mean is removed (no DC)
    and a 2D Hann window suppresses the border discontinuities that would otherwise leak energy
    into high frequencies (an axis of 4 px or less is not windowed; see `_hann`). Returns 0.0 for a constant image.
    """
    gray = resize_to_model_scale(to_gray_float(image), reference_size)
    gray = gray - gray.mean()
    if np.abs(gray).max() < 1e-6:  # constant image: only floating-point residue is left
        return 0.0
    window = np.outer(_hann(gray.shape[0]), _hann(gray.shape[1]))
    power = np.abs(np.fft.rfft2(gray * window)) ** 2
    distance = radial_frequency(gray.shape, real_input=True)
    # rfft2 stores half of the columns; interior columns stand for two conjugate bins.
    weights = np.full(power.shape[1], 2.0)
    weights[0] = 1.0
    if gray.shape[1] % 2 == 0:
        weights[-1] = 1.0
    power = power * weights[None, :]
    total = power[distance > 0].sum()
    if total <= 0:
        return 0.0
    return float(power[distance > cutoff].sum() / total)


def log_magnitude_spectrum(image: np.ndarray) -> np.ndarray:
    """Centered log-magnitude spectrum log(1 + |F(u, v)|), scaled to uint8 for display."""
    spectrum = np.fft.fftshift(np.fft.fft2(to_gray_float(image)))
    magnitude = np.log1p(np.abs(spectrum))
    span = magnitude.max() - magnitude.min()
    if span == 0:
        return np.zeros(magnitude.shape, dtype=np.uint8)
    return np.rint(255.0 * (magnitude - magnitude.min()) / span).astype(np.uint8)


def transfer_function_image(transfer: Callable[[np.ndarray], np.ndarray], size: int = 256) -> np.ndarray:
    """Centered picture of H(u, v) for slides, scaled to uint8 (black = 0, white = max)."""
    distance = np.fft.fftshift(radial_frequency((size, size)))
    values = transfer(distance)
    peak = values.max()
    if peak <= 0:
        return np.zeros((size, size), dtype=np.uint8)
    return np.rint(255.0 * values / peak).astype(np.uint8)
