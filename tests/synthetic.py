"""Synthetic test images; no dataset images are ever stored in the repository."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image


def scene(height: int = 120, width: int = 160, seed: int = 0) -> np.ndarray:
    """Asymmetric BGR scene with sharp edges, so flips and rotations are distinguishable."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[: height // 2, : width // 2] = (255, 0, 0)
    image[: height // 2, width // 2 :] = (0, 255, 0)
    image[height // 2 :, : width // 3] = (0, 0, 255)
    image[height // 2 :, width // 3 :] = (40, 200, 220)
    cv2.putText(image, "EPI", (width // 8, height // 2 + height // 4), cv2.FONT_HERSHEY_SIMPLEX, height / 120, (255, 255, 255), 2)
    for x in range(0, width, 6):
        cv2.line(image, (x, 0), (x, height // 8), (0, 0, 0), 1)
    noise = rng.integers(-8, 9, image.shape)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def low_contrast(height: int = 120, width: int = 160) -> np.ndarray:
    gradient = np.linspace(100, 140, width, dtype=np.float32)[None, :].repeat(height, 0)
    return np.dstack([gradient, gradient * 0.95, gradient * 1.05]).astype(np.uint8)


def encode_cv2(image: np.ndarray, ext: str) -> bytes:
    ok, data = cv2.imencode(ext, image)
    assert ok
    return data.tobytes()


def encode_pil(image: np.ndarray, fmt: str, orientation: int | None = None, **save_args) -> bytes:
    """Encode a BGR array with Pillow, optionally tagging EXIF Orientation."""
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    if orientation is not None:
        exif = pil.getexif()
        exif[0x0112] = orientation
        save_args["exif"] = exif.tobytes()
    pil.save(buffer, format=fmt, **save_args)
    return buffer.getvalue()
