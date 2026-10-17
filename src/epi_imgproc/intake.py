"""Image intake: validate raw bytes, decode them, and apply the EXIF orientation policy.

Orientation policy: the EXIF Orientation tag (0x0112) is read and applied to the pixels, so the
output is upright the way viewers display the photo, and the output carries no EXIF at all. This
matches what OpenCV's ``imread``/``imdecode`` with ``IMREAD_COLOR`` do (and therefore what
Ultralytics does when it reads image files), but it is implemented explicitly so it does not
depend on the OpenCV version. A missing, unreadable, or out-of-range tag is treated as 1 (no-op).
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB
MAX_PIXELS = 40_000_000  # width * height; guards against decompression bombs

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
EXIF_ORIENTATION_TAG = 0x0112


class IntakeError(Exception):
    """Input rejected before processing. ``code`` is stable; ``message`` is for humans (pt-BR)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class DecodedImage:
    pixels: np.ndarray  # BGR, uint8, shape (H, W, 3), already oriented
    source_format: str  # "jpeg" or "png"
    exif_orientation: int  # tag value that was applied, 1..8
    stored_size: tuple[int, int]  # (width, height) of the raster as stored, before orientation


def sniff_format(data: bytes) -> str | None:
    """Identify the container by its magic bytes; file names and MIME types are not trusted."""
    if data.startswith(JPEG_MAGIC):
        return "jpeg"
    if data.startswith(PNG_MAGIC):
        return "png"
    return None


def read_exif_orientation(image: Image.Image) -> int:
    """Orientation from the raw EXIF block only.

    ``Image.getexif()`` is avoided on purpose: when EXIF has no Orientation it falls back to XMP
    ``tiff:Orientation``, which OpenCV (and so Ultralytics) ignores. Call after ``load()`` so a
    PNG eXIf chunk placed after IDAT is seen too.
    """
    raw = image.info.get("exif")
    if not raw:
        return 1
    try:
        exif = Image.Exif()
        exif.load(raw)
        value = exif.get(EXIF_ORIENTATION_TAG, 1)
    except Exception:  # malformed EXIF must not reject an otherwise valid image
        return 1
    if isinstance(value, int) and 1 <= value <= 8:
        return value
    return 1


def apply_exif_orientation(pixels: np.ndarray, orientation: int) -> np.ndarray:
    """Return a new array transformed per EXIF 2.3 Orientation (same mapping as Pillow's exif_transpose)."""
    if orientation == 2:
        return cv2.flip(pixels, 1)
    if orientation == 3:
        return cv2.rotate(pixels, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(pixels, 0)
    if orientation == 5:
        return cv2.transpose(pixels)
    if orientation == 6:
        return cv2.rotate(pixels, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.transpose(pixels), -1)
    if orientation == 8:
        return cv2.rotate(pixels, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return pixels.copy()


def decode_image(
    data: bytes,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_pixels: int = MAX_PIXELS,
) -> DecodedImage:
    """Validate and decode JPEG/PNG bytes into an upright BGR uint8 array.

    Alpha is dropped without compositing, grayscale is expanded to 3 channels, and 16-bit PNGs
    are reduced to 8 bits: the same conversions ``cv2.IMREAD_COLOR`` applies when Ultralytics
    reads a file.
    """
    if not data:
        raise IntakeError("empty", "Arquivo vazio: nenhum byte de imagem foi recebido.")
    if len(data) > max_bytes:
        raise IntakeError(
            "too_large",
            f"Arquivo com {len(data)} bytes excede o limite de {max_bytes} bytes (10 MiB).",
        )
    source_format = sniff_format(data)
    if source_format is None:
        raise IntakeError(
            "unsupported_format",
            "Formato não suportado: envie JPEG ou PNG (o formato é identificado pelo conteúdo, não pela extensão).",
        )

    corrupt = IntakeError(
        "corrupt", f"Imagem {source_format.upper()} corrompida ou truncada: não foi possível decodificar."
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                width, height = probe.size
                if width * height > max_pixels:
                    raise IntakeError(
                        "too_many_pixels",
                        f"Imagem de {width}x{height} pixels excede o limite de {max_pixels} pixels.",
                    )
                probe.load()  # full decode: catches truncated or damaged data that OpenCV would pad silently
                orientation = read_exif_orientation(probe)
    except IntakeError:
        raise
    except Image.DecompressionBombError as exc:
        raise IntakeError("too_many_pixels", f"Imagem excede o limite de pixels: {exc}") from exc
    except Exception as exc:
        raise corrupt from exc

    buffer = np.frombuffer(data, dtype=np.uint8)
    pixels = cv2.imdecode(buffer, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if pixels is None or pixels.shape[:2] != (height, width):
        raise corrupt
    oriented = apply_exif_orientation(pixels, orientation)
    return DecodedImage(
        pixels=np.ascontiguousarray(oriented),
        source_format=source_format,
        exif_orientation=orientation,
        stored_size=(width, height),
    )


def encode_png(pixels: np.ndarray) -> bytes:
    """Lossless PNG encoding; no metadata is written."""
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return encoded.tobytes()
