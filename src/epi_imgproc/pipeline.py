"""The treatment path of the service, in one function: decode -> technique -> encode.

The REST endpoint, the A/B evaluation and the phase-2 dataset batch all call `process_image`, so a
photo treated by the batch goes through exactly the steps the endpoint applies. Callers differ only
in the arguments: the endpoint answers PNG under the upload limits, the batch writes JPEG and may
raise the pixel limit.
"""

from __future__ import annotations

from dataclasses import dataclass

from epi_imgproc import frequency, intake, techniques

MEDIA_TYPES = {"png": "image/png", "jpeg": "image/jpeg"}


@dataclass(frozen=True)
class ProcessedImage:
    content: bytes  # the encoded output image
    output_format: str  # "png" or "jpeg"
    technique: str
    width: int  # of the output, which is the size of the upright input
    height: int
    source_format: str  # "jpeg" or "png"
    exif_orientation: int  # tag value that was applied to the input, 1..8
    sharpness: float | None  # high-frequency energy ratio of the input, before the technique

    @property
    def media_type(self) -> str:
        return MEDIA_TYPES[self.output_format]


def process_image(
    data: bytes,
    technique: str = techniques.DEFAULT_TECHNIQUE,
    *,
    output_format: str = "png",
    jpeg_quality: int = intake.JPEG_QUALITY,
    max_bytes: int = intake.MAX_UPLOAD_BYTES,
    max_pixels: int = intake.MAX_PIXELS,
    measure_sharpness: bool = True,
) -> ProcessedImage:
    """Decode image bytes, apply one registered technique, and encode the result.

    Raises `intake.IntakeError` for input the intake rejects and `ValueError` for an unknown
    technique or output format. The output has the pixel grid of the upright input, so
    normalized boxes stay valid.
    """
    if technique not in techniques.TECHNIQUES:
        raise ValueError(f"unknown technique: {technique!r}")
    if output_format not in MEDIA_TYPES:
        raise ValueError(f"unknown output format: {output_format!r}")
    decoded = intake.decode_image(data, max_bytes=max_bytes, max_pixels=max_pixels)
    sharpness = frequency.high_frequency_energy_ratio(decoded.pixels) if measure_sharpness else None
    processed = techniques.apply_technique(technique, decoded.pixels)
    if output_format == "png":
        content = intake.encode_png(processed)
    else:
        content = intake.encode_jpeg(processed, jpeg_quality)
    height, width = processed.shape[:2]
    return ProcessedImage(
        content=content,
        output_format=output_format,
        technique=technique,
        width=width,
        height=height,
        source_format=decoded.source_format,
        exif_orientation=decoded.exif_orientation,
        sharpness=sharpness,
    )
