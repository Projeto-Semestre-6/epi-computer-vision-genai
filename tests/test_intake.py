import io

import cv2
import numpy as np
import pytest
from PIL import Image

from epi_imgproc import intake
from synthetic import encode_cv2, encode_pil, scene


def imdecode_color(data: bytes) -> np.ndarray:
    """What Ultralytics gets when it reads the same file (EXIF applied by OpenCV)."""
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def test_sniff_format_uses_magic_bytes():
    assert intake.sniff_format(encode_cv2(scene(), ".jpg")) == "jpeg"
    assert intake.sniff_format(encode_cv2(scene(), ".png")) == "png"
    assert intake.sniff_format(encode_cv2(scene(), ".bmp")) is None
    assert intake.sniff_format(b"GIF89a") is None


def test_png_round_trip_is_lossless():
    image = scene()
    decoded = intake.decode_image(encode_cv2(image, ".png"))
    assert decoded.source_format == "png"
    assert decoded.exif_orientation == 1
    assert np.array_equal(decoded.pixels, image)
    assert np.array_equal(imdecode_color(intake.encode_png(decoded.pixels)), image)


def test_jpeg_decode_matches_opencv():
    data = encode_cv2(scene(), ".jpg")
    decoded = intake.decode_image(data)
    assert decoded.source_format == "jpeg"
    assert np.array_equal(decoded.pixels, imdecode_color(data))


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_matches_opencv(fmt, orientation):
    image = scene(90, 160)
    data = encode_pil(image, fmt, orientation, **({"quality": 95} if fmt == "JPEG" else {}))
    decoded = intake.decode_image(data)
    assert decoded.exif_orientation == orientation
    assert decoded.stored_size == (160, 90)
    assert np.array_equal(decoded.pixels, imdecode_color(data))


@pytest.mark.parametrize("orientation", [5, 6, 7, 8])
def test_rotating_orientations_swap_width_and_height(orientation):
    decoded = intake.decode_image(encode_pil(scene(90, 160), "PNG", orientation))
    assert decoded.pixels.shape == (160, 90, 3)


def test_orientation_6_rotates_clockwise():
    image = scene(90, 160)
    decoded = intake.decode_image(encode_pil(image, "PNG", 6))
    assert np.array_equal(decoded.pixels, np.rot90(image, k=-1))


@pytest.mark.parametrize("value", [0, 9, 65535])
def test_out_of_range_orientation_is_ignored(value):
    image = scene()
    decoded = intake.decode_image(encode_pil(image, "PNG", value))
    assert decoded.exif_orientation == 1
    assert np.array_equal(decoded.pixels, image)


def test_grayscale_alpha_and_16bit_png_become_bgr_uint8():
    gray = cv2.cvtColor(scene(), cv2.COLOR_BGR2GRAY)
    rgba = cv2.cvtColor(scene(), cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = 0
    deep = scene().astype(np.uint16) * 257
    for array in (gray, rgba, deep):
        data = encode_cv2(array, ".png")
        decoded = intake.decode_image(data)
        assert decoded.pixels.dtype == np.uint8
        assert decoded.pixels.shape == (120, 160, 3)
        assert np.array_equal(decoded.pixels, imdecode_color(data))


def test_empty_input():
    with pytest.raises(intake.IntakeError) as error:
        intake.decode_image(b"")
    assert error.value.code == "empty"


@pytest.mark.parametrize("ext", [".bmp", ".webp", ".tiff"])
def test_unsupported_formats(ext):
    with pytest.raises(intake.IntakeError) as error:
        intake.decode_image(encode_cv2(scene(), ext))
    assert error.value.code == "unsupported_format"


def test_gif_is_unsupported():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="GIF")
    with pytest.raises(intake.IntakeError) as error:
        intake.decode_image(buffer.getvalue())
    assert error.value.code == "unsupported_format"


@pytest.mark.parametrize(
    "mangle",
    [
        lambda data: data[:100],
        lambda data: data[: len(data) // 2],
        lambda data: data[:8] + b"\x00" * 64,
    ],
    ids=["first-100-bytes", "half", "magic-only"],
)
@pytest.mark.parametrize("ext", [".jpg", ".png"])
def test_truncated_or_garbage_after_magic_is_corrupt(mangle, ext):
    data = mangle(encode_cv2(scene(), ext))
    with pytest.raises(intake.IntakeError) as error:
        intake.decode_image(data)
    assert error.value.code == "corrupt"


def test_png_with_damaged_pixel_data_is_corrupt():
    data = bytearray(encode_cv2(scene(), ".png"))
    start = data.index(b"IDAT") + 20
    data[start : start + 40] = bytes(40)
    with pytest.raises(intake.IntakeError) as error:
        intake.decode_image(bytes(data))
    assert error.value.code == "corrupt"


def test_size_limit_boundary():
    limit = 1000
    at_limit = intake.PNG_MAGIC + b"\x00" * (limit - len(intake.PNG_MAGIC))
    with pytest.raises(intake.IntakeError) as error:
        intake.decode_image(at_limit, max_bytes=limit)
    assert error.value.code == "corrupt"  # exactly at the limit is not "too large"
    with pytest.raises(intake.IntakeError) as error:
        intake.decode_image(at_limit + b"\x00", max_bytes=limit)
    assert error.value.code == "too_large"


def test_default_limit_is_10_mib():
    assert intake.MAX_UPLOAD_BYTES == 10 * 1024 * 1024


def test_pixel_limit():
    data = encode_cv2(scene(120, 160), ".png")
    with pytest.raises(intake.IntakeError) as error:
        intake.decode_image(data, max_pixels=120 * 160 - 1)
    assert error.value.code == "too_many_pixels"
    assert intake.decode_image(data, max_pixels=120 * 160).pixels.shape == (120, 160, 3)


def test_encoded_png_has_no_exif():
    decoded = intake.decode_image(encode_pil(scene(), "JPEG", 6))
    png = intake.encode_png(decoded.pixels)
    assert b"eXIf" not in png and b"Exif" not in png
    assert Image.open(io.BytesIO(png)).getexif().get(0x0112) is None


XMP_ORIENTATION_6 = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="6"/></rdf:RDF></x:xmpmeta>'
)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_orientation_only_in_xmp_is_ignored(fmt):
    """Review finding 3: Pillow's getexif() falls back to XMP; OpenCV and the contract use EXIF 0x0112 only."""
    from PIL.PngImagePlugin import PngInfo

    image = scene(90, 160)
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    if fmt == "JPEG":
        pil.save(buffer, format="JPEG", quality=95, xmp=XMP_ORIENTATION_6.encode())
    else:
        info = PngInfo()
        info.add_itxt("XML:com.adobe.xmp", XMP_ORIENTATION_6)
        pil.save(buffer, format="PNG", pnginfo=info)
    data = buffer.getvalue()
    decoded = intake.decode_image(data)
    assert decoded.exif_orientation == 1
    assert np.array_equal(decoded.pixels, imdecode_color(data))
