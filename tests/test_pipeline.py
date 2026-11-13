import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from epi_imgproc import evaluation, frequency, intake, pipeline, techniques
from epi_imgproc.api import app
from phase2_data import photo
from synthetic import encode_cv2, encode_pil, scene


def decode(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


@pytest.mark.parametrize("name", list(techniques.TECHNIQUES))
def test_process_image_is_decode_then_technique_then_png(name):
    data = encode_cv2(scene(), ".jpg")
    decoded = intake.decode_image(data)
    result = pipeline.process_image(data, name)
    assert result.content == intake.encode_png(techniques.apply_technique(name, decoded.pixels))
    assert (result.output_format, result.media_type, result.technique) == ("png", "image/png", name)
    assert (result.width, result.height) == (160, 120)
    assert result.source_format == "jpeg" and result.exif_orientation == 1
    assert result.sharpness == frequency.high_frequency_energy_ratio(decoded.pixels)


def test_default_technique_is_none_and_lossless():
    image = scene()
    result = pipeline.process_image(encode_cv2(image, ".png"))
    assert result.technique == "none"
    assert np.array_equal(decode(result.content), image)


@pytest.mark.parametrize("name", list(techniques.TECHNIQUES))
def test_jpeg_output_keeps_the_pixel_grid(name):
    data = encode_cv2(scene(90, 150), ".jpg")
    result = pipeline.process_image(data, name, output_format="jpeg", measure_sharpness=False)
    assert result.content.startswith(intake.JPEG_MAGIC)
    assert result.media_type == "image/jpeg" and result.sharpness is None
    assert decode(result.content).shape == (90, 150, 3)
    expected = techniques.apply_technique(name, intake.decode_image(data).pixels)
    assert result.content == intake.encode_jpeg(expected)


@pytest.mark.parametrize("name", list(techniques.TECHNIQUES))
def test_jpeg_output_stays_close_to_the_treated_pixels(name):
    """Quality 95 on a smooth picture: the mean error is under 2 gray levels (PNG would be 0)."""
    data = encode_cv2(photo(3), ".png")
    expected = techniques.apply_technique(name, intake.decode_image(data).pixels)
    written = decode(pipeline.process_image(data, name, output_format="jpeg").content)
    assert 0 < np.abs(written.astype(int) - expected.astype(int)).mean() < 2.0


def test_jpeg_encoding_parameters_are_the_documented_ones():
    image = scene()
    data = intake.encode_jpeg(image)
    with Image.open(io.BytesIO(data)) as written:
        with Image.open(io.BytesIO(encode_pil(image, "JPEG", quality=95, subsampling="4:2:0"))) as reference:
            assert written.quantization == reference.quantization  # libjpeg tables of quality 95
        assert [component[1:3] for component in written.layer] == [(2, 2), (1, 1), (1, 1)]  # 4:2:0
        assert not written.info.get("progressive") and "exif" not in written.info and "icc_profile" not in written.info
    assert data == intake.encode_jpeg(image, intake.JPEG_QUALITY) == intake.encode_jpeg(image.copy())  # deterministic
    described = intake.jpeg_encoding()
    assert (described["quality"], described["chroma_subsampling"], described["progressive"]) == (95, "4:2:0", False)
    assert len(intake.encode_jpeg(image, 50)) < len(data)
    with pytest.raises(ValueError):
        intake.encode_jpeg(image, 0)


def test_pixel_limit_is_a_parameter_and_the_service_default_is_unchanged():
    data = encode_cv2(scene(120, 160), ".jpg")  # 19,200 pixels
    assert intake.MAX_PIXELS == 40_000_000 and intake.MAX_UPLOAD_BYTES == 10 * 1024 * 1024
    with pytest.raises(intake.IntakeError) as rejected:
        pipeline.process_image(data, max_pixels=19_199)
    assert rejected.value.code == "too_many_pixels"
    assert pipeline.process_image(data, max_pixels=19_200).width == 160
    with pytest.raises(intake.IntakeError) as too_large:
        pipeline.process_image(data, max_bytes=100)
    assert too_large.value.code == "too_large"


def flat_png(width: int, height: int) -> bytes:
    """A valid grayscale PNG of one color: tens of kilobytes on disk, whatever the pixel count."""
    return encode_cv2(np.zeros((height, width), np.uint8), ".png")


def test_the_service_refuses_a_photo_just_over_40_megapixels():
    """8000x5001 is 8,000 pixels over the service limit and well under the batch limit (50 MP)."""
    data = flat_png(8000, 5001)
    assert len(data) < intake.MAX_UPLOAD_BYTES  # refused for its pixels, not for its bytes
    with pytest.raises(intake.IntakeError) as rejected:
        pipeline.process_image(data)  # the defaults of the function are the limits of the service
    assert rejected.value.code == "too_many_pixels"
    response = TestClient(app).post("/process", files={"file": ("grande.png", data, "image/png")})
    assert response.status_code == 413
    assert response.json() == {
        "code": "too_many_pixels",
        "detail": "Imagem de 8000x5001 pixels excede o limite de 40000000 pixels.",
    }


def test_exactly_40_megapixels_is_within_the_service_limit():
    """8000x5000 passes the pixel check: cut in half, the file is refused as corrupt, not for its pixels.

    (A whole photo of that size takes hundreds of megabytes to treat, too much for a unit test.)
    """
    data = flat_png(8000, 5000)
    with pytest.raises(intake.IntakeError) as rejected:
        pipeline.process_image(data[: len(data) // 2])
    assert rejected.value.code == "corrupt"
    response = TestClient(app).post("/process", files={"file": ("limite.png", data[: len(data) // 2], "image/png")})
    assert (response.status_code, response.json()["code"]) == (400, "corrupt")


def test_exif_orientation_is_applied_before_the_technique():
    result = pipeline.process_image(encode_pil(scene(90, 160), "JPEG", 6, quality=95), "clahe", output_format="jpeg")
    assert result.exif_orientation == 6
    assert (result.width, result.height) == (90, 160)
    with Image.open(io.BytesIO(result.content)) as written:
        assert written.size == (90, 160) and "exif" not in written.info


def test_unknown_technique_and_format_are_rejected_before_decoding():
    with pytest.raises(ValueError, match="unknown technique"):
        pipeline.process_image(b"not an image", "sepia")
    with pytest.raises(ValueError, match="unknown output format"):
        pipeline.process_image(encode_cv2(scene(), ".png"), output_format="webp")
    with pytest.raises(intake.IntakeError):
        pipeline.process_image(b"not an image")


@pytest.fixture
def calls(monkeypatch):
    """Record every call of the one treatment function, still running it."""
    recorded = []
    original = pipeline.process_image

    def spy(*args, **kwargs):
        recorded.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "process_image", spy)
    return recorded


def test_the_endpoint_goes_through_process_image_with_the_service_defaults(calls):
    data = encode_cv2(scene(), ".jpg")
    response = TestClient(app).post("/process", files={"file": ("foto.jpg", data, "image/jpeg")}, params={"technique": "clahe"})
    assert response.status_code == 200
    assert calls == [((data, "clahe"), {})]  # PNG, 10 MiB, 40 MP, sharpness measured: nothing overridden
    assert response.content == pipeline.process_image(data, "clahe").content
    assert response.headers["content-type"] == "image/png"


def test_the_ab_evaluation_goes_through_process_image(calls, tmp_path):
    source = tmp_path / "valid" / "images" / "a.jpg"
    source.parent.mkdir(parents=True)
    (tmp_path / "valid" / "labels").mkdir()
    data = encode_cv2(scene(), ".jpg")
    source.write_bytes(data)
    (tmp_path / "valid" / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    prepared = evaluation.prepare_variant("gamma", [source], {0: 0}, {0: "helmet"}, tmp_path / "out")
    assert prepared.images == 1 and prepared.sharpness == {}
    assert calls == [((data, "gamma"), {"measure_sharpness": False})]
    assert (tmp_path / "out" / "gamma" / "images" / "val" / "a.png").read_bytes() == pipeline.process_image(data, "gamma").content
