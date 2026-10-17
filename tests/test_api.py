import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import REPO
from epi_imgproc import api, intake, techniques
from epi_imgproc.api import app
from synthetic import encode_cv2, encode_pil, scene

client = TestClient(app)


def post(data: bytes, filename: str = "foto.jpg", **params):
    return client.post("/process", files={"file": (filename, data, "application/octet-stream")}, params=params)


def png_pixels(response) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["default"] == "none"
    assert set(body["techniques"]) == set(techniques.TECHNIQUES)


def test_default_is_lossless_passthrough():
    image = scene()
    response = post(encode_cv2(image, ".png"), "foto.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-technique"] == "none"
    assert np.array_equal(png_pixels(response), image)


@pytest.mark.parametrize("name", list(techniques.TECHNIQUES))
def test_every_technique_returns_png_of_same_size(name):
    response = post(encode_cv2(scene(), ".jpg"), technique=name)
    assert response.status_code == 200
    assert response.content.startswith(intake.PNG_MAGIC)
    assert png_pixels(response).shape == (120, 160, 3)
    assert response.headers["x-technique"] == name
    assert response.headers["x-image-size"] == "160x120"
    assert 0.0 <= float(response.headers["x-sharpness-hf-ratio"]) <= 1.0


def test_sharpness_header_is_measured_before_the_technique():
    data = encode_cv2(scene(), ".png")
    plain = post(data, technique="none").headers["x-sharpness-hf-ratio"]
    blurred = post(data, technique="gaussian_blur").headers["x-sharpness-hf-ratio"]
    assert plain == blurred


def test_blurry_input_has_lower_sharpness_header():
    image = scene(240, 320)
    sharp = float(post(encode_cv2(image, ".png")).headers["x-sharpness-hf-ratio"])
    soft = float(post(encode_cv2(cv2.GaussianBlur(image, (0, 0), 2.0), ".png")).headers["x-sharpness-hf-ratio"])
    assert soft < sharp


def test_exif_orientation_is_applied_and_reported():
    image = scene(90, 160)
    response = post(encode_pil(image, "JPEG", 6, quality=95))
    assert response.status_code == 200
    assert response.headers["x-exif-orientation"] == "6"
    assert response.headers["x-image-size"] == "90x160"
    assert png_pixels(response).shape == (160, 90, 3)
    assert b"eXIf" not in response.content


def test_filename_and_content_type_are_ignored():
    response = client.post("/process", files={"file": ("foto.gif", encode_cv2(scene(), ".png"), "image/gif")})
    assert response.status_code == 200
    assert response.headers["x-source-format"] == "png"


@pytest.mark.parametrize(
    "data, status, code",
    [
        (b"", 400, "empty"),
        (b"GIF89a" + bytes(20), 415, "unsupported_format"),
        (encode_cv2(scene(), ".bmp"), 415, "unsupported_format"),
        (encode_cv2(scene(), ".jpg")[:300], 400, "corrupt"),
        (encode_cv2(scene(), ".png")[:300], 400, "corrupt"),
    ],
    ids=["empty", "gif", "bmp", "truncated-jpeg", "truncated-png"],
)
def test_rejections(data, status, code):
    response = post(data)
    assert response.status_code == status
    body = response.json()
    assert body["code"] == code
    assert body["detail"]


def test_file_over_10_mib_is_413():
    data = intake.PNG_MAGIC + bytes(intake.MAX_UPLOAD_BYTES)
    response = post(data)
    assert response.status_code == 413
    assert response.json()["code"] == "too_large"


def test_declared_body_over_limit_is_refused_early():
    response = client.post(
        "/process",
        content=b"x",
        headers={"content-length": str(intake.MAX_UPLOAD_BYTES * 2), "content-type": "multipart/form-data; boundary=b"},
    )
    assert response.status_code == 413


def test_unknown_technique_is_422():
    assert post(encode_cv2(scene(), ".png"), technique="sepia").status_code == 422


def test_missing_file_is_422():
    assert client.post("/process").status_code == 422


def test_api_does_not_import_ultralytics_or_torch():
    code = "import epi_imgproc.api, sys; print('ultralytics' in sys.modules, 'torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    assert result.stdout.split() == ["False", "False"]


def test_default_port_is_8001_everywhere_it_is_documented():
    assert api.DEFAULT_PORT == 8001
    for name, text in [
        ("api.py", api.__doc__),
        ("contrato-api.md", (REPO / "docs" / "contrato-api.md").read_text()),
        ("README.md", (REPO / "README.md").read_text()),
    ]:
        commands = re.findall(r"epi_imgproc\.api:app[^\n`]*--port (\d+)", text)
        assert commands and set(commands) == {str(api.DEFAULT_PORT)}, name
    contract = (REPO / "docs" / "contrato-api.md").read_text()
    assert set(re.findall(r"localhost:(\d+)/process", contract)) == {str(api.DEFAULT_PORT)}
