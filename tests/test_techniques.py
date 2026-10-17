import cv2
import numpy as np
import pytest

from epi_imgproc import frequency, techniques
from synthetic import low_contrast, scene

ALL = list(techniques.TECHNIQUES)


def lightness(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float64)


@pytest.mark.parametrize("name", ALL)
def test_shape_dtype_purity_and_determinism(name):
    image = scene()
    original = image.copy()
    first = techniques.apply_technique(name, image)
    second = techniques.apply_technique(name, image)
    assert first.shape == image.shape and first.dtype == np.uint8
    assert first is not image
    assert np.array_equal(image, original)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (3, 2), (7, 5)])
def test_tiny_images_do_not_crash(name, shape):
    image = np.full((*shape, 3), 90, np.uint8)
    assert techniques.apply_technique(name, image).shape == image.shape


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("bad", [np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.float32), np.zeros((4, 4, 4), np.uint8)])
def test_rejects_non_bgr_uint8(name, bad):
    with pytest.raises(ValueError):
        techniques.apply_technique(name, bad)


def test_unknown_technique():
    with pytest.raises(ValueError):
        techniques.apply_technique("sepia", scene())


def test_none_is_identity():
    image = scene()
    assert np.array_equal(techniques.none(image), image)


def test_gamma():
    image = scene()
    assert np.array_equal(techniques.gamma_correction(image, gamma=1.0), image)
    assert techniques.gamma_correction(image, gamma=0.8).mean() > image.mean()
    assert techniques.gamma_correction(image, gamma=1.25).mean() < image.mean()
    with pytest.raises(ValueError):
        techniques.gamma_correction(image, gamma=0)


def test_histogram_equalization_and_clahe_raise_contrast():
    image = low_contrast()
    base = lightness(image).std()
    assert lightness(techniques.histogram_equalization(image)).std() > 2 * base
    assert lightness(techniques.clahe(image)).std() > base


def test_lightness_techniques_keep_hue():
    """Contrast techniques act on L only, so a flat color stays the same color."""
    image = np.full((32, 32, 3), (0, 140, 255), np.uint8)  # vest orange
    for name in ("unsharp_mask", "fft_lowpass", "fft_high_emphasis"):
        assert np.abs(techniques.apply_technique(name, image).astype(int) - image).max() <= 1


def test_unsharp_with_zero_amount_is_near_identity():
    image = scene()
    out = techniques.unsharp_mask(image, amount=0.0)
    assert np.abs(out.astype(int) - image).max() <= 1


@pytest.mark.parametrize(
    "name, direction",
    [("gaussian_blur", -1), ("fft_lowpass", -1), ("unsharp_mask", +1), ("fft_high_emphasis", +1)],
)
def test_sharpness_moves_in_the_expected_direction(name, direction):
    image = scene(240, 320)
    before = frequency.high_frequency_energy_ratio(image)
    after = frequency.high_frequency_energy_ratio(techniques.apply_technique(name, image))
    assert (after - before) * direction > 0


def test_neighborhood_parameters_scale_with_resolution():
    """sigma is in model-scale pixels: a 1280 px image gets twice the native sigma."""
    image = cv2.resize(scene(), (1280, 960))
    assert frequency.model_scale(image) == 2.0
    expected = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0, sigmaY=2.0)
    assert np.array_equal(techniques.gaussian_blur(image, sigma=1.0), expected)


def test_fft_lowpass_matches_spatial_blur_of_lightness():
    image = scene(480, 640)  # model scale 1: sigma 1.6 px, where both paths agree
    cutoff = 0.1
    sigma = 1.0 / (2.0 * np.pi * cutoff) * frequency.model_scale(image)
    in_frequency = techniques.fft_lowpass(image, cutoff=cutoff)
    in_space = techniques._apply_to_lightness(image, lambda l: cv2.GaussianBlur(l, (0, 0), sigmaX=sigma, sigmaY=sigma))
    difference = np.abs(in_frequency.astype(int) - in_space)
    assert difference.mean() < 0.05 and difference.max() <= 1


@pytest.mark.parametrize("name", ["fft_lowpass", "fft_high_emphasis"])
def test_frequency_padding_of_short_axis_is_bounded(name, monkeypatch):
    """Review finding 2: the pad comes from the long side; applied to a short axis it made a
    40 x 1,000,000 px image (within the 40 MP limit) allocate 60-150 GiB and return 500."""
    shapes = []
    real_rfft2 = np.fft.rfft2

    def recording_rfft2(array, *args, **kwargs):
        shapes.append(array.shape)
        return real_rfft2(array, *args, **kwargs)

    monkeypatch.setattr(np.fft, "rfft2", recording_rfft2)
    image = np.full((8, 20000, 3), 120, np.uint8)
    image[:, ::50] = 30
    assert techniques.apply_technique(name, image).shape == image.shape
    rows, cols = shapes[0]
    assert rows <= 2 * image.shape[0]
    assert cols < image.shape[1] + 2 * 1000
