import cv2
import numpy as np
import pytest

from epi_imgproc import frequency
from synthetic import scene


def test_radial_frequency_layout():
    distance = frequency.radial_frequency((8, 8))
    assert distance[0, 0] == 0
    assert distance.max() == pytest.approx(np.sqrt(0.5**2 + 0.5**2))
    half = frequency.radial_frequency((8, 10), real_input=True)
    assert half.shape == (8, 6)


@pytest.mark.parametrize("cutoff", [0.05, 0.1])
def test_frequency_lowpass_equals_spatial_gaussian(cutoff):
    """Convolution theorem (course slide 19): the same Gaussian, two computational paths.

    Holds for sigma >= ~1.5 px. Below 1 px (cutoff 0.25 -> sigma 0.64) OpenCV's sampled kernel
    and the continuous transfer function differ by aliasing, a few gray levels on sharp edges.
    """
    channel = cv2.cvtColor(scene(160, 200), cv2.COLOR_BGR2GRAY).astype(np.float64)
    sigma = 1.0 / (2.0 * np.pi * cutoff)
    in_frequency = frequency.filter_channel(
        channel, lambda d: frequency.gaussian_lowpass(d, cutoff), pad=frequency.gaussian_pad(cutoff)
    )
    in_space = cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    assert np.abs(in_frequency - in_space).mean() < 0.5
    assert np.abs(in_frequency - in_space).max() < 3.0


def test_high_frequency_emphasis_keeps_dc_and_boosts_edges():
    distance = frequency.radial_frequency((64, 64))
    transfer = frequency.high_frequency_emphasis(distance, cutoff=0.1, k1=1.0, k2=0.5)
    assert transfer[0, 0] == pytest.approx(1.0)
    assert transfer.max() == pytest.approx(1.5, abs=0.01)


def test_sharpness_drops_with_blur():
    image = scene(240, 320)
    values = [frequency.high_frequency_energy_ratio(image)]
    for sigma in (0.8, 1.5, 3.0):
        values.append(frequency.high_frequency_energy_ratio(cv2.GaussianBlur(image, (0, 0), sigma)))
    assert all(a > b for a, b in zip(values, values[1:]))
    assert 0.0 <= values[-1] < values[0] <= 1.0


def test_sharpness_of_constant_image_is_zero():
    assert frequency.high_frequency_energy_ratio(np.full((50, 70, 3), 128, np.uint8)) == 0.0
    assert frequency.high_frequency_energy_ratio(np.full((1, 1, 3), 7, np.uint8)) == 0.0


@pytest.mark.parametrize("width, height", [(640, 2), (2, 640), (640, 3), (3, 640), (640, 4), (4, 640), (640, 5)])
def test_sharpness_of_a_very_short_side_follows_the_detail(width, height):
    """Codex review #3: np.hanning(2) is [0, 0], so a 640x2 image used to score 0 whatever its detail.
    With 3 or 4 rows the window itself put half or more of the energy at the cutoff and above (a smooth
    ramp scored 0.67 and 0.5); from 5 rows on, the Hann window is kept.

    Alternating black and white lines along the long side are pure Nyquist detail; a smooth ramp
    has almost none. The image keeps its size at model scale (longest side already 640).
    """
    along = np.arange(max(width, height))
    stripes = np.where(along % 2 == 0, 0, 255).astype(np.uint8)
    ramp = np.linspace(0, 255, along.size).astype(np.uint8)

    def image(line: np.ndarray) -> np.ndarray:
        rows = np.tile(line, (min(width, height), 1))
        gray = rows if height < width else rows.T
        return np.repeat(gray[:, :, None], 3, axis=2)

    assert image(stripes).shape == (height, width, 3)
    sharp = frequency.high_frequency_energy_ratio(image(stripes))
    smooth = frequency.high_frequency_energy_ratio(image(ramp))
    assert sharp > 0.9
    assert smooth < 0.05


@pytest.mark.parametrize("factor", [2, 4])
def test_sharpness_is_measured_at_model_scale(factor):
    """An upscaled copy adds no detail. At model scale it keeps about half of the original value
    (the resampling round trip attenuates content near Nyquist); measured at native resolution it
    would collapse by more than 10x and flag every high-resolution photo as blurry."""
    image = scene(160, 160)
    large = cv2.resize(image, (160 * factor, 160 * factor), interpolation=cv2.INTER_CUBIC)
    small = frequency.high_frequency_energy_ratio(image, reference_size=160)
    at_model_scale = frequency.high_frequency_energy_ratio(large, reference_size=160)
    at_native_scale = frequency.high_frequency_energy_ratio(large, reference_size=160 * factor)
    assert 0.4 < at_model_scale / small < 1.1
    assert at_native_scale / small < 0.1


def test_log_magnitude_spectrum_is_centered():
    spectrum = frequency.log_magnitude_spectrum(scene(64, 96))
    assert spectrum.shape == (64, 96) and spectrum.dtype == np.uint8
    assert spectrum[32, 48] == 255  # DC term of a positive image dominates


def test_transfer_function_image():
    picture = frequency.transfer_function_image(lambda d: frequency.gaussian_lowpass(d, 0.1), size=64)
    assert picture[32, 32] == 255 and picture[0, 0] < 10


def test_short_axis_uses_one_reflect_period():
    """A 6-row channel with a 33 px pad: one mirrored period keeps the result equal to the spatial filter."""
    channel = cv2.cvtColor(scene(6, 400), cv2.COLOR_BGR2GRAY).astype(np.float64)
    cutoff = 0.02
    sigma = 1.0 / (2.0 * np.pi * cutoff)
    pad = frequency.gaussian_pad(cutoff)
    assert pad > channel.shape[0]
    in_frequency = frequency.filter_channel(channel, lambda d: frequency.gaussian_lowpass(d, cutoff), pad=pad)
    in_space = cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    assert in_frequency.shape == channel.shape
    assert np.abs(in_frequency - in_space).max() < 0.5
