"""Figures for the presentation: before/after per technique, FFT spectra, and the results table.

Figures are display copies (longest side 640 px, the model's input size); the service itself
never resizes. Labels use plain ASCII because OpenCV's Hershey fonts have no accents.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from epi_imgproc import frequency, intake, techniques

DISPLAY_SIZE = frequency.MODEL_INPUT_SIZE
BAR_HEIGHT = 28


def select_evenly(items: Sequence, count: int = 3) -> list:
    """Deterministic spread over a sorted list: first, middle, last for count=3."""
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[0]]
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def display_copy(image: np.ndarray) -> np.ndarray:
    scale = DISPLAY_SIZE / float(max(image.shape[:2]))
    if scale >= 1.0:
        return image
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _as_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image


def with_caption(image: np.ndarray, text: str) -> np.ndarray:
    image = _as_bgr(image)
    bar = np.full((BAR_HEIGHT, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(bar, text, (6, BAR_HEIGHT - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 8) -> np.ndarray:
    left, right = _as_bgr(left), _as_bgr(right)
    height = max(left.shape[0], right.shape[0])

    def pad(image: np.ndarray) -> np.ndarray:
        return cv2.copyMakeBorder(image, 0, height - image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    spacer = np.full((height, gap, 3), 255, dtype=np.uint8)
    return np.hstack([pad(left), spacer, pad(right)])


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(intake.encode_png(image))


def transfer_functions() -> dict[str, np.ndarray]:
    """H(u, v) of the frequency-domain techniques, with their model-scale parameters."""
    lowpass = techniques.TECHNIQUES["fft_lowpass"].params
    emphasis = techniques.TECHNIQUES["fft_high_emphasis"].params
    return {
        "fft_lowpass": frequency.transfer_function_image(
            lambda d: frequency.gaussian_lowpass(d, lowpass["cutoff"])
        ),
        "fft_high_emphasis": frequency.transfer_function_image(
            lambda d: frequency.high_frequency_emphasis(d, emphasis["cutoff"], emphasis["k1"], emphasis["k2"])
        ),
    }


def make_figures(sources: Sequence[tuple[str, np.ndarray]], out_dir: Path, note: str = "") -> list[Path]:
    """Write before/after pairs and spectra for every technique and every (name, BGR image) source."""
    written = []
    for name, original in sources:
        before = display_copy(original)
        before_hf = frequency.high_frequency_energy_ratio(original)
        for technique in techniques.TECHNIQUES.values():
            if technique.name == "none":
                continue
            processed = technique(original)
            after = display_copy(processed)
            after_hf = frequency.high_frequency_energy_ratio(processed)
            pair = side_by_side(
                with_caption(before, f"antes {note}".strip()),
                with_caption(after, f"depois: {technique.name}"),
            )
            path = out_dir / "antes-depois" / technique.name / f"{name}.png"
            _write_png(path, pair)
            written.append(path)
            spectra = side_by_side(
                with_caption(frequency.log_magnitude_spectrum(before), f"espectro antes  HF={before_hf:.4f}"),
                with_caption(frequency.log_magnitude_spectrum(after), f"espectro depois  HF={after_hf:.4f}"),
            )
            path = out_dir / "espectros" / technique.name / f"{name}.png"
            _write_png(path, spectra)
            written.append(path)
    for name, image in transfer_functions().items():
        path = out_dir / "espectros" / "filtros" / f"{name}_H.png"
        _write_png(path, with_caption(image, f"H(u,v) {name}"))
        written.append(path)
    return written


def copy_results_table(results_md: Path, out_dir: Path) -> Path:
    target = out_dir / "tabela-resultados.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    if results_md.exists():
        shutil.copyfile(results_md, target)
        results_csv = results_md.with_suffix(".csv")
        if results_csv.exists():
            shutil.copyfile(results_csv, out_dir / "tabela-resultados.csv")
    else:
        target.write_text(f"Sem resultados: {results_md} não existe (a avaliação A/B não rodou).\n")
    return target


def synthetic_scene(height: int = 480, width: int = 640, seed: int = 0) -> np.ndarray:
    """Worker-like synthetic scene with a periodic fence (spectral peaks) and sensor noise."""
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    sky = np.linspace(230, 150, height // 2, dtype=np.float32)[:, None]
    image[: height // 2] = np.dstack([sky + 10, sky, sky - 40]).clip(0, 255).astype(np.uint8)
    image[height // 2 :] = (70, 110, 120)
    for x in range(0, width, 12):  # fence: a periodic pattern shows up as peaks in the spectrum
        cv2.line(image, (x, height // 2 - 60), (x, height // 2 + 40), (60, 60, 60), 2)
    cx = width // 2
    cv2.rectangle(image, (cx - 60, 190), (cx + 60, 380), (0, 140, 255), -1)  # vest
    for y in (250, 300):
        cv2.rectangle(image, (cx - 60, y), (cx + 60, y + 10), (200, 230, 230), -1)  # reflective stripes
    cv2.circle(image, (cx, 150), 38, (150, 180, 220), -1)  # face
    cv2.ellipse(image, (cx, 128), (48, 30), 0, 180, 360, (0, 220, 255), -1)  # helmet
    cv2.rectangle(image, (cx - 30, 140), (cx + 30, 156), (40, 40, 40), -1)  # goggles band
    noise = rng.normal(0, 6, image.shape)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
