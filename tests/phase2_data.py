"""A small synthetic Roboflow export for the phase-2 batch tests; no dataset image is stored in the repository.

What the archive holds, by design:

- exact copies (same bytes): `a` twice inside train, `b` in train and valid, `c` in train and test,
  `w` in valid and test;
- near copies across splits: train `d` is valid `d` plus noise; train `e` is test `e` mirrored plus noise;
- labels without a trailing newline, as Roboflow writes them; train `f` mixes a polygon and boxes,
  train `g` has polygons only.

`chain_entries` adds two chains across the three splits to that archive:

- near, near: train `p` ~ valid `p` ~ test `p`, with train `p` and test `p` farther apart than
  NEAR_THRESHOLD;
- exact, near: a noisy copy of `b` in test, so valid `b` (the exact copy kept instead of train `b`)
  leaves as a near copy.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np

from synthetic import encode_cv2

NAMES = ["boots", "gloves", "goggles", "helmet", "no-boots", "no-gloves", "no-goggles", "no-helmet", "no-vest", "vest"]
DATA_YAML = (
    "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n"
    f"nc: {len(NAMES)}\nnames: {NAMES}\n\n"
    "roboflow:\n  workspace: synthetic\n  project: ppe\n  version: 1\n  license: CC BY 4.0\n"
)
NEAR_THRESHOLD = 3.0  # near copies here measure about 1; different photos measure tens of gray levels

BOX_ROWS = "3 0.5 0.3 0.2 0.2\n9 0.5 0.6 0.3 0.4"
POLYGON_ROW = "1 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6"  # encloses x 0.1..0.3, y 0.2..0.6


def photo(seed: int, height: int = 120, width: int = 160) -> np.ndarray:
    """Smooth random BGR picture; different seeds give visibly different photos."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    image = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    cv2.putText(image, str(seed), (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return image


def noisy(image: np.ndarray, seed: int, amplitude: int = 3) -> np.ndarray:
    rng = np.random.default_rng(1000 + seed)
    noise = rng.integers(-amplitude, amplitude + 1, image.shape)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def jpeg(image: np.ndarray) -> bytes:
    return encode_cv2(image, ".jpg")


def export_entries() -> dict[str, bytes]:
    """Entry name -> bytes of the synthetic export."""
    a, b, c, d, e, w = (jpeg(photo(seed)) for seed in (1, 2, 3, 4, 5, 6))
    images = {
        "train": {
            "a_jpg.rf.1.jpg": a,
            "a_jpg.rf.2.jpg": a,
            "b_jpg.rf.1.jpg": b,
            "c_jpg.rf.1.jpg": c,
            "d_jpg.rf.1.jpg": jpeg(noisy(photo(4), 1)),
            "e_jpg.rf.1.jpg": jpeg(noisy(photo(5)[:, ::-1], 2)),
            "f_jpg.rf.1.jpg": jpeg(photo(7)),
            "g_jpg.rf.1.jpg": jpeg(photo(8)),
        },
        "valid": {
            "b_jpg.rf.7.jpg": b,
            "d_jpg.rf.9.jpg": d,
            "v_jpg.rf.1.jpg": jpeg(photo(9)),
            "w_jpg.rf.1.jpg": w,
            "x_jpg.rf.1.jpg": jpeg(photo(10)),
        },
        "test": {
            "c_jpg.rf.3.jpg": c,
            "e_jpg.rf.3.jpg": e,
            "t_jpg.rf.1.jpg": jpeg(photo(11)),
            "w_jpg.rf.0.jpg": w,
        },
    }
    labels = {
        "train/a_jpg.rf.2": "3 0.5 0.3 0.2 0.2",  # other classes than the copy that stays
        "train/c_jpg.rf.1": "3 0.5 0.31 0.2 0.2\n9 0.5 0.6 0.3 0.4",  # same classes, other coordinates
        "train/f_jpg.rf.1": f"{BOX_ROWS}\n{POLYGON_ROW}",
        "train/g_jpg.rf.1": f"{POLYGON_ROW}\n0 0.5 0.5 0.6 0.5 0.6 0.7",
    }
    entries = {"data.yaml": DATA_YAML.encode(), "README.roboflow.txt": b"synthetic export\n"}
    for split, files in images.items():
        for name, data in files.items():
            stem = Path(name).stem
            entries[f"{split}/images/{name}"] = data
            entries[f"{split}/labels/{stem}.txt"] = labels.get(f"{split}/{stem}", BOX_ROWS).encode()
    return entries


CHAIN = {"train": "p_jpg.rf.1.jpg", "valid": "p_jpg.rf.2.jpg", "test": "p_jpg.rf.3.jpg"}
CHAIN_STEP = 2  # gray levels between neighbours of the chain: one step is under NEAR_THRESHOLD, two are over
CHAIN_EXACT = {"train": "b_jpg.rf.1.jpg", "valid": "b_jpg.rf.7.jpg", "test": "b_jpg.rf.5.jpg"}


def brighter(image: np.ndarray, levels: int) -> np.ndarray:
    """The same picture `levels` gray levels brighter; the range is halved first so nothing clips."""
    return (image // 2 + 60 + levels).astype(np.uint8)


def chain_entries() -> dict[str, bytes]:
    """The synthetic export plus the two chains across train, valid and test (see the module docstring)."""
    entries = export_entries()
    base = photo(12)
    for step, split in enumerate(("test", "valid", "train")):
        entries[f"{split}/images/{CHAIN[split]}"] = jpeg(brighter(base, step * CHAIN_STEP))
        entries[f"{split}/labels/{Path(CHAIN[split]).stem}.txt"] = BOX_ROWS.encode()
    entries[f"test/images/{CHAIN_EXACT['test']}"] = jpeg(noisy(photo(2), 3))
    entries[f"test/labels/{Path(CHAIN_EXACT['test']).stem}.txt"] = BOX_ROWS.encode()
    return entries


def write_export(path: Path, entries: dict[str, bytes] | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in (export_entries() if entries is None else entries).items():
            archive.writestr(name, data)
    return path


def export_bytes(entries: dict[str, bytes] | None = None) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in (export_entries() if entries is None else entries).items():
            archive.writestr(name, data)
    buffer.seek(0)
    return buffer


def tree(root: Path) -> dict[str, bytes]:
    """Every file under `root` with its bytes, to prove that a refused run changed nothing."""
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}
