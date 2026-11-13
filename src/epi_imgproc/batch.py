"""Phase-2 dataset batch: a raw copy without repeated photos and one treated copy per technique.

Layout written under a destination root, which is never inside the repository:

    cru/                     image bytes exactly as in the Roboflow export; labels as boxes only
    tratado-<technique>/     the same files, treated by `pipeline.process_image`, saved as JPEG

Every copy has train/, valid/ and test/ (each with images/ and labels/) and its own data.yaml.
Raw and treated training are only comparable when the copies hold exactly the same photos and
labels, so a photo that cannot be treated stops the batch instead of being skipped. Nothing is
ever deleted or overwritten: each step refuses a target that already has files, and every file
is created in exclusive mode.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import time
import warnings
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import PIL
import yaml
from PIL import Image

from epi_imgproc import duplicates, evaluation, intake, pipeline, techniques
from epi_imgproc.duplicates import Removal, SourceImage

SPLITS = ("train", "valid", "test")
RAW_COPY = "cru"
TREATMENT_INFO = "tratamento.json"
BATCH_MAX_PIXELS = 50_000_000  # the batch's own pixel limit; the endpoint keeps intake.MAX_PIXELS
JPEG_SUFFIXES = (".jpg", ".jpeg")

_ENTRY = re.compile(r"^(train|valid|test)/(images|labels)/([^/\\]+)$")

LABELS_EQUAL = "iguais"
LABELS_COORDINATES = "coordenadas"  # same classes, other coordinates
LABELS_CLASSES = "classes"  # different classes


class BatchError(Exception):
    """The batch cannot go on. It never skips a photo to keep going."""


class DestinationNotEmptyError(BatchError):
    """The target already has files; this code never deletes or overwrites."""


def treated_copy_name(technique: str) -> str:
    return f"tratado-{technique}"


def label_name(image_name: str) -> str:
    return Path(image_name).stem + ".txt"


# --- files: nothing is deleted, nothing is overwritten --------------------------------------------


def require_empty(path: Path) -> None:
    """Refuse a target that exists with anything inside (or that is not a directory)."""
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise DestinationNotEmptyError(
            f"{path} already has files. This script never deletes or overwrites: "
            "choose another destination or remove it yourself."
        )


def require_absent(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise DestinationNotEmptyError(f"already exists, and this script never overwrites: {', '.join(existing)}")


def write_new(path: Path, data: bytes) -> None:
    """Create a file; fails with FileExistsError instead of replacing one."""
    with path.open("xb") as handle:
        handle.write(data)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- labels ---------------------------------------------------------------------------------------


def boxes_only_label(text: str) -> tuple[str, int]:
    """Turn polygon rows (class x1 y1 x2 y2 ...) into their enclosing box; every other byte stays.

    Returns the new text and the number of rows converted. Ultralytics 8.4 drops an image whose
    label file mixes polygon and box rows, so the raw copy holds boxes only. The box is written
    with full float precision, as the export writes its own coordinates.
    """
    lines = text.split("\n")
    converted = 0
    for index, line in enumerate(lines):
        parts = line.split()
        if len(parts) in (0, 5):
            continue
        coordinates = parts[1:]
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(f"line {index + 1}: {len(coordinates)} values is neither a box nor a polygon")
        try:
            values = [float(value) for value in coordinates]
        except ValueError:
            raise ValueError(f"line {index + 1}: polygon coordinate is not a number") from None
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"line {index + 1}: polygon coordinate is not finite")
        box = evaluation.polygon_to_box(values)
        lines[index] = " ".join([parts[0], *(repr(value) for value in box)]) + ("\r" if line.endswith("\r") else "")
        converted += 1
    return "\n".join(lines), converted


def label_rows(text: str) -> list[list[str]]:
    return [line.split() for line in text.splitlines() if line.strip()]


def class_counts(text: str, names: Mapping[int, str]) -> Counter:
    """Boxes per class name in one label file."""
    return Counter(names[int(row[0])] for row in label_rows(text))


def compare_labels(first: str, second: str) -> str:
    """LABELS_EQUAL, LABELS_COORDINATES (same classes, other coordinates) or LABELS_CLASSES."""
    rows_first = sorted(" ".join(row) for row in label_rows(first))
    rows_second = sorted(" ".join(row) for row in label_rows(second))
    if rows_first == rows_second:
        return LABELS_EQUAL
    classes_first = sorted(row.split()[0] for row in rows_first)
    classes_second = sorted(row.split()[0] for row in rows_second)
    return LABELS_COORDINATES if classes_first == classes_second else LABELS_CLASSES


# --- the source archive ---------------------------------------------------------------------------


def _decode_label(data: bytes, where: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BatchError(f"{where}: label file is not UTF-8 text") from exc


def read_image(archive: zipfile.ZipFile, image: SourceImage) -> bytes:
    return archive.read(f"{image.split}/images/{image.name}")


def read_label(archive: zipfile.ZipFile, image: SourceImage) -> str:
    """Label of one image as it goes into the raw copy: polygons already turned into boxes."""
    where = f"{image.split}/labels/{label_name(image.name)}"
    try:
        return boxes_only_label(_decode_label(archive.read(where), where))[0]
    except ValueError as exc:
        raise BatchError(f"{where}: {exc}") from exc


def read_archive_index(archive: zipfile.ZipFile) -> tuple[list[SourceImage], dict]:
    """Check the export layout and hash every image. Returns the images and the parsed data.yaml.

    Only `<split>/images/<file>`, `<split>/labels/<file>` and the root `data.yaml` are read; entry
    names are validated, so nothing in the archive can choose where a file is written.
    """
    names: dict[tuple[str, str], list[str]] = {(split, kind): [] for split in SPLITS for kind in ("images", "labels")}
    configs = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        match = _ENTRY.match(info.filename)
        if match:
            split, kind, name = match.groups()
            if name in (".", "..") or not name.isprintable():
                raise BatchError(f"unsafe entry name in the archive: {info.filename!r}")
            names[(split, kind)].append(name)
        elif info.filename == "data.yaml":
            configs.append(archive.read(info))
    if not configs:
        raise BatchError("the archive has no data.yaml at its root")
    if any(config != configs[0] for config in configs):
        raise BatchError("the archive has data.yaml entries with different contents")
    config = yaml.safe_load(configs[0])
    if not isinstance(config, dict) or "names" not in config:
        raise BatchError("data.yaml in the archive has no class names")

    images = []
    for split in SPLITS:
        image_names, label_names = names[(split, "images")], names[(split, "labels")]
        for kind, entries in (("images", image_names), ("labels", label_names)):
            if len(set(entries)) != len(entries):
                raise BatchError(f"the archive repeats entry names in {split}/{kind}")
        not_jpeg = [name for name in image_names if Path(name).suffix.lower() not in JPEG_SUFFIXES]
        if not_jpeg:
            raise BatchError(f"{split}: treated copies are JPEG and keep the file names, but found {not_jpeg[:3]}")
        expected = {label_name(name) for name in image_names}
        if len(expected) != len(image_names) or expected != set(label_names):
            without_label = len(expected - set(label_names))
            without_image = len(set(label_names) - expected)
            raise BatchError(
                f"{split}: images and labels do not match one to one "
                f"({without_label} images without label, {without_image} labels without image)"
            )
        for name in sorted(image_names):
            images.append(SourceImage(split, name, sha256_bytes(archive.read(f"{split}/images/{name}"))))
    if not images:
        raise BatchError("the archive has no images under train/, valid/ or test/")
    return images, config


# --- which photos stay ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    kept: list[SourceImage]
    removals: list[Removal]
    near_copy_threshold: float | None  # None: only exact copies were removed


def plan_selection(
    archive: zipfile.ZipFile,
    images: Sequence[SourceImage],
    near_copy_threshold: float | None = None,
) -> Selection:
    """Decide which photos stay: exact copies first, then (with a threshold) near copies across splits."""
    kept, removals = duplicates.resolve_exact_copies(images)
    if near_copy_threshold is not None:
        thumbnails = np.stack([duplicates.thumbnail(read_image(archive, image)) for image in kept])
        pairs = duplicates.cross_split_pairs(kept, thumbnails, near_copy_threshold)
        kept, near = duplicates.resolve_near_copies(kept, pairs, near_copy_threshold)
        removals = removals + near
    return Selection(list(kept), list(removals), near_copy_threshold)


# --- the raw copy ---------------------------------------------------------------------------------


def dataset_yaml(config: Mapping) -> str:
    """data.yaml of one copy. Paths are relative to the file's own folder, so a copy can be moved."""
    names = evaluation.names_to_dict(config["names"])
    if sorted(names) != list(range(len(names))):
        raise BatchError("class indices in data.yaml are not 0..n-1")
    document: dict = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": [names[index] for index in range(len(names))],
    }
    if "roboflow" in config:
        document["roboflow"] = config["roboflow"]  # provenance and license of the source dataset
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True)


def build_raw_copy(
    archive: zipfile.ZipFile,
    kept: Sequence[SourceImage],
    config: Mapping,
    dest_root: Path,
) -> dict:
    """Write `<dest_root>/cru`: original image bytes and boxes-only labels of the photos that stay.

    `dest_root` must be empty or absent. Every label is read and converted before the first byte
    is written, so bad input leaves nothing behind.
    """
    require_empty(dest_root)
    raw_root = dest_root / RAW_COPY
    labels: dict[SourceImage, bytes] = {}
    polygon_rows = polygon_files = 0
    for image in kept:
        where = f"{image.split}/labels/{label_name(image.name)}"
        try:
            text, converted = boxes_only_label(_decode_label(archive.read(where), where))
        except ValueError as exc:
            raise BatchError(f"{where}: {exc}") from exc
        labels[image] = text.encode("utf-8")
        polygon_rows += converted
        polygon_files += bool(converted)
    config_text = dataset_yaml(config)

    for split in SPLITS:
        (raw_root / split / "images").mkdir(parents=True, exist_ok=True)
        (raw_root / split / "labels").mkdir(parents=True, exist_ok=True)
    image_bytes = 0
    for image in kept:
        data = read_image(archive, image)
        if sha256_bytes(data) != image.sha256:
            raise BatchError(f"{image.split}/images/{image.name} changed while the archive was being read")
        write_new(raw_root / image.split / "images" / image.name, data)
        write_new(raw_root / image.split / "labels" / label_name(image.name), labels[image])
        image_bytes += len(data)
    write_new(raw_root / "data.yaml", config_text.encode("utf-8"))
    return {
        "images": dict(Counter(image.split for image in kept)),
        "image_bytes": image_bytes,
        "polygon_rows_converted": polygon_rows,
        "label_files_with_polygons": polygon_files,
    }


# --- copies on disk -------------------------------------------------------------------------------


def list_copy(copy_root: Path) -> dict[str, list[str]]:
    """Image file names of each split, sorted; fails unless images and labels match one to one."""
    if not (copy_root / "data.yaml").is_file():
        raise BatchError(f"{copy_root} has no data.yaml: it is not a finished copy")
    listing = {}
    for split in SPLITS:
        image_dir, label_dir = copy_root / split / "images", copy_root / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise BatchError(f"{copy_root / split} has no images/ and labels/")
        images = sorted(path.name for path in image_dir.iterdir())
        labels = sorted(path.name for path in label_dir.iterdir())
        if sorted(label_name(name) for name in images) != labels:
            raise BatchError(f"{copy_root / split}: images and labels do not match one to one")
        listing[split] = images
    return listing


def stored_size(path: Path) -> tuple[int, int, str]:
    """(width, height, format) from the file header, without decoding the pixels."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as probe:
            return probe.size[0], probe.size[1], str(probe.format)


def _preflight(
    raw_root: Path,
    listing: Mapping[str, Sequence[str]],
    max_pixels: int,
) -> dict[tuple[str, str], tuple[int, int]]:
    """Check every raw image against the batch limits before anything is written."""
    sizes, problems = {}, []
    for split, names in listing.items():
        for name in names:
            path = raw_root / split / "images" / name
            where = f"{split}/images/{name}"
            if Path(name).suffix.lower() not in JPEG_SUFFIXES:
                problems.append(f"{where}: treated copies are JPEG and keep the file name")
                continue
            if path.stat().st_size > intake.MAX_UPLOAD_BYTES:
                problems.append(f"{where}: {path.stat().st_size} bytes exceeds {intake.MAX_UPLOAD_BYTES}")
            try:
                width, height, _ = stored_size(path)
            except Exception as exc:
                problems.append(f"{where}: cannot read the image header ({type(exc).__name__}: {exc})")
                continue
            if width * height > max_pixels:
                pixels = width * height
                problems.append(f"{where}: {width}x{height} = {pixels} pixels exceeds the limit of {max_pixels}")
            sizes[(split, name)] = (width, height)
    if problems:
        shown = "\n  ".join(problems[:20])
        raise BatchError(f"{len(problems)} image(s) cannot be treated; nothing was written:\n  {shown}")
    return sizes


def library_versions() -> dict:
    return {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
    }


def build_treated_copy(
    dest_root: Path,
    technique: str,
    max_pixels: int = BATCH_MAX_PIXELS,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Write `<dest_root>/tratado-<technique>` from `<dest_root>/cru`.

    Images go through `pipeline.process_image`, the function the endpoint calls, and are saved as
    JPEG under the same file names; labels and data.yaml are copied byte for byte. No quality
    measure is taken on any split. Any failure raises `BatchError` naming the photo.
    """
    if technique not in techniques.TECHNIQUES:
        raise BatchError(f"unknown technique: {technique!r}")
    raw_root = dest_root / RAW_COPY
    target = dest_root / treated_copy_name(technique)
    require_empty(target)
    listing = list_copy(raw_root)
    sizes = _preflight(raw_root, listing, max_pixels)
    total = sum(len(names) for names in listing.values())

    started = time.monotonic()
    for split in SPLITS:
        (target / split / "images").mkdir(parents=True, exist_ok=True)
        (target / split / "labels").mkdir(parents=True, exist_ok=True)
    done = image_bytes = 0
    for split in SPLITS:
        for name in listing[split]:
            where = f"{split}/images/{name}"
            try:
                result = pipeline.process_image(
                    (raw_root / split / "images" / name).read_bytes(),
                    technique,
                    output_format="jpeg",
                    max_pixels=max_pixels,
                    measure_sharpness=False,
                )
            except Exception as exc:  # never skipped: a copy with one photo less invalidates the comparison
                raise BatchError(
                    f"{technique}: {where} failed ({type(exc).__name__}: {exc}). The batch stopped; "
                    f"{target} is incomplete and must be removed by hand before trying again."
                ) from exc
            if (result.width, result.height) != sizes[(split, name)]:
                raise BatchError(
                    f"{technique}: {where} came out as {result.width}x{result.height}, the raw file stores "
                    f"{sizes[(split, name)][0]}x{sizes[(split, name)][1]} (EXIF orientation {result.exif_orientation})"
                )
            write_new(target / split / "images" / name, result.content)
            label = label_name(name)
            write_new(target / split / "labels" / label, (raw_root / split / "labels" / label).read_bytes())
            image_bytes += len(result.content)
            done += 1
            if progress is not None:
                progress(done, total)
    write_new(target / "data.yaml", (raw_root / "data.yaml").read_bytes())
    registered = techniques.TECHNIQUES[technique]
    info = {
        "copia": target.name,
        "tecnica": technique,
        "parametros": registered.params,
        "dominio": registered.domain,
        "jpeg": intake.jpeg_encoding(),
        "limite_pixels": max_pixels,
        "limite_bytes": intake.MAX_UPLOAD_BYTES,
        "imagens": {split: len(listing[split]) for split in SPLITS},
        "bytes_imagens": image_bytes,
        "segundos": round(time.monotonic() - started, 1),
        "concluido_em": datetime.now().astimezone().isoformat(timespec="seconds"),
        "versoes": library_versions(),
    }
    # Written last: its presence marks a copy that was treated to the end.
    text = json.dumps(info, indent=2, ensure_ascii=False) + "\n"
    write_new(target / TREATMENT_INFO, text.encode("utf-8"))
    return json.loads(text)


def treated_copies(dest_root: Path) -> list[str]:
    """Techniques, in registry order, that have a finished treated copy under `dest_root`."""
    return [name for name in techniques.TECHNIQUES if (dest_root / treated_copy_name(name) / TREATMENT_INFO).is_file()]


def read_treatment_info(dest_root: Path, technique: str) -> dict:
    return json.loads((dest_root / treated_copy_name(technique) / TREATMENT_INFO).read_text(encoding="utf-8"))


# --- verification ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def check_selection_on_disk(selection: Selection, dest_root: Path) -> list[str]:
    """Differences between the photos that should stay and the photos in the raw copy (empty when equal)."""
    listing = list_copy(dest_root / RAW_COPY)
    problems = []
    for split in SPLITS:
        planned = {image.name for image in selection.kept if image.split == split}
        on_disk = set(listing[split])
        if planned != on_disk:
            problems.append(
                f"{split}: {len(on_disk - planned)} file(s) on disk that the selection removes, "
                f"{len(planned - on_disk)} file(s) of the selection missing on disk"
            )
    return problems


def verify_copies(archive: zipfile.ZipFile, dest_root: Path, technique_names: Sequence[str]) -> list[Check]:
    """The acceptance checks that need no model: raw copy against the archive, treated copies against raw.

    Every check here starts from the files found in the raw copy; `verify_against_source` adds the
    two that start from the archive (class names, and which photos must be there).
    """
    checks = []
    raw_root = dest_root / RAW_COPY
    raw = list_copy(raw_root)
    total = sum(len(names) for names in raw.values())

    hashes: Counter = Counter()
    not_in_archive, label_differences, not_boxes = [], [], []
    for split in SPLITS:
        for name in raw[split]:
            data = (raw_root / split / "images" / name).read_bytes()
            hashes[sha256_bytes(data)] += 1
            where = f"{split}/images/{name}"
            try:
                same = archive.read(where) == data
            except KeyError:
                same = False
            if not same:
                not_in_archive.append(where)
            label_where = f"{split}/labels/{label_name(name)}"
            label = (raw_root / label_where).read_bytes()
            if any(len(row) != 5 for row in label_rows(label.decode("utf-8"))):
                not_boxes.append(label_where)
            try:
                expected = boxes_only_label(archive.read(label_where).decode("utf-8"))[0].encode("utf-8")
            except (KeyError, ValueError):
                expected = None
            if expected != label:
                label_differences.append(label_where)
    repeated = sum(1 for count in hashes.values() if count > 1)
    detail = f"{total} images, {len(hashes)} distinct SHA-256"
    checks.append(Check("cru: no image SHA-256 repeats", repeated == 0, detail))
    checks.append(
        Check(
            "cru: every image is byte-identical to the archive entry of the same split and name",
            not not_in_archive,
            f"{total - len(not_in_archive)} of {total} identical" + _examples(not_in_archive),
        )
    )
    checks.append(
        Check(
            "cru: every label is the archive's label with polygon rows turned into boxes, nothing else changed",
            not label_differences and not not_boxes,
            f"{total - len(label_differences)} of {total} as expected; {len(not_boxes)} file(s) with a non-box row"
            + _examples(label_differences + not_boxes),
        )
    )

    for technique in technique_names:
        copy_root = dest_root / treated_copy_name(technique)
        prefix = f"{copy_root.name}:"
        try:
            listing = list_copy(copy_root)
        except BatchError as exc:
            checks.append(Check(f"{prefix} is a complete copy", False, str(exc)))
            continue
        finished = (copy_root / TREATMENT_INFO).is_file()
        checks.append(
            Check(
                f"{prefix} same file list as cru in every split",
                listing == raw and finished,
                ", ".join(f"{split} {len(listing[split])}" for split in SPLITS)
                + ("" if finished else f"; {TREATMENT_INFO} is missing (the copy was not treated to the end)"),
            )
        )
        if listing != raw:
            continue
        sizes_differ, labels_differ, not_jpeg = [], [], []
        for split in SPLITS:
            for name in listing[split]:
                treated_path = copy_root / split / "images" / name
                width, height, image_format = stored_size(treated_path)
                if image_format != "JPEG":
                    not_jpeg.append(f"{split}/images/{name}")
                if (width, height) != stored_size(raw_root / split / "images" / name)[:2]:
                    sizes_differ.append(f"{split}/images/{name}")
                label = f"{split}/labels/{label_name(name)}"
                if (copy_root / label).read_bytes() != (raw_root / label).read_bytes():
                    labels_differ.append(label)
        checks.append(
            Check(
                f"{prefix} every image is a JPEG with the dimensions of the raw one",
                not sizes_differ and not not_jpeg,
                f"{total - len(sizes_differ)} of {total} with equal dimensions; {len(not_jpeg)} not JPEG"
                + _examples(sizes_differ + not_jpeg),
            )
        )
        checks.append(
            Check(
                f"{prefix} labels identical to cru, byte for byte",
                not labels_differ and (copy_root / "data.yaml").read_bytes() == (raw_root / "data.yaml").read_bytes(),
                f"{total - len(labels_differ)} of {total} identical; data.yaml identical: "
                f"{(copy_root / 'data.yaml').read_bytes() == (raw_root / 'data.yaml').read_bytes()}"
                + _examples(labels_differ),
            )
        )
    return checks


def _examples(items: Sequence[str], limit: int = 3) -> str:
    return f" (e.g. {', '.join(items[:limit])})" if items else ""


def _data_yaml_difference(path: Path, expected: Mapping) -> str:
    """How the data.yaml at `path` differs from the expected document; empty when it says the same."""
    try:
        found = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        return f"cannot be read ({type(exc).__name__})"
    if found == expected:
        return ""
    if not isinstance(found, dict):
        return "is not a mapping"
    try:
        names = evaluation.names_to_dict(found.get("names") or [])
    except (TypeError, ValueError):
        names = {}
    for index, name in enumerate(expected["names"]):
        if names.get(index) != name:
            return f"class {index} is {names.get(index)!r}, the archive says {name!r}"
    for key, value in expected.items():
        if found.get(key) != value:
            return f"{key} is {found.get(key)!r}, expected {value!r}"
    return f"unexpected keys: {', '.join(sorted(str(key) for key in set(found) - set(expected)))}"


def verify_against_source(
    archive: zipfile.ZipFile,
    dest_root: Path,
    technique_names: Sequence[str],
    near_copy_threshold: float | None = None,
) -> list[Check]:
    """What the archive says the copies must hold: the class of each index and the photos.

    `verify_copies` compares the files it finds with the archive and with each other, so it cannot
    see a change made alike in every copy. These two checks start from the archive instead:

    - the data.yaml of every copy must say what `dataset_yaml` writes from the archive's data.yaml
      (class names in index order, nc, split paths), so renaming classes in all copies fails;
    - the raw copy must hold exactly the photos of the selection, computed again from the archive
      with the near-copy threshold the raw copy was built with, so a photo missing from every copy
      fails even when the manifest was written again without it.
    """
    images, config = read_archive_index(archive)
    expected = yaml.safe_load(dataset_yaml(config))
    copy_names = [RAW_COPY, *[treated_copy_name(name) for name in technique_names]]
    differences = []
    for copy_name in copy_names:
        difference = _data_yaml_difference(dest_root / copy_name / "data.yaml", expected)
        if difference:
            differences.append(f"{copy_name}: {difference}")
    checks = [
        Check(
            "data.yaml of every copy has the class names and split paths of the archive",
            not differences,
            f"{len(copy_names) - len(differences)} of {len(copy_names)} as expected; {expected['nc']} classes"
            + _examples(differences),
        )
    ]

    selection = plan_selection(archive, images, near_copy_threshold)
    problems = check_selection_on_disk(selection, dest_root)
    threshold = "none (exact copies only)" if near_copy_threshold is None else f"{near_copy_threshold:g}"
    detail = f"the selection keeps {len(selection.kept)} of {len(images)} photos; near-copy threshold: {threshold}"
    if problems:
        detail += (
            f"; {'; '.join(problems)}. The threshold must be the one the raw copy was built with "
            "(--near-copy-threshold)"
        )
    name = "cru: holds exactly the photos that the selection from the archive keeps"
    checks.append(Check(name, not problems, detail))
    return checks


def ultralytics_scan(data_yaml: Path) -> dict[str, dict]:
    """What Ultralytics' own dataset reading keeps of each split of one copy.

    Runs `check_det_dataset` on the data.yaml and builds a `YOLODataset` per split, which scans
    every image and label the way training does. Returns, per data.yaml key (train, val, test):
    files on disk, images kept, boxes kept, the scan counters and Ultralytics' messages. The scan
    is forced (an existing labels.cache is ignored) and writing labels.cache is disabled, so the
    dataset folder is left untouched. Needs ultralytics; call `evaluation.isolate_ultralytics`
    first. `check_det_dataset` looks for Arial.ttf in the Ultralytics config dir and downloads it
    when missing.
    """
    from ultralytics.data import dataset as yolo
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_yaml), autodownload=False)

    class Scan(yolo.YOLODataset):
        def _load_or_scan_cache(self, cache_path, cache_hash):
            return self.cache_labels(cache_path), False

        def cache_labels(self, path):
            cache = super().cache_labels(path)
            found, missing, empty, corrupt, total = cache["results"]
            self.scan = {
                "files": total,
                "labels_found": found,
                "labels_missing": missing,
                "labels_empty": empty,
                "corrupt": corrupt,
                "messages": list(cache["msgs"]),
            }
            return cache

    def keep_in_memory(prefix, path, cache, version):
        cache["version"] = version  # what the real function adds before writing labels.cache

    scans = {}
    save_cache = yolo.save_dataset_cache_file
    yolo.save_dataset_cache_file = keep_in_memory
    try:
        for key in ("train", "val", "test"):
            dataset = Scan(img_path=data[key], data=data, task="detect", augment=False, prefix=f"{key}: ")
            scans[key] = {
                **dataset.scan,
                "kept": len(dataset.labels),
                "boxes": int(sum(len(label["cls"]) for label in dataset.labels)),
            }
    finally:
        yolo.save_dataset_cache_file = save_cache
    return scans
