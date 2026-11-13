"""Repeated photos in a dataset: exact copies (same bytes) and near copies across splits.

A model must not be validated or tested on a photo it was trained on, so one photo may stay in
one split only. Splits are kept in the order test > valid > train: when a photo is repeated, the
copy in the split that comes first in that order stays.

Exact copies are files with the same SHA-256. Near copies are measured on 64x64 grayscale
thumbnails (aspect ratio ignored): the distance between two photos is the mean absolute
difference of the thumbnails, in gray levels (0..255), minimized over the 8 mirrorings and
90-degree rotations of one of them. Nothing here reads files or decides a threshold.

The near-copy rule (`resolve_near_copies`): a photo leaves when it is within the threshold of a
photo of a higher-priority split that remains in the dataset. A photo close only to photos that
leave is kept.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

SPLIT_ORDER = ("test", "valid", "train")  # kept first ... removed first
THUMBNAIL_SIZE = 64
COARSE_BLOCK = 8  # 8x8 block means of the thumbnail: a 64-value lower bound used to prune pairs

EXACT_COPY = "exact_copy"
NEAR_COPY = "near_copy"

# The 8 symmetries of a square: 4 rotations, each with and without mirroring.
SYMMETRIES = (
    ("identity", lambda t: t),
    ("mirror", lambda t: t[..., ::-1]),
    ("rotate_90", lambda t: np.rot90(t, 1, axes=(-2, -1))),
    ("rotate_180", lambda t: np.rot90(t, 2, axes=(-2, -1))),
    ("rotate_270", lambda t: np.rot90(t, 3, axes=(-2, -1))),
    ("mirror_rotate_90", lambda t: np.rot90(t[..., ::-1], 1, axes=(-2, -1))),
    ("mirror_rotate_180", lambda t: np.rot90(t[..., ::-1], 2, axes=(-2, -1))),
    ("mirror_rotate_270", lambda t: np.rot90(t[..., ::-1], 3, axes=(-2, -1))),
)

_COARSE_SLACK = 1e-3  # float32 rounding of the lower bound must never prune a pair at the limit
_COARSE_CHUNK_VALUES = 16_000_000  # float32 values per chunk of the pruning pass (64 MB)
_EXACT_CHUNK_PAIRS = 2048  # pairs per chunk of the exact pass (64 MB of float64 at 64x64)


@dataclass(frozen=True)
class SourceImage:
    split: str
    name: str  # file name inside <split>/images
    sha256: str

    @property
    def keep_order(self) -> tuple[int, str]:
        """Sort key of the copy that stays: split priority, then file name (code point order)."""
        return (SPLIT_ORDER.index(self.split), self.name)


@dataclass(frozen=True)
class Removal:
    image: SourceImage
    reason: str  # EXACT_COPY or NEAR_COPY
    # The photo this one was compared with. Near copy: a photo that remains in the dataset. Exact
    # copy: the file kept from its group, which a later near-copy pass may still remove.
    kept: SourceImage
    distance: float | None = None  # near copies only
    symmetry: str | None = None  # near copies only: how the kept thumbnail was turned to match


@dataclass(frozen=True)
class NearPair:
    lower: int  # index of the image in the lower-priority split
    higher: int  # index of the image in the higher-priority split
    distance: float
    symmetry: str


def resolve_exact_copies(images: Sequence[SourceImage]) -> tuple[list[SourceImage], list[Removal]]:
    """Keep one file per SHA-256: test > valid > train, then the first name in alphabetical order."""
    groups: dict[str, list[SourceImage]] = defaultdict(list)
    for image in images:
        groups[image.sha256].append(image)
    keeper = {sha: min(group, key=lambda image: image.keep_order) for sha, group in groups.items()}
    kept, removals = [], []
    for image in images:
        if keeper[image.sha256] == image:
            kept.append(image)
        else:
            removals.append(Removal(image, EXACT_COPY, keeper[image.sha256]))
    return kept, removals


def thumbnail(data: bytes, size: int = THUMBNAIL_SIZE) -> np.ndarray:
    """Grayscale thumbnail (size x size, float32, 0..255) of encoded image bytes; aspect ratio is ignored."""
    gray = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("image bytes could not be decoded")
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def thumbnail_distance(first: np.ndarray, second: np.ndarray) -> tuple[float, str]:
    """Distance between two thumbnails and the symmetry of `second` that reaches it.

    Reference definition of the measure; `close_pairs` computes the same values for whole sets.
    """
    best, best_name = float("inf"), ""
    for name, transform in SYMMETRIES:
        distance = float(np.abs(first.astype(np.float64) - transform(second)).mean())
        if distance < best:
            best, best_name = distance, name
    return best, best_name


def _coarse(thumbnails: np.ndarray) -> np.ndarray:
    """Block means, flattened to (n, values). |mean(a) - mean(b)| <= mean(|a - b|) per block."""
    count, rows, cols = thumbnails.shape
    if rows % COARSE_BLOCK or cols % COARSE_BLOCK:
        raise ValueError(f"thumbnail side must be a multiple of {COARSE_BLOCK}")
    blocks = thumbnails.reshape(count, rows // COARSE_BLOCK, COARSE_BLOCK, cols // COARSE_BLOCK, COARSE_BLOCK)
    return blocks.mean(axis=(2, 4)).reshape(count, -1)


def close_pairs(lower: np.ndarray, higher: np.ndarray, max_distance: float) -> list[tuple[int, int, float, str]]:
    """Every (i, j, distance, symmetry) with distance(lower[i], higher[j]) <= max_distance.

    Exact: the block-mean distance never exceeds the thumbnail distance, so pruning by it loses no
    pair; survivors are measured on the full thumbnails. Both inputs are (n, size, size) float32.
    """
    best: dict[tuple[int, int], tuple[float, str]] = {}
    if not len(lower) or not len(higher):
        return []
    lower_coarse = _coarse(lower)
    rows_per_chunk = max(1, _COARSE_CHUNK_VALUES // (len(higher) * lower_coarse.shape[1]))
    for name, transform in SYMMETRIES:
        turned = np.ascontiguousarray(transform(higher))
        turned_coarse = _coarse(turned)
        for start in range(0, len(lower), rows_per_chunk):
            chunk = lower_coarse[start : start + rows_per_chunk]
            bound = np.abs(chunk[:, None, :] - turned_coarse[None, :, :]).mean(axis=2)
            rows, cols = np.nonzero(bound <= max_distance + _COARSE_SLACK)
            rows += start
            for offset in range(0, len(rows), _EXACT_CHUNK_PAIRS):
                i = rows[offset : offset + _EXACT_CHUNK_PAIRS]
                j = cols[offset : offset + _EXACT_CHUNK_PAIRS]
                distances = np.abs(lower[i].astype(np.float64) - turned[j]).mean(axis=(1, 2))
                for a, b, distance in zip(i.tolist(), j.tolist(), distances.tolist()):
                    if distance <= max_distance and distance < best.get((a, b), (float("inf"), ""))[0]:
                        best[(a, b)] = (distance, name)
    return [(a, b, distance, name) for (a, b), (distance, name) in sorted(best.items())]


def cross_split_pairs(images: Sequence[SourceImage], thumbnails: np.ndarray, max_distance: float) -> list[NearPair]:
    """All pairs of images in different splits within `max_distance`, as indices into `images`."""
    by_split = {split: [i for i, image in enumerate(images) if image.split == split] for split in SPLIT_ORDER}
    unknown = {image.split for image in images} - set(SPLIT_ORDER)
    if unknown:
        raise ValueError(f"unknown splits: {sorted(unknown)}")
    pairs = []
    for position, split in enumerate(SPLIT_ORDER[1:], start=1):
        lower = np.asarray(by_split[split], dtype=np.int64)
        higher = np.asarray([i for other in SPLIT_ORDER[:position] for i in by_split[other]], dtype=np.int64)
        if not len(lower) or not len(higher):
            continue
        for a, b, distance, symmetry in close_pairs(thumbnails[lower], thumbnails[higher], max_distance):
            pairs.append(NearPair(int(lower[a]), int(higher[b]), distance, symmetry))
    return pairs


def resolve_near_copies(
    images: Sequence[SourceImage],
    pairs: Sequence[NearPair],
    threshold: float,
) -> tuple[list[SourceImage], list[Removal]]:
    """Near copies across splits leave the lower-priority split.

    The rule: an image leaves when it is within `threshold` of an image of a higher-priority
    split that remains in the dataset. The nearest such image is recorded as its pair.

    Splits are decided from the highest priority down (valid against test, then train against
    what remains of test and valid), so "remains" is already settled when it is asked. In a chain
    (train a ~ valid v ~ test t, with a and t farther apart than the threshold) v leaves and a
    stays: the only image a is close to does not remain. Removing a as well would be the rule
    "within the threshold of any image of a higher-priority split", which is not the one used.

    No pair of kept images in different splits is within the threshold afterwards. Near copies
    inside one split are not touched.
    """
    candidates: dict[int, list[NearPair]] = defaultdict(list)
    for pair in pairs:
        if pair.distance <= threshold:
            candidates[pair.lower].append(pair)
    removed: dict[int, Removal] = {}
    for split in SPLIT_ORDER[1:]:
        for index, image in enumerate(images):
            if image.split != split:
                continue
            # `removed` is final for every higher-priority split here, so this is "remains in the dataset"
            staying = [pair for pair in candidates.get(index, ()) if pair.higher not in removed]
            if staying:
                nearest = min(staying, key=lambda pair: (pair.distance, images[pair.higher].keep_order))
                removed[index] = Removal(image, NEAR_COPY, images[nearest.higher], nearest.distance, nearest.symmetry)
    kept = [image for index, image in enumerate(images) if index not in removed]
    return kept, [removed[index] for index in sorted(removed)]


def nearest_partners(pairs: Sequence[NearPair]) -> list[NearPair]:
    """For each lower-priority image that has pairs, its closest pair; sorted by distance."""
    nearest: dict[int, NearPair] = {}
    for pair in pairs:
        current = nearest.get(pair.lower)
        if current is None or (pair.distance, pair.higher) < (current.distance, current.higher):
            nearest[pair.lower] = pair
    return sorted(nearest.values(), key=lambda pair: (pair.distance, pair.lower))
