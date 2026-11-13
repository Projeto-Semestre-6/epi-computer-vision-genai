import numpy as np
import pytest

from epi_imgproc import duplicates
from epi_imgproc.duplicates import NearPair, SourceImage
from phase2_data import jpeg, noisy, photo


def image(split: str, name: str, sha: str = "") -> SourceImage:
    return SourceImage(split, name, sha or f"{split}/{name}")


def test_exact_copies_keep_test_then_valid_then_train():
    images = [
        image("train", "a.jpg", "A"), image("valid", "a2.jpg", "A"), image("test", "zz.jpg", "A"),
        image("train", "b.jpg", "B"), image("valid", "b.jpg", "B"),
        image("train", "c.jpg", "C"),
    ]  # fmt: skip
    kept, removals = duplicates.resolve_exact_copies(images)
    assert [(i.split, i.name) for i in kept] == [("test", "zz.jpg"), ("valid", "b.jpg"), ("train", "c.jpg")]
    assert [(r.image.split, r.image.name, r.kept.split, r.kept.name) for r in removals] == [
        ("train", "a.jpg", "test", "zz.jpg"),
        ("valid", "a2.jpg", "test", "zz.jpg"),
        ("train", "b.jpg", "valid", "b.jpg"),
    ]
    assert all(r.reason == duplicates.EXACT_COPY and r.distance is None for r in removals)


def test_exact_copies_in_one_split_keep_the_first_name_in_alphabetical_order():
    images = [image("train", "x.rf.b.jpg", "S"), image("train", "x.rf.a.jpg", "S"), image("train", "x.rf.c.jpg", "S")]
    kept, removals = duplicates.resolve_exact_copies(images)
    assert [i.name for i in kept] == ["x.rf.a.jpg"]
    assert sorted(r.image.name for r in removals) == ["x.rf.b.jpg", "x.rf.c.jpg"]
    assert {r.kept.name for r in removals} == {"x.rf.a.jpg"}
    # code point order: an uppercase name comes before a lowercase one
    kept, _ = duplicates.resolve_exact_copies([image("valid", "video.jpg", "T"), image("valid", "Video.jpg", "T")])
    assert [i.name for i in kept] == ["Video.jpg"]


def test_without_repeats_nothing_is_removed_and_order_is_kept():
    images = [image("train", "b.jpg"), image("train", "a.jpg"), image("test", "c.jpg")]
    assert duplicates.resolve_exact_copies(images) == (images, [])


def test_thumbnail_ignores_size_and_aspect_ratio():
    small = duplicates.thumbnail(jpeg(photo(1, 120, 160)))
    assert small.shape == (64, 64) and small.dtype == np.float32
    large = duplicates.thumbnail(jpeg(photo(1, 360, 240)))  # same 6x8 pattern drawn at another size and shape
    assert duplicates.thumbnail_distance(small, large)[0] < 8  # only the seed number is drawn differently
    with pytest.raises(ValueError):
        duplicates.thumbnail(b"not an image")


def test_distance_separates_noisy_copies_from_other_photos():
    original = duplicates.thumbnail(jpeg(photo(4)))
    copy = duplicates.thumbnail(jpeg(noisy(photo(4), 1)))
    other = duplicates.thumbnail(jpeg(photo(5)))
    assert duplicates.thumbnail_distance(original, original) == (0.0, "identity")
    near, symmetry = duplicates.thumbnail_distance(original, copy)
    assert 0 < near < 2 and symmetry == "identity"
    assert duplicates.thumbnail_distance(original, other)[0] > 20


@pytest.mark.parametrize(
    "transform, expected",
    [
        (lambda t: t[:, ::-1], {"mirror"}),
        (lambda t: np.rot90(t, 1), {"rotate_90", "rotate_270"}),
        (lambda t: np.rot90(t, 2), {"rotate_180"}),
        (lambda t: t[::-1, :], {"mirror_rotate_180"}),
        (lambda t: t.T, {"mirror_rotate_90", "mirror_rotate_270"}),
    ],
    ids=["mirrored", "quarter-turn", "half-turn", "upside-down", "transposed"],
)
def test_distance_sees_through_mirroring_and_quarter_turns(transform, expected):
    original = duplicates.thumbnail(jpeg(photo(3)))
    turned = np.ascontiguousarray(transform(original))
    distance, symmetry = duplicates.thumbnail_distance(original, turned)
    assert distance == 0.0 and symmetry in expected
    assert duplicates.thumbnail_distance(turned, original)[0] == 0.0  # symmetric
    plain = float(np.abs(original - turned).mean())
    assert plain > 10  # without the symmetries the same photo would look like another one


def random_sets(seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 255, (30, 64, 64)).astype(np.float32)
    smooth = np.stack([np.kron(rng.uniform(0, 255, (8, 8)), np.ones((8, 8))) for _ in range(30)]).astype(np.float32)
    higher = np.concatenate([base[:10], smooth[:10]])
    lower = np.concatenate(
        [
            higher[:5] + rng.normal(0, 1.5, (5, 64, 64)).astype(np.float32),  # noisy copies
            higher[10:13][:, :, ::-1] + 1.0,  # mirrored, slightly brighter
            np.rot90(higher[13:15], 1, axes=(1, 2)).copy(),  # quarter turn
            smooth[10:20] + rng.normal(0, 6, (10, 64, 64)).astype(np.float32),  # unrelated, same statistics
            base[10:20],  # unrelated
        ]
    )
    return lower, higher


@pytest.mark.parametrize("max_distance", [0.5, 2.0, 30.0, 300.0])
def test_close_pairs_equals_the_pairwise_definition(max_distance):
    lower, higher = random_sets()
    expected = {}
    for i in range(len(lower)):
        for j in range(len(higher)):
            distance, symmetry = duplicates.thumbnail_distance(lower[i], higher[j])
            if distance <= max_distance:
                expected[(i, j)] = (distance, symmetry)
    found = {(i, j): (distance, symmetry) for i, j, distance, symmetry in duplicates.close_pairs(lower, higher, max_distance)}
    assert set(found) == set(expected)
    for key, (distance, symmetry) in expected.items():
        assert found[key][0] == pytest.approx(distance, abs=1e-9)
        assert found[key][1] == symmetry
    if max_distance == 2.0:
        assert {(i, j) for i, j in found} == {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 10), (6, 11), (7, 12), (8, 13), (9, 14)}
        assert found[(5, 10)][1] == "mirror" and found[(8, 13)][1] in {"rotate_90", "rotate_270"}


def test_close_pairs_with_empty_input_and_bad_thumbnail_size():
    lower, higher = random_sets()
    assert duplicates.close_pairs(lower[:0], higher, 5.0) == []
    assert duplicates.close_pairs(lower, higher[:0], 5.0) == []
    with pytest.raises(ValueError):
        duplicates.close_pairs(np.zeros((1, 60, 60), np.float32), np.zeros((1, 60, 60), np.float32), 5.0)


def test_cross_split_pairs_point_from_the_lower_priority_split_to_the_higher():
    lower, higher = random_sets()
    train = [image("train", f"t{i}.jpg") for i in range(5)]
    images = [*train, image("valid", "v0.jpg"), *[image("test", f"x{i}.jpg") for i in range(5)]]
    thumbnails = np.concatenate([lower[:5], higher[:1] + 0.5, higher[:5]])  # valid v0 is a copy of test x0 too
    pairs = duplicates.cross_split_pairs(images, thumbnails, 3.0)
    as_names = {(images[p.lower].name, images[p.higher].name) for p in pairs}
    assert as_names == {*[(f"t{i}.jpg", f"x{i}.jpg") for i in range(5)], ("t0.jpg", "v0.jpg"), ("v0.jpg", "x0.jpg")}
    assert all(images[p.lower].split != images[p.higher].split for p in pairs)
    with pytest.raises(ValueError, match="unknown splits"):
        duplicates.cross_split_pairs([image("val", "a.jpg")], thumbnails[:1], 3.0)


def test_near_copies_leave_the_lower_priority_split():
    images = [image("train", name) for name in ("a.jpg", "b.jpg", "c.jpg")]
    images += [image("valid", "v.jpg"), image("valid", "w.jpg"), image("test", "t.jpg")]
    pairs = [
        NearPair(0, 3, 1.0, "identity"),  # train a ~ valid v
        NearPair(0, 5, 0.4, "mirror"),  # train a ~ test t, closer
        NearPair(1, 4, 2.5, "identity"),  # train b ~ valid w, above the threshold used below
        NearPair(2, 3, 9.0, "identity"),
    ]
    kept, removals = duplicates.resolve_near_copies(images, pairs, 2.0)
    assert [i.name for i in kept] == ["b.jpg", "c.jpg", "v.jpg", "w.jpg", "t.jpg"]
    (removal,) = removals
    assert (removal.image.name, removal.kept.name, removal.distance, removal.symmetry) == ("a.jpg", "t.jpg", 0.4, "mirror")
    assert removal.reason == duplicates.NEAR_COPY
    kept, removals = duplicates.resolve_near_copies(images, pairs, 2.5)  # the threshold itself is included
    assert sorted(r.image.name for r in removals) == ["a.jpg", "b.jpg"]


def test_a_photo_stays_when_its_only_near_copy_leaves():
    """train a ~ valid v ~ test t, but train a is not within the threshold of test t."""
    images = [image("train", "a.jpg"), image("train", "b.jpg")]
    images += [image("valid", "v.jpg"), image("valid", "w.jpg"), image("test", "t.jpg")]
    pairs = [
        NearPair(2, 4, 1.5, "identity"),  # valid v ~ test t
        NearPair(0, 2, 1.5, "identity"),  # train a ~ valid v
        NearPair(1, 3, 1.0, "identity"),  # train b ~ valid w
        NearPair(1, 2, 0.5, "identity"),  # train b ~ valid v, closer
    ]
    kept, removals = duplicates.resolve_near_copies(images, pairs, 2.0)
    assert [(r.image.name, r.kept.name) for r in removals] == [("b.jpg", "w.jpg"), ("v.jpg", "t.jpg")]  # b's nearest (v) leaves
    assert [i.name for i in kept] == ["a.jpg", "w.jpg", "t.jpg"]
    # no pair of kept photos in different splits is within the threshold
    staying = {images.index(i) for i in kept}
    assert not [p for p in pairs if p.distance <= 2.0 and p.lower in staying and p.higher in staying]


def test_no_pairs_no_removals():
    images = [image("train", "a.jpg"), image("test", "t.jpg")]
    assert duplicates.resolve_near_copies(images, [], 2.0) == (images, [])


def test_nearest_partners():
    pairs = [NearPair(0, 5, 2.0, "identity"), NearPair(0, 6, 0.5, "mirror"), NearPair(1, 5, 1.0, "identity")]
    assert duplicates.nearest_partners(pairs) == [pairs[1], pairs[2]]
    assert duplicates.nearest_partners([]) == []
