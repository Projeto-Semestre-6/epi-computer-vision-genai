import json
import zipfile

import numpy as np
import pytest
import yaml
from PIL import Image

from conftest import REPO, load_script
from epi_imgproc import batch, batch_report, duplicates, intake, pipeline, techniques
from phase2_data import BOX_ROWS, CHAIN, CHAIN_EXACT, NAMES, NEAR_THRESHOLD, POLYGON_ROW
from phase2_data import chain_entries, export_bytes, export_entries, jpeg, photo, tree, write_export

EXACT_REMOVED = {
    ("train", "a_jpg.rf.2.jpg"), ("train", "b_jpg.rf.1.jpg"), ("train", "c_jpg.rf.1.jpg"), ("valid", "w_jpg.rf.1.jpg"),
}  # fmt: skip
NEAR_REMOVED = {("train", "d_jpg.rf.1.jpg"), ("train", "e_jpg.rf.1.jpg")}


@pytest.fixture
def export(tmp_path):
    return write_export(tmp_path / "export.zip")


def build_raw(export, dest, threshold=None):
    with zipfile.ZipFile(export) as archive:
        images, config = batch.read_archive_index(archive)
        selection = batch.plan_selection(archive, images, threshold)
        stats = batch.build_raw_copy(archive, selection.kept, config, dest)
    return selection, stats


# --- labels ---------------------------------------------------------------------------------------


def test_polygon_rows_become_the_enclosing_box_and_every_other_byte_stays():
    text = f"{BOX_ROWS}\n{POLYGON_ROW}"
    converted, count = batch.boxes_only_label(text)
    assert count == 1
    assert converted.startswith(BOX_ROWS + "\n") and not converted.endswith("\n")
    row = converted.split("\n")[-1].split()
    assert row[0] == "1" and len(row) == 5
    assert [float(value) for value in row[1:]] == pytest.approx([0.2, 0.4, 0.2, 0.4])  # x 0.1..0.3, y 0.2..0.6
    assert batch.boxes_only_label(BOX_ROWS) == (BOX_ROWS, 0)
    assert batch.boxes_only_label("") == ("", 0)
    assert batch.boxes_only_label(f"\n{BOX_ROWS}\n\n") == (f"\n{BOX_ROWS}\n\n", 0)  # blank lines and final newline stay


def test_polygon_conversion_keeps_line_endings_and_full_precision():
    converted, count = batch.boxes_only_label(f"3 0.5 0.3 0.2 0.2\r\n{POLYGON_ROW}\r\n")
    assert count == 1 and converted.count("\r\n") == 2 and converted.startswith("3 0.5 0.3 0.2 0.2\r\n")
    polygon = (
        "9 0.41210937499999983 0.5130854608141997 0.43847656249999983 0.5101557733141997 "
        "0.45410156249999983 0.48671827331419965"
    )
    row = batch.boxes_only_label(polygon)[0].split()
    x_center, y_center, width, height = (float(value) for value in row[1:])
    assert x_center - width / 2 == pytest.approx(0.41210937499999983, abs=1e-15)
    assert x_center + width / 2 == pytest.approx(0.45410156249999983, abs=1e-15)
    assert y_center - height / 2 == pytest.approx(0.48671827331419965, abs=1e-15)
    assert y_center + height / 2 == pytest.approx(0.5130854608141997, abs=1e-15)


@pytest.mark.parametrize(
    "text",
    [
        "0 0.1 0.2 0.3",
        "0 0.1 0.2 0.3 0.4 0.5",
        "0 0.1 0.2 0.3 0.4 0.5 0.6 0.7",
        "0 0.1 0.2 0.3 0.4 0.5 x",
        "0 0.1 0.2 0.3 0.4 0.5 nan",
    ],
    ids=["three values", "five values", "seven values", "not a number", "not finite"],
)
def test_rows_that_are_neither_box_nor_polygon_are_rejected(text):
    with pytest.raises(ValueError, match="line 2"):
        batch.boxes_only_label(f"{BOX_ROWS.splitlines()[0]}\n{text}")


def test_compare_labels():
    reordered = "9 0.5 0.6 0.3 0.4\n3 0.5 0.3 0.2 0.2\n"
    assert batch.compare_labels("3 0.5 0.3 0.2 0.2\n9 0.5 0.6 0.3 0.4", reordered) == batch.LABELS_EQUAL
    assert batch.compare_labels("3 0.5 0.3 0.2 0.2", "3 0.5 0.31 0.2 0.2") == batch.LABELS_COORDINATES
    assert batch.compare_labels("3 0.5 0.3 0.2 0.2", "7 0.5 0.3 0.2 0.2") == batch.LABELS_CLASSES
    assert batch.compare_labels("3 0.5 0.3 0.2 0.2", "3 0.5 0.3 0.2 0.2\n3 0.1 0.1 0.1 0.1") == batch.LABELS_CLASSES
    assert batch.class_counts(BOX_ROWS, dict(enumerate(NAMES))) == {"helmet": 1, "vest": 1}


# --- the archive ----------------------------------------------------------------------------------


def test_archive_index_lists_and_hashes_every_image(export):
    with zipfile.ZipFile(export) as archive:
        images, config = batch.read_archive_index(archive)
    assert [image.split for image in images] == ["train"] * 8 + ["valid"] * 5 + ["test"] * 4
    assert [image.name for image in images[:3]] == ["a_jpg.rf.1.jpg", "a_jpg.rf.2.jpg", "b_jpg.rf.1.jpg"]
    by_name = {(image.split, image.name): image.sha256 for image in images}
    assert by_name[("train", "a_jpg.rf.1.jpg")] == by_name[("train", "a_jpg.rf.2.jpg")] == batch.sha256_bytes(jpeg(photo(1)))
    assert by_name[("train", "c_jpg.rf.1.jpg")] == by_name[("test", "c_jpg.rf.3.jpg")]
    assert config["names"] == NAMES


def broken(change):
    entries = export_entries()
    change(entries)
    return zipfile.ZipFile(export_bytes(entries))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda e: e.pop("train/labels/f_jpg.rf.1.txt"), "1 images without label"),
        (lambda e: e.update({"valid/labels/extra.txt": b""}), "1 labels without image"),
        (lambda e: e.update({"test/images/z.png": b"x", "test/labels/z.txt": b""}), "JPEG"),
        (lambda e: e.update({"test/images/..": b"x"}), "unsafe entry name"),
        (lambda e: e.pop("data.yaml"), "no data.yaml"),
        (lambda e: e.update({"data.yaml": b"train: x\n"}), "no class names"),
        (lambda e: [e.pop(name) for name in list(e) if "/" in name], "no images"),
    ],
    ids=["image without label", "label without image", "not jpeg", "unsafe name", "no yaml", "yaml without names", "empty"],
)
def test_archive_with_an_unexpected_layout_is_refused(change, message):
    with broken(change) as archive, pytest.raises(batch.BatchError, match=message):
        batch.read_archive_index(archive)


def test_repeated_data_yaml_entries_must_be_identical(tmp_path):
    entries = export_entries()
    for extra, accepted in ((entries["data.yaml"], True), (entries["data.yaml"] + b"#\n", False)):
        path = tmp_path / f"export-{accepted}.zip"
        with zipfile.ZipFile(path, "w") as archive, pytest.warns(UserWarning, match="Duplicate name"):
            for name, data in entries.items():
                archive.writestr(name, data)
            archive.writestr("data.yaml", extra)
        with zipfile.ZipFile(path) as archive:
            if accepted:
                assert len(batch.read_archive_index(archive)[0]) == 17
            else:
                with pytest.raises(batch.BatchError, match="different contents"):
                    batch.read_archive_index(archive)


# --- which photos stay ----------------------------------------------------------------------------


def test_selection_without_threshold_removes_exact_copies_only(export):
    with zipfile.ZipFile(export) as archive:
        images, _ = batch.read_archive_index(archive)
        selection = batch.plan_selection(archive, images)
    assert {(r.image.split, r.image.name) for r in selection.removals} == EXACT_REMOVED
    assert {r.reason for r in selection.removals} == {duplicates.EXACT_COPY}
    kept_by = {(r.image.split, r.image.name): (r.kept.split, r.kept.name) for r in selection.removals}
    assert kept_by == {
        ("train", "a_jpg.rf.2.jpg"): ("train", "a_jpg.rf.1.jpg"),  # same split: first name in alphabetical order
        ("train", "b_jpg.rf.1.jpg"): ("valid", "b_jpg.rf.7.jpg"),  # valid > train
        ("train", "c_jpg.rf.1.jpg"): ("test", "c_jpg.rf.3.jpg"),  # test > train
        ("valid", "w_jpg.rf.1.jpg"): ("test", "w_jpg.rf.0.jpg"),  # test > valid
    }
    assert len(selection.kept) == 13 and len({image.sha256 for image in selection.kept}) == 13
    assert selection.near_copy_threshold is None


def test_selection_with_threshold_also_removes_near_copies_across_splits(export):
    with zipfile.ZipFile(export) as archive:
        images, _ = batch.read_archive_index(archive)
        selection = batch.plan_selection(archive, images, NEAR_THRESHOLD)
    near = {(r.image.split, r.image.name): r for r in selection.removals if r.reason == duplicates.NEAR_COPY}
    assert set(near) == NEAR_REMOVED
    noisy_copy, mirrored = near[("train", "d_jpg.rf.1.jpg")], near[("train", "e_jpg.rf.1.jpg")]
    assert (noisy_copy.kept.split, noisy_copy.kept.name, noisy_copy.symmetry) == ("valid", "d_jpg.rf.9.jpg", "identity")
    assert (mirrored.kept.split, mirrored.kept.name, mirrored.symmetry) == ("test", "e_jpg.rf.3.jpg", "mirror")
    assert 0 < noisy_copy.distance <= NEAR_THRESHOLD and 0 < mirrored.distance <= NEAR_THRESHOLD
    assert {(r.image.split, r.image.name) for r in selection.removals} == EXACT_REMOVED | NEAR_REMOVED
    kept_in_train = sorted(image.name for image in selection.kept if image.split == "train")
    assert kept_in_train == ["a_jpg.rf.1.jpg", "f_jpg.rf.1.jpg", "g_jpg.rf.1.jpg"]


def test_a_chain_across_the_three_splits_loses_only_the_photo_in_the_middle():
    """train p ~ valid p ~ test p, with train p and test p farther apart than the threshold.

    The rule: a photo leaves when it is within the threshold of a photo of a higher-priority split
    that remains in the dataset. valid p leaves because test p remains; train p stays, because
    valid p, the only photo it is close to, does not remain.
    """
    entries = chain_entries()
    thumbnails = {split: duplicates.thumbnail(entries[f"{split}/images/{name}"]) for split, name in CHAIN.items()}

    def measure(first: str, second: str) -> float:
        return duplicates.thumbnail_distance(thumbnails[first], thumbnails[second])[0]

    assert measure("train", "valid") <= NEAR_THRESHOLD and measure("valid", "test") <= NEAR_THRESHOLD
    assert measure("train", "test") > NEAR_THRESHOLD
    with zipfile.ZipFile(export_bytes(entries)) as archive:
        images, _ = batch.read_archive_index(archive)
        selection = batch.plan_selection(archive, images, NEAR_THRESHOLD)
        kept_thumbnails = np.stack([duplicates.thumbnail(batch.read_image(archive, image)) for image in selection.kept])
    removed = {(r.image.split, r.image.name): r for r in selection.removals}
    kept = {(image.split, image.name) for image in selection.kept}
    middle = removed[("valid", CHAIN["valid"])]
    assert (middle.reason, middle.kept.split, middle.kept.name) == (duplicates.NEAR_COPY, "test", CHAIN["test"])
    assert middle.distance == pytest.approx(measure("valid", "test"))
    assert ("train", CHAIN["train"]) in kept and ("test", CHAIN["test"]) in kept
    # exactly these photos leave: the chains take the two photos of valid, and no photo of train
    in_the_middle = {("valid", CHAIN["valid"]), ("valid", CHAIN_EXACT["valid"])}
    assert set(removed) == EXACT_REMOVED | NEAR_REMOVED | in_the_middle
    # every near copy that left is within the threshold of a photo that remains ...
    near = [r for r in selection.removals if r.reason == duplicates.NEAR_COPY]
    assert all((r.kept.split, r.kept.name) in kept and r.distance <= NEAR_THRESHOLD for r in near)
    # ... and no two photos that remain, in different splits, are within the threshold
    assert duplicates.cross_split_pairs(selection.kept, kept_thumbnails, NEAR_THRESHOLD) == []


# --- the raw copy ---------------------------------------------------------------------------------


def test_raw_copy_holds_the_archive_bytes_and_boxes_only_labels(export, tmp_path):
    dest = tmp_path / "dest"
    selection, stats = build_raw(export, dest)
    entries = export_entries()
    listing = batch.list_copy(dest / "cru")
    assert {split: len(names) for split, names in listing.items()} == {"train": 5, "valid": 4, "test": 4}
    assert stats["images"] == {"train": 5, "valid": 4, "test": 4}
    hashes = set()
    for split, names in listing.items():
        for name in names:
            data = (dest / "cru" / split / "images" / name).read_bytes()
            assert data == entries[f"{split}/images/{name}"]
            hashes.add(batch.sha256_bytes(data))
            label = (dest / "cru" / split / "labels" / batch.label_name(name)).read_bytes()
            if name[0] not in "fg":
                assert label == entries[f"{split}/labels/{batch.label_name(name)}"]  # untouched, no newline added
    assert len(hashes) == 13
    mixed = (dest / "cru" / "train" / "labels" / "f_jpg.rf.1.txt").read_text()
    assert mixed.startswith(BOX_ROWS + "\n1 ") and [len(row) for row in batch.label_rows(mixed)] == [5, 5, 5]
    only_polygons = (dest / "cru" / "train" / "labels" / "g_jpg.rf.1.txt").read_text()
    rows = batch.label_rows(only_polygons)
    assert [row[0] for row in rows] == ["1", "0"] and all(len(row) == 5 for row in rows)
    assert (stats["polygon_rows_converted"], stats["label_files_with_polygons"]) == (3, 2)
    # the copy that stays keeps its own label
    assert (dest / "cru" / "train" / "labels" / "a_jpg.rf.1.txt").read_text() == BOX_ROWS
    assert (dest / "cru" / "test" / "labels" / "c_jpg.rf.3.txt").read_text() == BOX_ROWS
    assert selection.kept and not (dest / "cru" / "train" / "images" / "a_jpg.rf.2.jpg").exists()


def test_raw_copy_data_yaml_is_relative_and_keeps_names_and_license(export, tmp_path):
    build_raw(export, tmp_path / "dest")
    config = yaml.safe_load((tmp_path / "dest" / "cru" / "data.yaml").read_text())
    assert config == {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 10,
        "names": NAMES,
        "roboflow": {"workspace": "synthetic", "project": "ppe", "version": 1, "license": "CC BY 4.0"},
    }
    with pytest.raises(batch.BatchError, match="0..n-1"):
        batch.dataset_yaml({"names": {1: "helmet"}})


def test_raw_copy_refuses_a_destination_with_files_and_changes_nothing(export, tmp_path):
    dest = tmp_path / "dest"
    dest.mkdir()
    build_raw(export, dest)  # an empty folder is fine
    before = tree(dest)
    with pytest.raises(batch.DestinationNotEmptyError, match="never deletes"):
        build_raw(export, dest)
    other = tmp_path / "other"
    other.mkdir()
    (other / "notas.txt").write_text("minhas")
    with pytest.raises(batch.DestinationNotEmptyError):
        build_raw(export, other)
    assert tree(dest) == before and tree(other) == {"notas.txt": b"minhas"}
    (tmp_path / "arquivo").write_text("x")
    with pytest.raises(batch.DestinationNotEmptyError):
        build_raw(export, tmp_path / "arquivo")


def test_a_bad_label_stops_the_raw_copy_before_anything_is_written(tmp_path):
    entries = export_entries()
    entries["test/labels/t_jpg.rf.1.txt"] = b"0 0.1 0.2 0.3 0.4 0.5"
    path = write_export(tmp_path / "bad.zip", entries)
    with pytest.raises(batch.BatchError, match="test/labels/t_jpg.rf.1.txt: line 1"):
        build_raw(path, tmp_path / "dest")
    assert not (tmp_path / "dest").exists()


def test_write_new_never_overwrites(tmp_path):
    batch.write_new(tmp_path / "a.bin", b"1")
    with pytest.raises(FileExistsError):
        batch.write_new(tmp_path / "a.bin", b"2")
    assert (tmp_path / "a.bin").read_bytes() == b"1"
    with pytest.raises(batch.DestinationNotEmptyError):
        batch.require_absent([tmp_path / "missing", tmp_path / "a.bin"])
    batch.require_absent([tmp_path / "missing"])


# --- treated copies -------------------------------------------------------------------------------


def test_treated_copy_has_the_same_files_dimensions_and_labels(export, tmp_path):
    dest = tmp_path / "dest"
    build_raw(export, dest)
    raw_before = tree(dest / "cru")
    info = batch.build_treated_copy(dest, "clahe")
    assert tree(dest / "cru") == raw_before  # the raw copy is only read
    raw, treated = dest / "cru", dest / "tratado-clahe"
    assert batch.list_copy(treated) == batch.list_copy(raw)
    for split, names in batch.list_copy(raw).items():
        for name in names:
            source = (raw / split / "images" / name).read_bytes()
            written = (treated / split / "images" / name).read_bytes()
            assert written == pipeline.process_image(source, "clahe", output_format="jpeg").content
            assert written.startswith(intake.JPEG_MAGIC) and written != source
            with Image.open(raw / split / "images" / name) as before, Image.open(treated / split / "images" / name) as after:
                assert after.size == before.size == (160, 120) and after.format == "JPEG"
            label = batch.label_name(name)
            assert (treated / split / "labels" / label).read_bytes() == (raw / split / "labels" / label).read_bytes()
    assert (treated / "data.yaml").read_bytes() == (raw / "data.yaml").read_bytes()
    stored = json.loads((treated / batch.TREATMENT_INFO).read_text())
    assert stored == info
    assert stored["tecnica"] == "clahe" and stored["parametros"] == {"clip_limit": 2.0, "tile_grid": [8, 8]}
    assert stored["jpeg"]["quality"] == 95 and stored["jpeg"]["chroma_subsampling"] == "4:2:0"
    assert stored["imagens"] == {"train": 5, "valid": 4, "test": 4} and stored["limite_pixels"] == batch.BATCH_MAX_PIXELS
    assert stored["bytes_imagens"] == sum(len(data) for path, data in tree(treated).items() if "/images/" in path)
    assert batch.treated_copies(dest) == ["clahe"]


def test_none_copy_is_the_control_decoded_and_reencoded_only(export, tmp_path):
    dest = tmp_path / "dest"
    build_raw(export, dest)
    batch.build_treated_copy(dest, "none")
    name = "t_jpg.rf.1.jpg"
    source = (dest / "cru" / "test" / "images" / name).read_bytes()
    written = (dest / "tratado-none" / "test" / "images" / name).read_bytes()
    assert written == intake.encode_jpeg(intake.decode_image(source).pixels)


def test_the_batch_goes_through_process_image_and_measures_no_quality(export, tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    build_raw(export, dest)
    calls = []
    original = pipeline.process_image

    def spy(*args, **kwargs):
        calls.append((args[1], kwargs))
        result = original(*args, **kwargs)
        assert result.sharpness is None
        return result

    monkeypatch.setattr(pipeline, "process_image", spy)
    batch.build_treated_copy(dest, "gamma", max_pixels=123_456)
    expected = {"output_format": "jpeg", "max_pixels": 123_456, "measure_sharpness": False}
    assert calls == [("gamma", expected)] * 13  # every photo of train, valid and test


def test_treated_copy_refuses_an_existing_target_and_never_overwrites(export, tmp_path):
    dest = tmp_path / "dest"
    build_raw(export, dest)
    batch.build_treated_copy(dest, "gamma")
    before = tree(dest)
    with pytest.raises(batch.DestinationNotEmptyError):
        batch.build_treated_copy(dest, "gamma")
    assert tree(dest) == before
    with pytest.raises(batch.BatchError, match="unknown technique"):
        batch.build_treated_copy(dest, "sepia")
    with pytest.raises(batch.BatchError, match="not a finished copy"):
        batch.build_treated_copy(tmp_path / "nowhere", "gamma")


def test_a_photo_that_fails_stops_the_batch_and_is_not_skipped(export, tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    build_raw(export, dest)
    original = pipeline.process_image
    seen = []

    def failing(data, *args, **kwargs):
        seen.append(data)
        if len(seen) == 3:
            raise MemoryError("Unable to allocate 2.1 GiB")
        return original(data, *args, **kwargs)

    monkeypatch.setattr(pipeline, "process_image", failing)
    message = r"train/images/e_jpg\.rf\.1\.jpg failed \(MemoryError: Unable to allocate"
    with pytest.raises(batch.BatchError, match=message) as error:
        batch.build_treated_copy(dest, "unsharp_mask")
    assert "incomplete" in str(error.value)
    assert len(seen) == 3  # stopped at the failing photo
    target = dest / "tratado-unsharp_mask"
    assert sorted(path.name for path in (target / "train" / "images").iterdir()) == ["a_jpg.rf.1.jpg", "d_jpg.rf.1.jpg"]
    assert not (target / batch.TREATMENT_INFO).exists() and batch.treated_copies(dest) == []  # never counted as finished


def test_the_pixel_limit_belongs_to_the_batch_and_is_checked_before_writing(export, tmp_path):
    dest = tmp_path / "dest"
    build_raw(export, dest)
    with pytest.raises(batch.BatchError, match="13 image.s. cannot be treated; nothing was written") as error:
        batch.build_treated_copy(dest, "none", max_pixels=19_199)
    assert "160x120 = 19200 pixels exceeds the limit of 19199" in str(error.value)
    assert not (dest / "tratado-none").exists()
    assert batch.build_treated_copy(dest, "none", max_pixels=19_200)["limite_pixels"] == 19_200
    assert batch.BATCH_MAX_PIXELS > intake.MAX_PIXELS == 40_000_000  # the endpoint keeps its own limit


def test_an_unreadable_raw_image_is_reported_before_anything_is_written(export, tmp_path):
    dest = tmp_path / "dest"
    build_raw(export, dest)
    (dest / "cru" / "valid" / "images" / "v_jpg.rf.1.jpg").write_bytes(b"\xff\xd8\xff" + bytes(20))
    with pytest.raises(batch.BatchError, match=r"valid/images/v_jpg\.rf\.1\.jpg: cannot read the image header"):
        batch.build_treated_copy(dest, "none")
    assert not (dest / "tratado-none").exists()


def test_list_copy_requires_images_and_labels_to_match(export, tmp_path):
    dest = tmp_path / "dest"
    build_raw(export, dest)
    (dest / "cru" / "test" / "labels" / "t_jpg.rf.1.txt").rename(dest / "cru" / "test" / "labels" / "u.txt")
    with pytest.raises(batch.BatchError, match="do not match one to one"):
        batch.list_copy(dest / "cru")


# --- verification ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A complete build: raw copy without exact and near copies, plus the 8 treated copies."""
    root = tmp_path_factory.mktemp("phase2")
    export = write_export(root / "export.zip")
    dest = root / "dest"
    build_raw(export, dest, NEAR_THRESHOLD)
    for name in techniques.TECHNIQUES:
        batch.build_treated_copy(dest, name)
    return export, dest


def copy_tree(source, target):
    for relative, data in tree(source).items():
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        (target / relative).write_bytes(data)
    return target


def test_verification_passes_on_a_complete_build(built):
    export, dest = built
    with zipfile.ZipFile(export) as archive:
        checks = batch.verify_copies(archive, dest, list(techniques.TECHNIQUES))
    assert len(checks) == 3 + 3 * len(techniques.TECHNIQUES)
    assert all(check.passed for check in checks), [check for check in checks if not check.passed]
    assert "11 images, 11 distinct SHA-256" in checks[0].detail


def test_verification_catches_what_would_invalidate_the_comparison(built, tmp_path):
    export, original = built
    dest = copy_tree(original, tmp_path / "dest")
    raw = dest / "cru"
    with (dest / "tratado-gamma" / "valid" / "labels" / "v_jpg.rf.1.txt").open("a") as handle:
        handle.write("\n0 0.5 0.5 0.1 0.1")  # a label that differs from the raw one
    (dest / "tratado-clahe" / "test" / "images" / "t_jpg.rf.1.jpg").write_bytes(jpeg(photo(11, 60, 80)))  # other size
    (dest / "tratado-none" / "train" / "images" / "a_jpg.rf.1.jpg").unlink()  # a photo less
    (dest / "tratado-none" / "train" / "labels" / "a_jpg.rf.1.txt").unlink()
    (dest / "tratado-gaussian_blur" / batch.TREATMENT_INFO).unlink()  # not treated to the end
    (raw / "train" / "images" / "copy.jpg").write_bytes((raw / "train" / "images" / "a_jpg.rf.1.jpg").read_bytes())
    (raw / "train" / "labels" / "copy.txt").write_text(f"{BOX_ROWS}\n{POLYGON_ROW}")
    with zipfile.ZipFile(export) as archive:
        checks = {check.name: check for check in batch.verify_copies(archive, dest, list(techniques.TECHNIQUES))}
    failed = {name for name, check in checks.items() if not check.passed}
    assert failed == {
        "cru: no image SHA-256 repeats",
        "cru: every image is byte-identical to the archive entry of the same split and name",
        "cru: every label is the archive's label with polygon rows turned into boxes, nothing else changed",
        *[f"tratado-{name}: same file list as cru in every split" for name in techniques.TECHNIQUES],
    }
    assert "is missing" in checks["tratado-gaussian_blur: same file list as cru in every split"].detail

    dest = copy_tree(original, tmp_path / "dest2")
    with (dest / "tratado-gamma" / "valid" / "labels" / "v_jpg.rf.1.txt").open("a") as handle:
        handle.write("\n0 0.5 0.5 0.1 0.1")
    (dest / "tratado-clahe" / "test" / "images" / "t_jpg.rf.1.jpg").write_bytes(jpeg(photo(11, 60, 80)))
    (dest / "tratado-fft_lowpass" / "test" / "images" / "t_jpg.rf.1.jpg").write_bytes(intake.encode_png(photo(11)))
    with zipfile.ZipFile(export) as archive:
        failed = {check.name for check in batch.verify_copies(archive, dest, list(techniques.TECHNIQUES)) if not check.passed}
    assert failed == {
        "tratado-gamma: labels identical to cru, byte for byte",
        "tratado-clahe: every image is a JPEG with the dimensions of the raw one",
        "tratado-fft_lowpass: every image is a JPEG with the dimensions of the raw one",
    }


def test_selection_on_disk_must_match_the_plan(built):
    export, dest = built
    with zipfile.ZipFile(export) as archive:
        images, _ = batch.read_archive_index(archive)
        assert batch.check_selection_on_disk(batch.plan_selection(archive, images, NEAR_THRESHOLD), dest) == []
        problems = batch.check_selection_on_disk(batch.plan_selection(archive, images), dest)
    assert problems == ["train: 0 file(s) on disk that the selection removes, 2 file(s) of the selection missing on disk"]


def test_ultralytics_reads_every_copy_and_the_scan_leaves_no_cache(built, tmp_path, monkeypatch):
    pytest.importorskip("ultralytics")
    monkeypatch.setattr("ultralytics.data.utils.check_font", lambda *args, **kwargs: None)  # no font download in tests
    export, original = built
    before = tree(original)
    for copy_name in ("cru", "tratado-none", "tratado-fft_high_emphasis"):
        scans = batch.ultralytics_scan(original / copy_name / "data.yaml")
        assert {key: (scan["files"], scan["kept"], scan["corrupt"]) for key, scan in scans.items()} == {
            "train": (3, 3, 0), "val": (4, 4, 0), "test": (4, 4, 0),
        }  # fmt: skip
        assert scans["train"]["boxes"] == 2 + 3 + 2 and scans["train"]["messages"] == []
    assert tree(original) == before  # no labels.cache, nothing touched

    # The same reading does drop a photo whose label still mixes a polygon with boxes: the check can fail.
    dest = copy_tree(original / "cru", tmp_path / "unconverted")
    (dest / "train" / "labels" / "f_jpg.rf.1.txt").write_bytes(export_entries()["train/labels/f_jpg.rf.1.txt"])
    scan = batch.ultralytics_scan(dest / "data.yaml")["train"]
    assert (scan["files"], scan["kept"], scan["corrupt"]) == (3, 2, 1)
    assert "labels mix segment and detection rows" in scan["messages"][0]


# --- the script -----------------------------------------------------------------------------------


def test_script_builds_reports_finishes_and_verifies(export, tmp_path, capsys):
    script = load_script("build_phase2_dataset")
    dest = tmp_path / "fase2"
    common = ["--dest", str(dest)]
    assert script.main(["raw", "--zip", str(export), *common]) == 0
    assert "exact copies left out: train 3, valid 1, test 0" in capsys.readouterr().out
    assert script.main(["near-copies", "--threshold", str(NEAR_THRESHOLD), *common]) == 0
    out = capsys.readouterr().out
    assert "would leave train 2, valid 0, test 0" in out and "<- proposed" in out
    assert (dest / batch_report.NEAR_COPY_SHEET).read_bytes().startswith(intake.PNG_MAGIC)
    assert script.main(["treat", "--techniques", "none", "clahe", *common]) == 0
    assert script.main(["treat", "--techniques", "none", *common]) == 2  # refuses, deletes nothing
    assert "already has files" in capsys.readouterr().err
    assert script.main(["finish", "--zip", str(export), *common]) == 2  # six treated copies still missing
    remaining = [name for name in techniques.TECHNIQUES if name not in ("none", "clahe")]
    assert script.main(["treat", "--techniques", *remaining, *common]) == 0
    assert script.main(["verify", "--zip", str(export), *common]) == 1  # the manifest does not exist yet
    assert script.main(["finish", "--zip", str(export), *common]) == 0
    capsys.readouterr()
    assert script.main(["verify", "--zip", str(export), *common]) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and f"{3 + 3 * 8 + 2 + 2} of {3 + 3 * 8 + 2 + 2} checks passed" in out
    for name in (batch_report.MANIFEST, batch_report.REMOVED, batch_report.README, batch_report.SAMPLES):
        assert (dest / name).is_file()
    before = tree(dest)
    assert script.main(["raw", "--zip", str(export), *common]) == 2
    assert script.main(["finish", "--zip", str(export), *common]) == 2
    assert tree(dest) == before


def test_script_with_threshold_builds_the_final_raw_copy_in_one_go(export, tmp_path, capsys):
    script = load_script("build_phase2_dataset")
    dest = tmp_path / "fase2"
    assert script.main(["raw", "--zip", str(export), "--dest", str(dest), "--near-copy-threshold", str(NEAR_THRESHOLD)]) == 0
    out = capsys.readouterr().out
    assert f"near copies left out (threshold {NEAR_THRESHOLD:g}): train 2, valid 0, test 0" in out
    assert {split: len(names) for split, names in batch.list_copy(dest / "cru").items()} == {"train": 3, "valid": 4, "test": 4}


def test_script_near_copies_reports_zero_pairs_when_no_near_copy_is_left(export, tmp_path, capsys):
    script = load_script("build_phase2_dataset")
    common = ["--dest", str(tmp_path / "fase2")]
    assert script.main(["raw", "--zip", str(export), "--near-copy-threshold", str(NEAR_THRESHOLD), *common]) == 0
    capsys.readouterr()
    assert script.main(["near-copies", "--threshold", str(NEAR_THRESHOLD), *common]) == 0  # not a traceback
    out = capsys.readouterr().out
    assert "cross-split pairs with measure <= 8: 0" in out
    assert "would leave train 0, valid 0, test 0; would stay train 3, valid 4, test 4" in out
    assert "contact sheet (0 pairs)" in out


COPIES = ["cru", *[f"tratado-{name}" for name in techniques.TECHNIQUES]]


@pytest.fixture(scope="module")
def finished_builds(tmp_path_factory):
    """Two finished builds, manifest included: exact copies only, and near copies removed too."""
    root = tmp_path_factory.mktemp("phase2-verify")
    export = write_export(root / "export.zip")
    builds = {}
    for threshold in (None, NEAR_THRESHOLD):
        dest = root / f"dest-{threshold}"
        build_raw(export, dest, threshold)
        for name in techniques.TECHNIQUES:
            batch.build_treated_copy(dest, name)
        with zipfile.ZipFile(export) as archive:
            batch_report.finish(archive, export.name, "0" * 64, dest, threshold)
        builds[threshold] = dest
    return export, builds


def run_verify(export, dest, capsys, *options):
    """Exit code of `verify` and the names of the checks that failed."""
    script = load_script("build_phase2_dataset")
    code = script.main(["verify", "--zip", str(export), "--dest", str(dest), *options])
    lines = capsys.readouterr().out.splitlines()
    return code, [line.removeprefix("FAIL  ") for line in lines if line.startswith("FAIL")]


def test_verify_fails_when_every_data_yaml_swaps_two_class_names(finished_builds, tmp_path, capsys):
    """Images, labels and manifest untouched, but class 3 is called no-helmet and class 7 helmet in the 9 copies."""
    export, builds = finished_builds
    dest = copy_tree(builds[None], tmp_path / "dest")
    assert run_verify(export, dest, capsys) == (0, [])
    for copy_name in COPIES:
        path = dest / copy_name / "data.yaml"
        config = yaml.safe_load(path.read_text())
        helmet, no_helmet = config["names"].index("helmet"), config["names"].index("no-helmet")
        config["names"][helmet], config["names"][no_helmet] = "no-helmet", "helmet"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
    code, failed = run_verify(export, dest, capsys)
    assert code == 1 and len(failed) == 1
    assert failed[0].startswith("data.yaml of every copy has the class names and split paths of the archive: 0 of 9")
    assert "class 3 is 'no-helmet', the archive says 'helmet'" in failed[0]


def test_verify_fails_when_one_data_yaml_points_a_split_at_other_photos(finished_builds, tmp_path, capsys):
    export, builds = finished_builds
    dest = copy_tree(builds[None], tmp_path / "dest")
    path = dest / "cru" / "data.yaml"
    path.write_text(path.read_text().replace("val: valid/images", "val: test/images"))
    code, failed = run_verify(export, dest, capsys)
    assert code == 1
    assert any(name.startswith("data.yaml of every copy") and "8 of 9" in name and "cru: val is" in name for name in failed)


def test_verify_fails_when_a_photo_is_missing_from_every_copy_and_the_manifest_was_written_again(
    finished_builds, tmp_path, capsys
):
    """The copies agree with each other and with the manifest, but not with the archive."""
    export, builds = finished_builds
    dest = copy_tree(builds[None], tmp_path / "dest")
    for copy_name in COPIES:
        (dest / copy_name / "train" / "images" / "f_jpg.rf.1.jpg").unlink()
        (dest / copy_name / "train" / "labels" / "f_jpg.rf.1.txt").unlink()
    (dest / batch_report.MANIFEST).unlink()
    rows = batch_report.manifest_rows(dest, list(techniques.TECHNIQUES))
    (dest / batch_report.MANIFEST).write_bytes(batch_report._csv_bytes(batch_report.MANIFEST_HEADER, rows))
    code, failed = run_verify(export, dest, capsys)
    assert code == 1 and len(failed) == 1
    assert failed[0].startswith("cru: holds exactly the photos that the selection from the archive keeps")
    assert "train: 0 file(s) on disk that the selection removes, 1 file(s) of the selection missing on disk" in failed[0]


def test_verify_fails_when_a_removed_photo_is_back_in_every_copy(finished_builds, tmp_path, capsys):
    """An exact copy that the selection leaves out, put back in the 9 copies with a manifest that lists it."""
    export, builds = finished_builds
    dest = copy_tree(builds[None], tmp_path / "dest")
    entries = export_entries()
    for copy_name in COPIES:
        source = dest / copy_name / "valid"
        (dest / copy_name / "train" / "images" / "b_jpg.rf.1.jpg").write_bytes(
            (source / "images" / "b_jpg.rf.7.jpg").read_bytes()
        )
        (dest / copy_name / "train" / "labels" / "b_jpg.rf.1.txt").write_bytes(entries["train/labels/b_jpg.rf.1.txt"])
    (dest / batch_report.MANIFEST).unlink()
    rows = batch_report.manifest_rows(dest, list(techniques.TECHNIQUES))
    (dest / batch_report.MANIFEST).write_bytes(batch_report._csv_bytes(batch_report.MANIFEST_HEADER, rows))
    code, failed = run_verify(export, dest, capsys)
    assert code == 1
    assert any("train: 1 file(s) on disk that the selection removes, 0 file(s)" in name for name in failed)


def test_verify_recomputes_the_selection_with_the_near_copy_threshold_it_is_given(finished_builds, capsys):
    export, builds = finished_builds
    threshold = ["--near-copy-threshold", str(NEAR_THRESHOLD)]
    assert run_verify(export, builds[NEAR_THRESHOLD], capsys, *threshold) == (0, [])
    assert run_verify(export, builds[None], capsys) == (0, [])
    # the threshold has to be the one the raw copy was built with
    code, failed = run_verify(export, builds[NEAR_THRESHOLD], capsys)
    assert code == 1 and len(failed) == 1
    assert "near-copy threshold: none" in failed[0] and "2 file(s) of the selection missing on disk" in failed[0]
    assert "--near-copy-threshold" in failed[0]
    code, failed = run_verify(export, builds[None], capsys, *threshold)
    assert code == 1 and len(failed) == 1
    assert f"near-copy threshold: {NEAR_THRESHOLD:g}" in failed[0]
    assert "train: 2 file(s) on disk that the selection removes" in failed[0]


def test_script_never_writes_into_the_repository_and_reports_missing_input(tmp_path, capsys):
    script = load_script("build_phase2_dataset")
    assert script.main(["raw", "--zip", str(tmp_path / "none.zip"), "--dest", str(REPO / "runs" / "fase2")]) == 2
    assert "inside the repository" in capsys.readouterr().err
    assert not (REPO / "runs").exists()
    assert script.main(["raw", "--zip", str(tmp_path / "none.zip"), "--dest", str(tmp_path / "dest")]) == 2
    assert not (tmp_path / "dest").exists()
    assert script.main(["treat", "--dest", str(tmp_path / "dest")]) == 2
    assert script.DEFAULT_DEST.name == "epi-dataset-fase2" and not script.DEFAULT_DEST.is_relative_to(REPO)
