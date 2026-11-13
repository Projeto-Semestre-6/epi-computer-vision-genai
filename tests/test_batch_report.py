import csv
import zipfile

import cv2
import numpy as np
import pytest

from epi_imgproc import batch, batch_report, duplicates, techniques
from epi_imgproc.duplicates import NearPair
from phase2_data import CHAIN, CHAIN_EXACT, NAMES, NEAR_THRESHOLD, chain_entries, tree, write_export
from test_batch import build_raw


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def png_shape(path):
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR).shape


def test_pick_across_range_spreads_pairs_over_the_measure():
    distances = [0.0, 0.1, 0.2, 0.3, 1.9, 2.1, 3.0, 4.2, 7.9]
    nearest = [NearPair(i, 100 + i, distance, "identity") for i, distance in enumerate(distances)]
    picked = batch_report.pick_across_range(nearest, 5, 8.0)  # grid 0, 2, 4, 6, 8; each pair is used once
    assert [pair.distance for pair in picked] == [0.0, 1.9, 3.0, 4.2, 7.9]  # sorted by measure
    assert batch_report.pick_across_range(nearest[:2], 5, 8.0) == nearest[:2]  # fewer pairs than asked: all of them
    assert batch_report.pick_across_range([], 5, 8.0) == []


@pytest.fixture
def raw_only(tmp_path):
    export = write_export(tmp_path / "export.zip")
    dest = tmp_path / "dest"
    build_raw(export, dest)
    return export, dest


def test_near_copy_report_writes_sheet_and_pairs_and_counts_what_would_leave(raw_only):
    _, dest = raw_only
    before = tree(dest / "cru")
    report = batch_report.near_copy_report(dest, NEAR_THRESHOLD, max_distance=8.0, sheet_pairs=40)
    assert tree(dest / "cru") == before  # the report changes nothing in the copy
    assert report["images"] == {"train": 5, "valid": 4, "test": 4}
    assert report["removed"] == {"train": 2} and report["remaining"] == {"train": 3, "valid": 4, "test": 4}
    assert report["removed_by_symmetry"] == {"nenhuma": 1, "espelhada": 1}
    assert report["removed_by_threshold"][NEAR_THRESHOLD] == {"train": 2, "valid": 0, "test": 0}
    assert report["removed_by_threshold"][8.0] == {"train": 2, "valid": 0, "test": 0}
    assert report["class_names"] == NAMES
    assert report["boxes_before"]["train"]["helmet"] == 4 and report["boxes_before"]["train"]["gloves"] == 2
    lost = report["boxes_lost"]["train"]
    assert (lost["helmet"], lost["vest"], sum(lost.values())) == (2, 2, 4)
    assert sum(report["boxes_lost"]["valid"].values()) == 0
    assert report["kept_because_the_only_near_copies_also_leave"] == 0

    rows = read_csv(dest / batch_report.NEAR_COPY_PAIRS)
    assert len(rows) == report["pairs_measured"] == 2
    assert [(r["split_a"], r["arquivo_a"], r["split_b"], r["arquivo_b"], r["transformacao"]) for r in rows] == sorted(
        [
            ("train", "d_jpg.rf.1.jpg", "valid", "d_jpg.rf.9.jpg", "nenhuma"),
            ("train", "e_jpg.rf.1.jpg", "test", "e_jpg.rf.3.jpg", "espelhada"),
        ],
        key=lambda row: next(float(r["medida"]) for r in rows if r["arquivo_a"] == row[1]),
    )
    assert all(0 < float(r["medida"]) <= NEAR_THRESHOLD and r["b_e_o_mais_proximo_de_a"] == "1" for r in rows)
    assert sorted(r["par_na_folha"] for r in rows) == ["1", "2"]
    height, width, _ = png_shape(dest / batch_report.NEAR_COPY_SHEET)
    assert width > 5 * 2 * 224 and height > 224  # one row of five pair tiles, plus the title


def test_near_copy_report_refuses_to_overwrite(raw_only):
    _, dest = raw_only
    batch_report.near_copy_report(dest, NEAR_THRESHOLD)
    before = tree(dest)
    with pytest.raises(batch.DestinationNotEmptyError):
        batch_report.near_copy_report(dest, 1.0)
    assert tree(dest) == before
    with pytest.raises(batch.BatchError, match="above the largest distance"):
        batch_report.near_copy_report(dest, 9.0, max_distance=8.0)


def test_near_copy_report_without_any_pair_across_splits_still_writes_its_files(tmp_path):
    """A raw copy whose near copies are already out is ordinary input, not an error."""
    export = write_export(tmp_path / "export.zip")
    dest = tmp_path / "dest"
    build_raw(export, dest, NEAR_THRESHOLD)
    before = tree(dest / "cru")
    report = batch_report.near_copy_report(dest, NEAR_THRESHOLD, max_distance=8.0, sheet_pairs=40)
    assert tree(dest / "cru") == before
    assert (report["pairs_measured"], report["sheet_pairs"]) == (0, 0)
    assert report["removed"] == {} and report["remaining"] == {"train": 3, "valid": 4, "test": 4}
    assert all(counts == {"train": 0, "valid": 0, "test": 0} for counts in report["removed_by_threshold"].values())
    assert report["kept_because_the_only_near_copies_also_leave"] == 0
    assert sum(sum(lost.values()) for lost in report["boxes_lost"].values()) == 0
    assert (dest / batch_report.NEAR_COPY_PAIRS).read_text() == ",".join(batch_report.NEAR_COPY_HEADER) + "\n"
    height, width, _ = png_shape(dest / batch_report.NEAR_COPY_SHEET)
    assert width > 5 * 2 * 224 and height > 3 * 24  # the three title lines, as wide as a row of five pair tiles


def test_near_copy_sheet_without_pairs_is_the_title_over_an_empty_row():
    def never(image):
        raise AssertionError("there is no photo to load")

    sheet = batch_report.near_copy_sheet([], never, threshold=2.0, max_distance=8.0, columns=2, cell=64)
    assert sheet.dtype == np.uint8 and sheet.shape[2] == 3
    assert sheet.shape[1] == 2 * (2 * 64 + 6) + 3 * 12  # the width of a sheet with pairs
    assert sheet.shape[0] == 3 * 24 + (24 + 64) + 12  # title lines, one blank row of tiles, bottom margin


def test_near_copy_sheet_marks_what_leaves():
    images = [duplicates.SourceImage("train", "a.jpg", "1"), duplicates.SourceImage("test", "b.jpg", "2")]
    picture = np.full((30, 60, 3), 200, np.uint8)
    entries = [
        (NearPair(0, 1, 0.5, "mirror"), images[0], images[1], True),
        (NearPair(0, 1, 5.0, "identity"), images[0], images[1], False),
    ]
    sheet = batch_report.near_copy_sheet(entries, lambda image: picture, threshold=2.0, max_distance=8.0, columns=2, cell=64)
    assert sheet.dtype == np.uint8 and sheet.shape[2] == 3
    assert sheet.shape[1] == 2 * (2 * 64 + 6) + 3 * 12
    red = (sheet[..., 2] == 255) & (sheet[..., 0] == 190)
    assert red.any() and red[:, : sheet.shape[1] // 2].sum() == red.sum()  # only the first tile's bar is red


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase2-report")
    export = write_export(root / "export.zip")
    dest = root / "dest"
    build_raw(export, dest, NEAR_THRESHOLD)
    for name in techniques.TECHNIQUES:
        batch.build_treated_copy(dest, name)
    with zipfile.ZipFile(export) as archive:
        result = batch_report.finish(archive, export.name, "0" * 64, dest, NEAR_THRESHOLD)
    return export, dest, result


def test_finish_writes_one_manifest_row_per_image_of_every_copy(finished):
    _, dest, result = finished
    rows = read_csv(dest / batch_report.MANIFEST)
    assert len(rows) == result["manifest_rows"] == 9 * 11
    assert list(rows[0]) == batch_report.MANIFEST_HEADER
    assert [row["copia"] for row in rows[::11]] == ["cru", *[f"tratado-{name}" for name in techniques.TECHNIQUES]]
    assert {row["tecnica"] for row in rows if row["copia"] == "cru"} == {""}
    assert {row["tecnica"] for row in rows if row["copia"] == "tratado-fft_lowpass"} == {"fft_lowpass"}
    for row in rows:
        path = dest / row["copia"] / row["split"] / "images" / row["arquivo"]
        assert row["sha256"] == batch.sha256_bytes(path.read_bytes()) and int(row["bytes"]) == path.stat().st_size
        assert (row["largura"], row["altura"]) == ("160", "120")
    by_file = {}
    for row in rows:
        by_file.setdefault((row["split"], row["arquivo"]), set()).add((row["rotulo_sha256"], row["caixas"]))
    assert len(by_file) == 11 and all(len(labels) == 1 for labels in by_file.values())  # same label in the 9 copies
    assert next(row["caixas"] for row in rows if row["arquivo"] == "f_jpg.rf.1.jpg") == "3"
    assert len({row["sha256"] for row in rows if row["copia"] == "cru"}) == 11


def test_finish_lists_every_removed_photo_with_reason_pair_and_label_divergence(finished):
    _, dest, result = finished
    rows = read_csv(dest / batch_report.REMOVED)
    assert list(rows[0]) == batch_report.REMOVED_HEADER
    columns = ("motivo", "split_do_par", "par", "transformacao", "rotulos_divergiam", "divergencia")
    table = {(r["split"], r["arquivo"]): tuple(r[column] for column in columns) for r in rows}
    assert table == {
        ("train", "a_jpg.rf.2.jpg"): ("copia_exata", "train", "a_jpg.rf.1.jpg", "", "sim", "classes"),
        ("train", "b_jpg.rf.1.jpg"): ("copia_exata", "valid", "b_jpg.rf.7.jpg", "", "nao", "iguais"),
        ("train", "c_jpg.rf.1.jpg"): ("copia_exata", "test", "c_jpg.rf.3.jpg", "", "sim", "coordenadas"),
        ("train", "d_jpg.rf.1.jpg"): ("quase_copia", "valid", "d_jpg.rf.9.jpg", "nenhuma", "nao", "iguais"),
        ("train", "e_jpg.rf.1.jpg"): ("quase_copia", "test", "e_jpg.rf.3.jpg", "espelhada", "nao", "iguais"),
        ("valid", "w_jpg.rf.1.jpg"): ("copia_exata", "test", "w_jpg.rf.0.jpg", "", "nao", "iguais"),
    }
    assert [r["medida"] == "" for r in rows] == [r["motivo"] == "copia_exata" for r in rows]
    assert result["removed_rows"] == 6
    assert result["removed"] == {("copia_exata", "train"): 3, ("copia_exata", "valid"): 1, ("quase_copia", "train"): 2}
    assert {r["par_permanece"] for r in rows} == {"sim"}  # no chain in this export: every pair is in the dataset


@pytest.fixture(scope="module")
def chained(tmp_path_factory):
    """A finished build of the export with the two chains across the splits (phase2_data.chain_entries)."""
    root = tmp_path_factory.mktemp("phase2-chain")
    export = write_export(root / "export.zip", chain_entries())
    dest = root / "dest"
    build_raw(export, dest, NEAR_THRESHOLD)
    for name in techniques.TECHNIQUES:
        batch.build_treated_copy(dest, name)
    with zipfile.ZipFile(export) as archive:
        result = batch_report.finish(archive, export.name, "0" * 64, dest, NEAR_THRESHOLD)
    return export, dest, result


def test_removed_list_says_whether_the_pair_remains_in_the_dataset(chained):
    """train b is an exact copy of valid b, and valid b then leaves as a near copy of test b."""
    _, dest, _ = chained
    rows = read_csv(dest / batch_report.REMOVED)
    assert list(rows[0]) == batch_report.REMOVED_HEADER
    assert "par_que_ficou" not in rows[0]  # the old name said the pair stayed, which was false for chains
    on_disk = {(split, name) for split, names in batch.list_copy(dest / "cru").items() for name in names}
    assert all((r["split"], r["arquivo"]) not in on_disk for r in rows)
    for row in rows:  # the column says what is on disk
        assert row["par_permanece"] == ("sim" if (row["split_do_par"], row["par"]) in on_disk else "nao"), row
    by_file = {(r["split"], r["arquivo"]): r for r in rows}
    gone = [key for key, row in by_file.items() if row["par_permanece"] == "nao"]
    assert gone == [("train", CHAIN_EXACT["train"])]
    exact = by_file[("train", CHAIN_EXACT["train"])]
    assert (exact["motivo"], exact["split_do_par"], exact["par"]) == ("copia_exata", "valid", CHAIN_EXACT["valid"])
    assert (exact["rotulos_divergiam"], exact["divergencia"]) == ("nao", "iguais")  # compared with the pair's label
    # the pair that left has its own row, which leads to a photo that remains
    follow = by_file[(exact["split_do_par"], exact["par"])]
    assert (follow["motivo"], follow["split_do_par"], follow["par"]) == ("quase_copia", "test", CHAIN_EXACT["test"])
    assert follow["par_permanece"] == "sim"
    assert {r["par_permanece"] for r in rows if r["motivo"] == "quase_copia"} == {"sim"}
    # which photos leave is unchanged: the chain keeps train p
    assert ("train", CHAIN["train"]) in on_disk and ("valid", CHAIN["valid"]) in by_file
    assert len(rows) == 8 and len(on_disk) == 13


def test_readme_states_the_near_copy_rule_and_counts_the_pairs_that_left(chained, finished):
    _, dest, _ = chained
    text = (dest / batch_report.README).read_text(encoding="utf-8")
    assert (
        "Regra: sai a foto que está a até 3,00 de uma foto de split de maior prioridade (test > valid > train) "
        "que permanece no dataset."
    ) in text
    assert "a foto de valid sai e a de train continua" in text  # the chain, spelled out
    assert "nenhum par de splits diferentes está dentro do limiar" in text
    assert "(nenhum arquivo do grupo está no dataset): 1." in text
    assert "`par_permanece` diz se o par está no dataset" in text
    assert "Linhas com `nao`: 1 de `copia_exata` e 0 de `quase_copia`" in text
    assert "compara o rótulo da foto removida com o do par" in text
    assert "par_que_ficou" not in text and "com o da que ficou" not in text
    assert "Quase-cópias entre splits: 4 (2 de train, 2 de valid, 0 de test), com limiar 3,00" in text

    plain = (finished[1] / batch_report.README).read_text(encoding="utf-8")  # the export without chains
    assert "(nenhum arquivo do grupo está no dataset): 0." in plain
    assert "Nenhuma linha tem `nao`" in plain and "Linhas com `nao`" not in plain


def test_finish_writes_readme_with_technique_and_jpeg_parameters(finished):
    _, dest, result = finished
    text = (dest / batch_report.README).read_text(encoding="utf-8")
    for name, technique in techniques.TECHNIQUES.items():
        assert f"`tratado-{name}` | `{technique.function.__name__}`" in text
    assert "clip_limit=2.0, tile_grid=[8, 8]" in text and "cutoff=0.1, k1=1.0, k2=0.5" in text and "gamma=0.8" in text
    assert "Qualidade 95; subamostragem de croma 4:2:0; baseline (não progressivo)" in text
    assert f"{batch.BATCH_MAX_PIXELS} pixels por foto (o serviço mantém 40000000)" in text
    assert "export.zip" in text and "0" * 64 in text and "CC BY 4.0" in text
    assert "Cópias exatas (mesmo SHA-256): 4 (3 de train, 1 de valid, 0 de test)" in text
    assert "Quase-cópias entre splits: 2 (2 de train, 0 de valid, 0 de test), com limiar 3,00" in text
    assert "| train | 3 | 7 |" in text and "| total | 11 | 23 |" in text
    assert "3 linhas em formato de polígono, em 2 arquivos" in text
    assert (result["polygon_rows_converted"], result["label_files_with_polygons"]) == (3, 2)
    assert all(f"`{name}`" in text for name in result["samples"])
    assert "quase-copias.png" not in text  # the report was not run for this build


def test_readme_mentions_the_near_copy_sheet_when_it_exists(tmp_path):
    export = write_export(tmp_path / "export.zip")
    dest = tmp_path / "dest"
    build_raw(export, dest)
    batch_report.near_copy_report(dest, NEAR_THRESHOLD)
    for name in techniques.TECHNIQUES:
        batch.build_treated_copy(dest, name)
    with zipfile.ZipFile(export) as archive:
        batch_report.finish(archive, export.name, "0" * 64, dest, None)
    assert "`quase-copias.png` e `quase-copias.csv`" in (dest / batch_report.README).read_text(encoding="utf-8")


def test_finish_writes_samples_of_validation_photos_in_nine_columns(finished):
    _, dest, result = finished
    assert result["samples"] == batch.list_copy(dest / "cru")["valid"]  # 4 validation photos here, 6 asked
    height, width, _ = png_shape(dest / batch_report.SAMPLES)
    assert width == 9 * 256 + 8 * 4
    assert height == 24 + 4 * (256 + 4)
    sheet, names = batch_report.samples_sheet(dest, list(techniques.TECHNIQUES), count=2, cell=64)
    assert len(names) == 2 and sheet.shape == (24 + 2 * 68, 9 * 64 + 8 * 4, 3)
    raw_tile, blurred_tile = sheet[28:92, 0:64], sheet[28:92, 4 * 68 : 4 * 68 + 64]
    assert not np.array_equal(raw_tile, blurred_tile)  # the copies really differ


def test_finish_never_overwrites_and_manifest_check_detects_changes(finished, tmp_path):
    export, dest, _ = finished
    before = tree(dest)
    with zipfile.ZipFile(export) as archive, pytest.raises(batch.DestinationNotEmptyError):
        batch_report.finish(archive, export.name, "0" * 64, dest, NEAR_THRESHOLD)
    assert tree(dest) == before
    names = list(techniques.TECHNIQUES)
    assert batch_report.check_manifest(dest, names).passed
    assert not batch_report.check_manifest(dest, names[:-1]).passed
    copy = tmp_path / "copy"
    for relative, data in before.items():
        (copy / relative).parent.mkdir(parents=True, exist_ok=True)
        (copy / relative).write_bytes(data)
    target = copy / "tratado-gamma" / "valid" / "images" / "v_jpg.rf.1.jpg"
    target.write_bytes(target.read_bytes() + b"\x00")
    assert not batch_report.check_manifest(copy, names).passed
    (copy / batch_report.MANIFEST).unlink()
    assert not batch_report.check_manifest(copy, names).passed


def test_finish_needs_every_copy_and_the_selection_that_is_on_disk(tmp_path):
    export = write_export(tmp_path / "export.zip")
    dest = tmp_path / "dest"
    build_raw(export, dest)  # exact copies only
    for name in list(techniques.TECHNIQUES)[:-1]:
        batch.build_treated_copy(dest, name)
    with zipfile.ZipFile(export) as archive:
        with pytest.raises(batch.BatchError, match="treated copies not finished: fft_high_emphasis"):
            batch_report.finish(archive, export.name, "0" * 64, dest, None)
        batch.build_treated_copy(dest, "fft_high_emphasis")
        with pytest.raises(batch.BatchError, match="does not hold the photos this selection keeps"):
            batch_report.finish(archive, export.name, "0" * 64, dest, NEAR_THRESHOLD)
        outputs = (batch_report.MANIFEST, batch_report.REMOVED, batch_report.README, batch_report.SAMPLES)
        assert not any((dest / name).exists() for name in outputs)
        result = batch_report.finish(archive, export.name, "0" * 64, dest, None)
    assert result["removed"] == {("copia_exata", "train"): 3, ("copia_exata", "valid"): 1}
    text = (dest / batch_report.README).read_text(encoding="utf-8")
    assert "Quase-cópias entre splits: não foram removidas." in text
    assert "segunda etapa" not in text and "Nenhuma linha tem `nao`" in text  # no second step, so every pair remains


def test_box_counts_per_split(finished):
    _, dest, _ = finished
    counts = batch_report.box_counts(dest / "cru")
    assert counts["train"] == {"helmet": 2, "vest": 2, "gloves": 2, "boots": 1}
    assert sum(counts["valid"].values()) == 8 and sum(counts["test"].values()) == 8
    assert counts == batch_report.box_counts(dest / "tratado-clahe")
