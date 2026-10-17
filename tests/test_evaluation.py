import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from conftest import REPO, VINI_WEIGHTS, load_script
from epi_imgproc import evaluation
from synthetic import encode_cv2, encode_pil, scene

MODEL_NAMES = {
    0: "boots", 1: "gloves", 2: "goggles", 3: "helmet", 4: "no-boots",
    5: "no-gloves", 6: "no-goggles", 7: "no-helmet", 8: "no-vest", 9: "vest",
}  # fmt: skip
DATASET_NAMES = [
    "helmet", "no-helmet", "vest", "no-vest", "goggles",
    "no-goggles", "boots", "no-boots", "gloves", "no-gloves",
]  # fmt: skip


def test_normalize_class_name():
    assert evaluation.normalize_class_name(" No_Helmet ") == "no-helmet"
    assert evaluation.normalize_class_name("no helmet") == "no-helmet"


def test_remap_identity_and_permutation():
    identity = evaluation.build_class_remap(MODEL_NAMES, MODEL_NAMES)
    assert all(k == v for k, v in identity.items())
    remap = evaluation.build_class_remap(evaluation.names_to_dict(DATASET_NAMES), MODEL_NAMES)
    assert remap[0] == 3  # helmet
    assert remap[1] == 7  # no-helmet
    assert remap[5] == 6  # no-goggles
    assert sorted(remap.values()) == list(range(10))


def test_remap_rejects_unknown_and_duplicate_names():
    with pytest.raises(ValueError, match="absent"):
        evaluation.build_class_remap({0: "helmet", 1: "mask"}, MODEL_NAMES)
    with pytest.raises(ValueError):
        evaluation.build_class_remap({0: "helmet", 1: "Helmet"}, MODEL_NAMES)
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.build_class_remap({0: "helmet"}, {0: "helmet", 1: "HELMET"})


def test_remap_label_text():
    text = "0 0.5 0.5 0.2 0.2\n\n5 0.1 0.2 0.3 0.4 0.5 0.6\n"
    # the polygon (0.1,0.2) (0.3,0.4) (0.5,0.6) becomes its enclosing box
    assert evaluation.remap_label_text(text, {0: 3, 5: 6}) == "3 0.5 0.5 0.2 0.2\n6 0.300000 0.400000 0.400000 0.400000\n"
    assert evaluation.remap_label_text("", {0: 3}) == ""
    with pytest.raises(ValueError, match="neither"):
        evaluation.remap_label_text("0 0.1 0.2 0.3 0.4 0.5", {0: 3})
    with pytest.raises(ValueError, match="neither"):
        evaluation.remap_label_text("0 0.1 0.2 0.3 0.4 0.5 0.6 0.7", {0: 3})
    with pytest.raises(ValueError):
        evaluation.remap_label_text("12 0.5 0.5 0.2 0.2", {0: 3})
    with pytest.raises(ValueError):
        evaluation.remap_label_text("x 0.5 0.5 0.2 0.2", {0: 3})


def roboflow_layout(root: Path, names=DATASET_NAMES, val_entry="../valid/images") -> Path:
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump({"train": "../train/images", "val": val_entry, "test": "../test/images", "nc": len(names), "names": names})
    )
    return data_yaml


def test_resolve_split_dir_roboflow_convention(tmp_path):
    data_yaml = roboflow_layout(tmp_path)
    data = yaml.safe_load(data_yaml.read_text())
    assert evaluation.find_data_yaml(tmp_path) == data_yaml
    assert evaluation.resolve_split_dir(data_yaml, data, "val") == (tmp_path / "valid" / "images").resolve()


def test_resolve_split_dir_with_path_key(tmp_path):
    (tmp_path / "ds" / "images" / "val").mkdir(parents=True)
    data_yaml = tmp_path / "cfg" / "data.yaml"
    data_yaml.parent.mkdir()
    data = {"path": str(tmp_path / "ds"), "val": "images/val"}
    assert evaluation.resolve_split_dir(data_yaml, data) == (tmp_path / "ds" / "images" / "val").resolve()


def test_resolve_split_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluation.resolve_split_dir(tmp_path / "data.yaml", {"val": "nowhere"})
    with pytest.raises(FileNotFoundError):
        evaluation.find_data_yaml(tmp_path)


def test_label_path_for():
    assert evaluation.label_path_for(Path("/d/valid/images/a.b.jpg")) == Path("/d/valid/labels/a.b.txt")
    assert evaluation.label_path_for(Path("/images/x/images/a.png")) == Path("/images/x/labels/a.txt")
    with pytest.raises(ValueError):
        evaluation.label_path_for(Path("/d/valid/a.jpg"))


def test_list_images_rejects_colliding_stems(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"1")
    (tmp_path / "a.png").write_bytes(b"2")
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.list_images(tmp_path)


def fake_model(modules, depth=0.33, width=0.25, names=MODEL_NAMES):
    layers = [[-1, 1, module, []] for module in modules]
    return SimpleNamespace(
        yaml={"depth_multiple": depth, "width_multiple": width, "backbone": layers, "head": []}, names=names
    )


def test_describe_checkpoint():
    checkpoint = {
        "model": fake_model(["Conv", "C2f", "SPPF"]),
        "version": "8.4.166",
        "train_args": {"imgsz": 640, "model": "yolov8n.pt", "iou": 0.7},
    }
    info = evaluation.describe_checkpoint(checkpoint)
    assert info["variant"] == "YOLOv8n"
    assert info["imgsz"] == 640
    assert info["names"][3] == "helmet"
    eleven = evaluation.describe_checkpoint({"model": fake_model(["Conv", "C3k2", "C2PSA"])})
    assert eleven["family"] == "YOLO11"


def metrics_row(map50, recalls):
    return {
        "map50_readme": map50,
        "map50_all": map50,
        "map50_95_all": map50 / 2,
        "per_class": {
            name: {"ap50": map50, "recall": recall, "recall_max_f1": recall}
            for name, recall in zip(evaluation.README_CLASSES, recalls)
        },
    }


def test_decision_rule():
    base = metrics_row(0.50, [0.6] * 6)
    assert evaluation.decide(base, metrics_row(0.51, [0.6] * 6)) == (True, [])
    passed, reasons = evaluation.decide(base, metrics_row(0.50, [0.7] * 6))
    assert not passed and "mAP50" in reasons[0]
    passed, reasons = evaluation.decide(base, metrics_row(0.60, [0.6, 0.6, 0.6, 0.6, 0.6, 0.59]))
    assert not passed and "no-goggles" in reasons[0]
    passed, reasons = evaluation.decide(metrics_row(0.5, [0.6] * 5 + [None]), metrics_row(0.6, [0.7] * 6))
    assert not passed and "não verificável" in reasons[0]


def fake_stats():
    """Two images in the shape of Ultralytics' metrics.stats (tp columns: IoU 0.50, 0.55, ...)."""
    tp = np.zeros((6, 10), bool)
    tp[[0, 1, 2, 3, 5], 0] = True  # matched at IoU 0.5
    tp[4, 1:] = True  # only the stricter IoU columns: must not count
    return {
        "tp": [tp[:4], tp[4:], np.zeros((0, 10), bool)],
        "conf": [np.array([0.9, 0.25, 0.2499, 0.6]), np.array([0.8, 0.3]), np.zeros(0)],
        "pred_cls": [np.array([3.0, 3.0, 3.0, 7.0]), np.array([9.0, 2.0]), np.zeros(0)],
        "target_cls": [np.array([3.0, 3.0, 3.0, 3.0, 7.0]), np.array([9.0, 2.0, 2.0]), np.array([8.0])],
    }


def test_true_positive_counts_exact_threshold_and_iou_column():
    true_positives, instances = evaluation.true_positive_counts(fake_stats(), 10, confidence=0.25)
    # helmet (3): matches at 0.9 and exactly 0.25 count, the one at 0.2499 does not; vest (9) has none at IoU 0.5
    assert true_positives.tolist() == [0, 0, 1, 2, 0, 0, 0, 1, 0, 0]
    assert instances.tolist() == [0, 0, 2, 4, 0, 0, 0, 1, 1, 1]
    nothing, no_instances = evaluation.true_positive_counts({}, 10)
    assert nothing.tolist() == [0] * 10 and no_instances.tolist() == [0] * 10


def test_class_metrics_from_validator():
    present = [2, 3, 7, 8, 9]  # no-goggles (6) has no instances
    all_ap = np.tile(np.linspace(0.1, 0.5, len(present))[:, None], (1, 10))
    box = SimpleNamespace(
        ap_class_index=np.array(present), all_ap=all_ap, r=np.array([0.9] * len(present)), map50=0.3, map=0.2,
    )  # fmt: skip
    metrics = SimpleNamespace(box=box)
    result = evaluation.class_metrics_from_validator(metrics, MODEL_NAMES, fake_stats(), confidence=0.25)
    helmet = result["per_class"]["helmet"]
    assert helmet["recall"] == 2 / 4 and helmet["true_positives"] == 2 and helmet["instances"] == 4
    assert result["per_class"]["vest"]["recall"] == 0.0
    assert result["per_class"]["goggles"]["recall"] == 1 / 2
    assert result["per_class"]["helmet"]["ap50"] == pytest.approx(all_ap[1, 0])
    assert result["per_class"]["no-goggles"] == {
        "ap50": None, "recall": None, "true_positives": None, "recall_max_f1": None, "instances": 0,
    }  # fmt: skip
    assert result["map50_readme"] == pytest.approx(all_ap[:, 0].mean())


def test_results_table_has_baseline_and_one_row_per_technique():
    rows = [
        {"technique": "none", "metrics": metrics_row(0.5, [0.6] * 6), "verdict": "linha de base"},
        {"technique": "clahe", "metrics": metrics_row(0.4, [0.5] * 5 + [None]), "verdict": "reprovada"},
    ]
    table = evaluation.results_table(rows)
    lines = table.strip().splitlines()
    assert len(lines) == 4
    for name in evaluation.README_CLASSES:
        assert f"recall {name}" in lines[0]
    assert "n/d" in lines[3]
    validation = {
        "conf": 0.001, "iou": 0.7, "imgsz": 640, "rect": True, "max_det": 300, "batch": 16,
        "recall_confidence": 0.25, "recall_iou": 0.5,
    }  # fmt: skip
    captioned = evaluation.results_table(rows, validation)
    assert captioned.startswith(table)
    caption = captioned[len(table):]
    assert "IoU do NMS 0.7" in caption and "confiança ≥ 0.25" in caption and "IoU ≥ 0.5" in caption
    assert "contagem exata" in caption


MIXED_POLYGON_ROW = "2 0.1 0.1 0.4 0.1 0.4 0.3 0.1 0.3"


def corrected_yaml_copy(dataset_root: Path, target: Path) -> Path:
    """Like the copy made for the real data: absolute `path`, only the val split, original untouched."""
    original = yaml.safe_load((dataset_root / "data.yaml").read_text())
    copy = {"path": str(dataset_root), "val": "valid/images", "nc": original["nc"], "names": original["names"]}
    target.write_text(yaml.safe_dump(copy))
    return target


def write_dataset(root: Path, count: int = 3) -> Path:
    data_yaml = roboflow_layout(root)
    for index in range(count):
        image = scene(240, 320, seed=index)
        data = encode_cv2(image, ".jpg") if index % 2 == 0 else encode_pil(image, "JPEG", 6, quality=95)
        (root / "valid" / "images" / f"img{index}.jpg").write_bytes(data)
        rows = [f"{c} 0.5 0.5 0.3 0.3" for c in range(6)]
        if index == 1:  # Roboflow exports can mix box and polygon rows in one file
            rows.append(MIXED_POLYGON_ROW)
        (root / "valid" / "labels" / f"img{index}.txt").write_text("\n".join(rows) + "\n")
    (root / "test" / "images" / "never.jpg").write_bytes(b"must not be read")
    return data_yaml


def test_prepare_variant_writes_png_labels_and_yaml(tmp_path):
    write_dataset(tmp_path / "ds")
    (tmp_path / "ds" / "valid" / "images" / "broken.jpg").write_bytes(b"\xff\xd8\xff" + bytes(10))
    images = evaluation.list_images(tmp_path / "ds" / "valid" / "images")
    remap = evaluation.build_class_remap(evaluation.names_to_dict(DATASET_NAMES), MODEL_NAMES)
    prepared = evaluation.prepare_variant("none", images, remap, MODEL_NAMES, tmp_path / "out")
    assert prepared.images == 3
    assert prepared.skipped == {"broken.jpg": "corrupt"}
    assert set(prepared.sharpness) == {"img0.jpg", "img1.jpg", "img2.jpg"}
    assert (tmp_path / "out" / "none" / "images" / "val" / "img1.png").read_bytes().startswith(b"\x89PNG")
    labels = (tmp_path / "out" / "none" / "labels" / "val" / "img0.txt").read_text().split("\n")
    assert [line.split()[0] for line in labels if line] == ["3", "7", "9", "8", "2", "6"]
    config = yaml.safe_load(prepared.data_yaml.read_text())
    assert config["names"][3] == "helmet" and Path(config["path"]).is_absolute()


def test_baseline_matches_what_opencv_reads(tmp_path):
    write_dataset(tmp_path)
    images = evaluation.list_images(tmp_path / "valid" / "images")
    assert evaluation.baseline_pixel_mismatches(images) == []


@pytest.mark.skipif(not VINI_WEIGHTS.exists(), reason="best.pt not available")
def test_end_to_end_with_real_weights_on_synthetic_data(tmp_path):
    """Plumbing only: synthetic images, so the metrics themselves mean nothing."""
    write_dataset(tmp_path / "dados")
    original_yaml = (tmp_path / "dados" / "data.yaml").read_text()
    info = evaluation.run_ab_evaluation(
        data_root=tmp_path / "empty",
        data_yaml=corrected_yaml_copy(tmp_path / "dados", tmp_path / "data-corrigido.yaml"),
        weights=VINI_WEIGHTS,
        work_dir=tmp_path / "ab",
        project_dir=tmp_path / "runs",
        technique_names=["clahe"],
        batch=2,
        workers=0,
    )
    assert info["checkpoint"]["variant"] == "YOLOv8n"
    assert info["imgsz"] == 640
    assert info["train_args"]["model"] == "yolov8n.pt"
    assert info["remap_is_identity"] is False
    assert info["images"] == 3
    assert info["baseline_pixel_mismatches"] == []
    assert [row["technique"] for row in info["rows"]] == ["none", "clahe"]
    assert info["rows"][1]["verdict"] in {"aprovada", "reprovada"}
    # Ultralytics 8.4 drops label files that mix boxes and polygons; every arm must score all 3 images.
    assert [row["images_seen"] for row in info["rows"]] == [3, 3]
    assert info["all_images_scored_in_every_arm"] is True
    assert (tmp_path / "dados" / "data.yaml").read_text() == original_yaml
    table = (tmp_path / "ab" / "results.md").read_text().strip().splitlines()
    assert sum(line.startswith("|") for line in table) == 4
    assert "IoU do NMS 0.7" in "\n".join(table)
    assert info["rows"][0]["validation"]["conf"] == pytest.approx(0.001)
    assert info["rows"][0]["validation"]["recall_iou"] == 0.5
    assert info["same_validation_in_every_arm"] is True
    for row in info["rows"]:
        for values in row["metrics"]["per_class"].values():
            assert values["recall"] == values["true_positives"] / values["instances"]
    assert (tmp_path / "ab" / "results.csv").exists()
    assert (tmp_path / "ab" / "sharpness_val.csv").exists()
    assert {p.name for p in (tmp_path / "runs").iterdir()} == {"none", "clahe"}
    assert not (REPO / "runs").exists()
    variant_images = sorted(p.name for p in (tmp_path / "ab" / "variants" / "none" / "images" / "val").iterdir())
    assert variant_images == ["img0.png", "img1.png", "img2.png"]  # test split never read


def test_eval_script_without_data_exits_2(tmp_path, monkeypatch):
    for variable in ("YOLO_CONFIG_DIR", "YOLO_AUTOINSTALL", "YOLO_OFFLINE"):
        monkeypatch.setenv(variable, os.environ.get(variable, ""))
    script = load_script("run_ab_eval")
    args = ["--data-root", str(tmp_path / "dados"), "--work-dir", str(tmp_path / "ab"), "--project-dir", str(tmp_path / "runs")]
    assert script.main(args) == 2
    (tmp_path / "dados" / "valid" / "images").mkdir(parents=True)
    (tmp_path / "dados" / "data.yaml").write_text("names: [helmet]\nval: ../valid/images\n")
    assert script.main(args) == 2  # data.yaml present but no validation images


def test_prepare_variant_discards_files_from_a_previous_run(tmp_path):
    """Review finding 1: Ultralytics scores every file in the folder, so stale PNGs would be counted."""
    write_dataset(tmp_path / "ds", count=3)
    images = evaluation.list_images(tmp_path / "ds" / "valid" / "images")
    remap = evaluation.build_class_remap(evaluation.names_to_dict(DATASET_NAMES), MODEL_NAMES)
    evaluation.prepare_variant("none", images, remap, MODEL_NAMES, tmp_path / "out")
    prepared = evaluation.prepare_variant("none", images[:1], remap, MODEL_NAMES, tmp_path / "out")
    assert prepared.images == 1
    root = tmp_path / "out" / "none"
    assert sorted(p.name for p in (root / "images" / "val").iterdir()) == ["img0.png"]
    assert sorted(p.name for p in (root / "labels" / "val").iterdir()) == ["img0.txt"]
