"""A/B evaluation of the enhancement techniques with the downstream YOLO model (best.pt).

Every arm, the baseline included, goes through the service pipeline (decode -> EXIF policy ->
technique -> lossless PNG), so the only difference between rows is the technique. Labels are
rewritten to the model's class indices by class name. Ultralytics is imported lazily so the
service and the pure helpers never depend on it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from epi_imgproc import frequency, intake, techniques

README_CLASSES = ("helmet", "no-helmet", "vest", "no-vest", "goggles", "no-goggles")
RECALL_CONFIDENCE = 0.25  # default `conf` of the model service's /detect (backend/main.py)
RECALL_IOU_COLUMN = 0  # column of Ultralytics' `tp` matrix for its first IoU threshold, 0.5
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
TOLERANCE = 1e-9


# --- class names and labels ---------------------------------------------------------------------


def normalize_class_name(name: str) -> str:
    return "-".join(str(name).strip().lower().replace("_", " ").split())


def names_to_dict(names: Sequence[str] | Mapping[int, str]) -> dict[int, str]:
    if isinstance(names, Mapping):
        return {int(index): str(name) for index, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def build_class_remap(dataset_names: Mapping[int, str], model_names: Mapping[int, str]) -> dict[int, int]:
    """Map dataset class index -> model class index by normalized name."""
    model_by_name: dict[str, int] = {}
    for index, name in model_names.items():
        key = normalize_class_name(name)
        if key in model_by_name:
            raise ValueError(f"model has duplicate class name {name!r}")
        model_by_name[key] = index
    remap: dict[int, int] = {}
    missing = []
    for index, name in dataset_names.items():
        key = normalize_class_name(name)
        if key not in model_by_name:
            missing.append(name)
        else:
            remap[index] = model_by_name[key]
    if missing:
        raise ValueError(f"dataset classes absent from the model: {missing}")
    if len(set(remap.values())) != len(remap):
        raise ValueError("two dataset classes map to the same model class")
    return remap


def polygon_to_box(coordinates: Sequence[float]) -> tuple[float, float, float, float]:
    """Enclosing box (x_center, y_center, width, height) of a normalized polygon x1 y1 x2 y2 ...

    Same min/max rule as Ultralytics' `segments2boxes`.
    """
    xs, ys = coordinates[0::2], coordinates[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def remap_label_text(text: str, remap: Mapping[int, int]) -> str:
    """Rewrite the class index of each YOLO label line and turn polygon rows into boxes.

    Box rows (class x y w h) keep their coordinates verbatim. Polygon rows (class x1 y1 x2 y2 ...)
    become their enclosing box, as Ultralytics does for polygon-only files; Ultralytics 8.4 drops
    the whole image when one file mixes both kinds ("labels mix segment and detection rows").
    """
    lines = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        try:
            class_index = int(parts[0])
        except ValueError:
            raise ValueError(f"line {line_number}: class index {parts[0]!r} is not an integer") from None
        if class_index not in remap:
            raise ValueError(f"line {line_number}: class index {class_index} is not in data.yaml names")
        coordinates = parts[1:]
        if len(coordinates) != 4:
            if len(coordinates) < 6 or len(coordinates) % 2:
                raise ValueError(f"line {line_number}: {len(coordinates)} values is neither a box nor a polygon")
            box = polygon_to_box([float(value) for value in coordinates])
            coordinates = [f"{value:.6f}" for value in box]
        lines.append(" ".join([str(remap[class_index]), *coordinates]))
    return "\n".join(lines) + ("\n" if lines else "")


# --- dataset discovery --------------------------------------------------------------------------


def find_data_yaml(root: Path) -> Path:
    candidates = sorted(root.rglob("data.yaml"), key=lambda path: (len(path.parts), str(path)))
    if not candidates:
        raise FileNotFoundError(f"no data.yaml under {root}")
    return candidates[0]


def resolve_split_dir(data_yaml: Path, data: Mapping, split: str = "val") -> Path:
    """Find the image directory of a split, tolerating Roboflow's `../valid/images` convention."""
    entry = data.get(split)
    if isinstance(entry, (list, tuple)):
        raise ValueError(f"split {split!r} lists several directories; pass --images explicitly")
    base = data_yaml.parent
    candidates: list[Path] = []
    if entry:
        entry_path = Path(str(entry))
        if entry_path.is_absolute():
            candidates.append(entry_path)
        else:
            if data.get("path"):
                root = Path(str(data["path"]))
                candidates.append((root if root.is_absolute() else base / root) / entry_path)
            candidates += [base / entry_path, base / str(entry).removeprefix("../")]
    candidates += [base / "valid" / "images", base / "val" / "images", base / "images" / "val"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"no image directory for split {split!r}; tried {[str(c) for c in candidates]}")


def label_path_for(image_path: Path) -> Path:
    """Ultralytics convention: the last `images` directory component becomes `labels`."""
    parts = list(image_path.parts)
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"{image_path} is not inside an `images` directory")


def list_images(directory: Path) -> list[Path]:
    images = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    stems = [p.stem for p in images]
    if len(set(stems)) != len(stems):
        raise ValueError(f"duplicate file stems in {directory}; PNG outputs would collide")
    return images


# --- variant preparation ------------------------------------------------------------------------


@dataclass
class PreparedVariant:
    technique: str
    root: Path
    data_yaml: Path
    images: int
    skipped: dict[str, str] = field(default_factory=dict)
    sharpness: dict[str, float] = field(default_factory=dict)
    polygon_rows: int = 0


def prepare_variant(
    technique: str,
    images: Iterable[Path],
    remap: Mapping[int, int],
    model_names: Mapping[int, str],
    out_root: Path,
) -> PreparedVariant:
    """Write `<out_root>/<technique>/images/val/*.png`, remapped labels, and a data.yaml."""
    root = out_root / technique
    if root.exists():
        shutil.rmtree(root)  # Ultralytics scores every file in the folder: no leftovers from earlier runs
    image_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    prepared = PreparedVariant(technique, root, root / "data.yaml", 0)
    for source in images:
        try:
            decoded = intake.decode_image(source.read_bytes())
        except intake.IntakeError as exc:
            prepared.skipped[source.name] = exc.code
            continue
        if technique == "none":
            prepared.sharpness[source.name] = frequency.high_frequency_energy_ratio(decoded.pixels)
        processed = techniques.apply_technique(technique, decoded.pixels)
        (image_dir / f"{source.stem}.png").write_bytes(intake.encode_png(processed))
        label_source = label_path_for(source)
        text = label_source.read_text() if label_source.exists() else ""
        prepared.polygon_rows += sum(1 for line in text.splitlines() if len(line.split()) > 5)
        (label_dir / f"{source.stem}.txt").write_text(remap_label_text(text, remap))
        prepared.images += 1
    names = {index: model_names[index] for index in sorted(model_names)}
    config = {"path": str(root.resolve()), "train": "images/val", "val": "images/val", "names": names}
    prepared.data_yaml.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return prepared


def baseline_pixel_mismatches(images: Iterable[Path]) -> list[str]:
    """Images whose decoded baseline differs from what cv2.imread (used by Ultralytics) returns."""
    import cv2

    mismatches = []
    for source in images:
        try:
            ours = intake.decode_image(source.read_bytes()).pixels
        except intake.IntakeError:
            continue
        theirs = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if theirs is None or theirs.shape != ours.shape or not np.array_equal(theirs, ours):
            mismatches.append(source.name)
    return mismatches


# --- checkpoint ---------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_checkpoint(checkpoint: Mapping) -> dict:
    """Model family and scale from an Ultralytics checkpoint dict (no torch needed here)."""
    model = checkpoint.get("ema") or checkpoint.get("model")
    model_yaml = dict(getattr(model, "yaml", None) or {})
    modules = {layer[2] for section in ("backbone", "head") for layer in model_yaml.get(section, [])}
    if "C3k2" in modules or "C2PSA" in modules:
        family = "YOLO11"
    elif "C2f" in modules:
        family = "YOLOv8"
    else:
        family = "unknown"
    scales = {(0.33, 0.25): "n", (0.33, 0.5): "s", (0.67, 0.75): "m", (1.0, 1.0): "l", (1.0, 1.25): "x"}
    depth, width = model_yaml.get("depth_multiple"), model_yaml.get("width_multiple")
    scale = scales.get((depth, width), "?") if family == "YOLOv8" else model_yaml.get("scale", "?")
    train_args = dict(checkpoint.get("train_args") or {})
    return {
        "variant": f"{family}{scale}",
        "family": family,
        "depth_multiple": depth,
        "width_multiple": width,
        "names": names_to_dict(getattr(model, "names", {}) or {}),
        "ultralytics_version": checkpoint.get("version"),
        "date": checkpoint.get("date"),
        "train_args": train_args,
        "imgsz": train_args.get("imgsz"),
    }


# --- metrics and decision -----------------------------------------------------------------------


def true_positive_counts(
    stats: Mapping[str, Sequence[np.ndarray]],
    num_classes: int,
    confidence: float = RECALL_CONFIDENCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class true positives with confidence >= `confidence`, and annotated instances.

    `stats` holds the per-image arrays that Ultralytics' DetectionValidator accumulates in
    `metrics.stats`: `tp` (one row per prediction, one column per IoU threshold 0.50:0.95, after
    Ultralytics' own matching), `conf`, `pred_cls` and `target_cls`.
    """

    def joined(key: str, empty: np.ndarray) -> np.ndarray:
        values = stats.get(key) or []
        return np.concatenate([np.asarray(value) for value in values], 0) if values else empty

    tp = joined("tp", np.zeros((0, RECALL_IOU_COLUMN + 1), bool))
    hits = tp[:, RECALL_IOU_COLUMN].astype(bool) & (joined("conf", np.zeros(0)) >= confidence)
    predicted = joined("pred_cls", np.zeros(0)).astype(int)
    true_positives = np.bincount(predicted[hits], minlength=num_classes)
    instances = np.bincount(joined("target_cls", np.zeros(0)).astype(int), minlength=num_classes)
    return true_positives, instances


def class_metrics_from_validator(
    metrics,
    model_names: Mapping[int, str],
    stats: Mapping[str, Sequence[np.ndarray]],
    confidence: float = RECALL_CONFIDENCE,
) -> dict:
    """Per-class AP50 and recall from an Ultralytics DetMetrics object, keyed by README class name.

    `recall` is an exact count: true positives at IoU 0.5 with confidence >= `confidence`
    (`true_positive_counts` over the validator's `stats`) divided by the annotated instances.
    `recall_max_f1` is what Ultralytics prints (at the confidence that maximizes mean F1).
    Classes without instances in the split are reported as None.
    """
    box = metrics.box
    by_index = {int(c): position for position, c in enumerate(box.ap_class_index)}
    index_by_name = {normalize_class_name(name): index for index, name in model_names.items()}
    true_positives, instances_by_index = true_positive_counts(stats, len(model_names), confidence)
    per_class = {}
    for name in README_CLASSES:
        index = index_by_name.get(name, -1)
        instances = None
        if 0 <= index < len(instances_by_index):
            instances = int(instances_by_index[index])
        position = by_index.get(index)
        if position is None or not instances:
            per_class[name] = {
                "ap50": None, "recall": None, "true_positives": None, "recall_max_f1": None, "instances": instances,
            }  # fmt: skip
            continue
        per_class[name] = {
            "ap50": float(box.all_ap[position, 0]),
            "recall": int(true_positives[index]) / instances,
            "true_positives": int(true_positives[index]),
            "recall_max_f1": float(box.r[position]),
            "instances": instances,
        }
    ap_values = [v["ap50"] for v in per_class.values() if v["ap50"] is not None]
    return {
        "map50_readme": float(np.mean(ap_values)) if ap_values else None,
        "map50_all": float(box.map50),
        "map50_95_all": float(box.map),
        "per_class": per_class,
    }


def decide(baseline: Mapping, candidate: Mapping) -> tuple[bool, list[str]]:
    """Pre-registered rule: mAP50 (README classes) strictly up and no README-class recall down."""
    reasons = []
    base_map, cand_map = baseline.get("map50_readme"), candidate.get("map50_readme")
    if base_map is None or cand_map is None:
        reasons.append("mAP50 indisponível")
    elif not cand_map > base_map + TOLERANCE:
        reasons.append(f"mAP50 não aumentou ({cand_map:.4f} vs {base_map:.4f})")
    for name in README_CLASSES:
        base_recall = baseline["per_class"][name]["recall"]
        cand_recall = candidate["per_class"][name]["recall"]
        if base_recall is None or cand_recall is None:
            reasons.append(f"{name} sem instâncias: regra não verificável")
        elif cand_recall < base_recall - TOLERANCE:
            reasons.append(f"recall de {name} caiu ({cand_recall:.4f} vs {base_recall:.4f})")
    return not reasons, reasons


def _fmt(value: float | None) -> str:
    return "n/d" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.4f}"


def results_caption(validation: Mapping) -> str:
    """How each metric of the table was measured, with the parameters the validator really used."""
    return (
        "Como cada métrica foi medida (validação, mesmo `best.pt` em todas as linhas):\n"
        f"- mAP50: validação padrão do Ultralytics (`model.val`), com conf {validation['conf']}, "
        f"IoU do NMS {validation['iou']}, imgsz {validation['imgsz']}, rect={validation['rect']} e "
        f"max_det {validation['max_det']}. \"mAP50 (6 classes)\" é a média do AP50 das 6 classes do README; "
        "\"mAP50 (todas)\" é a média das classes do modelo.\n"
        "- Recall: contagem exata, nas predições dessa mesma validação. Para cada classe, as predições "
        f"com confiança ≥ {validation['recall_confidence']} que o Ultralytics casou com uma caixa anotada "
        f"da mesma classe com IoU ≥ {validation['recall_iou']} (verdadeiros positivos), divididas pelo "
        "número de caixas anotadas da classe. Não é lido da curva recall × confiança.\n"
        "- Regra registrada antes da rodada: uma técnica só vira padrão se o mAP50 (6 classes) subir e "
        "o recall de nenhuma das 6 classes cair em relação a `none`.\n"
    )


def results_table(rows: Sequence[Mapping], validation: Mapping | None = None) -> str:
    """Markdown table: one row per arm, mAP50 (README classes) and exact recall of each README class.

    With `validation` (the parameters captured from the validator), a caption follows the table.
    """
    header = [
        "técnica",
        "imagens",
        "mAP50 (6 classes)",
        *[f"recall {name}" for name in README_CLASSES],
        "mAP50 (todas)",
        "veredito",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        metrics = row["metrics"]
        cells = [
            row["technique"],
            str(row.get("images_seen", "n/d")),
            _fmt(metrics["map50_readme"]),
            *[_fmt(metrics["per_class"][name]["recall"]) for name in README_CLASSES],
            _fmt(metrics["map50_all"]),
            row.get("verdict", ""),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"
    return table if validation is None else table + "\n" + results_caption(validation)


def write_results_csv(rows: Sequence[Mapping], path: Path) -> None:
    fields = ["technique", "images_prepared", "images_seen", "instances", "map50_readme", "map50_all", "map50_95_all"]
    fields += ["verdict", "reasons"]
    for name in README_CLASSES:
        fields += [f"instances_{name}", f"ap50_{name}", f"true_positives_{name}", f"recall_{name}"]
        fields += [f"recall_max_f1_{name}"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            record = {
                "technique": row["technique"],
                "images_prepared": row.get("images"),
                "images_seen": row.get("images_seen"),
                "instances": row.get("instances"),
                "map50_readme": metrics["map50_readme"],
                "map50_all": metrics["map50_all"],
                "map50_95_all": metrics["map50_95_all"],
                "verdict": row.get("verdict", ""),
                "reasons": "; ".join(row.get("reasons", [])),
            }
            for name in README_CLASSES:
                values = metrics["per_class"][name]
                record[f"instances_{name}"] = values.get("instances")
                record[f"ap50_{name}"] = values["ap50"]
                record[f"true_positives_{name}"] = values.get("true_positives")
                record[f"recall_{name}"] = values["recall"]
                record[f"recall_max_f1_{name}"] = values["recall_max_f1"]
            writer.writerow(record)


# --- orchestration (needs ultralytics + torch) --------------------------------------------------


def isolate_ultralytics(config_dir: Path) -> None:
    """Keep Ultralytics from installing packages or using ~/.config settings.

    Must run before `ultralytics` is first imported. YOLO_OFFLINE does not stop every download:
    in Ultralytics 8.4, `check_font` still fetches Arial.ttf into `config_dir` during `model.val`
    when the file is not there yet.
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["YOLO_OFFLINE"] = "True"
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir / "matplotlib"))


def load_checkpoint(weights: Path) -> dict:
    import torch

    # The checkpoint pickles Ultralytics classes, so weights_only=False is required (trusted file).
    return torch.load(weights, map_location="cpu", weights_only=False)


def run_ab_evaluation(
    data_root: Path,
    weights: Path,
    work_dir: Path,
    project_dir: Path,
    technique_names: Sequence[str] | None = None,
    limit: int | None = None,
    batch: int = 16,
    workers: int = 2,
    data_yaml: Path | None = None,
) -> dict:
    """Prepare every arm, validate it on CPU, apply the decision rule, and write the results.

    `data_yaml` overrides discovery under `data_root` (e.g. a copy with corrected paths).
    """
    if data_yaml is None:
        data_yaml = find_data_yaml(data_root)  # fail fast, before loading torch, when there is no data
    data = yaml.safe_load(data_yaml.read_text())
    dataset_names = names_to_dict(data["names"])
    val_dir = resolve_split_dir(data_yaml, data, "val")
    images = list_images(val_dir)
    if limit:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"no images in {val_dir}")

    if "ultralytics" not in sys.modules:
        isolate_ultralytics(work_dir / ".ultralytics-config")
    from ultralytics import YOLO

    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_info = describe_checkpoint(load_checkpoint(weights))
    checkpoint_info["sha256"] = sha256_file(weights)
    imgsz = int(checkpoint_info["imgsz"] or frequency.MODEL_INPUT_SIZE)
    iou = float(checkpoint_info["train_args"].get("iou") or 0.7)
    model = YOLO(str(weights))
    model_names = names_to_dict(model.names)
    remap = build_class_remap(dataset_names, model_names)

    captured: dict = {}

    def record_stats(validator) -> None:
        # Ultralytics clears metrics.stats before on_val_end, so keep the lists as of the last batch.
        captured["stats"] = {key: list(values) for key, values in validator.metrics.stats.items()}

    def record_scored(validator) -> None:
        # What Ultralytics really evaluated; it silently skips images whose labels it rejects.
        captured["seen"] = int(validator.seen)
        captured["instances"] = int(validator.metrics.nt_per_class.sum())
        args = validator.args
        captured["validation"] = {
            "conf": float(args.conf),
            "iou": float(args.iou),
            "imgsz": args.imgsz,
            "rect": bool(args.rect),
            "max_det": int(args.max_det),
            "batch": int(args.batch),
            "recall_confidence": RECALL_CONFIDENCE,
            "recall_iou": round(float(validator.iouv[RECALL_IOU_COLUMN]), 6),
        }

    model.add_callback("on_val_batch_end", record_stats)
    model.add_callback("on_val_end", record_scored)

    names = list(technique_names or techniques.TECHNIQUES)
    if "none" in names:
        names.remove("none")
    names.insert(0, "none")

    run_info = {
        "checkpoint": {k: v for k, v in checkpoint_info.items() if k != "train_args"},
        "train_args": checkpoint_info["train_args"],
        "data_yaml": str(data_yaml),
        "val_dir": str(val_dir),
        "images": len(images),
        "dataset_names": dataset_names,
        "model_names": model_names,
        "remap": remap,
        "remap_is_identity": all(k == v for k, v in remap.items()),
        "imgsz": imgsz,
        "iou": iou,
        "recall_confidence": RECALL_CONFIDENCE,
        "baseline_pixel_mismatches": baseline_pixel_mismatches(images),
        "technique_params": {n: techniques.TECHNIQUES[n].params for n in names},
    }

    rows = []
    for name in names:
        prepared = prepare_variant(name, images, remap, model_names, work_dir / "variants")
        if prepared.sharpness:
            with (work_dir / "sharpness_val.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image", "hf_ratio"])
                writer.writerows(sorted(prepared.sharpness.items()))
        metrics = model.val(
            data=str(prepared.data_yaml),
            split="val",
            imgsz=imgsz,
            batch=batch,
            device="cpu",
            iou=iou,
            workers=workers,
            plots=False,
            save_json=False,
            project=str(project_dir.resolve()),
            name=name,
            exist_ok=True,
            verbose=False,
        )
        rows.append(
            {
                "technique": name,
                "metrics": class_metrics_from_validator(metrics, model_names, captured.pop("stats", {})),
                "validation": captured.pop("validation", None),
                "images": prepared.images,
                "images_seen": captured.pop("seen", None),
                "instances": captured.pop("instances", None),
                "polygon_rows_converted": prepared.polygon_rows,
                "skipped": prepared.skipped,
            }
        )

    complete = {row["technique"]: row["images_seen"] == len(images) for row in rows}
    run_info["all_images_scored_in_every_arm"] = all(complete.values())
    baseline = rows[0]["metrics"]
    rows[0]["verdict"] = "linha de base" if complete["none"] else "inválida"
    for row in rows[1:]:
        if not (complete["none"] and complete[row["technique"]]):
            row["verdict"] = "inválida"
            row["reasons"] = [f"Ultralytics avaliou {row['images_seen']} de {len(images)} imagens"]
            continue
        passed, reasons = decide(baseline, row["metrics"])
        row["verdict"] = "aprovada" if passed else "reprovada"
        row["reasons"] = reasons

    run_info["same_validation_in_every_arm"] = all(row["validation"] == rows[0]["validation"] for row in rows)
    (work_dir / "results.md").write_text(results_table(rows, rows[0]["validation"]))
    write_results_csv(rows, work_dir / "results.csv")
    run_info["rows"] = rows
    (work_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False, default=str))
    return run_info
