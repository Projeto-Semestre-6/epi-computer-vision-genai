"""Segmentation-equivalent figures: best.pt boxes and the per-PPE state rule of README "Etapa 5".

Usage (from the repository root):
    .venv/bin/python scripts/make_segmentation_figures.py
    .venv/bin/python scripts/make_segmentation_figures.py --list-absence-candidates 10

Inputs go through the service pipeline (decode -> EXIF -> technique `none`), exactly what the model
service receives. Detection uses the defaults of Vini's POST /detect (conf 0.25, NMS IoU 0.45) and
the checkpoint's imgsz (640). Only validation images are read; the test split never is.

Figures: the images listed in apresentacao/imagens-escolhidas.txt, plus one example of
"ausência identificada" for the helmet: the first validation image, in alphabetical order, whose
labels and model both give that state and that is not in REJECTED_BY_EYE (watermarks, logos or
overlaid text can only be checked by looking at the image).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from epi_imgproc import evaluation, intake  # noqa: E402

EVAL_ROOT = Path.home() / "unisagrado" / "epi-avaliacao"
WEIGHTS = Path.home() / "unisagrado" / "epi-vini-referencia" / "best.pt"
CONF, IOU, IMGSZ = 0.25, 0.45, 640  # POST /detect defaults (backend/main.py) and train_args.imgsz
DISPLAY_SIZE = 640  # larger photos are shrunk for the slide only; detection runs on the full image

PPE = {  # README classes: (positive, negative)
    "Capacete": ("helmet", "no-helmet"),
    "Colete": ("vest", "no-vest"),
    "Óculos": ("goggles", "no-goggles"),
}
IN_SCOPE = {name for pair in PPE.values() for name in pair}
NEGATIVE = {negative for _, negative in PPE.values()}
IDENTIFIED, ABSENT, INDETERMINATE = "identificado", "ausência identificada", "indeterminado"
NO_DETECTION, CONFLICT = "nenhuma detecção", "conflito"

# Helmet-absence candidates rejected after looking at them (validation image name -> reason).
REJECTED_BY_EYE: dict[str, str] = {
    "00004_jpg.rf.569936908230ca3c13fd57eb33d111da.jpg": "marca d'água de banco de imagens (Colourbox) no centro",
}

POSITIVE_COLOR, NEGATIVE_COLOR = (33, 102, 172), (178, 24, 43)
# DejaVu (with accents) ships with matplotlib, which this .venv has as an Ultralytics dependency.
FONT_DIR = Path(np.__file__).resolve().parents[1] / "matplotlib" / "mpl-data" / "fonts" / "ttf"
try:
    FONT = ImageFont.truetype(str(FONT_DIR / "DejaVuSans.ttf"), 15)
    FONT_BOLD = ImageFont.truetype(str(FONT_DIR / "DejaVuSans-Bold.ttf"), 15)
except OSError:
    FONT = FONT_BOLD = ImageFont.load_default(size=15)


# --- README "Etapa 5": threshold, out-of-scope classes, state per PPE ---------------------------


def ppe_states(detections: Iterable[tuple[str, float]], min_confidence: float = CONF) -> dict[str, dict]:
    """State of each PPE from (class name, confidence) detections of one image.

    Detections below `min_confidence` and classes outside the README scope are ignored. A PPE is
    "identificado" when only its positive class is detected, "ausência identificada" when only its
    negative class is, and "indeterminado" otherwise: with no detection ("não detectado" is never
    "ausente") or with both (conflict: one main worker cannot both wear and lack the item).
    """
    kept = [(name, conf) for name, conf in detections if name in IN_SCOPE and conf >= min_confidence]
    states = {}
    for ppe, (positive, negative) in PPE.items():
        pos = sorted((conf for name, conf in kept if name == positive), reverse=True)
        neg = sorted((conf for name, conf in kept if name == negative), reverse=True)
        if pos and neg:
            state, reason = INDETERMINATE, CONFLICT
        elif pos:
            state, reason = IDENTIFIED, None
        elif neg:
            state, reason = ABSENT, None
        else:
            state, reason = INDETERMINATE, NO_DETECTION
        states[ppe] = {"estado": state, "motivo": reason, positive: pos, negative: neg}
    return states


def discarded_classes(detections: Iterable[tuple[str, float]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, _ in detections:
        if name not in IN_SCOPE:
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def truth_label(info: dict) -> str:
    """In the labels, no annotation means "not annotated", not an inference failure."""
    if info["motivo"] == NO_DETECTION:
        return "não anotado"
    if info["motivo"] == CONFLICT:
        return "as duas classes anotadas"
    return info["estado"]


def state_text(info: dict) -> str:
    return info["estado"] if info["motivo"] != CONFLICT else f"{info['estado']} (conflito)"


def helmet_absence_candidates(
    images: Iterable[Path],
    truth_states: Callable[[Path], dict],
    model_states: Callable[[Path], dict],
) -> Iterator[Path]:
    """Images, in alphabetical order, whose labels and model both give "ausência identificada" for the helmet.

    The model only runs on images whose labels qualify.
    """
    for path in sorted(images, key=lambda p: p.name):
        if truth_states(path)["Capacete"]["estado"] != ABSENT:
            continue
        if model_states(path)["Capacete"]["estado"] == ABSENT:
            yield path


# --- figures -----------------------------------------------------------------------------------


def fmt(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def display_scale(width: int, height: int) -> float:
    return min(1.0, DISPLAY_SIZE / max(width, height))


def draw_figure(pixels_bgr: np.ndarray, boxes: Sequence[dict], states: dict, truth: dict, discarded: dict) -> Image.Image:
    rgb = Image.fromarray(pixels_bgr[:, :, ::-1].copy())
    source_size = rgb.size
    scale = display_scale(*source_size)
    if scale < 1.0:
        rgb = rgb.resize((round(source_size[0] * scale), round(source_size[1] * scale)), Image.Resampling.LANCZOS)
    width, height = rgb.size
    annotated = rgb.copy()
    draw = ImageDraw.Draw(annotated)
    for box in boxes:
        color = NEGATIVE_COLOR if box["classe"] in NEGATIVE else POSITIVE_COLOR
        x0, y0, x1, y1 = (value * scale for value in box["xyxy"])
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        text = f"{box['classe']} {fmt(box['conf'])}"
        tw = draw.textlength(text, font=FONT)
        ty = max(0, y0 - 19)
        draw.rectangle((x0, ty, x0 + tw + 6, ty + 19), fill=color)
        draw.text((x0 + 3, ty + 1), text, fill=(255, 255, 255), font=FONT)

    gap, header, row_h = 10, 26, 22
    panel_h = header + row_h * (len(PPE) + 2) + 10
    canvas = Image.new("RGB", (2 * width + gap, header + height + panel_h), (255, 255, 255))
    canvas.paste(rgb, (0, header))
    canvas.paste(annotated, (width + gap, header))
    draw = ImageDraw.Draw(canvas)
    before = "Antes: entrada (saída do serviço, técnica none)"
    if scale < 1.0:
        before = f"Antes: entrada (técnica none), reduzida de {source_size[0]}x{source_size[1]} para exibição"
    draw.text((4, 4), before, fill=(0, 0, 0), font=FONT_BOLD)
    draw.text((width + gap + 4, 4), f"Depois: YOLOv8n (best.pt), conf ≥ {fmt(CONF)}, IoU {fmt(IOU)}, imgsz {IMGSZ}", fill=(0, 0, 0), font=FONT_BOLD)
    y = header + height + 8
    draw.text((4, y), "Estado por EPI (modelo)", fill=(0, 0, 0), font=FONT_BOLD)
    draw.text((width + gap + 4, y), "Gabarito (rótulos do dataset)", fill=(0, 0, 0), font=FONT_BOLD)
    for ppe, info in states.items():
        y += row_h
        positive, negative = PPE[ppe]
        evidence = [f"{positive} {fmt(c)}" for c in info[positive]] + [f"{negative} {fmt(c)}" for c in info[negative]]
        line = f"{ppe}: {state_text(info)}" + (f"  ({'; '.join(evidence)})" if evidence else f"  ({NO_DETECTION})")
        color = NEGATIVE_COLOR if info["estado"] == ABSENT else (0, 0, 0)
        draw.text((4, y), line, fill=color, font=FONT)
        draw.text((width + gap + 4, y), f"{ppe}: {truth_label(truth[ppe])}", fill=(0, 0, 0), font=FONT)
    y += row_h
    dropped = ", ".join(f"{name} ×{count}" for name, count in discarded.items()) or "nenhuma"
    draw.text((4, y), f"Descartadas (fora do escopo do README): {dropped}", fill=(90, 90, 90), font=FONT)
    return canvas


class Segmenter:
    """best.pt with the /detect defaults, plus the labels of the validation split."""

    def __init__(self, weights: Path, dataset_names: dict[int, str]):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.dataset_names = dataset_names

    def detect(self, image_path: Path) -> tuple[np.ndarray, list[dict]]:
        pixels = intake.decode_image(image_path.read_bytes()).pixels
        result = self.model.predict(source=pixels, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False, save=False)[0]
        boxes = [
            {"classe": self.model.names[int(cls)], "conf": round(conf, 4), "xyxy": [round(v, 1) for v in xyxy]}
            for cls, conf, xyxy in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())
        ]
        return pixels, boxes

    def label_detections(self, image_path: Path) -> list[tuple[str, float]]:
        label = evaluation.label_path_for(image_path)
        rows = [line.split() for line in label.read_text().splitlines() if line.strip()] if label.exists() else []
        return [(self.dataset_names[int(row[0])], 1.0) for row in rows]

    def model_states(self, image_path: Path) -> dict:
        _, boxes = self.detect(image_path)
        return ppe_states((box["classe"], box["conf"]) for box in boxes)

    def truth_states(self, image_path: Path) -> dict:
        return ppe_states(self.label_detections(image_path))

    def figure(self, image_path: Path, out_png: Path, root: Path) -> dict:
        pixels, boxes = self.detect(image_path)
        detections = [(box["classe"], box["conf"]) for box in boxes]
        states = ppe_states(detections)
        truth = self.truth_states(image_path)
        in_scope = [box for box in boxes if box["classe"] in IN_SCOPE]
        discarded = discarded_classes(detections)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        draw_figure(pixels, in_scope, states, truth, discarded).save(out_png)
        return {
            "imagem": image_path.name,
            "tamanho": f"{pixels.shape[1]}x{pixels.shape[0]}",
            "reduzida_para_exibicao": display_scale(pixels.shape[1], pixels.shape[0]) < 1.0,
            "deteccoes_no_escopo": in_scope,
            "descartadas": discarded,
            "estado_modelo": {ppe: info["estado"] for ppe, info in states.items()},
            "motivo_indeterminado": {ppe: info["motivo"] for ppe, info in states.items() if info["motivo"]},
            "estado_gabarito": {ppe: truth_label(info) for ppe, info in truth.items()},
            "figura": str(out_png.relative_to(root)),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=EVAL_ROOT / "dados")
    parser.add_argument("--data-yaml", type=Path, default=None, help="data.yaml to use instead of searching --data-root")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--presentation-dir", type=Path, default=EVAL_ROOT / "apresentacao")
    parser.add_argument("--config-dir", type=Path, default=EVAL_ROOT / "ab" / ".ultralytics-config")
    parser.add_argument("--list-absence-candidates", type=int, metavar="N", default=0,
                        help="only print the first N helmet-absence candidates (before REJECTED_BY_EYE)")
    args = parser.parse_args(argv)

    try:
        data_yaml = args.data_yaml or evaluation.find_data_yaml(args.data_root)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    data = yaml.safe_load(data_yaml.read_text())
    val_dir = evaluation.resolve_split_dir(data_yaml, data, "val")
    images = evaluation.list_images(val_dir)
    evaluation.isolate_ultralytics(args.config_dir)
    segmenter = Segmenter(args.weights, evaluation.names_to_dict(data["names"]))
    candidates = helmet_absence_candidates(images, segmenter.truth_states, segmenter.model_states)

    if args.list_absence_candidates:
        for _, path in zip(range(args.list_absence_candidates), candidates):
            print(path.name, REJECTED_BY_EYE.get(path.name, ""))
        return 0

    root = args.presentation_dir
    out_dir = root / "segmentacao"
    chosen = [Path(line) for line in (root / "imagens-escolhidas.txt").read_text().split()]
    records = [segmenter.figure(path, out_dir / f"{path.stem}.png", root) for path in chosen]
    absence = next((path for path in candidates if path.name not in REJECTED_BY_EYE), None)
    output = {
        "parametros": {"conf": CONF, "iou_nms": IOU, "imgsz": IMGSZ, "pesos": str(args.weights)},
        "regra": "identificado: só a positiva; ausência identificada: só a negativa; indeterminado: nenhuma "
        "das duas ou as duas (conflito). Detecções abaixo de conf e classes fora do escopo são ignoradas.",
        "imagens": records,
        "ausencia_capacete": None,
    }
    if absence is not None:
        record = segmenter.figure(absence, out_dir / f"{absence.stem}.png", root)
        record["criterio"] = (
            "primeira imagem da validação, em ordem alfabética, em que o gabarito e o modelo dão "
            "'ausência identificada' para o capacete e que não tem marca d'água, logotipo nem texto sobreposto"
        )
        record["rejeitadas_ao_olhar"] = REJECTED_BY_EYE
        output["ausencia_capacete"] = record
    (out_dir / "estados.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    for record in [*records, *([output["ausencia_capacete"]] if absence else [])]:
        print(record["imagem"][:40], record["estado_modelo"], "| gabarito:", record["estado_gabarito"], "| descartadas:", record["descartadas"])
    if absence is None:
        print("no helmet-absence image left after REJECTED_BY_EYE", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
