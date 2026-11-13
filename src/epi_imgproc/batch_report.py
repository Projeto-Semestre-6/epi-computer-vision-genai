"""Files the phase-2 batch leaves at the destination root for people to read.

    quase-copias.png / .csv   near copies across splits: contact sheet and every pair measured
    manifesto.csv             one row per image file of every copy
    removidas.csv             every photo left out, why, its pair, and whether the pair remains
    LEIA-ME.md                parameters of each technique and of the JPEG encoding
    amostras.png              validation photos in 9 columns: raw and the 8 treated copies

File names, column names and the README are in Brazilian Portuguese, like the folder names.
Image captions are plain ASCII because OpenCV's Hershey fonts have no accents.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from epi_imgproc import batch, duplicates, evaluation, intake, presentation, techniques
from epi_imgproc.batch import BatchError, Selection
from epi_imgproc.duplicates import NearPair, SourceImage

NEAR_COPY_SHEET = "quase-copias.png"
NEAR_COPY_PAIRS = "quase-copias.csv"
MANIFEST = "manifesto.csv"
REMOVED = "removidas.csv"
README = "LEIA-ME.md"
SAMPLES = "amostras.png"

SUMMARY_THRESHOLDS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
NEAR_COPY_HEADER = [
    "medida", "transformacao", "split_a", "arquivo_a", "split_b", "arquivo_b",
    "b_e_o_mais_proximo_de_a", "par_na_folha",
]  # fmt: skip

REASONS = {duplicates.EXACT_COPY: "copia_exata", duplicates.NEAR_COPY: "quase_copia"}
SYMMETRIES = {
    "identity": "nenhuma",
    "mirror": "espelhada",
    "rotate_90": "girada_90",
    "rotate_180": "girada_180",
    "rotate_270": "girada_270",
    "mirror_rotate_90": "espelhada_girada_90",
    "mirror_rotate_180": "espelhada_girada_180",
    "mirror_rotate_270": "espelhada_girada_270",
}
TECHNIQUE_NOTES = {
    "none": "nenhum tratamento: a foto é só decodificada e regravada em JPEG (controle)",
    "histogram_equalization": "equalização global do histograma no canal L (CIELAB)",
    "clahe": "equalização adaptativa com limite de contraste (CLAHE) no canal L",
    "gamma": "correção gama em cada canal BGR",
    "gaussian_blur": "suavização gaussiana nos três canais",
    "unsharp_mask": "máscara de nitidez (unsharp mask) no canal L",
    "fft_lowpass": "passa-baixas gaussiano no domínio da frequência, canal L",
    "fft_high_emphasis": "ênfase de altas frequências no domínio da frequência, canal L",
}

_CAPTION_HEIGHT = 24
_BACKGROUND = 48


# --- small helpers --------------------------------------------------------------------------------


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _decimal(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _gib(size: int) -> str:
    return _decimal(size / 2**30)


def _fit(image: np.ndarray, cell: int) -> np.ndarray:
    """Letterbox a BGR image into a cell x cell tile, keeping its aspect ratio."""
    scale = cell / float(max(image.shape[:2]))
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    tile = np.full((cell, cell, 3), _BACKGROUND, dtype=np.uint8)
    top, left = (cell - size[1]) // 2, (cell - size[0]) // 2
    tile[top : top + size[1], left : left + size[0]] = resized
    return tile


def _caption(width: int, text: str, color: tuple[int, int, int] = (255, 255, 255), scale: float = 0.45) -> np.ndarray:
    bar = np.full((_CAPTION_HEIGHT, width, 3), color, dtype=np.uint8)
    cv2.putText(bar, text, (5, _CAPTION_HEIGHT - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return bar


def _decode(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise BatchError(f"{path} could not be decoded")
    return image


def _class_names(copy_root: Path) -> dict[int, str]:
    return evaluation.names_to_dict(yaml.safe_load((copy_root / "data.yaml").read_text(encoding="utf-8"))["names"])


def box_counts(copy_root: Path, listing: Mapping[str, Sequence[str]] | None = None) -> dict[str, Counter]:
    """Boxes per class name in each split of a copy on disk."""
    names = _class_names(copy_root)
    listing = listing or batch.list_copy(copy_root)
    counts = {}
    for split in batch.SPLITS:
        total: Counter = Counter()
        for name in listing[split]:
            text = (copy_root / split / "labels" / batch.label_name(name)).read_text(encoding="utf-8")
            total += batch.class_counts(text, names)
        counts[split] = total
    return counts


# --- near copies across splits --------------------------------------------------------------------


def load_raw_images(raw_root: Path) -> tuple[list[SourceImage], np.ndarray]:
    """Every image of the raw copy with its thumbnail, in split and name order."""
    listing = batch.list_copy(raw_root)
    images, thumbnails = [], []
    for split in batch.SPLITS:
        for name in listing[split]:
            data = (raw_root / split / "images" / name).read_bytes()
            images.append(SourceImage(split, name, batch.sha256_bytes(data)))
            thumbnails.append(duplicates.thumbnail(data))
    return images, np.stack(thumbnails)


def pick_across_range(nearest: Sequence[NearPair], count: int, max_distance: float) -> list[NearPair]:
    """`count` pairs spread over [0, max_distance]: for each value of an even grid, the unused pair closest to it."""
    available = list(nearest)
    chosen = []
    for step in range(count):
        if not available:
            break
        target = max_distance * step / max(count - 1, 1)
        pick = min(available, key=lambda pair: (abs(pair.distance - target), pair.distance, pair.lower))
        available.remove(pick)
        chosen.append(pick)
    return sorted(chosen, key=lambda pair: (pair.distance, pair.lower))


def near_copy_sheet(
    entries: Sequence[tuple[NearPair, SourceImage, SourceImage, bool]],
    load_image: Callable[[SourceImage], np.ndarray],
    threshold: float,
    max_distance: float,
    columns: int = 5,
    cell: int = 224,
) -> np.ndarray:
    """Contact sheet: one tile per pair (lower-priority photo on the left), with measure and splits.

    Without pairs the sheet is the title over one row of empty tiles.
    """
    gap = 6
    tile_width = 2 * cell + gap
    tiles = []
    for number, (pair, lower, higher, leaves) in enumerate(entries, start=1):
        symmetry = "" if pair.symmetry == "identity" else f" {SYMMETRIES[pair.symmetry].replace('_', ' ')}"
        verdict = "SAI" if leaves else "fica"
        text = f"{number:02d}  medida {pair.distance:.2f}  {lower.split} | {higher.split}{symmetry}  {verdict}"
        bar = _caption(tile_width, text, (190, 190, 255) if leaves else (255, 255, 255))
        spacer = np.full((cell, gap, 3), _BACKGROUND, dtype=np.uint8)
        photos = np.hstack([_fit(load_image(lower), cell), spacer, _fit(load_image(higher), cell)])
        tiles.append(np.vstack([bar, photos]))
    blank = np.full((_CAPTION_HEIGHT + cell, tile_width, 3), _BACKGROUND, dtype=np.uint8)
    while not tiles or len(tiles) % columns:  # at least one row: no pair across splits is ordinary input
        tiles.append(blank)
    margin = np.full((_CAPTION_HEIGHT + cell, gap * 2, 3), _BACKGROUND, dtype=np.uint8)
    rows = []
    for start in range(0, len(tiles), columns):
        row = [margin]
        for tile in tiles[start : start + columns]:
            row += [tile, margin]
        rows.append(np.hstack(row))
        rows.append(np.full((gap * 2, rows[-1].shape[1], 3), _BACKGROUND, dtype=np.uint8))
    body = np.vstack(rows)
    title = [
        f"Quase-copias entre splits: {len(entries)} pares ao longo da medida, de 0 a {max_distance:g}. "
        "Esquerda: foto do split de menor prioridade. Direita: foto que fica (test > valid > train).",
        "Medida: diferenca media absoluta entre miniaturas 64x64 em tons de cinza (0 a 255), "
        "a menor entre os 8 espelhamentos e giros de 90 graus.",
        f"Limiar proposto: {threshold:g}. Barra vermelha e 'SAI': a foto da esquerda sairia com esse limiar. "
        f"Nomes dos arquivos de cada par: coluna par_na_folha de {NEAR_COPY_PAIRS}.",
    ]
    bars = [_caption(body.shape[1], line, scale=0.5) for line in title]
    return np.vstack([*bars, body])


def near_copy_report(
    dest_root: Path,
    threshold: float,
    max_distance: float = 8.0,
    sheet_pairs: int = 40,
) -> dict:
    """Measure near copies across the splits of the raw copy; write the contact sheet and the pair list.

    Changes nothing in the copies. Returns the figures needed to decide: photos that would leave
    each split at several thresholds and, at `threshold`, the boxes per class each split loses.
    """
    if threshold > max_distance:
        raise BatchError("the proposed threshold is above the largest distance measured")
    sheet_path, pairs_path = dest_root / NEAR_COPY_SHEET, dest_root / NEAR_COPY_PAIRS
    batch.require_absent([sheet_path, pairs_path])
    raw_root = dest_root / batch.RAW_COPY
    images, thumbnails = load_raw_images(raw_root)
    pairs = duplicates.cross_split_pairs(images, thumbnails, max_distance)
    nearest = duplicates.nearest_partners(pairs)
    chosen = pick_across_range(nearest, sheet_pairs, max_distance)
    sheet_number = {(pair.lower, pair.higher): number for number, pair in enumerate(chosen, start=1)}
    nearest_keys = {(pair.lower, pair.higher) for pair in nearest}

    kept, removals = duplicates.resolve_near_copies(images, pairs, threshold)
    leaving = {removal.image for removal in removals}

    rows = [
        [
            f"{pair.distance:.4f}",
            SYMMETRIES[pair.symmetry],
            images[pair.lower].split,
            images[pair.lower].name,
            images[pair.higher].split,
            images[pair.higher].name,
            int((pair.lower, pair.higher) in nearest_keys),
            sheet_number.get((pair.lower, pair.higher), ""),
        ]
        for pair in sorted(pairs, key=lambda pair: (pair.distance, pair.lower, pair.higher))
    ]
    header = NEAR_COPY_HEADER
    entries = [(pair, images[pair.lower], images[pair.higher], images[pair.lower] in leaving) for pair in chosen]
    sheet = near_copy_sheet(
        entries, lambda image: _decode(raw_root / image.split / "images" / image.name), threshold, max_distance
    )
    batch.write_new(pairs_path, _csv_bytes(header, rows))
    batch.write_new(sheet_path, intake.encode_png(sheet))

    names = _class_names(raw_root)

    def boxes(selected: Sequence[SourceImage]) -> dict[str, dict[str, int]]:
        totals = {split: Counter() for split in batch.SPLITS}
        for image in selected:
            path = raw_root / image.split / "labels" / batch.label_name(image.name)
            totals[image.split] += batch.class_counts(path.read_text(encoding="utf-8"), names)
        return {split: {name: totals[split][name] for name in names.values()} for split in batch.SPLITS}

    by_threshold = {}
    for value in sorted({*[t for t in SUMMARY_THRESHOLDS if t <= max_distance], threshold}):
        counts = Counter(removal.image.split for removal in duplicates.resolve_near_copies(images, pairs, value)[1])
        by_threshold[value] = {split: counts[split] for split in batch.SPLITS}
    index_of = {image: index for index, image in enumerate(images)}
    within = {pair.lower for pair in pairs if pair.distance <= threshold}
    return {
        "threshold": threshold,
        "max_distance": max_distance,
        "images": dict(Counter(image.split for image in images)),
        "pairs_measured": len(pairs),
        "sheet_pairs": len(chosen),
        "removed_by_threshold": by_threshold,
        "removed": dict(Counter(removal.image.split for removal in removals)),
        "remaining": dict(Counter(image.split for image in kept)),
        "removed_by_symmetry": dict(Counter(SYMMETRIES[removal.symmetry] for removal in removals)),
        "kept_because_the_only_near_copies_also_leave": len(within - {index_of[image] for image in leaving}),
        "boxes_before": boxes(images),
        "boxes_lost": boxes([removal.image for removal in removals]),
        "class_names": [names[index] for index in sorted(names)],
        "sheet": str(sheet_path),
        "pairs_csv": str(pairs_path),
    }


# --- manifest, removed photos, samples, README ----------------------------------------------------


def manifest_rows(dest_root: Path, technique_names: Sequence[str]) -> list[list]:
    """One row per image file of every copy: copy, technique, split, name, SHA-256, dimensions, label."""
    rows = []
    copies = [(batch.RAW_COPY, ""), *[(batch.treated_copy_name(name), name) for name in technique_names]]
    for copy_name, technique in copies:
        copy_root = dest_root / copy_name
        listing = batch.list_copy(copy_root)
        for split in batch.SPLITS:
            for name in listing[split]:
                path = copy_root / split / "images" / name
                data = path.read_bytes()
                width, height, _ = batch.stored_size(path)
                label = (copy_root / split / "labels" / batch.label_name(name)).read_bytes()
                rows.append(
                    [
                        copy_name, technique, split, name, batch.sha256_bytes(data), width, height, len(data),
                        batch.sha256_bytes(label), len(batch.label_rows(label.decode("utf-8"))),
                    ]
                )  # fmt: skip
    return rows


MANIFEST_HEADER = [
    "copia", "tecnica", "split", "arquivo", "sha256", "largura", "altura", "bytes", "rotulo_sha256", "caixas",
]  # fmt: skip
REMOVED_HEADER = [
    "arquivo", "split", "motivo", "par", "split_do_par", "par_permanece", "medida", "transformacao",
    "rotulos_divergiam", "divergencia",
]  # fmt: skip
PAIR_REMAINS = {True: "sim", False: "nao"}


def pairs_that_left(selection: Selection) -> list[duplicates.Removal]:
    """Removals whose pair is not in the dataset either.

    Exact copies are resolved first and near copies are then decided among the files kept, so the
    file kept from a group of exact copies can leave in the second step. The pair of a near copy
    always remains.
    """
    remaining = set(selection.kept)
    return [removal for removal in selection.removals if removal.kept not in remaining]


def removed_rows(archive: zipfile.ZipFile, selection: Selection) -> list[list]:
    """One row per photo left out: file, split, reason, its pair, whether the pair remains, label divergence.

    The pair is the photo the removed one was compared with: the file kept from its group of exact
    copies, or the nearest near copy that remains. `par_permanece` says whether the pair is in the
    dataset; when it is not, the pair has a row of its own, which names a photo that remains.
    Labels are compared with the pair's label, whether or not the pair remains.
    """
    rows = []
    remaining = set(selection.kept)
    order = {split: position for position, split in enumerate(batch.SPLITS)}
    for removal in sorted(selection.removals, key=lambda r: (order[r.image.split], r.image.name)):
        removed_label, kept_label = batch.read_label(archive, removal.image), batch.read_label(archive, removal.kept)
        comparison = batch.compare_labels(removed_label, kept_label)
        rows.append(
            [
                removal.image.name,
                removal.image.split,
                REASONS[removal.reason],
                removal.kept.name,
                removal.kept.split,
                PAIR_REMAINS[removal.kept in remaining],
                "" if removal.distance is None else f"{removal.distance:.4f}",
                "" if removal.symmetry is None else SYMMETRIES[removal.symmetry],
                "nao" if comparison == batch.LABELS_EQUAL else "sim",
                comparison,
            ]
        )
    return rows


def samples_sheet(
    dest_root: Path,
    technique_names: Sequence[str],
    count: int = 6,
    cell: int = 256,
) -> tuple[np.ndarray, list[str]]:
    """`count` validation photos, one per row; columns: the raw copy and each treated copy."""
    raw_root = dest_root / batch.RAW_COPY
    chosen = presentation.select_evenly(batch.list_copy(raw_root)["valid"], count)
    copies = [batch.RAW_COPY, *[batch.treated_copy_name(name) for name in technique_names]]
    labels = [batch.RAW_COPY, *technique_names]
    gap = 4
    spacer = np.full((cell, gap, 3), _BACKGROUND, dtype=np.uint8)
    header = []
    for label in labels:
        header += [_caption(cell, label), np.full((_CAPTION_HEIGHT, gap, 3), _BACKGROUND, dtype=np.uint8)]
    rows = [np.hstack(header[:-1])]
    for name in chosen:
        tiles = []
        for copy_name in copies:
            tiles += [_fit(_decode(dest_root / copy_name / "valid" / "images" / name), cell), spacer]
        rows.append(np.full((gap, rows[0].shape[1], 3), _BACKGROUND, dtype=np.uint8))
        rows.append(np.hstack(tiles[:-1]))
    return np.vstack(rows), list(chosen)


def readme_text(
    dest_root: Path,
    archive_name: str,
    archive_sha256: str,
    config: Mapping,
    selection: Selection,
    polygon_rows: int,
    polygon_files: int,
    technique_names: Sequence[str],
    sample_names: Sequence[str],
) -> str:
    raw_root = dest_root / batch.RAW_COPY
    listing = batch.list_copy(raw_root)
    counts = box_counts(raw_root, listing)
    class_names = [name for _, name in sorted(_class_names(raw_root).items())]
    infos = {name: batch.read_treatment_info(dest_root, name) for name in technique_names}
    roboflow = config.get("roboflow") or {}
    removed = Counter((removal.reason, removal.image.split) for removal in selection.removals)

    def per_split(reason: str) -> str:
        return ", ".join(f"{removed[(reason, split)]} de {split}" for split in batch.SPLITS)

    lines = [
        "# Dataset da fase 2: cópia crua e cópias tratadas",
        "",
        f"Gerado em {datetime.now().astimezone().strftime('%d/%m/%Y %H:%M %Z')} por "
        "`scripts/build_phase2_dataset.py` (repositório epi-computer-vision-genai).",
        "",
        "## Origem",
        "",
        f"- Arquivo: `{archive_name}`, SHA-256 `{archive_sha256}`. O zip não foi alterado.",
        f"- Export do Roboflow: workspace `{roboflow.get('workspace', '?')}`, "
        f"projeto `{roboflow.get('project', '?')}`, "
        f"versão {roboflow.get('version', '?')}, licença {roboflow.get('license', '?')}"
        + (f" ({roboflow['url']})." if roboflow.get("url") else "."),
        "",
        "## Estrutura",
        "",
        "```",
        "cru/                    bytes originais do zip, sem fotos repetidas; rótulos só com caixas",
        "tratado-<técnica>/      as mesmas fotos, tratadas e gravadas em JPEG; rótulos iguais aos de cru/",
        "  train/ valid/ test/   cada um com images/ e labels/",
        "  data.yaml             caminhos relativos à pasta da cópia (a pasta pode ser movida)",
        "  tratamento.json       só nas tratadas: parâmetros, tamanho e tempo",
        f"{MANIFEST}  {REMOVED}  {SAMPLES}  {README}",
        "```",
        "",
        "Todas as cópias têm os mesmos nomes de arquivo em cada split e rótulos idênticos byte a byte. "
        "As cópias tratadas passaram por `epi_imgproc.pipeline.process_image`, a mesma função que o endpoint "
        "`/process` chama; a única diferença é a saída em JPEG (o endpoint devolve PNG).",
        "",
        "## Fotos e caixas por split (iguais em todas as cópias)",
        "",
        "| Split | Imagens | Caixas | " + " | ".join(class_names) + " |",
        "|---|---|---|" + "---|" * len(class_names),
    ]
    def count_row(label: str, images: int, boxes: Counter) -> str:
        cells = [label, str(images), str(sum(boxes.values())), *[str(boxes[name]) for name in class_names]]
        return "| " + " | ".join(cells) + " |"

    image_total = sum(len(names) for names in listing.values())
    for split in batch.SPLITS:
        lines.append(count_row(split, len(listing[split]), counts[split]))
    lines += [
        count_row("total", image_total, sum((counts[split] for split in batch.SPLITS), Counter())),
        "",
        "## Fotos removidas",
        "",
    ]
    exact = sum(count for (reason, _), count in removed.items() if reason == duplicates.EXACT_COPY)
    near = sum(count for (reason, _), count in removed.items() if reason == duplicates.NEAR_COPY)
    gone = pairs_that_left(selection)
    gone_by_reason = Counter(removal.reason for removal in gone)
    exact_rule = (
        f"- Cópias exatas (mesmo SHA-256): {exact} ({per_split(duplicates.EXACT_COPY)}). Fica um arquivo por grupo: "
        "prioridade test > valid > train e, no mesmo split, o primeiro nome em ordem alfabética "
        "(ordem dos códigos de caractere, a de `LC_ALL=C sort`). Vale o rótulo do arquivo mantido."
    )
    if selection.near_copy_threshold is None:
        lines += [exact_rule, "- Quase-cópias entre splits: não foram removidas."]
    else:
        threshold = _decimal(selection.near_copy_threshold)
        groups_gone = len({removal.kept for removal in gone if removal.reason == duplicates.EXACT_COPY})
        lines += [
            exact_rule + " As quase-cópias são decididas depois, só entre os arquivos mantidos; grupos em que o "
            "arquivo mantido saiu nessa segunda etapa (nenhum arquivo do grupo está no dataset): "
            f"{groups_gone}.",
            f"- Quase-cópias entre splits: {near} ({per_split(duplicates.NEAR_COPY)}), com limiar {threshold}.",
            "  - Medida: diferença média absoluta entre as miniaturas 64x64 em tons de cinza (0 a 255) das duas fotos, "
            "a menor entre os 8 espelhamentos e giros de 90° de uma delas. Duas fotos são quase-cópias quando a "
            "medida é menor ou igual ao limiar.",
            f"  - Regra: sai a foto que está a até {threshold} de uma foto de split de maior prioridade "
            "(test > valid > train) que permanece no dataset. A decisão segue essa ordem: primeiro as fotos de "
            "valid, contra as de test; depois as de train, contra as de test e as de valid que permanecem. "
            "No caso em cadeia (train parecida com valid, valid parecida com test, train fora do limiar de test), "
            "a foto de valid sai e a de train continua, porque a foto de que ela está perto não permanece. "
            "Entre as fotos que permanecem, nenhum par de splits diferentes está dentro do limiar. "
            "Quase-cópias dentro do mesmo split não foram removidas.",
        ]
    if gone:
        pairs_gone = (
            f"Linhas com `nao`: {gone_by_reason[duplicates.EXACT_COPY]} de `copia_exata` e "
            f"{gone_by_reason[duplicates.NEAR_COPY]} de `quase_copia`; nelas o par saiu depois, como quase-cópia, "
            "e tem a própria linha, que leva a uma foto que permanece."
        )
    else:
        pairs_gone = "Nenhuma linha tem `nao`."
    lines += [
        f"- Lista completa em `{REMOVED}`, uma linha por foto removida: `motivo` é `copia_exata` ou `quase_copia`; "
        "`par` e `split_do_par` identificam a foto com que a removida foi comparada (o arquivo mantido do grupo de "
        "cópias exatas, ou a quase-cópia mais próxima que permanece); `par_permanece` diz se o par está no dataset "
        f"(`sim` ou `nao`). {pairs_gone} `rotulos_divergiam` compara o rótulo da foto removida com o do par "
        "(`divergencia`: `iguais`, `coordenadas` = mesmas classes com outras coordenadas, `classes` = classes "
        "diferentes).",
        "",
        "## Rótulos",
        "",
        f"- {polygon_rows} linhas em formato de polígono, em {polygon_files} arquivos, viraram a caixa que envolve os "
        "pontos (mínimo e máximo de x e de y). O resto de cada arquivo ficou igual, byte a byte.",
        "- As cópias tratadas receberam os rótulos de `cru/` byte a byte.",
        "",
        "## Técnicas",
        "",
        "| Cópia | Função em `epi_imgproc.techniques` | Parâmetros | O que faz |",
        "|---|---|---|---|",
    ]
    for name in technique_names:
        technique = techniques.TECHNIQUES[name]
        parameters = ", ".join(f"{key}={json.dumps(value)}" for key, value in technique.params.items()) or "nenhum"
        note = TECHNIQUE_NOTES.get(name) or (technique.function.__doc__ or "").strip().splitlines()[0]
        lines.append(f"| `{batch.treated_copy_name(name)}` | `{technique.function.__name__}` | {parameters} | {note} |")
    jpeg = infos[technique_names[0]]["jpeg"] if technique_names else intake.jpeg_encoding()
    limits = sorted({info["limite_pixels"] for info in infos.values()})
    lines += [
        "",
        "Sigmas e frequências de corte são definidos na escala do modelo (lado maior com 640 px) e convertidos "
        "para a resolução de cada foto. Nenhuma técnica muda o tamanho da imagem.",
        "",
        "## Codificação JPEG das cópias tratadas",
        "",
        f"- Qualidade {jpeg['quality']}; subamostragem de croma {jpeg['chroma_subsampling']}; "
        f"{'progressivo' if jpeg['progressive'] else 'baseline (não progressivo)'}; "
        f"tabelas de Huffman {'otimizadas' if jpeg['optimized_huffman_tables'] else 'padrão (sem otimização)'}; "
        f"intervalo de reinício {jpeg['restart_interval']}; sem EXIF e sem perfil ICC (só o cabeçalho JFIF).",
        f"- Codificador: {jpeg['encoder']}. Os mesmos parâmetros em todas as cópias tratadas.",
        f"- Limites do lote: {', '.join(str(limit) for limit in limits)} pixels por foto (o serviço mantém "
        f"{intake.MAX_PIXELS}) e {intake.MAX_UPLOAD_BYTES} bytes por arquivo.",
        "- O lote não calcula nenhuma medida de qualidade (brilho, nitidez) em nenhum split.",
        "",
        "## Tamanho e tempo de cada cópia",
        "",
        "| Cópia | Imagens | Tamanho das imagens (GiB) | Tempo (min) |",
        "|---|---|---|---|",
    ]
    raw_bytes = sum(
        (raw_root / split / "images" / name).stat().st_size for split in batch.SPLITS for name in listing[split]
    )
    lines.append(f"| `{batch.RAW_COPY}` | {image_total} | {_gib(raw_bytes)} | (extração do zip) |")
    for name in technique_names:
        info = infos[name]
        lines.append(
            f"| `{info['copia']}` | {sum(info['imagens'].values())} | {_gib(info['bytes_imagens'])} | "
            f"{_decimal(info['segundos'] / 60, 1)} |"
        )
    versions = infos[technique_names[0]]["versoes"] if technique_names else batch.library_versions()
    lines += [
        "",
        "Versões usadas no tratamento: " + ", ".join(f"{key} {value}" for key, value in versions.items()) + ".",
        "",
        "## Arquivos da raiz",
        "",
        f"- `{MANIFEST}`: uma linha por imagem de cada cópia "
        f"({', '.join(f'`{column}`' for column in MANIFEST_HEADER)}). Em `cru`, a coluna `tecnica` fica vazia.",
        f"- `{REMOVED}`: as fotos que ficaram de fora.",
        f"- `{SAMPLES}`: {len(sample_names)} fotos de validação, uma por linha, em {len(technique_names) + 1} colunas "
        "(`cru` e as cópias tratadas, na ordem da tabela de técnicas). "
        f"Fotos: {', '.join(f'`{name}`' for name in sample_names)}.",
    ]
    if (dest_root / NEAR_COPY_SHEET).exists() or (dest_root / NEAR_COPY_PAIRS).exists():
        lines.append(
            f"- `{NEAR_COPY_SHEET}` e `{NEAR_COPY_PAIRS}`: folha de contato e lista dos pares de quase-cópias entre "
            "splits, medidos em `cru/` no momento em que o comando `near-copies` rodou. As marcas SAI/fica da folha "
            "usam o limiar proposto naquele momento, escrito no título da folha."
        )
    lines += [
        "",
        "## Uso",
        "",
        "No Ultralytics, aponte `data` para o `data.yaml` da cópia, por exemplo `tratado-clahe/data.yaml`. "
        "O endpoint tem de aplicar a mesma técnica da cópia usada no treino.",
        "",
    ]
    return "\n".join(lines)


def finish(
    archive: zipfile.ZipFile,
    archive_name: str,
    archive_sha256: str,
    dest_root: Path,
    near_copy_threshold: float | None,
) -> dict:
    """Write manifesto.csv, removidas.csv, amostras.png and LEIA-ME.md once every copy is complete.

    The selection is computed again from the archive and must be exactly what the raw copy holds.
    """
    targets = [dest_root / name for name in (MANIFEST, REMOVED, SAMPLES, README)]
    batch.require_absent(targets)
    technique_names = list(techniques.TECHNIQUES)
    missing = [name for name in technique_names if name not in batch.treated_copies(dest_root)]
    if missing:
        raise BatchError(f"treated copies not finished: {', '.join(missing)}")
    images, config = batch.read_archive_index(archive)
    selection = batch.plan_selection(archive, images, near_copy_threshold)
    problems = batch.check_selection_on_disk(selection, dest_root)
    if problems:
        raise BatchError(
            "cru/ does not hold the photos this selection keeps "
            f"(near-copy threshold: {near_copy_threshold}): " + "; ".join(problems)
        )
    raw_listing = batch.list_copy(dest_root / batch.RAW_COPY)
    different = [
        name for name in technique_names if batch.list_copy(dest_root / batch.treated_copy_name(name)) != raw_listing
    ]
    if different:
        raise BatchError(f"treated copies with a file list different from cru/: {', '.join(different)}")

    polygon_rows = polygon_files = 0
    for image in selection.kept:
        where = f"{image.split}/labels/{batch.label_name(image.name)}"
        converted = batch.boxes_only_label(archive.read(where).decode("utf-8"))[1]
        polygon_rows += converted
        polygon_files += bool(converted)
    manifest = manifest_rows(dest_root, technique_names)
    removed = removed_rows(archive, selection)
    sheet, sample_names = samples_sheet(dest_root, technique_names)
    text = readme_text(
        dest_root,
        archive_name,
        archive_sha256,
        config,
        selection,
        polygon_rows,
        polygon_files,
        technique_names,
        sample_names,
    )
    batch.write_new(dest_root / MANIFEST, _csv_bytes(MANIFEST_HEADER, manifest))
    batch.write_new(dest_root / REMOVED, _csv_bytes(REMOVED_HEADER, removed))
    batch.write_new(dest_root / SAMPLES, intake.encode_png(sheet))
    batch.write_new(dest_root / README, text.encode("utf-8"))
    return {
        "manifest_rows": len(manifest),
        "removed_rows": len(removed),
        "removed": dict(Counter((REASONS[r.reason], r.image.split) for r in selection.removals)),
        "samples": sample_names,
        "polygon_rows_converted": polygon_rows,
        "label_files_with_polygons": polygon_files,
    }


def check_manifest(dest_root: Path, technique_names: Sequence[str]) -> batch.Check:
    """The manifest on disk must describe exactly the files on disk, hashes included."""
    path = dest_root / MANIFEST
    if not path.is_file():
        return batch.Check(f"{MANIFEST} matches the files on disk", False, "file is missing")
    with path.open(newline="", encoding="utf-8") as handle:
        stored = [row for row in csv.reader(handle)]
    expected = [MANIFEST_HEADER, *[[str(value) for value in row] for row in manifest_rows(dest_root, technique_names)]]
    return batch.Check(
        f"{MANIFEST} matches the files on disk",
        stored == expected,
        f"{len(stored) - 1} rows for {len(technique_names) + 1} copies",
    )
