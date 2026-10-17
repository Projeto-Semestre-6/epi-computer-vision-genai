"""Presentation material: before/after of 3 validation images per technique, FFT spectra, results table.

Usage (from the repository root):
    .venv/bin/python scripts/make_presentation.py              # 3 validation images from dados/
    .venv/bin/python scripts/make_presentation.py --synthetic  # synthetic scene, into apresentacao/sintetico/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from epi_imgproc import evaluation, intake, presentation  # noqa: E402

EVAL_ROOT = Path.home() / "unisagrado" / "epi-avaliacao"

SYNTHETIC_NOTE = """# Material sintético

Estas figuras usam uma cena **sintética** gerada por código (`presentation.synthetic_scene`), não imagens do dataset.
Servem só para mostrar o método (antes/depois e espectros). Não use como evidência de desempenho.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=EVAL_ROOT / "dados")
    parser.add_argument("--data-yaml", type=Path, default=None, help="data.yaml to use instead of searching --data-root")
    parser.add_argument("--out-dir", type=Path, default=EVAL_ROOT / "apresentacao")
    parser.add_argument("--results", type=Path, default=EVAL_ROOT / "ab" / "results.md")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)

    if args.synthetic:
        out_dir = args.out_dir / "sintetico"
        sources = [(f"cena-sintetica-{seed}", presentation.synthetic_scene(seed=seed)) for seed in range(args.count)]
        written = presentation.make_figures(sources, out_dir, note="(sintetico)")
        (out_dir / "LEIA-ME.md").write_text(SYNTHETIC_NOTE)
    else:
        try:
            data_yaml = args.data_yaml or evaluation.find_data_yaml(args.data_root)
        except FileNotFoundError as exc:
            print(f"error: {exc}; use --synthetic to illustrate the method without data", file=sys.stderr)
            return 2
        val_dir = evaluation.resolve_split_dir(data_yaml, yaml.safe_load(data_yaml.read_text()), "val")
        chosen = presentation.select_evenly(evaluation.list_images(val_dir), args.count)
        sources = [(path.stem, intake.decode_image(path.read_bytes()).pixels) for path in chosen]
        out_dir = args.out_dir
        written = presentation.make_figures(sources, out_dir)
        (out_dir / "imagens-escolhidas.txt").write_text("\n".join(str(path) for path in chosen) + "\n")
    written.append(presentation.copy_results_table(args.results, out_dir))
    print(f"{len(written)} files under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
