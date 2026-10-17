"""A/B evaluation of the enhancement techniques on the validation split, on CPU.

Usage (from the repository root):
    .venv/bin/python scripts/run_ab_eval.py
    .venv/bin/python scripts/run_ab_eval.py --techniques clahe gamma --limit 50

Outputs go to ~/unisagrado/epi-avaliacao/ (never to runs/ inside the repository).
The test split is never read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from epi_imgproc import evaluation, techniques  # noqa: E402

HOME = Path.home() / "unisagrado"
EVAL_ROOT = HOME / "epi-avaliacao"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=EVAL_ROOT / "dados")
    parser.add_argument("--data-yaml", type=Path, default=None, help="data.yaml to use instead of searching --data-root")
    parser.add_argument("--weights", type=Path, default=HOME / "epi-vini-referencia" / "best.pt")
    parser.add_argument("--work-dir", type=Path, default=EVAL_ROOT / "ab")
    parser.add_argument("--project-dir", type=Path, default=EVAL_ROOT / "runs")
    parser.add_argument("--techniques", nargs="+", choices=list(techniques.TECHNIQUES), default=None)
    parser.add_argument("--limit", type=int, default=None, help="only the first N validation images (smoke runs)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    evaluation.isolate_ultralytics(args.work_dir / ".ultralytics-config")
    try:
        info = evaluation.run_ab_evaluation(
            data_root=args.data_root,
            weights=args.weights,
            work_dir=args.work_dir,
            project_dir=args.project_dir,
            technique_names=args.techniques,
            limit=args.limit,
            batch=args.batch,
            workers=args.workers,
            data_yaml=args.data_yaml,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"checkpoint: {info['checkpoint']['variant']} sha256={info['checkpoint']['sha256']}")
    print(f"images: {info['images']}  remap identity: {info['remap_is_identity']}")
    print("images scored per arm: " + ", ".join(f"{r['technique']}={r['images_seen']}" for r in info["rows"]))
    print(f"baseline pixel mismatches vs cv2.imread: {len(info['baseline_pixel_mismatches'])}")
    print((args.work_dir / "results.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
