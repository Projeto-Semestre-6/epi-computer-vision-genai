"""Build the phase-2 dataset: a raw copy without repeated photos and one treated copy per technique.

Usage (from the repository root), in this order:
    .venv/bin/python scripts/build_phase2_dataset.py raw [--near-copy-threshold 2.0]
    .venv/bin/python scripts/build_phase2_dataset.py near-copies --threshold 2.0
    .venv/bin/python scripts/build_phase2_dataset.py treat [--techniques clahe gamma]
    .venv/bin/python scripts/build_phase2_dataset.py finish [--near-copy-threshold 2.0]
    .venv/bin/python scripts/build_phase2_dataset.py verify [--near-copy-threshold 2.0] [--ultralytics-config DIR]

raw          extracts the Roboflow export into <dest>/cru, leaving out exact copies (same SHA-256)
             and, with --near-copy-threshold, near copies across splits. <dest> must be empty.
near-copies  measures near copies across the splits of <dest>/cru and writes a contact sheet and
             the list of pairs; it changes nothing in the copies.
treat        writes <dest>/tratado-<technique> from <dest>/cru, one technique after the other,
             through the same function the endpoint calls; JPEG quality 95.
finish       writes manifesto.csv, removidas.csv, amostras.png and LEIA-ME.md.
verify       checks the copies against the archive and against each other: bytes, labels, class
             names in every data.yaml, and that cru holds exactly the photos the selection keeps
             (pass the --near-copy-threshold that raw was run with). With --ultralytics-config
             it also runs Ultralytics' own dataset reading on every copy.

Outputs go to ~/unisagrado/epi-dataset-fase2/, never into the repository. The script refuses a
target that already has files and never deletes or overwrites anything; the archive is only read.
A photo that cannot be treated stops the run (it is never skipped).
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from epi_imgproc import batch, batch_report, duplicates, evaluation, techniques  # noqa: E402

HOME = Path.home() / "unisagrado"
DEFAULT_ZIP = HOME / "epi-avaliacao" / "dataset-v1.zip"
DEFAULT_DEST = HOME / "epi-dataset-fase2"
PROGRESS_EVERY = 500


def _per_split(counts: dict) -> str:
    return ", ".join(f"{split} {counts.get(split, 0)}" for split in batch.SPLITS)


def command_raw(args: argparse.Namespace) -> int:
    started = time.monotonic()
    batch.require_empty(args.dest)  # fail before hashing the archive
    print(f"archive: {args.zip}  sha256={evaluation.sha256_file(args.zip)}")
    with zipfile.ZipFile(args.zip) as archive:
        images, config = batch.read_archive_index(archive)
        selection = batch.plan_selection(archive, images, args.near_copy_threshold)
        stats = batch.build_raw_copy(archive, selection.kept, config, args.dest)
    removed = Counter((removal.reason, removal.image.split) for removal in selection.removals)

    def left_out(reason: str) -> str:
        return _per_split({split: removed[(reason, split)] for split in batch.SPLITS})

    print(f"images in the archive: {_per_split(Counter(image.split for image in images))} (total {len(images)})")
    print(f"exact copies left out: {left_out(duplicates.EXACT_COPY)}")
    if args.near_copy_threshold is None:
        print("near copies across splits: not removed (no --near-copy-threshold)")
    else:
        print(f"near copies left out (threshold {args.near_copy_threshold:g}): {left_out(duplicates.NEAR_COPY)}")
    written = stats["images"]
    print(f"images written to {args.dest / batch.RAW_COPY}: {_per_split(written)} (total {sum(written.values())})")
    print(
        f"polygon rows turned into boxes: {stats['polygon_rows_converted']} "
        f"in {stats['label_files_with_polygons']} label files"
    )
    print(f"image bytes: {stats['image_bytes']}  elapsed: {time.monotonic() - started:.1f} s")
    return 0


def command_near_copies(args: argparse.Namespace) -> int:
    started = time.monotonic()
    report = batch_report.near_copy_report(args.dest, args.threshold, args.max_distance, args.pairs)
    names = report["class_names"]
    print(f"images in cru/: {_per_split(report['images'])}")
    print(f"cross-split pairs with measure <= {report['max_distance']:g}: {report['pairs_measured']}")
    print("photos that would leave, by threshold (train / valid / test):")
    for threshold, counts in report["removed_by_threshold"].items():
        mark = "  <- proposed" if threshold == report["threshold"] else ""
        print(f"  <= {threshold:>4g}: {counts['train']:5d} / {counts['valid']:4d} / {counts['test']:3d}{mark}")
    print(
        f"at threshold {report['threshold']:g}: would leave {_per_split(report['removed'])}; "
        f"would stay {_per_split(report['remaining'])}"
    )
    print(f"  how the kept copy is turned to match: {report['removed_by_symmetry']}")
    print(
        "  photos within the threshold that stay because their only near copies also leave: "
        f"{report['kept_because_the_only_near_copies_also_leave']}"
    )
    print("boxes per class: before -> lost (per split)")
    for split in ("train", "valid"):
        before, lost = report["boxes_before"][split], report["boxes_lost"][split]
        print(f"  {split}: total {sum(before.values())} -> -{sum(lost.values())}")
        print("    " + ", ".join(f"{name} {before[name]} -> -{lost[name]}" for name in names))
    print(f"contact sheet ({report['sheet_pairs']} pairs): {report['sheet']}")
    print(f"all pairs: {report['pairs_csv']}")
    print(f"elapsed: {time.monotonic() - started:.1f} s")
    return 0


def command_treat(args: argparse.Namespace) -> int:
    names = args.techniques or list(techniques.TECHNIQUES)
    for name in names:  # every target is checked before the first copy starts
        batch.require_empty(args.dest / batch.treated_copy_name(name))
    for name in names:
        started = time.monotonic()

        def progress(done: int, total: int, name: str = name, started: float = started) -> None:
            if done % PROGRESS_EVERY == 0 or done == total:
                print(f"  [{name}] {done}/{total}  {time.monotonic() - started:.0f} s", flush=True)

        info = batch.build_treated_copy(args.dest, name, args.max_pixels, progress)
        print(
            f"{info['copia']}: {sum(info['imagens'].values())} images ({_per_split(info['imagens'])}), "
            f"{info['bytes_imagens']} bytes ({info['bytes_imagens'] / 2**30:.2f} GiB), {info['segundos']} s",
            flush=True,
        )
    return 0


def command_finish(args: argparse.Namespace) -> int:
    started = time.monotonic()
    archive_sha256 = evaluation.sha256_file(args.zip)
    with zipfile.ZipFile(args.zip) as archive:
        result = batch_report.finish(archive, args.zip.name, archive_sha256, args.dest, args.near_copy_threshold)
    print(f"{batch_report.MANIFEST}: {result['manifest_rows']} rows")
    print(f"{batch_report.REMOVED}: {result['removed_rows']} rows {result['removed']}")
    print(f"{batch_report.SAMPLES}: {', '.join(result['samples'])}")
    print(
        f"{batch_report.README}: written; polygon rows turned into boxes in cru/: "
        f"{result['polygon_rows_converted']} in {result['label_files_with_polygons']} files"
    )
    print(f"elapsed: {time.monotonic() - started:.1f} s")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    names = batch.treated_copies(args.dest)
    missing = [name for name in techniques.TECHNIQUES if name not in names]
    with zipfile.ZipFile(args.zip) as archive:
        checks = batch.verify_copies(archive, args.dest, names)
        checks += batch.verify_against_source(archive, args.dest, names, args.near_copy_threshold)
    detail = f"{len(names)} of {len(techniques.TECHNIQUES)}" + (f"; missing: {missing}" if missing else "")
    checks.append(batch.Check("every technique has a finished treated copy", not missing, detail))
    checks.append(batch_report.check_manifest(args.dest, names))
    if args.ultralytics_config is not None:
        evaluation.isolate_ultralytics(args.ultralytics_config)
        for copy_name in [batch.RAW_COPY, *[batch.treated_copy_name(name) for name in names]]:
            listing = batch.list_copy(args.dest / copy_name)
            scans = batch.ultralytics_scan(args.dest / copy_name / "data.yaml")
            for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
                scan = scans[key]
                complete = scan["kept"] == scan["files"] == len(listing[split]) and scan["corrupt"] == 0
                detail = (
                    f"{scan['kept']} of {len(listing[split])} images kept, {scan['corrupt']} corrupt, "
                    f"{scan['boxes']} boxes, {len(scan['messages'])} warning(s) from Ultralytics"
                )
                name = f"Ultralytics reads {copy_name}/{split} without discarding any image"
                checks.append(batch.Check(name, complete, detail))
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'}  {check.name}: {check.detail}")
    failed = sum(not check.passed for check in checks)
    print(f"{len(checks) - failed} of {len(checks)} checks passed")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name: str, handler, needs_zip: bool) -> argparse.ArgumentParser:
        sub = commands.add_parser(name)
        sub.set_defaults(handler=handler)
        sub.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="destination root (outside the repository)")
        if needs_zip:
            sub.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="Roboflow export; only read, never changed")
        return sub

    raw = add("raw", command_raw, needs_zip=True)
    near = add("near-copies", command_near_copies, needs_zip=False)
    treat = add("treat", command_treat, needs_zip=False)
    finish = add("finish", command_finish, needs_zip=True)
    verify = add("verify", command_verify, needs_zip=True)
    for sub in (raw, finish, verify):
        sub.add_argument(
            "--near-copy-threshold",
            type=float,
            default=None,
            help="leave out near copies across splits with measure <= this value "
            "(finish and verify: the value raw was run with)",
        )
    near.add_argument("--threshold", type=float, required=True, help="threshold proposed in the report")
    near.add_argument("--max-distance", type=float, default=8.0, help="largest measure listed and shown")
    near.add_argument("--pairs", type=int, default=40, help="pairs on the contact sheet")
    treat.add_argument(
        "--techniques",
        nargs="+",
        choices=list(techniques.TECHNIQUES),
        default=None,
        help="default: all, in registry order",
    )
    treat.add_argument(
        "--max-pixels",
        type=int,
        default=batch.BATCH_MAX_PIXELS,
        help="pixel limit of the batch (the endpoint keeps its own)",
    )
    verify.add_argument(
        "--ultralytics-config",
        type=Path,
        default=None,
        help="writable folder for Ultralytics settings; enables the Ultralytics dataset reading",
    )
    args = parser.parse_args(argv)

    repository = Path(__file__).resolve().parents[1]
    if args.dest.resolve().is_relative_to(repository):
        print(f"error: {args.dest} is inside the repository; dataset images are never written there", file=sys.stderr)
        return 2
    try:
        return args.handler(args)
    except (batch.BatchError, FileNotFoundError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
