"""Quality control of zephyr's predictions on videos without a thermistor.

``zephyr predict --quality-control`` writes one ``<video stem>.qc.json`` per video beside
its prediction: label-free measures of the prediction, over the whole clip and per
window, each placed among the checkpoint's validation windows (scores, never a verdict;
the same with or without a thermistor). Those windows are the checkpoint's reference,
which training writes; ``reference`` makes one for a checkpoint without it.

CLI
---
    zephyr quality-control reference --checkpoint best.pt --clips <list.toml>...
        --cache <dir> --out qc-reference.json [--groups <g>...] [--max-clips N]
"""

import argparse
from pathlib import Path

from ..config import QualityReference, _unchecked_list
from .reference import Reference
from .report import build_reference, load_reports, machine, quality_report

__all__ = ["Reference", "build_reference", "load_reports", "main", "quality_report"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser(
        "reference",
        help="Measure a checkpoint once on held-out labelled clips (its validation "
        "animals).",
    )
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--clips", type=Path, nargs="+", required=True)
    command.add_argument(
        "--groups", nargs="+", help="Keep only these groups of the lists."
    )
    command.add_argument("--cache", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True, help="Reference file.")
    command.add_argument("--device", help="Default: cuda if any.")
    command.add_argument("--amp", choices=["bf16", "fp16", "off"])
    command.add_argument("--infer-window", type=int, default=1024)
    command.add_argument("--frame-chunk", type=int, default=256)
    command.add_argument("--max-clips", type=int)
    return parser


def _selections(paths: list[Path], groups: list[str] | None) -> list[dict]:
    """One selection per list, each keeping the *groups* it holds (all without)."""
    if groups is None:
        return [{"clips": path} for path in paths]
    out, found = [], set()
    for path in paths:
        present = {c.group for c in _unchecked_list(path.resolve()).resolve()}
        mine = [g for g in groups if g in present]
        found |= set(mine)
        if mine:
            out.append({"clips": path, "groups": mine})
    missing = sorted(set(groups) - found)
    if missing:
        raise SystemExit(f"--groups {missing} are in none of the clip lists")
    return out


def configure(argv: list[str]) -> QualityReference:
    """The validated settings the command line *argv* asks for."""
    args = _parser().parse_args(argv)
    return QualityReference.model_validate(
        {
            "checkpoint": args.checkpoint,
            "features_dir": args.cache,
            "device": args.device,
            "amp": args.amp,
            "infer_window": args.infer_window,
            "frame_chunk": args.frame_chunk,
            "out": args.out,
            "calibration": _selections(args.clips, args.groups),
            "max_clips": args.max_clips,
        },
        context={"base": Path.cwd()},
    )


def main(argv: list[str] | None = None) -> None:
    import sys

    config = configure(sys.argv[1:] if argv is None else argv)
    device, amp = machine(config, None, None)
    build_reference(config, device, amp)
