"""Quality control of zephyr's predictions on videos without a thermistor.

``run`` writes one ``<video stem>.qc.json`` per video with label-free measures of the
network's ordinary prediction, over the whole clip and per window, each with its
pass/fail. ``reference`` (optional) calibrates a checkpoint's onset head once, on
labelled clips it never trained on, for the expected-precision measure.

CLI
---
    zephyr quality-control run --checkpoint best.pt --clips <list.toml>... --cache <dir>
        --out <dir> [--reference qc-reference.json] [--groups <g>...] [--window-s 30]
        [--threshold <measure>=<bound>...]
    zephyr quality-control reference --checkpoint best.pt --clips <list.toml>...
        --cache <dir> --out qc-reference.json [--groups <g>...] [--max-clips N]
"""

import argparse
from pathlib import Path

from ..config import QualityControl, QualityParams, QualityReference, _unchecked_list
from .reference import Reference
from .report import build_reference, load_reports, machine, run

__all__ = ["Reference", "build_reference", "load_reports", "main", "run"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (
        ("run", "Write one quality report per video."),
        (
            "reference",
            "Calibrate a checkpoint's onset head on held-out labelled clips.",
        ),
    ):
        command = sub.add_parser(name, help=help_)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--clips", type=Path, nargs="+", required=True)
        command.add_argument(
            "--groups", nargs="+", help="Keep only these groups of the lists."
        )
        command.add_argument("--cache", type=Path, required=True)
        command.add_argument(
            "--out",
            type=Path,
            required=True,
            help="Report folder (run) or reference file (reference).",
        )
        command.add_argument("--device", help="Default: cuda if any.")
        command.add_argument("--amp", choices=["bf16", "fp16", "off"])
        command.add_argument("--infer-window", type=int, default=1024)
        command.add_argument("--frame-chunk", type=int, default=256)
        if name == "run":
            defaults = QualityParams()
            command.add_argument("--reference", type=Path)
            command.add_argument("--window-s", type=float, default=defaults.window_s)
            command.add_argument(
                "--threshold",
                action="append",
                default=[],
                metavar="MEASURE=BOUND",
                help="Override a pass bound (repeat; BOUND nan: not checked).",
            )
            command.add_argument(
                "--no-traces", action="store_true", help="Skip the parquet traces."
            )
        else:
            command.add_argument("--max-clips", type=int)
    return parser


def _thresholds(pairs: list[str]) -> dict[str, float]:
    out = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--threshold {pair!r} is not MEASURE=BOUND")
        out[name.strip()] = float(value)
    return out


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


def configure(argv: list[str]) -> QualityControl | QualityReference:
    """The validated settings the command line *argv* asks for."""
    args = _parser().parse_args(argv)
    common = {
        "checkpoint": args.checkpoint,
        "features_dir": args.cache,
        "device": args.device,
        "amp": args.amp,
        "infer_window": args.infer_window,
        "frame_chunk": args.frame_chunk,
    }
    clips = _selections(args.clips, args.groups)
    context = {"base": Path.cwd()}
    if args.command == "reference":
        return QualityReference.model_validate(
            common
            | {"out": args.out, "calibration": clips, "max_clips": args.max_clips},
            context=context,
        )
    params = {
        "window_s": args.window_s,
        "thresholds": _thresholds(args.threshold),
        "save_traces": not args.no_traces,
    }
    return QualityControl.model_validate(
        common
        | {
            "output_dir": args.out,
            "reference": args.reference,
            "clips": clips,
            "params": params,
        },
        context=context,
    )


def main(argv: list[str] | None = None) -> None:
    import sys

    config = configure(sys.argv[1:] if argv is None else argv)
    device, amp = machine(config, None, None)
    if isinstance(config, QualityReference):
        build_reference(config, device, amp)
    else:
        written = run(config, device, amp)
        print(f"{len(written)} report(s) in {config.output_dir}")
