"""Quality control of zephyr's predictions on videos without a thermistor.

``reference`` measures, once per checkpoint, what new videos are compared with (a JSON
file shipped beside it); ``run`` writes one ``<video stem>.qc.json`` per video with
label-free measures of its prediction, over the whole clip and per window. Both read a
TOML file (:class:`~zephyr.config.QualityReference`, :class:`~zephyr.config.QualityControl`).

CLI
---
    zephyr quality-control reference <reference.toml> [--device cuda] [--amp bf16]
    zephyr quality-control run <quality-control.toml> [--device cuda] [--amp bf16]
"""

import argparse
from pathlib import Path

from ..config import QualityControl, QualityReference, load
from .reference import Reference
from .report import build_reference, load_reports, machine, run

__all__ = ["Reference", "build_reference", "load_reports", "main", "run"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (
        ("reference", "Measure a checkpoint's reference (QualityReference TOML)."),
        ("run", "Write one quality report per video (QualityControl TOML)."),
    ):
        command = sub.add_parser(name, help=help_)
        command.add_argument("config", type=Path)
        command.add_argument("--device", help="Default: the file's, else cuda if any.")
        command.add_argument("--amp", choices=["bf16", "fp16", "off"])
    args = parser.parse_args(argv)

    if args.command == "reference":
        config = load(QualityReference, args.config)
        build_reference(config, *machine(config, args.device, args.amp))
    else:
        config = load(QualityControl, args.config)
        written = run(config, *machine(config, args.device, args.amp))
        print(f"{len(written)} report(s) in {config.output_dir}")
