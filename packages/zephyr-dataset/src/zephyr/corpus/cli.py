"""The ``zephyr-dataset`` command: one subcommand per step, each taking the spec.

Steps are resumable and run in this order; ``build`` runs them all. Only machine
settings (``--workers``, ``--limit``, ``--refresh``, ``--profile``) are flags; the rest
is in the spec.

CLI
---
    zephyr-dataset inventory <spec.toml> [--refresh]   # DocDB -> docdb.json.gz, inventory.parquet
    zephyr-dataset fetch <spec.toml> [--workers 16] [--limit N]
    zephyr-dataset lists <spec.toml>       # sessions.toml
    zephyr-dataset qc <spec.toml>          # marks/spans of unreviewed clips, qc.parquet (local root)
    zephyr-dataset select <spec.toml>      # results.json
    zephyr-dataset videos <spec.toml> [--workers 16] [--limit N]   # videos/, cameras.parquet
    zephyr-dataset sync <spec.toml> [--workers 32]   # sync.parquet: Harp read-dump times
    zephyr-dataset folds <spec.toml>       # folds.parquet: train / val / test A, B, C (after select)
    zephyr-dataset clips <spec.toml> [--workers 8] [--limit N]     # clips/, clips.parquet, clips.toml
    zephyr-dataset cross_validation <spec.toml>   # cross_validation.parquet (after folds)
    zephyr-dataset status <spec.toml>
    zephyr-dataset build <spec.toml>       # all of the above
    zephyr-dataset clips <spec.toml> --profile IAmAnywhere   # AWS profile of an s3:// root
    zephyr-dataset clips <spec.toml> --prune   # delete unassigned sessions' and dropped cameras' clips (delete rights)
    zephyr-dataset clips <spec.toml> --prune --dry-run   # list them, delete nothing
"""

import argparse
import os
from pathlib import Path

from zephyr.config import load

from .config import DatasetSpec

STEPS = [
    "inventory",
    "fetch",
    "lists",
    "qc",
    "select",
    "videos",
    "sync",
    "folds",
    "clips",
    "status",
]
OPTIONAL = ["cross_validation"]
"""Steps ``build`` leaves out: they read the published split and write beside it."""


def step(name: str, spec: DatasetSpec, args: argparse.Namespace) -> None:
    if name == "inventory":
        from . import inventory

        inventory.build(spec, workers=args.workers or 32, refresh=args.refresh)
    elif name == "fetch":
        from . import fetch

        fetch.run(spec, workers=args.workers or 16, limit=args.limit)
    elif name == "lists":
        from . import lists

        lists.run(spec)
    elif name == "qc":
        from . import qc

        qc.run(spec)
    elif name == "select":
        from . import select

        select.run(spec, workers=args.workers or 32)
    elif name == "videos":
        from . import videos

        videos.run(spec, workers=args.workers or 16, limit=args.limit)
    elif name == "sync":
        from . import sync

        sync.run(spec, workers=args.workers or 32, limit=args.limit)
    elif name == "folds":
        from . import folds

        folds.run(spec, workers=args.workers or 6)
    elif name == "clips":
        from . import clips

        clips.run(
            spec,
            workers=args.workers or 8,
            limit=args.limit,
            prune_unassigned=args.prune,
            dry_run=args.dry_run,
        )
    elif name == "status":
        from . import status

        status.run(spec)
    elif name == "cross_validation":
        from . import cross_validation

        cross_validation.run(spec)


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="zephyr-dataset",
        description=__doc__.splitlines()[0],
        epilog=__doc__.split("CLI\n---\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=[*STEPS, *OPTIONAL, "build"])
    parser.add_argument("spec", type=Path)
    parser.add_argument("--workers", type=int, help="threads for S3 work")
    parser.add_argument(
        "--limit",
        type=int,
        help="fetch, videos, sync, clips: only the first N sessions",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="inventory: query DocDB again instead of reading docdb.json.gz",
    )
    parser.add_argument(
        "--profile", help="AWS profile for an s3:// root (sets AWS_PROFILE)"
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="clips: first delete clips of sessions folds.parquet does not assign "
        "and of cameras a [[review]] drops (needs a profile allowed to delete)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="clips --prune: only list the files it would delete",
    )
    args = parser.parse_args(argv)
    if args.dry_run and not args.prune:
        parser.error("--dry-run goes with clips --prune")
    if args.profile:
        os.environ["AWS_PROFILE"] = args.profile
    spec = load(DatasetSpec, args.spec)
    for name in STEPS if args.command == "build" else [args.command]:
        step(name, spec, args)


if __name__ == "__main__":
    run()
