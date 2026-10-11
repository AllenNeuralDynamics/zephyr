"""The ``zephyr-dataset`` command: one subcommand per step, each taking the spec.

Steps are resumable and run in this order; ``build`` runs them all. Only machine
settings (``--workers``, ``--limit``, ``--refresh``) are flags; the rest is in the spec.

CLI
---
    zephyr-dataset inventory <spec.toml> [--refresh]   # DocDB -> docdb.json.gz, inventory.parquet
    zephyr-dataset fetch <spec.toml> [--workers 16] [--limit N]
    zephyr-dataset lists <spec.toml>       # sessions.toml
    zephyr-dataset qc <spec.toml>          # marks/spans of unreviewed clips, qc.parquet
    zephyr-dataset select <spec.toml>      # results.json
    zephyr-dataset videos <spec.toml> [--workers 16] [--limit N]   # videos/, cameras.parquet
    zephyr-dataset status <spec.toml>
    zephyr-dataset build <spec.toml>       # all of the above
"""

import argparse
from pathlib import Path

from zephyr.config import load

from .config import DatasetSpec

STEPS = ["inventory", "fetch", "lists", "qc", "select", "videos", "status"]


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
    elif name == "status":
        from . import status

        status.run(spec)


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="zephyr-dataset",
        description=__doc__.splitlines()[0],
        epilog=__doc__.split("CLI\n---\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=[*STEPS, "build"])
    parser.add_argument("spec", type=Path)
    parser.add_argument("--workers", type=int, help="threads for S3 work")
    parser.add_argument(
        "--limit", type=int, help="fetch, videos: only the first N sessions"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="inventory: query DocDB again instead of reading docdb.json.gz",
    )
    args = parser.parse_args(argv)
    spec = load(DatasetSpec, args.spec)
    for name in STEPS if args.command == "build" else [args.command]:
        step(name, spec, args)


if __name__ == "__main__":
    run()
