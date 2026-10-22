"""The desktop app: a project, from videos to a trained network and its predictions.

A project is a folder (:mod:`.project`): its clip lists, ``experiments/``, ``cache/``,
``runs/`` and ``predictions/``. Pages walk through it in order (Project, Clips, Boxes,
Breathing, Train, Results, Predict); each writes only clip lists and new experiments,
and runs work through the ``zephyr`` commands in a local child process (:mod:`.jobs`).
Needs the ``ui`` extra (PySide6, pyqtgraph) and a display; this module does not.

CLI
---
    zephyr gui                        # choose or create a project
    zephyr gui my-project/            # a project folder
    zephyr gui my-project/clips.toml  # a project, on this clip list
"""

import argparse
from pathlib import Path

UI_HINT = (
    "the zephyr app needs the `ui` extra: `uv sync --extra ui` in a clone, or "
    "`pip install 'zephyr[ui]'`"
)


def need_ui() -> None:
    """Exit with how to install the ``ui`` extra if it is missing."""
    from importlib.util import find_spec

    if find_spec("PySide6") is None or find_spec("pyqtgraph") is None:
        raise SystemExit(UI_HINT)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "path", nargs="?", type=Path, help="Project folder, or a clip list in one."
    )
    args = parser.parse_args(argv)
    need_ui()
    from .app import run_app

    run_app(args.path)
