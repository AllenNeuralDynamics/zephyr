"""The ``zephyr-benchmarks`` command: one subcommand per benchmark step.

Each subcommand forwards its arguments, unparsed, to that module's own ``main()``;
``zephyr-benchmarks <subcommand> --help`` prints its flags. ``run`` and ``evaluate``
are zephyr's own, taking the same files plus ``train_params.arch``.

CLI
---
    zephyr-benchmarks run <experiment.toml> [--smoke] [--folds <name>...]
    zephyr-benchmarks evaluate --checkpoint <best.pt> --clips <list.toml>... --cache <dir>
    zephyr-benchmarks pixel|facemap|timing <experiment.toml> [--fold <name>] ...
    zephyr-benchmarks collect --runs <dir>
    zephyr-benchmarks report --runs <dir> --out <dir>
    zephyr-benchmarks occlusion --checkpoint <best.pt> --clips <list.toml>... --cache <dir> --out <file.npz>
"""

import argparse


def _run(args: list[str]) -> None:
    from zephyr import run

    from .runner import RUNNER

    run.main(args, runner=RUNNER)


def _evaluate(args: list[str]) -> None:
    from zephyr import evaluate

    from .runner import load_checkpoint

    evaluate.main(args, loader=load_checkpoint)


def _collect(command: str):
    def forward(args: list[str]) -> None:
        from . import collect

        collect.main([command, *args])

    return forward


def _occlusion(args: list[str]) -> None:
    from . import occlusion

    occlusion.main(args)


def _report(args: list[str]) -> None:
    from . import report

    report.main(args)


COMMANDS = {
    "run": _run,
    "evaluate": _evaluate,
    "pixel": _collect("pixel"),
    "facemap": _collect("facemap"),
    "timing": _collect("timing"),
    "collect": _collect("collect"),
    "report": _report,
    "occlusion": _occlusion,
}


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="zephyr-benchmarks",
        description=__doc__.splitlines()[0],
        epilog=__doc__.split("CLI\n---\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    COMMANDS[parsed.command](parsed.args)


if __name__ == "__main__":
    run()
