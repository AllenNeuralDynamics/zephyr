"""Run an experiment: train each of its (fold, seed), then score the fold's test groups.

Each (fold, seed) trains into ``{output_dir}/{fold name}/seed-{seed}/``, which receives
the resolved config, training video list, checkpoints and ``evaluation.json`` (and
``traces/`` with ``save_traces`` or ``--save-traces``). Machine
settings (device, precision, workers) come from the experiment or CLI, never a fold:
they change how fast a result arrives, not what it is. What is trained and how it is
loaded is a :class:`Runner`, zephyr's own unless a caller passes another.

CLI
---
    zephyr run examples/experiment.toml
    zephyr run examples/experiment.toml --smoke
    zephyr run examples/experiment.toml --folds fold --seeds 1
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import Experiment, Fold, load
from .evaluate import TRACES, Loader, evaluate_groups, load_checkpoint
from .train import ZEPHYR, Architecture, Machine, OnScored, train_fold


@dataclass(frozen=True)
class Runner:
    """What the experiment file loads as, what trains, and how checkpoints load."""

    experiment: type[Experiment] = Experiment
    architecture: Callable[[Fold], Architecture] = lambda fold: ZEPHYR
    loader: Loader = load_checkpoint


ZEPHYR_RUNNER = Runner()


def smoke_fold(fold: Fold) -> Fold:
    """The same fold, cut to two optimiser steps: proves the plumbing, not the science."""
    params = fold.train_params.model_copy(update={"epochs": 1, "steps_per_epoch": 2})
    return fold.model_copy(update={"train_params": params})


def needs_training(run_dir: Path, resume: bool) -> bool:
    """Whether *run_dir* still trains: it has no ``best.pt`` yet, or, with *resume*, it
    stopped part way (a ``last.pt`` but no ``evaluation.json``, which is written only
    once training ended); a finished run is only scored again."""
    if not (run_dir / "best.pt").exists():
        return True
    stopped = (run_dir / "last.pt").exists() and not (
        run_dir / "evaluation.json"
    ).exists()
    return resume and stopped


def run_fold(
    fold: Fold,
    seed: int,
    run_dir: Path,
    features_dir: Path,
    machine: Machine,
    *,
    smoke: bool = False,
    resume: bool = False,
    runner: Runner = ZEPHYR_RUNNER,
    save_traces: bool = False,
    on_scored: OnScored | None = None,
) -> None:
    """Train one fold with one seed, then evaluate its test groups (writing the traces
    scored into ``traces/`` with *save_traces*; *on_scored* goes to ``train_fold``)."""
    if smoke:
        fold = smoke_fold(fold)
    if fold.init_from is not None and not fold.init_from.is_file():
        raise SystemExit(f"init_from checkpoint {fold.init_from} does not exist")
    best = run_dir / "best.pt"
    if not needs_training(run_dir, resume):
        print(f"{run_dir}: already trained, skipping to evaluation", flush=True)
    else:
        train_fold(
            fold,
            seed,
            run_dir,
            features_dir,
            machine,
            resume=resume,
            architecture=runner.architecture(fold),
            on_scored=on_scored,
        )
    params = fold.train_params
    evaluate_groups(
        [best],
        fold.test_clips(),
        features_dir,
        fold.preprocess,
        device=machine.device,
        amp=machine.amp,
        window=params.infer_window,
        frame_chunk=params.frame_chunk,
        out=run_dir / "evaluation.json",
        loader=runner.loader,
        traces_dir=run_dir / TRACES if save_traces else None,
    )


def choose_machine(args: argparse.Namespace, defaults: dict) -> Machine:
    """CLI flag, else the experiment's value, else a default -- per setting.

    An experiment that leaves a setting out loads it as ``None``, so each
    fallback tests for ``None`` rather than for a missing key.
    """

    def pick(name: str, default):
        for value in (getattr(args, name), defaults.get(name)):
            if value is not None:
                return value
        return default

    return Machine(
        device=pick("device", "cuda" if torch.cuda.is_available() else "cpu"),
        amp=pick("amp", "bf16"),
        num_workers=pick("num_workers", 4),
    )


def main(
    argv: list[str] | None = None,
    *,
    runner: Runner = ZEPHYR_RUNNER,
    on_scored: OnScored | None = None,
) -> None:
    """Run the experiment *argv* names; *on_scored* (Python callers only, e.g. the GUI)
    sees each validation scoring of every fold and seed."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path, help="Experiment TOML file.")
    parser.add_argument(
        "--folds", nargs="+", help="Run only the folds with these names (default: all)."
    )
    parser.add_argument(
        "--cache", type=Path, help="Feature cache (default: the experiment's)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Run directory root (default: the experiment's).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Seeds to run (default: the experiment's).",
    )
    parser.add_argument("--device", help="Default: the experiment's, else cuda if any.")
    parser.add_argument("--amp", choices=["bf16", "fp16", "off"])
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Cut every fold to 1 epoch of 2 steps, to check the plumbing.  "
        "Writes under {output dir}/smoke/ unless --output-dir is given.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue runs that have a last.pt instead of refusing the directory.",
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Also write each test clip's trace into traces/ (or the experiment's "
        "save_traces); it changes no score.",
    )
    args = parser.parse_args(argv)

    experiment = load(runner.experiment, args.config)
    try:
        folds = experiment.folds(args.folds)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    seeds = args.seeds or experiment.seeds
    features_dir = args.cache or experiment.features_dir
    output_dir = args.output_dir or experiment.output_dir
    defaults = {
        "device": experiment.device,
        "amp": experiment.amp,
        "num_workers": experiment.num_workers,
    }

    machine = choose_machine(args, defaults)
    if args.smoke and args.output_dir is None:
        # A finished run is skipped straight to evaluation, so a smoke best.pt in
        # the real directory would later be reported as the trained model.
        output_dir = output_dir / "smoke"

    for fold in folds:
        for seed in seeds:
            run_dir = output_dir.resolve() / fold.name / f"seed-{seed}"
            print(f"\n=== {fold.name}  seed {seed}  -> {run_dir}", flush=True)
            run_fold(
                fold,
                seed,
                run_dir,
                features_dir.resolve(),
                machine,
                smoke=args.smoke,
                resume=args.resume,
                runner=runner,
                save_traces=args.save_traces or experiment.save_traces,
                on_scored=on_scored,
            )


if __name__ == "__main__":
    main()
