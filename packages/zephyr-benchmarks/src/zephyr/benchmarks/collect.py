"""Benchmark baselines for the manuscript, run on the same fold as the networks.

A fold of an experiment supplies the clips: its training lists fit each baseline and
its test groups score it, so every method sees the protocol the networks do.
Network baselines (TS-CAN, PhysNet) are trained with ``zephyr-benchmarks run`` on
a fold whose ``train_params.arch`` names them.

CLI
---
    zephyr-benchmarks pixel   EXPERIMENT [--fold NAME] [--methods flow pca snr]
    zephyr-benchmarks facemap EXPERIMENT [--fold NAME] [--n-components 100]
    zephyr-benchmarks timing  EXPERIMENT [--fold NAME] --checkpoint NAME=best.pt ...
    zephyr-benchmarks collect --runs EXPERIMENT_OUTPUT_DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np

from zephyr import features
from zephyr.config import Fold, load

from .config import BenchmarkExperiment

METRICS = ("correlation", "inhale_f1", "exhale_f1", "kl_ibi")
DEFAULT_OUT = Path("benchmarks/baselines")


def _entries(fold: Fold, cache: Path) -> tuple[list, dict[str, list]]:
    """The fold's training entries and its labelled test entries per group."""
    recipe = fold.preprocess
    train = [
        e
        for clips in fold.train_clips()
        for e in features.require(clips, recipe, cache)
    ]
    groups = {
        name: features.require([c for c in clips if c.labelled], recipe, cache)
        for name, clips in fold.test_clips().items()
    }
    return train, groups


def run_timing(
    fold: Fold,
    cache: Path,
    out_root: Path,
    checkpoints: dict[str, Path],
    *,
    n_clips: int,
    devices: list[str],
) -> dict:
    """Time every method per clip on the fold's test clips; write ``timing.json``."""
    import torch

    from . import timing

    _, groups = _entries(fold, cache)
    entries = [e for members in groups.values() for e in members]
    clips = timing.pick_clips(entries, n_clips)
    warmup = timing.pick_warmup(entries, clips)
    cached = timing.warm_file_cache([*clips, warmup])
    print(f"file cache warmed: {cached / 1e9:.1f} GB read", flush=True)
    fs = fold.preprocess.select_fs
    methods: list[dict] = []

    def add(name: str, device: str, fn, params_m: float | None = None) -> None:
        print(f"timing {name} on {device} ({len(clips)} clips)", flush=True)
        result = timing.time_method(fn, clips, warmup=warmup)
        methods.append(
            {"method": name, "device": device, "params_m": params_m} | result
        )
        print(f"  mean {result['summary']['mean_s']:.2f} s/clip", flush=True)

    for method in ("flow", "pca", "snr"):
        add(f"pixel {method}", "cpu", timing.pixel_method(method, fs, 2))
    lags = np.arange(-15, 16, 3)
    for variant in ("motion", "movie", "both"):
        add(
            f"facemap-style {variant}",
            "cpu",
            timing.facemap_method(
                variant,
                out_root / "facemap" / "bases.npz",
                n_components=100,
                lags=lags,
                seed=0,
            ),
        )
    for device_name in devices:
        if device_name == "cuda" and not torch.cuda.is_available():
            print("skipping cuda timing: no GPU", flush=True)
            continue
        device = torch.device(device_name)
        for name, path in checkpoints.items():
            fn, params_m = timing.network_method(path, device)
            add(name, device_name, fn, params_m)

    data = {
        "n_clips": len(clips),
        "clip_seconds": float(np.mean([e.n_output / timing.CLIP_FS for e in clips])),
        "clip_ids": [e.clip_id for e in clips],
        "hardware": timing.hardware(),
        "flow_preprocessing": timing.estimate_flow_preprocessing(clips[0]),
        "methods": methods,
    }
    timing.write(out_root / "timing.json", data)
    print(f"wrote {out_root / 'timing.json'}")
    return data


def _timing_markdown(timing_path: Path) -> str:
    data = json.loads(timing_path.read_text())
    seconds = data["clip_seconds"]
    lines = [
        "## Inference time per clip",
        "",
        (
            f"Mean ± sd over {data['n_clips']} test clips (each {seconds:.0f} s of "
            "video), from the preprocessed crops on disk to the finished trace. "
            "Video decoding and preprocessing are not included."
        ),
        "",
        "| method | device | params (M) | seconds per clip | x real time |",
        "| --- | --- | --- | --- | --- |",
    ]
    for m in data["methods"]:
        s = m["summary"]
        spread = f" ± {s['sd_s']:.2f}" if s.get("sd_s") is not None else ""
        params = "—" if m.get("params_m") is None else f"{m['params_m']:.2f}"
        lines.append(
            f"| {m['method']} | {m['device']} | {params} | "
            f"{s['mean_s']:.2f}{spread} | {seconds / s['mean_s']:.0f}x |"
        )
    flow = data.get("flow_preprocessing")
    if flow:
        lines += [
            "",
            (
                "Optical flow is computed in preprocessing, not above: about "
                f"{flow['seconds_per_clip']:.0f} s per clip "
                f"({flow['per_frame_ms']:.2f} ms per frame, CPU). Add it for any "
                "method that reads the flow channel (pixel flow, zephyr with flow)."
            ),
        ]
    return "\n".join(lines) + "\n"


def _aggregate(method: str, variant: str, results: list[dict]) -> list[dict]:
    """One row per group: each metric's mean and sd across *results* (seeds)."""
    rows = []
    for group in sorted({g for r in results for g in r["groups"]}):
        row = {
            "method": method,
            "variant": variant,
            "group": group,
            "n_seeds": len(results),
        }
        for metric in METRICS:
            values = np.array(
                [
                    r["groups"][group]["summary"].get(metric)
                    for r in results
                    if group in r["groups"]
                ],
                dtype=float,
            )
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
        rows.append(row)
    return rows


def _markdown(rows: list[dict]) -> str:
    lines = [
        "| method | variant | group | n | correlation | inhale F1 | exhale F1 | KL-IBI |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        cells = []
        for metric in METRICS:
            mean, sd = r[f"{metric}_mean"], r[f"{metric}_sd"]
            cells.append(
                "—" if mean is None else f"{mean:.3f}" + (f" ± {sd:.3f}" if sd else "")
            )
        lines.append(
            f"| {r['method']} | {r['variant']} | {r['group']} | {r['n_seeds']} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _onset_head_row(fold: str, results: list[dict]) -> dict | None:
    """Inhale F1 from the multitask onset head, across seeds.

    The head only scores inhale onsets, so the other metrics are left empty.
    """
    values = np.array(
        [r["groups"]["all"]["summary"].get("head_inhale_f1") for r in results],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    row = {
        "method": "zephyr (onset head)",
        "variant": fold,
        "group": "all",
        "n_seeds": len(results),
    }
    for metric in METRICS:
        row[f"{metric}_mean"] = None
        row[f"{metric}_sd"] = None
    row["inhale_f1_mean"] = float(values.mean())
    row["inhale_f1_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
    return row


def _best_markdown(rows: list[dict]) -> str:
    """Each method's best score per metric over its variants, 'all' group only.

    Not a like-for-like ranking: variants are picked by their own test score, so
    this shows what each method can reach, not what a tuned run would report.
    """
    lines = [
        "## Best per method (all recordings)",
        "",
        "| method | correlation | inhale F1 | exhale F1 | KL-IBI (lower is better) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for method in dict.fromkeys(r["method"] for r in rows):
        members = [r for r in rows if r["method"] == method and r["group"] == "all"]
        cells = []
        for metric in METRICS:
            scored = [r for r in members if r.get(f"{metric}_mean") is not None]
            if not scored:
                cells.append("—")
                continue
            pick = (min if metric == "kl_ibi" else max)(
                scored, key=lambda r: r[f"{metric}_mean"]
            )
            cells.append(f"{pick[f'{metric}_mean']:.3f} ({pick['variant']})")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def collect(out_root: Path, runs_dir: Path | None) -> list[dict]:
    """Tabulate every baseline under *out_root* and the runs under *runs_dir*.

    *runs_dir* is a ``zephyr run`` or ``zephyr-benchmarks run`` output
    directory (``{fold}/seed-{n}/``); each fold is one row set, seeds
    aggregated, and a fold trained as a baseline architecture is labelled with it.
    """
    rows: list[dict] = []
    for path in sorted(out_root.glob("pixel/*/*/evaluation.json")):
        rows += _aggregate(
            f"pixel-{path.parent.parent.name}",
            path.parent.name,
            [json.loads(path.read_text())],
        )
    for path in sorted(out_root.glob("facemap/*/evaluation.json")):
        rows += _aggregate("facemap", path.parent.name, [json.loads(path.read_text())])
    if runs_dir is not None and runs_dir.exists():
        for fold_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            paths = sorted(fold_dir.glob("seed-*/evaluation.json"))
            if not paths:
                continue
            results = [json.loads(p.read_text()) for p in paths]
            config = paths[0].parent / "config.json"
            arch = "zephyr"
            if config.exists():
                params = json.loads(config.read_text())["fold"]
                arch = params.get("train_params", {}).get("arch", "zephyr")
            rows += _aggregate(arch, fold_dir.name, results)
            head = _onset_head_row(fold_dir.name, results)
            if head is not None:
                rows.append(head)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "results.json").write_text(json.dumps(rows, indent=2))
    text = _markdown(rows) + "\n" + _best_markdown(rows)
    if (out_root / "timing.json").exists():
        text += "\n" + _timing_markdown(out_root / "timing.json")
    (out_root / "results.md").write_text(text, encoding="utf-8")
    print(f"wrote {out_root / 'results.md'}")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("pixel", "facemap", "timing"):
        command = sub.add_parser(name)
        command.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
        command.add_argument("experiment", type=Path)
        command.add_argument(
            "--fold", help="Which of its folds (needed when it has several)."
        )
        command.add_argument(
            "--cache", type=Path, help="Feature cache (default: the experiment's)."
        )
        if name == "pixel":
            command.add_argument("--methods", nargs="+", default=["flow", "pca", "snr"])
            command.add_argument("--bin", type=int, default=2)
        elif name == "facemap":
            command.add_argument("--n-components", type=int, default=100)
            command.add_argument("--samples-per-clip", type=int, default=2000)
            command.add_argument("--bin", type=int, default=2)
            command.add_argument(
                "--max-lag", type=int, default=15, help="Output frames."
            )
            command.add_argument("--lag-step", type=int, default=3)
            command.add_argument("--seed", type=int, default=0)
        else:
            command.add_argument("--n-clips", type=int, default=8)
            command.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
            command.add_argument(
                "--checkpoint",
                action="append",
                default=[],
                metavar="NAME=PATH",
                help="A network to time; repeat for several.",
            )
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    collect_parser.add_argument("--runs", type=Path, help="A run output directory.")
    args = parser.parse_args(argv)

    if args.command == "collect":
        collect(args.out_root, args.runs)
        return

    experiment = load(BenchmarkExperiment, args.experiment)
    if args.fold is None and len(experiment.fold) > 1:
        raise SystemExit(
            f"{args.experiment} has folds {[f.name for f in experiment.fold]}; "
            "choose one with --fold"
        )
    try:
        (fold,) = experiment.folds(None if args.fold is None else [args.fold])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.cache = args.cache or experiment.features_dir
    if args.command == "timing":
        checkpoints = {
            name: Path(path)
            for name, _, path in (c.partition("=") for c in args.checkpoint)
        }
        run_timing(
            fold,
            args.cache,
            args.out_root,
            checkpoints,
            n_clips=args.n_clips,
            devices=args.devices,
        )
        return

    from . import baselines

    train, groups = _entries(fold, args.cache)
    if args.command == "pixel":
        baselines.run_pixel(
            train,
            groups,
            methods=args.methods,
            out_root=args.out_root,
            fs=fold.preprocess.select_fs,
            output_fs=fold.preprocess.output_fs,
            bin_factor=args.bin,
        )
    else:
        baselines.run_facemap(
            train,
            groups,
            out_root=args.out_root,
            n_components=args.n_components,
            samples_per_clip=args.samples_per_clip,
            bin_factor=args.bin,
            max_lag=args.max_lag,
            lag_step=args.lag_step,
            seed=args.seed,
        )
