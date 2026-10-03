"""Benchmark baselines for the manuscript, run on the same split as ``zephyr benchmark``.

Paths, splits, epoch budget and seeds come from ``artifacts/benchmark.toml`` so
every method sees exactly the protocol zephyr's own sweep does.

CLI
---
    zephyr baseline pixel   [--methods flow pca snr]
    zephyr baseline facemap [--n-components 100]
    zephyr baseline net     --arch tscan|physnet [--lr 3e-4] [--seeds 17 42] [--dry-run]
    zephyr baseline timing  [--n-clips 8] [--devices cuda cpu]
    zephyr baseline collect
"""

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from .benchmark import (
    DEFAULT_CONFIG,
    METRICS,
    BenchmarkConfig,
    Job,
    Objective,
    evaluate_command,
    load_config,
    train_command,
)

DEFAULT_OUT = Path("benchmarks/baselines-v1")
ZEPHYR_RESULTS = Path("benchmarks/input-objective-v1/results.json")

ARCH_ARGS = {
    # Temporal augmentation off: TS-CAN's motion stream differences consecutive
    # frames, and a stretched window would change what "consecutive" means.
    "tscan": ["--arch", "tscan", "--time-stretch", "1", "--select-jitter", "0"],
    "physnet": [
        "--arch",
        "physnet",
        "--time-stretch",
        "1",
        "--select-jitter",
        "0",
        "--window",
        "128",
        "--scales",
        "1",
    ],
}


def net_config(
    config: BenchmarkConfig, arch: str, out_root: Path, seeds
) -> BenchmarkConfig:
    return replace(
        config,
        output_root=out_root / arch,
        representations=("gray",),
        objectives=(Objective("signal", 1.0, 0.0),),
        seeds=tuple(seeds or config.seeds),
    )


def net_train_command(
    config: BenchmarkConfig, job: Job, arch: str, *, lr, resume: bool
) -> list[str]:
    command = train_command(config, job, resume=resume) + ARCH_ARGS[arch]
    if lr is not None:
        command += ["--lr", str(lr)]
    return command


def run_net(config: BenchmarkConfig, arch: str, *, lr, dry_run: bool) -> None:
    for job in config.jobs():
        run_dir = config.run_dir(job)
        if not (run_dir / "best.pt").exists():
            command = net_train_command(
                config, job, arch, lr=lr, resume=(run_dir / "last.pt").exists()
            )
            print(subprocess.list2cmdline(command), flush=True)
            if not dry_run:
                subprocess.run(command, check=True)
        if not (run_dir / "evaluation.json").exists():
            command = evaluate_command(config, job)
            print(subprocess.list2cmdline(command), flush=True)
            if not dry_run:
                subprocess.run(command, check=True)


NETWORK_CHECKPOINTS = {
    "physnet": "{out}/physnet/runs/gray__signal__seed-17/best.pt",
    "tscan": "{out}/tscan/runs/gray__signal__seed-17/best.pt",
    "zephyr gray": "{zephyr}/runs/gray__multitask__seed-17/best.pt",
    "zephyr gray+diff+flow": "{zephyr}/runs/gray-diff-flow__multitask__seed-17/best.pt",
}
"""Checkpoints timed (seed 17 of each). Inference cost does not depend on the seed."""


def run_timing(
    config: BenchmarkConfig, out_root: Path, *, n_clips: int, devices: list[str]
) -> dict:
    """Time every method per clip on the test split; write ``timing.json``."""
    import numpy as np
    import torch

    from .baselines import timing
    from .dataset import load_manifest

    manifest, entries = load_manifest(
        config.features_dir, config.test_split, config.camera
    )
    clips = timing.pick_clips(entries, n_clips)
    warmup = timing.pick_warmup(entries, clips)
    cached = timing.warm_file_cache([*clips, warmup])
    print(f"file cache warmed: {cached / 1e9:.1f} GB read", flush=True)
    fs = manifest["select_fs_hz"]
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
        for name, template in NETWORK_CHECKPOINTS.items():
            path = Path(template.format(out=out_root, zephyr=ZEPHYR_RESULTS.parent))
            if not path.exists():
                print(f"skipping {name}: {path} not found", flush=True)
                continue
            fn, params_m = timing.network_method(path, device)
            add(name, device_name, fn, params_m)

    data = {
        "split": config.test_split,
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


def _strata(result: dict) -> dict:
    return result.get("strata") or {"all": {"summary": result["summary_by_session"]}}


def _aggregate(method: str, variant: str, results: list[dict]) -> list[dict]:
    rows = []
    for stratum in sorted({s for r in results for s in _strata(r)}):
        row = {
            "method": method,
            "variant": variant,
            "stratum": stratum,
            "n_seeds": len(results),
        }
        for metric in METRICS:
            values = np.array(
                [
                    _strata(r)[stratum]["summary"].get(metric)
                    for r in results
                    if stratum in _strata(r)
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
        "| method | variant | stratum | n | correlation | inhale F1 | exhale F1 | KL-IBI |",
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
            f"| {r['method']} | {r['variant']} | {r['stratum']} | {r['n_seeds']} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _onset_head_rows(runs_dir: Path) -> list[dict]:
    """Zephyr's inhale F1 from the multitask onset head, across seeds.

    The head only scores inhale onsets, so the other metrics are left empty.
    """
    by_variant: dict[str, list[dict]] = {}
    for path in sorted(runs_dir.glob("*/head_evaluation.json")):
        representation, objective, _ = path.parent.name.split("__")
        by_variant.setdefault(
            f"{representation.replace('-', '+')}/{objective}", []
        ).append(json.loads(path.read_text()))
    rows = []
    for variant, results in by_variant.items():
        values = np.array(
            [r["strata"]["all"]["summary"]["head_inhale_f1"] for r in results],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        row = {
            "method": "zephyr (onset head)",
            "variant": variant,
            "stratum": "all",
            "n_seeds": len(results),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = None
            row[f"{metric}_sd"] = None
        row["inhale_f1_mean"] = float(values.mean()) if len(values) else None
        row["inhale_f1_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
        rows.append(row)
    return rows


def _best_markdown(rows: list[dict]) -> str:
    """Each method's best score per metric over its variants, 'all' stratum only.

    Not a like-for-like ranking: variants are picked by their own test score, so
    this shows what each method can reach, not what a tuned run would report.
    """
    lines = [
        "## Best per method (all sessions)",
        "",
        "| method | correlation | inhale F1 | exhale F1 | KL-IBI (lower is better) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for method in dict.fromkeys(r["method"] for r in rows):
        members = [r for r in rows if r["method"] == method and r["stratum"] == "all"]
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


def collect(out_root: Path, zephyr_results: Path | None) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(out_root.glob("pixel/*/*/evaluation.json")):
        rows += _aggregate(
            f"pixel-{path.parent.parent.name}",
            path.parent.name,
            [json.loads(path.read_text())],
        )
    for path in sorted(out_root.glob("facemap/*/evaluation.json")):
        rows += _aggregate("facemap", path.parent.name, [json.loads(path.read_text())])
    for arch in ("tscan", "physnet"):
        results = [
            json.loads(p.read_text())
            for p in sorted(out_root.glob(f"{arch}/runs/*/evaluation.json"))
        ]
        if results:
            rows += _aggregate(arch, "-", results)
    if zephyr_results is not None and zephyr_results.exists():
        for r in json.loads(zephyr_results.read_text())["summary_across_seeds"]:
            rows.append(
                {
                    "method": "zephyr",
                    "variant": f"{r['representation']}/{r['objective']}",
                    "stratum": r["stratum"],
                    "n_seeds": r["n_seeds"],
                }
                | {
                    f"{m}_{s}": r.get(f"{m}_{s}")
                    for m in METRICS
                    for s in ("mean", "sd")
                }
            )
        rows += _onset_head_rows(zephyr_results.parent / "runs")
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pixel")
    p.add_argument("--methods", nargs="+", default=["flow", "pca", "snr"])
    p.add_argument("--bin", type=int, default=2)
    f = sub.add_parser("facemap")
    f.add_argument("--n-components", type=int, default=100)
    f.add_argument("--samples-per-clip", type=int, default=2000)
    f.add_argument("--bin", type=int, default=2)
    f.add_argument("--max-lag", type=int, default=15, help="Output frames.")
    f.add_argument("--lag-step", type=int, default=3)
    f.add_argument("--seed", type=int, default=0)
    n = sub.add_parser("net")
    n.add_argument("--arch", required=True, choices=sorted(ARCH_ARGS))
    n.add_argument("--lr", type=float)
    n.add_argument("--seeds", type=int, nargs="+")
    n.add_argument("--dry-run", action="store_true")
    t = sub.add_parser("timing")
    t.add_argument("--n-clips", type=int, default=8)
    t.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    sub.add_parser("collect")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.command == "net":
        run_net(
            net_config(config, args.arch, args.out_root, args.seeds),
            args.arch,
            lr=args.lr,
            dry_run=args.dry_run,
        )
        return
    if args.command == "timing":
        run_timing(config, args.out_root, n_clips=args.n_clips, devices=args.devices)
        return
    if args.command == "collect":
        collect(args.out_root, ZEPHYR_RESULTS)
        return

    from .baselines import run
    from .dataset import load_manifest

    manifest, train = load_manifest(
        config.features_dir, config.train_split, config.camera
    )
    _, test = load_manifest(config.features_dir, config.test_split, config.camera)
    scoring = {
        "packaged_root": config.packaged_root,
        "split": config.test_split,
        "split_manifest": config.split_manifest,
    }
    if args.command == "pixel":
        run.run_pixel(
            train,
            test,
            methods=args.methods,
            out_root=args.out_root,
            fs=manifest["select_fs_hz"],
            output_fs=manifest["output_fs_hz"],
            bin_factor=args.bin,
            scoring=scoring,
        )
    else:
        run.run_facemap(
            train,
            test,
            out_root=args.out_root,
            n_components=args.n_components,
            samples_per_clip=args.samples_per_clip,
            bin_factor=args.bin,
            max_lag=args.max_lag,
            lag_step=args.lag_step,
            seed=args.seed,
            scoring=scoring,
        )
