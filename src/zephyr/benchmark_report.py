"""Build the final benchmark report and directly score the auxiliary onset head.

The ordinary challenge evaluation derives inhalation events from the reconstructed
breathing trace.  This module adds a separate, explicitly labelled diagnostic for
the multitask models' onset head, then combines it with the challenge metrics and
the training histories in publication-oriented figures and a Markdown table.

Examples
--------
    zephyr benchmark-report head
    zephyr benchmark-report head --job gray__multitask__seed-17
    zephyr benchmark-report report
    zephyr benchmark-report all
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

from .benchmark import DEFAULT_CONFIG, BenchmarkConfig, Job, load_config
from .dataset import load_manifest
from .evaluate import load_checkpoint, load_test_strata, truth_frame
from .evaluation import score_clip
from .infer import predict_clip
from .signal import BREATHING_SIGNAL_COLUMN, TIME_COLUMN

HEAD_FILENAME = "head_evaluation.json"
HEAD_THRESHOLD = 0.5
HEAD_MIN_DISTANCE_S = 0.05
REPRESENTATION_LABELS = {
    "gray": "Gray",
    "gray+diff": "Gray + diff",
    "gray+flow": "Gray + flow",
    "gray+diff+flow": "Gray + diff + flow",
}
OBJECTIVE_LABELS = {
    "signal": "Signal only",
    "multitask": "Signal + event detection",
}
STRATUM_LABELS = {
    "new_animals": "Held-out animals",
    "known_animals_new_date": "Held-out sessions",
}
COLORS = {"signal": "#0072B2", "multitask": "#D55E00"}
MARKERS = {"signal": "o", "multitask": "D"}


def _write_json_atomic(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def _head_event_indices(probability: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(probability) != len(times):
        raise ValueError("onset probability and time arrays have different lengths")
    if len(times) < 2:
        return np.empty(0, dtype=int)
    sampling_rate = 1.0 / float(np.median(np.diff(times)))
    distance = max(1, int(round(HEAD_MIN_DISTANCE_S * sampling_rate)))
    peaks, _ = find_peaks(probability, height=HEAD_THRESHOLD, distance=distance)
    return peaks.astype(int)


def _head_summary(rows: list[dict]) -> dict[str, float | int]:
    values = np.asarray([row["head_inhale_f1"] for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return {
        "head_inhale_f1": float(values.mean()) if len(values) else float("nan"),
        "head_inhale_f1_sd": (
            float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        ),
        "head_inhale_f1_n": len(values),
    }


def _aggregate_head_sessions(rows: list[dict]) -> list[dict]:
    result = []
    for session_idx in sorted({int(row["session_idx"]) for row in rows}):
        members = [row for row in rows if int(row["session_idx"]) == session_idx]
        result.append(
            {
                "session_idx": session_idx,
                "n_clips": len(members),
                **_head_summary(members),
            }
        )
    return result


def evaluate_head_job(
    config: BenchmarkConfig,
    job: Job,
    *,
    force: bool = False,
) -> Path | None:
    if job.objective.w_onset <= 0:
        return None

    run_dir = config.run_dir(job)
    output = run_dir / HEAD_FILENAME
    if output.is_file() and not force:
        print(f"skip {job.job_id}: {HEAD_FILENAME} already exists", flush=True)
        return output

    checkpoint = run_dir / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")

    device = torch.device(config.device)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[config.amp]
    model, mean, std, _ = load_checkpoint(checkpoint, device)
    _, entries = load_manifest(config.features_dir, config.test_split, config.camera)
    strata = load_test_strata(config.split_manifest)
    sessions = sorted({index for values in strata.values() for index in values})
    entries = [entry for entry in entries if entry.session_idx in set(sessions)]

    print(
        f"head evaluation {job.job_id}: {len(entries)} clips / "
        f"{len(sessions)} sessions",
        flush=True,
    )
    rows = []
    for clip_number, entry in enumerate(
        sorted(entries, key=lambda value: (value.session_idx, value.part)), start=1
    ):
        signal, onset_probability = predict_clip(
            model,
            entry,
            mean,
            std,
            device=device,
            amp_dtype=amp_dtype,
        )
        times = np.load(entry.times)
        n = min(len(times), len(signal), len(onset_probability))
        times = times[:n]
        signal = signal[:n]
        onset_probability = onset_probability[:n]
        head_indices = _head_event_indices(onset_probability, times)
        head_times = times[head_indices]

        predicted = pd.DataFrame(
            {
                TIME_COLUMN: times.astype(np.float64),
                BREATHING_SIGNAL_COLUMN: signal.astype(np.float64),
            }
        )
        score = score_clip(
            truth_frame(
                config.packaged_root,
                config.test_split,
                entry.session_idx,
                entry.part,
            ),
            predicted,
            predicted_onset_times_s=head_times,
        )
        rows.append(
            {
                "clip_id": entry.clip_id,
                "session_idx": entry.session_idx,
                "part": entry.part,
                "head_inhale_f1": score.inhale_f1,
                "n_predicted_head_events": len(head_indices),
                "head_probability_min": float(onset_probability.min()),
                "head_probability_max": float(onset_probability.max()),
                "head_probability_mean": float(onset_probability.mean()),
            }
        )
        print(
            f"  [{clip_number:02d}/{len(entries)}] {entry.clip_id}: "
            f"head inhale F1={score.inhale_f1:.3f}, events={len(head_indices)}",
            flush=True,
        )

    session_rows = _aggregate_head_sessions(rows)
    stratum_results = {}
    for name, indices in strata.items():
        members = [row for row in session_rows if row["session_idx"] in set(indices)]
        stratum_results[name] = {
            "sessions": indices,
            "summary": _head_summary(members),
        }
    stratum_results["all"] = {
        "sessions": sessions,
        "summary": _head_summary(session_rows),
    }
    payload = {
        "job_id": job.job_id,
        "checkpoint": str(checkpoint),
        "split": config.test_split,
        "detector": {
            "kind": "local_maxima",
            "input": "sigmoid(onset_head_logits)",
            "height_threshold": HEAD_THRESHOLD,
            "minimum_distance_s": HEAD_MIN_DISTANCE_S,
            "event_match_tolerance_s": 0.017,
            "threshold_selection": "fixed a priori; not tuned on holdout data",
        },
        "clips": rows,
        "per_session": session_rows,
        "strata": stratum_results,
    }
    _write_json_atomic(output, payload)
    print(f"wrote {output}", flush=True)
    return output


def evaluate_heads(
    config: BenchmarkConfig,
    selected_jobs: list[str] | None,
    *,
    force: bool,
) -> None:
    jobs = [job for job in config.jobs() if job.objective.w_onset > 0]
    if selected_jobs:
        wanted = set(selected_jobs)
        unknown = wanted - {job.job_id for job in jobs}
        if unknown:
            raise SystemExit(
                "unknown or non-multitask job(s): " + ", ".join(sorted(unknown))
            )
        jobs = [job for job in jobs if job.job_id in wanted]
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job.job_id}", flush=True)
        evaluate_head_job(config, job, force=force)


def _load_combined_rows(config: BenchmarkConfig) -> list[dict]:
    results = json.loads((config.output_root / "results.json").read_text())
    rows = [
        dict(row)
        for row in results["summary_by_seed"]
        if row["stratum"] in STRATUM_LABELS
    ]
    by_key = {
        (row["representation"], row["objective"], row["seed"], row["stratum"]): row
        for row in rows
    }
    missing = []
    for job in config.jobs():
        if job.objective.w_onset <= 0:
            continue
        path = config.run_dir(job) / HEAD_FILENAME
        if not path.is_file():
            missing.append(job.job_id)
            continue
        head = json.loads(path.read_text())
        for stratum in STRATUM_LABELS:
            key = (job.representation, job.objective.name, job.seed, stratum)
            by_key[key]["head_inhale_f1"] = head["strata"][stratum]["summary"][
                "head_inhale_f1"
            ]
    if missing:
        raise SystemExit(
            f"missing {HEAD_FILENAME} for {len(missing)} multitask jobs; run the "
            f"`head` command first. First missing: {missing[0]}"
        )
    return rows


def _summarise_combined(rows: list[dict]) -> list[dict]:
    summary = []
    metrics = ("correlation", "inhale_f1", "head_inhale_f1")
    for stratum in STRATUM_LABELS:
        for representation in REPRESENTATION_LABELS:
            for objective in OBJECTIVE_LABELS:
                members = [
                    row
                    for row in rows
                    if row["stratum"] == stratum
                    and row["representation"] == representation
                    and row["objective"] == objective
                ]
                result = {
                    "stratum": stratum,
                    "representation": representation,
                    "objective": objective,
                    "n_seeds": len(members),
                }
                for metric in metrics:
                    values = np.asarray(
                        [row[metric] for row in members if metric in row], dtype=float
                    )
                    result[f"{metric}_mean"] = (
                        float(values.mean()) if len(values) else None
                    )
                    result[f"{metric}_min"] = (
                        float(values.min()) if len(values) else None
                    )
                    result[f"{metric}_max"] = (
                        float(values.max()) if len(values) else None
                    )
                summary.append(result)
    return summary


def _plot_performance_detailed(config: BenchmarkConfig, rows: list[dict]) -> None:
    metric_specs = [
        ("correlation", "Signal correlation", "Pearson correlation"),
        ("inhale_f1", "Inhalation F1 from signal", "F1 score"),
        ("head_inhale_f1", "Inhalation F1 from event head", "F1 score"),
    ]
    representations = list(REPRESENTATION_LABELS)
    objectives = list(OBJECTIVE_LABELS)
    strata = list(STRATUM_LABELS)
    x = np.arange(len(representations), dtype=float)
    objective_offsets = {"signal": -0.14, "multitask": 0.14}
    seed_jitter = dict(zip([17, 42, 101, 202, 314], np.linspace(-0.045, 0.045, 5)))

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 17,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(19, 10), sharey="col")
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.82,
        bottom=0.17,
        hspace=0.42,
        wspace=0.22,
    )

    column_limits = {}
    for metric, _, _ in metric_specs:
        values = [float(row[metric]) for row in rows if metric in row]
        low, high = min(values), max(values)
        padding = max(0.006, (high - low) * 0.12)
        column_limits[metric] = (low - padding, high + padding)

    for row_index, stratum in enumerate(strata):
        for column_index, (metric, title, ylabel) in enumerate(metric_specs):
            ax = axes[row_index, column_index]
            for representation_index, representation in enumerate(representations):
                applicable_objectives = (
                    ["multitask"] if metric == "head_inhale_f1" else objectives
                )
                for objective in applicable_objectives:
                    members = sorted(
                        (
                            row
                            for row in rows
                            if row["stratum"] == stratum
                            and row["representation"] == representation
                            and row["objective"] == objective
                            and metric in row
                        ),
                        key=lambda row: row["seed"],
                    )
                    values = np.asarray([row[metric] for row in members], dtype=float)
                    base = representation_index + (
                        0.0
                        if metric == "head_inhale_f1"
                        else objective_offsets[objective]
                    )
                    positions = np.asarray(
                        [base + seed_jitter.get(row["seed"], 0.0) for row in members]
                    )
                    ax.scatter(
                        positions,
                        values,
                        s=45,
                        marker=MARKERS[objective],
                        color=COLORS[objective],
                        edgecolor="white",
                        linewidth=0.6,
                        alpha=0.9,
                        zorder=3,
                    )
                    mean = float(values.mean())
                    ax.plot(
                        [base - 0.085, base + 0.085],
                        [mean, mean],
                        color="#202020",
                        linewidth=2.5,
                        solid_capstyle="round",
                        zorder=4,
                    )
            ax.set_ylim(*column_limits[metric])
            if row_index == 0:
                ax.set_title(title, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.set_xticks(
                x,
                [REPRESENTATION_LABELS[value] for value in representations],
                rotation=18,
                ha="right",
            )
            ax.grid(axis="y", alpha=0.25, linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            if metric == "head_inhale_f1":
                ax.text(
                    0.02,
                    0.04,
                    "Signal-only: not applicable",
                    transform=ax.transAxes,
                    color="#555555",
                    fontsize=9,
                )

    fig.text(
        0.014,
        0.62,
        "Held-out animals\n(unseen subjects)",
        va="center",
        ha="center",
        rotation=90,
        fontweight="bold",
    )
    fig.text(
        0.014,
        0.285,
        "Held-out sessions\n(known animals, new dates)",
        va="center",
        ha="center",
        rotation=90,
        fontweight="bold",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[objective],
            linestyle="none",
            markerfacecolor=COLORS[objective],
            markeredgecolor="white",
            markersize=8,
            label=OBJECTIVE_LABELS[objective],
        )
        for objective in objectives
    ]
    legend_handles.append(
        Line2D([0], [0], color="#202020", linewidth=2.5, label="Five-network mean")
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncols=3,
        frameon=False,
    )
    fig.suptitle("CNN–TCN final held-out benchmark", y=0.975, fontweight="bold")
    fig.text(
        0.5,
        0.035,
        "Each point is one network. Head events use fixed probability ≥ 0.5 local maxima; black bars show means.",
        ha="center",
        color="#444444",
    )
    fig.savefig(
        config.output_root / "final-report-metrics.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        config.output_root / "final-report-metrics.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _rolling_mean(values: np.ndarray, width: int = 9) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < width:
        return np.arange(len(values)), values
    smooth = np.convolve(values, np.ones(width) / width, mode="valid")
    return np.arange(width // 2, width // 2 + len(smooth)), smooth


def _plot_losses_per_network(config: BenchmarkConfig) -> list[dict]:
    representations = list(REPRESENTATION_LABELS)
    objectives = list(OBJECTIVE_LABELS)
    seeds = list(config.seeds)
    jobs = {
        (job.representation, job.objective.name, job.seed): job for job in config.jobs()
    }
    fig, axes = plt.subplots(
        len(representations) * len(objectives),
        len(seeds),
        figsize=(20, 23),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.95,
        bottom=0.05,
        hspace=0.28,
        wspace=0.12,
    )
    convergence = []
    for representation_index, representation in enumerate(representations):
        for objective_index, objective in enumerate(objectives):
            row_index = representation_index * len(objectives) + objective_index
            for column_index, seed in enumerate(seeds):
                ax = axes[row_index, column_index]
                job = jobs[(representation, objective, seed)]
                history_path = config.run_dir(job) / "history.json"
                history = json.loads(history_path.read_text())
                epochs = np.asarray([row["epoch"] for row in history], dtype=float)
                loss = np.asarray([row["train_loss"] for row in history], dtype=float)
                smooth_x, smooth = _rolling_mean(loss)
                color = COLORS[objective]
                ax.plot(epochs, loss, color=color, alpha=0.22, linewidth=0.7)
                ax.plot(smooth_x, smooth, color=color, linewidth=1.7)
                ax.set_yscale("log")
                ax.grid(alpha=0.20, linewidth=0.6)
                ax.spines[["top", "right"]].set_visible(False)
                final_mean = float(loss[-10:].mean())
                slope = float(np.polyfit(epochs[-20:], loss[-20:], 1)[0])
                ax.text(
                    0.97,
                    0.90,
                    f"final {final_mean:.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="#333333",
                )
                convergence.append(
                    {
                        "job_id": job.job_id,
                        "representation": representation,
                        "objective": objective,
                        "seed": seed,
                        "epochs": len(history),
                        "first_10_epoch_mean_loss": float(loss[:10].mean()),
                        "last_10_epoch_mean_loss": final_mean,
                        "last_20_epoch_loss_slope_per_epoch": slope,
                    }
                )
                if row_index == 0:
                    ax.set_title(f"Seed {seed}", fontweight="bold")
                if column_index == 0:
                    ax.set_ylabel(
                        f"{REPRESENTATION_LABELS[representation]}\n"
                        f"{OBJECTIVE_LABELS[objective]}\nLoss"
                    )
                if row_index == len(representations) * len(objectives) - 1:
                    ax.set_xlabel("Epoch")

    fig.suptitle(
        "Training loss for all 40 networks",
        y=0.99,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.018,
        "Thin lines: epoch values. Thick lines: nine-epoch moving average. Logarithmic loss axis shared by all panels.",
        ha="center",
        color="#444444",
    )
    fig.savefig(
        config.output_root / "all-training-loss-curves.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        config.output_root / "all-training-loss-curves.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return convergence


def _plot_performance(config: BenchmarkConfig, rows: list[dict]) -> None:
    """Two-row report with correlation and all applicable inhalation F1s."""
    representations = list(REPRESENTATION_LABELS)
    strata = list(STRATUM_LABELS)
    x = np.arange(len(representations), dtype=float)
    seed_jitter = dict(zip([17, 42, 101, 202, 314], np.linspace(-0.035, 0.035, 5)))
    green = "#009E73"

    correlation_series = [
        ("signal", "correlation", "Signal only", COLORS["signal"], "o", -0.12),
        (
            "multitask",
            "correlation",
            "Signal + event detection",
            COLORS["multitask"],
            "D",
            0.12,
        ),
    ]
    f1_series = [
        (
            "signal",
            "inhale_f1",
            "Signal only: F1 from signal",
            COLORS["signal"],
            "o",
            -0.20,
        ),
        (
            "multitask",
            "inhale_f1",
            "Multitask: F1 from signal",
            COLORS["multitask"],
            "D",
            0.0,
        ),
        (
            "multitask",
            "head_inhale_f1",
            "Multitask: F1 from event head",
            green,
            "^",
            0.20,
        ),
    ]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 17,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey="col")
    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.80,
        bottom=0.17,
        hspace=0.42,
        wspace=0.24,
    )

    correlation_values = [float(row["correlation"]) for row in rows]
    f1_values = [float(row["inhale_f1"]) for row in rows]
    f1_values += [
        float(row["head_inhale_f1"]) for row in rows if "head_inhale_f1" in row
    ]
    limits = {}
    for name, values in (("correlation", correlation_values), ("f1", f1_values)):
        low, high = min(values), max(values)
        padding = max(0.006, (high - low) * 0.10)
        limits[name] = (low - padding, high + padding)

    for row_index, stratum in enumerate(strata):
        for column_index, (series, ylabel, limit_name) in enumerate(
            (
                (correlation_series, "Pearson correlation", "correlation"),
                (f1_series, "Inhalation F1 score", "f1"),
            )
        ):
            ax = axes[row_index, column_index]
            for representation_index, representation in enumerate(representations):
                for objective, metric, _, color, marker, offset in series:
                    members = sorted(
                        (
                            row
                            for row in rows
                            if row["stratum"] == stratum
                            and row["representation"] == representation
                            and row["objective"] == objective
                            and metric in row
                        ),
                        key=lambda row: row["seed"],
                    )
                    values = np.asarray([row[metric] for row in members], dtype=float)
                    base = representation_index + offset
                    positions = np.asarray(
                        [base + seed_jitter.get(row["seed"], 0.0) for row in members]
                    )
                    ax.scatter(
                        positions,
                        values,
                        s=48,
                        marker=marker,
                        color=color,
                        edgecolor="white",
                        linewidth=0.6,
                        alpha=0.9,
                        zorder=3,
                    )
                    mean = float(values.mean())
                    ax.plot(
                        [base - 0.075, base + 0.075],
                        [mean, mean],
                        color="#202020",
                        linewidth=2.5,
                        solid_capstyle="round",
                        zorder=4,
                    )
            ax.set_ylim(*limits[limit_name])
            if row_index == 0:
                ax.set_title(
                    "Signal correlation"
                    if column_index == 0
                    else "Inhalation event detection",
                    fontweight="bold",
                )
            ax.set_ylabel(ylabel)
            ax.set_xticks(
                x,
                [REPRESENTATION_LABELS[value] for value in representations],
                rotation=18,
                ha="right",
            )
            ax.grid(axis="y", alpha=0.25, linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.025,
        0.615,
        "Held-out animals\n(unseen subjects)",
        va="center",
        ha="center",
        rotation=90,
        fontweight="bold",
    )
    fig.text(
        0.025,
        0.285,
        "Held-out sessions\n(known animals, new dates)",
        va="center",
        ha="center",
        rotation=90,
        fontweight="bold",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            label=label,
        )
        for _, _, label, color, marker, _ in f1_series
    ]
    legend_handles.append(
        Line2D([0], [0], color="#202020", linewidth=2.5, label="Five-network mean")
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncols=2,
        frameon=False,
    )
    fig.suptitle("CNN–TCN final held-out benchmark", y=0.975, fontweight="bold")
    fig.text(
        0.5,
        0.035,
        "Each point is one network. Head events use fixed probability ≥ 0.5 local maxima; black bars show means.",
        ha="center",
        color="#444444",
    )
    fig.savefig(
        config.output_root / "final-report-metrics.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        config.output_root / "final-report-metrics.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _plot_losses(config: BenchmarkConfig) -> list[dict]:
    """Eight panels, one per condition, with all five seeds overlaid."""
    representations = list(REPRESENTATION_LABELS)
    objectives = list(OBJECTIVE_LABELS)
    seeds = list(config.seeds)
    jobs = {
        (job.representation, job.objective.name, job.seed): job for job in config.jobs()
    }
    seed_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7A4EAB"]
    fig, axes = plt.subplots(
        len(representations),
        len(objectives),
        figsize=(13, 15),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.11,
        right=0.985,
        top=0.90,
        bottom=0.07,
        hspace=0.30,
        wspace=0.14,
    )
    convergence = []
    for representation_index, representation in enumerate(representations):
        for objective_index, objective in enumerate(objectives):
            ax = axes[representation_index, objective_index]
            for seed, color in zip(seeds, seed_colors):
                job = jobs[(representation, objective, seed)]
                history = json.loads((config.run_dir(job) / "history.json").read_text())
                epochs = np.asarray([row["epoch"] for row in history], dtype=float)
                loss = np.asarray([row["train_loss"] for row in history], dtype=float)
                smooth_x, smooth = _rolling_mean(loss)
                ax.plot(epochs, loss, color=color, alpha=0.12, linewidth=0.7)
                ax.plot(
                    smooth_x, smooth, color=color, linewidth=1.6, label=f"Seed {seed}"
                )
                final_mean = float(loss[-10:].mean())
                convergence.append(
                    {
                        "job_id": job.job_id,
                        "representation": representation,
                        "objective": objective,
                        "seed": seed,
                        "epochs": len(history),
                        "first_10_epoch_mean_loss": float(loss[:10].mean()),
                        "last_10_epoch_mean_loss": final_mean,
                        "last_20_epoch_loss_slope_per_epoch": float(
                            np.polyfit(epochs[-20:], loss[-20:], 1)[0]
                        ),
                    }
                )
            ax.set_yscale("log")
            ax.grid(alpha=0.20, linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if representation_index == 0:
                ax.set_title(OBJECTIVE_LABELS[objective], fontweight="bold")
            if objective_index == 0:
                ax.set_ylabel(f"{REPRESENTATION_LABELS[representation]}\nTraining loss")
            if representation_index == len(representations) - 1:
                ax.set_xlabel("Epoch")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncols=5,
        frameon=False,
    )
    fig.suptitle(
        "Training loss by input and objective",
        y=0.985,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "Five seeds per panel. Faint lines are epoch values; solid lines are nine-epoch moving averages. Shared logarithmic loss axis.",
        ha="center",
        color="#444444",
    )
    fig.savefig(
        config.output_root / "all-training-loss-curves.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        config.output_root / "all-training-loss-curves.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return convergence


def _format_cell(row: dict, metric: str, best: float | None) -> str:
    mean = row[f"{metric}_mean"]
    if mean is None:
        return "N/A"
    cell = f"{mean:.4f} ({row[f'{metric}_min']:.4f}/{row[f'{metric}_max']:.4f})"
    return f"**{cell}**" if best is not None and np.isclose(mean, best) else cell


def _write_summary_table(config: BenchmarkConfig, summary: list[dict]) -> None:
    lines = [
        "# Final CNN–TCN benchmark summary",
        "",
        "Cells report mean (minimum/maximum) across five trained networks. Bold indicates the highest mean within each holdout stratum and metric. Head F1 is not applicable to signal-only networks.",
    ]
    for stratum, stratum_label in STRATUM_LABELS.items():
        rows = [row for row in summary if row["stratum"] == stratum]
        best = {}
        for metric in ("correlation", "inhale_f1", "head_inhale_f1"):
            values = [
                row[f"{metric}_mean"]
                for row in rows
                if row[f"{metric}_mean"] is not None
            ]
            best[metric] = max(values) if values else None
        lines.extend(
            [
                "",
                f"## {stratum_label}",
                "",
                "| Input | Objective | Correlation | Inhalation F1 from signal | Inhalation F1 from head |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for representation in REPRESENTATION_LABELS:
            for objective in OBJECTIVE_LABELS:
                row = next(
                    value
                    for value in rows
                    if value["representation"] == representation
                    and value["objective"] == objective
                )
                lines.append(
                    f"| {REPRESENTATION_LABELS[representation]} | "
                    f"{OBJECTIVE_LABELS[objective]} | "
                    f"{_format_cell(row, 'correlation', best['correlation'])} | "
                    f"{_format_cell(row, 'inhale_f1', best['inhale_f1'])} | "
                    f"{_format_cell(row, 'head_inhale_f1', best['head_inhale_f1'])} |"
                )
    (config.output_root / "final-report-summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_report(config: BenchmarkConfig) -> None:
    rows = _load_combined_rows(config)
    summary = _summarise_combined(rows)
    _plot_performance(config, rows)
    convergence = _plot_losses(config)
    _write_summary_table(config, summary)
    _write_json_atomic(
        config.output_root / "final-report-results.json",
        {
            "benchmark": config.name,
            "head_detector": {
                "kind": "local_maxima",
                "height_threshold": HEAD_THRESHOLD,
                "minimum_distance_s": HEAD_MIN_DISTANCE_S,
                "event_match_tolerance_s": 0.017,
                "threshold_selection": "fixed a priori; not tuned on holdout data",
            },
            "per_network": rows,
            "summary": summary,
            "training_convergence": convergence,
        },
    )
    print(f"wrote final report under {config.output_root}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("head", "report", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--job", action="append", dest="jobs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command in {"head", "all"}:
        evaluate_heads(config, args.jobs, force=args.force)
    if args.command in {"report", "all"}:
        if args.jobs:
            raise SystemExit("--job can only be used with the `head` command")
        build_report(config)
