"""Score checkpoints against held-out labelled clips, the way the benchmark does.

Truth is the raw thermistor trace, scored by ``.evaluation.score_clip``. ``zephyr
evaluate`` and the evaluation after :mod:`zephyr.run` share :func:`evaluate_groups`:
named groups of clips in, one result out. Clips sharing a video or recording with a
checkpoint's recorded training set are refused (older checkpoints only warn). Multitask
checkpoints also report the onset head's inhale F1. Several checkpoints are z-scored
then averaged; ``--plot`` (one checkpoint) writes :mod:`.plot_diagnosis` figures.
Only zephyr checkpoints load here; other networks bring their own loader.

CLI
---
    zephyr evaluate --checkpoint runs/example/fold/seed-42/best.pt \\
        --clips <list.toml> --cache data/features-example
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks

from zephyr.channels import ChannelSet
from zephyr.model import BreathingNet

from . import features
from .config import ClipList, ResolvedClip, load
from .evaluation import score_clip
from .features import ClipEntry
from .infer import predict_clip
from .plot_diagnosis import ClipPrediction, rate_breakdown, reserved_grid
from .signal import BREATHING_SIGNAL_COLUMN, TIME_COLUMN

HEAD_THRESHOLD = 0.5
HEAD_MIN_DISTANCE_S = 0.05
"""Onset-head detector: local maxima of the probability above a fixed
threshold, at least this far apart.  Fixed a priori; never tuned on holdout data."""


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[BreathingNet, np.ndarray, np.ndarray, dict]:
    """Rebuild a model from a training checkpoint, with its own normalisation.

    ``model`` holds whichever weights the run selected -- the EMA copy when
    averaging was enabled -- so nothing here needs to know how it was trained.

    The channel selection comes from the checkpoint, not from the current
    config: it fixes the first convolution's shape, so reading it from
    anywhere else would build a model the weights do not fit.  Checkpoints
    written before channel selection existed have no such field and trained on
    every stored channel, which their own feature config records.  Returned
    ``mean``/``std`` stay full-width -- :func:`~.infer.predict_clip` slices
    them to the model.  Checkpoints without an ``arch`` field are zephyr
    CNN-TCNs; any other ``arch`` is refused.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    arch = state.get("arch", "zephyr")
    if arch != "zephyr":
        raise SystemExit(
            f"{path} is a {arch} checkpoint; this loads only zephyr checkpoints, "
            "so score it with the loader of the package that trained it"
        )
    model = BreathingNet(channels=checkpoint_channels(state)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state["mean"], state["std"], state


def checkpoint_channels(state: dict) -> ChannelSet:
    """The channels a checkpoint's network reads; see :func:`load_checkpoint`."""
    return ChannelSet.parse(
        state.get("channels") or state["feature_config"]["channel_names"]
    )


Loader = Callable[
    [Path, torch.device], tuple[torch.nn.Module, np.ndarray, np.ndarray, dict]
]
"""Same contract as :func:`load_checkpoint`."""


def refuse_leaked(states: list[dict], clips: list[ResolvedClip]) -> None:
    """Refuse to score clips any checkpoint trained on, by video or recording.

    A fine-tuned checkpoint is checked against its parents' clips too.  One with
    no training record, or descended from one, cannot be fully checked: that is
    warned about, not refused, so old checkpoints stay usable.
    """
    leaked = []
    for state in states:
        videos = state.get("train_videos")
        if videos is None or not state.get("lineage_known", True):
            print(
                "WARNING: checkpoint (or a checkpoint it was initialised from) "
                "records no training clips, so leakage cannot be fully checked; "
                "make sure it never saw these recordings."
            )
        if videos is None:
            continue
        trained_videos = {*videos, *state.get("inherited_videos", ())}
        trained_recordings = {
            *state.get("train_recordings", ()),
            *state.get("inherited_recordings", ()),
        }
        for clip in clips:
            if str(clip.video) in trained_videos:
                leaked.append(f"{clip.name} (trained on this video)")
            elif clip.recording_id in trained_recordings:
                leaked.append(f"{clip.name} (trained on its recording)")
    if leaked:
        raise SystemExit(
            f"refusing to score: {len(leaked)} clip(s) were not held out by every "
            "checkpoint, so this would not be an unbiased estimate: "
            f"{sorted(set(leaked))}"
        )


def zscore(x: np.ndarray) -> np.ndarray:
    scale = x.std()
    return (x - x.mean()) / (scale if scale > 0 else 1.0)


def predict_entry(
    models: list[tuple[BreathingNet, np.ndarray, np.ndarray]],
    entry: ClipEntry,
    device: torch.device,
    *,
    window: int,
    frame_chunk: int,
    amp_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Ensemble prediction for one clip, z-scored per model then averaged.

    Each model is z-scored before averaging: they are trained on a correlation
    loss that leaves output scale free, so averaging raw outputs would weight by
    whichever run happened to settle on the largest amplitude.  Also returns the
    onset-head probability when there is exactly one model, else ``None``.
    """
    outputs = [
        predict_clip(
            model,
            entry,
            mean,
            std,
            window=window,
            device=device,
            frame_chunk=frame_chunk,
            amp_dtype=amp_dtype,
        )
        for model, mean, std in models
    ]
    stack = [zscore(signal) for signal, _ in outputs]
    onset = outputs[0][1] if len(outputs) == 1 else None
    return zscore(np.mean(stack, axis=0)), onset


def truth_frame(entry: ClipEntry) -> pd.DataFrame:
    """Raw thermistor trace -- already the columns ``score_clip`` expects."""
    return pd.read_parquet(entry.thermistor)


def head_event_indices(probability: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Inhale events the onset head asserts: thresholded local maxima."""
    if len(probability) != len(times):
        raise ValueError("onset probability and time arrays have different lengths")
    if len(times) < 2:
        return np.empty(0, dtype=int)
    sampling_rate = 1.0 / float(np.median(np.diff(times)))
    distance = max(1, round(HEAD_MIN_DISTANCE_S * sampling_rate))
    peaks, _ = find_peaks(probability, height=HEAD_THRESHOLD, distance=distance)
    return peaks.astype(int)


METRIC_FIELDS = ("correlation", "inhale_f1", "exhale_f1", "kl_ibi")
HEAD_FIELD = "head_inhale_f1"


def _fields(rows: list[dict]) -> tuple[str, ...]:
    head = any(r.get(HEAD_FIELD) is not None for r in rows)
    return METRIC_FIELDS + ((HEAD_FIELD,) if head else ())


def summarise_rows(rows: list[dict]) -> dict[str, float]:
    """Mean and spread for metric rows, ignoring undefined values."""
    summary: dict[str, float] = {}
    for field in _fields(rows):
        values = np.array(
            [r[field] for r in rows if r.get(field) is not None], dtype=float
        )
        values = values[np.isfinite(values)]
        summary[field] = float(values.mean()) if len(values) else float("nan")
        summary[f"{field}_sd"] = (
            float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        )
        summary[f"{field}_n"] = len(values)
    return summary


def aggregate_recordings(rows: list[dict]) -> list[dict]:
    """Average clip metrics within recording, the independent sampling unit."""
    recordings: list[dict] = []
    for recording in dict.fromkeys(r["recording"] for r in rows):
        members = [r for r in rows if r["recording"] == recording]
        summary = summarise_rows(members)
        recordings.append(
            {"recording": recording, "n_clips": len(members)}
            | {field: summary[field] for field in _fields(rows)}
        )
    return recordings


def build_result(groups: dict[str, list[dict]]) -> dict:
    """Clip, recording and group summaries plus the composite, as JSON-ready data.

    *groups* maps each group's name to its clip rows (each carrying a
    ``recording``).  Shared by checkpoint evaluation and any other scorer so
    all report identically.  ``"all"`` is reserved for the summary across groups.
    """
    if "all" in groups:
        raise ValueError("'all' is reserved for the summary across groups")
    rows = [
        row | {"group": name} for name, members in groups.items() for row in members
    ]
    summary = summarise_rows(rows)

    group_results = {}
    per_recording: list[dict] = []
    for name, members in groups.items():
        recordings = aggregate_recordings(members)
        per_recording += [{"group": name} | r for r in recordings]
        group_results[name] = {
            "recordings": [r["recording"] for r in recordings],
            "summary": summarise_rows(recordings),
        }
    summary_by_recording = summarise_rows(per_recording)
    group_results["all"] = {
        "recordings": [r["recording"] for r in per_recording],
        "summary": summary_by_recording,
    }
    # Weighted summary of the four metrics, for orientation only.
    composite = (
        0.50 * summary["inhale_f1"]
        + 0.20 * summary["exhale_f1"]
        + 0.20 * max(0.0, summary["correlation"])
        + 0.10 * float(np.exp(-summary["kl_ibi"]))
    )
    return {
        "clips": rows,
        "summary": summary,
        "per_recording": per_recording,
        "summary_by_recording": summary_by_recording,
        "groups": group_results,
        "composite": composite,
    }


def score_entries(
    models: list[tuple[BreathingNet, np.ndarray, np.ndarray]],
    entries: list[ClipEntry],
    device: torch.device,
    *,
    window: int,
    frame_chunk: int,
    amp_dtype: torch.dtype | None,
    head: bool = False,
    plot_data: list[ClipPrediction] | None = None,
) -> list[dict]:
    """Score each labelled entry: one row of metrics per clip, printed as it goes."""
    header = f"{'clip':40s}{'corr':>8s}{'inh_f1':>8s}{'exh_f1':>8s}{'kl_ibi':>8s}" + (
        f"{'head_f1':>8s}" if head else ""
    )
    print(header)
    print("-" * len(header))
    rows = []
    for entry in entries:
        signal, onset = predict_entry(
            models,
            entry,
            device,
            window=window,
            frame_chunk=frame_chunk,
            amp_dtype=amp_dtype,
        )
        times = np.load(entry.times)
        n = min(len(times), len(signal))
        signal, times = signal[:n], times[:n]
        predicted = pd.DataFrame(
            {
                TIME_COLUMN: times.astype(np.float64),
                BREATHING_SIGNAL_COLUMN: signal.astype(np.float64),
            }
        )
        truth = truth_frame(entry)
        score = score_clip(truth, predicted)
        row = {"clip_id": entry.clip_id, "recording": entry.recording}
        row |= score.to_dict()
        line = (
            f"{entry.clip_id:40s}{score.correlation:+8.3f}"
            f"{score.inhale_f1:8.3f}{score.exhale_f1:8.3f}{score.kl_ibi:8.3f}"
        )
        if head:
            head_indices = head_event_indices(onset[:n], times)
            head_score = score_clip(
                truth, predicted, predicted_onset_times_s=times[head_indices]
            )
            row[HEAD_FIELD] = head_score.to_dict()["inhale_f1"]
            row["n_head_events"] = len(head_indices)
            line += f"{head_score.inhale_f1:8.3f}"
        rows.append(row)
        if plot_data is not None:
            plot_data.append(ClipPrediction(entry, signal, times, truth))
        print(line, flush=True)
    print("-" * len(header))
    return rows


def has_onset_head(states: list[dict]) -> bool:
    """Whether the (single) checkpoint was trained with an onset objective."""
    if len(states) != 1:
        return False
    recorded = states[0].get("params") or states[0].get("args") or {}
    return float(recorded.get("w_onset", 0.0)) > 0


def evaluate_groups(
    checkpoints: list[Path],
    groups: dict[str, list[ResolvedClip]],
    features_dir: Path,
    recipe,
    *,
    device: str,
    amp: str,
    window: int = 1024,
    frame_chunk: int = 256,
    out: Path | None = None,
    plot_dir: Path | None = None,
    plot_options: dict | None = None,
    loader: Loader = load_checkpoint,
) -> dict:
    """Score *checkpoints* on named groups of clips; the one evaluation path.

    Unlabelled clips (no thermistor) are skipped: there is nothing to score them
    against.  Returns the :func:`build_result` dict, also written to *out*.
    """
    torch_device = torch.device(device)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[amp]
    models, states = [], []
    for path in checkpoints:
        model, mean, std, state = loader(path, torch_device)
        models.append((model, mean, std))
        states.append(state)
        print(f"loaded {path}  epoch {state.get('epoch')}")

    labelled = {
        name: [c for c in clips if c.labelled] for name, clips in groups.items()
    }
    skipped = sum(len(clips) - len(labelled[name]) for name, clips in groups.items())
    if skipped:
        print(f"skipping {skipped} unlabelled clip(s): nothing to score them against")
    labelled = {name: clips for name, clips in labelled.items() if clips}
    if not labelled:
        raise SystemExit("no labelled clips to score")
    refuse_leaked(states, [c for clips in labelled.values() for c in clips])

    head = has_onset_head(states)
    plot_data: list[ClipPrediction] | None = [] if plot_dir is not None else None
    rows_by_group: dict[str, list[dict]] = {}
    for name, clips in labelled.items():
        entries = features.require(clips, recipe, features_dir)
        print(
            f"\ngroup {name!r}: {len(entries)} clips / "
            f"{len({e.recording for e in entries})} recordings with "
            f"{len(models)} model(s)\n"
        )
        rows_by_group[name] = score_entries(
            models,
            entries,
            torch_device,
            window=window,
            frame_chunk=frame_chunk,
            amp_dtype=amp_dtype,
            head=head,
            plot_data=plot_data,
        )

    result = build_result(rows_by_group)
    for name, group in result["groups"].items():
        summary = group["summary"]
        line = (
            f"{name:28s} corr {summary['correlation']:+.3f}  "
            f"inhale F1 {summary['inhale_f1']:.3f}  exhale F1 "
            f"{summary['exhale_f1']:.3f}  KL-IBI {summary['kl_ibi']:.3f}"
        )
        if head:
            line += f"  head F1 {summary[HEAD_FIELD]:.3f}"
        print(line)
    print(f"\ncomposite (0.5/0.2/0.2/0.1): {result['composite']:.4f}")

    result = {"checkpoints": [str(p) for p in checkpoints]} | result
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"wrote {out}")

    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        options = plot_options or {}
        recordings = list(
            dict.fromkeys(
                r["recording"] for rows in rows_by_group.values() for r in rows
            )
        )
        rate_breakdown(
            plot_data,
            recordings,
            plot_dir / "rate_breakdown.png",
            checkpoints[0].parent.name,
            n_shifts=options.get("null_shifts", 5),
            corr_window_s=options.get("corr_window_s", 3.0),
            corr_hop_s=options.get("corr_hop_s", 1.5),
            corr_min_breaths=options.get("corr_min_breaths", 3),
        )
        model, mean, std = models[0]
        reserved_grid(
            model,
            mean,
            std,
            torch_device,
            [p.entry for p in plot_data],
            recordings,
            plot_dir / "reserved_grid.png",
            window=options.get("grid_window", 512),
            infer_window=window,
            frame_chunk=frame_chunk,
            amp_dtype=amp_dtype,
        )
        print(f"wrote diagnostic plots -> {plot_dir}")
    return result


def main(argv: list[str] | None = None, *, loader: Loader = load_checkpoint) -> None:
    parser = argparse.ArgumentParser(description="Score checkpoints on held-out clips.")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--clips",
        type=Path,
        nargs="+",
        required=True,
        help="Clip list(s) to score; each is reported as its own group.",
    )
    parser.add_argument(
        "--name",
        help="Group name for a single --clips list (default: the file's stem).",
    )
    parser.add_argument(
        "--cache", type=Path, required=True, help="Feature cache directory."
    )
    parser.add_argument("--infer-window", type=int, default=1024)
    parser.add_argument("--frame-chunk", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    parser.add_argument("--out", type=Path, help="Write the full result as JSON.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write the rate-breakdown and reserved-grid diagnostic plots "
        "(see .plot_diagnosis) into --plot-dir.  Needs exactly one --checkpoint: "
        "the reserved-grid panel reads one model's own onset head.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Where the diagnostic plots land. Defaults to a `diagnosis/` "
        "folder next to the checkpoint being scored, so plots from different "
        "runs never collide or need telling apart by filename.",
    )
    parser.add_argument("--grid-window", type=int, default=512)
    parser.add_argument("--corr-window-s", type=float, default=3.0)
    parser.add_argument("--corr-hop-s", type=float, default=1.5)
    parser.add_argument("--corr-min-breaths", type=int, default=3)
    parser.add_argument(
        "--null-shifts",
        type=int,
        default=5,
        help="Circular shifts of the predicted onset train used for the rate-"
        "breakdown plot's chance baseline.",
    )
    args = parser.parse_args(argv)

    if args.plot and len(args.checkpoint) != 1:
        raise SystemExit(
            f"--plot needs exactly one --checkpoint (the reserved-grid panel "
            f"reads one model's own onset head); got {len(args.checkpoint)}"
        )
    if args.name and len(args.clips) != 1:
        raise SystemExit("--name names one group, so it needs exactly one --clips list")

    lists = {(args.name or path.stem): load(ClipList, path) for path in args.clips}
    recipes = {lst.preprocess for lst in lists.values()}
    if len(recipes) != 1:
        raise SystemExit("the clip lists disagree on [preprocess]")
    evaluate_groups(
        args.checkpoint,
        {name: lst.resolve() for name, lst in lists.items()},
        args.cache,
        recipes.pop(),
        device=args.device,
        amp=args.amp,
        window=args.infer_window,
        frame_chunk=args.frame_chunk,
        out=args.out,
        plot_dir=(
            args.plot_dir or args.checkpoint[0].parent / "diagnosis"
            if args.plot
            else None
        ),
        plot_options={
            "grid_window": args.grid_window,
            "corr_window_s": args.corr_window_s,
            "corr_hop_s": args.corr_hop_s,
            "corr_min_breaths": args.corr_min_breaths,
            "null_shifts": args.null_shifts,
        },
        loader=loader,
    )


if __name__ == "__main__":
    main()
