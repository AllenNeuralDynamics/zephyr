"""Noisy frames: a video's prediction again with Gaussian noise added to every frame.

A null for quality control: how a network's label-free measures move as its input
degrades, beside its true scores. Each ``--sigmas`` level (gray levels, 0-255) adds
noise to the frames the network sees (cached frames only, through a wrapper around the
network, as :mod:`.occlusion` hides patches), and is measured as ``zephyr
quality-control run`` measures a video, with its agreement with the ordinary prediction
and, for a labelled clip, its truth: one ``<stem>.probe.json`` per video (with
``--save-traces``, each level's trace in ``<stem>.probe.parquet``).

CLI
---
    zephyr-benchmarks probe --checkpoint <best.pt> --clips <list.toml>... --cache <dir>
        --out <dir> [--groups G...] [--max-clips N] [--sigmas 4 8 16 32 64]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from zephyr import features
from zephyr.config import QualityParams, QualitySelection, ResolvedClip, canonical
from zephyr.evaluate import (
    checkpoint_head_threshold,
    has_onset_head,
    load_checkpoint,
    refuse_leaked,
    score_span,
    truth_events,
    truth_frame,
)
from zephyr.evaluation import event_f1, frame_tolerance_s
from zephyr.features import ClipEntry
from zephyr.infer import predict_clip
from zephyr.quality_control.metrics import NAN, Video, find_events, measure, spans
from zephyr.quality_control.reference import checkpoint_hash, jsonable
from zephyr.quality_control.report import _spread_out

REPORT = ".probe.json"
TRACES = ".probe.parquet"
SIGMAS = (4.0, 8.0, 16.0, 32.0, 64.0)
"""Noise standard deviations, in gray levels of the 0-255 frames."""
AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}


class Noisy(nn.Module):
    """A network that sees its input with Gaussian noise of *sigma* gray levels added.

    The input is already standardised by the checkpoint's *std*, so the noise is
    ``sigma / std`` there. Everything :func:`zephyr.infer.predict_clip` reads from a
    model is passed through; *seed* makes a clip's noise repeatable.
    """

    def __init__(self, model: nn.Module, std: float) -> None:
        super().__init__()
        self.model = model
        self.std = std
        self.sigma = 0.0
        self.generator: torch.Generator | None = None
        self.receptive_field = model.receptive_field

    def reseed(self, sigma: float, seed: int) -> None:
        self.sigma = sigma
        self.generator = torch.Generator().manual_seed(seed)

    def forward(self, frames: torch.Tensor, *args, **kwargs):
        """*frames* is ``(batch, time, 1, height, width)``."""
        if self.sigma > 0:
            noise = torch.randn(frames.shape, generator=self.generator)
            frames = frames + noise.to(frames.device, frames.dtype) * (
                self.sigma / self.std
            )
        return self.model(frames, *args, **kwargs)


def agreement(base: Video, other: Video, span) -> dict[str, float]:
    """How far *other* reproduces the ordinary prediction *base* over *span*: trace
    correlation, and inhale F1 against base's inhales (at the frame tolerance)."""
    mask = (base.times >= span[0]) & (base.times < span[1])
    a, b = base.signal[mask], other.signal[mask]
    a, b = a - a.mean(), b - b.mean()
    scale = float(np.sqrt((a * a).sum() * (b * b).sum()))

    def inhales(video: Video) -> np.ndarray:
        at = video.times[video.events.inhale]
        return at[(at >= span[0]) & (at < span[1])]

    return {
        "trace_r": float((a * b).sum() / scale) if scale > 0 else NAN,
        "inhale_f1": event_f1(
            inhales(base), inhales(other), frame_tolerance_s(base.times)
        ),
    }


class Prober:
    """One network, predicting a clip at every noise level and measuring each."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        network, self.mean, self.std, self.state = load_checkpoint(
            args.checkpoint, self.device
        )
        self.model = Noisy(network, self.std)
        self.head = has_onset_head([self.state])
        self.threshold = checkpoint_head_threshold(self.state)
        self.params = QualityParams()

    def predict(self, entry: ClipEntry, sigma: float, seed: int) -> Video:
        self.model.reseed(sigma, seed)
        signal, onset = predict_clip(
            self.model,
            entry,
            self.mean,
            self.std,
            window=self.args.infer_window,
            device=self.device,
            frame_chunk=self.args.frame_chunk,
            amp_dtype=AMP[self.args.amp],
        )
        times = np.load(entry.times)
        onset = onset if self.head else None
        events = find_events(times, signal, onset, self.threshold)
        return Video(times, signal, onset, events)

    def video(
        self, clip: ResolvedClip, entry: ClipEntry, seed: int
    ) -> tuple[dict, pd.DataFrame]:
        """The ordinary prediction and every noise level, each over the whole clip
        and per window; and each level's trace, onset probability and inhales
        (``signal_<sigma>``, ``onset_<sigma>``, ``inhale_<sigma>``) beside ``time``."""
        truth = None
        if entry.thermistor is not None and entry.thermistor.is_file():
            truth = (truth_frame(entry), truth_events(entry))
        base = self.predict(entry, 0.0, seed)
        whole = (float(base.times[0]), float(np.nextafter(base.times[-1], np.inf)))

        def scores(video: Video, span, whole_clip=False) -> dict:
            p = self.params
            out = measure(
                video,
                span,
                ceiling_hz=p.breathing_ceiling_hz,
                ambiguous_floor=p.ambiguous_floor,
                reference=None,
                flat_ratio=p.flat_ratio,
            )
            out["agreement"] = agreement(base, video, span)
            if truth is not None:
                inhale_s = video.times[video.events.inhale] if self.head else None
                out["truth"] = score_span(
                    truth, video.times, video.signal, inhale_s, span, whole_clip
                )
            return out

        results, traces = [], {"time": base.times}
        for sigma in (0.0, *self.args.sigmas):
            video = base if sigma == 0 else self.predict(entry, sigma, seed)
            inhale = np.zeros(len(video.times), bool)
            inhale[video.events.inhale] = True
            traces |= {
                f"signal_{sigma:g}": video.signal,
                f"onset_{sigma:g}": video.onset,
                f"inhale_{sigma:g}": inhale,
            }
            results.append(
                {
                    "sigma": sigma,
                    "summary": scores(video, whole, whole_clip=True),
                    "windows": [
                        {"start": a, "end": b} | scores(video, (a, b))
                        for a, b in spans(base.times, self.args.window_s)
                    ],
                }
            )
        report = {
            "clip_id": entry.clip_id,
            "video": canonical(clip.video) if clip.video else None,
            "recording": entry.recording,
            "labelled": truth is not None,
            "duration_s": float(base.times[-1] - base.times[0]),
            "conditions": results,
        }
        return report, pd.DataFrame(traces)


def _selections(paths: list[Path], groups: list[str] | None):
    """The clips of *paths* (only *groups*, when given) and their one recipe."""
    selections = []
    for path in paths:
        if groups is None:
            selections.append(QualitySelection(clips=path))
            continue
        present = {c.group for c in QualitySelection(clips=path).resolved()}
        mine = [g for g in groups if g in present]
        if mine:
            selections.append(QualitySelection(clips=path, groups=mine))
    recipes = {s.clip_list().preprocess for s in selections}
    if len(recipes) != 1:
        raise SystemExit("the clip lists disagree on [preprocess]")
    return [c for s in selections for c in s.resolved()], recipes.pop()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predictions from noisy frames.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clips", type=Path, nargs="+", required=True)
    parser.add_argument("--groups", nargs="+", help="Keep only these groups.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Report folder.")
    parser.add_argument("--max-clips", type=int, help="Spread over the selection.")
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=list(SIGMAS),
        help="Noise standard deviations, gray levels.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-traces", action="store_true", help="Also write each level's trace."
    )
    parser.add_argument("--window-s", type=float, default=30.0)
    parser.add_argument("--infer-window", type=int, default=1024)
    parser.add_argument("--frame-chunk", type=int, default=256)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--amp", default="bf16", choices=list(AMP))
    args = parser.parse_args(argv)

    clips, recipe = _selections(args.clips, args.groups)
    clips = _spread_out(clips, args.max_clips)
    prober = Prober(args)
    refuse_leaked([prober.state], clips)
    entries = features.require(clips, recipe, args.cache)
    meta = {
        "checkpoint": canonical(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash(args.checkpoint),
        "head_threshold": prober.threshold if prober.head else None,
        "sigmas": list(args.sigmas),
        "seed": args.seed,
        "window_s": args.window_s,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for k, (clip, entry) in enumerate(zip(clips, entries, strict=True)):
        out = args.out / f"{clip.source.stem}{REPORT}"
        if out.is_file():
            continue
        print(f"[{k + 1}/{len(clips)}] {entry.clip_id}", flush=True)
        report, traces = prober.video(clip, entry, args.seed + k)
        if args.save_traces:
            traces.to_parquet(args.out / f"{clip.source.stem}{TRACES}", index=False)
        result = jsonable(report | meta)
        part = out.with_name(out.name + ".part")
        part.write_text(json.dumps(result, indent=1))
        part.replace(out)
    print(f"{len(clips)} probe report(s) in {args.out}")


# --- reading probes back -------------------------------------------------------------------


def load_probes(folder: Path, windows: bool = False) -> pd.DataFrame:
    """Every noise level of *folder*'s reports, one row per video and level (``sigma``
    0: the ordinary prediction), over the whole clip or, with *windows*, per window:
    ``stem``, ``sigma``, the measures, ``agree_<name>`` and ``truth_<metric>``."""
    rows = []
    for path in sorted(Path(folder).glob(f"*{REPORT}")):
        data = json.loads(path.read_text())
        ident = {"stem": path.name.removesuffix(REPORT), "recording": data["recording"]}
        for condition in data["conditions"]:
            spans_ = condition["windows"] if windows else [condition["summary"]]
            for span in spans_:
                span = dict(span)
                agree = span.pop("agreement", None) or {}
                truth = span.pop("truth", None) or {}
                rows.append(
                    ident
                    | {"sigma": condition["sigma"]}
                    | span
                    | {f"agree_{k}": v for k, v in agree.items()}
                    | {f"truth_{k}": v for k, v in truth.items()}
                )
    return pd.DataFrame(rows)


def probe_traces(folder: Path, stem: str) -> dict[float, pd.DataFrame]:
    """One video's saved traces (``--save-traces``), per noise level: ``time``,
    ``signal``, ``onset`` and ``inhale``."""
    table = pd.read_parquet(Path(folder) / f"{stem}{TRACES}")
    levels = [c.removeprefix("signal_") for c in table if c.startswith("signal_")]
    return {
        float(level): pd.DataFrame(
            {
                "time": table["time"],
                "signal": table[f"signal_{level}"],
                "onset": table[f"onset_{level}"],
                "inhale": table[f"inhale_{level}"],
            }
        )
        for level in levels
    }
