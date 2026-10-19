"""Build a checkpoint's reference, and write one quality report per video.

A report (``<video stem>.qc.json``) holds the video's measures over the whole clip
(``summary``) and per window (``windows``); a labelled clip also gets a ``truth`` block
(its real scores, never used by a measure) so the measures can be checked against it.
``<video stem>.qc.npz`` keeps the trace, onset probability and the draws' spread.
"""

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import features
from ..config import (
    QualityControl,
    QualityParams,
    QualityReference,
    ResolvedClip,
    canonical,
    dump,
)
from ..evaluate import (
    has_onset_head,
    load_checkpoint,
    refuse_leaked,
    truth_events,
    truth_frame,
)
from ..evaluation import frame_tolerance_s, match_events, score_clip
from ..events import in_spans
from ..features import ClipEntry
from ..infer import predict_clip, predict_samples
from ..signal import BREATHING_SIGNAL_COLUMN, TIME_COLUMN
from . import frames
from .metrics import Video, find_events, measure, spans, spread
from .reference import (
    INTERVAL_EDGES,
    Reference,
    checkpoint_hash,
    embedding_statistics,
    fit_calibration,
    interval_density,
    jsonable,
    quantiles,
)

AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}
SETTINGS = "quality-control.json"
"""The run's settings, written once into its output folder."""
REPORT = ".qc.json"
TRACES = ".qc.npz"


def machine(config, device: str | None, amp: str | None) -> tuple[str, str]:
    """CLI flag, else the config's value, else cuda if any and bf16."""
    device = device or config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return device, amp or config.amp or "bf16"


def _predict(model, mean, std, entry, config, device, amp, passes=0, seed=0):
    return predict_samples(
        model,
        entry,
        mean,
        std,
        passes=passes,
        seed=seed,
        window=config.infer_window,
        device=device,
        frame_chunk=config.frame_chunk,
        amp_dtype=AMP[amp],
    )


def _sample_rows(embeddings: np.ndarray, n: int) -> np.ndarray:
    """Up to *n* frames spread evenly over those with an embedding."""
    known = np.flatnonzero(np.isfinite(embeddings).all(axis=1))
    if len(known) <= n:
        return known
    return known[np.linspace(0, len(known) - 1, n).round().astype(int)]


def _truth_intervals(entry: ClipEntry) -> np.ndarray:
    """Intervals between the clip's true inhales, none reaching into an excluded span."""
    found = truth_events(entry)
    onsets = np.sort(np.asarray(found.inhale, float))
    if len(onsets) < 2:
        return np.empty(0)
    start, stop = onsets[:-1], onsets[1:]
    keep = np.ones(len(start), bool)
    for lo, hi in np.asarray(found.excluded, float).reshape(-1, 2):
        keep &= ~((start <= hi) & (stop >= lo))
    return (stop - start)[keep]


def _head_matches(
    entry: ClipEntry, times, onset, inhale
) -> tuple[np.ndarray, np.ndarray]:
    """Peak height of each inhale the head asserts outside excluded spans, and whether
    it matches a true inhale."""
    found = truth_events(entry)
    spans_ = np.asarray(found.excluded, float).reshape(-1, 2)
    asserted = inhale[~in_spans(times[inhale], spans_)]
    truth = np.asarray(found.inhale, float)
    truth = truth[~in_spans(truth, spans_)]
    real = np.zeros(len(asserted))
    for _, j in match_events(truth, times[asserted], frame_tolerance_s(times)):
        real[j] = 1
    return onset[asserted], real


def build_reference(config: QualityReference, device: str, amp: str) -> dict:
    """Measure what :class:`~.reference.Reference` holds and write it to ``config.out``."""
    torch_device = torch.device(device)
    model, mean, std, state = load_checkpoint(config.checkpoint, torch_device)
    head = has_onset_head([state])
    known = [c for s in config.clips for c in s.resolved()]
    unseen = [c for s in config.calibration for c in s.resolved()]
    unlabelled = [c.name for c in unseen if not c.labelled]
    if unlabelled:
        raise SystemExit(f"calibration clips need a thermistor: {unlabelled}")
    refuse_leaked([state], unseen)
    recipe = config.preprocess

    embeddings, image, intervals = [], [], []
    entries = features.require(known, recipe, config.features_dir, model.channels)
    for k, entry in enumerate(entries):
        print(f"[{k + 1}/{len(entries)}] {entry.clip_id}", flush=True)
        samples = _predict(model, mean, std, entry, config, torch_device, amp)
        rows = _sample_rows(samples.embeddings, config.frames_per_clip)
        embeddings.append(samples.embeddings[rows])
        if frames.has_gray(entry):
            image.append(frames.statistics(entry, rows))
        if entry.events is not None:
            intervals.append(_truth_intervals(entry))
    centre, precision = embedding_statistics(
        np.concatenate(embeddings), config.shrinkage
    )
    reference = Reference(
        checkpoint_sha256="",
        embedding_mean=centre,
        embedding_precision=precision,
        distance_quantiles=np.empty(0),
        image_quantiles=None,
        interval_edges=INTERVAL_EDGES,
        interval_density=None,
        onset_calibration=None,
        record={},
    )

    distances, peaks, real = [], [], []
    if unseen:
        entries = features.require(unseen, recipe, config.features_dir, model.channels)
        for k, entry in enumerate(entries):
            print(f"[calibration {k + 1}/{len(entries)}] {entry.clip_id}", flush=True)
            samples = _predict(model, mean, std, entry, config, torch_device, amp)
            rows = _sample_rows(samples.embeddings, config.frames_per_clip)
            distances.append(reference.distance(samples.embeddings[rows]))
            if head:
                times = np.load(entry.times)
                events = find_events(times, samples.signal, samples.onset)
                p, r = _head_matches(entry, times, samples.onset, events.inhale)
                peaks.append(p)
                real.append(r)
    else:
        distances = [reference.distance(e) for e in embeddings]

    record = {
        "checkpoint": canonical(config.checkpoint),
        "checkpoint_sha256": checkpoint_hash(config.checkpoint),
        "epoch": state.get("epoch"),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "zephyr_version": version("zephyr"),
        "clips": [c.name for c in known],
        "calibration_clips": [c.name for c in unseen],
        "embedding": {
            "mean": centre,
            "precision": precision,
            "distance_quantiles": quantiles(np.concatenate(distances)),
            "distances_from": "calibration" if unseen else "clips",
            "frames": int(sum(len(e) for e in embeddings)),
        },
        "image": (
            {
                name: quantiles(np.concatenate([i[name] for i in image]))
                for name in frames.STATISTICS
            }
            if image
            else None
        ),
        "intervals": {
            "edges": INTERVAL_EDGES,
            "density": interval_density(np.concatenate(intervals))
            if intervals
            else None,
            "n": int(sum(len(i) for i in intervals)),
        },
        "onset_calibration": (
            fit_calibration(np.concatenate(peaks), np.concatenate(real))
            if peaks
            else None
        ),
        "settings": dump(config),
    }
    record = jsonable(record)
    config.out.parent.mkdir(parents=True, exist_ok=True)
    config.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {config.out}")
    return record


# --- one report per video ----------------------------------------------------------------


def _truth(truth, times, signal, inhale_s, span, whole) -> dict | None:
    """The clip's real scores over *span*, as ``zephyr evaluate`` scores a clip (over
    the *whole* clip exactly so: every true event, as it passes them). *truth* is the
    clip's thermistor frame and cached event set."""
    thermistor, found = truth
    t = thermistor[TIME_COLUMN].to_numpy()
    rows = (t >= span[0] - 1.0) & (t < span[1] + 1.0)
    mask = (times >= span[0]) & (times < span[1])
    if rows.sum() < 2 or mask.sum() < 2:
        return None
    inhale, exhale = np.asarray(found.inhale, float), np.asarray(found.exhale, float)
    if not whole:
        inhale = inhale[(inhale >= span[0]) & (inhale < span[1])]
        exhale = exhale[(exhale >= span[0]) & (exhale < span[1])]
        if inhale_s is not None:
            inhale_s = inhale_s[(inhale_s >= span[0]) & (inhale_s < span[1])]
    predicted = pd.DataFrame(
        {
            TIME_COLUMN: times[mask].astype(np.float64),
            BREATHING_SIGNAL_COLUMN: signal[mask].astype(np.float64),
        }
    )
    score = score_clip(
        thermistor[rows],
        predicted,
        truth_onset_times_s=inhale,
        truth_offset_times_s=exhale,
        predicted_onset_times_s=inhale_s,
        excluded_s=found.excluded,
    )
    return score.to_dict()


def _video(
    entries: list[tuple],
    entry: ClipEntry,
    config: QualityControl,
    device,
    amp,
    reference: Reference | None,
    head: bool,
) -> tuple[Video, np.ndarray]:
    """Predict *entry* with the first model and its draws (dropout or the others)."""
    params = config.params
    model, mean, std = entries[0]
    passes = params.dropout_passes if params.agreement == "dropout" else 0
    samples = _predict(
        model, mean, std, entry, config, device, amp, passes=passes, seed=params.seed
    )
    times = np.load(entry.times)
    onset = samples.onset if head else None
    draws, draw_onsets = samples.signal_draws, samples.onset_draws
    if params.agreement == "checkpoints":
        outputs = [
            predict_clip(
                m,
                entry,
                mu,
                sd,
                window=config.infer_window,
                device=device,
                frame_chunk=config.frame_chunk,
                amp_dtype=AMP[amp],
            )
            for m, mu, sd in entries[1:]
        ]
        draws = np.stack([s for s, _ in outputs])
        draw_onsets = np.stack([o for _, o in outputs])
    draw_events = [
        find_events(times, d, o if head else None)
        for d, o in zip(draws, draw_onsets, strict=True)
    ]
    video = Video(
        times=times,
        signal=samples.signal,
        onset=onset,
        events=find_events(times, samples.signal, onset),
        draws=draws,
        draw_events=draw_events,
        frame_times=np.load(entry.frame_times),
        distances=(
            reference.distance(samples.embeddings) if reference is not None else None
        ),
        image=frames.statistics(entry) if frames.has_gray(entry) else None,
    )
    return video, samples.onset


def report(
    clip: ResolvedClip,
    entry: ClipEntry,
    video: Video,
    params: QualityParams,
    reference: Reference | None,
    head: bool,
    meta: dict,
) -> dict:
    """One video's report: its measures over the whole clip and per window."""
    truth = None
    if entry.thermistor is not None:
        truth = (truth_frame(entry), truth_events(entry))

    def scores(span, whole=False):
        out = measure(
            video,
            span,
            band=params.rate_band_hz,
            jump=params.rate_jump,
            ambiguous_floor=params.ambiguous_floor,
            reference=reference,
        )
        if truth is not None:
            inhale_s = video.times[video.events.inhale] if head else None
            out["truth"] = _truth(
                truth, video.times, video.signal, inhale_s, span, whole
            )
        return out

    whole_span = (float(video.times[0]), float(np.nextafter(video.times[-1], np.inf)))
    windows = [
        {"start": a, "end": b} | scores((a, b))
        for a, b in spans(video.times, params.window_s)
    ]
    return jsonable(
        {
            "clip_id": entry.clip_id,
            "video": canonical(clip.video),
            "recording": entry.recording,
            "labelled": entry.thermistor is not None,
            "inhale_events": "onset head" if head else "DSP on the trace",
            "draws": len(video.draws),
            "duration_s": float(video.times[-1] - video.times[0]),
            **meta,
            "summary": scores(whole_span, whole=True),
            "windows": windows,
        }
    )


def _write_settings(config: QualityControl, checkpoint_sha: str) -> None:
    """Record the run's settings; refuse a folder holding reports made otherwise."""
    settings = {
        "checkpoints": [canonical(p) for p in config.checkpoints],
        "checkpoint_sha256": checkpoint_sha,
        "reference": canonical(config.reference) if config.reference else None,
        "params": dump(config.params),
    }
    path = config.output_dir / SETTINGS
    if path.is_file():
        before = json.loads(path.read_text())
        if before != settings:
            raise SystemExit(
                f"{config.output_dir} holds reports made with other settings "
                f"({path}); use a new output_dir"
            )
        return
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2))


def run(config: QualityControl, device: str, amp: str) -> list[Path]:
    """Write a report for every selected video that has none yet; returns them all."""
    torch_device = torch.device(device)
    models, states = [], []
    for path in config.checkpoints:
        model, mean, std, state = load_checkpoint(path, torch_device)
        models.append((model, mean, std))
        states.append(state)
    head = has_onset_head(states[:1])
    reference = None
    if config.reference is not None:
        reference = Reference.load(config.reference)
        if reference.checkpoint_sha256 != checkpoint_hash(config.checkpoints[0]):
            raise SystemExit(
                f"{config.reference} was made for another checkpoint than "
                f"{config.checkpoints[0]}"
            )
    clips = [c for s in config.clips for c in s.resolved()]
    stems = [c.source.stem for c in clips]
    repeated = sorted({s for s in stems if stems.count(s) > 1})
    if repeated:
        raise SystemExit(
            f"videos share a file name, so their reports would: {repeated}"
        )
    refuse_leaked(states, clips)
    channels = models[0][0].channels
    entries = features.require(clips, config.preprocess, config.features_dir, channels)
    sha = checkpoint_hash(config.checkpoints[0])
    _write_settings(config, sha)
    meta = {
        "checkpoint": canonical(config.checkpoints[0]),
        "checkpoint_sha256": sha,
        "reference": canonical(config.reference) if config.reference else None,
        "agreement": config.params.agreement,
        "zephyr_version": version("zephyr"),
        "params": dump(config.params),
    }
    written = []
    for k, (clip, entry) in enumerate(zip(clips, entries, strict=True)):
        out = config.output_dir / f"{clip.source.stem}{REPORT}"
        written.append(out)
        if out.is_file():
            continue
        print(f"[{k + 1}/{len(clips)}] {entry.clip_id}", flush=True)
        video, onset = _video(models, entry, config, torch_device, amp, reference, head)
        result = report(clip, entry, video, config.params, reference, head, meta)
        if config.params.save_traces:
            np.savez_compressed(
                config.output_dir / f"{clip.source.stem}{TRACES}",
                times=video.times,
                signal=video.signal,
                onset=onset,
                spread=spread(video),
                inhale=video.times[video.events.inhale],
            )
        part = out.with_name(out.name + ".part")
        part.write_text(json.dumps(result, indent=1))
        part.replace(out)
    return written


# --- reading reports back ------------------------------------------------------------------


def load_reports(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every report in *folder*: one row per video (its ``summary``) and one per window,
    each with the report's identity; ``truth`` scores become ``truth_<metric>`` columns."""

    def flat(row: dict) -> dict:
        truth = row.pop("truth", None) or {}
        return row | {f"truth_{k}": v for k, v in truth.items()}

    videos, windows = [], []
    for path in sorted(Path(folder).glob(f"*{REPORT}")):
        data = json.loads(path.read_text())
        ident = {
            "stem": path.name.removesuffix(REPORT),
            "clip_id": data["clip_id"],
            "recording": data["recording"],
        }
        videos.append(
            ident | {"duration_s": data["duration_s"]} | flat(data["summary"])
        )
        windows += [ident | flat(w) for w in data["windows"]]
    return pd.DataFrame(videos), pd.DataFrame(windows)
