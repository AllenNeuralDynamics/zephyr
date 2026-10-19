"""Build a checkpoint's reference, and write one quality report per video.

A report (``<video stem>.qc.json``) holds the video's measures over the whole clip
(``summary``) and per window (``windows``); a labelled clip also gets a ``truth`` block
(its real scores, never used by a measure) so the measures can be checked against it.
``<video stem>.qc.parquet`` keeps the trace, onset probability and inhales.
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
    Fold,
    QualityControl,
    QualityParams,
    QualityReference,
    ReferenceSampling,
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
from ..infer import predict_clip
from ..signal import BREATHING_SIGNAL_COLUMN, TIME_COLUMN
from . import embeddings
from .metrics import Video, find_events, measure, spans
from .reference import (
    Reference,
    checkpoint_hash,
    embedding_statistics,
    fit_calibration,
    jsonable,
    quantiles,
)

AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}
SETTINGS = "quality-control.json"
"""The run's settings, written once into its output folder."""
REPORT = ".qc.json"
TRACES = ".qc.parquet"


def machine(config, device: str | None, amp: str | None) -> tuple[str, str]:
    """CLI flag, else the config's value, else cuda if any and bf16."""
    device = device or config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return device, amp or config.amp or "bf16"


def _predict(model, mean, std, entry, inference, device, amp):
    """The network's ordinary prediction: trace and onset probability."""
    return predict_clip(
        model,
        entry,
        mean,
        std,
        window=inference.infer_window,
        device=device,
        frame_chunk=inference.frame_chunk,
        amp_dtype=AMP[amp],
    )


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


def _spread_out(clips: list[ResolvedClip], n: int | None) -> list[ResolvedClip]:
    """Up to *n* of *clips*, spread evenly over the list (all when ``None``)."""
    if n is None or len(clips) <= n:
        return list(clips)
    return [clips[i] for i in np.linspace(0, len(clips) - 1, n).round().astype(int)]


def measure_reference(
    checkpoint: Path,
    known: list[ResolvedClip],
    unseen: list[ResolvedClip],
    recipe,
    features_dir: Path,
    out: Path,
    sampling: ReferenceSampling,
    inference,
    device: str,
    amp: str,
) -> dict:
    """Measure what :class:`~.reference.Reference` holds for *checkpoint* from the clips
    it knows and labelled clips it never trained on, and write it to *out*.

    Embeddings come from ``frames_per_clip`` sampled frames of each clip, never a
    whole-clip pass; only the calibration clips are predicted whole, for the onset
    head's calibration. *inference* gives ``infer_window`` and ``frame_chunk``.
    """
    torch_device = torch.device(device)
    model, mean, std, state = load_checkpoint(checkpoint, torch_device)
    head = has_onset_head([state])
    known = _spread_out(known, sampling.max_clips)
    unseen = _spread_out(unseen, sampling.max_clips)
    unlabelled = [c.name for c in unseen if not c.labelled]
    if unlabelled:
        raise SystemExit(f"calibration clips need a thermistor: {unlabelled}")
    refuse_leaked([state], unseen)

    def embed(entry):
        vectors, _ = embeddings.sample(
            model,
            mean,
            std,
            entry,
            sampling.frames_per_clip,
            torch_device,
            AMP[amp],
        )
        return vectors

    entries = features.require(known, recipe, features_dir, model.channels)
    vectors = []
    for k, entry in enumerate(entries):
        print(f"[reference {k + 1}/{len(entries)}] {entry.clip_id}", flush=True)
        vectors.append(embed(entry))
    centre, precision = embedding_statistics(
        np.concatenate(vectors), sampling.shrinkage
    )
    reference = Reference(
        checkpoint_sha256="",
        embedding_mean=centre,
        embedding_precision=precision,
        distance_quantiles=np.empty(0),
        onset_calibration=None,
        record={},
    )

    distances, peaks, real = [], [], []
    if unseen:
        entries = features.require(unseen, recipe, features_dir, model.channels)
        for k, entry in enumerate(entries):
            print(f"[calibration {k + 1}/{len(entries)}] {entry.clip_id}", flush=True)
            distances.append(reference.distance(embed(entry)))
            if head:
                signal, onset = _predict(
                    model, mean, std, entry, inference, torch_device, amp
                )
                times = np.load(entry.times)
                events = find_events(times, signal, onset)
                p, r = _head_matches(entry, times, onset, events.inhale)
                peaks.append(p)
                real.append(r)
    else:
        distances = [reference.distance(v) for v in vectors]

    record = {
        "checkpoint": canonical(checkpoint),
        "checkpoint_sha256": checkpoint_hash(checkpoint),
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
            "frames": int(sum(len(v) for v in vectors)),
        },
        "onset_calibration": (
            fit_calibration(np.concatenate(peaks), np.concatenate(real))
            if peaks
            else None
        ),
        "sampling": dump(sampling),
    }
    record = jsonable(record)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out}")
    return record


def build_reference(config: QualityReference, device: str, amp: str) -> dict:
    """``zephyr quality-control reference``: the reference *config* describes."""
    return measure_reference(
        config.checkpoint,
        [c for s in config.clips for c in s.resolved()],
        [c for s in config.calibration for c in s.resolved()],
        config.preprocess,
        config.features_dir,
        config.out,
        config,
        config,
        device,
        amp,
    )


FOLD_REFERENCE = "qc-reference.json"
"""What ``zephyr run`` writes beside each checkpoint."""


def fold_reference(
    fold: Fold, run_dir: Path, features_dir: Path, device: str, amp: str, smoke: bool
) -> Path | None:
    """The reference of a fold's ``best.pt``, from its training clips and calibrated on
    its ``[fold.val]`` clips; kept if it exists, skipped when the fold turns it off."""
    params = fold.quality_reference
    out = run_dir / FOLD_REFERENCE
    if not params.enabled:
        return None
    if out.is_file():
        return out
    if smoke:
        params = params.model_copy(update={"max_clips": 1, "frames_per_clip": 10})
    measure_reference(
        run_dir / "best.pt",
        [c for clips in fold.train_clips() for c in clips],
        fold.val_clips(),
        fold.preprocess,
        features_dir,
        out,
        params,
        fold.train_params,
        device,
        amp,
    )
    return out


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
    model,
    mean,
    std,
    entry: ClipEntry,
    config: QualityControl,
    device,
    amp,
    reference: Reference | None,
    head: bool,
) -> tuple[Video, np.ndarray]:
    """The network's prediction for *entry* (one ordinary pass) and, with a reference,
    the embedding distances of ``embedding_frames`` sampled frames."""
    signal, onset = _predict(model, mean, std, entry, config, device, amp)
    times = np.load(entry.times)
    distances = None
    if reference is not None:
        vectors, _ = embeddings.sample(
            model, mean, std, entry, config.params.embedding_frames, device, AMP[amp]
        )
        distances = reference.distance(vectors)
    judged = onset if head else None
    video = Video(
        times=times,
        signal=signal,
        onset=judged,
        events=find_events(times, signal, judged),
        distances=distances,
    )
    return video, onset


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
            ceiling_hz=params.breathing_ceiling_hz,
            ambiguous_floor=params.ambiguous_floor,
            reference=reference,
            flat_ratio=params.flat_ratio,
            whole=whole,
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
            "duration_s": float(video.times[-1] - video.times[0]),
            **meta,
            "summary": scores(whole_span, whole=True),
            "windows": windows,
        }
    )


def traces(video: Video, onset: np.ndarray) -> pd.DataFrame:
    """One row per output sample: time, trace, onset probability and whether an
    inhale was found there."""
    inhale = np.zeros(len(video.times), bool)
    inhale[video.events.inhale] = True
    return pd.DataFrame(
        {
            "time": video.times,
            "signal": video.signal,
            "onset": onset,
            "inhale": inhale,
        }
    )


def _write_settings(config: QualityControl, checkpoint_sha: str) -> None:
    """Record the run's settings; refuse a folder holding reports made otherwise."""
    settings = {
        "checkpoint": canonical(config.checkpoint),
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
    model, mean, std, state = load_checkpoint(config.checkpoint, torch_device)
    head = has_onset_head([state])
    sha = checkpoint_hash(config.checkpoint)
    reference = None
    if config.reference is not None:
        reference = Reference.load(config.reference)
        if reference.checkpoint_sha256 != sha:
            raise SystemExit(
                f"{config.reference} was made for another checkpoint than "
                f"{config.checkpoint}"
            )
    clips = [c for s in config.clips for c in s.resolved()]
    stems = [c.source.stem for c in clips]
    repeated = sorted({s for s in stems if stems.count(s) > 1})
    if repeated:
        raise SystemExit(
            f"videos share a file name, so their reports would: {repeated}"
        )
    refuse_leaked([state], clips)
    entries = features.require(
        clips, config.preprocess, config.features_dir, model.channels
    )
    _write_settings(config, sha)
    meta = {
        "checkpoint": canonical(config.checkpoint),
        "checkpoint_sha256": sha,
        "reference": canonical(config.reference) if config.reference else None,
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
        video, onset = _video(
            model, mean, std, entry, config, torch_device, amp, reference, head
        )
        result = report(clip, entry, video, config.params, reference, head, meta)
        if config.params.save_traces:
            traces(video, onset).to_parquet(
                config.output_dir / f"{clip.source.stem}{TRACES}", index=False
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
