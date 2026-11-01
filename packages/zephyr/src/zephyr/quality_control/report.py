"""Build a checkpoint's reference, and one quality report per predicted video.

A reference (:func:`reference_record`) is fitted on labelled clips the checkpoint never
trained on: the only place quality control reads a thermistor. A report
(:func:`quality_report`, written by ``zephyr predict --quality-control`` as
``<video stem>.qc.json``) holds every measure over the whole clip and per window and,
with a reference, each one's percentile among its validation windows: scores from the
prediction alone, never a verdict, the same whether or not the video has a thermistor.
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
    QualityParams,
    QualityReference,
    ResolvedClip,
    canonical,
    dump,
)
from ..evaluate import (
    checkpoint_head_threshold,
    has_onset_head,
    load_checkpoint,
    refuse_leaked,
    score_span,
    truth_events,
    truth_frame,
)
from ..evaluation import frame_tolerance_s, match_events
from ..events import in_spans
from ..features import ClipEntry
from ..infer import predict_clip
from .metrics import MEASURES, Video, find_events, measure, spans
from .reference import (
    Reference,
    checkpoint_hash,
    fit_calibration,
    jsonable,
    percentiles,
)

AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}
REFERENCE = "qc-reference.json"
"""The reference training writes beside ``best.pt``."""
REPORT = ".qc.json"


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
    unseen: list[ResolvedClip],
    recipe,
    features_dir: Path,
    out: Path,
    max_clips: int | None,
    inference,
    device: str,
    amp: str,
) -> dict:
    """Calibrate *checkpoint*'s onset head on labelled clips it never trained on, and
    write the :class:`~.reference.Reference` to *out*. *inference* gives
    ``infer_window`` and ``frame_chunk``."""
    torch_device = torch.device(device)
    model, mean, std, state = load_checkpoint(checkpoint, torch_device)
    if not has_onset_head([state]):
        raise SystemExit(
            f"{checkpoint} has no onset head: there is nothing to calibrate"
        )
    unseen = _spread_out(unseen, max_clips)
    unlabelled = [c.name for c in unseen if not c.labelled]
    if unlabelled:
        raise SystemExit(f"calibration clips need a thermistor: {unlabelled}")
    refuse_leaked([state], unseen)
    threshold = checkpoint_head_threshold(state)

    predicted = []
    entries = features.require(unseen, recipe, features_dir)
    for k, entry in enumerate(entries):
        print(f"[calibration {k + 1}/{len(entries)}] {entry.clip_id}", flush=True)
        signal, onset = _predict(model, mean, std, entry, inference, torch_device, amp)
        predicted.append((entry, np.load(entry.times), signal, onset))
    record = reference_record(checkpoint, state, predicted, threshold, QualityParams())
    write_reference(record, out)
    return record


def reference_record(
    checkpoint: Path,
    state: dict,
    predicted: list[tuple[ClipEntry, np.ndarray, np.ndarray, np.ndarray]],
    threshold: float,
    params: QualityParams,
) -> dict:
    """The reference of *checkpoint* (its loaded *state*) from its *predicted* labelled
    clips (each entry, output times, trace and onset probability), its inhales at
    onset *threshold*: the head's calibration, then every measure on the clips'
    *params*-wide windows (with that calibration) beside the windows' true scores."""
    peaks, real, videos = [], [], []
    for entry, times, signal, onset in predicted:
        n = min(len(times), len(signal), len(onset))
        times, signal, onset = times[:n], signal[:n], onset[:n]
        events = find_events(times, signal, onset, threshold)
        p, r = _head_matches(entry, times, onset, events.inhale)
        peaks.append(p)
        real.append(r)
        videos.append((entry, Video(times, signal, onset, events)))
    calibration = fit_calibration(np.concatenate(peaks), np.concatenate(real))
    reference = Reference("", calibration, {})
    windows = []
    for entry, video in videos:
        truth = (truth_frame(entry), truth_events(entry))
        for span in spans(video.times, params.window_s):
            values = measure(
                video,
                span,
                ceiling_hz=params.breathing_ceiling_hz,
                ambiguous_floor=params.ambiguous_floor,
                reference=reference,
                flat_ratio=params.flat_ratio,
            )
            inhale_s = video.times[video.events.inhale]
            scores = score_span(truth, video.times, video.signal, inhale_s, span)
            windows.append(
                {"clip_id": entry.clip_id, "start": span[0], "end": span[1]}
                | values
                | {f"truth_{k}": v for k, v in (scores or {}).items()}
            )
    return jsonable(
        {
            "checkpoint": canonical(checkpoint),
            "checkpoint_sha256": checkpoint_hash(checkpoint),
            "epoch": state.get("epoch"),
            "head_threshold": threshold,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "zephyr_version": version("zephyr"),
            "calibration_clips": [entry.clip_id for entry, _ in videos],
            "onset_calibration": calibration,
            "validation": {"window_s": params.window_s, "windows": windows},
        }
    )


def write_reference(record: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {out}")


def build_reference(config: QualityReference, device: str, amp: str) -> dict:
    """``zephyr quality-control reference``: the reference *config* describes."""
    return measure_reference(
        config.checkpoint,
        [c for s in config.calibration for c in s.resolved()],
        config.preprocess,
        config.features_dir,
        config.out,
        config.max_clips,
        config,
        device,
        amp,
    )


# --- one report per video ----------------------------------------------------------------


def load_reference(path: Path, sha: str, threshold: float) -> Reference:
    """The reference at *path*, refused unless it was made for the checkpoint of
    SHA-256 *sha* at its onset *threshold*."""
    reference = Reference.load(path)
    if reference.checkpoint_sha256 != sha:
        raise SystemExit(f"{path} was made for another checkpoint")
    calibrated_at = reference.record.get("head_threshold")
    if calibrated_at is not None and calibrated_at != threshold:
        raise SystemExit(
            f"{path} calibrated the head's inhales at onset threshold {calibrated_at}, "
            f"not {threshold}; build it again"
        )
    return reference


def quality_report(
    video: Video,
    params: QualityParams,
    reference: Reference | None,
    head: bool,
    meta: dict,
) -> dict:
    """One video's report from its prediction alone: every measure over the whole clip
    and per window, with their percentiles among the *reference*'s validation windows
    when it has them. Whatever else the video comes with, the report is the same."""
    validation = reference.validation if reference is not None else None

    def scores(span) -> dict:
        out = measure(
            video,
            span,
            ceiling_hz=params.breathing_ceiling_hz,
            ambiguous_floor=params.ambiguous_floor,
            reference=reference,
            flat_ratio=params.flat_ratio,
        )
        if validation:
            out["percentiles"] = percentiles(out, validation, MEASURES)
        return out

    whole = (float(video.times[0]), float(np.nextafter(video.times[-1], np.inf)))
    return jsonable(
        {
            "inhale_events": "onset head" if head else "DSP on the trace",
            "duration_s": float(video.times[-1] - video.times[0]),
            **meta,
            "params": dump(params),
            "summary": scores(whole),
            "windows": [
                {"start": a, "end": b} | scores((a, b))
                for a, b in spans(video.times, params.window_s)
            ],
        }
    )


def write_report(report: dict, path: Path) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(report, indent=1))
    part.replace(path)


# --- reading reports back ------------------------------------------------------------------


def load_reports(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every report in *folder*: one row per video (its ``summary``) and one per window,
    each with the video's ``stem``; percentiles become ``pct_<measure>`` columns."""

    def flat(row: dict) -> dict:
        row = dict(row)
        position = row.pop("percentiles", None) or {}
        return row | {f"pct_{k}": v for k, v in position.items()}

    videos, windows = [], []
    for path in sorted(Path(folder).glob(f"*{REPORT}")):
        data = json.loads(path.read_text())
        ident = {"stem": path.name.removesuffix(REPORT)}
        videos.append(
            ident | {"duration_s": data["duration_s"]} | flat(data["summary"])
        )
        windows += [ident | flat(w) for w in data["windows"]]
    return pd.DataFrame(videos), pd.DataFrame(windows)
