"""Whole-clip prediction by stitching overlapping windows.

A 300 s clip is 2.5 GB as float32, so inference runs in windows although the model has
no length limit. Windows overlap and their edges are discarded rather than butt-joined:
within ~126 frames (half the TCN's receptive field) of an edge a prediction comes from
zero-padding, which would stamp an artefact into the trace every window. With the margin
trimmed the stitched trace equals an unbounded forward pass. Each window reads the
selection frames covering its span; there is no time-stretch.
"""

import contextlib
from dataclasses import dataclass

import numpy as np
import torch

from zephyr.model import BreathingNet, resample_embeddings

from . import features
from .dataset import INTERP_MARGIN, ClipEntry

INPUT_MARGIN = INTERP_MARGIN
"""Same slack the training windows keep."""


def window_starts(n_frames: int, window: int, hop: int) -> list[int]:
    """Window start offsets covering ``[0, n_frames)`` with the last one flush.

    The final window is snapped back to ``n_frames - window`` so the tail is
    covered by real data instead of padding, which costs a little extra overlap
    and nothing else.
    """
    if n_frames <= window:
        return [0]
    starts = list(range(0, n_frames - window + 1, hop))
    if starts[-1] != n_frames - window:
        starts.append(n_frames - window)
    return starts


def window_spans(n_out: int, window: int, margin: int):
    """``(start, stop, lo, hi)`` per window: the output samples it predicts, and the
    part kept, its padding-contaminated edges trimmed except where it sits against the
    true start or end of the clip (there the padding is real)."""
    window = min(window, n_out)
    margin = min(margin, (window - 1) // 2)
    hop = max(1, window - 2 * margin)
    for start in window_starts(n_out, window, hop):
        stop = start + window
        lo = start + (margin if start > 0 else 0)
        hi = stop - (margin if stop < n_out else 0)
        yield start, stop, lo, hi


def frame_range(frame_times: np.ndarray, out_times: np.ndarray) -> tuple[int, int]:
    """The selection frames covering *out_times*, plus :data:`INPUT_MARGIN` slack."""
    first = max(0, int(np.searchsorted(frame_times, out_times[0])) - INPUT_MARGIN)
    last = min(
        len(frame_times),
        int(np.searchsorted(frame_times, out_times[-1])) + 1 + INPUT_MARGIN,
    )
    return first, last


def channel_stats(channels, mean: np.ndarray, std: np.ndarray):
    """Per-channel normalisation of *channels*, shaped to broadcast over frames."""
    selected_mean, selected_std = channels.take_stats(mean, std)
    mean_t = torch.from_numpy(np.asarray(selected_mean, np.float32)).view(1, -1, 1, 1)
    std_t = torch.from_numpy(np.asarray(selected_std, np.float32)).view(1, -1, 1, 1)
    return mean_t, std_t


def window_inputs(array, channels, frame_times, out_times, stats, device):
    """The normalised frames covering *out_times* and both time axes (relative to the
    window's first output time), on *device*; also the frames' ``(first, last)``."""
    first, last = frame_range(frame_times, out_times)
    mean_t, std_t = stats
    # Cast in numpy: the memmap slice is read-only, and torch.from_numpy
    # warns on non-writable storage.  The cast has to copy anyway.
    block = torch.from_numpy(
        array.read(slice(first, last), channels).astype(np.float32)
    )
    block = ((block - mean_t) / std_t).unsqueeze(0).to(device, non_blocking=True)
    origin = out_times[0]
    t_in = (
        torch.from_numpy((frame_times[first:last] - origin).astype(np.float32))
        .unsqueeze(0)
        .to(device, non_blocking=True)
    )
    t_out = (
        torch.from_numpy((out_times - origin).astype(np.float32))
        .unsqueeze(0)
        .to(device, non_blocking=True)
    )
    return block, t_in, t_out, (first, last)


@torch.no_grad()
def predict_clip(
    model: BreathingNet,
    entry: ClipEntry,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    window: int = 1024,
    margin: int | None = None,
    device: torch.device | str = "cuda",
    frame_chunk: int = 256,
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict the full trace for one clip.

    *mean* and *std* cover every stored channel; both they and the stored array
    are sliced to ``model.channels`` here.

    Returns
    -------
    (signal, onset_prob)
        Both aligned with ``np.load(entry.times)``, the clip's 60 Hz output
        grid.  *signal* is z-scored over the clip -- the scorer's ``correlation``
        is amplitude-invariant, and a per-clip z-score is the closest thing to
        a canonical choice.
    """
    model.eval()
    if margin is None:
        margin = model.receptive_field // 2

    channels = model.channels
    array = features.reader(entry)
    frame_times = np.load(entry.frame_times)
    grid = np.load(entry.times)
    n_out = len(grid)
    window = min(window, n_out)
    # A margin at or past half the window would leave no interior to keep.
    margin = min(margin, (window - 1) // 2)
    hop = max(1, window - 2 * margin)

    mean_t, std_t = channel_stats(channels, mean, std)

    signal = np.zeros(n_out, np.float32)
    onset = np.zeros(n_out, np.float32)
    filled = np.zeros(n_out, bool)

    for start, stop, lo, hi in window_spans(n_out, window, margin):
        out_times = grid[start:stop]
        block, t_in, t_out, _ = window_inputs(
            array, channels, frame_times, out_times, (mean_t, std_t), device
        )

        if amp_dtype is not None:
            with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype):
                pred_signal, pred_onset = model(block, t_in, t_out, chunk=frame_chunk)
        else:
            pred_signal, pred_onset = model(block, t_in, t_out, chunk=frame_chunk)

        pred_signal = pred_signal.float().squeeze(0).cpu().numpy()
        pred_onset = torch.sigmoid(pred_onset.float()).squeeze(0).cpu().numpy()

        signal[lo:hi] = pred_signal[lo - start : hi - start]
        onset[lo:hi] = pred_onset[lo - start : hi - start]
        filled[lo:hi] = True

    if not filled.all():
        raise RuntimeError(
            f"{entry.clip_id}: {int((~filled).sum())} samples were never predicted "
            f"(window={window}, margin={margin}, hop={hop})"
        )

    # Part of the model interface, unused by zephyr: a network trained on a
    # transformed target may define postprocess(signal, fs) to rebuild the trace,
    # once, on the stitched whole clip and before the z-score.
    postprocess = getattr(model, "postprocess", None)
    if postprocess is not None:
        fs = 1.0 / float(np.median(np.diff(grid)))
        signal = postprocess(signal, fs).astype(np.float32)

    scale = signal.std()
    signal = (signal - signal.mean()) / (scale if scale > 0 else 1.0)
    return signal, onset


def _autocast(device, amp_dtype):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype)


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    scale = x.std(axis=-1, keepdims=True)
    return (x - x.mean(axis=-1, keepdims=True)) / np.where(scale > 0, scale, 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class Samples:
    """One clip's prediction with its dropout draws and frame embeddings.

    *signal*/*onset* are what :func:`predict_clip` returns; *signal_draws* and
    *onset_draws* ``(passes, T)`` the same with the temporal network's dropout on
    (each draw z-scored); *embeddings* ``(n_frames, embed)`` the encoder's output per
    selection frame (NaN where no window kept the frame).
    """

    signal: np.ndarray
    onset: np.ndarray
    signal_draws: np.ndarray
    onset_draws: np.ndarray
    embeddings: np.ndarray


def _dropout(model: BreathingNet, on: bool) -> None:
    """Switch only the temporal network's dropout (never its batch norms)."""
    for module in model.temporal.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train(on)


@torch.no_grad()
def predict_samples(
    model: BreathingNet,
    entry: ClipEntry,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    passes: int = 0,
    seed: int = 0,
    window: int = 1024,
    device: torch.device | str = "cuda",
    frame_chunk: int = 256,
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> Samples:
    """:func:`predict_clip` plus *passes* Monte Carlo dropout draws and the embeddings.

    The encoder runs once per window; only the temporal network is re-run, all draws
    as one batch. Draws are seeded by *seed* (the global torch generator), so a clip's
    draws do not depend on what ran before it.
    """
    model.eval()
    torch.manual_seed(seed)
    margin = model.receptive_field // 2
    channels = model.channels
    array = features.reader(entry)
    frame_times = np.load(entry.frame_times)
    grid = np.load(entry.times)
    n_out = len(grid)
    stats = channel_stats(channels, mean, std)

    signal = np.zeros(n_out, np.float32)
    onset = np.zeros(n_out, np.float32)
    signal_draws = np.zeros((passes, n_out), np.float32)
    onset_draws = np.zeros((passes, n_out), np.float32)
    embeddings: np.ndarray | None = None
    filled = np.zeros(n_out, bool)

    for start, stop, lo, hi in window_spans(n_out, window, margin):
        out_times = grid[start:stop]
        block, t_in, t_out, (first, last) = window_inputs(
            array, channels, frame_times, out_times, stats, device
        )
        with _autocast(device, amp_dtype):
            embedded = model.encode(block, chunk=frame_chunk)
            resampled = resample_embeddings(embedded, t_in, t_out)
            outputs = [model.temporal(resampled)]
            if passes:
                _dropout(model, True)
                try:
                    outputs.append(model.temporal(resampled.expand(passes, -1, -1)))
                finally:
                    _dropout(model, False)
        keep = slice(lo - start, hi - start)
        s, o = (x.float().cpu().numpy() for x in outputs[0])
        signal[lo:hi], onset[lo:hi] = s[0, keep], _sigmoid(o[0, keep])
        if passes:
            s, o = (x.float().cpu().numpy() for x in outputs[1])
            signal_draws[:, lo:hi] = s[:, keep]
            onset_draws[:, lo:hi] = _sigmoid(o[:, keep])
        filled[lo:hi] = True

        # Each frame is kept by the window whose kept span holds its time; the
        # clip's ends also keep the frames before its first and after its last
        # output sample.
        vectors = embedded.float().squeeze(0).cpu().numpy()
        if embeddings is None:
            embeddings = np.full(
                (len(frame_times), vectors.shape[1]), np.nan, np.float32
            )
        lo_t = -np.inf if lo == 0 else grid[lo]
        hi_t = np.inf if hi == n_out else grid[hi]
        times = frame_times[first:last]
        mine = (times >= lo_t) & (times < hi_t)
        embeddings[first:last][mine] = vectors[mine]

    if not filled.all():
        raise RuntimeError(f"{entry.clip_id}: some samples were never predicted")
    assert embeddings is not None
    return Samples(
        signal=_zscore_rows(signal),
        onset=onset,
        signal_draws=_zscore_rows(signal_draws),
        onset_draws=onset_draws,
        embeddings=embeddings,
    )
