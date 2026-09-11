"""Whole-clip prediction by stitching overlapping windows.

A 300 s clip is 18 001 frames of 4x96x96, which is 2.5 GB as float32 -- so
inference runs in windows even though the model itself has no length limit.
Windows are overlapped and their edges discarded rather than butt-joined: the
TCN's receptive field is 253 frames, so a prediction fewer than ~126 frames from
a window boundary is computed from zero-padding rather than from data, and
butt-joining would stamp a visible artefact into the trace every window.  With
the margin trimmed, every output sample comes from a fully-populated receptive
field, and the stitched trace is identical to what an unbounded forward pass
would produce.
"""

import numpy as np
import torch

from .dataset import ClipEntry
from .model import BreathingNet


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
        Both length ``entry.n_frames``.  *signal* is z-scored over the clip --
        the scorer's ``max_xcorr`` is amplitude-invariant, and a per-clip z-score
        is the closest thing to a canonical choice.
    """
    model.eval()
    if margin is None:
        margin = model.receptive_field // 2

    channels = model.channels
    array = np.load(entry.features, mmap_mode="r")
    n_frames = len(array)
    window = min(window, n_frames)
    # A margin at or past half the window would leave no interior to keep.
    margin = min(margin, (window - 1) // 2)
    hop = max(1, window - 2 * margin)

    selected_mean, selected_std = channels.take_stats(mean, std)
    mean_t = torch.from_numpy(np.asarray(selected_mean, np.float32)).view(1, -1, 1, 1)
    std_t = torch.from_numpy(np.asarray(selected_std, np.float32)).view(1, -1, 1, 1)

    signal = np.zeros(n_frames, np.float32)
    onset = np.zeros(n_frames, np.float32)
    filled = np.zeros(n_frames, bool)

    for start in window_starts(n_frames, window, hop):
        stop = start + window
        # Cast in numpy: the memmap slice is read-only, and torch.from_numpy
        # warns on non-writable storage.  The cast has to copy anyway.
        block = torch.from_numpy(
            channels.take(np.asarray(array[start:stop])).astype(np.float32)
        )
        block = ((block - mean_t) / std_t).unsqueeze(0).to(device, non_blocking=True)

        if amp_dtype is not None:
            with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype):
                pred_signal, pred_onset = model(block, chunk=frame_chunk)
        else:
            pred_signal, pred_onset = model(block, chunk=frame_chunk)

        pred_signal = pred_signal.float().squeeze(0).cpu().numpy()
        pred_onset = torch.sigmoid(pred_onset.float()).squeeze(0).cpu().numpy()

        # Trim the padding-contaminated edges, except where the window sits
        # against the true start or end of the clip -- there the padding is real.
        lo = start + (margin if start > 0 else 0)
        hi = stop - (margin if stop < n_frames else 0)
        signal[lo:hi] = pred_signal[lo - start : hi - start]
        onset[lo:hi] = pred_onset[lo - start : hi - start]
        filled[lo:hi] = True

    if not filled.all():
        raise RuntimeError(
            f"{entry.clip_id}: {int((~filled).sum())} frames were never predicted "
            f"(window={window}, margin={margin}, hop={hop})"
        )

    scale = signal.std()
    signal = (signal - signal.mean()) / (scale if scale > 0 else 1.0)
    return signal, onset
