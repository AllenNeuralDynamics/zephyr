"""Per-frame image statistics of a clip's stored gray crop: what the camera delivered."""

import numpy as np

from .. import features
from ..channels import ChannelSet
from ..features import ClipEntry

GRAY = ChannelSet.parse("gray")
STATISTICS = ("brightness", "contrast", "sharpness", "motion")
"""Mean gray level, its spread, the Laplacian's variance (focus), and the mean
absolute change from the frame before."""
CHUNK = 2048


def has_gray(entry: ClipEntry) -> bool:
    return "gray" in entry.groups


def statistics(
    entry: ClipEntry, rows: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """:data:`STATISTICS` of the frames *rows* (all frames when ``None``)."""
    reader = features.reader(entry)
    rows = np.arange(entry.n_frames) if rows is None else np.asarray(rows, int)
    out = {name: np.empty(len(rows), np.float32) for name in STATISTICS}
    for at in range(0, len(rows), CHUNK):
        part = rows[at : at + CHUNK]
        frames = reader.read(part, GRAY)[:, 0].astype(np.float32)
        before = reader.read(np.maximum(part - 1, 0), GRAY)[:, 0].astype(np.float32)
        laplacian = (
            frames[:, :-2, 1:-1]
            + frames[:, 2:, 1:-1]
            + frames[:, 1:-1, :-2]
            + frames[:, 1:-1, 2:]
            - 4 * frames[:, 1:-1, 1:-1]
        )
        block = slice(at, at + len(part))
        out["brightness"][block] = frames.mean(axis=(1, 2))
        out["contrast"][block] = frames.std(axis=(1, 2))
        out["sharpness"][block] = laplacian.var(axis=(1, 2))
        out["motion"][block] = np.abs(frames - before).mean(axis=(1, 2))
    return out
