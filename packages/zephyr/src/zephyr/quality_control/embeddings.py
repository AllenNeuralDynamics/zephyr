"""The frame encoder's embeddings of a few frames spread over a clip.

Head-fixed mice barely move, so a handful of frames says whether the crop looks like
what the network trained on, at a cost of a few encoder passes instead of the clip's.
"""

import contextlib

import numpy as np
import torch

from .. import features
from ..features import ClipEntry


def spread_rows(n_frames: int, n: int) -> np.ndarray:
    """*n* frame indices spread evenly over ``[0, n_frames)``, ends excluded."""
    n = min(n, n_frames)
    return np.linspace(0, n_frames, n + 2)[1:-1].astype(int)


@torch.no_grad()
def sample(
    model,
    mean: np.ndarray,
    std: np.ndarray,
    entry: ClipEntry,
    n: int,
    device,
    amp_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings ``(n, embed)`` of *n* frames spread over *entry*, and their times.

    A frame goes through the encoder with the neighbours its first layer reads
    (``temporal_kernel``), so its embedding is the one a whole-clip pass would give.
    """
    model.eval()
    halo = model.encoder.temporal_kernel // 2
    reader = features.reader(entry)
    times = np.load(entry.frame_times)
    rows = spread_rows(len(times), n)
    mu, sd = (
        torch.from_numpy(np.asarray(v, np.float32)).view(1, -1, 1, 1)
        for v in model.channels.take_stats(mean, std)
    )
    out = []
    for row in rows:
        lo, hi = max(0, row - halo), min(len(times), row + halo + 1)
        block = torch.from_numpy(
            reader.read(slice(lo, hi), model.channels).astype(np.float32)
        )
        block = ((block - mu) / sd).unsqueeze(0).to(device)
        with (
            torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype)
            if amp_dtype is not None
            else contextlib.nullcontext()
        ):
            embedded = model.encode(block)
        out.append(embedded[0, row - lo].float().cpu().numpy())
    return np.stack(out), times[rows]
