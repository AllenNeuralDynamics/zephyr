"""Render a clip as a GIF of the scrubber figure; call it from a script, not a cell."""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from . import plots, results


def pipeline_gif(
    clip: results.TestClip,
    traces: pd.DataFrame,
    truth: pd.DataFrame,
    t0: float,
    path: Path,
    seconds: float = 10.0,
    fps: int = 10,
) -> Path:
    """Write *seconds* from *t0* as a GIF: video, the four channels and the traces."""
    frames: list[Image.Image] = []
    for t in t0 + np.arange(round(seconds * fps)) / fps:
        fig = plots.scrubber(
            results.raw_frame(clip, t),
            results.channel_frames(clip.entry, t),
            clip.box,
            t,
            traces,
            truth,
            t0,
            seconds,
        )
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=80)
        plt.close(fig)
        frames.append(Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE))
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0
    )
    return path
