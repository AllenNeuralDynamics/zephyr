"""Render clips as the scrubber figure in motion; call from a script, not a cell.

:func:`session_movie` writes a whole test recording (all its videos, in order) as an
MP4 in real time: the video, the four network inputs and a scrolling trace window.
:func:`pipeline_gif` writes a short excerpt as a GIF.

    uv run --project .. python -m utils.animation 6 figures/rec-6.mp4   # from notebooks/
"""

import argparse
import io
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter
from PIL import Image

from zephyr import video

from . import plots, results, style

CURSOR: float = 0.7
"""Where the current time sits in the scrolling trace window (0 left, 1 right)."""


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


def recording_clips(recording: str) -> list[results.TestClip]:
    """The test videos of one recording (``"6"``), in name order (part 1 first)."""
    clips = [
        clip
        for name, clip in sorted(results.clips().items())
        if results.recording_label(clip.entry) == f"rec {recording}"
    ]
    if not clips:
        raise ValueError(f"no test recording {recording!r}")
    return clips


def _frames(clip: results.TestClip, fps: float, seconds: float | None):
    """``(time, frame)`` of the video every 1/*fps* s, decoded in one pass."""
    capture = cv2.VideoCapture(str(clip.video))
    native = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    k, m = 0, 0
    for block in video.iter_frames(clip.video, scale_to=clip.target_size):
        for frame in block:
            if k == round(m * native / fps):
                t = k / native
                if seconds is not None and t > seconds:
                    return
                yield t, frame
                m += 1
            k += 1


def _render(fig, axes, clip, part, parts, writer, fps, window_s, seconds) -> None:
    for ax in axes.values():
        ax.clear()
    entry = clip.entry
    out = results.zephyr_outputs(entry)
    raw = results.truth(entry)
    truth_t = out["Time"].to_numpy()
    truth = np.interp(truth_t, raw["Time"].to_numpy(), raw["Signal"].to_numpy())
    features = np.load(entry.features, mmap_mode="r")
    frame_times = np.load(entry.frame_times)

    video_image = plots.frame_with_box(
        axes["video"], np.zeros(clip.target_size[::-1]), clip.box
    )
    planes = results.channel_planes(features[0])
    channel_images = {}
    for name in style.CHANNELS:
        channel_images[name] = plots.channel_image(axes[name], name, planes[name])
        axes[name].set_title(name, color=style.CHANNELS[name])
    trace = axes["trace"]
    trace.plot(
        truth_t,
        (truth - truth.mean()) / truth.std(),
        color=style.TRUTH,
        label="Thermistor",
    )
    trace.plot(truth_t, out["Zephyr"], color=style.color("Zephyr"), label="Zephyr")
    cursor = trace.axvline(0, color="#d55e00", linewidth=1)
    trace.set_ylim(-3.5, 3.0)
    trace.set_xlabel("Time (s)")
    trace.set_ylabel("Breathing (z-score)")
    trace.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2)
    title = axes["video"].set_title("")

    for t, frame in _frames(clip, fps, seconds):
        if t > truth_t[-1]:
            break
        video_image.set_data(frame)
        index = min(int(np.searchsorted(frame_times, t)), len(frame_times) - 1)
        for name, plane in results.channel_planes(features[index]).items():
            channel_images[name].set_data(plane)
        cursor.set_xdata([t, t])
        trace.set_xlim(t - CURSOR * window_s, t + (1 - CURSOR) * window_s)
        title.set_text(
            f"{results.recording_label(entry)}, part {part}/{parts}   t = {t:.2f} s"
        )
        writer.grab_frame()


def session_movie(
    recording: str,
    path: Path,
    *,
    fps: int = 30,
    window_s: float = 10.0,
    dpi: int = 150,
    seconds: float | None = None,
) -> Path:
    """Write test recording *recording* as an H.264 MP4 playing in real time.

    Needs Zephyr's cached outputs on its videos (see
    :func:`results.zephyr_outputs`). *seconds* limits each video, for a preview.
    """
    style.use_style()
    fig, axes = style.mosaic(
        [["video", *style.CHANNELS], ["trace"] * 5],
        "double",
        0.5,
        width_ratios=[1.33, 1, 1, 1, 1],
        height_ratios=[1, 0.9],
    )
    writer = FFMpegWriter(
        fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"]
    )
    clips = recording_clips(recording)
    path.parent.mkdir(parents=True, exist_ok=True)
    # An MP4 is unplayable until its index is written at the end, so render under
    # a temporary name and only take the final name once the file is complete.
    partial = path.with_name(f"{path.stem}.partial{path.suffix}")
    with writer.saving(fig, str(partial), dpi):
        for part, clip in enumerate(clips, 1):
            _render(fig, axes, clip, part, len(clips), writer, fps, window_s, seconds)
    plt.close(fig)
    partial.replace(path)
    return path


if __name__ == "__main__":
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=session_movie.__doc__)
    parser.add_argument("recording", help='test recording number, e.g. "6"')
    parser.add_argument("out", type=Path, help="output .mp4")
    parser.add_argument("--seconds", type=float, help="limit each video (preview)")
    args = parser.parse_args()
    print(session_movie(args.recording, args.out, seconds=args.seconds))
