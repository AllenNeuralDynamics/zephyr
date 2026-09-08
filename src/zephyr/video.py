"""Grayscale frame decoding through an ffmpeg pipe.

``cv2.VideoCapture`` frame-by-frame decoding is the bottleneck in a naive
pipeline.  Piping raw grayscale out of a single ffmpeg process instead decodes
much faster at reduced resolution, because scaling and colour conversion
happen inside ffmpeg's own pipeline.

Frame indexing
--------------
:func:`iter_frames` decodes from the start of the file so that yielded frame
``i`` is row ``i`` of the clip's timestamp table.  Passing ``start_s`` enables
ffmpeg's accurate seek, which is good enough for grabbing a representative
frame (as :mod:`.annotate` does when caching them) but should be avoided when
exact frame indices matter, since a stream-copied file's first frame may sit
up to one GOP before the requested window.
"""

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np


Box = tuple[int, int, int, int]
"""Crop box in full-resolution pixels, as ``(x, y, width, height)``."""


def probe_size(video: Path) -> tuple[int, int]:
    """Return the ``(width, height)`` of a video's first stream."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def _build_command(
    video: Path,
    *,
    scale_to: tuple[int, int],
    crop: Box | None,
    start_s: float | None,
    dur_s: float | None,
) -> list[str]:
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start_s is not None:
        cmd += ["-ss", f"{start_s:.6f}"]
    cmd += ["-i", str(video)]
    if dur_s is not None:
        cmd += ["-t", f"{dur_s:.6f}"]

    # Emit exactly the frames that are in the file, in presentation order.
    # ffmpeg's default for a non-streamcopy output is constant frame rate, which
    # *duplicates or drops* frames to force a uniform cadence -- and a camera
    # whose timestamps are close to but not exactly uniform will silently gain
    # or lose frames under that default, shifting every later frame off its
    # timestamp.
    cmd += ["-fps_mode", "passthrough"]

    # Scale first, then crop: crop boxes are expressed in the *downsampled* frame,
    # which is the frame the annotator actually placed them on.  `area` averages
    # over the source footprint; the default bilinear kernel samples too sparsely
    # when downscaling and aliases away exactly the kind of small-amplitude
    # texture motion this pipeline is trying to measure.
    filters = [f"scale={scale_to[0]}:{scale_to[1]}:flags=area"]
    if crop is not None:
        x, y, w, h = crop
        filters.append(f"crop={w}:{h}:{x}:{y}")
    filters.append("format=gray")

    return [*cmd, "-vf", ",".join(filters), "-f", "rawvideo", "-"]


def iter_frames(
    video: Path,
    *,
    scale_to: tuple[int, int],
    crop: Box | None = None,
    start_s: float | None = None,
    dur_s: float | None = None,
    chunk_frames: int = 256,
) -> Iterator[np.ndarray]:
    """Yield uint8 blocks of at most *chunk_frames* frames.

    The frame is first scaled to *scale_to*, then optionally cropped, so *crop*
    is in ``scale_to`` coordinates and the yielded shape is the crop's height and
    width (or ``scale_to`` reversed when there is no crop).  There is no second
    rescale after the crop: what you crop is what you get.

    Blocks are views into a buffer that is not reused, so they stay valid after
    the next iteration, but they are read-only — copy before mutating.
    """
    out_w, out_h = (crop[2], crop[3]) if crop is not None else scale_to
    cmd = _build_command(
        video, scale_to=scale_to, crop=crop, start_s=start_s, dur_s=dur_s
    )
    frame_bytes = out_w * out_h

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(frame_bytes * chunk_frames)
            if not buf:
                break
            n = len(buf) // frame_bytes
            if n == 0:
                break
            yield np.frombuffer(buf, dtype=np.uint8, count=n * frame_bytes).reshape(
                n, out_h, out_w
            )
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed on {video.name}: {stderr.strip()}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def decode_window(
    video: Path,
    *,
    scale_to: tuple[int, int],
    crop: Box | None = None,
    start_s: float | None = None,
    dur_s: float | None = None,
) -> np.ndarray:
    """Decode a window fully into one uint8 array; see :func:`iter_frames`."""
    blocks = list(
        iter_frames(video, scale_to=scale_to, crop=crop, start_s=start_s, dur_s=dur_s)
    )
    if not blocks:
        raise ValueError(f"No frames decoded from {video}")
    return np.concatenate(blocks, axis=0)


def clamp_box(box: Box, frame_w: int, frame_h: int) -> Box:
    """Shift *box* so it lies inside a ``frame_w`` × ``frame_h`` frame.

    The box is translated rather than resized, so the output keeps the requested
    width and height (shrinking only if the box is larger than the frame). This
    keeps every clip's crop the same shape, which the model requires.
    """
    x, y, w, h = box
    w, h = min(w, frame_w), min(h, frame_h)
    x = int(np.clip(x, 0, frame_w - w))
    y = int(np.clip(y, 0, frame_h - h))
    return x, y, w, h
