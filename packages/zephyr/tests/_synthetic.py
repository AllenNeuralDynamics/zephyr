"""A tiny real dataset for tests that decode, train or predict: needs ffmpeg.

Each clip is a 64x48 gray video at 60 Hz whose centre brightens with a breathing sine,
its frame timestamps, and a 500 Hz thermistor of the same sine (inhale up), named as
the default ``[derive]`` expects.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
SIZE = (64, 48)
FPS = 60
BOX = (16, 8, 32, 32)


def write_video(
    path: Path,
    frames: np.ndarray,
    fps: int = FPS,
    codec: tuple[str, ...] = ("-qp", "0"),
) -> None:
    """Encode uint8 gray *frames* (T, H, W) losslessly enough for tests (*codec*: the
    x264 settings, e.g. a short GOP with B-frames)."""
    height, width = frames.shape[1:]
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            *codec,
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        input=frames.tobytes(),
        check=True,
    )


def make_clip(
    folder: Path,
    group: int,
    part: int,
    *,
    seconds: float = 20.0,
    rate_hz: float = 3.0,
    sensor_hz: float = 500.0,
    thermistor: bool = True,
    t0: float = 100.0,
) -> Path:
    """Write one clip's video, timestamps and (optionally) thermistor; the video."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(group * 10 + part)
    n = int(seconds * FPS)
    times = t0 + np.arange(n) / FPS
    phase = rng.uniform(0, 2 * np.pi)
    breath = np.sin(2 * np.pi * rate_hz * (times - t0) + phase)
    width, height = SIZE
    frames = np.full((n, height, width), 60, np.uint8)
    yy, xx = np.mgrid[:height, :width]
    blob = np.exp(-((xx - 32) ** 2 + (yy - 24) ** 2) / 60.0)
    frames += (blob[None] * (60 + 50 * breath[:, None, None])).astype(np.uint8)
    stem = f"video_face_{group}_part_{part}"
    video = folder / f"{stem}.mp4"
    write_video(video, frames)
    pd.DataFrame({"Time": times}).to_parquet(folder / f"{stem}.parquet")
    if thermistor:
        sensor_t = np.arange(times[0] - 1, times[-1] + 1, 1 / sensor_hz)
        signal = 500 + 100 * np.sin(2 * np.pi * rate_hz * (sensor_t - t0) + phase)
        pd.DataFrame({"Time": sensor_t, "Signal": signal}).to_parquet(
            folder / f"thermistor_{group}_part_{part}.parquet"
        )
    return video


def make_dataset(root: Path, groups=(1, 2, 3), parts=(1,), **kwargs) -> Path:
    """Clips of *groups* x *parts* under ``root/data`` and ``root/clips.toml`` listing
    them, each boxed; returns the list."""
    videos = [make_clip(root / "data", g, p, **kwargs) for g in groups for p in parts]
    lines = [f"[preprocess]\ntarget_size = {list(SIZE)}\n"]
    for video in videos:
        rel = video.relative_to(root).as_posix()
        lines.append(f'[[clip]]\nvideo = "{rel}"\nbox = {list(BOX)}\n')
    path = root / "clips.toml"
    path.write_text("\n".join(lines))
    return path


TINY_TRAINING = """
[fold.train_params]
epochs = 2
steps_per_epoch = 2
window = 256
batch_size = 2
warmup_steps = 1
score_every = 1
min_epochs = 0
val_windows = 4
infer_window = 512
frame_chunk = 64
"""


def write_experiment(
    root: Path,
    clips: Path,
    *,
    train=("1",),
    val=("2",),
    test=("3",),
    name: str = "experiment",
) -> Path:
    """An experiment of one fold over *clips*: training cut to a few steps on CPU."""
    rel = Path("..") / clips.relative_to(root)
    folder = root / "experiments"
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}.toml"
    path.write_text(
        f"""seeds = [1]
output_dir = "../runs/{name}"
features_dir = "../cache"
device = "cpu"
amp = "off"
num_workers = 0

[[fold]]
name = "fold"
[[fold.train]]
clips = "{rel.as_posix()}"
groups = {list(train)}
[fold.val]
clips = "{rel.as_posix()}"
groups = {list(val)}
[fold.test]
held_out = {{ clips = "{rel.as_posix()}", groups = {list(test)} }}
{TINY_TRAINING}"""
    )
    return path
