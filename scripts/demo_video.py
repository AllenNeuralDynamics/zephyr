"""Make the demo video: three camera views of a held-out mouse, each with the crop its
network reads (picture in picture), over scrolling lanes of the thermistor and the
face, side and front networks' traces, their inhalations as dashed lines.

    uv run python scripts/demo_video.py prepare <workdir> [--span <session>_span<k>]
    uv run python scripts/demo_video.py render <workdir> <out.mp4> [--still <s>]

`prepare` downloads the span's three videos and frame times, reads each network's crop
box from its run, and saves the four traces and each network's inhalations (its saved
trace and onset output, at its own threshold) for the minute whose breathing rate
varies most. The runs are the results manifest's (notebooks/utils/runs.py).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))

SPAN = "828425_2026-02-06_20-34-23_span1"  # held out by fold cv_0
FOLD = "cv_0"
NETWORKS = {
    "face": ("Zephyr", "FaceCamera"),
    "side": ("Zephyr, side camera", "SideCamera"),
    "front": ("Zephyr, front camera", "FrontCamera"),
}
FPS, MINUTE, HALF = 60, 60.0, 3.0  # output rate, clip length, half the trace window (s)
W, H = 1920, 1080
BG = "#000000"
COL = {"truth": "#FFFFFF", "face": "#00E5FF", "side": "#FF3EA5", "front": "#A6FF3D"}
NAME = {
    "truth": "Thermistor",
    "face": "Face camera",
    "side": "Side camera",
    "front": "Front camera",
}
PIP_LEFT = {"face": True, "side": False, "front": True}
"""Which lower corner of each view holds its picture in picture (away from its box)."""
VIDEO_SIZE = (720, 540)


def _box(config: dict, video: str) -> list[int]:
    """The crop box of the clip whose video ends with *video* in a run's config, and
    the frame size it was placed on (``target_size`` of its recipe)."""
    boxes, sizes = [], []

    def walk(x):
        if isinstance(x, dict):
            if "target_size" in x:
                sizes.append(x["target_size"])
            if str(x.get("video", "")).endswith(video) and "box" in x:
                boxes.append(x["box"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(config)
    if not boxes or not sizes:
        raise KeyError(f"no box or recipe frame size for {video}")
    # Boxes are in the frame the run's recipe resizes to: scale them to the video's.
    scale = VIDEO_SIZE[0] / sizes[0][0]
    return [round(v * scale) for v in boxes[0]]


def prepare(work: Path, span: str) -> None:
    from utils import runs
    from zephyr.evaluate import head_event_indices

    work.mkdir(parents=True, exist_ok=True)
    s3 = runs._s3()
    dataset = "aind-scratch-data/vr-foraging/zephyr-dataset/clips"
    for _, camera in NETWORKS.values():
        for ext in ("mp4", "parquet"):
            target = work / f"{span}_{camera}.{ext}"
            if not target.is_file():
                s3.get(f"{dataset}/{span}_{camera}.{ext}", str(target))
    truth = runs.truth(f"{span}_FaceCamera")
    out = {
        "truth_t": truth["times"],
        "truth": truth["signal"],
        "truth_inh": truth["inhale"],
    }
    boxes = {}
    for key, (label, camera) in NETWORKS.items():
        name = runs.cv_name(label, FOLD)
        p = runs.prediction(name, f"{span}_{camera}")
        found = head_event_indices(p["onset"], p["times"], runs.head_threshold(name))
        out |= {
            f"{key}_t": p["times"],
            key: p["signal"],
            f"{key}_inh": p["times"][found],
        }
        m = runs.RUNS[name]
        config = runs._fetch(
            f"{runs.folder(m)}/{runs.seeds(runs.folder(m))[0]}/config.json"
        )
        boxes[key] = _box(json.loads(Path(config).read_text()), f"{span}_{camera}.mp4")
    inhale = truth["inhale"]
    rate, mid = 1 / np.diff(inhale), inhale[:-1]

    def spread(t0: float) -> float:
        inside = (mid >= t0) & (mid < t0 + MINUTE)
        return float(np.std(rate[inside])) if inside.sum() > 10 else 0.0

    out["start"] = max(np.arange(inhale[0], inhale[-1] - MINUTE, 1.0), key=spread)
    np.savez(work / "traces.npz", span=span, **out)
    (work / "boxes.json").write_text(json.dumps(boxes))
    print(f"minute from {out['start'] - truth['times'][0]:.1f} s; boxes {boxes}")


def render(work: Path, out: Path, still: float | None) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.collections import LineCollection

    z = np.load(work / "traces.npz")
    boxes = json.loads((work / "boxes.json").read_text())
    span = str(z["span"])
    t0 = float(z["start"])
    times = t0 + np.arange(int(MINUTE * FPS)) / FPS

    def lane(key):
        t, x = z[f"{key}_t"], z[key]
        keep = (t >= t0 - HALF) & (t <= t0 + MINUTE + HALF)
        x = x[keep]
        return t[keep] - t0, (x - x.mean()) / (x.std() or 1.0)

    frame_at = {}
    for key, (_, camera) in NETWORKS.items():
        ft = pd.read_parquet(work / f"{span}_{camera}.parquet")["Time"].to_numpy()
        frame_at[key] = np.clip(np.searchsorted(ft, times), 0, len(ft) - 1)

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "white"})
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    views, pips = {}, {}
    for i, key in enumerate(NETWORKS):
        left = (30 + i * 630) / W
        ax = fig.add_axes((left, 1 - 495 / H, 600 / W, 450 / H))
        ax.set_axis_off()
        views[key] = ax.imshow(
            np.zeros(VIDEO_SIZE[::-1]),
            cmap="gray",
            vmin=0,
            vmax=255,
            extent=(0, VIDEO_SIZE[0], VIDEO_SIZE[1], 0),
        )
        x, y, w, h = boxes[key]
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=COL[key], lw=2))
        ax.text(12, 30, NAME[key], color=COL[key], fontsize=17, weight="bold")
        pip_x = left + (5 if PIP_LEFT[key] else 400) / W
        pip = fig.add_axes((pip_x, 1 - 490 / H, 195 / W, 195 / H))
        pip.set_xticks([])
        pip.set_yticks([])
        for s in pip.spines.values():
            s.set_color(COL[key])
            s.set_linewidth(2.5)
        pips[key] = pip.imshow(np.zeros((96, 96)), cmap="gray", vmin=0, vmax=255)

    tr = fig.add_axes((215 / W, 60 / H, 1675 / W, 440 / H), facecolor=BG)
    step = 6.0
    for row, key in enumerate(["truth", *NETWORKS]):
        y0 = -step * row
        t, x = lane(key)
        tr.plot(t, x + y0, color=COL[key], lw=1.8)
        inh = z[f"{key}_inh"] - t0
        inh = inh[(inh > -HALF) & (inh < MINUTE + HALF)]
        tr.add_collection(
            LineCollection(
                [[(s, y0 - 2.6), (s, y0 + 2.6)] for s in inh],
                colors=COL[key],
                linewidths=1.1,
                linestyles=(0, (3, 3)),
                alpha=0.85,
            )
        )
        tr.text(
            -0.01,
            y0,
            NAME[key],
            transform=tr.get_yaxis_transform(),
            ha="right",
            va="center",
            color=COL[key],
            fontsize=15,
            weight="bold",
        )
    playhead = tr.axvline(0, color="white", lw=1.5, alpha=0.6)
    tr.set_ylim(-step * 3 - 3.2, 3.2)
    tr.set_yticks([])
    for s in ("left", "right", "top"):
        tr.spines[s].set_visible(False)
    tr.spines["bottom"].set_color("#888888")
    tr.tick_params(axis="x", colors="#aaaaaa", labelsize=12)
    tr.set_xlabel("Time (s)", color="#aaaaaa", fontsize=13)
    clock = fig.text(
        1 - 30 / W, 1 - 525 / H, "", ha="right", color="#aaaaaa", fontsize=14
    )
    fig.text(
        30 / W,
        1 - 528 / H,
        "Zephyr: respiration from video in a held-out mouse",
        color="white",
        fontsize=17,
        weight="bold",
    )
    # Key to the dashed lines, left of the clock.
    key = fig.add_axes((1440 / W, 1 - 532 / H, 40 / W, 20 / H), facecolor=BG)
    key.plot([0, 1], [0.5, 0.5], color="#cccccc", lw=1.6, ls=(0, (3, 3)))
    key.set_axis_off()
    fig.text(1490 / W, 1 - 528 / H, "inhalation onsets", color="#cccccc", fontsize=14)

    caps = {
        k: cv2.VideoCapture(str(work / f"{span}_{c}.mp4"))
        for k, (_, c) in NETWORKS.items()
    }
    position = {k: -1 for k in NETWORKS}

    def seek(key, index):
        caps[key].set(cv2.CAP_PROP_POS_FRAMES, index)
        position[key] = index - 1

    def frame(key, index):
        """Video frame *index* of camera *key*, read forward, in gray."""
        if index <= position[key] or index - position[key] > 2000:
            seek(key, index)
        while position[key] < index - 1:
            caps[key].grab()
            position[key] += 1
        ok, img = caps[key].read()
        position[key] += 1
        if not ok:
            return np.zeros(VIDEO_SIZE[::-1], np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def draw(i):
        now = times[i] - t0
        for key in NETWORKS:
            img = frame(key, frame_at[key][i])
            views[key].set_data(img)
            x, y, w, h = boxes[key]
            pips[key].set_data(cv2.resize(img[y : y + h, x : x + w], (96, 96)))
        tr.set_xlim(now - HALF, now + HALF)
        playhead.set_xdata([now, now])
        clock.set_text(f"{now:5.1f} s")
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())

    first = int(still * FPS) if still is not None else 0
    for key in NETWORKS:
        seek(key, frame_at[key][first])
    if still is not None:
        plt.imsave(out, draw(first))
        return
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{W}x{H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            # about 2.4 Mbit/s: a minute stays under 20 MB
            "-b:v",
            "2400k",
            "-maxrate",
            "3000k",
            "-bufsize",
            "6000k",
            "-preset",
            "slow",
            "-movflags",
            "+faststart",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    for i in range(len(times)):
        ff.stdin.write(draw(i).tobytes())
        if i % 600 == 0:
            print(f"{i} / {len(times)} frames", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("workdir", type=Path)
    p.add_argument("--span", default=SPAN)
    r = sub.add_parser("render")
    r.add_argument("workdir", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--still", type=float, help="write one frame (s into the minute)")
    args = parser.parse_args()
    if args.step == "prepare":
        prepare(args.workdir, args.span)
    else:
        render(args.workdir, args.out, args.still)


if __name__ == "__main__":
    main()
