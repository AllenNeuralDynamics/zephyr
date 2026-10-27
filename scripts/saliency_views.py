"""Where and when the side and front networks look, for Supplementary Fig. 6.

    uv run python scripts/saliency_views.py <clip-list folder> <workdir>

For each held-out unit of the face maps' fold (the notebook's saliency cache), the same
recording span from the side and front cameras: downloads its video, makes its crops
with zephyr's preprocessing from the camera's clip list (`clips-<camera>-720.toml`, as
the zephyr-compute-sandbox keeps them), and runs `analysis.clip_saliency` with that
camera's network of the fold. Writes `saliency-<camera>-<fold>.npz` beside the face
cache, which the notebook draws.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))

CAMERAS = {
    "side": ("SideCamera", "Zephyr, side camera"),
    "front": ("FrontCamera", "Zephyr, front camera"),
}
REMOTE = "/data/zephyr-dataset"
BINS = np.arange(-1.0, 1.0 + 1e-9, 1 / 60) - 1 / 120


def _span(clip) -> str:
    """A clip's recording span: its video's name without the camera."""
    return Path(clip.video).stem.rsplit("_", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lists", type=Path, help="folder of clips-<camera>-720.toml")
    parser.add_argument("workdir", type=Path, help="videos and feature caches go here")
    args = parser.parse_args()
    root = args.workdir.resolve()
    (root / "clips").mkdir(parents=True, exist_ok=True)
    # The lists name the dataset as the capsule mounts it; this folder stands for it.
    os.environ["ZEPHYR_PATH_MAP"] = f"{root.as_posix()}={REMOTE}"

    from utils import runs
    from zephyr.benchmarks import analysis
    from zephyr.config import ClipList, load
    from zephyr.features import lookup
    from zephyr.preprocess import preprocess_clips

    face = np.load(runs.PREDICTIONS / "saliency-face-cv.npz", allow_pickle=True)
    fold = "cv_0"
    keep = face["fold"] == fold
    spans = {
        str(s).rsplit("_", 1)[0]: str(u)
        for s, u in zip(face["stem"][keep], face["unit"][keep], strict=True)
    }
    s3 = runs._s3()
    bucket = "aind-scratch-data/vr-foraging/zephyr-dataset/clips"
    dev = runs.device()
    for camera, (suffix, label) in CAMERAS.items():
        # The list's own entries for these spans, unchanged (boxes and all), naming the
        # files here; loading a list needs every file it names.
        source = args.lists / f"clips-{camera}-720.toml"
        header, *entries = source.read_text().split("\n[[clip]]\n")
        wanted = [e for e in entries if any(f"/{s}_{suffix}." in e for s in spans)]
        present = [s for s in spans if any(f"/{s}_{suffix}." in e for e in wanted)]
        for span in present:
            for name in (
                f"{span}_{suffix}.mp4",
                f"{span}_{suffix}.parquet",
                f"{span}_thermistor.parquet",
            ):
                target = root / "clips" / name
                if target.is_file():
                    continue
                local = runs.root() / "dataset" / "clips" / name
                if local.is_file():
                    shutil.copyfile(local, target)
                else:
                    # under another name until complete, so a stopped run leaves none
                    part = target.with_name(target.name + ".part")
                    s3.get(f"{bucket}/{name}", str(part))
                    part.replace(target)
        local_list = root / source.name
        text = "\n[[clip]]\n".join([header, *wanted])
        local_list.write_text(text.replace(REMOTE, root.as_posix()))
        clip_list = load(ClipList, local_list)
        clips = clip_list.resolve()
        cache = root / "cache" / f"features-720x540-box96x96-60hz-{suffix}"
        preprocess_clips(clips, clip_list.preprocess, cache, workers=4)

        name = runs.cv_name(label, fold)
        model, mean, std, _, _ = runs.network(name)
        model = model.to(dev).float().eval()
        control = analysis.randomized(model)
        rows = {
            k: []
            for k in ("occlusion", "trace", "gradient", "random", "crop", "profile")
        }
        units, stems, whole, boxes = [], [], [], []
        for c in clips:
            entry = lookup(c, clip_list.preprocess, cache)
            truth = analysis.truth(entry)
            found = analysis.clip_saliency(
                model, control, entry, mean, std, truth["inhale"], BINS, device=dev
            )
            for k in ("occlusion", "trace", "gradient", "random", "profile"):
                rows[k].append(found[k])
            frames = np.load(entry.frames, mmap_mode="r")
            rows["crop"].append(
                np.asarray(frames[:: max(1, len(frames) // 50), 0]).mean(0)
            )
            # The whole first frame at the list's frame size, and the box on it.
            video = cv2.VideoCapture(str(c.video))
            _, image = video.read()
            video.release()
            size = tuple(clip_list.preprocess.target_size)
            image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), size)
            whole.append(image)
            boxes.append(c.box)
            units.append(spans[_span(c)])
            stems.append(Path(c.video).stem)
            print(camera, Path(c.video).stem, flush=True)
        out = {k: np.array(v) for k, v in rows.items()}
        out |= {
            "frame": np.array(whole),
            "box": np.array(boxes),
            "unit": np.array(units),
            "fold": np.array([fold] * len(units)),
            "stem": np.array(stems),
            "offset_s": (BINS[:-1] + BINS[1:]) / 2,
        }
        np.savez(runs.PREDICTIONS / f"saliency-{camera}-{fold}.npz", **out)
        print("wrote", camera, len(units), "units", flush=True)


if __name__ == "__main__":
    main()
