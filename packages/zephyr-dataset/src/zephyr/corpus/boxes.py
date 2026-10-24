"""Crop boxes for one camera's clips: a template match first, then your corrections.

A script to run, not a build step; the work folder holds a local list per camera (e.g.
``face_camera_snout.toml``: that camera's clips of ``clips.toml``, whose files are not on
this machine) and one frame per clip, the session's thumbnail. ``publish`` writes the
boxes back into the dataset's ``clips.toml``.

    uv run python -m zephyr.corpus.boxes prepare <spec.toml> --work boxes
    uv run zephyr gui boxes/face_camera_snout.toml --page boxes   # frames: boxes/frames
    uv run python -m zephyr.corpus.boxes publish <spec.toml> --work boxes
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tomlkit

from zephyr.annotate import centre_box, frame_path
from zephyr.config import load

from . import rois, storage
from .config import DatasetSpec, Layout
from .lists import validate


def camera_entries(doc, camera: str) -> list:
    return [e for e in doc["clip"] if str(e["video"]).endswith(f"_{camera}.mp4")]


def work_list(work: Path, camera: str) -> Path:
    """The work folder's list of *camera*'s clips: ``SideCamera`` -> ``side_camera_snout.toml``."""
    return work / f"{re.sub(r'(?<!^)(?=[A-Z])', '_', camera).lower()}_snout.toml"


def prepare(spec, work: Path, camera: str, box_size: int, min_score: float) -> None:
    layout = Layout(spec.root)
    doc = tomlkit.parse(layout.clip_list.read_text(encoding="utf-8"))
    entries = camera_entries(doc, camera)
    target = tuple(doc["preprocess"]["target_size"])
    crops = rois.example_crops(layout, camera)
    matcher = rois.Template(crops)
    gate = rois.Gate((45, 110), (40, 110), (500, 610), (250, 350), min_score)
    frames = work / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    sessions = sorted({str(e["group"]) for e in entries})
    found, rows = {}, []
    for n, sid in enumerate(sessions, 1):
        image = rois.decode(layout.thumbnail(sid, camera).read_bytes())
        picked = rois.pick(*matcher([image])[0], gate)
        gray = np.asarray(image.convert("L"))
        scale = image.width / target[0]
        if picked is not None:
            box = picked[0]
            centre = ((box[0] + box[2]) / 2 / scale, (box[1] + box[3]) / 2 / scale)
            found[sid] = list(centre_box(*centre, box_size, target))
        rows.append({"session_id": sid, "score": picked[1] if picked else np.nan})
        for e in (e for e in entries if e["group"] == sid):
            np.save(frame_path(frames, (work / str(e["video"])).resolve()), gray)
        if n % 100 == 0:
            print(f"boxes: {n} of {len(sessions)} sessions")
    for e in entries:
        if "box" not in e and e["group"] in found:  # a box put there by hand stays
            e["box"] = found[e["group"]]
    out = tomlkit.document()
    for key in ("preprocess", "events"):
        if key in doc:
            out.add(key, doc[key])
    out.add("clip", tomlkit.aot())
    for e in entries:
        out["clip"].append(e)
    work_list(work, camera).write_text(tomlkit.dumps(out), encoding="utf-8")
    pd.DataFrame(rows).to_csv(work / "auto.csv", index=False)
    done = sum(
        1 for sid in sessions if any("box" in e for e in entries if e["group"] == sid)
    )
    print(
        f"boxes: {done} of {len(sessions)} sessions have a box; {len(sessions) - done} for the UI"
    )


def publish(spec, work: Path, camera: str) -> None:
    layout = Layout(spec.root)
    path = work_list(work, camera)
    face = tomlkit.parse(path.read_text(encoding="utf-8"))
    doc = tomlkit.parse(layout.clip_list.read_text(encoding="utf-8"))
    if list(doc["preprocess"]["target_size"]) != list(
        face["preprocess"]["target_size"]
    ):
        raise SystemExit(f"boxes: {path.name} changed the target size; set it back")
    boxes = {
        str(e["video"]): [int(v) for v in e["box"]] for e in face["clip"] if "box" in e
    }
    for e in doc["clip"]:
        if str(e["video"]) in boxes:
            e["box"] = boxes[str(e["video"])]
    validate(doc, layout.clip_list)
    storage.write_text(layout.clip_list, tomlkit.dumps(doc))
    entries = camera_entries(doc, camera)
    left = {str(e["group"]) for e in entries if "box" not in e}
    print(
        f"boxes: {len(boxes)} of {len(entries)} {camera} clips boxed; {len(left)} sessions without"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m zephyr.corpus.boxes", description=__doc__
    )
    p.add_argument("step", choices=["prepare", "publish"])
    p.add_argument("spec", type=Path, help="the dataset spec (only its root is used)")
    p.add_argument("--work", type=Path, default=Path("boxes"))
    p.add_argument("--camera", default="FaceCamera")
    p.add_argument(
        "--box-size", type=int, default=96, help="in the clip list's target frame"
    )
    p.add_argument("--min-score", type=float, default=0.58)
    p.add_argument("--profile", help="AWS profile for an s3:// root (sets AWS_PROFILE)")
    a = p.parse_args(argv)
    if a.profile:
        os.environ["AWS_PROFILE"] = a.profile
    spec = load(DatasetSpec, a.spec)
    a.work.mkdir(parents=True, exist_ok=True)
    if a.step == "prepare":
        prepare(spec, a.work, a.camera, a.box_size, a.min_score)
    else:
        publish(spec, a.work, a.camera)


if __name__ == "__main__":
    main()
