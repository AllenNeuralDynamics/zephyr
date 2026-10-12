"""A helper that proposes a snout ROI per thumbnail; not a build step, nothing in the spec.

    uv run python -m zephyr.corpus.rois <spec.toml> [--sample 30] [--preview demo.png]

A pretrained detector (OWLv2; needs the package's ``rois`` extra) is shown snouts cut
from our thumbnails (``EXAMPLES``, ``--example``) or asked for ``--prompt`` text. Only boxes
a snout can be by size and centre (``--width`` ... ``--center-y``, thumbnail pixels) are
kept, the best wins, and the ROI is a ``--size`` square around it moved by ``--offset``.
Writes ``rois.csv`` and a preview (box red, ROI yellow) to ``--out``, never into the dataset.
"""

import argparse
import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from zephyr.config import load

from . import storage
from .config import DatasetSpec, Layout

BATCH = 8  # thumbnails per text detector call
COLUMNS = [
    "session_id",
    "status",
    "score",
    "box_x0",
    "box_y0",
    "box_x1",
    "box_y1",
    "x0",
    "y0",
]
EXAMPLES = [  # (session, nose tip pad in thumbnail pixels): read off the thumbnails by eye
    ("806527_2026-02-17_16-11-39", (527, 258, 597, 318)),
    ("815103_2026-03-31_20-15-23", (530, 277, 598, 336)),
    ("828415_2026-02-03_00-20-24", (520, 275, 590, 343)),
    ("841301_2026-03-04_20-50-44", (507, 257, 575, 330)),
    ("841301_2026-03-27_20-02-29", (525, 259, 594, 330)),
    ("841302_2026-03-18_20-31-39", (514, 259, 585, 330)),
]


@dataclass(frozen=True)
class Gate:
    """What a snout box can be, in pixels of the thumbnail: ``(min, max)`` ranges."""

    width: tuple[float, float]
    height: tuple[float, float]
    center_x: tuple[float, float]
    center_y: tuple[float, float]
    min_score: float


def pick(boxes: np.ndarray, scores: np.ndarray, gate: Gate):
    """The best scoring ``(x0, y0, x1, y1)`` box that passes the gate, and its score; or
    None."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    ok = scores >= gate.min_score
    for value, (low, high) in [
        (w, gate.width),
        (h, gate.height),
        (cx, gate.center_x),
        (cy, gate.center_y),
    ]:
        ok &= (value >= low) & (value <= high)
    if not ok.any():
        return None
    best = np.flatnonzero(ok)[np.argmax(scores[ok])]
    return boxes[best], float(scores[best])


def square(box, size: int, offset, shape: tuple[int, int]) -> tuple[int, int]:
    """Top left of the *size* square around the box, moved by *offset* ``(x, y)``, kept
    inside an image of *shape* ``(width, height)``."""
    if min(shape) < size:
        raise ValueError(f"a {shape[0]}x{shape[1]} image is smaller than the ROI")
    cx = (box[0] + box[2]) / 2 + offset[0]
    cy = (box[1] + box[3]) / 2 + offset[1]
    return (
        int(np.clip(round(cx - size / 2), 0, shape[0] - size)),
        int(np.clip(round(cy - size / 2), 0, shape[1] - size)),
    )


class Detector:
    """A ``transformers`` OWLv2 detector, on the GPU when there is one: asked for
    *prompts*, or shown *examples* (images of the thing; a box scores its best one)."""

    def __init__(self, model: str, prompts=None, examples=None) -> None:
        import torch
        from transformers import AutoProcessor, Owlv2ForObjectDetection

        self.torch = torch
        self.prompts = prompts
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = (
            Owlv2ForObjectDetection.from_pretrained(model).eval().to(self.device)
        )
        self.queries = [
            self.processor(images=e, return_tensors="pt")["pixel_values"].to(
                self.device
            )
            for e in examples or []
        ]

    def __call__(self, images: list) -> list[tuple[np.ndarray, np.ndarray]]:
        """Every box the model proposes in each image (pixels) with its best score."""
        if self.queries:
            return [self.guided(image) for image in images]
        torch = self.torch
        inputs = self.processor(
            text=[self.prompts] * len(images), images=images, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            found = self.model(**inputs)
        sides = torch.tensor([[max(im.size)] * 2 for im in images])
        found = self.processor.post_process_grounded_object_detection(
            found, target_sizes=sides, threshold=0.0
        )
        return [(f["boxes"].cpu().numpy(), f["scores"].cpu().numpy()) for f in found]

    def guided(self, image) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"]
        pixels = pixels.to(self.device)
        scores = []
        for query in self.queries:
            with torch.no_grad():
                found = self.model.image_guided_detection(
                    pixel_values=pixels, query_pixel_values=query
                )
            scores.append(torch.sigmoid(found.logits.max(-1).values[0]))
        side = max(image.size)  # the processor pads to a square, bottom and right
        c = found.target_pred_boxes[0]
        boxes = torch.stack(
            [c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2]
            + [c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2],
            1,
        )
        best = torch.stack(scores).max(0).values
        return (boxes * side).cpu().numpy(), best.cpu().numpy()


class Template:
    """Normalised cross-correlation of the examples (resized to their mean size, a little
    blurred) against the image: a box of that size at every position, scored by the mean
    correlation over the examples."""

    def __init__(self, examples: list, blur: float = 2.0) -> None:
        import cv2

        self.cv2, self.blur = cv2, blur
        self.w = round(np.mean([e.width for e in examples]))
        self.h = round(np.mean([e.height for e in examples]))
        self.templates = [self.gray(e.resize((self.w, self.h))) for e in examples]

    def gray(self, image) -> np.ndarray:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        return self.cv2.GaussianBlur(gray, (0, 0), self.blur) if self.blur else gray

    def __call__(self, images: list) -> list[tuple[np.ndarray, np.ndarray]]:
        out = []
        for image in images:
            gray = self.gray(image)
            maps = [
                self.cv2.matchTemplate(gray, t, self.cv2.TM_CCOEFF_NORMED)
                for t in self.templates
            ]
            score = np.mean(maps, axis=0)
            y0, x0 = np.indices(score.shape)
            boxes = np.stack([x0, y0, x0 + self.w, y0 + self.h], -1).reshape(-1, 4)
            out.append((boxes.astype(float), score.reshape(-1)))
        return out


def example_crops(layout: Layout, camera: str, examples=None) -> list:
    """The labelled snouts, cut from their sessions' thumbnails."""
    return [
        decode(layout.thumbnail(s, camera).read_bytes()).crop(tuple(map(int, b)))
        for s, b in examples or EXAMPLES
    ]


def decode(data: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")


def preview(
    images: dict, table: pd.DataFrame, size: int, path: Path, columns=6
) -> None:
    """A grid of the thumbnails with the chosen box (red) and the ROI (yellow)."""
    from PIL import Image, ImageDraw

    tiles = []
    for r in table.itertuples():
        image = images[r.session_id].copy()
        draw = ImageDraw.Draw(image)
        if r.status == "ok":
            draw.rectangle(
                [r.box_x0, r.box_y0, r.box_x1, r.box_y1], outline="red", width=3
            )
            draw.rectangle(
                [r.x0, r.y0, r.x0 + size, r.y0 + size], outline="yellow", width=3
            )
            label = f"{r.session_id[:26]} {r.score:.2f}"
        else:
            label = f"{r.session_id[:26]} NO ROI"
        draw.rectangle([0, 0, 330, 22], fill="black")
        draw.text((4, 4), label, fill="white")
        tiles.append(image.resize((image.width // 2, image.height // 2)))
    w, h = max(t.width for t in tiles), max(t.height for t in tiles)
    grid = Image.new("RGB", (w * columns, h * -(-len(tiles) // columns)))
    for i, tile in enumerate(tiles):
        grid.paste(tile, ((i % columns) * w, (i // columns) * h))
    grid.save(path)


def propose(
    layout: Layout,
    detector,
    camera: str,
    gate: Gate,
    offset,
    size: int,
    sample: int | None = None,
    seed: int = 0,
    skip=(),
) -> tuple[pd.DataFrame, dict]:
    """One row per thumbnail of *camera* (a random *sample* of them, or all; never the
    *skip* sessions), and the decoded images by session."""
    suffix = f"_{camera}.png"
    sessions = sorted(
        n.removesuffix(suffix)
        for n in storage.names(layout.thumbnail_dir, f"*{suffix}")
        if n.removesuffix(suffix) not in skip
    )
    if not sessions:
        raise SystemExit(f"rois: no {camera} thumbnails in {layout.thumbnail_dir}")
    if sample is not None and sample < len(sessions):
        rng = np.random.default_rng(seed)
        sessions = sorted(rng.choice(sessions, sample, replace=False))
    rows, images = [], {}
    for start in range(0, len(sessions), BATCH):
        ids = sessions[start : start + BATCH]
        batch = [decode(layout.thumbnail(i, camera).read_bytes()) for i in ids]
        for sid, image, found in zip(ids, batch, detector(batch)):
            images[sid] = image
            row = dict.fromkeys(COLUMNS, np.nan) | {"session_id": sid, "status": "none"}
            best = pick(*found, gate)
            if best is not None:
                box, score = best
                x0, y0 = square(box, size, offset, image.size)
                row.update(status="ok", score=score, x0=x0, y0=y0)
                row.update(zip(COLUMNS[3:7], map(float, box)))
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS), images


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m zephyr.corpus.rois", description=__doc__
    )
    p.add_argument("spec", type=Path, help="the dataset spec (only its root is used)")
    p.add_argument("--prompt", action="append", help="ask for text instead of examples")
    p.add_argument(
        "--example",
        action="append",
        nargs=5,
        metavar=("SESSION", "X0", "Y0", "X1", "Y1"),
        help="a snout to show the detector; default: EXAMPLES",
    )
    p.add_argument(
        "--template", action="store_true", help="match the examples, no network"
    )
    p.add_argument("--camera", default="FaceCamera")
    for name, value in [
        ("width", (45, 110)),
        ("height", (40, 110)),
        ("center-x", (500, 610)),
        ("center-y", (250, 350)),
    ]:
        p.add_argument(
            f"--{name}", type=float, nargs=2, default=value, metavar=("MIN", "MAX")
        )
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--offset", type=float, nargs=2, default=(0, 0), metavar=("X", "Y"))
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    p.add_argument("--sample", type=int, help="only this many random thumbnails")
    p.add_argument("--seed", type=int, default=0, help="which ones --sample takes")
    p.add_argument("--out", type=Path, default=Path("rois"))
    p.add_argument("--preview", type=Path, help="default: <out>/rois_preview.png")
    p.add_argument("--profile", help="AWS profile for an s3:// root (sets AWS_PROFILE)")
    a = p.parse_args(argv)
    if a.profile:
        os.environ["AWS_PROFILE"] = a.profile
    layout = Layout(load(DatasetSpec, a.spec).root)
    if a.prompt and not a.template:
        examples = []
    elif a.example:
        examples = [(s, tuple(map(float, b))) for s, *b in a.example]
    else:
        examples = EXAMPLES
    crops = example_crops(layout, a.camera, examples)
    gate = Gate(a.width, a.height, a.center_x, a.center_y, a.min_score)
    table, images = propose(
        layout,
        Template(crops) if a.template else Detector(a.model, a.prompt, crops),
        a.camera,
        gate,
        a.offset,
        a.size,
        a.sample,
        a.seed,
        skip={s for s, _ in examples},
    )
    a.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out / "rois.csv", index=False)
    preview(images, table, a.size, a.preview or a.out / "rois_preview.png")
    print(
        f"{int((table.status == 'ok').sum())} of {len(table)} have a ROI; see {a.out}"
    )


if __name__ == "__main__":
    main()
