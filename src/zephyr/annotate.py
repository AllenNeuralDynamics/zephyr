"""Choose a downsample target and hand-place one crop box per session.

**Downsample target** — the working frame size, in pixels. The video is scaled
to this before anything else, so it sets the resolution the model sees.

**Crop box** — placed *on the downsampled frame*, in downsampled pixels. Its
size is the model's input shape, so it is global across sessions; its position
is per session.

There is no automatic box placement: it is done by hand, once per session, and
verified across a session's part frames with the ``1`` / ``2`` keys below.

Usage
-----
    # once: cache one native-resolution frame per clip
    python -m breathing_cnn_tcn.annotate --prepare

    # then choose the geometry and place the boxes
    python -m breathing_cnn_tcn.annotate --width 360 --height 270 --box-size 96

Boxes auto-save to ``--out`` on every edit, so the window can be closed at any
point and reopened to resume.

Keys: click places the box centre | arrows nudge 1 px, shift+arrows 10 px |
``[`` / ``]`` resize the box for every session | ``n`` / ``p`` next / previous
session | ``1`` / ``2`` switch part frame | ``c`` copy the previous session's box
| ``r`` re-centre on the frame.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .clips import (
    PRIVATE_SPLIT,
    PUBLIC_SPLIT,
    SessionRef,
    discover_clips,
    discover_sessions,
)
from .video import Box, clamp_box, decode_window, probe_size

DEFAULT_BOX_SIZE = 128
PREPARE_START_S = 150.0

SPLIT_TITLES = {
    PUBLIC_SPLIT: "Public  (train)  -  thermistor available",
    PRIVATE_SPLIT: "Private (test)   -  video only",
}


# ---------------------------------------------------------------------------
# Geometry -- free of any UI state, so it can be tested directly
# ---------------------------------------------------------------------------


def centre_box(
    centre_x: float, centre_y: float, size: int, frame_size: tuple[int, int]
) -> Box:
    """Square box of *size* centred on the given point, shifted to stay in frame."""
    return clamp_box(
        (round(centre_x - size / 2), round(centre_y - size / 2), size, size),
        *frame_size,
    )


def box_centre(box: Box) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def downsample(frame: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Area-average *frame* down to ``(width, height)``.

    ``INTER_AREA`` is the same box-average that ffmpeg's ``scale=...:flags=area``
    applies during preprocessing, so a box placed on this image lands on the same
    pixels there.
    """
    width, height = target
    if (frame.shape[1], frame.shape[0]) == (width, height):
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


@dataclass
class SessionBoxState:
    """The chosen geometry, plus where every session's box currently sits.

    Boxes are stored in *target_size* coordinates -- the downsampled frame the
    annotator placed them on.  Holds no Tk or matplotlib objects, so each
    transition below is testable.
    """

    session_keys: list[str]
    native_size: tuple[int, int]
    target_size: tuple[int, int]
    boxes: dict[str, Box] = field(default_factory=dict)
    placed: set[str] = field(default_factory=set)
    index: int = 0
    box_size: int = DEFAULT_BOX_SIZE
    part_index: int = 0

    def __post_init__(self) -> None:
        self.box_size = min(self.box_size, *self.target_size)
        centre = (self.target_size[0] / 2, self.target_size[1] / 2)
        for key in self.session_keys:
            seed = box_centre(self.boxes[key]) if key in self.boxes else centre
            self.boxes[key] = centre_box(*seed, self.box_size, self.target_size)

    @property
    def key(self) -> str:
        return self.session_keys[self.index]

    @property
    def box(self) -> Box:
        return self.boxes[self.key]

    @property
    def scale_from_native(self) -> float:
        """Native pixels per target pixel, for reporting."""
        return self.native_size[0] / self.target_size[0]

    def is_placed(self, key: str) -> bool:
        """Whether *key* was set by hand rather than left at its default."""
        return key in self.placed

    def select(self, key: str) -> None:
        if key in self.session_keys:
            self.index = self.session_keys.index(key)
            self.part_index = 0

    def advance(self, step: int) -> None:
        self.index = int(np.clip(self.index + step, 0, len(self.session_keys) - 1))
        self.part_index = 0

    def place(self, centre_x: float, centre_y: float) -> None:
        self.boxes[self.key] = centre_box(
            centre_x, centre_y, self.box_size, self.target_size
        )
        self.placed.add(self.key)

    def nudge(self, dx: int, dy: int) -> None:
        cx, cy = box_centre(self.box)
        self.place(cx + dx, cy + dy)

    def resize(self, delta: int) -> None:
        """Change the box size for *every* session -- one input shape for the model."""
        self.box_size = int(np.clip(self.box_size + delta, 16, min(self.target_size)))
        for key, box in self.boxes.items():
            self.boxes[key] = centre_box(
                *box_centre(box), self.box_size, self.target_size
            )

    def set_target(self, width: int, height: int) -> None:
        """Change the downsample target, carrying every box across proportionally.

        Box *size* is deliberately left alone: it is the model's input shape, so
        the annotator picks it directly.  Changing the target therefore changes
        how much of the frame that shape covers, which is the actual trade-off.
        """
        width = int(np.clip(width, 16, self.native_size[0]))
        height = int(np.clip(height, 16, self.native_size[1]))
        ratio_x = width / self.target_size[0]
        ratio_y = height / self.target_size[1]
        self.target_size = (width, height)
        self.box_size = min(self.box_size, width, height)
        for key, box in self.boxes.items():
            cx, cy = box_centre(box)
            self.boxes[key] = centre_box(
                cx * ratio_x, cy * ratio_y, self.box_size, self.target_size
            )

    def reset(self) -> None:
        """Return this session's box to the frame centre, unplaced."""
        self.place(self.target_size[0] / 2, self.target_size[1] / 2)
        self.placed.discard(self.key)

    def copy_previous(self) -> None:
        if self.index > 0:
            self.place(*box_centre(self.boxes[self.session_keys[self.index - 1]]))

    def set_part(self, part_index: int, n_parts: int) -> None:
        self.part_index = int(np.clip(part_index, 0, max(0, n_parts - 1)))

    def progress(self) -> tuple[int, int]:
        return len(self.placed), len(self.session_keys)


# ---------------------------------------------------------------------------
# Frame cache
# ---------------------------------------------------------------------------


def prepare_frames(packaged_root: Path, split: str, camera: str, out_dir: Path) -> None:
    """Cache one native-resolution frame per clip so the UI opens instantly.

    Stored at native resolution, not downsampled, so the downsample target stays
    a live choice in the UI rather than something baked into the cache.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = [
        c
        for c in discover_clips(packaged_root, split, camera=camera)
        if c.exists(camera)
    ]
    print(f"caching frames for {len(clips)} {split}/{camera} clips -> {out_dir}")

    for i, clip in enumerate(clips, start=1):
        path = out_dir / f"frame_{camera}_{clip.clip_id}.npy"
        if path.exists():
            print(f"  [{i}/{len(clips)}] {clip.clip_id}: cached")
            continue
        size = probe_size(clip.video(camera))
        frame = decode_window(
            clip.video(camera),
            scale_to=size,
            start_s=PREPARE_START_S,
            dur_s=1.0 / 60.0,
        )[0]
        np.save(path, frame)
        print(f"  [{i}/{len(clips)}] {clip.clip_id}: wrote {path.name}")


def load_session_frames(
    session: SessionRef, camera: str, frame_dir: Path
) -> list[np.ndarray]:
    """Native-resolution frames for a session, one per part, in part order."""
    frames = []
    for clip in session.clips:
        path = frame_dir / f"frame_{camera}_{clip.clip_id}.npy"
        if path.exists():
            frames.append(np.load(path))
    return frames


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_boxes(
    path: Path, state: SessionBoxState, sessions: list[SessionRef], camera: str
) -> None:
    """Write the geometry and every session's box, expanded to its clips."""
    by_key = {s.key: s for s in sessions}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "camera": camera,
                "native_size": list(state.native_size),
                "target_size": list(state.target_size),
                "box_size": state.box_size,
                "n_placed": len(state.placed),
                "sessions": {
                    key: {
                        "box": list(state.boxes[key]),
                        "placed": key in state.placed,
                        "split": by_key[key].split,
                        "session_idx": by_key[key].session_idx,
                        "clip_ids": by_key[key].clip_ids,
                    }
                    for key in state.session_keys
                    if key in by_key
                },
            },
            indent=2,
        )
    )


@dataclass
class SavedGeometry:
    """What a boxes file records, for consumers that must reproduce the crop."""

    target_size: tuple[int, int]
    box_size: int
    boxes: dict[str, Box]
    placed: set[str]


def _target_size(data: dict) -> tuple[int, int]:
    """The coordinate space a boxes file's boxes are expressed in.

    Files written before downsampling existed have no ``target_size``: back then
    boxes were placed on the full-resolution frame, so that file's ``frame_size``
    *is* the coordinate space and carries over unchanged.
    """
    if "target_size" in data:
        return tuple(data["target_size"])
    if "frame_size" in data:
        return tuple(data["frame_size"])
    raise KeyError(
        "boxes file records neither 'target_size' nor 'frame_size', so there is "
        "no way to know what coordinate space its boxes are in"
    )


def load_boxes(path: Path) -> SavedGeometry | None:
    """Read back a saved boxes file, or ``None`` if it does not exist yet."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    sessions = data.get("sessions", {})
    boxes = {k: tuple(v["box"]) for k, v in sessions.items()}
    # An older file's box_size may disagree with the boxes it holds; trust the
    # boxes, since those are what was actually placed.
    sizes = {b[2] for b in boxes.values()}
    box_size = sizes.pop() if len(sizes) == 1 else int(data.get("box_size", 0))
    return SavedGeometry(
        target_size=_target_size(data),
        box_size=box_size or DEFAULT_BOX_SIZE,
        boxes=boxes,
        placed={k for k, v in sessions.items() if v.get("placed")},
    )


def clip_boxes(path: Path) -> tuple[tuple[int, int], dict[str, Box]]:
    """Flatten a boxes file into ``(target_size, {clip_id: box})``.

    Used by :mod:`.preprocess`, which works clip by clip and needs the target
    size to reproduce the same downsample before cropping.
    """
    data = json.loads(path.read_text())
    return _target_size(data), {
        clip_id: tuple(entry["box"])
        for entry in data.get("sessions", {}).values()
        for clip_id in entry.get("clip_ids", [])
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def run_ui(
    sessions: list[SessionRef],
    state: SessionBoxState,
    camera: str,
    frame_dir: Path,
    out_path: Path,
) -> None:
    """Session list on the left, downsampled frame with the box on the right.

    Requires a display.  Everything else in this module runs headless.
    """
    import tkinter as tk
    from tkinter import ttk

    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.patches as patches
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    by_key = {s.key: s for s in sessions}
    native_cache: dict[str, list[np.ndarray]] = {}

    def frames_for(key: str) -> list[np.ndarray]:
        if key not in native_cache:
            native_cache[key] = load_session_frames(by_key[key], camera, frame_dir)
        return native_cache[key]

    root = tk.Tk()
    root.title(f"Crop ROI per session - {camera} camera")
    root.geometry("1500x900")

    # ---- left: session tree ------------------------------------------------
    left = ttk.Frame(root, padding=(8, 8))
    left.pack(side=tk.LEFT, fill=tk.Y)
    ttk.Label(left, text="Sessions", font=("TkDefaultFont", 11, "bold")).pack(
        anchor="w"
    )

    tree = ttk.Treeview(
        left,
        columns=("session", "roi"),
        show="tree headings",
        height=34,
        selectmode="browse",
    )
    for column, heading, width, anchor in (
        ("#0", "Session", 170, "w"),
        ("session", "Group", 80, "center"),
        ("roi", "ROI", 70, "center"),
    ):
        tree.heading(column, text=heading)
        tree.column(column, width=width, anchor=anchor, stretch=False)
    tree.pack(side=tk.LEFT, fill=tk.Y)

    scroll = ttk.Scrollbar(left, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    scroll.pack(side=tk.LEFT, fill=tk.Y)
    tree.tag_configure("placed", foreground="#0a7a28")
    tree.tag_configure("auto", foreground="#999999")

    item_for: dict[str, str] = {}
    for split in (PUBLIC_SPLIT, PRIVATE_SPLIT):
        members = [s for s in sessions if s.split == split]
        if not members:
            continue
        parent = tree.insert(
            "", "end", text=f"{SPLIT_TITLES[split]}   [{len(members)}]", open=True
        )
        for session in members:
            item_for[session.key] = tree.insert(
                parent,
                "end",
                text=f"   {session.label}",
                values=(session.session_idx, ""),
            )

    def refresh_row(key: str) -> None:
        placed = state.is_placed(key)
        tree.item(
            item_for[key],
            values=(by_key[key].session_idx, "set" if placed else "auto"),
            tags=("placed" if placed else "auto",),
        )

    # ---- right: controls + canvas -----------------------------------------
    right = ttk.Frame(root, padding=(8, 8))
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    controls = ttk.Frame(right)
    controls.pack(side=tk.TOP, fill=tk.X)

    width_var = tk.StringVar(value=str(state.target_size[0]))
    height_var = tk.StringVar(value=str(state.target_size[1]))
    size_var = tk.StringVar(value=str(state.box_size))
    part_var = tk.StringVar(value="")
    info_var = tk.StringVar(value="")
    progress_var = tk.StringVar(value="")

    figure = Figure(figsize=(9, 7), layout="constrained")
    axis = figure.add_subplot(111)
    axis.axis("off")
    canvas = FigureCanvasTkAgg(figure, master=right)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    rect = patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor="cyan", lw=2)
    axis.add_patch(rect)
    (crosshair,) = axis.plot([], [], "c+", ms=16, mew=2)
    image = None

    def redraw() -> None:
        nonlocal image
        key = state.key
        session = by_key[key]
        frames = frames_for(key)

        if frames:
            state.set_part(state.part_index, len(frames))
            data = downsample(frames[state.part_index], state.target_size)
            part_var.set(f"part {session.clips[state.part_index].part}")
        else:
            data = np.zeros(state.target_size[::-1], dtype=np.uint8)
            part_var.set("no cached frame")

        extent = (0, state.target_size[0], state.target_size[1], 0)
        if image is None:
            image = axis.imshow(data, cmap="gray", extent=extent, aspect="equal")
        else:
            image.set_data(data)
            image.set_extent(extent)
            image.set_clim(0, 255)

        box = state.box
        rect.set_bounds(*box)
        rect.set_edgecolor("cyan" if state.is_placed(key) else "yellow")
        crosshair.set_data(*[[v] for v in box_centre(box)])

        axis.set_title(
            f"{key}   {session.label}\n"
            f"frame {state.target_size[0]}x{state.target_size[1]} "
            f"({state.scale_from_native:.2f}x downsample)   "
            f"box {state.box_size}px = {state.box_size * state.scale_from_native:.0f} "
            f"native px   at {box[:2]}",
            fontsize=10,
        )
        canvas.draw_idle()

        done, total = state.progress()
        progress_var.set(f"{done} / {total} placed")
        info_var.set(f"clips: {', '.join(session.clip_ids)}")
        width_var.set(str(state.target_size[0]))
        height_var.set(str(state.target_size[1]))
        size_var.set(str(state.box_size))

    def commit() -> None:
        refresh_row(state.key)
        save_boxes(out_path, state, sessions, camera)
        redraw()

    def select_key(key: str) -> None:
        state.select(key)
        tree.selection_set(item_for[key])
        tree.see(item_for[key])
        redraw()

    def apply_target() -> None:
        try:
            width, height = int(width_var.get()), int(height_var.get())
        except ValueError:
            redraw()
            return
        state.set_target(width, height)
        save_boxes(out_path, state, sessions, camera)
        redraw()

    def set_divisor(divisor: int) -> None:
        state.set_target(
            state.native_size[0] // divisor, state.native_size[1] // divisor
        )
        save_boxes(out_path, state, sessions, camera)
        redraw()

    # ---- events ------------------------------------------------------------

    def on_tree_select(_event=None) -> None:
        selection = tree.selection()
        if not selection:
            return
        for key, item in item_for.items():
            if item == selection[0]:
                state.select(key)
                redraw()
                return

    def on_click(event) -> None:
        if event.inaxes is axis and event.xdata is not None:
            state.place(event.xdata, event.ydata)
            commit()

    def on_key(event) -> str | None:
        key = event.keysym
        step = 10 if event.state & 0x0001 else 1
        moves = {
            "Left": (-step, 0),
            "Right": (step, 0),
            "Up": (0, -step),
            "Down": (0, step),
        }
        if key in moves:
            state.nudge(*moves[key])
            commit()
        elif key in ("n", "N"):
            state.advance(1)
            select_key(state.key)
        elif key in ("p", "P"):
            state.advance(-1)
            select_key(state.key)
        elif key in ("c", "C"):
            state.copy_previous()
            commit()
        elif key in ("r", "R"):
            state.reset()
            commit()
        elif key in ("1", "2", "3"):
            state.set_part(int(key) - 1, len(frames_for(state.key)))
            redraw()
        elif key == "bracketright":
            state.resize(8)
            commit()
        elif key == "bracketleft":
            state.resize(-8)
            commit()
        else:
            return None
        return "break"

    tree.bind("<<TreeviewSelect>>", on_tree_select)
    canvas.mpl_connect("button_press_event", on_click)
    root.bind("<Key>", on_key)

    # ---- control widgets ---------------------------------------------------
    ttk.Label(controls, text="Downsample to:").pack(side=tk.LEFT)
    ttk.Entry(controls, textvariable=width_var, width=6).pack(side=tk.LEFT, padx=2)
    ttk.Label(controls, text="x").pack(side=tk.LEFT)
    ttk.Entry(controls, textvariable=height_var, width=6).pack(side=tk.LEFT, padx=2)
    ttk.Button(controls, text="Apply", command=apply_target).pack(side=tk.LEFT, padx=4)
    for label, divisor in (("native", 1), ("1/2", 2), ("1/3", 3), ("1/4", 4)):
        ttk.Button(
            controls,
            text=label,
            width=6,
            command=lambda d=divisor: set_divisor(d),
        ).pack(side=tk.LEFT, padx=1)

    ttk.Separator(controls, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)
    ttk.Label(controls, text="Box:").pack(side=tk.LEFT)
    ttk.Button(
        controls, text="-", width=3, command=lambda: (state.resize(-8), commit())
    ).pack(side=tk.LEFT)
    ttk.Label(controls, textvariable=size_var, width=5, anchor="center").pack(
        side=tk.LEFT
    )
    ttk.Button(
        controls, text="+", width=3, command=lambda: (state.resize(8), commit())
    ).pack(side=tk.LEFT)

    ttk.Separator(controls, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)
    ttk.Label(controls, textvariable=part_var).pack(side=tk.LEFT)
    ttk.Separator(controls, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)
    ttk.Label(controls, textvariable=progress_var).pack(side=tk.LEFT)

    status = ttk.Frame(right, padding=(0, 6))
    status.pack(side=tk.BOTTOM, fill=tk.X)
    ttk.Label(status, textvariable=info_var, justify="left").pack(side=tk.LEFT)
    ttk.Label(
        status,
        text="click=place   arrows=nudge (shift x10)   [ ]=box size   "
        "n/p=session   1/2=part   c=copy prev   r=re-centre",
        foreground="#666666",
    ).pack(side=tk.RIGHT)

    def on_close() -> None:
        save_boxes(out_path, state, sessions, camera)
        print(f"saved {len(state.boxes)} session boxes -> {out_path}")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    for key in state.session_keys:
        refresh_row(key)
    select_key(state.session_keys[state.index])
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Choose a downsample target and hand-place a crop box per session."
    )
    parser.add_argument("--packaged-root", type=Path, default=Path("data"))
    parser.add_argument("--camera", default="face", choices=["face", "side"])
    parser.add_argument(
        "--width", type=int, help="Downsample target width (default: native)"
    )
    parser.add_argument(
        "--height", type=int, help="Downsample target height (default: native)"
    )
    parser.add_argument("--box-size", type=int, default=DEFAULT_BOX_SIZE)
    parser.add_argument(
        "--splits",
        nargs="*",
        default=[PUBLIC_SPLIT, PRIVATE_SPLIT],
        help="Splits to list, in order.",
    )
    parser.add_argument("--frame-dir", type=Path, default=Path("artifacts/frames"))
    parser.add_argument("--out", type=Path, help="Session boxes JSON")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Cache one frame per clip for every requested split, then exit.",
    )
    args = parser.parse_args()

    if args.prepare:
        for split in args.splits:
            prepare_frames(args.packaged_root, split, args.camera, args.frame_dir)
        return

    out_path = args.out or Path(
        f"baseline-cnn-tcn/artifacts/session_boxes_{args.camera}.json"
    )
    sessions = discover_sessions(
        args.packaged_root,
        args.splits,
        camera=args.camera,
    )
    if not sessions:
        raise SystemExit(f"No {args.camera} sessions under {args.packaged_root}")

    native = probe_size(sessions[0].clips[0].video(args.camera))
    saved = load_boxes(out_path)
    if saved:
        print(
            f"resuming from {out_path}: {len(saved.placed)} of {len(saved.boxes)} "
            f"placed, target {saved.target_size[0]}x{saved.target_size[1]}, "
            f"box {saved.box_size}"
        )

    # Explicit flags win over the saved geometry, so a run can change the target.
    if args.width or args.height:
        target = (args.width or native[0], args.height or native[1])
    elif saved:
        target = saved.target_size
    else:
        target = native

    explicit_box_size = args.box_size != DEFAULT_BOX_SIZE
    box_size = args.box_size if explicit_box_size or saved is None else saved.box_size

    keys = {s.key for s in sessions}
    state = SessionBoxState(
        session_keys=[s.key for s in sessions],
        native_size=native,
        # Start on whichever grid the boxes were saved on, then move to the
        # requested target below, so existing placements are carried across
        # rather than reinterpreted as coordinates on a different-sized frame.
        target_size=saved.target_size if saved else target,
        boxes={k: v for k, v in (saved.boxes if saved else {}).items() if k in keys},
        placed={k for k in (saved.placed if saved else set()) if k in keys},
        box_size=box_size,
    )
    if state.target_size != target:
        state.set_target(*target)
        if explicit_box_size:
            # set_target only shrinks box_size to fit; an explicit request wins.
            state.resize(box_size - state.box_size)

    missing = [
        c.clip_id
        for s in sessions
        for c in s.clips
        if not (args.frame_dir / f"frame_{args.camera}_{c.clip_id}.npy").exists()
    ]
    if missing:
        print(
            f"WARNING: {len(missing)} clips have no cached frame and will show "
            "blank. Run with --prepare first."
        )

    print(
        f"{len(sessions)} sessions | native {native[0]}x{native[1]} | "
        f"target {target[0]}x{target[1]} | box {state.box_size} | -> {out_path}"
    )
    run_ui(sessions, state, args.camera, args.frame_dir, out_path)


if __name__ == "__main__":
    main()
