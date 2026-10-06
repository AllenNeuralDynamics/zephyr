"""Choose a downsample target and hand-place crop boxes for a clip list.

Boxes are placed on the frame downsampled to the list's ``[preprocess] target_size``.
One box covers a whole recording (the camera does not move within it); ``o`` gives a
single clip its own. Boxes are written back as ``box = [x, y, w, h]`` with tomlkit,
leaving the rest of the file untouched; an unplaced recording gets no box and
preprocessing refuses it.

Keys: click place | arrows nudge (shift x10) | [ ] resize | n/p recording | 1/2/3 clip |
o own box | c copy previous | r re-centre.

CLI
---
    zephyr annotate list.toml --prepare   # cache one frame per clip, once
    zephyr annotate list.toml --width 360 --height 270 --box-size 96
"""

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .clips import write_boxes
from .config import ClipList, ResolvedClip, load
from .video import Box, clamp_box, decode_window, probe_size

DEFAULT_BOX_SIZE = 128
PREPARE_START_S = 150.0


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


@dataclass(frozen=True)
class Recording:
    """The clips of one recording, in clip-list order: one camera placement."""

    key: str
    label: str
    clips: tuple[ResolvedClip, ...]


def recordings_of(clips: list[ResolvedClip]) -> list[Recording]:
    """Group *clips* by recording, keeping the list's order."""
    grouped: dict[str, list[ResolvedClip]] = {}
    for clip in clips:
        grouped.setdefault(clip.recording_id, []).append(clip)
    return [
        Recording(
            key, f"{members[0].video.parent.name}/{members[0].group}", tuple(members)
        )
        for key, members in grouped.items()
    ]


@dataclass
class BoxState:
    """The chosen geometry, plus where every recording's box currently sits.

    Boxes are stored in *target_size* coordinates -- the downsampled frame the
    annotator placed them on.  Holds no Tk or matplotlib objects, so each
    transition below is testable.
    """

    recordings: list[Recording]
    native_size: tuple[int, int]
    target_size: tuple[int, int]
    boxes: dict[str, Box] = field(default_factory=dict)
    overrides: dict[Path, Box] = field(default_factory=dict)
    placed: set[str] = field(default_factory=set)
    index: int = 0
    box_size: int = DEFAULT_BOX_SIZE
    part_index: int = 0

    def __post_init__(self) -> None:
        self.box_size = min(self.box_size, *self.target_size)
        centre = (self.target_size[0] / 2, self.target_size[1] / 2)
        for recording in self.recordings:
            seed = (
                box_centre(self.boxes[recording.key])
                if recording.key in self.boxes
                else centre
            )
            self.boxes[recording.key] = centre_box(
                *seed, self.box_size, self.target_size
            )
        for video, box in self.overrides.items():
            self.overrides[video] = centre_box(
                *box_centre(box), self.box_size, self.target_size
            )

    @property
    def recording(self) -> Recording:
        return self.recordings[self.index]

    @property
    def key(self) -> str:
        return self.recording.key

    @property
    def clip(self) -> ResolvedClip:
        return self.recording.clips[self.part_index]

    @property
    def has_override(self) -> bool:
        return self.clip.video in self.overrides

    @property
    def box(self) -> Box:
        """The box of the displayed clip: its own override, else its recording's."""
        return self.overrides.get(self.clip.video, self.boxes[self.key])

    @property
    def scale_from_native(self) -> float:
        """Native pixels per target pixel, for reporting."""
        return self.native_size[0] / self.target_size[0]

    def is_placed(self, key: str) -> bool:
        """Whether *key* was set by hand rather than left at its default."""
        return key in self.placed

    def select(self, key: str) -> None:
        for i, recording in enumerate(self.recordings):
            if recording.key == key:
                self.index = i
                self.part_index = 0

    def advance(self, step: int) -> None:
        self.index = int(np.clip(self.index + step, 0, len(self.recordings) - 1))
        self.part_index = 0

    def place(self, centre_x: float, centre_y: float) -> None:
        box = centre_box(centre_x, centre_y, self.box_size, self.target_size)
        if self.has_override:
            self.overrides[self.clip.video] = box
        else:
            self.boxes[self.key] = box
        self.placed.add(self.key)

    def nudge(self, dx: int, dy: int) -> None:
        cx, cy = box_centre(self.box)
        self.place(cx + dx, cy + dy)

    def resize(self, delta: int) -> None:
        """Change the box size for *every* clip -- one input shape for the model."""
        self.box_size = int(np.clip(self.box_size + delta, 16, min(self.target_size)))
        for table in (self.boxes, self.overrides):
            for key, box in table.items():
                table[key] = centre_box(
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
        for table in (self.boxes, self.overrides):
            for key, box in table.items():
                cx, cy = box_centre(box)
                table[key] = centre_box(
                    cx * ratio_x, cy * ratio_y, self.box_size, self.target_size
                )

    def reset(self) -> None:
        """Return this recording's box to the frame centre, unplaced."""
        self.overrides.pop(self.clip.video, None)
        self.boxes[self.key] = centre_box(
            self.target_size[0] / 2,
            self.target_size[1] / 2,
            self.box_size,
            self.target_size,
        )
        self.placed.discard(self.key)

    def copy_previous(self) -> None:
        if self.index > 0:
            self.place(*box_centre(self.boxes[self.recordings[self.index - 1].key]))

    def toggle_override(self) -> None:
        """Give the displayed clip a box of its own (from its recording's), or drop it."""
        video = self.clip.video
        if video in self.overrides:
            del self.overrides[video]
        else:
            self.overrides[video] = self.boxes[self.key]
            self.placed.add(self.key)

    def set_part(self, part_index: int) -> None:
        n_parts = len(self.recording.clips)
        self.part_index = int(np.clip(part_index, 0, max(0, n_parts - 1)))

    def progress(self) -> tuple[int, int]:
        return len(self.placed), len(self.recordings)

    def clip_boxes(self) -> dict[Path, Box]:
        """The box of every clip whose recording has been placed."""
        return {
            clip.video: self.overrides.get(clip.video, self.boxes[recording.key])
            for recording in self.recordings
            if recording.key in self.placed
            for clip in recording.clips
        }


def initial_state(
    clips: list[ResolvedClip],
    native_size: tuple[int, int],
    target_size: tuple[int, int],
    box_size: int,
) -> BoxState:
    """State from a clip list's existing boxes: a recording's first box is its
    own, and any clip whose box differs from that is an override."""
    recordings = recordings_of(clips)
    boxes: dict[str, Box] = {}
    overrides: dict[Path, Box] = {}
    for recording in recordings:
        have = [c for c in recording.clips if c.box is not None]
        if not have:
            continue
        boxes[recording.key] = have[0].box
        overrides |= {c.video: c.box for c in have if c.box != have[0].box}
    placed = set(boxes)
    sizes = {b[2] for b in [*boxes.values(), *overrides.values()]}
    # Trust the boxes already placed over the flag: those are what preprocessing
    # will crop to.
    box_size = sizes.pop() if len(sizes) == 1 else box_size
    return BoxState(
        recordings,
        native_size,
        target_size,
        boxes=boxes,
        overrides=overrides,
        placed=placed,
        box_size=box_size,
    )


def save_boxes(path: Path, state: BoxState) -> None:
    """Write the geometry and every placed box back into the clip list."""
    write_boxes(path, state.clip_boxes(), state.target_size)


# ---------------------------------------------------------------------------
# Frame cache
# ---------------------------------------------------------------------------


def frame_path(frame_dir: Path, video: Path) -> Path:
    """Cache file for *video*'s frame, keyed by its path."""
    digest = hashlib.sha256(str(video).encode()).hexdigest()[:10]
    return frame_dir / f"frame_{video.stem}-{digest}.npy"


def prepare_frames(clips: list[ResolvedClip], frame_dir: Path) -> None:
    """Cache one native-resolution frame per clip so the UI opens instantly.

    Stored at native resolution, not downsampled, so the downsample target stays
    a live choice in the UI rather than something baked into the cache.
    """
    frame_dir.mkdir(parents=True, exist_ok=True)
    print(f"caching frames for {len(clips)} clips -> {frame_dir}")

    for i, clip in enumerate(clips, start=1):
        path = frame_path(frame_dir, clip.video)
        if path.exists():
            print(f"  [{i}/{len(clips)}] {clip.name}: cached")
            continue
        size = probe_size(clip.video)
        frame = decode_window(
            clip.video,
            scale_to=size,
            start_s=PREPARE_START_S,
            dur_s=1.0 / 60.0,
        )[0]
        np.save(path, frame)
        print(f"  [{i}/{len(clips)}] {clip.name}: wrote {path.name}")


def load_recording_frames(recording: Recording, frame_dir: Path) -> list[np.ndarray]:
    """Native-resolution frames for a recording, one per clip, in clip order."""
    frames = []
    for clip in recording.clips:
        path = frame_path(frame_dir, clip.video)
        if path.exists():
            frames.append(np.load(path))
    return frames


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def run_ui(state: BoxState, frame_dir: Path, out_path: Path) -> None:
    """Recording list on the left, downsampled frame with the box on the right.

    Requires a display.  Everything else in this module runs headless.
    """
    import tkinter as tk
    from tkinter import ttk

    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib import patches
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    by_key = {r.key: r for r in state.recordings}
    native_cache: dict[str, list[np.ndarray]] = {}

    def frames_for(key: str) -> list[np.ndarray]:
        if key not in native_cache:
            native_cache[key] = load_recording_frames(by_key[key], frame_dir)
        return native_cache[key]

    root = tk.Tk()
    root.title("Crop ROI per recording")
    root.geometry("1500x900")

    # ---- left: recording tree ----------------------------------------------
    left = ttk.Frame(root, padding=(8, 8))
    left.pack(side=tk.LEFT, fill=tk.Y)
    ttk.Label(left, text="Recordings", font=("TkDefaultFont", 11, "bold")).pack(
        anchor="w"
    )

    tree = ttk.Treeview(
        left,
        columns=("clips", "roi"),
        show="tree headings",
        height=34,
        selectmode="browse",
    )
    for column, heading, width, anchor in (
        ("#0", "Recording", 190, "w"),
        ("clips", "Clips", 60, "center"),
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
    folders: dict[str, str] = {}
    for recording in state.recordings:
        folder = recording.clips[0].video.parent
        if str(folder) not in folders:
            folders[str(folder)] = tree.insert("", "end", text=folder.name, open=True)
        item_for[recording.key] = tree.insert(
            folders[str(folder)],
            "end",
            text=f"   {recording.label}",
            values=(len(recording.clips), ""),
        )

    def refresh_row(key: str) -> None:
        placed = state.is_placed(key)
        tree.item(
            item_for[key],
            values=(len(by_key[key].clips), "set" if placed else "auto"),
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
        recording = state.recording
        frames = frames_for(key)

        if frames:
            state.set_part(state.part_index)
            data = downsample(frames[state.part_index], state.target_size)
            suffix = "  (own box)" if state.has_override else ""
            part_var.set(f"{state.clip.video.stem}{suffix}")
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
        rect.set_linestyle("--" if state.has_override else "-")
        crosshair.set_data(*[[v] for v in box_centre(box)])

        axis.set_title(
            f"{recording.label}\n"
            f"frame {state.target_size[0]}x{state.target_size[1]} "
            f"({state.scale_from_native:.2f}x downsample)   "
            f"box {state.box_size}px = {state.box_size * state.scale_from_native:.0f} "
            f"native px   at {box[:2]}",
            fontsize=10,
        )
        canvas.draw_idle()

        done, total = state.progress()
        progress_var.set(f"{done} / {total} placed")
        info_var.set(f"clips: {', '.join(c.video.stem for c in recording.clips)}")
        width_var.set(str(state.target_size[0]))
        height_var.set(str(state.target_size[1]))
        size_var.set(str(state.box_size))

    def commit() -> None:
        refresh_row(state.key)
        save_boxes(out_path, state)
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
        save_boxes(out_path, state)
        redraw()

    def set_divisor(divisor: int) -> None:
        state.set_target(
            state.native_size[0] // divisor, state.native_size[1] // divisor
        )
        save_boxes(out_path, state)
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
        elif key in ("o", "O"):
            state.toggle_override()
            commit()
        elif key in ("1", "2", "3"):
            state.set_part(int(key) - 1)
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
        "n/p=recording   1/2/3=clip   o=own box   c=copy prev   r=re-centre",
        foreground="#666666",
    ).pack(side=tk.RIGHT)

    def on_close() -> None:
        save_boxes(out_path, state)
        print(f"saved {len(state.placed)} recording box(es) -> {out_path}")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    for recording in state.recordings:
        refresh_row(recording.key)
    select_key(state.recordings[state.index].key)
    root.mainloop()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Choose a downsample target and hand-place crop boxes."
    )
    parser.add_argument("clips", type=Path, help="Clip list TOML file to annotate.")
    parser.add_argument(
        "--width", type=int, help="Downsample target width (default: the list's)"
    )
    parser.add_argument(
        "--height", type=int, help="Downsample target height (default: the list's)"
    )
    parser.add_argument("--box-size", type=int, default=DEFAULT_BOX_SIZE)
    parser.add_argument("--frame-dir", type=Path, default=Path("artifacts/frames"))
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Cache one frame per clip, then exit.",
    )
    args = parser.parse_args(argv)

    clip_list = load(ClipList, args.clips)
    clips = clip_list.resolve()
    if args.prepare:
        prepare_frames(clips, args.frame_dir)
        return

    native = probe_size(clips[0].video)
    saved_target = clip_list.preprocess.target_size
    target = (args.width or saved_target[0], args.height or saved_target[1])
    # Start on the grid the boxes were saved on, then move to the requested
    # target, so existing placements are carried across rather than
    # reinterpreted as coordinates on a different-sized frame.
    state = initial_state(clips, native, saved_target, args.box_size)
    if args.box_size != DEFAULT_BOX_SIZE:
        state.resize(args.box_size - state.box_size)
    if state.target_size != target:
        state.set_target(*target)
    print(
        f"resuming: {len(state.placed)} of {len(state.recordings)} recordings placed"
        if state.placed
        else "no boxes placed yet"
    )

    missing = [
        c.name for c in clips if not frame_path(args.frame_dir, c.video).exists()
    ]
    if missing:
        print(
            f"WARNING: {len(missing)} clips have no cached frame and will show "
            "blank. Run with --prepare first."
        )

    print(
        f"{len(state.recordings)} recordings | native {native[0]}x{native[1]} | "
        f"target {target[0]}x{target[1]} | box {state.box_size} | -> {args.clips}"
    )
    run_ui(state, args.frame_dir, args.clips)
