"""Choose a downsample target and hand-place crop boxes for a clip list (the logic).

Boxes are placed on the frame downsampled to the list's ``[preprocess] target_size``.
One box covers a whole recording (the camera does not move within it); ``o`` gives a
single clip its own. Boxes are written back as ``box = [x, y, w, h]`` with tomlkit,
leaving the rest of the file untouched; an unplaced recording gets no box and
preprocessing refuses it.

Keys (the window is :mod:`zephyr.gui.boxes`): click place | arrows nudge (shift x10) |
[ ] resize | n/p recording | 1/2/3 clip | o own box | c copy previous | r re-centre.

CLI
---
    zephyr annotate list.toml --prepare   # cache one frame per clip, once
    zephyr annotate list.toml --width 360 --height 270 --box-size 96
    zephyr annotate list.toml --remote --frame-dir frames   # clips kept elsewhere: frames only
    zephyr annotate events list.toml   # choose each clip's event detector: zephyr.annotate_events
"""

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .clips import write_boxes
from .config import ClipList, ResolvedClip, load, require_videos
from .video import Box, clamp_box, decode_window, probe_size

DEFAULT_BOX_SIZE = 128
PREPARE_START_S = 150.0
"""Where the cached frame is taken: this far in, or halfway through a shorter clip."""


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
    annotator placed them on.  Holds no Qt objects, so each
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
    """Write the geometry and every placed box back into the clip list; a recording
    re-centred (unplaced) loses its box."""
    unplaced = {
        clip.video
        for recording in state.recordings
        if recording.key not in state.placed
        for clip in recording.clips
    }
    write_boxes(path, state.clip_boxes(), state.target_size, clear=unplaced)


# ---------------------------------------------------------------------------
# Frame cache
# ---------------------------------------------------------------------------


def frame_path(frame_dir: Path, video: Path) -> Path:
    """Cache file for *video*'s frame, keyed by its path."""
    digest = hashlib.sha256(str(video).encode()).hexdigest()[:10]
    return frame_dir / f"frame_{video.stem}-{digest}.npy"


def frame_time(clip: ResolvedClip) -> float:
    """Seconds into *clip* of its cached frame."""
    times = pd.read_parquet(clip.timestamps, columns=["Time"])["Time"].to_numpy()
    return min(PREPARE_START_S, float(times[-1] - times[0]) / 2)


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
            start_s=frame_time(clip),
            dur_s=1.0 / 60.0,
        )[0]
        np.save(path, frame)
        print(f"  [{i}/{len(clips)}] {clip.name}: wrote {path.name}")


def frame_size(clips: list[ResolvedClip], frame_dir: Path) -> tuple[int, int]:
    """``(width, height)`` of the first cached frame, for clips that are not here."""
    for clip in clips:
        path = frame_path(frame_dir, clip.video)
        if path.exists():
            height, width = np.load(path).shape[:2]
            return width, height
    raise SystemExit(f"--remote: no cached frame in {frame_dir} for these clips")


def load_recording_frames(recording: Recording, frame_dir: Path) -> list[np.ndarray]:
    """Native-resolution frames for a recording, one per clip, in clip order."""
    frames = []
    for clip in recording.clips:
        path = frame_path(frame_dir, clip.video)
        if path.exists():
            frames.append(np.load(path))
    return frames


def open_state(
    path: Path,
    frame_dir: Path,
    *,
    width: int | None = None,
    height: int | None = None,
    box_size: int = DEFAULT_BOX_SIZE,
    remote: bool = False,
) -> BoxState:
    """The box state of the clip list at *path*, on the target ``width x height``
    (default: the list's), its existing boxes carried across."""
    clip_list = load(ClipList, path, check_files=not remote)
    clips = clip_list.resolve()
    try:
        require_videos(clips, "placing boxes")
    except ValueError as exc:
        raise SystemExit(f"{path}: {exc}") from None
    native = frame_size(clips, frame_dir) if remote else probe_size(clips[0].video)
    saved_target = clip_list.preprocess.target_size
    target = (width or saved_target[0], height or saved_target[1])
    # Start on the grid the boxes were saved on, then move to the requested
    # target, so existing placements are carried across rather than
    # reinterpreted as coordinates on a different-sized frame.
    state = initial_state(clips, native, saved_target, box_size)
    if box_size != DEFAULT_BOX_SIZE:
        state.resize(box_size - state.box_size)
    if state.target_size != target:
        state.set_target(*target)
    return state


def missing_frames(state: BoxState, frame_dir: Path) -> list[str]:
    """Clips with no cached frame (they show blank)."""
    return [
        clip.name
        for recording in state.recordings
        for clip in recording.clips
        if not frame_path(frame_dir, clip.video).exists()
    ]


def main(argv: list[str] | None = None) -> None:
    if argv and argv[0] == "events":
        from . import annotate_events

        annotate_events.main(argv[1:])
        return
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
    parser.add_argument(
        "--remote",
        action="store_true",
        help="The clips are not on this machine: do not check they exist and take "
        "the frame size from the cached frames in --frame-dir.",
    )
    args = parser.parse_args(argv)

    if args.prepare and args.remote:
        raise SystemExit("--prepare decodes the videos, which --remote has not got")
    if args.prepare:
        clips = load(ClipList, args.clips).resolve()
        try:
            require_videos(clips, "placing boxes")
        except ValueError as exc:
            raise SystemExit(f"{args.clips}: {exc}") from None
        prepare_frames(clips, args.frame_dir)
        return

    from .gui import need_ui

    need_ui()
    state = open_state(
        args.clips,
        args.frame_dir,
        width=args.width,
        height=args.height,
        box_size=args.box_size,
        remote=args.remote,
    )
    print(
        f"resuming: {len(state.placed)} of {len(state.recordings)} recordings placed"
        if state.placed
        else "no boxes placed yet"
    )
    missing = missing_frames(state, args.frame_dir)
    if missing:
        print(
            f"WARNING: {len(missing)} clips have no cached frame and will show "
            "blank. Run with --prepare first."
        )
    native, target = state.native_size, state.target_size
    print(
        f"{len(state.recordings)} recordings | native {native[0]}x{native[1]} | "
        f"target {target[0]}x{target[1]} | box {state.box_size} | -> {args.clips}"
    )
    from .gui.boxes import run_ui

    run_ui(state, args.frame_dir, args.clips)
