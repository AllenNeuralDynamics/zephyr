"""Choose a downsample target and hand-place crop boxes for a clip list (the logic).

Boxes are placed on the frame downsampled to the list's ``[preprocess] target_size``
(or a clip's own ``frame``). One box covers a whole recording (the camera does not move
within it); ``o`` or a frame of its own gives a single clip its own. Boxes are written back as ``box = [x, y, w, h]`` with tomlkit,
leaving the rest of the file untouched; an unplaced recording gets no box and
preprocessing refuses it.

Boxes are placed by hand in the app (``zephyr gui <list> --page boxes``,
:mod:`zephyr.gui.boxes`); :class:`BoxState` is that page's logic.

CLI
---
    zephyr annotate list.toml --prepare --frame-dir frames   # one frame per clip, once
    zephyr annotate list.toml --auto --frame-dir frames [--min-score 0.5]   # match, save
    zephyr annotate predict.toml --auto --frame-dir frames --checkpoint best.pt
    zephyr annotate events list.toml --auto   # the default detector: zephyr.annotate_events
"""

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .clips import scale_box, scaled_frame, write_boxes
from .config import ClipList, ResolvedClip, load, require_videos
from .video import Box, clamp_box, decode_window, probe_duration, probe_size

DEFAULT_BOX_SIZE = 128
PREPARE_START_S = 150.0
"""Where the cached frame is taken: this far in, or halfway through a shorter clip."""
AUTO_MIN_SCORE = 0.5
"""The correlation a match needs for Auto to place a box (else it stays unplaced)."""
AUTO_BLUR = 2.0
"""Gaussian sigma (target pixels) of the frames and templates Auto matches."""


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
    matched: dict[str, float] = field(default_factory=dict)
    """Recordings :meth:`auto` placed, with their match score, until moved by hand."""
    excluded: set[str] = field(default_factory=set)
    """Recordings never to use (``selection = "exclude"`` on their clips): no box."""
    restored: set[str] = field(default_factory=set)
    """Recordings excluded here and then used again: their clips lose ``selection``."""
    locked: bool = False
    """Box and frame size are a network's (``--checkpoint``): only the place moves."""
    frames: dict[Path, tuple[int, int]] = field(default_factory=dict)
    """Clips on a frame of their own (not *target_size*); each has its own box."""

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
        for recording in self.recordings:
            for clip in recording.clips:
                if clip.video in self.frames and clip.video not in self.overrides:
                    self.overrides[clip.video] = scale_box(
                        self.boxes[recording.key],
                        self.target_size,
                        self.frames[clip.video],
                    )
        for video, box in self.overrides.items():
            self.overrides[video] = centre_box(
                *box_centre(box), self.box_size, self.frame_of(video)
            )

    def frame_of(self, video: Path) -> tuple[int, int]:
        """The frame *video*'s box is placed on: its own, else the list's."""
        return self.frames.get(video, self.target_size)

    @property
    def frame(self) -> tuple[int, int]:
        """The displayed clip's frame."""
        return self.frame_of(self.clip.video)

    @property
    def input_scale(self) -> float:
        """The displayed clip's frame over the list's (1: the list's own)."""
        return self.frame[0] / self.target_size[0]

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
        """Native pixels per frame pixel of the displayed clip, for reporting."""
        return self.native_size[0] / self.frame[0]

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
        box = centre_box(centre_x, centre_y, self.box_size, self.frame)
        if self.has_override:
            self.overrides[self.clip.video] = box
        else:
            self.boxes[self.key] = box
        self.placed.add(self.key)
        self.matched.pop(self.key, None)
        if self.key in self.excluded:
            self.excluded.discard(self.key)
            self.restored.add(self.key)

    def auto(self, frames, min_score: float = AUTO_MIN_SCORE) -> dict[str, float]:
        """Box every unplaced recording where the hand-placed boxes match its frame.

        The crop of each recording placed by hand (not by an earlier match) is a
        template; *frames* gives a recording's native frames. A recording whose best
        normalised cross-correlation (mean over the templates) reaches *min_score*
        gets a box there, placed and kept in :attr:`matched` with its score until it
        is moved; the others stay unplaced. Returns every tried recording's score.
        """
        by_hand = [r for r in self.recordings if self.is_placed(r.key)]
        by_hand = [r for r in by_hand if r.key not in self.matched]
        templates = []
        for recording in by_hand:
            image = self._gray(frames(recording))
            if image is not None:
                x, y, w, h = self.boxes[recording.key]
                templates.append(image[y : y + h, x : x + w])
        if not templates:
            raise ValueError("place a box by hand first: it is what Auto looks for")
        size = templates[0].shape[1]
        scores = {}
        for recording in self.recordings:
            if self.is_placed(recording.key) or recording.key in self.excluded:
                continue
            image = self._gray(frames(recording))
            if image is None:
                continue
            score = np.mean(
                [cv2.matchTemplate(image, t, cv2.TM_CCOEFF_NORMED) for t in templates],
                axis=0,
            )
            y, x = np.unravel_index(np.argmax(score), score.shape)
            scores[recording.key] = float(score[y, x])
            if scores[recording.key] >= min_score:
                self.boxes[recording.key] = centre_box(
                    x + size / 2, y + size / 2, self.box_size, self.target_size
                )
                self.placed.add(recording.key)
                self.matched[recording.key] = scores[recording.key]
        return scores

    def _gray(self, frames: list[np.ndarray]) -> np.ndarray | None:
        """A recording's first frame at the target size, gray and a little blurred."""
        if not frames:
            return None
        image = downsample(frames[0], self.target_size).astype(np.float32)
        if image.ndim == 3:
            image = image.mean(axis=2)
        return cv2.GaussianBlur(image, (0, 0), AUTO_BLUR)

    def nudge(self, dx: int, dy: int) -> None:
        cx, cy = box_centre(self.box)
        self.place(cx + dx, cy + dy)

    def resize(self, delta: int) -> None:
        """Change the box size for *every* clip -- one input shape for the model."""
        if self.locked:
            return
        self.box_size = int(np.clip(self.box_size + delta, 16, min(self.target_size)))
        for key, box in self.boxes.items():
            self.boxes[key] = centre_box(
                *box_centre(box), self.box_size, self.target_size
            )
        for video, box in self.overrides.items():
            self.overrides[video] = centre_box(
                *box_centre(box), self.box_size, self.frame_of(video)
            )

    def set_target(self, width: int, height: int) -> None:
        """Change the downsample target, carrying every box across proportionally.

        Box *size* is deliberately left alone: it is the model's input shape, so
        the annotator picks it directly.  Changing the target therefore changes
        how much of the frame that shape covers, which is the actual trade-off.
        """
        if self.locked:
            return
        width = int(np.clip(width, 16, self.native_size[0]))
        height = int(np.clip(height, 16, self.native_size[1]))
        ratio_x = width / self.target_size[0]
        ratio_y = height / self.target_size[1]
        self.target_size = (width, height)
        self.box_size = min(self.box_size, width, height)
        for table in (self.boxes, self.overrides):
            for key, box in table.items():
                if key in self.frames:  # on a frame of its own, which stays
                    continue
                cx, cy = box_centre(box)
                table[key] = centre_box(
                    cx * ratio_x, cy * ratio_y, self.box_size, self.target_size
                )

    def set_input_scale(self, scale: float) -> None:
        """Put the displayed clip on *scale* times the list's frame (1: the list's
        own), its box carried onto the same part of the image, as a box of its own.
        The box keeps its size in pixels: the network's input."""
        video = self.clip.video
        new = scaled_frame(self.target_size, scale)
        old = self.frame
        if new == old:
            return
        moved = scale_box(self.box, old, new)
        if new == self.target_size:
            self.frames.pop(video, None)
        else:
            self.frames[video] = new
        self.overrides[video] = centre_box(*box_centre(moved), self.box_size, new)
        self.placed.add(self.key)

    def reset(self) -> None:
        """Return this recording's box to the frame centre, unplaced."""
        self.overrides.pop(self.clip.video, None)
        self.boxes[self.key] = centre_box(
            self.target_size[0] / 2,
            self.target_size[1] / 2,
            self.box_size,
            self.target_size,
        )
        own = self.frames.get(self.clip.video)
        if own is not None:  # a frame of its own keeps a box of its own
            self.overrides[self.clip.video] = centre_box(
                own[0] / 2, own[1] / 2, self.box_size, own
            )
        self.placed.discard(self.key)
        self.matched.pop(self.key, None)

    def toggle_exclude(self) -> None:
        """Mark this recording never to be used (its box removed), or use it again."""
        if self.key in self.excluded:
            self.excluded.discard(self.key)
            self.restored.add(self.key)
            return
        self.reset()
        for clip in self.recording.clips:
            if clip.video not in self.frames:
                self.overrides.pop(clip.video, None)
        self.excluded.add(self.key)
        self.restored.discard(self.key)

    def copy_previous(self) -> None:
        if self.index > 0:
            self.place(*box_centre(self.boxes[self.recordings[self.index - 1].key]))

    def toggle_override(self) -> None:
        """Give the displayed clip a box of its own (from its recording's), or drop it;
        a clip on its own frame keeps its own."""
        video = self.clip.video
        if video in self.frames:
            return
        if video in self.overrides:
            del self.overrides[video]
        else:
            self.overrides[video] = self.boxes[self.key]
            self.placed.add(self.key)

    def set_part(self, part_index: int) -> None:
        n_parts = len(self.recording.clips)
        self.part_index = int(np.clip(part_index, 0, max(0, n_parts - 1)))

    def progress(self) -> tuple[int, int]:
        """Recordings placed, of those not excluded."""
        return len(self.placed), len(self.recordings) - len(self.excluded)

    def clip_boxes(self) -> dict[Path, Box]:
        """The box of every clip whose recording has been placed, and of every clip on
        a frame of its own (which always has one)."""
        return {
            clip.video: self.overrides.get(clip.video, self.boxes[recording.key])
            for recording in self.recordings
            for clip in recording.clips
            if recording.key in self.placed or clip.video in self.frames
        }


def initial_state(
    clips: list[ResolvedClip],
    native_size: tuple[int, int],
    target_size: tuple[int, int],
    box_size: int,
    excluded: frozenset[Path] = frozenset(),
) -> BoxState:
    """State from a clip list's existing boxes: a recording's first box is its
    own, and any clip whose box differs from that is an override. A recording whose
    clips are all *excluded* videos is excluded."""
    recordings = recordings_of(clips)
    frames = {c.video: c.frame for c in clips if c.frame and c.frame != target_size}
    boxes: dict[str, Box] = {}
    overrides: dict[Path, Box] = {}
    for recording in recordings:
        have = [c for c in recording.clips if c.box is not None]
        overrides |= {c.video: c.box for c in have if c.video in frames}
        have = [c for c in have if c.video not in frames]
        if have:
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
        frames=frames,
        excluded={
            r.key for r in recordings if all(c.video in excluded for c in r.clips)
        },
    )


def save_boxes(path: Path, state: BoxState) -> None:
    """Write the geometry and every placed box back into the clip list; a recording
    re-centred (unplaced) loses its box."""
    unplaced = {
        clip.video
        for recording in state.recordings
        if recording.key not in state.placed
        for clip in recording.clips
        if clip.video not in state.frames
    }
    videos = {r.key: {c.video for c in r.clips} for r in state.recordings}
    write_boxes(
        path,
        state.clip_boxes(),
        state.target_size,
        frames={
            clip.video: state.frames.get(clip.video)
            for recording in state.recordings
            for clip in recording.clips
        },
        clear=unplaced,
        exclude=set().union(*(videos[k] for k in state.excluded)),
        restore=set().union(*(videos[k] for k in state.restored)),
    )


# ---------------------------------------------------------------------------
# Frame cache
# ---------------------------------------------------------------------------


def frame_path(frame_dir: Path, video: Path) -> Path:
    """Cache file for *video*'s frame, keyed by its path."""
    digest = hashlib.sha256(str(video).encode()).hexdigest()[:10]
    return frame_dir / f"frame_{video.stem}-{digest}.npy"


def frame_time(clip: ResolvedClip) -> float:
    """Seconds into *clip* of its cached frame."""
    if clip.frame_rate is not None:  # no frame count needed: the container's length
        return min(PREPARE_START_S, probe_duration(clip.video) / 2)
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
    checkpoint: Path | None = None,
) -> BoxState:
    """The box state of the clip list at *path*, on the target ``width x height``
    (default: the list's), its existing boxes carried across. With *checkpoint* the
    box is that network's crop and the frame its size, both locked."""
    clip_list = load(ClipList, path, check_files=not remote)
    if checkpoint is not None:
        from .clips import checkpoint_recipe

        recipe, crop = checkpoint_recipe(checkpoint)
        if tuple(recipe["target_size"]) != clip_list.preprocess.target_size:
            raise SystemExit(
                f"{path} is not on {checkpoint}'s {recipe['target_size']} frames: "
                "make it with `zephyr clips scan --checkpoint`"
            )
        if crop is None or crop[0] != crop[1]:
            raise SystemExit(f"{checkpoint} does not say one square crop size")
        width = height = None
        box_size = crop[0]
    clips = clip_list.resolve(include_excluded=True)  # shown, to be used again
    excluded = frozenset(c.video for c in clips) - {
        c.video for c in clip_list.resolve()
    }
    try:
        require_videos(clips, "placing boxes")
    except ValueError as exc:
        raise SystemExit(f"{path}: {exc}") from None
    used = [c for c in clips if c.video not in excluded] or clips
    native = frame_size(clips, frame_dir) if remote else probe_size(used[0].video)
    saved_target = clip_list.preprocess.target_size
    target = (width or saved_target[0], height or saved_target[1])
    # Start on the grid the boxes were saved on, then move to the requested
    # target, so existing placements are carried across rather than
    # reinterpreted as coordinates on a different-sized frame.
    state = initial_state(clips, native, saved_target, box_size, excluded)
    if box_size != DEFAULT_BOX_SIZE or checkpoint is not None:
        state.resize(box_size - state.box_size)
    if state.target_size != target:
        state.set_target(*target)
    state.locked = checkpoint is not None
    return state


def missing_frames(state: BoxState, frame_dir: Path) -> list[str]:
    """Clips with no cached frame (they show blank); excluded recordings need none."""
    return [
        clip.name
        for recording in state.recordings
        if recording.key not in state.excluded
        for clip in recording.clips
        if not frame_path(frame_dir, clip.video).exists()
    ]


def main(argv: list[str] | None = None) -> None:
    if argv and argv[0] == "events":
        from . import annotate_events

        annotate_events.main(argv[1:])
        return
    parser = argparse.ArgumentParser(
        description="Cache a frame per clip, or box the rest from the boxes placed "
        "(placing them by hand is the app's: zephyr gui <list> --page boxes)."
    )
    parser.add_argument("clips", type=Path, help="Clip list TOML file.")
    parser.add_argument("--frame-dir", type=Path, default=Path("artifacts/frames"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--prepare", action="store_true", help="Cache one frame per clip, then exit."
    )
    action.add_argument(
        "--auto",
        action="store_true",
        help="Box the unplaced recordings where the list's boxes match their cached "
        "frame, and save (A in the app's Boxes page).",
    )
    parser.add_argument("--min-score", type=float, default=AUTO_MIN_SCORE)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="With --auto: the boxes are this network's crop size (a list from "
        "`clips scan --checkpoint`).",
    )
    args = parser.parse_args(argv)
    if args.prepare:
        clips = load(ClipList, args.clips).resolve()
        try:
            require_videos(clips, "placing boxes")
        except ValueError as exc:
            raise SystemExit(f"{args.clips}: {exc}") from None
        prepare_frames(clips, args.frame_dir)
        return
    # Matching reads only the cached frames: the videos may be elsewhere.
    state = open_state(
        args.clips, args.frame_dir, remote=True, checkpoint=args.checkpoint
    )
    try:
        scores = state.auto(
            lambda r: load_recording_frames(r, args.frame_dir), args.min_score
        )
    except ValueError as exc:
        raise SystemExit(f"{args.clips}: {exc}") from None
    save_boxes(args.clips, state)
    for recording in state.recordings:
        if recording.key in scores:
            kept = "boxed" if recording.key in state.matched else "left unplaced"
            print(f"  {recording.label}: {scores[recording.key]:.2f} {kept}")
    print(
        f"matched {len(state.matched)} of {len(scores)} recording(s); "
        f"{len(state.placed)} of {len(state.recordings)} placed -> {args.clips}"
    )
