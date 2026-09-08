"""Clip discovery: sessions, parts, and how clips are found on disk.

Packaged files are named ``{stream}_{session}_part_{part}.{ext}`` with session
indices restarting at 1 within each split, so an index is only meaningful
alongside its split -- ``session_idx`` is the whole identity a clip carries,
and the key everything downstream groups by.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

CAMERAS = ("face", "side")

PUBLIC_SPLIT = "train"
"""Packaged split that carries ground-truth training targets."""

PRIVATE_SPLIT = "test"
"""Packaged split held out for scoring.  Its labels must not be used for
training, validation, or model selection — see :func:`assert_public`."""

_CLIP_RE = re.compile(r"^thermistor_(\d+)_part_(\d+)\.parquet$")


@dataclass(frozen=True)
class ClipRef:
    """A single packaged clip: one 300 s window of one session."""

    root: Path
    split: str
    session_idx: int
    part: int

    @property
    def suffix(self) -> str:
        """The ``{session}_part_{part}`` stem shared by every file in the clip."""
        return f"{self.session_idx}_part_{self.part}"

    @property
    def clip_id(self) -> str:
        return f"{self.split}_{self.suffix}"

    @property
    def dir(self) -> Path:
        return self.root / self.split

    def video(self, camera: str) -> Path:
        return self.dir / f"video_{camera}_{self.suffix}.mp4"

    def frame_times_path(self, camera: str) -> Path:
        return self.dir / f"video_{camera}_{self.suffix}.parquet"

    def thermistor_path(self) -> Path:
        return self.dir / f"thermistor_{self.suffix}.parquet"

    def exists(self, camera: str) -> bool:
        return self.video(camera).exists() and self.frame_times_path(camera).exists()


def discover_clips(packaged_root: Path, split: str = PUBLIC_SPLIT) -> list[ClipRef]:
    """Enumerate clips in ``{packaged_root}/{split}``, sorted by session then part.

    Clips are found by their thermistor parquet, so an unlabelled split
    enumerates as empty -- which is what stops a private directory being
    preprocessed into targetless features that then fail obscurely in training.
    """
    clips: list[ClipRef] = []
    for path in sorted((packaged_root / split).glob("thermistor_*.parquet")):
        match = _CLIP_RE.match(path.name)
        if match is None:
            continue
        clips.append(
            ClipRef(
                root=packaged_root,
                split=split,
                session_idx=int(match.group(1)),
                part=int(match.group(2)),
            )
        )
    return sorted(clips, key=lambda c: (c.session_idx, c.part))


@dataclass(frozen=True)
class SessionRef:
    """One recording session and the clips packaged from it.

    A session is the natural unit for anything the camera geometry determines --
    a crop box, most obviously -- because the camera is not repositioned
    between a session's parts.  It is also the unit for grouped validation, so
    a session's clips always land on the same side of a train/test split.
    """

    split: str
    session_idx: int
    clips: tuple[ClipRef, ...]

    @property
    def key(self) -> str:
        """Stable identifier, unique across splits (session indices restart)."""
        return f"{self.split}_{self.session_idx}"

    @property
    def clip_ids(self) -> list[str]:
        return [c.clip_id for c in self.clips]

    @property
    def is_public(self) -> bool:
        return self.split == PUBLIC_SPLIT

    @property
    def label(self) -> str:
        return f"session {self.session_idx}"


def discover_sessions(
    packaged_root: Path,
    splits: Sequence[str] = (PUBLIC_SPLIT, PRIVATE_SPLIT),
    *,
    camera: str | None = None,
) -> list[SessionRef]:
    """Group clips into sessions, across one or more splits.

    Parameters
    ----------
    packaged_root:
        Directory holding the split sub-directories.
    splits:
        Splits to enumerate, in the order they should be presented.
    camera:
        When given, only clips having video for this camera are included, and
        sessions left with no clips are dropped.
    """
    sessions: list[SessionRef] = []
    for split in splits:
        if not (packaged_root / split).is_dir():
            continue
        by_index: dict[int, list[ClipRef]] = {}
        for clip in discover_clips(packaged_root, split):
            if camera is not None and not clip.exists(camera):
                continue
            by_index.setdefault(clip.session_idx, []).append(clip)

        for session_idx in sorted(by_index):
            group = sorted(by_index[session_idx], key=lambda c: c.part)
            sessions.append(
                SessionRef(
                    split=split,
                    session_idx=session_idx,
                    clips=tuple(group),
                )
            )
    return sessions
