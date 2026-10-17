"""Every input zephyr consumes is a TOML file, validated here.

A clip list names videos; an experiment holds its folds (which lists train and which
are scored) and the seeds they run with. Nothing else knows the dataset layout;
the example dataset appears only in ``examples/`` and zephyr-benchmarks. Every path
field is a :data:`RelPath`: relative to the file it is written in, absolute once loaded.

A clip's ``group`` plus its video's folder is its *recording*, the unit leakage is
judged on: parts of one session share an animal and a camera placement. The same group
label in two folders is two recordings.
"""

import argparse
import functools
import itertools
import json
import os
import re
import string
import tomllib
from pathlib import Path
from typing import Annotated, ClassVar, Literal, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from . import events
from .channels import FLOW_CLIP_PX, FLOW_SCALE_PX, MOTION_TAU_S, ChannelSet
from .rates import DEFAULT_RATE_BINS_HZ
from .signal import CANONICAL_BREATHING_SAMPLING_RATE

OUTPUT_FS = CANONICAL_BREATHING_SAMPLING_RATE
"""The default output grid (``PreprocessParams.output_fs``), and always the scorer's:
predictions on another grid are resampled to it before any metric."""

Box = tuple[int, int, int, int]
"""Crop box ``(x, y, width, height)`` in ``target_size`` pixels."""


def _anchor(path: Path, info: ValidationInfo) -> Path:
    base = Path((info.context or {}).get("base", "."))
    return (base / path).resolve()


RelPath = Annotated[Path, AfterValidator(_anchor)]
"""A path written relative to the file it appears in, absolute once loaded."""

PATH_MAP_ENV = "ZEPHYR_PATH_MAP"
"""``<copy>=<original>[;<copy>=<original>...]``: folders on this machine that are copies of
folders elsewhere (e.g. ``C:/Data/zephyr-dataset=/data/zephyr-dataset``). See
:func:`canonical`."""


def _fold(text: str) -> str:
    """*text* for comparing paths: case-blind where the file system is (Windows)."""
    return text.lower() if os.name == "nt" else text


@functools.cache
def _path_map(text: str) -> tuple[tuple[str, str], ...]:
    pairs = []
    for item in filter(None, (part.strip() for part in text.split(";"))):
        copy, sep, original = item.partition("=")
        if not sep or not copy.strip() or not original.strip():
            raise ValueError(f"{PATH_MAP_ENV}: {item!r} is not <copy>=<original>")
        root = Path(copy.strip()).resolve().as_posix().rstrip("/")
        pairs.append((_fold(root), original.strip().rstrip("/")))
    return tuple(pairs)


def canonical(path: Path) -> str:
    """*path* as the cache keys, sidecars and run records name it: in ``/`` form, and
    inside a copied folder (:data:`PATH_MAP_ENV`) as the original's path, so a copy of a
    dataset finds the caches made from the original and its checkpoints' training
    records. Without a map, ``path.as_posix()``."""
    text = path.as_posix()
    folded = _fold(text)
    for root, original in _path_map(os.environ.get(PATH_MAP_ENV, "")):
        if folded == root or folded.startswith(root + "/"):
            return original + text[len(root) :]
    return text


class _Model(BaseModel):
    """Frozen and strict: a typo in a TOML key is an error, never a default."""

    model_config = ConfigDict(extra="forbid", frozen=True)


M = TypeVar("M", bound=_Model)


def load(model: type[M], path: Path, check_files: bool = True) -> M:
    """Read the TOML file at *path* into *model*, resolving paths against it. A clip
    list's files must exist unless *check_files* is False (clips kept elsewhere)."""
    path = Path(path).resolve()
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    try:
        return model.model_validate(
            data, context={"base": path.parent, "check_files": check_files}
        )
    except ValidationError as exc:
        # Name the file: errors from a fold's nested clip lists are otherwise
        # impossible to place.
        raise ValueError(f"{path}: {exc}") from exc


def dump(model: _Model) -> dict:
    """JSON-ready dump with absolute paths -- what a run records of its inputs."""
    return model.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Clip lists
# ---------------------------------------------------------------------------


class PreprocessParams(_Model):
    """How raw video becomes channel arrays.  Part of every feature cache key."""

    target_size: tuple[PositiveInt, PositiveInt]
    """Working frame size ``(width, height)``; the video is scaled to this
    before cropping, and clip boxes are expressed in these pixels.  No default:
    a box means nothing without the frame it was placed on."""
    select_fs: float = Field(OUTPUT_FS, gt=0)
    """Rate of the frames the CNN sees, in Hz. Below the output rate, the TCN's input
    is interpolated up to it (half the CNN work at 30 Hz)."""
    motion_tau_s: float = Field(MOTION_TAU_S, gt=0)
    flow_scale_px: float = Field(FLOW_SCALE_PX, gt=0)
    flow_clip_px: float = Field(FLOW_CLIP_PX, gt=0)
    onset_sigma_s: float = Field(0.020, gt=0)
    output_fs: float = Field(OUTPUT_FS, gt=0)
    """Rate of the grid the network predicts on and trains against, in Hz. The TCN's
    dilations count samples of this grid, so its context in seconds scales with
    1 / output_fs. Scoring is always on the 60 Hz grid."""

    KEY_DEFAULTS: ClassVar[dict[str, float]] = {"output_fs": OUTPUT_FS}
    """Fields added after caches existed: left out of the cache key at these values,
    so the caches made before them still match."""


class Derive(_Model):
    """How a clip's omitted fields are derived from its video file name.

    The defaults match the example dataset, whose files are named
    ``video_{camera}_{session}_part_{part}.mp4`` next to
    ``thermistor_{session}_part_{part}.parquet``.
    """

    pattern: str = r"video_.+_(?P<group>\d+)_part_(?P<part>\d+)$"
    """Regex searched in the video stem.  Must capture ``group``; every other
    named group is available to the templates."""
    timestamps: str = "{stem}.parquet"
    thermistor: str = "thermistor_{group}_part_{part}.parquet"
    """Templates over ``{stem}`` and the pattern's named groups, relative to
    the video's folder."""

    @field_validator("pattern")
    @classmethod
    def _capture_group(cls, value: str) -> str:
        try:
            compiled = re.compile(value)
        except re.error as exc:
            raise ValueError(f"pattern is not a valid regex: {exc}") from exc
        if "group" not in compiled.groupindex:
            raise ValueError("pattern must capture a named group 'group'")
        return value

    @model_validator(mode="after")
    def _templates_use_known_fields(self) -> "Derive":
        known = {"stem", *re.compile(self.pattern).groupindex}
        for name in ("timestamps", "thermistor"):
            fields = {
                field
                for _, field, _, _ in string.Formatter().parse(getattr(self, name))
                if field
            }
            if fields - known:
                raise ValueError(
                    f"{name} template uses {sorted(fields - known)}, but only "
                    f"{sorted(known)} are available"
                )
        return self

    def fields(self, video: Path) -> dict[str, str] | None:
        """Template fields for *video*, or ``None`` if the pattern misses."""
        match = re.search(self.pattern, video.stem)
        if match is None:
            return None
        return {"stem": video.stem, **match.groupdict()}


class EventsSource(_Model):
    """Which breathing events a clip uses: a detector and its parameters.

    Training (the onset head, rate balancing) and scoring both read them; see
    :mod:`zephyr.events`.  Set for a whole list with ``[events]`` or for one clip
    with ``events = {...}`` next to its ``box`` (``zephyr annotate events``);
    without either, ``peak_detection`` with its defaults.
    """

    method: str = events.DEFAULT_METHOD
    """Name of a detector in :data:`zephyr.events.DETECTORS`."""
    params: dict[str, float] = {}
    """Its parameters; omitted ones keep the detector's defaults."""

    @model_validator(mode="after")
    def _known_method(self) -> "EventsSource":
        events.detector_params(self.method, self.params)
        return self

    def for_clip(self, clip: "ResolvedClip", span=None) -> "events.Events":
        """The detector's events on the clip's thermistor, with its excluded spans."""
        found = events.for_clip(self.method, self.params, clip.thermistor, span)
        return events.Events.of(found.inhale, found.exhale, clip.excluded)

    def identity(self, clip: "ResolvedClip") -> dict:
        """What the clip's cached events depend on; spans enter only when present, so
        caches made without any still match."""
        out = events.identity(self.method, self.params)
        if clip.excluded:
            out["excluded"] = [list(span) for span in clip.excluded]
        return out


class Clip(_Model):
    """One clip as written in a clip list; omitted fields are derived.

    Without a ``video`` the entry is a thermistor-only recording (e.g. a whole session):
    it names its ``thermistor`` and has no timestamps or box. The detector tuner and
    ``zephyr clips select`` read such entries; everything that needs frames refuses
    them (:func:`require_videos`).
    """

    video: RelPath | None = None
    """The clip's video; omitted for a thermistor-only entry."""
    timestamps: RelPath | None = None
    """Frame timestamp parquet; derived when omitted."""
    thermistor: RelPath | Literal[False] | None = None
    """Ground-truth trace; derived when omitted, ``false`` for an unlabelled
    clip (inference only)."""
    group: str | None = None
    """Recording label within the video's folder; derived when omitted."""
    box: Box | None = None
    """Hand-placed crop box in ``target_size`` pixels; see ``zephyr annotate``."""
    events: "EventsSource | None" = None
    """This clip's own detector, overriding the list's ``[events]``."""
    selection: Literal["include", "exclude"] | None = None
    """Curation mark set from ``zephyr annotate events``.  ``"exclude"``: never use
    this video; it is left out of everything downstream (preprocessing, folds,
    training, scoring).  ``"include"`` is a note only; neither touches the clip's
    other settings."""
    reviewed: bool = False
    """Whether a person has confirmed ``selection`` (and the spans) in ``zephyr
    annotate events``; ``zephyr clips select`` never changes a reviewed clip."""
    excluded: list[tuple[float, float]] = []
    """Spans ``[start, end]`` (seconds on the thermistor's clock) whose events are
    untrusted: nothing is scored there and no training window overlaps them."""

    included: list[tuple[float, float]] = []
    """Spans ``[start, end]`` marked as good stretches of data from ``zephyr annotate
    events``: a note only, nothing downstream reads them."""

    @field_validator("excluded", "included")
    @classmethod
    def _ordered_spans(cls, value: list[tuple[float, float]]):
        for start, end in value:
            if not start < end:
                raise ValueError(f"span [{start}, {end}] needs start < end")
        return sorted(value)

    @field_validator("box")
    @classmethod
    def _positive_box(cls, value: Box | None) -> Box | None:
        if value is not None and (
            value[0] < 0 or value[1] < 0 or value[2] <= 0 or value[3] <= 0
        ):
            raise ValueError(f"box {list(value)} needs x, y >= 0 and w, h > 0")
        return value

    @model_validator(mode="after")
    def _video_or_thermistor(self) -> "Clip":
        if self.video is None:
            if not isinstance(self.thermistor, Path):
                raise ValueError(
                    "a clip needs a video, or (thermistor only) a thermistor"
                )
            if self.timestamps is not None or self.box is not None:
                raise ValueError(
                    f"{self.thermistor.name}: a thermistor-only clip has no "
                    "timestamps or box"
                )
        return self


class ResolvedClip(_Model):
    """A clip with every field concrete; what the rest of the package consumes.
    ``video`` and ``timestamps`` are None for a thermistor-only entry."""

    video: Path | None
    timestamps: Path | None
    thermistor: Path | None
    group: str
    box: Box | None
    excluded: tuple[tuple[float, float], ...] = ()
    events: EventsSource = EventsSource()
    """Its list's ``[events]``; the default detector when the list has none."""

    @property
    def labelled(self) -> bool:
        return self.thermistor is not None

    @property
    def source(self) -> Path:
        """The file that identifies the clip: its video, else its thermistor."""
        source = self.video if self.video is not None else self.thermistor
        assert source is not None  # Clip requires one of them
        return source

    @property
    def recording(self) -> tuple[Path, str]:
        """Identity leakage is judged on; see the module docstring."""
        return (self.source.parent, self.group)

    @property
    def recording_id(self) -> str:
        """The recording as one string, for reports and run records."""
        return f"{canonical(self.source.parent)}#{self.group}"

    @property
    def name(self) -> str:
        """Short human label: folder and file stem."""
        return f"{self.source.parent.name}/{self.source.stem}"


def require_videos(clips: list[ResolvedClip], what: str) -> None:
    """Refuse thermistor-only clips where frames are needed."""
    missing = [c.name for c in clips if c.video is None]
    if missing:
        raise ValueError(
            f"{what} needs videos; {len(missing)} clip(s) are thermistor only: {missing}"
        )


def resolve_clip(
    clip: Clip, derive: Derive, events_source: EventsSource | None = None
) -> ResolvedClip:
    """Fill a clip's omitted fields from its video name, or say why not."""
    events_source = clip.events or events_source or EventsSource()
    if clip.video is None:  # thermistor only: nothing to derive
        assert isinstance(clip.thermistor, Path)
        return ResolvedClip(
            video=None,
            timestamps=None,
            thermistor=clip.thermistor,
            group=clip.group if clip.group is not None else clip.thermistor.stem,
            box=None,
            excluded=tuple(clip.excluded),
            events=events_source,
        )
    fields = derive.fields(clip.video)

    def derived(label: str, template: str | None) -> str:
        if fields is None:
            raise ValueError(
                f"{clip.video.name}: cannot derive {label}: pattern "
                f"{derive.pattern!r} does not match the video name; give "
                f"{label} explicitly"
            )
        assert template is not None
        return template.format(**fields)

    group = clip.group if clip.group is not None else derived("group", "{group}")
    if clip.timestamps is not None:
        timestamps = clip.timestamps
    else:
        timestamps = clip.video.parent / derived("timestamps", derive.timestamps)
    if clip.thermistor is False:
        thermistor = None
    elif clip.thermistor is not None:
        thermistor = clip.thermistor
    else:
        thermistor = clip.video.parent / derived("thermistor", derive.thermistor)
    return ResolvedClip(
        video=clip.video,
        timestamps=timestamps,
        thermistor=thermistor,
        group=group,
        box=clip.box,
        excluded=tuple(clip.excluded),
        events=events_source,
    )


class ClipList(_Model):
    """A named set of clips sharing one preprocessing recipe."""

    preprocess: PreprocessParams
    derive: Derive = Derive()
    events: EventsSource | None = None
    """Which breathing events the clips use; see :class:`EventsSource`."""
    clip: list[Clip] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_clips(self, info: ValidationInfo) -> "ClipList":
        resolved = self.resolve()
        if not resolved:
            raise ValueError("every clip of the list is excluded")

        sources = [c.source for c in self.resolve(include_excluded=True)]
        duplicated = sorted({str(v) for v in sources if sources.count(v) > 1})
        if duplicated:
            raise ValueError(f"videos listed more than once: {duplicated}")

        missing = [
            str(path)
            for c in resolved
            for path in (c.video, c.timestamps, c.thermistor)
            if path is not None
            and not path.is_file()
            and (info.context or {}).get("check_files", True)
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} file(s) do not exist (mark an unlabelled clip "
                f"`thermistor = false`): {missing}"
            )

        width, height = self.preprocess.target_size
        sizes = set()
        for c in resolved:
            if c.box is None:
                continue
            x, y, w, h = c.box
            if x + w > width or y + h > height:
                raise ValueError(
                    f"{c.name}: box {list(c.box)} lies outside the "
                    f"{width}x{height} target frame"
                )
            sizes.add((w, h))
        if len(sizes) > 1:
            raise ValueError(
                f"boxes of mixed sizes {sorted(sizes)}: every clip must crop to "
                "the same shape, since the model has a single input shape"
            )
        return self

    def resolve(self, include_excluded: bool = False) -> list[ResolvedClip]:
        """The clips in file order, every field concrete.  Excluded clips are left
        out unless *include_excluded* (only the detector tuner wants them)."""
        return [
            resolve_clip(c, self.derive, self.events)
            for c in self.clip
            if include_excluded or c.selection != "exclude"
        ]


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


class Augmentation(_Model):
    """Training-time augmentation; every default is today's ``train.py`` default.

    Mirrors :class:`zephyr.augment.AugmentConfig`, whose docstring explains
    each knob.  The augmentation defaults here are *on*, unlike the
    config's own no-op defaults, because they are what a training run wants.
    """

    select_jitter: float = Field(0.25, ge=0)
    motion_noise: float = Field(2.0, ge=0)
    shift_px: int = Field(4, ge=0)
    scale: float = Field(0.0, ge=0, lt=1)
    rotation: float = Field(0.0, ge=0, le=180)
    blur: float = Field(0.0, ge=0)
    brightness: float = Field(0.15, ge=0)
    contrast: float = Field(0.2, ge=0)
    noise: float = Field(2.0, ge=0)
    flip: float = Field(0.0, ge=0, le=1)


class TrainParams(_Model):
    """Every training knob; defaults are the ones ``zephyr train`` used to have."""

    channels: str = "gray+diff+flow"
    """Channels to train on (``gray``, ``diff``, ``flow``, ``flow_x``, ...);
    preprocessing stores all of them, so this selects without reprocessing."""
    window: PositiveInt = 512
    batch_size: PositiveInt = 4
    epochs: PositiveInt = 200
    steps_per_epoch: PositiveInt = 200
    lr: float = Field(3e-4, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    warmup_steps: int = Field(200, ge=0)
    grad_clip: float = Field(1.0, ge=0)
    dropout: float = Field(0.1, ge=0, lt=1)
    ema_decay: float = Field(0.999, ge=0, lt=1)
    """Exponential moving average of the weights; 0 disables it."""
    save_every: int = Field(0, ge=0)
    """Also keep the weights every this many epochs as ``epoch-<n>.pt``, scorable
    with ``evaluate``; 0 keeps only ``best.pt`` and ``last.pt``.  Mid-run weights
    were trained with the learning rate not yet decayed."""
    augmentation: Augmentation = Augmentation()
    rate_balance: float = Field(0.0, ge=0, le=1)
    """Draw training windows by their true breathing rate: weight ``share(bin) **
    -rate_balance``. 0 keeps today's draw; 1 gives every rate bin equal mass."""
    rate_bins_hz: tuple[float, ...] = DEFAULT_RATE_BINS_HZ
    """Rate bin edges for ``rate_balance``; rates outside count in the end bins."""
    signal_pool: PositiveInt = 1
    """The signal head reads the TCN's output average-pooled by this factor in
    time and upsampled back (smoother trace). 1 is the network without the
    branch."""
    temporal_kernel: PositiveInt = 1
    """Frames the encoder's first layer spans (odd): 1 is the per-frame network; 3 or 5
    make it a 3D convolution that sees change between neighbouring frames itself."""
    onset_input: Literal["full", "pooled", "both"] = "full"
    """What the onset head reads: the TCN's full-rate output, the pooled branch's,
    or both concatenated. ``pooled`` and ``both`` need ``signal_pool > 1``."""
    w_corr: float = Field(1.0, ge=0)
    w_onset: float = Field(0.5, ge=0)
    scales: tuple[PositiveInt, ...] = (1, 4, 16)
    val_fraction: float = Field(0.25, ge=0, lt=1)
    """Time-tail of every training clip held back for early stopping; 0
    disables validation (a fixed budget of ``epochs``)."""
    score_every: PositiveInt = 4
    patience: int = Field(8, ge=0)
    min_epochs: int = Field(24, ge=0)
    val_windows: PositiveInt = 2048
    infer_window: PositiveInt = 1024
    frame_chunk: PositiveInt = 256

    @field_validator("channels")
    @classmethod
    def _known_channels(cls, value: str) -> str:
        ChannelSet.parse(value)
        return value

    @field_validator("rate_bins_hz")
    @classmethod
    def _increasing_bins(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if (
            len(value) < 2
            or value[0] <= 0
            or any(b <= a for a, b in itertools.pairwise(value))
        ):
            raise ValueError(
                f"rate_bins_hz {list(value)} needs two or more increasing edges > 0"
            )
        return value

    @field_validator("temporal_kernel")
    @classmethod
    def _odd_kernel(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError(f"temporal_kernel must be odd (centred), got {value}")
        return value

    @model_validator(mode="after")
    def _zoom_needs_no_flow(self) -> "TrainParams":
        flow = {"flow_x", "flow_y"} & set(ChannelSet.parse(self.channels).names)
        for knob in ("scale", "rotation"):
            if getattr(self.augmentation, knob) > 0 and flow:
                raise ValueError(
                    f"augmentation {knob} cannot warp the flow channels {sorted(flow)}: "
                    "train on channels without flow (gray, gray+diff) or leave it at 0"
                )
        return self

    @model_validator(mode="after")
    def _onset_input_needs_the_branch(self) -> "TrainParams":
        if self.onset_input != "full" and self.signal_pool < 2:
            raise ValueError(
                f"onset_input = {self.onset_input!r} reads the pooled branch, "
                "which needs signal_pool > 1"
            )
        return self


def _load_list(path: Path) -> ClipList:
    return load(ClipList, path)


class Selection(_Model):
    """A clip list, optionally narrowed to some of its groups (recordings).

    Lets one list hold every session of a dataset while each fold picks which
    sessions train and which are scored.
    """

    clips: RelPath
    groups: list[str] | None = Field(None, min_length=1)
    """Groups to keep; omitted keeps the whole list."""

    def select(self, clip_list: ClipList) -> list[ResolvedClip]:
        """The selected clips in file order; unknown group names are an error."""
        resolved = clip_list.resolve()
        if self.groups is None:
            return resolved
        present = {c.group for c in resolved}
        unknown = sorted(set(self.groups) - present)
        if unknown:
            raise ValueError(
                f"{self.clips.name}: groups {unknown} are not in the list "
                f"(it has {sorted(present)})"
            )
        return [c for c in resolved if c.group in self.groups]


class TrainSource(Selection):
    """Clips to train on and their share of the sampled windows."""

    weight: float = Field(1.0, gt=0)
    """Relative to the other sources: weights 0.5 / 0.5 and 1 / 1 are the same."""


FOLD_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class Fold(_Model):
    """One train/test arrangement: what trains, what is scored, from where.

    Written as a ``[[fold]]`` block of an experiment; its *name* is the run
    directory, so it must be a plain file name.
    """

    name: str
    """The fold's run directory under the experiment's ``output_dir``."""
    init_from: RelPath | None = None
    """Checkpoint whose weights to start from (fresh optimiser and schedule)."""
    train: list[TrainSource] = Field(min_length=1)
    test: dict[str, Selection] = Field(min_length=1)
    """Report group name -> clips to score; the names are the config author's.
    A bare path is shorthand for the whole list."""
    val: Selection | None = None
    """Clips, scored whole, that select ``best.pt`` and stop training early, in place
    of the time-tail of the training clips (``val_fraction``). They never train and
    must share no recording with the training or test clips."""
    train_params: TrainParams = TrainParams()

    @field_validator("name")
    @classmethod
    def _plain_name(cls, value: str) -> str:
        if not FOLD_NAME.match(value):
            raise ValueError(
                f"fold name {value!r} must be a plain file name "
                "(letters, digits, '.', '_', '-')"
            )
        return value

    @field_validator("test", mode="before")
    @classmethod
    def _bare_paths(cls, value):
        if isinstance(value, dict):
            return {
                name: {"clips": entry} if isinstance(entry, str) else entry
                for name, entry in value.items()
            }
        return value

    @field_validator("test")
    @classmethod
    def _no_reserved_name(cls, value: dict[str, Selection]) -> dict[str, Selection]:
        if "all" in value:
            raise ValueError("'all' is reserved for the summary across test groups")
        return value

    def train_lists(self) -> list[ClipList]:
        return [_load_list(source.clips) for source in self.train]

    def test_lists(self) -> dict[str, ClipList]:
        return {name: _load_list(s.clips) for name, s in self.test.items()}

    @property
    def preprocess(self) -> PreprocessParams:
        return self.train_lists()[0].preprocess

    def train_clips(self) -> list[list[ResolvedClip]]:
        """Selected training clips, one list per ``[[train]]`` source."""
        return [s.select(lst) for s, lst in zip(self.train, self.train_lists())]

    def test_clips(self) -> dict[str, list[ResolvedClip]]:
        return {
            name: self.test[name].select(lst) for name, lst in self.test_lists().items()
        }

    def val_list(self) -> ClipList | None:
        return _load_list(self.val.clips) if self.val is not None else None

    def val_clips(self) -> list[ResolvedClip]:
        """The selected validation clips; empty without ``val``."""
        return self.val.select(self.val_list()) if self.val is not None else []

    @model_validator(mode="after")
    def _check_fold(self) -> "Fold":
        train_lists, test_lists = self.train_lists(), self.test_lists()
        train = self.train_clips()
        test = self.test_clips()
        val = self.val_clips()
        val_lists = [self.val_list()] if self.val is not None else []
        everything = [*train_lists, *test_lists.values(), *val_lists]
        recipes = {lst.preprocess for lst in everything}
        if len(recipes) > 1:
            raise ValueError(
                "clip lists disagree on [preprocess]; one model is trained and "
                f"scored on one recipe, got {len(recipes)} different ones"
            )

        selected = [c for clips in [*train, *test.values(), val] for c in clips]
        require_videos(selected, "a fold")
        sizes = {(c.box[2], c.box[3]) for c in selected if c.box is not None}
        unboxed = [c.name for c in selected if not c.box]
        if unboxed:
            raise ValueError(
                f"{len(unboxed)} clip(s) have no box; place them with "
                f"`zephyr annotate`: {unboxed}"
            )
        if len(sizes) > 1:
            raise ValueError(f"clip lists crop to different box sizes {sorted(sizes)}")

        train_clips = [c for clips in train for c in clips]
        unlabelled = [c.name for c in train_clips if not c.labelled]
        if unlabelled:
            raise ValueError(f"training clips need a thermistor: {unlabelled}")

        train_videos = {c.video for c in train_clips}
        train_recordings = {c.recording for c in train_clips}
        for name, clips in test.items():
            for c in clips:
                if c.video in train_videos:
                    raise ValueError(
                        f"test group {name!r}: {c.name} is also a training clip"
                    )
                if c.recording in train_recordings:
                    raise ValueError(
                        f"test group {name!r}: {c.name} shares recording "
                        f"{c.recording_id} with a training clip"
                    )
        if self.val is not None:
            self._check_val(val, train_clips, test)
        return self

    def _check_val(
        self,
        val: list[ResolvedClip],
        train: list[ResolvedClip],
        test: dict[str, list[ResolvedClip]],
    ) -> None:
        """Validation clips are scored, never trained on, and choose the checkpoint:
        they need labels and must not overlap training (leakage) or a test group
        (a test score chosen on itself)."""
        if not val:
            raise ValueError("val selects no clips")
        if "val_fraction" in self.train_params.model_fields_set:
            raise ValueError(
                "val and train_params.val_fraction are two ways to validate; "
                "with val, leave val_fraction out"
            )
        unlabelled = [c.name for c in val if not c.labelled]
        if unlabelled:
            raise ValueError(f"validation clips need a thermistor: {unlabelled}")
        against = {"training": train} | {
            f"test group {n!r}": cs for n, cs in test.items()
        }
        for what, clips in against.items():
            recordings = {c.recording for c in clips}
            for c in val:
                if c.recording in recordings:
                    raise ValueError(
                        f"val: {c.name} shares recording {c.recording_id} with {what}"
                    )


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


class Experiment(_Model):
    """Folds to run (inline ``[[fold]]`` blocks), the seeds each runs with, and
    where results go.  A subclass may hold a richer fold by redeclaring *fold*."""

    fold: list[Fold] = Field(min_length=1)
    """Each runs into ``{output_dir}/{name}/seed-{seed}/``."""
    seeds: list[int] = Field(min_length=1)
    """Each fold runs once per seed; the seed belongs here, not to the fold."""
    output_dir: RelPath
    features_dir: RelPath
    """Feature cache the folds read; filled by ``zephyr preprocess``."""
    device: str | None = None
    amp: Literal["bf16", "fp16", "off"] | None = None
    num_workers: int | None = Field(None, ge=0)
    """Machine settings, not science: unset means ``zephyr run``'s defaults."""

    @field_validator("seeds")
    @classmethod
    def _distinct_seeds(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError(f"seeds must be distinct, got {value}")
        return value

    @model_validator(mode="after")
    def _distinct_names(self) -> "Experiment":
        names = [fold.name for fold in self.fold]
        if len(set(names)) != len(names):
            raise ValueError(
                f"folds share a name, so their runs would collide: {names}"
            )
        return self

    def folds(self, names: list[str] | None = None) -> list[Fold]:
        """The folds called *names* (all when ``None``), in the file's order."""
        if names is None:
            return list(self.fold)
        known = {fold.name: fold for fold in self.fold}
        unknown = sorted(set(names) - set(known))
        if unknown:
            raise ValueError(
                f"no fold named {unknown}; the experiment has {sorted(known)}"
            )
        return [fold for fold in self.fold if fold.name in names]


# ---------------------------------------------------------------------------
# JSON schemas, for editor validation of the TOML files
# ---------------------------------------------------------------------------

SCHEMAS = {"clip-list": ClipList, "experiment": Experiment}


def write_schemas(directory: Path) -> list[Path]:
    """Write one JSON schema per config file type into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, model in SCHEMAS.items():
        path = directory / f"{name}.schema.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2))
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write JSON schemas of the clip list and experiment files."
    )
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    for path in write_schemas(args.directory):
        print(f"wrote {path}")
