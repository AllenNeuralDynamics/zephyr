"""The dataset spec (one TOML file) and the layout of the dataset folder it names.

Every setting is written out in the spec, QC thresholds included, so a dataset never
changes when a default of zephyr's does. ``root`` is a URL (``s3://...``) or a path
relative to the spec file.
"""

from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationInfo
from upath import UPath

from zephyr.config import EventsSource, PreprocessParams
from zephyr.session_selection import Settings

from . import storage


def _root(value: str, info: ValidationInfo) -> str:
    if "://" in value:
        return value.rstrip("/")
    return str((Path((info.context or {}).get("base", ".")) / value).resolve())


Root = Annotated[str, AfterValidator(_root)]
"""A URL, kept as is, or a path relative to the spec file, absolute once loaded."""


class _Model(BaseModel):
    """Frozen and strict: a typo in a key is an error, never a default."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Source(_Model):
    """Where the sessions come from."""

    bucket: str = "aind-open-data"
    """S3 bucket of the raw and derived assets (read anonymously)."""
    docdb: str = "https://api.allenneuraldynamics.org/v2/metadata_index/data_assets"
    """DocDB collection the assets are queried from (read-only REST API)."""
    process: str = "primary-nwb-packaging-vr-foraging"
    """Processing step whose derived assets define the sessions (DocDB)."""
    device: str = "SniffDetector"
    """Harp device folder (``behavior/<device>.harp``) holding the thermistor."""
    signal: str = "RawVoltage"
    """Register of that device read as the thermistor signal."""
    cameras: list[str] = ["FaceCamera", "SideCamera"]
    """Camera folders (``behavior-videos/<camera>``, each with ``video.mp4`` and
    ``metadata.csv``) a selected session needs."""


class Query(_Model):
    """Which inventoried sessions are fetched and scored."""

    until: datetime | None = None
    """Latest creation time of an asset kept (UTC when no zone is given; a bare date is
    its midnight): later packagings are ignored, so a rebuild picks the same sessions
    (with ``docdb.json.gz``, from the same metadata)."""
    min_dataset_version: str | None = None
    """Lowest ``dataset_version`` kept (PEP 440 ordering)."""
    include_sessions: list[str] = []
    """Sessions kept whatever their version."""
    exclude_sessions: list[str] = []
    """Sessions never kept."""
    exclude_rigs: list[str] = []
    """Rig names never kept."""


class Qc(_Model):
    """Every threshold of :class:`zephyr.session_selection.Settings`, all required."""

    window_s: float = Field(gt=0)
    min_span_s: float = Field(gt=0)
    max_outside: float = Field(ge=0, le=1)
    max_breaks: float = Field(ge=0, le=1)
    outlier_z: float = Field(gt=0)
    min_snr_gap_db: float = Field(ge=0)
    snr_ceiling_db: float
    min_amplitude: float = Field(ge=0)
    artifact_z: float = Field(gt=0)
    artifact_pad_s: float = Field(ge=0)

    def settings(self) -> Settings:
        return Settings(**{f.name: getattr(self, f.name) for f in fields(Settings)})


class Select(_Model):
    """The final rule: how many clips of what length each passing session gives."""

    clip_s: float = Field(gt=0)
    clips_per_session: int = Field(ge=1)
    seed: int
    """Global seed; each session draws from ``seed`` and its session_id."""
    require_cameras: bool = True
    """Keep only sessions with every one of ``Source.cameras``."""


class Clips(_Model):
    """How ``clips`` cuts the selected spans."""

    thermistor_pad_s: float = Field(10.0, ge=0)
    """Thermistor kept beyond the clip's videos at each end, so the event detector's
    filter settles before the first frame."""


class Folds(_Model):
    """How ``folds`` splits the selected sessions (why: the package README). Animals are
    held out whole (test A, validation); test B holds out sessions of training animals;
    ``group_c`` animals are held out by session only."""

    seed: int
    camera: str
    """Camera whose frame rate defines group C."""
    test_animals: int = Field(ge=1)
    val_animals: int = Field(ge=1)
    min_sessions: int = Field(ge=1)
    """Sessions an animal needs to be held out whole."""
    cap_train: int = Field(ge=1)
    cap_val: int = Field(ge=1)
    cap_test: int = Field(ge=1)
    cap_test_b: int = Field(ge=0)
    """Most sessions per animal and rig (spread over its dates) in each set."""
    test_b_one_in: int = Field(ge=1)
    """Test B takes at most one session in this many of an animal-rig pair ..."""
    train_keep: int = Field(ge=0)
    """... and leaves the pair at least this many for training."""
    tries: int = Field(ge=1)
    """Seeded draws of held-out animals; the best balanced one is kept."""
    rig_families: dict[str, str] = {}
    """Rig names that are one physical rig (``FRG.4-B`` -> ``4B``); others stand alone."""
    group_c_min_hz: float | None = None
    """Sessions whose camera runs faster than this form group C ..."""
    group_c: dict[str, tuple[int, int, int]] = {}
    """... split per animal into (train, val, test) sessions."""


class Review(_Model):
    """A person's decision on one session, written over the suggestions by ``qc`` (with
    ``reviewed = true``); fields left out keep what QC wrote."""

    session_id: str
    selection: Literal["include", "exclude"] | None = None
    excluded: list[tuple[float, float]] | None = None
    included: list[tuple[float, float]] | None = None


class DatasetSpec(_Model):
    root: Root
    """The dataset folder (see :class:`Layout`), local or remote."""
    source: Source = Source()
    query: Query = Query()
    preprocess: PreprocessParams
    """Written as the clip list's ``[preprocess]``."""
    events: EventsSource
    """The clip list's detector (``[events]``)."""
    qc: Qc
    select: Select
    clips: Clips = Clips()
    folds: Folds | None = None
    """The train / validation / test assignment (``folds`` refuses to run without)."""
    review: list[Review] = []
    """Reviews made by hand, so a rebuild from nothing gives the same corpus."""


class Layout:
    """Paths inside a dataset folder (:class:`upath.UPath`, local or remote)."""

    def __init__(self, root) -> None:
        self.root = storage.root(root)
        self.docdb = self.root / "docdb.json.gz"
        self.inventory = self.root / "inventory.parquet"
        self.thermistor_dir = self.root / "thermistor"
        self.session_list = self.root / "sessions.toml"
        self.qc = self.root / "qc.parquet"
        self.results = self.root / "results.json"
        self.video_dir = self.root / "videos"
        self.cameras = self.root / "cameras.parquet"
        self.clip_dir = self.root / "clips"
        self.clip_table = self.root / "clips.parquet"
        self.clip_list = self.root / "clips.toml"
        self.thumbnail_dir = self.root / "thumbnails"
        self.folds = self.root / "folds.parquet"

    @property
    def local(self) -> bool:
        return storage.is_local(self.root)

    def thermistor(self, session_id: str) -> UPath:
        """The whole session's thermistor."""
        return self.thermistor_dir / f"{session_id}.parquet"

    def timestamps(self, session_id: str, camera: str) -> UPath:
        """A whole session's repaired camera timestamps."""
        return self.video_dir / f"{session_id}_{camera}.parquet"

    def camera_metadata(self, session_id: str, camera: str) -> UPath:
        """A camera's ``metadata.csv`` as on S3, unmodified (a cache)."""
        return self.root / "camera_metadata" / f"{session_id}_{camera}.parquet"

    def clip(self, session_id: str, span: str, part: str, suffix: str) -> UPath:
        """A clip's file: *part* is a camera or ``thermistor``."""
        return self.clip_dir / f"{session_id}_{span}_{part}{suffix}"

    def thumbnail(self, session_id: str, camera: str) -> UPath:
        """A session's camera at full resolution (first frame of its first clip)."""
        return self.thumbnail_dir / f"{session_id}_{camera}.png"

    def sessions(self) -> list[str]:
        """Session ids with a fetched thermistor, sorted."""
        return sorted(
            n.removesuffix(".parquet")
            for n in storage.names(self.thermistor_dir, "*.parquet")
        )


def session_of(thermistor: Path | str) -> str:
    """The session_id of a session's thermistor file."""
    return Path(str(thermistor)).stem
