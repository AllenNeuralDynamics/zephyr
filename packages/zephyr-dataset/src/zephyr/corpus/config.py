"""The dataset spec (one TOML file) and the layout of the dataset folder it names.

Every setting is written out in the spec, QC thresholds included, so a dataset never
changes when a default of zephyr's does. ``root`` is relative to the spec file.
"""

from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from zephyr.config import EventsSource, PreprocessParams, RelPath
from zephyr.session_selection import Settings


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


class Review(_Model):
    """A person's decision on one session, written over the suggestions by ``qc`` (with
    ``reviewed = true``); fields left out keep what QC wrote."""

    session_id: str
    selection: Literal["include", "exclude"] | None = None
    excluded: list[tuple[float, float]] | None = None
    included: list[tuple[float, float]] | None = None


class DatasetSpec(_Model):
    root: RelPath
    """The dataset folder (see :class:`Layout`)."""
    source: Source = Source()
    query: Query = Query()
    preprocess: PreprocessParams
    """Written as the clip list's ``[preprocess]``."""
    events: EventsSource
    """The clip list's detector (``[events]``)."""
    qc: Qc
    select: Select
    review: list[Review] = []
    """Reviews made by hand, so a rebuild from nothing gives the same corpus."""


class Layout:
    """Paths inside a dataset folder."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.docdb = self.root / "docdb.json.gz"
        self.inventory = self.root / "inventory.parquet"
        self.thermistor_dir = self.root / "thermistor"
        self.clips = self.root / "sessions.toml"
        self.qc = self.root / "qc.parquet"
        self.results = self.root / "results.json"
        self.video_dir = self.root / "videos"
        self.cameras = self.root / "cameras.parquet"

    def thermistor(self, session_id: str) -> Path:
        """The whole session's thermistor."""
        return self.thermistor_dir / f"{session_id}.parquet"

    def timestamps(self, session_id: str, camera: str) -> Path:
        """A whole session's repaired camera timestamps."""
        return self.video_dir / f"{session_id}_{camera}.parquet"

    def camera_metadata(self, session_id: str, camera: str) -> Path:
        """A camera's ``metadata.csv`` as on S3, unmodified (a cache)."""
        return self.root / "camera_metadata" / f"{session_id}_{camera}.parquet"

    def sessions(self) -> list[str]:
        """Session ids with a fetched thermistor, sorted."""
        return sorted(p.stem for p in self.thermistor_dir.glob("*.parquet"))


def session_of(thermistor: Path | str) -> str:
    """The session_id of a session's thermistor file."""
    return Path(str(thermistor)).stem
