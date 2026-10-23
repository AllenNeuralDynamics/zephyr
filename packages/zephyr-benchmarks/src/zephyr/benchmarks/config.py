"""zephyr's config files, extended with the network a fold trains.

Every key zephyr's files take still means the same; ``train_params.arch`` names the
network (``zephyr`` when omitted, so every zephyr fold loads here unchanged) and
``train_params.tscan_img_size`` is TS-CAN's input size. :class:`ResultsManifest` is
which finished runs a report compares (``configs/results/``).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from zephyr.config import Experiment, Fold, TrainParams


class BenchmarkTrainParams(TrainParams):
    """zephyr's training knobs plus the network to train."""

    # The same names as nets.ARCHS (a test keeps them equal); not imported so
    # that this module loads without torch.
    arch: Literal["zephyr", "tscan", "physnet", "physformer", "rhythmformer"] = "zephyr"
    tscan_img_size: PositiveInt = 36

    @model_validator(mode="after")
    def _zephyr_options(self) -> "BenchmarkTrainParams":
        if self.arch != "zephyr" and (
            self.signal_pool != 1 or self.onset_input != "full"
        ):
            raise ValueError(
                f"signal_pool and onset_input are zephyr options; arch {self.arch!r}"
            )
        if self.arch != "zephyr" and self.w_onset != 0:
            raise ValueError(
                f"only zephyr has an onset head; arch {self.arch!r} needs w_onset = 0"
            )
        if self.arch != "zephyr" and self.w_phase != 0:
            raise ValueError(
                f"only zephyr has a phase head; arch {self.arch!r} needs w_phase = 0"
            )
        return self


class BenchmarkFold(Fold):
    train_params: BenchmarkTrainParams = BenchmarkTrainParams()


class BenchmarkExperiment(Experiment):
    fold: list[BenchmarkFold] = Field(min_length=1)


class ReportedRun(BaseModel):
    """One method as a report shows it: the run folder it was uploaded to."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str
    run: str
    """Folder under the manifest's ``root``: ``<run>/<fold>/seed-<n>/`` for a network,
    ``<run>/<fold>/{pixel,facemap}/`` for a baseline."""
    kind: Literal["network", "facemap", "pixel"] = "network"
    variant: str = ""
    """Facemap ``both``/``motion``/``movie``; pixel ``<method>/<blind|calibrated>``."""
    head: bool = False
    """Whether it is a zephyr network trained with its onset head (its inhale events come
    from the head, not DSP); no other network has one."""
    fold: str = ""
    """The run's fold folder, when it is not the manifest's ``fold``: ``<run>/<fold>/``
    for a network, ``<run>/<fold>/<kind>/<variant>/`` for a baseline."""

    @model_validator(mode="after")
    def _variant_names_a_baseline(self) -> "ReportedRun":
        if (self.kind == "network") == bool(self.variant):
            raise ValueError(f"{self.label}: only a facemap or pixel run has a variant")
        if self.head and self.kind != "network":
            raise ValueError(f"{self.label}: only a network has an onset head")
        return self


class CrossValidationRuns(BaseModel):
    """The methods retrained on each cross-validation fold: every ``method`` is read
    from each of ``folds`` (its ``fold`` is set per fold)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    folds: list[str] = Field(min_length=1)
    method: list[ReportedRun] = Field(min_length=1)
    variant: list[ReportedRun] = []
    """The first method trained differently on each fold, compared with it alone."""

    @model_validator(mode="after")
    def _labels_are_distinct(self) -> "CrossValidationRuns":
        labels = [r.label for r in (*self.method, *self.variant)]
        if len(set(labels)) != len(labels):
            raise ValueError(f"repeated labels in {labels}")
        if any(r.fold for r in (*self.method, *self.variant)):
            raise ValueError("a cross-validation method takes its fold from folds")
        return self


class ResultsManifest(BaseModel):
    """Which runs a report compares, under which labels (``configs/results/``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    root: str
    """Where the runs were uploaded: an ``s3://`` prefix or a local folder."""
    fold: str = "dev"
    method: list[ReportedRun] = Field(min_length=1)
    """The compared methods, in the order tables list them."""
    variant: list[ReportedRun] = []
    """The first method trained differently, compared with it alone."""
    learning_curve: list[ReportedRun] = []
    """The first method trained on parts of its training set, scored on the same tests."""
    cross_validation: CrossValidationRuns | None = None
    """Methods retrained on each cross-validation fold (labels as in ``method``)."""

    @model_validator(mode="after")
    def _labels_are_distinct(self) -> "ResultsManifest":
        labels = [r.label for r in (*self.method, *self.variant, *self.learning_curve)]
        if len(set(labels)) != len(labels):
            raise ValueError(f"repeated labels in {labels}")
        return self
