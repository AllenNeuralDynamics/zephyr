"""zephyr's config files, extended with the network a fold trains.

Every key zephyr's files take still means the same; ``train_params.arch`` names the
network (``zephyr`` when omitted, so every zephyr fold loads here unchanged) and
``train_params.tscan_img_size`` is TS-CAN's input size.
"""

from typing import Literal

from pydantic import Field, PositiveInt, model_validator

from zephyr.channels import ChannelSet
from zephyr.config import Experiment, Fold, TrainParams


class BenchmarkTrainParams(TrainParams):
    """zephyr's training knobs plus the network to train."""

    # The same names as nets.ARCHS (a test keeps them equal); not imported so
    # that this module loads without torch.
    arch: Literal["zephyr", "tscan", "physnet", "physnet_event"] = "zephyr"
    tscan_img_size: PositiveInt = 36

    @model_validator(mode="after")
    def _baseline_archs_read_gray(self) -> "BenchmarkTrainParams":
        if self.arch != "zephyr" and ChannelSet.parse(self.channels).names != ("gray",):
            raise ValueError(f"arch {self.arch!r} needs channels = 'gray'")
        if self.arch != "zephyr" and (
            self.signal_pool != 1 or self.onset_input != "full"
        ):
            raise ValueError(
                f"signal_pool and onset_input are zephyr options; arch {self.arch!r}"
            )
        return self


class BenchmarkFold(Fold):
    train_params: BenchmarkTrainParams = BenchmarkTrainParams()


class BenchmarkExperiment(Experiment):
    fold: list[BenchmarkFold] = Field(min_length=1)
