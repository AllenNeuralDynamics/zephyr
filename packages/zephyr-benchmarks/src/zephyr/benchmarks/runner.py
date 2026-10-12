"""Train and score every benchmarked network through zephyr's own run and evaluate.

:data:`RUNNER` hands :func:`zephyr.run.main` the benchmark config models, the
architecture each fold's ``train_params.arch`` names, and a loader that rebuilds any of
the networks, so TS-CAN and PhysNet train and score exactly as zephyr does.
"""

from pathlib import Path

import numpy as np
import torch

from zephyr.evaluate import checkpoint_channels
from zephyr.model import BreathingLoss
from zephyr.run import Runner
from zephyr.train import ZEPHYR, Architecture

from . import nets
from .config import BenchmarkExperiment, BenchmarkFold, BenchmarkTrainParams


def _build(channels, params: BenchmarkTrainParams, mean, std, **kwargs):
    return nets.build_model(
        params.arch, channels, dropout=params.dropout, mean=mean, std=std, **kwargs
    )


def _criterion(params: BenchmarkTrainParams) -> torch.nn.Module:
    if params.arch == "tscan":
        return nets.DerivativeMSELoss()
    return BreathingLoss(params.w_corr, params.w_onset, tuple(params.scales))


def _kwargs(params: BenchmarkTrainParams) -> dict:
    return {"img_size": params.tscan_img_size} if params.arch == "tscan" else {}


def architecture(fold: BenchmarkFold) -> Architecture:
    """The network *fold* trains: zephyr's own, or a comparison network."""
    if fold.train_params.arch == "zephyr":
        return ZEPHYR
    return Architecture(
        name=fold.train_params.arch, build=_build, criterion=_criterion, kwargs=_kwargs
    )


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, dict]:
    """:func:`zephyr.evaluate.load_checkpoint` for any network in :data:`nets.ARCHS`."""
    state = torch.load(path, map_location=device, weights_only=False)
    model = nets.build_model(
        state.get("arch", "zephyr"),
        checkpoint_channels(state),
        mean=state["mean"],
        std=state["std"],
        **state.get("arch_kwargs", {}),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state["mean"], state["std"], state


RUNNER = Runner(
    experiment=BenchmarkExperiment,
    architecture=architecture,
    loader=load_checkpoint,
)
