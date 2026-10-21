"""Train and score every benchmarked network through zephyr's own run and evaluate.

:data:`RUNNER` hands :func:`zephyr.run.main` the benchmark config models, the
architecture each fold's ``train_params.arch`` names, and a loader that rebuilds any of
the networks, so every network trains and scores exactly as zephyr does, each on its
own paper's loss.
"""

from pathlib import Path

import torch

from zephyr.evaluate import checkpoint_stats
from zephyr.model import BreathingLoss
from zephyr.run import Runner
from zephyr.train import ZEPHYR, Architecture

from . import nets, transformer_losses
from .config import BenchmarkExperiment, BenchmarkFold, BenchmarkTrainParams


def _build(params: BenchmarkTrainParams, mean: float, std: float, **kwargs):
    return nets.build_model(
        params.arch, dropout=params.dropout, mean=mean, std=std, **kwargs
    )


def _criterion(params: BenchmarkTrainParams, fs: float) -> torch.nn.Module:
    """Each network's own paper's loss; PhysNet's (negative Pearson) is zephyr's at
    ``scales = [1]``, without the onset term (``w_onset`` is 0 for every network but
    zephyr)."""
    if params.arch == "tscan":
        return nets.DerivativeMSELoss()
    if params.arch == "physformer":
        return transformer_losses.PhysFormerLoss(fs, params.epochs)
    if params.arch == "rhythmformer":
        return transformer_losses.RhythmFormerLoss(fs)
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
) -> tuple[torch.nn.Module, float, float, dict]:
    """:func:`zephyr.evaluate.load_checkpoint` for any network in :data:`nets.ARCHS`."""
    state = torch.load(path, map_location=device, weights_only=False)
    mean, std = checkpoint_stats(state)
    model = nets.build_model(
        state.get("arch", "zephyr"), mean=mean, std=std, **state.get("arch_kwargs", {})
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, mean, std, state


RUNNER = Runner(
    experiment=BenchmarkExperiment,
    architecture=architecture,
    loader=load_checkpoint,
)
