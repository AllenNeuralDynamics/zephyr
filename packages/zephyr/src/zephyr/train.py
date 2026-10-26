"""Train one fold with one seed, validating on a fold's ``val`` clips or a time tail.

With validation (``[fold.val]``, scored whole, or ``val_fraction > 0`` of every training
clip) each epoch reports ``val loss/corr`` (windowed, like the training loss) and, every
``score_every`` epochs, ``xcorr``/``f1`` from full-clip predictions, which select the
checkpoint and stop the run early; ``head_thresholds`` then picks ``best.pt``'s onset
threshold on the ``val`` clips. Everything that shapes
a run comes from a :class:`~zephyr.config.Fold`; the seed and machine are the caller's
(see :mod:`zephyr.run`). The network is :data:`ZEPHYR` unless a caller passes another
:class:`Architecture`.
"""

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from zephyr.augment import AugmentConfig
from zephyr.model import BreathingLoss, BreathingNet

from . import features
from .config import OUTPUT_FS, Fold, TrainParams, canonical, dump
from .dataset import (
    EpochRangeSampler,
    WindowDataset,
    frame_stats,
    stats_path,
)
from .evaluate import (
    HEAD_THRESHOLD,
    checkpoint_stats,
    choose_head_threshold,
    load_checkpoint,
)
from .evaluation import EVENT_TOLERANCE_FRAMES
from .features import ClipEntry
from .infer import predict_clip
from .signal import detect_inhalation_events
from .validation import event_f1, zero_lag_correlation


@dataclass(frozen=True)
class Machine:
    """Where a run executes; deliberately not part of a fold's science."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: str = "bf16"
    num_workers: int = 4

    @property
    def amp_dtype(self) -> torch.dtype | None:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[self.amp]


@dataclass(frozen=True)
class Architecture:
    """The network :func:`train_fold` trains and the loss it trains with.

    Anything with :class:`~zephyr.model.BreathingNet`'s interface fits.  *name* is
    recorded in every checkpoint as ``arch`` and *kwargs* as ``arch_kwargs``, so a
    loader can rebuild the network.
    """

    name: str
    build: Callable[..., torch.nn.Module]
    """``build(params, mean, std, **kwargs)``, *mean*/*std* the gray statistics."""
    criterion: Callable[[TrainParams, float], torch.nn.Module]
    """``criterion(params, fs)``, *fs* the output rate in Hz. A loss that changes over
    training defines ``set_epoch(epoch)``, called before each epoch (0-based)."""
    kwargs: Callable[[TrainParams], dict] = lambda params: {}


def _zephyr_kwargs(params: TrainParams) -> dict:
    """Network options that differ from the default, as ``arch_kwargs``; only those
    are recorded, so checkpoints of the default network are written as before."""
    kwargs: dict = {}
    if params.signal_pool > 1:
        kwargs["signal_pool"] = params.signal_pool
    if params.onset_input != "full":
        kwargs["onset_input"] = params.onset_input
    if params.temporal_kernel > 1:
        kwargs["temporal_kernel"] = params.temporal_kernel
    if params.w_phase > 0:
        kwargs["phase_head"] = True
    return kwargs


ZEPHYR = Architecture(
    name="zephyr",
    build=lambda params, mean, std, **kwargs: BreathingNet(
        dropout=params.dropout, **kwargs
    ),
    criterion=lambda params, fs: BreathingLoss(
        params.w_corr, params.w_onset, tuple(params.scales), params.w_phase
    ),
    kwargs=_zephyr_kwargs,
)


def snapshot_path(run_dir: Path, epoch: int, save_every: int) -> Path | None:
    """Where the weights after 0-based *epoch* are kept, or ``None``.

    Every *save_every*-th epoch is kept as ``epoch-<n>.pt`` (``n`` 1-based), with
    the weights a final ``best.pt`` would hold, so ``evaluate`` scores it as is.
    """
    if save_every and (epoch + 1) % save_every == 0:
        return run_dir / f"epoch-{epoch + 1:03d}.pt"
    return None


@dataclass
class Selection:
    """The best validation xcorr so far and the scorings since, which pick ``best.pt``
    and stop a run early. The first score of the averaged (EMA) weights starts afresh:
    the live weights are noisier and can spike above anything the averaged ones reach
    for many epochs, so a live best is no bar for them."""

    best: float = -np.inf
    since_best: int = 0
    on_ema: bool = False
    """Whether ``best`` was scored on the averaged weights."""

    def offer(self, xcorr: float, ema: bool) -> bool:
        """Count a scoring of the live (``ema=False``) or averaged weights; ``True`` if
        it is the new best."""
        if ema and not self.on_ema:
            self.best, self.since_best, self.on_ema = -np.inf, 0, True
        if xcorr > self.best:
            self.best, self.since_best = xcorr, 0
            return True
        self.since_best += 1
        return False


@dataclass(frozen=True)
class ScoredClip:
    """One validation clip as a whole-clip scoring predicted it, for a caller watching
    training (:func:`train_fold`'s *on_scored*): arrays on the clip's output grid."""

    entry: ClipEntry
    times: np.ndarray
    truth: np.ndarray
    signal: np.ndarray
    onset: np.ndarray
    span: tuple[float, float]
    """The scored fraction of the clip; only frames inside it reach the metrics."""
    scores: dict


OnScored = Callable[[int, list[ScoredClip]], None]
"""``on_scored(epoch, clips)``, called after each whole-clip scoring (0-based epoch)
with the predictions of the weights selection judged."""


def _mean(values: list[dict], key: str) -> float:
    return float(np.mean([v[key] for v in values])) if values else float("nan")


TARGETS = ("signal", "onset", "phase", "phase_mask")
"""The batch's targets, which :func:`_loss` reads."""


def _keys(values: list[dict]) -> tuple[str, ...]:
    """The loss terms a history row reports: ``phase`` only with a phase head."""
    return ("loss", "corr", "onset") + (
        ("phase",) if values and "phase" in values[0] else ()
    )


def _loss(
    criterion: torch.nn.Module, outputs: tuple, targets: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    """*criterion* on a network's outputs, in fp32, and the batch's *targets*; a
    network returning a phase after its signal and onset is scored on the phase
    target too."""
    signal, onset, *extra = (output.float() for output in outputs)
    phase = {}
    if extra:
        phase = {
            "phase_pred": extra[0],
            "phase_true": targets["phase"],
            "phase_mask": targets["phase_mask"],
        }
    return criterion(signal, onset, targets["signal"], targets["onset"], **phase)


@torch.no_grad()
def validate_windows(
    model: BreathingNet,
    loader: DataLoader,
    criterion: BreathingLoss,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    """Windowed validation loss, computed exactly like the training loss."""
    model.eval()
    parts: list[dict] = []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        t_in = batch["t_in"].to(device, non_blocking=True)
        t_out = batch["t_out"].to(device, non_blocking=True)
        targets = {k: batch[k].to(device, non_blocking=True) for k in TARGETS}
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                _, stats = _loss(criterion, model(features, t_in, t_out), targets)
        else:
            _, stats = _loss(criterion, model(features, t_in, t_out), targets)
        parts.append(stats)
    return {f"val_{k}": _mean(parts, k) for k in _keys(parts)}


def score_full_clips(
    model: BreathingNet,
    entries: list[ClipEntry],
    mean: float,
    std: float,
    device: torch.device,
    *,
    window: int,
    frame_chunk: int,
    amp_dtype: torch.dtype | None,
    fs: float = OUTPUT_FS,
    span: tuple[float, float] = (0.0, 1.0),
    keep: list[ScoredClip] | None = None,
) -> tuple[dict[str, float], list[dict]]:
    """Predict each clip end to end for local training diagnostics (each prediction
    also appended to *keep* when given).

    *span* restricts which fraction of each clip is *scored*.  The whole clip is
    still predicted -- inference is cheap, and it keeps every scored sample's
    receptive field fully populated with real context rather than zero-padding
    at the span boundary -- but only frames inside the span reach the metrics.
    Under ``val_fraction`` that is what keeps trained-on frames out of the
    reported number.
    """
    rows: list[dict] = []
    for entry in entries:
        pred, onset = predict_clip(
            model,
            entry,
            mean,
            std,
            window=window,
            device=device,
            frame_chunk=frame_chunk,
            amp_dtype=amp_dtype,
        )
        truth = pd.read_parquet(entry.target)["signal"].to_numpy(np.float64)
        n = min(len(truth), len(pred))
        whole_truth, whole_pred = truth[:n], pred[:n]
        truth, pred_n = truth[:n], pred[:n].astype(np.float64)
        if span != (0.0, 1.0):
            lo, hi = int(n * span[0]), int(n * span[1])
            truth, pred_n = truth[lo:hi], pred_n[lo:hi]

        corr = zero_lag_correlation(truth, pred_n)
        tolerance = EVENT_TOLERANCE_FRAMES / fs
        truth_on, truth_off = detect_inhalation_events(truth, fs)
        pred_on, pred_off = detect_inhalation_events(pred_n, fs)
        rows.append(
            {
                "clip_id": entry.clip_id,
                "recording": entry.recording,
                "correlation": corr,
                "inhale_f1": event_f1(truth_on / fs, pred_on / fs, tolerance),
                "exhale_f1": event_f1(truth_off / fs, pred_off / fs, tolerance),
                "n_truth_onsets": len(truth_on),
                "n_pred_onsets": len(pred_on),
            }
        )
        if keep is not None:
            keep.append(
                ScoredClip(
                    entry,
                    np.load(entry.times)[:n],
                    whole_truth,
                    whole_pred,
                    onset[:n],
                    span,
                    rows[-1],
                )
            )
    summary = {
        "xcorr": _mean(rows, "correlation"),
        "inhale_f1": _mean(rows, "inhale_f1"),
        "exhale_f1": _mean(rows, "exhale_f1"),
    }
    return summary, rows


def settle_head_threshold(
    path: Path,
    entries: list[ClipEntry] | None,
    params: TrainParams,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict:
    """Choose the onset threshold of the checkpoint at *path* on its validation clips
    *entries* (``None``: the fold has none, so it keeps :data:`HEAD_THRESHOLD`) over
    ``params.head_thresholds``, and store it in the checkpoint as ``head_threshold``."""
    model, mean, std, state = load_checkpoint(path, device)
    if entries is None:
        chosen = {
            "value": HEAD_THRESHOLD,
            "chosen_on": None,
            "rule": "the default: the fold has no validation clips",
        }
    else:
        clips = []
        for entry in entries:
            _, onset = predict_clip(
                model,
                entry,
                mean,
                std,
                window=params.infer_window,
                device=device,
                frame_chunk=params.frame_chunk,
                amp_dtype=amp_dtype,
            )
            times = np.load(entry.times)
            n = min(len(times), len(onset))
            clips.append((entry, times[:n], onset[:n]))
        chosen = choose_head_threshold(
            clips, params.head_thresholds, params.head_threshold_f1_tie
        )
    state["head_threshold"] = chosen
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return chosen


def cosine_schedule(step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay to 1% of the peak learning rate."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def augment_config(params: TrainParams) -> AugmentConfig:
    return AugmentConfig(**params.augmentation.model_dump())


def train_fold(
    fold: Fold,
    seed: int,
    run_dir: Path,
    features_dir: Path,
    machine: Machine,
    *,
    resume: bool = False,
    architecture: Architecture = ZEPHYR,
    on_scored: OnScored | None = None,
) -> Path:
    """Train *fold* with *seed* into *run_dir*; return the path of ``best.pt``
    (*on_scored*, if given, sees each whole-clip scoring's predictions).

    Writes ``config.json`` (every input model dumped with absolute paths) and
    ``train_videos.json`` into *run_dir*, and the same two records into every
    checkpoint, so a checkpoint says exactly what it was trained on --
    :mod:`zephyr.evaluate` refuses to score anything on that list.
    """
    params = fold.train_params
    torch.manual_seed(seed)
    device = torch.device(machine.device)
    amp_dtype = machine.amp_dtype
    recipe = fold.preprocess

    train_lists = fold.train_lists()
    sources = [
        features.require(clips, recipe, features_dir) for clips in fold.train_clips()
    ]
    weights = [source.weight for source in fold.train]
    train_entries = [e for source in sources for e in source]
    train_videos = [canonical(e.video) for e in train_entries]
    resolved = {
        "seed": seed,
        "fold": dump(fold),
        "train_clip_lists": [dump(lst) for lst in train_lists],
        "test_clip_lists": {n: dump(lst) for n, lst in fold.test_lists().items()},
    }
    if fold.val is not None:
        resolved["val_clip_list"] = dump(fold.val_list())

    init = None
    inherited_videos: list[str] = []
    lineage_known = True
    if fold.init_from is not None:
        init = torch.load(fold.init_from, map_location=device, weights_only=False)
        init_arch = init.get("arch", "zephyr")
        if init_arch != architecture.name:
            raise SystemExit(
                f"init_from {fold.init_from} is {init_arch}, "
                f"this fold is {architecture.name}"
            )
        # The weights expect the statistics they were trained with, so a
        # fine-tune keeps them rather than re-measuring on its own clips.
        mean, std = checkpoint_stats(init)
        # A fine-tune has also seen everything its parent saw.  A parent with
        # no record leaves this run's lineage unknown, which evaluate warns on.
        if init.get("train_videos") is None:
            lineage_known = False
        else:
            lineage_known = init.get("lineage_known", True)
            inherited_videos = sorted(
                {*init["train_videos"], *init.get("inherited_videos", ())}
            )
    else:
        mean, std = frame_stats(train_entries, stats_path(features_dir, train_entries))

    # Every training clip trains.  Validation, if any, is the fold's val clips
    # whole, or else the time-tail of the training clips -- see val_fraction.
    if fold.val is not None:
        val_entries = features.require(fold.val_clips(), recipe, features_dir)
        train_span, val_span = (0.0, 1.0), (0.0, 1.0)
        n_val = len({e.recording for e in val_entries})
        tail = f"validating on {len(val_entries)} val clips / {n_val} recordings"
    else:
        val_entries = train_entries
        train_span = (0.0, 1.0 - params.val_fraction)
        val_span = (1.0 - params.val_fraction, 1.0)
        tail = f"validating on the last {params.val_fraction:.0%} of each clip"
    scoring = fold.val is not None or params.val_fraction > 0
    if not scoring:
        tail = f"NO validation -- fixed {params.epochs} epochs, no early stopping"
    n_recordings = len({e.recording for e in train_entries})
    print(
        f"train {len(train_entries)} clips / {n_recordings} recordings in "
        f"{len(sources)} list(s) (weights {weights})  |  {tail}",
        flush=True,
    )
    print(f"gray mean {mean:.2f}  std {std:.2f}", flush=True)

    augment = augment_config(params)
    print(
        f"augmentation: {'on -> ' + str(augment) if augment.enabled else 'off'}",
        flush=True,
    )

    horizon = params.epochs * params.steps_per_epoch * params.batch_size
    grids = {
        "select_fs": recipe.select_fs,
        "output_fs": recipe.output_fs,
        "onset_sigma_s": recipe.onset_sigma_s,
    }
    train_set = WindowDataset(
        sources,
        window=params.window,
        mean=mean,
        std=std,
        length=horizon,
        seed=seed,
        augment=augment,
        span=train_span,
        weights=weights,
        rate_balance=params.rate_balance,
        rate_bins_hz=params.rate_bins_hz,
        **grids,
    )
    # Gridded and non-overlapping, so the windowed number is comparable epoch to
    # epoch; capped because a full grid over every held-out clip is more forward
    # passes than the signal in the number justifies.
    val_set = (
        WindowDataset(
            [val_entries],
            window=params.window,
            mean=mean,
            std=std,
            stride=params.window,
            span=val_span,
            **grids,
        )
        if scoring
        else None
    )
    if val_set is not None and len(val_set) > params.val_windows:
        val_set.index = val_set.index[:: len(val_set) // params.val_windows + 1]

    sampler = EpochRangeSampler(params.steps_per_epoch * params.batch_size)
    common = {
        "num_workers": machine.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": machine.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set, batch_size=params.batch_size, sampler=sampler, **common
    )
    val_loader = (
        DataLoader(val_set, batch_size=params.batch_size, shuffle=False, **common)
        if val_set is not None
        else None
    )

    arch_kwargs = architecture.kwargs(params)
    model = architecture.build(params, mean, std, **arch_kwargs).to(device)
    if init is not None:
        model.load_state_dict(init["model"])
        print(f"initialised from {fold.init_from} (epoch {init.get('epoch')})")
    criterion = architecture.criterion(params, recipe.output_fs)
    set_epoch = getattr(criterion, "set_epoch", None)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )
    total_steps = params.epochs * params.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: cosine_schedule(s, total_steps, params.warmup_steps)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is torch.float16)

    # The averaged weights start life anchored to the random initialisation and
    # relax toward the trained ones with a time constant of 1/(1-decay) steps --
    # 1000 at the 0.999 default.  Selecting on them before that has settled picks
    # a checkpoint that is mostly noise: measured, a 2-epoch (400-step) run scored
    # +0.118 on EMA against +0.716 live.  Three time constants is ~95% relaxed.
    ema_ready_step = int(3.0 / (1.0 - params.ema_decay)) if params.ema_decay > 0 else 0
    global_step = 0

    ema = None
    if params.ema_decay > 0:
        ema = AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(params.ema_decay),
            use_buffers=True,
        )

    if ema is not None:
        print(
            f"EMA decay {params.ema_decay}: selection switches to the averaged "
            f"weights after {ema_ready_step} steps "
            f"(~epoch {ema_ready_step // params.steps_per_epoch + 1}); "
            f"until then the log marks them '~'",
            flush=True,
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"{n_params / 1e6:.2f} M params | receptive field {model.receptive_field} "
        f"frames ({model.receptive_field / recipe.output_fs:.1f} s) | window {params.window} "
        f"x batch {params.batch_size} | amp {machine.amp}",
        flush=True,
    )

    if run_dir.exists() and not resume and any(run_dir.iterdir()):
        raise SystemExit(
            f"{run_dir} already exists and is not empty; resume that exact run "
            "or choose another output directory"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2))
    (run_dir / "train_videos.json").write_text(json.dumps(train_videos, indent=2))

    def save(
        path: Path,
        epoch: int,
        metrics: dict,
        per_clip: list[dict],
        weights: torch.nn.Module | None = None,
    ) -> None:
        """Write a checkpoint.

        ``model`` holds whichever weights were selected -- the EMA copy when
        averaging is on -- so downstream loaders need no special case.  The live
        weights and the averaging state are stored alongside so that ``last.pt``
        resumes exactly: optimiser, scheduler, and a cosine schedule restarted
        mid-run would otherwise not be the same schedule.
        """
        state = {
            "model": (weights or model).state_dict(),
            "model_live": model.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "ema_decay": params.ema_decay,
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best": selection.best,
            "since_best": selection.since_best,
            "best_on_ema": selection.on_ema,
            "history": history,
            "global_step": global_step,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            # What this checkpoint trained on: evaluate refuses to score any
            # clip whose video is listed here.
            "train_videos": train_videos,
            "inherited_videos": inherited_videos,
            "lineage_known": lineage_known,
            "init_from": str(fold.init_from) if fold.init_from else None,
            "seed": seed,
            "arch": architecture.name,
            "arch_kwargs": arch_kwargs,
            # The gray statistics the network's input is standardised with.
            "mean": mean,
            "std": std,
            "params": dump(params),
            "config": resolved,
            "feature_config": dump(recipe) | {"output_fs_hz": recipe.output_fs},
            "metrics": metrics,
            "per_clip": per_clip,
        }
        # Keep the previous checkpoint intact until the replacement is fully
        # serialized. A process or machine failure during torch.save then costs
        # at most one epoch instead of corrupting the only resumable file.
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    selection = Selection()
    history: list[dict] = []
    start_epoch = 0

    last_path = run_dir / "last.pt"
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["train_videos"] != train_videos:
            raise SystemExit(
                f"{last_path} was trained on a different list of clips than "
                "this fold names -- refusing to mix them."
            )
        if state["arch"] != architecture.name:
            raise SystemExit(
                f"{last_path} was trained as {state['arch']} but this fold asks for "
                f"{architecture.name} -- it cannot be resumed."
            )
        model.load_state_dict(state.get("model_live") or state["model"])
        if ema is not None and state.get("ema") is not None:
            ema.load_state_dict(state["ema"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        selection = Selection(state["best"], state["since_best"], state["best_on_ema"])
        history = state["history"]
        start_epoch = state["epoch"] + 1
        global_step = state.get("global_step", start_epoch * params.steps_per_epoch)
        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in state["cuda_rng_state_all"]]
            )
        if state.get("numpy_rng_state") is not None:
            np.random.set_state(state["numpy_rng_state"])
        if state.get("python_rng_state") is not None:
            random.setstate(state["python_rng_state"])
        print(
            f"resumed from {last_path} at epoch {start_epoch} "
            f"(best xcorr {selection.best:+.4f})"
        )
    elif resume:
        print(f"resume requested but {last_path} does not exist; starting fresh")

    for epoch in range(start_epoch, params.epochs):
        sampler.epoch = epoch
        if set_epoch is not None:
            set_epoch(epoch)
        model.train()
        started = time.perf_counter()
        parts: list[dict] = []
        for batch in train_loader:
            features_ = batch["features"].to(device, non_blocking=True)
            t_in = batch["t_in"].to(device, non_blocking=True)
            t_out = batch["t_out"].to(device, non_blocking=True)
            targets = {k: batch[k].to(device, non_blocking=True) for k in TARGETS}

            optimiser.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    outputs = model(features_, t_in, t_out)
                # Losses in fp32: the correlation term normalises by a sum of
                # squares over 512 samples, which is exactly the kind of
                # reduction that loses precision in half.
                loss, stats = _loss(criterion, outputs, targets)
            else:
                loss, stats = _loss(criterion, model(features_, t_in, t_out), targets)

            scaler.scale(loss).backward()
            if params.grad_clip:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            global_step += 1
            if ema is not None:
                ema.update_parameters(model)
            parts.append(stats)

        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0]}
        row |= {f"train_{k}": _mean(parts, k) for k in _keys(parts)}
        if val_loader is not None:
            row |= validate_windows(model, val_loader, criterion, device, amp_dtype)

        scored = scoring and (
            (epoch + 1) % params.score_every == 0 or epoch == params.epochs - 1
        )
        if scored:
            live_kept: list[ScoredClip] | None = [] if on_scored else None
            summary, per_clip = score_full_clips(
                model,
                val_entries,
                mean,
                std,
                device,
                window=params.infer_window,
                frame_chunk=params.frame_chunk,
                amp_dtype=amp_dtype,
                fs=recipe.output_fs,
                span=val_span,
                keep=live_kept,
            )
            kept = live_kept
            row |= summary
            # Scoring a handful of clips costs seconds against a ~115 s epoch, so
            # evaluating the averaged weights as well is close to free -- and it
            # is the only way to know whether EMA actually helped.
            selected, selected_clips = model, per_clip
            on_ema = False
            if ema is not None and global_step < ema_ready_step:
                row["ema_warming"] = True
            if ema is not None:
                ema_kept: list[ScoredClip] | None = [] if on_scored else None
                ema_summary, ema_clips = score_full_clips(
                    ema.module,
                    val_entries,
                    mean,
                    std,
                    device,
                    window=params.infer_window,
                    frame_chunk=params.frame_chunk,
                    amp_dtype=amp_dtype,
                    fs=recipe.output_fs,
                    span=val_span,
                    keep=ema_kept,
                )
                row |= {f"ema_{k}": v for k, v in ema_summary.items()}
                # Select on the averaged weights -- they are what would ship --
                # but only once they have relaxed away from the initialisation.
                if global_step >= ema_ready_step:
                    summary, per_clip = ema_summary, ema_clips
                    selected, selected_clips = ema.module, ema_clips
                    kept = ema_kept
                    on_ema = True

            improved = selection.offer(summary["xcorr"], ema=on_ema)
            if improved:
                save(
                    run_dir / "best.pt",
                    epoch,
                    summary,
                    selected_clips,
                    weights=selected,
                )
                (run_dir / "best_per_clip.json").write_text(
                    json.dumps(selected_clips, indent=2)
                )
            row["since_best"] = selection.since_best
            if on_scored is not None:
                on_scored(epoch, kept)

        row["seconds"] = round(time.perf_counter() - started, 1)
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        # Written every epoch, scored or not, so a crash costs one epoch.
        save(last_path, epoch, row, [])
        snapshot = snapshot_path(run_dir, epoch, params.save_every)
        if snapshot is not None:
            save(snapshot, epoch, row, [], weights=ema.module if ema else None)

        message = (
            f"[{epoch + 1:3d}/{params.epochs}] train corr {row['train_corr']:+.3f}"
        )
        if "val_corr" in row:
            message += f"  val corr {row['val_corr']:+.3f}  loss {row['val_loss']:.4f}"
        else:
            message += f"  loss {row['train_loss']:.4f}"
        if scored:
            marker = " *" if improved else ""
            message += f"  |  xcorr {row['xcorr']:+.3f}"
            if "ema_xcorr" in row:
                message += f"  ema {row['ema_xcorr']:+.3f}"
                if row.get("ema_warming"):
                    message += "~"
            message += f"  inhale_f1 {row['inhale_f1']:.3f}{marker}"
        print(f"{message}  ({row['seconds']:.0f}s)", flush=True)

        if (
            params.patience
            and scored
            and selection.since_best >= params.patience
            and epoch + 1 >= params.min_epochs
        ):
            print(
                f"early stop: {selection.since_best} scored evaluations "
                f"({selection.since_best * params.score_every} epochs) without "
                f"improving on {selection.best:+.4f}",
                flush=True,
            )
            break

    if not scoring:
        # Nothing was ever scored, so there is no "best" to choose -- the final
        # weights are the deliverable.  Written as best.pt too so downstream
        # loaders need no special case.
        save(run_dir / "best.pt", epoch, {}, [], weights=ema.module if ema else None)
        print(f"final weights (no validation) -> {run_dir / 'best.pt'}")
    else:
        print(
            f"best held-out correlation {selection.best:+.4f} -> {run_dir / 'best.pt'}"
        )
    if params.head_thresholds and architecture.name == "zephyr" and params.w_onset > 0:
        chosen = settle_head_threshold(
            run_dir / "best.pt",
            val_entries if fold.val is not None else None,
            params,
            device,
            amp_dtype,
        )
        print(
            f"onset threshold {chosen['value']:.2f} "
            + (
                f"(validation inhale F1 {chosen['val_inhale_f1']:.3f}, over-count "
                f"{chosen['val_overcount_per_min']:+.2f}/min)"
                if chosen["chosen_on"]
                else "(default: no validation clips)"
            ),
            flush=True,
        )
    return run_dir / "best.pt"
