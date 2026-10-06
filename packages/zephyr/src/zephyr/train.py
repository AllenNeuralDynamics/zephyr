"""Train one fold with one seed, with an optional time-sliced validation tail.

With ``val_fraction > 0`` each epoch reports ``val loss/corr`` (cheap, windowed, like
the training loss) and, every ``score_every`` epochs, ``xcorr``/``f1`` from full-clip
predictions on the held-back tail, which selects the checkpoint. Everything that shapes
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
from zephyr.channels import CHANNEL_NAMES, ChannelSet
from zephyr.model import BreathingLoss, BreathingNet

from . import features
from .config import OUTPUT_FS, Fold, TrainParams, dump
from .dataset import (
    EpochRangeSampler,
    WindowDataset,
    channel_stats,
    stats_path,
)
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
    """``build(channels, params, mean, std, **kwargs)``; stats are full-width."""
    criterion: Callable[[TrainParams], torch.nn.Module]
    kwargs: Callable[[TrainParams], dict] = lambda params: {}


ZEPHYR = Architecture(
    name="zephyr",
    build=lambda channels, params, mean, std: BreathingNet(
        channels=channels, dropout=params.dropout
    ),
    criterion=lambda params: BreathingLoss(
        params.w_corr, params.w_onset, tuple(params.scales)
    ),
)


def _mean(values: list[dict], key: str) -> float:
    return float(np.mean([v[key] for v in values])) if values else float("nan")


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
        signal = batch["signal"].to(device, non_blocking=True)
        onset = batch["onset"].to(device, non_blocking=True)
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                pred_signal, pred_onset = model(features, t_in, t_out)
                _, stats = criterion(
                    pred_signal.float(), pred_onset.float(), signal, onset
                )
        else:
            pred_signal, pred_onset = model(features, t_in, t_out)
            _, stats = criterion(pred_signal, pred_onset, signal, onset)
        parts.append(stats)
    return {f"val_{k}": _mean(parts, k) for k in ("loss", "corr", "onset")}


def score_full_clips(
    model: BreathingNet,
    entries: list[ClipEntry],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    *,
    window: int,
    frame_chunk: int,
    amp_dtype: torch.dtype | None,
    fs: float = OUTPUT_FS,
    span: tuple[float, float] = (0.0, 1.0),
) -> tuple[dict[str, float], list[dict]]:
    """Predict each clip end to end for local training diagnostics.

    *span* restricts which fraction of each clip is *scored*.  The whole clip is
    still predicted -- inference is cheap, and it keeps every scored sample's
    receptive field fully populated with real context rather than zero-padding
    at the span boundary -- but only frames inside the span reach the metrics.
    Under ``val_fraction`` that is what keeps trained-on frames out of the
    reported number.
    """
    rows: list[dict] = []
    for entry in entries:
        pred, _ = predict_clip(
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
        truth, pred_n = truth[:n], pred[:n].astype(np.float64)
        if span != (0.0, 1.0):
            lo, hi = int(n * span[0]), int(n * span[1])
            truth, pred_n = truth[lo:hi], pred_n[lo:hi]

        corr = zero_lag_correlation(truth, pred_n)
        truth_on, truth_off = detect_inhalation_events(truth, fs)
        pred_on, pred_off = detect_inhalation_events(pred_n, fs)
        rows.append(
            {
                "clip_id": entry.clip_id,
                "recording": entry.recording,
                "correlation": corr,
                "inhale_f1": event_f1(truth_on / fs, pred_on / fs),
                "exhale_f1": event_f1(truth_off / fs, pred_off / fs),
                "n_truth_onsets": len(truth_on),
                "n_pred_onsets": len(pred_on),
            }
        )
    summary = {
        "xcorr": _mean(rows, "correlation"),
        "inhale_f1": _mean(rows, "inhale_f1"),
        "exhale_f1": _mean(rows, "exhale_f1"),
    }
    return summary, rows


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
) -> Path:
    """Train *fold* with *seed* into *run_dir*; return the path of ``best.pt``.

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
    train_videos = [str(e.video) for e in train_entries]
    train_recordings = sorted({e.recording for e in train_entries})
    resolved = {
        "seed": seed,
        "fold": dump(fold),
        "train_clip_lists": [dump(lst) for lst in train_lists],
        "test_clip_lists": {n: dump(lst) for n, lst in fold.test_lists().items()},
    }

    try:
        channel_set = ChannelSet.parse(params.channels)
    except ValueError as exc:
        raise SystemExit(f"channels: {exc}") from exc

    init = None
    inherited_videos: list[str] = []
    inherited_recordings: list[str] = []
    lineage_known = True
    if fold.init_from is not None:
        init = torch.load(fold.init_from, map_location=device, weights_only=False)
        init_channels = tuple(
            init.get("channels") or init["feature_config"]["channel_names"]
        )
        init_arch = init.get("arch", "zephyr")
        if init_channels != channel_set.names or init_arch != architecture.name:
            raise SystemExit(
                f"init_from {fold.init_from} is {init_arch} on "
                f"{list(init_channels)}, this fold is {architecture.name} on "
                f"{list(channel_set.names)}"
            )
        # The weights expect the statistics they were trained with, so a
        # fine-tune keeps them rather than re-measuring on its own clips.
        mean, std = init["mean"], init["std"]
        # A fine-tune has also seen everything its parent saw.  A parent with
        # no record leaves this run's lineage unknown, which evaluate warns on.
        if init.get("train_videos") is None:
            lineage_known = False
        else:
            lineage_known = init.get("lineage_known", True)
            inherited_videos = sorted(
                {*init["train_videos"], *init.get("inherited_videos", ())}
            )
            inherited_recordings = sorted(
                {*init["train_recordings"], *init.get("inherited_recordings", ())}
            )
    else:
        mean, std = channel_stats(
            train_entries, stats_path(features_dir, train_entries)
        )

    # Every training clip trains.  Validation, if any, is the time-tail of
    # these same clips -- see val_fraction.
    train_span = (0.0, 1.0 - params.val_fraction)
    val_span = (1.0 - params.val_fraction, 1.0)
    scoring = params.val_fraction > 0

    tail = (
        f"validating on the last {params.val_fraction:.0%} of each clip"
        if scoring
        else f"NO validation -- fixed {params.epochs} epochs, no early stopping"
    )
    print(
        f"train {len(train_entries)} clips / {len(train_recordings)} recordings in "
        f"{len(sources)} list(s) (weights {weights})  |  {tail}",
        flush=True,
    )
    selected_mean, selected_std = channel_set.take_stats(mean, std)
    print(
        f"channels {channel_set} ({len(channel_set)} of {len(CHANNEL_NAMES)} stored)  "
        f"mean {np.round(selected_mean, 2)}  std {np.round(selected_std, 2)}",
        flush=True,
    )

    augment = augment_config(params)
    print(
        f"augmentation: {'on -> ' + str(augment) if augment.enabled else 'off'}",
        flush=True,
    )

    horizon = params.epochs * params.steps_per_epoch * params.batch_size
    grids = {
        "select_fs": recipe.select_fs,
        "output_fs": OUTPUT_FS,
        "motion_tau_s": recipe.motion_tau_s,
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
        channels=channel_set,
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
            [train_entries],
            window=params.window,
            mean=mean,
            std=std,
            stride=params.window,
            span=val_span,
            channels=channel_set,
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
    model = architecture.build(channel_set, params, mean, std, **arch_kwargs).to(device)
    if init is not None:
        model.load_state_dict(init["model"])
        print(f"initialised from {fold.init_from} (epoch {init.get('epoch')})")
    criterion = architecture.criterion(params)
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
        f"frames ({model.receptive_field / 60:.1f} s) | window {params.window} "
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
            "best": best,
            "since_best": since_best,
            "history": history,
            "global_step": global_step,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            # What this checkpoint trained on: evaluate refuses to score any
            # clip whose video or recording is listed here.
            "train_videos": train_videos,
            "train_recordings": train_recordings,
            "inherited_videos": inherited_videos,
            "inherited_recordings": inherited_recordings,
            "lineage_known": lineage_known,
            "init_from": str(fold.init_from) if fold.init_from else None,
            "seed": seed,
            # Not part of state_dict, but the weights are unusable without
            # it: it fixes the first conv's input width and which planes of
            # the stored array to feed it.
            "channels": list(channel_set.names),
            "arch": architecture.name,
            "arch_kwargs": arch_kwargs,
            # Full-width, over every stored channel, so checkpoints trained
            # on different selections stay comparable.  Consumers slice with
            # ChannelSet.take_stats.
            "mean": mean,
            "std": std,
            "params": dump(params),
            "config": resolved,
            "feature_config": dump(recipe)
            | {"channel_names": list(CHANNEL_NAMES), "output_fs_hz": OUTPUT_FS},
            "metrics": metrics,
            "per_clip": per_clip,
        }
        # Keep the previous checkpoint intact until the replacement is fully
        # serialized. A process or machine failure during torch.save then costs
        # at most one epoch instead of corrupting the only resumable file.
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    best = -np.inf
    since_best = 0
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
        resumed_channels = tuple(state["channels"])
        if resumed_channels != channel_set.names or state["arch"] != architecture.name:
            raise SystemExit(
                f"{last_path} was trained as {state['arch']} on "
                f"{list(resumed_channels)} but this fold asks for {architecture.name} "
                f"on {list(channel_set.names)} -- it cannot be resumed."
            )
        model.load_state_dict(state.get("model_live") or state["model"])
        if ema is not None and state.get("ema") is not None:
            ema.load_state_dict(state["ema"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best = state["best"]
        since_best = state.get("since_best", 0)
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
            f"resumed from {last_path} at epoch {start_epoch} (best xcorr {best:+.4f})"
        )
    elif resume:
        print(f"resume requested but {last_path} does not exist; starting fresh")

    for epoch in range(start_epoch, params.epochs):
        sampler.epoch = epoch
        model.train()
        started = time.perf_counter()
        parts: list[dict] = []
        for batch in train_loader:
            features_ = batch["features"].to(device, non_blocking=True)
            t_in = batch["t_in"].to(device, non_blocking=True)
            t_out = batch["t_out"].to(device, non_blocking=True)
            signal = batch["signal"].to(device, non_blocking=True)
            onset = batch["onset"].to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    pred_signal, pred_onset = model(features_, t_in, t_out)
                # Losses in fp32: the correlation term normalises by a sum of
                # squares over 512 samples, which is exactly the kind of
                # reduction that loses precision in half.
                loss, stats = criterion(
                    pred_signal.float(), pred_onset.float(), signal, onset
                )
            else:
                pred_signal, pred_onset = model(features_, t_in, t_out)
                loss, stats = criterion(pred_signal, pred_onset, signal, onset)

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
        row |= {f"train_{k}": _mean(parts, k) for k in ("loss", "corr", "onset")}
        if val_loader is not None:
            row |= validate_windows(model, val_loader, criterion, device, amp_dtype)

        scored = scoring and (
            (epoch + 1) % params.score_every == 0 or epoch == params.epochs - 1
        )
        if scored:
            summary, per_clip = score_full_clips(
                model,
                train_entries,
                mean,
                std,
                device,
                window=params.infer_window,
                frame_chunk=params.frame_chunk,
                amp_dtype=amp_dtype,
                span=val_span,
            )
            row |= summary
            # Scoring a handful of clips costs seconds against a ~115 s epoch, so
            # evaluating the averaged weights as well is close to free -- and it
            # is the only way to know whether EMA actually helped.
            selected, selected_clips = model, per_clip
            if ema is not None and global_step < ema_ready_step:
                row["ema_warming"] = True
            if ema is not None:
                ema_summary, ema_clips = score_full_clips(
                    ema.module,
                    train_entries,
                    mean,
                    std,
                    device,
                    window=params.infer_window,
                    frame_chunk=params.frame_chunk,
                    amp_dtype=amp_dtype,
                    span=val_span,
                )
                row |= {f"ema_{k}": v for k, v in ema_summary.items()}
                # Select on the averaged weights -- they are what would ship --
                # but only once they have relaxed away from the initialisation.
                if global_step >= ema_ready_step:
                    summary, per_clip = ema_summary, ema_clips
                    selected, selected_clips = ema.module, ema_clips

            if summary["xcorr"] > best:
                best = summary["xcorr"]
                since_best = 0
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
            else:
                since_best += 1
            row["since_best"] = since_best

        row["seconds"] = round(time.perf_counter() - started, 1)
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        # Written every epoch, scored or not, so a crash costs one epoch.
        save(last_path, epoch, row, [])

        message = (
            f"[{epoch + 1:3d}/{params.epochs}] train corr {row['train_corr']:+.3f}"
        )
        if "val_corr" in row:
            message += f"  val corr {row['val_corr']:+.3f}  loss {row['val_loss']:.4f}"
        else:
            message += f"  loss {row['train_loss']:.4f}"
        if scored:
            marker = " *" if row.get("ema_xcorr", row["xcorr"]) >= best else ""
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
            and since_best >= params.patience
            and epoch + 1 >= params.min_epochs
        ):
            print(
                f"early stop: {since_best} scored evaluations "
                f"({since_best * params.score_every} epochs) without improving on "
                f"{best:+.4f}",
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
        print(f"best held-out correlation {best:+.4f} -> {run_dir / 'best.pt'}")
    return run_dir / "best.pt"
