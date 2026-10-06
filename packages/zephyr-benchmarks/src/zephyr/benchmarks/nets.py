"""TS-CAN and PhysNet, reimplemented in PyTorch from the published architectures.

TS-CAN: Liu, Fromm, Patel & McDuff, NeurIPS 2020. PhysNet: Yu, Li & Zhao, BMVC 2019.
Deviations forced by the data: one gray channel, the mouse breathing band, PhysNet at
the 96 px crop.

Both are wrapped to :class:`~zephyr.model.BreathingNet`'s interface (``forward(features,
t_in, t_out, chunk=None) -> (signal, onset_logits)``, ``channels``, ``receptive_field``)
so training and scoring are shared. Onset logits are zeros; an optional
``postprocess(signal, fs)`` is applied by :func:`~zephyr.infer.predict_clip`.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.signal import butter, filtfilt
from scipy.sparse.linalg import spsolve
from torch import nn

from zephyr.channels import CHANNEL_NAMES, ChannelSet
from zephyr.model import BreathingNet, pearson_loss, resample_embeddings

from .common import BREATH_BAND_HZ

GRAY = ChannelSet.parse("gray")


def temporal_shift(x: torch.Tensor, n_segment: int, fold_div: int = 3) -> torch.Tensor:
    """Shift a third of the channels one frame forward and a third one frame back."""
    nt, c, h, w = x.shape
    x = x.view(nt // n_segment, n_segment, c, h, w)
    fold = c // fold_div
    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]
    out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]
    out[:, :, 2 * fold :] = x[:, :, 2 * fold :]
    return out.view(nt, c, h, w)


def attention_mask(g: torch.Tensor) -> torch.Tensor:
    """Normalise a soft mask to mean 0.5 over space, as TS-CAN does."""
    h, w = g.shape[-2:]
    return g / g.sum(dim=(2, 3), keepdim=True) * h * w * 0.5


class TSCAN(nn.Module):
    """Two-stream convolutional attention network with temporal shift.

    ``motion`` and ``appearance`` are ``(N, 1, S, S)`` with N a multiple of
    *frame_depth*, consecutive frames within each segment.  Returns ``(N,)``.
    """

    def __init__(
        self,
        frame_depth: int = 10,
        img_size: int = 36,
        filters1: int = 32,
        filters2: int = 64,
        dense: int = 128,
        dropout1: float = 0.25,
        dropout2: float = 0.5,
    ) -> None:
        super().__init__()
        self.frame_depth = frame_depth
        self.m1 = nn.Conv2d(1, filters1, 3, padding=1)
        self.m2 = nn.Conv2d(filters1, filters1, 3)
        self.m3 = nn.Conv2d(filters1, filters2, 3, padding=1)
        self.m4 = nn.Conv2d(filters2, filters2, 3)
        self.a1 = nn.Conv2d(1, filters1, 3, padding=1)
        self.a2 = nn.Conv2d(filters1, filters1, 3)
        self.a3 = nn.Conv2d(filters1, filters2, 3, padding=1)
        self.a4 = nn.Conv2d(filters2, filters2, 3)
        self.g1 = nn.Conv2d(filters1, 1, 1)
        self.g2 = nn.Conv2d(filters2, 1, 1)
        self.pool = nn.AvgPool2d(2)
        self.drop1 = nn.Dropout(dropout1)
        self.drop2 = nn.Dropout(dropout2)
        side = ((img_size - 2) // 2 - 2) // 2
        self.dense = nn.Linear(filters2 * side * side, dense)
        self.out = nn.Linear(dense, 1)

    def _tsm(self, x: torch.Tensor) -> torch.Tensor:
        return temporal_shift(x, self.frame_depth)

    def forward(self, motion: torch.Tensor, appearance: torch.Tensor) -> torch.Tensor:
        d = torch.tanh(self.m1(self._tsm(motion)))
        d = torch.tanh(self.m2(self._tsm(d)))
        r = torch.tanh(self.a2(torch.tanh(self.a1(appearance))))
        d = d * attention_mask(torch.sigmoid(self.g1(r)))
        d = self.drop1(self.pool(d))
        r = self.drop1(self.pool(r))
        d = torch.tanh(self.m3(self._tsm(d)))
        d = torch.tanh(self.m4(self._tsm(d)))
        r = torch.tanh(self.a4(torch.tanh(self.a3(r))))
        d = d * attention_mask(torch.sigmoid(self.g2(r)))
        d = self.drop1(self.pool(d))
        d = self.drop2(torch.tanh(self.dense(d.flatten(1))))
        return self.out(d).squeeze(-1)


def tarvainen_detrend(signal: np.ndarray, lam: float = 100.0) -> np.ndarray:
    """Smoothness-priors detrend (Tarvainen et al., 2002), solved sparsely.

    Same result as the dense ``(I - (I + lam^2 D'D)^-1) x`` the reference code uses,
    which at 18 000 samples would need a 2.6 GB matrix inverse.
    """
    n = len(signal)
    d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    system = (sparse.identity(n) + lam**2 * (d.T @ d)).tocsc()
    return signal - spsolve(system, signal)


def reconstruct_from_derivative(
    derivative: np.ndarray,
    fs: float,
    band: tuple[float, float] = BREATH_BAND_HZ,
    lam: float = 100.0,
) -> np.ndarray:
    """TS-CAN's own inference post-processing: integrate, detrend, band-pass.

    *derivative* is the forward difference ``d[k] = y[k+1] - y[k]`` (the paper's
    ``np.diff`` label), so the running sum up to but excluding ``k = n`` is exactly
    ``y[n]`` and the trace lands on the right frame with no lag.  The last sample
    has no successor and is not used.
    """
    integrated = np.concatenate(([0.0], np.cumsum(derivative[:-1].astype(np.float64))))
    detrended = tarvainen_detrend(integrated, lam)
    b, a = butter(1, band, btype="bandpass", fs=fs)
    return filtfilt(b, a, detrended)


class DerivativeMSELoss(nn.Module):
    """MSE against the standardised forward difference of the target.

    ``d[k] = y[k+1] - y[k]``, as in the paper.  The last sample of a window has no
    successor, so it is not scored.  Same call signature and stats keys as
    :class:`~zephyr.model.BreathingLoss`.
    """

    def forward(self, signal_pred, onset_logits, signal_true, onset_true):
        d = signal_true[..., 1:] - signal_true[..., :-1]
        d = d / d.std(dim=-1, keepdim=True).clamp_min(1e-6)
        pred = signal_pred[..., :-1]
        loss = F.mse_loss(pred, d)
        corr = 1.0 - pearson_loss(pred, d)
        return loss, {
            "loss": float(loss.detach()),
            "corr": float(corr.detach()),
            "onset": 0.0,
        }


class TSCANBreathing(nn.Module):
    """TS-CAN on the gray channel, predicting the trace's forward difference."""

    arch = "tscan"

    def __init__(
        self,
        channels: ChannelSet = GRAY,
        *,
        frame_depth: int = 10,
        img_size: int = 36,
        gray_mean: float = 0.0,
        gray_std: float = 1.0,
        band: tuple[float, float] = BREATH_BAND_HZ,
    ) -> None:
        super().__init__()
        if channels.names != ("gray",):
            raise ValueError(f"TS-CAN takes only the gray channel, got {channels}")
        self.channels = channels
        self.frame_depth = frame_depth
        self.img_size = img_size
        self.band = band
        self.net = TSCAN(frame_depth=frame_depth, img_size=img_size)
        self.register_buffer("gray_mean", torch.tensor(float(gray_mean)))
        self.register_buffer("gray_std", torch.tensor(float(gray_std)))

    @property
    def receptive_field(self) -> int:
        return 2 * self.frame_depth

    def forward(self, features, t_in, t_out, chunk=None):
        b, t = features.shape[:2]
        s = self.img_size
        x = F.interpolate(
            features[:, :, 0].reshape(b * t, 1, *features.shape[-2:]).float(),
            size=(s, s),
            mode="area",
        ).view(b, t, s, s)
        raw = (x * self.gray_std + self.gray_mean).clamp_min(1.0)
        motion = (raw[:, 1:] - raw[:, :-1]) / (raw[:, 1:] + raw[:, :-1])
        motion = motion / motion.flatten(1).std(dim=1).clamp_min(1e-6).view(b, 1, 1, 1)
        appearance = x[:, :-1]
        n = t - 1
        pad = (-n) % self.frame_depth
        if pad:
            motion = torch.cat([motion, motion[:, -1:].expand(-1, pad, -1, -1)], dim=1)
            appearance = torch.cat(
                [appearance, appearance[:, -1:].expand(-1, pad, -1, -1)], dim=1
            )
        out = self.net(motion.reshape(-1, 1, s, s), appearance.reshape(-1, 1, s, s))
        out = out.view(b, n + pad)[:, :n]
        # Output j describes the frame pair (j, j+1) and is the forward difference
        # y[j+1] - y[j], so it sits at frame j's time.
        signal = resample_embeddings(out.unsqueeze(-1), t_in[:, :-1], t_out).squeeze(-1)
        return signal, torch.zeros_like(signal)

    def postprocess(self, signal: np.ndarray, fs: float) -> np.ndarray:
        return reconstruct_from_derivative(signal, fs, self.band)


def _block(cin: int, cout: int, kernel, padding) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, kernel, stride=1, padding=padding),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
    )


def _up() -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose3d(64, 64, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
        nn.BatchNorm3d(64),
        nn.ELU(),
    )


class PhysNet(nn.Module):
    """3-D CNN encoder-decoder: ``(B, C, T, H, W)`` -> ``(B, T)``, T divisible by 4."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.b1 = _block(in_channels, 16, (1, 5, 5), (0, 2, 2))
        self.b2 = _block(16, 32, 3, 1)
        self.b3 = _block(32, 64, 3, 1)
        self.b4, self.b5, self.b6, self.b7, self.b8, self.b9 = (
            _block(64, 64, 3, 1) for _ in range(6)
        )
        self.up1, self.up2 = _up(), _up()
        self.pool_s = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.pool_st = nn.MaxPool3d(2, stride=2)
        self.spatial = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.head = nn.Conv3d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t = x.shape[:3]
        x = self.pool_s(self.b1(x))
        x = self.pool_st(self.b3(self.b2(x)))
        x = self.pool_st(self.b5(self.b4(x)))
        x = self.pool_s(self.b7(self.b6(x)))
        x = self.up2(self.up1(self.b9(self.b8(x))))
        return self.head(self.spatial(x)).view(b, t)


class PhysNetBreathing(nn.Module):
    """PhysNet on the gray channel, predicting the trace directly."""

    arch = "physnet"

    def __init__(self, channels: ChannelSet = GRAY) -> None:
        super().__init__()
        if channels.names != ("gray",):
            raise ValueError(f"PhysNet takes only the gray channel, got {channels}")
        self.channels = channels
        self.net = PhysNet(in_channels=1)

    @property
    def receptive_field(self) -> int:
        return 64

    def forward(self, features, t_in, t_out, chunk=None):
        t = features.shape[1]
        x = features.permute(0, 2, 1, 3, 4)  # (B, 1, T, H, W)
        pad = (-t) % 4
        if pad:
            x = torch.cat([x, x[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
        out = self.net(x)[:, :t]
        signal = resample_embeddings(out.unsqueeze(-1), t_in, t_out).squeeze(-1)
        return signal, torch.zeros_like(signal)


ARCHS = ("zephyr", "tscan", "physnet")
ARCH_OPTIONS = {
    "zephyr": set(),
    "tscan": {"img_size"},
    "physnet": set(),
}
"""Architecture options a checkpoint may record and :func:`build_model` accepts."""


def build_model(
    arch: str,
    channels: ChannelSet,
    *,
    dropout: float = 0.1,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    **arch_kwargs,
) -> nn.Module:
    """Construct any supported network.  *mean*/*std* are full-width channel stats.

    *arch_kwargs* are architecture options (TS-CAN: ``img_size``) that a
    checkpoint records so it can be rebuilt; the other networks take none.
    """
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")
    unsupported = set(arch_kwargs) - ARCH_OPTIONS[arch]
    if unsupported:
        raise ValueError(f"--arch {arch} does not take {sorted(unsupported)}")
    if arch == "zephyr":
        return BreathingNet(channels=channels, dropout=dropout)
    if arch == "tscan":
        gray = CHANNEL_NAMES.index("gray")
        return TSCANBreathing(
            channels,
            gray_mean=float(mean[gray]) if mean is not None else 0.0,
            gray_std=float(std[gray]) if std is not None else 1.0,
            **arch_kwargs,
        )
    return PhysNetBreathing(channels, **arch_kwargs)
