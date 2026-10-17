"""PhysFormer's and RhythmFormer's own training losses, as their authors' code computes them.

Both add to the negative Pearson a cross-entropy over the window's power spectrum, one bin
per beat per minute, against the bin of the target's rate. Deviations forced by the data:
the bins span the breathing band instead of heart rates, and the target rate is the
thermistor's, estimated per window as RhythmFormer's code does (PhysFormer read it from the
ECG). The ``_event`` variants add zephyr's onset loss at ``w_onset``.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, welch
from torch import nn

from zephyr.model import onset_bce, pearson_loss

from .common import BREATH_BAND_HZ
from .nets import tarvainen_detrend


def rate_bins(band: tuple[float, float] = BREATH_BAND_HZ) -> np.ndarray:
    """Every whole rate per minute in *band*, upper end excluded (``arange(40, 180)``)."""
    return np.arange(math.ceil(band[0] * 60), math.ceil(band[1] * 60), dtype=np.float64)


def window_rate(
    signal: np.ndarray, fs: float, band: tuple[float, float] = BREATH_BAND_HZ
) -> float:
    """The rate of *signal* per minute, as RhythmFormer's ``calculate_hr``: detrend
    (lambda 100), first-order band-pass, then the Welch peak strictly inside *band*."""
    x = tarvainen_detrend(np.asarray(signal, dtype=np.float64), 100.0)
    b, a = butter(1, band, btype="bandpass", fs=fs)
    x = filtfilt(b, a, x)
    f, p = welch(x, fs, nfft=int(1e5 / fs), nperseg=min(len(x) - 1, 256))
    inside = (f > band[0]) & (f < band[1])
    return float(f[inside][np.argmax(p[inside])] * 60)


def spectrum(signal: torch.Tensor, fs: float, bins: np.ndarray) -> torch.Tensor:
    """Power of each Hann-windowed ``(B, N)`` window at *bins* (per minute), divided by
    its sum: the papers' "analogous softmax", which they then use as logits."""
    n = signal.shape[-1]
    k = torch.as_tensor(bins / 60.0 * n / fs, dtype=signal.dtype, device=signal.device)
    steps = torch.arange(n, dtype=signal.dtype, device=signal.device)
    phase = 2 * math.pi * k[:, None] * steps[None] / n
    x = signal * torch.hann_window(n, periodic=False, dtype=signal.dtype).to(signal)
    power = (x @ phase.sin().T) ** 2 + (x @ phase.cos().T) ** 2
    return power / power.sum(dim=-1, keepdim=True)


def gaussian_label(target: torch.Tensor, n_bins: int, std: float) -> torch.Tensor:
    """PhysFormer's label distribution: the normal density at each bin around the target
    bin, floored at 1e-15 and not renormalised."""
    bins = torch.arange(n_bins, dtype=torch.float32, device=target.device)
    centre = target.float()[:, None]
    density = torch.exp(-((bins - centre) ** 2) / (2 * std**2))
    return (density / (math.sqrt(2 * math.pi) * std)).clamp_min(1e-15)


class _SpectralLoss(nn.Module):
    """What both losses share: the bins, each window's target bin and the onset term."""

    def __init__(
        self, fs: float, w_onset: float, band: tuple[float, float] = BREATH_BAND_HZ
    ) -> None:
        super().__init__()
        self.fs, self.w_onset, self.band = fs, w_onset, band
        self.bins = rate_bins(band)

    def target_bins(self, signal_true: torch.Tensor) -> torch.Tensor:
        """Each window's rate, truncated to a whole bin as the papers' ``int(hr)``."""
        windows = signal_true.detach().double().cpu().numpy()
        rates = np.array([window_rate(w, self.fs, self.band) for w in windows])
        index = rates.astype(int) - int(self.bins[0])
        index = np.clip(index, 0, len(self.bins) - 1)
        return torch.as_tensor(index, dtype=torch.long, device=signal_true.device)

    def finish(self, terms, signal_pred, signal_true, onset_logits, onset_true):
        onset = onset_bce(onset_logits, onset_true)
        total = sum(terms.values()) + self.w_onset * onset
        stats = {k: float(v.detach()) for k, v in terms.items()}
        return total, stats | {
            "loss": float(total.detach()),
            "corr": float(1.0 - pearson_loss(signal_pred, signal_true).detach()),
            "onset": float(onset.detach()),
        }


class PhysFormerLoss(_SpectralLoss):
    """``a * Pearson + b * (CE + KL)``, ``a = 0.1`` and ``b`` rising as ``5^(epoch/E)``.

    The authors trained for E = 25 epochs (``b = 5 ** (epoch / 25.0)``); here E is the
    run's own ``epochs``, so ``b`` climbs from 1 to almost 5 over training just as theirs.
    KL compares the softmax of the spectrum with :func:`gaussian_label` (std 1).
    """

    def __init__(self, fs: float, epochs: int, w_onset: float, **kwargs) -> None:
        super().__init__(fs, w_onset, **kwargs)
        self.epochs = epochs
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.a, self.b = 0.1, 5.0 ** (epoch / self.epochs)

    def forward(self, signal_pred, onset_logits, signal_true, onset_true):
        power = spectrum(signal_pred, self.fs, self.bins)
        target = self.target_bins(signal_true)
        label = gaussian_label(target, len(self.bins), std=1.0)
        kl = (label * (label.log() - F.log_softmax(power, dim=-1))).sum(-1).mean()
        ce = F.cross_entropy(power, target)
        terms = {
            "pearson": self.a * pearson_loss(signal_pred, signal_true),
            "frequency": self.b * (ce + kl),
        }
        return self.finish(terms, signal_pred, signal_true, onset_logits, onset_true)


class RhythmFormerLoss(_SpectralLoss):
    """``0.2 * Pearson + CE``: the terms of the authors' ``RhythmFormer_Loss`` that train.

    Theirs also adds ``1.0 * loss_hr``, a KL between peak positions computed on detached
    arrays, so it adds a constant and no gradient, and it computes a KL to a label
    distribution that it never uses; both are left out.
    """

    def forward(self, signal_pred, onset_logits, signal_true, onset_true):
        power = spectrum(signal_pred, self.fs, self.bins)
        terms = {
            "pearson": 0.2 * pearson_loss(signal_pred, signal_true),
            "frequency": F.cross_entropy(power, self.target_bins(signal_true)),
        }
        return self.finish(terms, signal_pred, signal_true, onset_logits, onset_true)
