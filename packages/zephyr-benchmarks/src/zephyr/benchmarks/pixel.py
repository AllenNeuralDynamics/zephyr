"""Non-learned breathing traces from pixels alone -- the floor every method must clear.

``flow``: mean optical flow over the crop on its dominant direction, integrated and
band-passed. ``pca``: per-pixel band-passed intensity, keeping the PCA component most
concentrated in one in-band peak (Poh et al., 2010). ``snr``: pixels weighted by in-band
variance fraction, signed to agree, averaged. None sees the thermistor; polarity is
fixed afterwards by :func:`.common.blind_polarity`.
"""

import numpy as np
from scipy.signal import welch

from .common import BREATH_BAND_HZ, FS, bandpass


def flow_projection(
    flow_x: np.ndarray, flow_y: np.ndarray, fs: float = FS
) -> np.ndarray:
    """``(T, H, W)`` flow in px/tau -> ``(T,)`` band-passed displacement."""
    n = len(flow_x)
    velocity = np.stack(
        [flow_x.reshape(n, -1).mean(1), flow_y.reshape(n, -1).mean(1)], axis=1
    ).astype(np.float64)
    velocity = bandpass(velocity - velocity.mean(0), fs)
    _, _, vt = np.linalg.svd(velocity, full_matrices=False)
    displacement = np.cumsum(velocity @ vt[0]) / fs
    return bandpass(displacement - displacement.mean(), fs)


def spectral_concentration(x: np.ndarray, fs: float = FS) -> float:
    """Peak / total Welch power inside the breathing band."""
    f, p = welch(x, fs=fs, nperseg=min(1024, len(x)))
    band = (f >= BREATH_BAND_HZ[0]) & (f <= BREATH_BAND_HZ[1])
    total = p[band].sum()
    return float(p[band].max() / total) if total > 0 else 0.0


def pixel_pca(frames: np.ndarray, fs: float = FS, n_components: int = 10) -> np.ndarray:
    """``(T, H, W)`` gray -> the most rhythmic of the top principal components."""
    x = frames.reshape(len(frames), -1).astype(np.float64)
    x = bandpass(x - x.mean(0), fs)
    _, evecs = np.linalg.eigh(x.T @ x)
    top = evecs[:, ::-1][:, :n_components]
    components = x @ top
    scores = [spectral_concentration(components[:, k], fs) for k in range(top.shape[1])]
    return components[:, int(np.argmax(scores))]


def snr_weighted(
    frames: np.ndarray, fs: float = FS, top_fraction: float = 0.1
) -> np.ndarray:
    """``(T, H, W)`` gray -> band-power-weighted, sign-aligned pixel average."""
    raw = frames.reshape(len(frames), -1).astype(np.float64)
    raw = raw - raw.mean(0)
    band = bandpass(raw, fs)
    total = raw.var(0)
    snr = band.var(0) / np.where(total > 0, total, np.inf)
    k = max(1, round(top_fraction * len(snr)))
    chosen = np.argsort(snr)[::-1][:k]
    reference = band[:, chosen[0]]
    signs = np.sign(band[:, chosen].T @ reference)
    signs[signs == 0] = 1.0
    weights = snr[chosen] * signs
    return band[:, chosen] @ weights / np.abs(weights).sum()
