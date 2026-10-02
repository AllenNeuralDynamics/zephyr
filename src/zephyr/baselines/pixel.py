"""Non-learned breathing traces from pixels alone -- the floor every method must clear.

Three methods, none of which ever sees the thermistor:

``flow``
    Mean optical flow over the crop, projected on its dominant direction (per-clip
    PCA of the 2-D velocity), integrated to displacement, band-passed.
``pca``
    Per-pixel band-passed intensity, per-clip PCA, keeping the component whose
    spectrum is most concentrated in one peak inside the breathing band -- the
    blind-source-separation recipe of camera pulse measurement (Poh et al., 2010).
``snr``
    Pixels weighted by the fraction of their variance inside the breathing band,
    the top fraction kept, each signed to agree with the best pixel, averaged.

Every output is unsigned in the sense that matters: polarity is fixed afterwards
by :func:`.common.blind_polarity`.
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
