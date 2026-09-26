"""Signal preparation for model targets, independent of competition scoring."""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks

CANONICAL_BREATHING_SAMPLING_RATE = 60.0
TIME_COLUMN = "Time"
BREATHING_SIGNAL_COLUMN = "Signal"


def filter_sniff_signal(values: np.ndarray, fs: float) -> np.ndarray:
    """Filter raw thermistor samples before resampling the training target."""
    b_hp, a_hp = butter(2, 0.2, "highpass", fs=fs)
    filtered = filtfilt(b_hp, a_hp, values)
    b_lp, a_lp = butter(2, 40.0, "lowpass", fs=fs)
    return filtfilt(b_lp, a_lp, filtered)


def detect_inhalation_events(
    signal: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Find positive inhale peaks and negative exhale troughs in a clean trace."""
    distance = max(1, int(fs * 0.05))
    prominence = 0.1 * float(np.ptp(signal))
    inhale, _ = find_peaks(signal, distance=distance, prominence=prominence)
    exhale, _ = find_peaks(-signal, distance=distance, prominence=prominence)
    return inhale.astype(int), exhale.astype(int)


def resample_uniform(
    thermistor: pd.DataFrame,
    *,
    target_fs: float = CANONICAL_BREATHING_SAMPLING_RATE,
) -> pd.DataFrame:
    """Resample a thermistor signal onto a uniform grid via linear interpolation.

    Used to bring ground-truth (thermistor, rate inferred from timestamps) and
    predicted signals onto a common time base before signal-level comparison.

    Parameters
    ----------
    thermistor:
        Input dataframe as extracted from a clip parquet.  Must contain the
        parquet schema columns ``Time`` (seconds, monotonically increasing)
        and ``Signal``.
    target_fs:
        Target sampling rate in Hz.  Default ``CANONICAL_BREATHING_SAMPLING_RATE``
        (the canonical scoring grid).

    Returns
    -------
    pd.DataFrame
        New dataframe with the same two columns, where ``Time`` is a
        uniform grid ``t[0], t[0]+1/target_fs, ...`` spanning the input range
        and ``Signal`` is linearly interpolated onto that grid.
    """
    t = thermistor[TIME_COLUMN].to_numpy(dtype=float)
    v = thermistor[BREATHING_SIGNAL_COLUMN].to_numpy(dtype=float)

    t_uniform = np.arange(t[0], t[-1], 1.0 / target_fs)
    interp_fn = interp1d(
        t, v, kind="linear", bounds_error=False, fill_value="extrapolate"
    )
    return pd.DataFrame(
        {TIME_COLUMN: t_uniform, BREATHING_SIGNAL_COLUMN: interp_fn(t_uniform)}
    )
