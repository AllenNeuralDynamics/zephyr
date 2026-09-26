"""Signal preparation for model targets, independent of competition scoring."""

import numpy as np
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
