"""What the ground-truth thermistor looks like, per clip and per set of clips.

Six features of each recording, for the training clips, the test clips and the
out-of-distribution rig, and how far each set sits from the training one.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import welch

from zephyr.signal import detect_inhalation_events, filter_sniff_signal

from . import results

FEATURES: dict[str, tuple[str, str, int, bool]] = {
    "rate_hz": ("Breathing rate", "inhales / s", 1, False),
    "amp": ("Amplitude", "ADC, p95 - p5 of the filtered trace", 0, True),
    "snr_db": ("Signal to noise", "dB, 0.5-15 Hz vs 25-100 Hz", 1, False),
    "ibi_cv": ("Interval variability", "CV of inhale intervals", 2, False),
    "breath_cv": (
        "Breath-size variability",
        "CV of per-breath peak-to-trough",
        2,
        False,
    ),
    "raw_levels": ("ADC resolution", "unique raw values", 0, True),
}
"""Feature -> (label, what it is, decimals shown, log scale). Amplitude and resolution
span an order of magnitude between clips, so their distances are taken in log10."""
SETS: tuple[str, ...] = ("Train", "Test", "OOD (side camera)")
SNR_SIGNAL_HZ: tuple[float, float] = (0.5, 15.0)
SNR_NOISE_HZ: tuple[float, float] = (25.0, 100.0)


def clip_features(thermistor: Path) -> dict[str, float]:
    """The :data:`FEATURES` of one thermistor recording (parquet, ``Time`` and
    ``Signal``). Inhales are the scorer's, found on the filtered trace."""
    frame = pd.read_parquet(thermistor)
    t = frame["Time"].to_numpy(float)
    raw = frame["Signal"].to_numpy(float)
    fs = 1.0 / float(np.median(np.diff(t)))
    trace = filter_sniff_signal(raw, fs)
    inhale, _ = detect_inhalation_events(trace, fs)
    intervals = np.diff(t[inhale])
    cycles = np.split(trace, inhale[1:])  # inhale to inhale
    sizes = np.array([c.max() - c.min() for c in cycles if len(c) > 2])
    freq, power = welch(trace, fs=fs, nperseg=int(4 * fs))

    def band(lo: float, hi: float) -> float:
        return float(power[(freq >= lo) & (freq <= hi)].sum())

    return {
        "rate_hz": len(inhale) / float(t[-1] - t[0]),
        "amp": float(np.percentile(trace, 95) - np.percentile(trace, 5)),
        "snr_db": 10 * np.log10(band(*SNR_SIGNAL_HZ) / band(*SNR_NOISE_HZ)),
        "ibi_cv": float(intervals.std() / intervals.mean()),
        "breath_cv": float(sizes.std() / sizes.mean()),
        "raw_levels": float(len(np.unique(raw))),
    }


def features() -> pd.DataFrame:
    """One row per clip: ``set``, ``clip`` and every feature, for the training clips,
    the test clips and the out-of-distribution clips."""
    groups: dict[str, list] = {
        "Train": results.train_entries(),
        "Test": [c.entry for c in results.clips().values()],
        "OOD (side camera)": [
            c.entry for c in results.clips(results.OOD_CLIPS).values()
        ],
    }
    rows = [
        {"set": name, "clip": results.short_name(entry)}
        | clip_features(Path(entry.thermistor))
        for name, entries in groups.items()
        for entry in entries
    ]
    return pd.DataFrame(rows)


def _scale(values: NDArray, feature: str) -> NDArray:
    return np.log10(values) if FEATURES[feature][3] else np.asarray(values, float)


def shift_from_train(table: pd.DataFrame) -> pd.DataFrame:
    """Every clip's features as distances from the training clips, in robust z:
    ``(x - median) / (1.4826 * MAD)`` of the training set (log10 for the features
    that span decades). Same layout as *table*."""
    out = table.copy()
    train = table[table["set"] == "Train"]
    for feature in FEATURES:
        reference = _scale(train[feature].to_numpy(), feature)
        median = np.median(reference)
        spread = 1.4826 * np.median(np.abs(reference - median))
        out[feature] = (_scale(table[feature].to_numpy(), feature) - median) / spread
    return out


def summary_table(table: pd.DataFrame) -> pd.DataFrame:
    """Median (interquartile range) of every feature in each set, and how far the
    out-of-distribution median sits from the training one, in robust z."""
    shifts = shift_from_train(table)
    rows = []
    for feature, (label, unit, decimals, _) in FEATURES.items():
        row = {"feature": label, "what": unit}
        for name in SETS:
            x = table.loc[table["set"] == name, feature]
            q1, med, q3 = np.percentile(x, [25, 50, 75])
            row[name] = f"{med:.{decimals}f} ({q1:.{decimals}f} - {q3:.{decimals}f})"
        row["OOD vs train (z)"] = round(
            float(shifts.loc[shifts["set"] == SETS[-1], feature].median()), 1
        )
        rows.append(row)
    return pd.DataFrame(rows)
