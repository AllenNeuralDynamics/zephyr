"""Local training diagnostics; final scores come from :mod:`.evaluation`."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def zero_lag_correlation(truth: np.ndarray, predicted: np.ndarray) -> float:
    """Pearson correlation for aligned traces."""
    truth = truth - truth.mean()
    predicted = predicted - predicted.mean()
    energy = np.sqrt(np.sum(truth**2) * np.sum(predicted**2))
    return float(np.sum(truth * predicted) / energy) if energy else float("nan")


def event_f1(
    truth_times_s: np.ndarray,
    predicted_times_s: np.ndarray,
    tolerance_s: float = 0.017,
) -> float:
    """One-to-one event F1 used only for training diagnostics."""
    n_truth, n_pred = len(truth_times_s), len(predicted_times_s)
    if n_truth == 0 and n_pred == 0:
        return float("nan")
    if n_truth == 0 or n_pred == 0:
        return 0.0
    delta = np.abs(truth_times_s[:, None] - predicted_times_s[None, :])
    feasible = delta <= tolerance_s
    if not feasible.any():
        return 0.0
    penalty = tolerance_s * (n_truth + n_pred + 1)
    rows, cols = linear_sum_assignment(np.where(feasible, delta, penalty))
    matches = sum(bool(feasible[i, j]) for i, j in zip(rows, cols, strict=True))
    return float(2 * matches / (n_truth + n_pred))
