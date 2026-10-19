"""What a checkpoint's quality control needs from labelled data, measured once.

A :class:`Reference` is a calibration of the onset head's peaks on labelled clips the
checkpoint never trained on (its validation animals): the chance that an inhale the
head asserts is real. It is a JSON file beside the checkpoint (written after training,
or by ``zephyr quality-control reference``), tied to it by its SHA-256.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

RELIABILITY_BINS = 10
EPS = 1e-6


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class Reference:
    checkpoint_sha256: str
    onset_calibration: dict | None
    record: dict
    """Everything else written (clips, settings, reliability): kept as read."""

    def calibrate(self, peak_probability: np.ndarray) -> np.ndarray:
        """Probability that an inhale the head asserts with this peak is real."""
        c = self.onset_calibration
        return _sigmoid(c["slope"] * logit(peak_probability) + c["intercept"])

    @classmethod
    def load(cls, path: Path) -> "Reference":
        data = json.loads(Path(path).read_text())
        return cls(
            checkpoint_sha256=data["checkpoint_sha256"],
            onset_calibration=data["onset_calibration"],
            record=data,
        )


def fit_calibration(peaks: np.ndarray, real: np.ndarray) -> dict | None:
    """Logistic map from a head peak's logit to the chance it matches a true inhale
    (slope and intercept), with a reliability table before and after."""
    peaks, real = np.asarray(peaks, float), np.asarray(real, float)
    if len(peaks) < 20 or real.min() == real.max():
        return None
    x = logit(peaks)

    def loss(theta):
        z = theta[0] * x + theta[1]
        return float(np.mean(np.logaddexp(0, z) - real * z))

    slope, intercept = minimize(loss, np.array([1.0, 0.0]), method="BFGS").x
    calibrated = _sigmoid(slope * x + intercept)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "events": len(peaks),
        "precision": float(real.mean()),
        "ece_raw": expected_calibration_error(peaks, real),
        "ece": expected_calibration_error(calibrated, real),
        "reliability": reliability(calibrated, real),
    }


def reliability(predicted: np.ndarray, real: np.ndarray) -> list[dict]:
    bins = np.clip((predicted * RELIABILITY_BINS).astype(int), 0, RELIABILITY_BINS - 1)
    rows = []
    for b in range(RELIABILITY_BINS):
        m = bins == b
        if m.any():
            rows.append(
                {
                    "predicted": float(predicted[m].mean()),
                    "observed": float(real[m].mean()),
                    "n": int(m.sum()),
                }
            )
    return rows


def expected_calibration_error(predicted: np.ndarray, real: np.ndarray) -> float:
    rows = reliability(np.asarray(predicted, float), np.asarray(real, float))
    total = sum(r["n"] for r in rows)
    return float(
        sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in rows) / total
    )


def jsonable(value):
    """*value* with NaN as ``None``, so the file is valid JSON."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
