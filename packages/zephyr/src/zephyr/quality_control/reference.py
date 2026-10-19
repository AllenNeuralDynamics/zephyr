"""What a checkpoint's quality control compares new videos with, measured once.

A :class:`Reference` holds the training clips' embedding distribution, the embedding
distances of unseen labelled animals, image-statistic quantiles, the population's breath
intervals, and a calibration of the onset head's peaks. It is a JSON file shipped beside
the checkpoint (``zephyr quality-control reference``), tied to it by its SHA-256.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

QUANTILES = np.arange(101)
"""Percentiles every distribution is stored at."""
INTERVAL_EDGES = np.geomspace(1 / 30, 4.0, 41)
"""Breath-interval bins (s): 30 Hz to one breath in 4 s, log-spaced."""
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
    embedding_mean: np.ndarray
    embedding_precision: np.ndarray
    distance_quantiles: np.ndarray
    image_quantiles: dict[str, np.ndarray] | None
    interval_edges: np.ndarray
    interval_density: np.ndarray | None
    onset_calibration: dict | None
    record: dict
    """Everything else written (clips, settings, reliability): kept as read."""

    def distance(self, embeddings: np.ndarray) -> np.ndarray:
        """Mahalanobis distance of each row from the training embeddings (NaN rows
        stay NaN)."""
        centred = np.asarray(embeddings, np.float64) - self.embedding_mean
        squared = np.einsum("ij,jk,ik->i", centred, self.embedding_precision, centred)
        return np.sqrt(np.maximum(squared, 0))

    def distance_percentile(self, distances: np.ndarray) -> np.ndarray:
        """Where each distance falls among an unseen animal's (0 to 100)."""
        return np.interp(distances, self.distance_quantiles, QUANTILES)

    def calibrate(self, peak_probability: np.ndarray) -> np.ndarray:
        """Probability that an inhale the head asserts with this peak is real."""
        c = self.onset_calibration
        return _sigmoid(c["slope"] * logit(peak_probability) + c["intercept"])

    @classmethod
    def load(cls, path: Path) -> "Reference":
        data = json.loads(Path(path).read_text())
        embedding, intervals = data["embedding"], data["intervals"]
        image = data["image"]
        return cls(
            checkpoint_sha256=data["checkpoint_sha256"],
            embedding_mean=np.asarray(embedding["mean"]),
            embedding_precision=np.asarray(embedding["precision"]),
            distance_quantiles=np.asarray(embedding["distance_quantiles"]),
            image_quantiles=(
                {k: np.asarray(v) for k, v in image.items()} if image else None
            ),
            interval_edges=np.asarray(intervals["edges"]),
            interval_density=(
                np.asarray(intervals["density"]) if intervals["density"] else None
            ),
            onset_calibration=data["onset_calibration"],
            record=data,
        )


def embedding_statistics(
    embeddings: np.ndarray, shrinkage: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and precision (inverse of the covariance shrunk toward a scaled identity)."""
    x = np.asarray(embeddings, np.float64)
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    target = np.trace(cov) / len(cov) * np.eye(len(cov))
    shrunk = (1 - shrinkage) * cov + shrinkage * target
    return mean, np.linalg.inv(shrunk)


def interval_density(intervals: np.ndarray) -> np.ndarray | None:
    """The population's breath intervals over :data:`INTERVAL_EDGES`, smoothed so no
    bin is empty."""
    intervals = np.asarray(intervals, float)
    if not len(intervals):
        return None
    edges = INTERVAL_EDGES
    clipped = np.clip(intervals, edges[0], np.nextafter(edges[-1], 0))
    counts = np.histogram(clipped, bins=edges)[0] + 0.5
    return counts / counts.sum()


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


def quantiles(values: np.ndarray) -> list[float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return np.percentile(values, QUANTILES).tolist() if len(values) else []


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
