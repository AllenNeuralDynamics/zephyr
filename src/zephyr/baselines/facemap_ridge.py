"""Facemap-style SVD features with a linear readout -- the field-standard baseline.

Facemap (Stringer et al., 2019; Syeda et al., 2024) summarises face video as the
top singular vectors of the *motion* (absolute frame difference) and of the
*movie* (raw frames).  Here the basis is fitted once on frames sampled from every
training clip and shared by all clips, so a single readout transfers between
animals; Facemap itself fits per video, which would leave no common space to
regress in.  Motion is rectified, so direction is lost and inhale and exhale can
look alike -- which is why the movie variant is reported too.

The readout is ridge regression on lagged copies of the per-clip z-scored
components: a learned linear filter, nothing more.  The regularisation strength
is chosen by leave-one-session-out, computed exactly from per-session Gram
matrices so no design matrix is ever held for more than one clip.
"""

import functools
import operator
from dataclasses import dataclass

import numpy as np

from .common import shift


@dataclass(frozen=True)
class SvdBasis:
    mean: np.ndarray  # (P,)
    components: np.ndarray  # (P, K)


def movie_matrix(frames: np.ndarray) -> np.ndarray:
    return frames.reshape(len(frames), -1).astype(np.float32)


def motion_matrix(frames: np.ndarray) -> np.ndarray:
    x = movie_matrix(frames)
    return np.abs(np.diff(x, axis=0, prepend=x[:1]))


def fit_basis(samples: list[np.ndarray], n_components: int) -> SvdBasis:
    """Top principal directions of the pooled ``(N_i, P)`` *samples*."""
    x = np.concatenate(samples).astype(np.float64)
    mean = x.mean(0)
    x -= mean
    _, evecs = np.linalg.eigh(x.T @ x)
    components = evecs[:, ::-1][:, :n_components]
    return SvdBasis(mean.astype(np.float32), components.astype(np.float32))


def project(matrix: np.ndarray, basis: SvdBasis) -> np.ndarray:
    return (matrix - basis.mean) @ basis.components


def zscore_columns(x: np.ndarray) -> np.ndarray:
    std = x.std(0)
    return (x - x.mean(0)) / np.where(std > 0, std, 1.0)


def lagged_design(features: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """``(T, K)`` -> ``(T, K * len(lags))``; column block *j* is delayed by ``lags[j]``."""
    return np.concatenate(
        [
            np.stack(
                [shift(features[:, k], int(lag)) for k in range(features.shape[1])], 1
            )
            for lag in lags
        ],
        axis=1,
    )


@dataclass(frozen=True)
class Gram:
    xtx: np.ndarray
    xty: np.ndarray
    yty: float
    n: int

    @classmethod
    def of(cls, x: np.ndarray, y: np.ndarray) -> "Gram":
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        return cls(x.T @ x, x.T @ y, float(y @ y), len(y))

    def __add__(self, other: "Gram") -> "Gram":
        return Gram(
            self.xtx + other.xtx,
            self.xty + other.xty,
            self.yty + other.yty,
            self.n + other.n,
        )

    def __sub__(self, other: "Gram") -> "Gram":
        return Gram(
            self.xtx - other.xtx,
            self.xty - other.xty,
            self.yty - other.yty,
            self.n - other.n,
        )


def total(grams: dict[int, Gram]) -> Gram:
    return functools.reduce(operator.add, grams.values())


def solve(gram: Gram, alpha: float) -> np.ndarray:
    return np.linalg.solve(gram.xtx + alpha * np.eye(len(gram.xty)), gram.xty)


def mse(gram: Gram, w: np.ndarray) -> float:
    return float((gram.yty - 2 * w @ gram.xty + w @ gram.xtx @ w) / gram.n)


def select_alpha(grams: dict[int, Gram]) -> tuple[float, list[dict]]:
    """Leave-one-session-out ridge strength; alphas scale with the mean feature energy."""
    whole = total(grams)
    scale = float(np.trace(whole.xtx)) / len(whole.xty)
    table = []
    for alpha in scale * np.logspace(-4, 2, 13):
        errors = [mse(g, solve(whole - g, alpha)) for g in grams.values()]
        table.append({"alpha": float(alpha), "loso_mse": float(np.mean(errors))})
    best = min(table, key=lambda row: row["loso_mse"])
    return best["alpha"], table
