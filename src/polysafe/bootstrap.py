"""Patient-level BCa bootstrap confidence intervals (PREREGISTRATION D2.1, D2.1a).

D2.1 replaces the percentile bootstrap with a patient-level BCa (bias-corrected and
accelerated) interval; D2.1a fixes how it is computed:

* a resample draws patients (``uniquepid``) with replacement and keeps every stay of a
  drawn patient; the acceleration comes from the leave-one-patient-out jackknife;
* every real interval uses ``N_BOOT`` = 10,000 resamples (the coverage simulation uses
  ``N_BOOT_SIM`` = 2,000 per simulated study);
* the statistic is computed on fixed out-of-fold predictions: nothing is refitted;
* a resample on which the statistic is undefined (NaN, e.g. one outcome class) is
  dropped, not redrawn, and counted;
* a non-finite bias correction, acceleration or adjusted percentile raises
  ``BCaUndefined``: no interval, and no other method is substituted.

``bca_from`` is the formula alone, given the full-sample estimate, the bootstrap
replicates and the jackknife values; ``bca`` produces those from a ``stat`` that takes
row indices. The coverage simulation computes its replicates with patient weights and
calls ``bca_from``, so the real intervals and the simulated ones share one formula.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from polysafe import power

SEED = 20261004
N_BOOT = 10_000  # D2.1a: every interval reported for the real cohort
N_BOOT_SIM = 2_000  # D2.1a: per simulated study in the coverage re-run
N = NormalDist()


class BCaUndefined(ValueError):
    """The BCa interval does not exist for this statistic (D2.1a): report the failure."""


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    method: str  # "bca"
    n_attempted: int  # resamples drawn
    n_used: int  # resamples on which the statistic was defined
    n_dropped: int  # resamples on which it was undefined
    z0: float  # bias correction
    accel: float  # acceleration


def bca_from(theta: float, boot, jack, level: float = 0.95) -> Interval:
    """BCa interval from the estimate, the bootstrap replicates and the jackknife values.

    NaN replicates are dropped and counted. Raises ``BCaUndefined`` when the bias
    correction, the acceleration or an adjusted percentile is not finite.
    """
    boot, jack = np.asarray(boot, float), np.asarray(jack, float)
    ok = boot[~np.isnan(boot)]
    if not np.isfinite(theta) or len(ok) == 0:
        raise BCaUndefined("estimate or every resample undefined")
    # bias correction: share of resamples below the estimate, ties counted half
    p = (np.sum(ok < theta) + 0.5 * np.sum(ok == theta)) / len(ok)
    if not 0 < p < 1:
        raise BCaUndefined(f"bias correction not finite (p = {p})")
    z0 = N.inv_cdf(p)
    # acceleration: skewness of the leave-one-patient-out jackknife values
    d = jack.mean() - jack
    with np.errstate(invalid="ignore", divide="ignore"):
        accel = float(np.sum(d ** 3) / (6 * np.sum(d ** 2) ** 1.5))
    if not np.isfinite(accel):
        raise BCaUndefined("acceleration not finite")
    z = N.inv_cdf(0.5 + level / 2)
    qs = []
    for zq in (-z, z):
        with np.errstate(invalid="ignore", divide="ignore"):
            adj = z0 + (z0 + zq) / (1 - accel * (z0 + zq))
        if not np.isfinite(adj):
            raise BCaUndefined("adjusted percentile not finite")
        qs.append(N.cdf(adj))
    low, high = np.quantile(ok, qs)
    return Interval(float(theta), float(low), float(high), "bca", len(boot), len(ok),
                    len(boot) - len(ok), float(z0), accel)


def _cluster_rows(groups) -> tuple[np.ndarray, list[np.ndarray]]:
    groups = np.asarray(groups)
    ids, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.cumsum(np.bincount(inverse, minlength=len(ids)))[:-1]
    return ids, np.split(order, bounds)


def bca(stat: Callable[[np.ndarray], float], groups, n_boot: int = N_BOOT,
        level: float = 0.95, seed: int = SEED) -> Interval:
    """Patient-level BCa interval for ``stat`` over rows clustered by ``groups``."""
    ids, members = _cluster_rows(groups)
    n = len(ids)
    if n < 2:
        raise ValueError("BCa needs at least two clusters")
    theta = float(stat(np.arange(len(np.asarray(groups)))))
    draws = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    boot = [stat(np.concatenate([members[c] for c in d])) for d in draws]
    jack = [stat(np.concatenate([members[j] for j in range(n) if j != i])) for i in range(n)]
    return bca_from(theta, boot, jack, level)


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """AUROC with ties counted half (``power.auc``); NaN if only one class is present."""
    y = np.asarray(y).astype(bool)
    if y.all() or not y.any():
        return float("nan")
    return power.auc(np.asarray(score, float), y)


def delta_auroc(p_with: np.ndarray, p_without: np.ndarray, y: np.ndarray):
    """``stat`` for the primary estimand: AUROC(with medication) - AUROC(without), paired."""
    p_with, p_without, y = map(np.asarray, (p_with, p_without, y))

    def stat(idx: np.ndarray) -> float:
        return auroc(p_with[idx], y[idx]) - auroc(p_without[idx], y[idx])

    return stat
