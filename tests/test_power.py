import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from polysafe import cv, labels, power


@pytest.fixture(scope="module")
def primary(tables):
    out = labels.label(tables, "A")[0]
    return out[out["primary_site"]].reset_index(drop=True)


def test_no_patient_spans_two_folds_on_the_real_cohort(primary):
    folds = cv.patient_folds(primary["aki"], primary["uniquepid"])
    assert folds.shape == (cv.N_REPEATS, len(primary))
    for r in range(cv.N_REPEATS):
        per_patient = pd.Series(folds[r]).groupby(primary["uniquepid"].to_numpy()).nunique()
        assert (per_patient == 1).all()


def test_folds_are_stratified(primary):
    folds = cv.patient_folds(primary["aki"], primary["uniquepid"])
    rates = primary.groupby(folds[0])["aki"].mean()
    assert rates.max() - rates.min() < 0.05


def test_weighted_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.round(rng.random(300), 2)  # rounding creates ties
    w = rng.integers(0, 3, (4, 300)).astype(float)
    for row in w:
        assert power.weighted_auc(p, y, row[None, :])[0] == pytest.approx(
            roc_auc_score(y, p, sample_weight=row))


def test_sigma_hits_the_target_auroc():
    s = power.sigma_for(0.75, 0.12)
    rng = np.random.default_rng(1)
    L = rng.standard_normal(100_000)
    y = L > np.quantile(L, 0.88)
    assert power.auc(L + s * rng.standard_normal(100_000), y) == pytest.approx(0.75, abs=0.005)


def test_kappa_gives_the_target_score_correlation():
    s1, s2 = power.sigma_for(0.75, 0.12), power.sigma_for(0.78, 0.12)
    k = power.kappa_for(0.9, s1, s2)
    rng = np.random.default_rng(2)
    L, z1, z2 = rng.standard_normal((3, 200_000))
    a, b = L + s1 * z1, L + s2 * (k * z1 + np.sqrt(1 - k * k) * z2)
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(0.9, abs=0.005)
    # Every rho in the grid is reachable; identical scores (rho = 1) are not when AUROCs differ.
    assert all(power.kappa_for(r, s1, s2) is not None for r in power.RHO_GRID)
    assert power.kappa_for(1.0, s1, s2) is None


def test_labels_have_exact_prevalence():
    pat = np.repeat(np.arange(100), 3)
    _, y = power.draw_labels(pat, 100, 0.5, 37, np.random.default_rng(3))
    assert y.sum() == 37


def test_pair_concordance():
    pat = np.array([0, 0, 1, 1, 1, 2])
    y = np.array([1, 1, 1, 0, 0, 1])
    # patient 0: 1 of 1 pairs both positive; patient 1: 0 of 3; patient 2 has one stay
    assert power.pair_concordance(y, pat) == pytest.approx(0.5)


def test_mde_interpolates_between_grid_points():
    g = pd.DataFrame({"delta": [0.0, 0.02, 0.04], "power": [0.03, 0.6, 0.9]})
    assert power.mde(g) == pytest.approx(0.02 + 0.2 / 0.3 * 0.02)
    assert np.isnan(power.mde(g.assign(power=[0.0, 0.1, 0.5])))
