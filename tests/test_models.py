"""Section 8 models (D7.1, D7.2). Model-fitting tests use synthetic data only."""

import numpy as np
import pandas as pd
import pytest

from polysafe import ascertain, features, models
from polysafe.tables import load

CLIN = ("a", "b", "hx1", "hx2")
SPEC = models.Spec("toy", "lr", CLIN, (("b",), ("hx1", "hx2")))


def _frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "hx1": rng.integers(0, 2, n).astype(float),
                      "hx2": rng.integers(0, 2, n).astype(float)})
    for c in models.MED_COLS:
        X[c] = (rng.random(n) < 0.2).astype(float)
    return X


# ---------------------------------------------------------------- design (D3.6, D4.4, D7.1)


def test_record_less_stays_get_the_training_share_and_no_indicator():
    tr, te = _frame(), _frame(5, seed=1)
    tr.loc[:3, list(models.MED_COLS)] = np.nan  # record-less training stays
    te.loc[0, list(models.MED_COLS)] = np.nan
    share = tr["med_vancomycin"].mean()  # pandas skips NaN: stays with a record only
    x_tr, x_te = models.lr_design(tr, te, SPEC, medication=True)
    raw_tr, raw_te = models.lr_design(tr, te, SPEC, medication=False)
    assert x_tr.shape[1] == raw_tr.shape[1] + 5  # five medication columns, no indicator
    col = raw_tr.shape[1]  # med_vancomycin
    filled = tr["med_vancomycin"].fillna(share)
    expect = (share - filled.mean()) / filled.std(ddof=0)
    assert x_te[0, col] == pytest.approx(expect)
    assert 0 < share < 1


def test_d44_indicators_median_imputation_and_training_standardisation():
    tr, te = _frame(), _frame(4, seed=2)
    tr.loc[0, "b"] = np.nan
    tr.loc[1, "hx1"] = np.nan
    te.loc[0, "a"] = np.nan
    x_tr, x_te = models.lr_design(tr, te, SPEC, medication=False)
    assert x_tr.shape[1] == 4 + 2  # four columns plus the b and merged hx indicators
    assert np.allclose(x_tr.mean(axis=0), 0) and np.allclose(x_tr.std(axis=0)[:4], 1)
    a = tr["a"]
    assert x_te[0, 0] == pytest.approx((a.median() - a.mean()) / a.std(ddof=0))


def test_a_constant_training_column_is_set_to_zero():
    tr, te = _frame(), _frame(3, seed=3)
    tr["hx2"] = 1.0
    x_tr, x_te = models.lr_design(tr, te, models.Spec("t", "lr", CLIN, ()), medication=False)
    assert (x_tr[:, 3] == 0).all() and (x_te[:, 3] == 0).all()


def test_full_lr_has_one_indicator_per_column_with_a_gap_and_lgbm_keeps_nan():
    tr, te = _frame(), _frame(3, seed=4)
    tr.loc[0, "a"] = np.nan
    tr.loc[:2, list(models.MED_COLS)] = np.nan
    full = models.Spec("f", "lr", CLIN, None)
    x_tr, _ = models.lr_design(tr, te, full, medication=True)
    assert x_tr.shape[1] == 4 + 1 + 5  # only "a" has a gap among the clinical columns
    g_tr, _ = models.lgbm_design(tr, te, models.Spec("g", "lgbm", CLIN, None), medication=True)
    assert np.isnan(g_tr[:3, 4:]).all()


def test_specs_follow_d44_and_d72():
    s = {x.name: x for x in models.specs(["age", "lactate", "temp_max", "bun"])}
    assert s["lr_reduced"].clinical == features.PRIMARY_LR_FEATURES
    assert s["lr_full"].clinical == ("age", "bun")  # LR_EXCLUDED removed
    assert s["lgbm"].clinical == ("age", "lactate", "temp_max", "bun")
    assert len(models.C_GRID) == 13 and models.C_GRID[0] == pytest.approx(1e-3)
    assert models.C_GRID[-1] == pytest.approx(1e3)
    assert models.LGBM_GRID[0] == {"num_leaves": 4, "min_child_samples": 40, "n_estimators": 100}
    assert len(models.LGBM_GRID) == 18
    assert "ondansetron" not in models.MEDICATION and "aminoglycosides" not in models.MEDICATION


# ---------------------------------------------------------------- tuning and CV


def test_tuning_ties_go_to_the_first_candidate(monkeypatch):
    monkeypatch.setattr(models, "fit_predict", lambda *a: np.full((a[4].shape[0],), 0.3))
    rng = np.random.default_rng(5)
    X = _frame(60)
    y = rng.integers(0, 2, 60)
    best, losses = models.choose(SPEC, False, X, y, np.arange(60), seed=1)
    assert best == models.C_GRID[0] and len(set(losses)) == 1


def test_cross_validation_pools_every_stay_and_pairs_share_folds():
    rng = np.random.default_rng(6)
    n = 80
    X = _frame(n, seed=6)
    y = (X["a"] + rng.normal(size=n) > 0.8).astype(int).to_numpy()
    groups = np.repeat(np.arange(40), 2)
    spec = models.Spec("toy", "lr", CLIN, (("b",),))
    pooled, tuning = models.cross_validate(X, y, groups, [spec], repeats=2)
    assert set(pooled) == {("toy", False), ("toy", True)}
    assert all(v.shape == (n,) and np.isfinite(v).all() for v in pooled.values())
    t = tuning.pivot_table(index=["repeat", "fold"], columns="medication", values="param",
                           aggfunc="size")
    assert len(t) == 2 * 5 and (t == 1).all().all()


def test_lightgbm_fits_with_missing_medication_features():
    pytest.importorskip("lightgbm")
    X = _frame(60, seed=7)
    X.loc[:5, list(models.MED_COLS)] = np.nan
    y = (X["a"] > 0).astype(int).to_numpy()
    spec = models.Spec("g", "lgbm", CLIN, None)
    p = models.fit_predict(spec, True, X[:50], y[:50], X[50:], models.LGBM_GRID[0])
    assert p.shape == (10,) and np.isfinite(p).all()


# ---------------------------------------------------------------- metrics


def test_calibration_is_zero_and_one_for_calibrated_predictions():
    rng = np.random.default_rng(8)
    lp = rng.normal(-2, 1.2, 20000)
    p = 1 / (1 + np.exp(-lp))
    y = (rng.random(20000) < p).astype(int)
    icpt, slope = models.calibration(p, y)
    assert icpt == pytest.approx(0, abs=0.05) and slope == pytest.approx(1, abs=0.05)
    over = 1 / (1 + np.exp(-2 * lp))  # overconfident: slope about 0.5
    assert models.calibration(over, y)[1] == pytest.approx(0.5, abs=0.05)


def test_net_benefit_and_treat_all():
    y = np.array([1, 0, 0, 1, 0])
    p = np.array([0.9, 0.5, 0.05, 0.3, 0.2])
    assert models.net_benefit(p, y, 0.25) == pytest.approx(2 / 5 - 1 / 5 * 0.25 / 0.75)
    dca = models.decision_curve({"m": p}, y)
    assert dca["threshold"].iloc[0] == 0.02 and dca["threshold"].iloc[-1] == 0.4
    assert len(dca) == 39
    assert dca["treat_all"].iloc[0] == pytest.approx(0.4 - 0.6 * 0.02 / 0.98)


def test_logit_is_clipped():
    assert np.isfinite(models.logit(np.array([0.0, 1.0]))).all()


def test_matched_cell_picks_the_nearest_grid_point():
    rows = [{"cohort": models.PRIMARY_COHORT, "prevalence": 0.1162, "a0": a0, "rho": r,
             "delta": d, "bca_coverage": 0.9 + d, "clin_coverage": 0.95,
             "comb_coverage": 0.94}
            for a0 in (0.7, 0.75, 0.8) for r in (0.6, 0.75, 0.9, 0.95) for d in (0.0, 0.02)]
    cov = pd.DataFrame(rows)
    mde = pd.DataFrame([{"cohort": models.PRIMARY_COHORT, "prevalence": 0.1162, "a0": a0,
                         "rho": r, "mde_bca": a0 / 10 + r / 100, "mde_pct": 0.05}
                        for a0 in (0.7, 0.75, 0.8) for r in (0.6, 0.75, 0.9, 0.95)])
    m = models.matched_cell(cov, mde, 0.1162, a0=0.77, rho=0.93, delta=-0.01)
    assert (m["cell_a0"], m["cell_rho"], m["cell_delta"]) == (0.75, 0.95, 0.0)
    assert m["cov_dauroc"] == pytest.approx(0.9) and m["mde_bca"] == pytest.approx(0.0845)
    assert m["cov_dauroc_min"] == pytest.approx(0.9) and m["cov_dauroc_max"] == pytest.approx(0.92)


# ---------------------------------------------------------------- real data, no fitting


def test_medication_features_on_the_primary_cohort():
    med = load("medication")
    stays = ascertain.primary_stays()
    m = models.medication_features(stays, med)
    assert list(m.columns) == list(models.MED_COLS) and len(m) == 671
    no_record = ~stays.isin(med["patientunitstayid"].unique())
    assert no_record.sum() == 20  # DATA_REPORT.md: 20 record-less stays in the 671
    assert m[no_record].isna().all().all() and m[~no_record].notna().all().all()
    prev = pd.read_csv(models.RESULTS / "exposure_prevalence.csv")
    prev = prev[(prev["arm"] == models.ARM) & (prev["variant"] == models.VARIANT)]
    counts = prev.set_index("exposure")["exposed_stays"]
    for exposure, col in models.MEDICATION.items():
        assert int(m[col].sum()) == counts[exposure]


def test_evaluate_and_report_end_to_end_on_synthetic_predictions():
    rng = np.random.default_rng(9)
    n = 300
    y = (rng.random(n) < 0.15).astype(int)
    groups = np.repeat(np.arange(150), 2)
    base = rng.normal(size=n) + 1.2 * y
    pooled = {}
    for name in ("lr_reduced", "lgbm", "lr_full"):
        pooled[(name, False)] = 1 / (1 + np.exp(-(base - 2)))
        pooled[(name, True)] = 1 / (1 + np.exp(-(base + 0.3 * y + rng.normal(0, 0.2, n) - 2)))
    grid = [(a0, r, d) for a0 in (0.7, 0.75, 0.8) for r in (0.6, 0.75, 0.9, 0.95)
            for d in (0.0, 0.02, 0.04)]
    cov = pd.DataFrame([{"cohort": models.PRIMARY_COHORT, "prevalence": 0.1162, "a0": a0,
                         "rho": r, "delta": d, "bca_coverage": 0.94, "clin_coverage": 0.98,
                         "comb_coverage": 0.96} for a0, r, d in grid])
    mde = pd.DataFrame([{"cohort": models.PRIMARY_COHORT, "prevalence": 0.1162, "a0": a0,
                         "rho": r, "mde_bca": 0.05, "mde_pct": 0.05}
                        for a0 in (0.7, 0.75, 0.8) for r in (0.6, 0.75, 0.9, 0.95)])
    spec_list = models.specs(["age"])
    metrics, comparison, curves, dca = models.evaluate(pooled, y, groups, spec_list, cov, mde,
                                                       0.1162, n_boot=200)
    assert len(metrics) == 6 and len(comparison) == 3
    assert (metrics["auroc_n_attempted"] == 200).all()
    assert (comparison["lo"] < comparison["dauroc"]).all()
    assert (comparison["dauroc"] < comparison["hi"]).all()
    assert set(dca.columns) >= {"threshold", "treat_all", "treat_none", "lr_reduced_medication"}
    X = pd.DataFrame({c: [0.0] * n for c in models.MED_COLS})
    X.iloc[:20] = np.nan
    tuning = pd.DataFrame({"repeat": range(10)})
    md = models.report(metrics, comparison, X, y, tuning)
    assert "20 stays without a medication record" in md and "Paired dAUROC" in md
