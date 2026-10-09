"""BCa coverage re-run of the section 9.1 simulation (D2.1a)."""

import numpy as np
import pandas as pd
import pytest

from polysafe import coverage, power

RESULTS = coverage.RESULTS


def test_tasks_are_the_section_9_1_tasks():
    tasks = coverage.tasks_from(RESULTS / "power_cohorts.csv", sims=3)
    raw = pd.read_csv(RESULTS / "power_simulation_raw.csv.gz")
    assert {(t[0], t[1]) for t in tasks} == set(zip(raw["cohort"], raw["prevalence"],
                                                     strict=True))
    assert len(tasks) == 2 * 3 * 3 and {t[2] for t in tasks} == {0.0}


def test_one_study_reproduces_the_committed_9_1_study(monkeypatch):
    # a reduced grid: the random stream is drawn before the grid loops, so cells match
    monkeypatch.setattr(power, "A0_GRID", (0.75,))
    monkeypatch.setattr(power, "DELTA_GRID", (0.0, 0.04))
    monkeypatch.setattr(power, "RHO_GRID", (0.75,))
    cohorts = power.load_cohorts(RESULTS / "labels_A.csv")
    name, prev, tau2, j = coverage.tasks_from(RESULTS / "power_cohorts.csv", sims=1)[0]
    power._init(cohorts, {})
    raw = pd.DataFrame(coverage.one_study((name, prev, tau2, j)))
    committed = pd.read_csv(RESULTS / "power_simulation_raw.csv.gz")
    committed = committed[(committed["cohort"] == name) & (committed["prevalence"] == prev)
                          & (committed["sim"] == j) & (committed["a0"] == 0.75)
                          & committed["delta"].isin([0.0, 0.04]) & (committed["rho"] == 0.75)]
    assert len(raw) == len(committed) == 2
    assert coverage.check_reproduces(raw, committed) <= 1e-12
    assert raw["bca_ok"].all() and raw["clin_ok"].all() and raw["comb_ok"].all()
    assert (raw["bca_lo"] < raw["est"]).all() and (raw["est"] < raw["bca_hi"]).all()


def test_check_reproduces_refuses_a_different_study():
    a = pd.DataFrame({"cohort": ["c"], "prevalence": [0.1], "a0": [0.75], "delta": [0.0],
                      "rho": [0.6], "sim": [0], "feasible": [True], "est": [0.01],
                      "auc_clin": [0.7], "lo": [-0.05], "hi": [0.06]})
    assert coverage.check_reproduces(a, a.copy()) == 0
    with pytest.raises(ValueError, match="does not reproduce"):
        coverage.check_reproduces(a, a.assign(lo=-0.04))


def test_coverage_counts_hits_and_leaves_failed_intervals_out():
    lo = pd.Series([0.0, 0.2, np.nan])
    hi = pd.Series([0.5, 0.4, np.nan])
    got = coverage.covered(lo, hi, pd.Series([0.1, 0.1, 0.1]))
    assert got.iloc[0] == 1.0 and got.iloc[1] == 0.0 and np.isnan(got.iloc[2])
    assert got.mean() == 0.5
