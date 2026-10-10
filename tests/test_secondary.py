"""Section 10 / D11.1 secondary analysis. Synthetic data only."""

import numpy as np

from polysafe import secondary


def test_adjustment_removes_confounding_by_severity():
    rng = np.random.default_rng(0)
    n = 3000
    sev = rng.normal(size=n)
    a = (rng.random(n) < 1 / (1 + np.exp(-1.5 * sev))).astype(int)  # sicker -> exposed
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2 + 1.5 * sev)))).astype(int)  # no drug effect
    crude, adj, ps = secondary.odds_ratios(sev[:, None], a, y)
    assert crude > 2.0 and 0.75 < adj < 1.33


def test_analyse_reports_intervals_and_overlap():
    rng = np.random.default_rng(1)
    n = 400
    z = rng.normal(size=(n, 2))
    a = (rng.random(n) < 0.3).astype(int)
    y = (rng.random(n) < 0.2).astype(int)
    r = secondary.analyse(z, a, y, np.arange(n), n_boot=50)
    assert r["crude_lo"] < r["crude_or"] < r["crude_hi"]
    assert r["resamples_used"] == 50 and r["exposed"] == a.sum()
    assert {"exposed_outside_unexposed_range", "ps_c_statistic"} <= set(r)
