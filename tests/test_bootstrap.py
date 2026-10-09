"""Patient-level BCa bootstrap (D2.1, D2.1a)."""

import numpy as np
import pytest

from polysafe import bootstrap, power


def test_auroc_matches_definition_and_is_nan_for_one_class():
    y = np.array([0, 0, 1, 1])
    assert bootstrap.auroc(np.array([0.1, 0.4, 0.35, 0.8]), y) == pytest.approx(0.75)
    assert bootstrap.auroc(np.array([0.5, 0.5, 0.5, 0.5]), y) == pytest.approx(0.5)
    assert np.isnan(bootstrap.auroc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_resample_counts_are_fixed_by_d21a():
    assert bootstrap.N_BOOT == 10_000 and bootstrap.N_BOOT_SIM == 2_000


def test_bca_on_iid_mean_is_close_to_the_normal_interval():
    x = np.random.default_rng(1).normal(10, 2, 400)
    ci = bootstrap.bca(lambda i: x[i].mean(), np.arange(400), n_boot=4000)
    half = 1.96 * x.std(ddof=1) / np.sqrt(400)
    assert ci.estimate == pytest.approx(x.mean())
    assert ci.low == pytest.approx(x.mean() - half, abs=0.03)
    assert ci.high == pytest.approx(x.mean() + half, abs=0.03)
    assert abs(ci.accel) < 0.01
    assert (ci.n_attempted, ci.n_used, ci.n_dropped) == (4000, 4000, 0)


def test_bca_corrects_skew_where_percentile_does_not():
    # exponential data: the mean's sampling distribution is right-skewed, so BCa shifts right
    x = np.random.default_rng(2).exponential(1.0, 60)
    ci = bootstrap.bca(lambda i: x[i].mean(), np.arange(60), n_boot=4000)
    assert ci.accel > 0
    assert (ci.high - ci.estimate) > (ci.estimate - ci.low)


def test_resampling_keeps_every_patients_rows_together():
    groups = np.repeat(np.arange(30), 3)  # 30 patients x 3 stays
    x = np.random.default_rng(3).normal(size=90)
    seen = []

    def stat(idx):
        seen.append(idx)
        return float(x[idx].mean())

    bootstrap.bca(stat, groups, n_boot=50)
    for idx in seen:
        counts = np.bincount(groups[idx], minlength=30)
        assert not (counts % 3).any()


def test_undefined_resamples_are_dropped_and_counted():
    y = np.array([1, 1, 1] + [0] * 37)  # few positives: some resamples have none
    p = np.random.default_rng(4).random(40)
    ci = bootstrap.bca(bootstrap.delta_auroc(p, p[::-1], y), np.arange(40), n_boot=500)
    assert 0 < ci.n_dropped < 500 and ci.n_used + ci.n_dropped == ci.n_attempted == 500


def test_non_finite_bca_quantities_raise_and_never_fall_back():
    jack = np.array([0.1, 0.2, 0.4])
    with pytest.raises(bootstrap.BCaUndefined, match="bias correction"):
        bootstrap.bca_from(0.0, np.ones(100), jack)  # every replicate above the estimate
    with pytest.raises(bootstrap.BCaUndefined, match="acceleration"):
        bootstrap.bca_from(1.0, np.r_[np.zeros(50), np.full(50, 2.0)], np.ones(5))
    with pytest.raises(bootstrap.BCaUndefined, match="acceleration"):
        bootstrap.bca_from(1.0, np.r_[np.zeros(50), np.full(50, 2.0)],
                           np.array([0.1, np.nan, 0.3]))
    with pytest.raises(bootstrap.BCaUndefined):
        bootstrap.bca_from(1.0, np.full(10, np.nan), jack)


def test_bca_from_reduces_to_percentile_when_unbiased_and_unaccelerated():
    boot = np.linspace(-1, 1, 2001)  # symmetric around the estimate, so z0 = 0
    ci = bootstrap.bca_from(0.0, boot, np.array([-1.0, 0.0, 1.0]))  # symmetric: accel = 0
    assert ci.z0 == 0 and ci.accel == 0
    assert (ci.low, ci.high) == pytest.approx(tuple(np.quantile(boot, [0.025, 0.975])))


def test_patient_weights_give_the_same_auroc_as_resampled_rows():
    # the coverage simulation resamples by patient weights (power.weighted_auc); a
    # resample of whole patients must give the same AUROC either way
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(40), 2)
    y, p = rng.integers(0, 2, 80), rng.random(80)
    p[::7] = p[1]  # include ties
    _, members = bootstrap._cluster_rows(groups)
    draw = rng.integers(0, 40, 40)
    by_rows = bootstrap.auroc(p[np.concatenate([members[c] for c in draw])],
                              y[np.concatenate([members[c] for c in draw])])
    w = np.bincount(draw, minlength=40)[groups].astype(float)[None, :]
    assert power.weighted_auc(p, y, w)[0] == pytest.approx(by_rows)


def test_delta_auroc_is_paired_on_the_same_rows():
    y = np.array([0, 0, 1, 1, 0, 1])
    good, bad = np.array([.1, .2, .9, .8, .3, .7]), np.array([.9, .8, .1, .2, .7, .3])
    stat = bootstrap.delta_auroc(good, bad, y)
    assert stat(np.arange(6)) == pytest.approx(1.0)
