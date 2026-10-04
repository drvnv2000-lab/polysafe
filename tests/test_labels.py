import math

import pandas as pd
import pytest

from polysafe import cohort, labels
from polysafe.cohort import T_MIN

H = 60
DAY = 24 * H


def brute_force_onset(points, base, end, rrt):
    """Onset written straight from PREREGISTRATION section 4, independent of the package.

    points: [(offset, value)] for one stay. base: (offset, value) or None.
    end: min(T + 7 d, ICU discharge). rrt: first RRT offset or None.
    """
    candidates = []
    for tj, vj in sorted(points):
        if not T_MIN < tj <= end:
            continue
        rise = any(ti < tj and tj - ti <= 48 * H and vj - vi >= 0.3 - 1e-9 for ti, vi in points)
        ratio = (base is not None and base[0] < tj <= base[0] + 7 * DAY
                 and vj >= 1.5 * base[1] - 1e-9)
        if rise or ratio:
            candidates.append(tj)
            break
    if rrt is not None and T_MIN < rrt <= end:
        candidates.append(rrt)
    return min(candidates) if candidates else None


@pytest.fixture(scope="module")
def labelled(tables):
    return {b: labels.label(tables, b) for b in ("A", "C")}


@pytest.mark.parametrize("b", ["A", "C"])
def test_labels_match_an_independent_reimplementation(tables, cr, labelled, b):
    out = labelled[b][0].set_index("patientunitstayid")
    discharge = tables["patient"].set_index("patientunitstayid")["unitdischargeoffset"]
    rrt = cohort.rrt_start(tables["treatment"])
    by_stay = {s: list(zip(g["offset"], g["value"], strict=True))
               for s, g in cr[cr["patientunitstayid"].isin(out.index)].groupby("patientunitstayid")}
    for stay, row in out.iterrows():
        pts = by_stay.get(stay, [])
        if b == "A":
            in_first_day = [p for p in pts if 0 <= p[0] <= T_MIN]
            base = min(in_first_day) if in_first_day else None
        else:
            pre = [p for p in pts if p[0] < 0]
            base = max(pre) if pre else None
        end = min(T_MIN + 7 * DAY, discharge[stay])
        expected = brute_force_onset(pts, base, end, rrt.get(stay))
        assert row["aki"] == (expected is not None), stay
        if expected is not None:
            assert row["onset_offset"] == expected, stay


def test_incidence_regression(labelled):
    """Measured on the verified demo files; a change here means the logic changed."""
    for b, n, k, n_prim, k_prim in [("A", 868, 107, 671, 78), ("C", 660, 71, 497, 49)]:
        out = labelled[b][0]
        prim = out[out["primary_site"]]
        assert (len(out), int(out["aki"].sum())) == (n, k)
        assert (len(prim), int(prim["aki"].sum())) == (n_prim, k_prim)


def test_every_onset_is_inside_the_window(tables, labelled):
    discharge = tables["patient"].set_index("patientunitstayid")["unitdischargeoffset"]
    for out, _, _ in labelled.values():
        pos = out[out["aki"]]
        end = (discharge.loc[pos["patientunitstayid"]].clip(upper=T_MIN + 7 * DAY)).to_numpy()
        assert (pos["onset_offset"] > T_MIN).all()
        assert (pos["onset_offset"].to_numpy() <= end).all()
        assert out.loc[~out["aki"], "onset_offset"].isna().all()
        assert set(pos["criterion"]) <= {"rise", "ratio", "rise+ratio", "rrt", "rise+rrt",
                                         "ratio+rrt", "rise+ratio+rrt"}


def test_followup_is_capped_at_seven_days(labelled):
    for out, _, _ in labelled.values():
        assert out["followup_h"].between(0, 7 * 24).all()


def _stay(points, discharge):
    cr = pd.DataFrame({"patientunitstayid": 1, "offset": [t for t, _ in points],
                       "value": [v for _, v in points]})
    end = pd.Series({1: min(T_MIN + 7 * DAY, discharge)})
    return cr, end


def test_onset_criterion_on_synthetic_stay():
    cr, end = _stay([(60, 1.0), (30 * H, 1.2), (40 * H, 1.6)], discharge=10 * DAY)
    on = labels.onsets(cr, cohort.baseline_a(cr), pd.Series(dtype=float), end)
    # 40 h draw: 1.6 - 1.2 = 0.4 rise within 48 h, and 1.6 >= 1.5 x baseline 1.0
    assert on.loc[1, "offset"] == 40 * H and on.loc[1, "criterion"] == "rise+ratio"


def test_creatinine_after_icu_discharge_is_censored():
    cr, end = _stay([(60, 1.0), (30 * H, 2.0)], discharge=28 * H)
    on = labels.onsets(cr, cohort.baseline_a(cr), pd.Series(dtype=float), end)
    assert on.empty


def test_rrt_in_window_is_aki_without_creatinine():
    cr, end = _stay([(60, 1.0), (30 * H, 1.0)], discharge=10 * DAY)
    on = labels.onsets(cr, cohort.baseline_a(cr), pd.Series({1: 50 * H}), end)
    assert on.loc[1, "offset"] == 50 * H and on.loc[1, "criterion"] == "rrt"


def test_rrt_after_window_is_not_aki():
    cr, end = _stay([(60, 1.0), (30 * H, 1.0)], discharge=3 * DAY)
    on = labels.onsets(cr, cohort.baseline_a(cr), pd.Series({1: 4 * DAY}), end)
    assert on.empty


def test_primary_population_is_site_level(tables, labelled):
    sites = cohort.recording_hospitals(tables["patient"], tables["medication"])
    assert len(sites) == 186 - 42  # PREREGISTRATION section 6
    out = labelled["A"][0]
    assert (out["primary_site"] == out["hospitalid"].isin(sites)).all()


def test_wilson_interval():
    lo, hi = labels.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    assert labels.wilson(0, 10)[0] == pytest.approx(0, abs=1e-12)
    assert all(math.isnan(x) for x in labels.wilson(0, 0))
