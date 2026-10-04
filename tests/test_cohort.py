import pandas as pd
import pytest

from polysafe import cohort
from polysafe.cohort import T_MIN

H = 60  # minutes per hour


def series(*points, stay=1):
    """Creatinine rows for one synthetic stay: (offset_minutes, value) pairs."""
    return pd.DataFrame(
        [{"patientunitstayid": stay, "offset": t, "value": v} for t, v in points]
    )


# ------------------------------------------------------------ creatinine


def test_serum_creatinine_excludes_urinary(tables, cr):
    assert len(cr) == 14_603  # DATA_REPORT rev 2: serum rows with a value
    urinary = tables["lab"]["labname"].eq("urinary creatinine").sum()
    assert urinary > 0 and cr["value"].notna().all()


def test_baseline_availability_matches_handoff(cr):
    a, c = cohort.baseline_a(cr), cohort.baseline_c(cr)
    assert len(a) == 2_092 and len(c) == 1_687  # HANDOFF.md, measured independently
    assert a["offset"].between(0, T_MIN).all()
    assert (c["offset"] < 0).all()


def test_baseline_a_is_first_draw_in_window():
    cr = series((-60, 9.0), (30, 1.0), (600, 2.0))
    assert cohort.baseline_a(cr).loc[1, "value"] == 1.0


def test_baseline_c_is_latest_pre_icu_draw():
    cr = series((-600, 1.0), (-60, 1.2), (30, 9.0))
    assert cohort.baseline_c(cr).loc[1, "value"] == 1.2


# ------------------------------------------------------------ KDIGO rules


@pytest.mark.parametrize(
    "points, hit",
    [
        (((0, 1.0), (47 * H, 1.3)), True),  # 0.3 rise inside 48 h
        (((0, 1.0), (49 * H, 1.3)), False),  # same rise, outside 48 h
        (((0, 1.0), (10 * H, 1.29)), False),  # just below threshold
        (((0, 2.0), (10 * H, 2.3)), True),  # 2.3 - 2.0 is 0.2999... in floating point
        (((0, 1.3), (10 * H, 1.0)), False),  # a fall is not a rise
        (((60, 1.0), (60, 1.4)), False),  # simultaneous draws are not a rise
    ],
)
def test_rise_rule(points, hit):
    # the 48 h window in isolation: no T cut-off in play
    assert (1 in cohort.rise_by(series(*points), until=7 * 24 * H)) is hit


def test_rise_rule_ignores_draws_after_until():
    cr = series((0, 1.0), (T_MIN + 1, 5.0))
    assert cohort.rise_by(cr, until=T_MIN) == set()


def test_rise_rule_pre_icu_toggle():
    cr = series((-4 * H, 1.0), (6 * H, 1.4))
    assert cohort.rise_by(cr, T_MIN, include_pre_icu=True) == {1}
    assert cohort.rise_by(cr, T_MIN, include_pre_icu=False) == set()


@pytest.mark.parametrize(
    "points, hit",
    [
        (((60, 1.0), (20 * H, 1.5)), True),  # 1.5x baseline before T
        (((60, 1.0), (20 * H, 1.49)), False),
        (((60, 1.0), (T_MIN + H, 3.0)), False),  # after T: outcome, not exclusion
    ],
)
def test_ratio_rule_against_baseline_a(points, hit):
    cr = series(*points)
    assert (1 in cohort.ratio_by(cr, cohort.baseline_a(cr), until=T_MIN)) is hit


def test_ratio_rule_window_is_seven_days_from_baseline():
    cr = series((-8 * 24 * H, 1.0), (60, 2.0))  # baseline C drawn 8 days before
    assert cohort.ratio_by(cr, cohort.baseline_c(cr), until=T_MIN) == set()


# ------------------------------------------------------------ ESRD


def test_esrd_markers_on_real_data(tables):
    m = cohort.esrd_markers(tables["pastHistory"], tables["diagnosis"],
                            tables["treatment"], tables["apacheApsVar"])
    by_source = m.groupby("source")["patientunitstayid"].apply(set)
    assert m["patientunitstayid"].nunique() == 89
    # APACHE's chronic-dialysis flag is fully corroborated by charted past history.
    assert by_source["APACHE dialysis"] <= by_source["pastHistory dialysis"]


def test_acute_dialysis_is_not_an_esrd_marker(tables):
    tr = tables["treatment"]
    acute = set(tr.loc[tr["treatmentstring"].str.contains(
        r"C V V H|for acute renal failure|emergent", na=False), "patientunitstayid"])
    tr_only = tr[tr["patientunitstayid"].isin(acute)]
    empty = tables["pastHistory"].iloc[0:0]
    m = cohort.esrd_markers(empty, tables["diagnosis"].iloc[0:0], tr_only,
                            tables["apacheApsVar"].iloc[0:0])
    chronic = set(tr_only.loc[tr_only["treatmentstring"].str.contains(
        "for chronic renal failure", na=False), "patientunitstayid"])
    assert set(m["patientunitstayid"]) == chronic


# ------------------------------------------------------------ cohort


@pytest.fixture(scope="module")
def built(tables):
    return {b: cohort.build_cohort(tables, baseline=b) for b in ("A", "C")}


def test_flow_is_arithmetically_consistent(built):
    for final, flow in built.values():
        for prev, step in zip(flow, flow[1:], strict=False):
            assert step.removed <= prev.remaining
            assert step.remaining == prev.remaining - step.removed
        assert flow[-1].remaining == set(final["patientunitstayid"])


def test_flow_counts_regression(built):
    """Measured on the verified demo files; a change here means the logic changed."""
    assert [len(s.remaining) for s in built["A"][1]] == [2520, 1627, 1565, 1397, 1394, 1248, 1034]
    assert [len(s.remaining) for s in built["C"][1]] == [2520, 1627, 1565, 1394, 1391, 935, 779]


def test_acute_rrt_step_removes_the_three_pi_confirmed_stays(built):
    rrt_step = next(s for s in built["A"][1] if s.name == "Acute RRT by T")
    assert rrt_step.removed == {826668, 1770612, 2860223}


def test_rrt_rule_ignores_access_procedures():
    tr = pd.DataFrame({
        "patientunitstayid": [1, 2, 3],
        "treatmentoffset": [100, 100, 100],
        "treatmentstring": ["renal|dialysis|insertion of venous catheter for hemodialysis",
                            "renal|dialysis|C V V H D",
                            "renal|dialysis|hemodialysis|emergent"],
    })
    assert set(cohort.rrt_start(tr).index) == {2, 3}


@pytest.mark.parametrize("b", ["A", "C"])
def test_final_cohort_respects_every_exclusion(tables, cr, built, b):
    final, _ = built[b]
    ids = set(final["patientunitstayid"])
    base = cohort.BASELINES[b](cr)
    assert (final["unitdischargeoffset"] >= T_MIN).all()
    esrd = cohort.esrd_markers(tables["pastHistory"], tables["diagnosis"],
                               tables["treatment"], tables["apacheApsVar"])
    assert not ids & set(esrd["patientunitstayid"])
    assert not ids & cohort.aki_by(cr, base)
    rrt = cohort.rrt_start(tables["treatment"])
    assert not ids & set(rrt[rrt <= T_MIN].index)
    assert ids <= set(base.index)
    assert ids <= set(cr.loc[cr["offset"] > T_MIN, "patientunitstayid"])


def test_creatinine_after_T_cannot_change_the_cohort(tables):
    """Leakage guard: rewrite every creatinine value drawn after T; the cohort must not move.

    (The last step does use the *presence* of a draw after T, to require an
    observable outcome; values after T must never matter.)
    """
    lab = tables["lab"].copy()
    after = (lab["labname"] == "creatinine") & (lab["labresultoffset"] > T_MIN)
    lab.loc[after, "labresult"] = lab.loc[after, "labresult"] * 10
    # Extra draws only where one with a value already exists, so presence is unchanged
    # (e.g. stay 624150's only post-T creatinine rows are blank).
    future = lab[after & lab["labresult"].notna()].assign(
        labresultoffset=lambda d: d["labresultoffset"] + 1, labresult=0.01)
    perturbed = {**tables, "lab": pd.concat([lab, future], ignore_index=True)}
    for b in ("A", "C"):
        before = set(cohort.build_cohort(tables, baseline=b)[0]["patientunitstayid"])
        after_ = set(cohort.build_cohort(perturbed, baseline=b)[0]["patientunitstayid"])
        assert before == after_
