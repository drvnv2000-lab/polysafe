import math

import pandas as pd
import pytest

from polysafe import cohort, labels
from polysafe.cohort import T_MIN

H = 60


def brute_force_onset(points, base):
    """KDIGO onset after T written straight from the HANDOFF text, independent of the package.

    points: [(offset, value)] for one stay. base: (offset, value) or None.
    """
    for tj, vj in sorted(points):
        if tj <= T_MIN:
            continue
        rise = any(ti < tj and tj - ti <= 48 * H and vj - vi >= 0.3 - 1e-9 for ti, vi in points)
        ratio = (base is not None and base[0] < tj <= base[0] + 7 * 24 * H
                 and vj >= 1.5 * base[1] - 1e-9)
        if rise or ratio:
            return tj
    return None


@pytest.fixture(scope="module")
def labelled(tables):
    return {b: labels.label(tables, b) for b in ("A", "C")}


@pytest.mark.parametrize("b", ["A", "C"])
def test_labels_match_an_independent_reimplementation(cr, labelled, b):
    out = labelled[b][0].set_index("patientunitstayid")
    by_stay = {s: list(zip(g["offset"], g["value"], strict=True))
               for s, g in cr[cr["patientunitstayid"].isin(out.index)].groupby("patientunitstayid")}
    for stay, row in out.iterrows():
        pts = by_stay[stay]
        if b == "A":
            in_window = [p for p in pts if 0 <= p[0] <= T_MIN]
            base = min(in_window) if in_window else None
        else:
            pre = [p for p in pts if p[0] < 0]
            base = max(pre) if pre else None
        expected = brute_force_onset(pts, base)
        assert row["aki"] == (expected is not None), stay
        if expected is not None:
            assert row["onset_offset"] == expected, stay


def test_incidence_regression(labelled):
    """Measured on the verified demo files; a change here means the logic changed."""
    a, c = labelled["A"][0], labelled["C"][0]
    assert (len(a), int(a["aki"].sum())) == (1034, 166)
    assert (len(c), int(c["aki"].sum())) == (779, 107)


def test_every_onset_is_after_T(labelled):
    for out, _, _ in labelled.values():
        assert (out.loc[out["aki"], "onset_offset"] > T_MIN).all()
        assert out.loc[~out["aki"], "onset_offset"].isna().all()
        assert set(out.loc[out["aki"], "criterion"]) <= {"rise", "ratio", "rise+ratio"}


def test_onset_criterion_on_synthetic_stay():
    cr = pd.DataFrame({"patientunitstayid": 1, "offset": [60, 30 * H, 40 * H],
                       "value": [1.0, 1.2, 1.6]})
    on = labels.onsets(cr, cohort.baseline_a(cr))
    # 40 h draw: 1.6 - 1.2 = 0.4 rise within 48 h, and 1.6 >= 1.5 x baseline 1.0
    assert on.loc[1, "offset"] == 40 * H and on.loc[1, "criterion"] == "rise+ratio"


def test_wilson_interval():
    lo, hi = labels.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    assert labels.wilson(0, 10)[0] == pytest.approx(0, abs=1e-12)
    assert all(math.isnan(x) for x in labels.wilson(0, 0))
