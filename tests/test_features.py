import numpy as np
import pandas as pd
import pytest

from polysafe import features
from polysafe.cohort import T_MIN


@pytest.fixture(scope="module")
def inputs():
    return features.load_feature_inputs()


@pytest.fixture(scope="module")
def stays():
    from polysafe import labels
    from polysafe.cohort import load_inputs
    return labels.label(load_inputs(), "A")[0]["patientunitstayid"]


@pytest.fixture(scope="module")
def X(inputs, stays):
    return features.build(inputs, stays)


def test_shape_and_no_apache(X, inputs, stays):
    assert X.shape == (868, 48)
    assert X.index.equals(pd.Index(stays))
    assert not [c for c in X.columns if "apache" in c.lower()]
    assert not [t for t in inputs if "apache" in t.lower()]


def test_nothing_after_T_can_change_a_feature(inputs, stays, X):
    """Leakage guard: rewrite every row after T in every time-stamped table."""
    t = {k: v.copy() for k, v in inputs.items()}
    lab = t["lab"]
    after = lab["labresultoffset"] > T_MIN
    lab.loc[after, "labresult"] *= 10
    extra = lab[after].assign(labresultoffset=lambda d: d["labresultoffset"] + 1, labresult=99.0)
    t["lab"] = pd.concat([lab, extra], ignore_index=True)
    v = t["vitalAperiodic"]
    v.loc[v["observationoffset"] > T_MIN, ["noninvasivesystolic", "noninvasivemean"]] = 41.0
    io = t["intakeOutput"]
    io.loc[io["intakeoutputoffset"] > T_MIN, "cellvaluenumeric"] *= 10
    ph = t["pastHistory"]
    late = ph["pasthistoryenteredoffset"] > T_MIN
    ph.loc[late, "pasthistorypath"] = "x/Cardiovascular (R)/Congestive Heart Failure/x"
    pd.testing.assert_frame_equal(features.build(t, stays), X)


def test_value_at_T_counts_and_one_minute_later_does_not():
    lab = pd.DataFrame({"patientunitstayid": [1, 1], "labresultoffset": [T_MIN, T_MIN + 1],
                        "labname": ["sodium", "sodium"], "labresult": [140.0, 120.0]})
    assert features.lab_features(lab).loc[1, "na"] == 140.0


def test_age_over_89_is_coded_90(inputs):
    p = inputs["patient"]
    demo = features.demographics(p)
    over = p.loc[p["age"] == "> 89", "patientunitstayid"]
    assert (demo.loc[over, "age"] == 90).all()


def test_comorbidities_missing_when_no_history_by_T():
    ph = pd.DataFrame({"patientunitstayid": [1, 2, 3],
                       "pasthistoryenteredoffset": [60, 60, T_MIN + 60],
                       "pasthistorypath": ["a/Cardiovascular (R)/Congestive Heart Failure/b",
                                           "a/No Health Problems/b",
                                           "a/Gastrointestinal (R)/Cirrhosis/b"]})
    c = features.comorbidities(ph)
    assert c.loc[1, "hx_chf"] == 1 and c.loc[2, "hx_chf"] == 0
    assert 3 not in c.index  # history entered only after T: unknown, not "absent"


def test_every_comorbidity_pattern_matches_real_rows(inputs):
    paths = inputs["pastHistory"]["pasthistorypath"]
    for name, pattern in features.COMORBIDITIES.items():
        assert paths.str.contains(pattern, na=False).any(), name


def test_urine_counts_are_not_volumes():
    io = pd.DataFrame({"patientunitstayid": [1, 1, 1], "intakeoutputoffset": [60, 120, 180],
                       "cellpath": ["flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine",
                                    "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Urine Count",
                                    "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|"
                                    "Mixed Urine/Stool Volume"],
                       "cellvaluenumeric": [300.0, 1.0, 400.0]})
    u = features.urine_features(io, pd.Series({1: 75.0}))
    assert u.loc[1, "urine_ml_0_24h"] == 300.0
    assert u.loc[1, "urine_ml_kg_h"] == pytest.approx(300 / 75 / 24)


def test_unmeasured_stays_stay_missing(X):
    assert X["lactate"].isna().any() and not np.isinf(X.select_dtypes("number")).any().any()
