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
    assert X.shape == (868, 74)
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
    vp = t["vitalPeriodic"]
    vp.loc[vp["observationoffset"] > T_MIN, ["heartrate", "respiration", "sao2"]] = 60.0
    adx = t["admissionDx"]
    adx.loc[adx["admitdxenteredoffset"] > T_MIN, "admitdxpath"] = (
        "admission diagnosis|All Diagnosis|Operative|Diagnosis|Trauma|x")
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


def test_pre_icu_lookback_is_seven_days():
    week = features.PRE_ICU_LOOKBACK_MIN
    lab = pd.DataFrame({"patientunitstayid": [1, 2], "labresultoffset": [-week, -week - 1],
                        "labname": ["albumin", "albumin"], "labresult": [3.0, 3.0]})
    out = features.lab_features(lab)
    assert out.loc[1, "albumin"] == 3.0 and 2 not in out.index


def test_temperature_in_fahrenheit_is_converted():
    vp = pd.DataFrame({"patientunitstayid": [1, 1, 2], "observationoffset": [60, 120, 60],
                       "heartrate": 80.0, "respiration": 18.0, "sao2": 97.0,
                       "temperature": [98.6, 37.0, 101.3]})
    out = features.periodic_vital_features(vp)
    assert out.loc[1, "temp_max"] == pytest.approx(37.0)
    assert out.loc[2, "temp_max"] == pytest.approx((101.3 - 32) * 5 / 9)


def test_admission_diagnosis_entered_after_T_is_missing():
    p = pd.DataFrame({"patientunitstayid": [1, 2], "unittype": ["MICU", "MICU"],
                      "unitadmitsource": ["Floor", None]})
    adx = pd.DataFrame({"patientunitstayid": [1, 2], "admitdxenteredoffset": [30, T_MIN + 5],
                        "admitdxpath": ["admission diagnosis|All Diagnosis|Non-operative|"
                                        "Diagnosis|Respiratory|Pneumonia"] * 2})
    out = features.admission_context(p, adx)
    assert out.loc[1, "dx_respiratory"] == 1 and out.loc[1, "dx_operative"] == 0
    assert out.loc[2, ["dx_operative", "dx_respiratory", "dx_gi"]].isna().all()
    assert out.loc[2, ["src_floor", "src_ed"]].isna().all()  # unknown source: missing, not 0
    assert out.loc[1, "src_floor"] == 1 and out.loc[1, "unit_medical"] == 1


def test_ethnicity_is_not_a_feature(X):
    assert not [c for c in X.columns if "ethnic" in c.lower() or "race" in c.lower()]


def test_lr_exclusions_are_real_columns(X):
    assert set(features.LR_EXCLUDED) <= set(X.columns)


def test_primary_lr_spec_matches_d4(X):
    """D4.4: 14 variables + 3 indicators = 17 parameters, none of them LR-excluded."""
    assert len(features.PRIMARY_LR_FEATURES) == 14
    assert set(features.PRIMARY_LR_FEATURES) <= set(X.columns)
    assert not set(features.PRIMARY_LR_FEATURES) & set(features.LR_EXCLUDED)
    flat = [c for group in features.PRIMARY_LR_INDICATORS for c in group]
    assert set(flat) <= set(features.PRIMARY_LR_FEATURES)
    assert len(features.PRIMARY_LR_FEATURES) + len(features.PRIMARY_LR_INDICATORS) == 17
    hx = X[["hx_ckd", "hx_chf", "hx_diabetes"]].isna()
    assert (hx.nunique(axis=1) == 1).all()  # the merged indicator is exact
