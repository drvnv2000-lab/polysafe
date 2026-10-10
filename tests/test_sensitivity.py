"""D8.1 sensitivity analyses. Synthetic data only."""

import numpy as np
import pandas as pd

from polysafe import models, sensitivity


def test_hospital_clusters_merge_hospitals_that_share_a_patient():
    hospital = [1, 1, 2, 3, 4, 4]
    patient = ["a", "b", "b", "c", "d", "e"]  # b links hospitals 1 and 2
    ids = sensitivity.hospital_clusters(hospital, patient)
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] not in (ids[0], ids[4]) and ids[4] == ids[5]
    assert len(set(ids)) == 3


def test_medication_features_are_missing_for_record_less_stays():
    stays = pd.Index([1, 2, 3], name="patientunitstayid")
    t = pd.DataFrame(False, index=stays, columns=list(models.MEDICATION))
    t.loc[1, "furosemide"] = True
    m = sensitivity.medication_features({("a", "v"): t}, "a", "v", stays, recorded={1, 2})
    assert list(m.columns) == list(models.MED_COLS)
    assert m.loc[1, "med_furosemide"] == 1.0 and m.loc[2, "med_furosemide"] == 0.0
    assert m.loc[3].isna().all()  # never coded as unexposed (section 6, D7.1)


def test_folds_without_event_are_reported():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert sensitivity.folds_without_event(y, np.arange(10), repeats=1)
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert not sensitivity.folds_without_event(y, np.arange(10), repeats=1)


def _eth_inputs():
    rng = np.random.default_rng(0)
    idx = pd.Index(range(60), name="patientunitstayid")
    y = pd.Series(0, index=idx)
    y.iloc[:12] = 1  # group A: 12 events
    y.iloc[40:43] = 1  # group B: 3 events
    eth = pd.Series(["A"] * 40 + ["B"] * 15 + [None] * 5, index=idx)
    p = rng.random(60)
    preds = pd.DataFrame({"p_lr_reduced_clinical": p,
                          "p_lr_reduced_medication": p + 0.1 * y.to_numpy()}, index=idx)
    return preds, y, pd.Series(idx.astype(str), index=idx), eth


def test_ethnicity_rules():
    t = sensitivity.ethnicity_table(*_eth_inputs(), n_boot=200).set_index("ethnicity")
    assert t.loc["A", "interval"] == "BCa" and not np.isnan(t.loc["A", "lo"])
    assert "point estimate only" in t.loc["B", "interval"] and np.isnan(t.loc["B", "lo"])
    assert not np.isnan(t.loc["B", "dauroc"])
    assert t.loc["missing", "interval"] == "no event: no AUROC"
    assert np.isnan(t.loc["missing", "auroc_clinical"])


def test_report_states_d92_and_the_reproduction_check():
    row = {"analysis": "ascertainment:hicl_gtc:primary", "group": "ascertainment",
           "arm": models.ARM, "variant": models.VARIANT, "n": 10, "events": 2, "clusters": 9,
           "auroc_clinical": 0.6, "auroc_medication": 0.65, "dauroc": 0.05, "lo": 0.01,
           "hi": 0.09, "rho_logit": 0.9, "cov_dauroc": 0.94,
           "coverage_cohort": models.PRIMARY_COHORT}
    stopped = {"analysis": "hospital_clusters", "group": "hospital_clusters", "n": 10,
               "events": 2, "clusters": 3, "stopped": "outer test fold without an AKI event"}
    md = sensitivity.report(pd.DataFrame([row, stopped]), sensitivity.ethnicity_table(
        *_eth_inputs(), n_boot=200), pd.Series({"dauroc": 0.05, "lo": 0.01, "hi": 0.09}),
        repeats=10, n_boot=10_000)
    assert "after the primary result was seen (D9.2)" in md
    assert "reproduces the primary dAUROC" in md
    assert "(primary analysis)" in md and "stopped: outer test fold" in md
