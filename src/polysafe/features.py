"""Clinical features at T (PREREGISTRATION section 8, clinical-only covariates).

Rule: **only data with offset <= T** (section 3, "no exception"). Every
time-stamped table is cut at T by ``_at_or_before_t`` before anything else is
computed, and ``tests/test_features.py`` rewrites every row after T to prove
the features do not move.

Missing stays missing (NaN). Absence of a record is never coded as normal or
as "condition absent", except for comorbidity flags of stays whose past
history was taken (see ``comorbidities``).

Pre-registration deviations implemented here:
  D3.1  APACHE scores excluded: eICU's APACHE day-1 values use lab results
        drawn after T for 74 of 868 stays (up to 7.3 h after T).
  D3.2  vitalPeriodic: heart rate, respiratory rate, temperature, SpO2.
  D3.3  unit type, admission source, coarse admission diagnosis.
  D3.4  ethnicity is not a feature (written separately for subgroup reporting).
  D3.6  ``LR_EXCLUDED``: labs left out of the logistic regression only.
  D3.7  pre-ICU measured values (labs, creatinine) look back at most 7 days.

Usage:
    python -m polysafe.features   # writes results/features_clinical.csv, FEATURES.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from polysafe import cohort
from polysafe.cohort import T_MIN
from polysafe.tables import load

PRE_ICU_LOOKBACK_MIN = 7 * 24 * 60  # D3.7

# D3.6 as amended by D4.2-D4.3: above 40% missing in the primary cohort, plus
# temperature. Kept for LightGBM; left out of every logistic regression.
LR_EXCLUDED = ("lactate", "ph", "phos", "inr", "temp_max")

# D4.4: the pre-specified primary logistic regression (14 variables / 17 parameters).
PRIMARY_LR_FEATURES = ("age", "cr_baseline", "cr_max", "bun", "hco3", "albumin", "hgb", "wbc",
                       "map_min", "urine_ml_kg_h", "hx_ckd", "hx_chf", "hx_diabetes",
                       "dx_operative")
# Missingness indicators in the primary LR: one each, plus one merged for the hx_* flags.
PRIMARY_LR_INDICATORS = (("urine_ml_kg_h",), ("albumin",), ("hx_ckd", "hx_chf", "hx_diabetes"))

# Most recent value at or before T. Panel: AKI-relevant labs available at T for
# >= 40% of the cohort.
LAB_PANEL = {
    "sodium": "na", "potassium": "k", "chloride": "cl", "bicarbonate": "hco3",
    "anion gap": "anion_gap", "BUN": "bun", "glucose": "glucose", "calcium": "ca",
    "magnesium": "mg", "phosphate": "phos", "Hgb": "hgb", "WBC x 1000": "wbc",
    "platelets x 1000": "plt", "albumin": "albumin", "total bilirubin": "bili",
    "AST (SGOT)": "ast", "ALT (SGPT)": "alt", "lactate": "lactate", "pH": "ph",
    "PT - INR": "inr",
}

# Urine volume cells (mL). Counts, occurrences and mixed urine/stool are not volumes.
URINE_VOLUME = (r"Output \(ml\)\|(?:Urine|URINE CATHETER|Urine Output-Foley|Urine Output-FOLEY"
                r"|SN Urine Output\(ml\)|Urine, void:|Urine Output \(mL\)-Urethral Catheter)$")

# Past-history flags, matched on pasthistorypath.
COMORBIDITIES = {
    "hx_hypertension": r"Hypertension Requiring Treatment",
    "hx_chf": r"Congestive Heart Failure",
    "hx_diabetes": r"Insulin Dependent Diabetes",  # matches both insulin and non-insulin
    "hx_ckd": r"Renal Insufficiency|renal failure- not currently dialyzed",
    "hx_renal_transplant": r"s/p Renal Transplant",
    "hx_cirrhosis": r"Cirrhosis",
    "hx_copd": r"/COPD/",
    "hx_cancer": r"Hematology/Oncology \(R\)/Cancer",
    "hx_cad": r"Myocardial Infarction|Coronary Artery Bypass|Procedural Coronary Intervention"
              r"|/Angina/",
    "hx_stroke": r"/Strokes/|/TIA\(s\)/",
    "hx_pvd": r"Peripheral Vascular Disease",
    "hx_arrhythmia": r"/Arrhythmias/",
    "hx_immunosuppression": r"Immunosuppression|Recent Steroid Use",
}


# D3.3: coarse categories. Unlisted non-missing values fall into "other".
UNIT_TYPE = {"Med-Surg ICU": "medsurg", "MICU": "medical", "SICU": "surgical",
             "Neuro ICU": "neuro", "Cardiac ICU": "cardiac", "CCU-CTICU": "cardiac",
             "CSICU": "cardiac", "CTICU": "cardiac"}
ADMIT_SOURCE = {"Emergency Department": "ed", "Floor": "floor", "Acute Care/Floor": "floor",
                "Operating Room": "or_recovery", "Recovery Room": "or_recovery",
                "PACU": "or_recovery", "Direct Admit": "direct",
                "Step-Down Unit (SDU)": "icu_stepdown", "Other ICU": "icu_stepdown",
                "ICU to SDU": "icu_stepdown", "ICU": "icu_stepdown"}
ADMIT_DX_SYSTEM = {"Cardiovascular": "cardiovascular", "Respiratory": "respiratory",
                   "Neurology": "neuro", "Gastrointestinal": "gi",
                   "Metabolic/Endocrine": "metabolic", "Genitourinary": "gu", "Trauma": "trauma"}
ADMIT_DX = (r"^admission diagnosis\|All Diagnosis\|(?P<op>Non-operative|Operative)\|Diagnosis\|"
            r"(?P<system>[^|]+)")


def _at_or_before_t(df: pd.DataFrame, offset: str, start: float = -np.inf) -> pd.DataFrame:
    """Rows with start <= offset <= T: the only data any feature may use."""
    return df[(df[offset] >= start) & (df[offset] <= T_MIN)]


def demographics(patient: pd.DataFrame) -> pd.DataFrame:
    """Admission-time fields only. Age '> 89' is coded 90 (eICU de-identification)."""
    p = patient.set_index("patientunitstayid")
    age = pd.to_numeric(p["age"].replace("> 89", "90"), errors="coerce")
    female = p["gender"].map({"Female": 1.0, "Male": 0.0})
    height = p["admissionheight"].where(p["admissionheight"].between(100, 250))
    weight = p["admissionweight"].where(p["admissionweight"].between(25, 350))
    return pd.DataFrame({"age": age, "female": female, "height_cm": height,
                         "weight_kg": weight, "bmi": weight / (height / 100) ** 2})


def creatinine_features(cr: pd.DataFrame) -> pd.DataFrame:
    w = _at_or_before_t(cr, "offset", start=-PRE_ICU_LOOKBACK_MIN)
    g = w.groupby("patientunitstayid")["value"]
    return pd.DataFrame({
        "cr_baseline": cohort.baseline_a(cr)["value"],  # first draw in [0, T]
        "cr_pre_icu": cohort.baseline_c(w)["value"],  # last draw before admission, <= 7 d back
        "cr_last": g.last(),
        "cr_max": g.max(),
    })


def lab_features(lab: pd.DataFrame) -> pd.DataFrame:
    w = _at_or_before_t(lab[lab["labname"].isin(LAB_PANEL) & lab["labresult"].notna()],
                        "labresultoffset", start=-PRE_ICU_LOOKBACK_MIN)
    w = w.sort_values(["patientunitstayid", "labresultoffset"], kind="stable")
    last = w.groupby(["patientunitstayid", "labname"])["labresult"].last().unstack()
    return last.rename(columns=LAB_PANEL).reindex(columns=list(LAB_PANEL.values()))


def vital_features(vitals: pd.DataFrame) -> pd.DataFrame:
    w = _at_or_before_t(vitals, "observationoffset", start=0)
    mbp = w["noninvasivemean"].where(w["noninvasivemean"].between(20, 200))
    sbp = w["noninvasivesystolic"].where(w["noninvasivesystolic"].between(40, 300))
    w = w.assign(mbp=mbp, sbp=sbp)
    g = w.groupby("patientunitstayid")
    return pd.DataFrame({"map_min": g["mbp"].min(), "map_mean": g["mbp"].mean(),
                         "sbp_min": g["sbp"].min(),
                         "map_below_65": g["mbp"].min().lt(65).astype(float)
                         .where(g["mbp"].count() > 0)})


def periodic_vital_features(vp: pd.DataFrame) -> pd.DataFrame:
    """D3.2: heart rate, respiratory rate, temperature (degC; degF converted), SpO2 in [0, T]."""
    w = _at_or_before_t(vp, "observationoffset", start=0)
    temp = w["temperature"].where(w["temperature"] <= 50, (w["temperature"] - 32) * 5 / 9)
    w = w.assign(hr=w["heartrate"].where(w["heartrate"].between(20, 250)),
                 rr=w["respiration"].where(w["respiration"].between(2, 80)),
                 spo2=w["sao2"].where(w["sao2"].between(50, 100)),
                 temp=temp.where(temp.between(30, 45)))
    g = w.groupby("patientunitstayid")
    return pd.DataFrame({"hr_mean": g["hr"].mean(), "hr_max": g["hr"].max(),
                         "rr_mean": g["rr"].mean(), "rr_max": g["rr"].max(),
                         "spo2_min": g["spo2"].min(), "temp_max": g["temp"].max()})


def _one_hot(values: pd.Series, prefix: str, levels) -> pd.DataFrame:
    """Indicator columns; a missing category gives NaN in every column of the group."""
    out = pd.DataFrame({f"{prefix}_{lv}": (values == lv).astype(float) for lv in levels},
                       index=values.index)
    return out.where(values.notna())


def admission_context(patient: pd.DataFrame, admission_dx: pd.DataFrame) -> pd.DataFrame:
    """D3.3: unit type, admission source, coarse admission diagnosis entered by T."""
    p = patient.set_index("patientunitstayid")
    unit = p["unittype"].map(UNIT_TYPE)  # every eICU demo unit type is listed
    src = p["unitadmitsource"].map(ADMIT_SOURCE).where(p["unitadmitsource"].isna(),
                                                       lambda s: s.fillna("other"))
    dx = admission_dx[admission_dx["admitdxenteredoffset"] <= T_MIN]
    parts = dx["admitdxpath"].str.extract(ADMIT_DX).assign(
        patientunitstayid=dx["patientunitstayid"], at=dx["admitdxenteredoffset"]).dropna()
    parts = parts.sort_values("at", kind="stable").groupby("patientunitstayid").first()
    system = parts["system"].map(ADMIT_DX_SYSTEM).fillna("other").reindex(p.index)
    operative = (parts["op"] == "Operative").astype(float).reindex(p.index)
    return pd.concat([
        _one_hot(unit, "unit", sorted(set(UNIT_TYPE.values()))),
        _one_hot(src, "src", sorted(set(ADMIT_SOURCE.values()) | {"other"})),
        operative.rename("dx_operative"),
        _one_hot(system, "dx", sorted(set(ADMIT_DX_SYSTEM.values()) | {"other"})),
    ], axis=1)


def urine_features(io: pd.DataFrame, weight_kg: pd.Series) -> pd.DataFrame:
    w = _at_or_before_t(io[io["cellpath"].str.contains(URINE_VOLUME, na=False)],
                        "intakeoutputoffset", start=0)
    ml = w.groupby("patientunitstayid")["cellvaluenumeric"].sum()
    return pd.DataFrame({"urine_ml_0_24h": ml,
                         "urine_ml_kg_h": ml / weight_kg.reindex(ml.index) / (T_MIN / 60)})


def comorbidities(past_history: pd.DataFrame) -> pd.DataFrame:
    """1/0 for stays whose past history was entered by T; NaN for stays with none by T."""
    w = past_history[past_history["pasthistoryenteredoffset"] <= T_MIN]
    taken = pd.Index(w["patientunitstayid"].unique())
    out = pd.DataFrame(index=taken)
    for name, pattern in COMORBIDITIES.items():
        hit = set(w.loc[w["pasthistorypath"].str.contains(pattern, na=False), "patientunitstayid"])
        out[name] = taken.isin(list(hit)).astype(float)
    return out


def build(tables: dict[str, pd.DataFrame], stays) -> pd.DataFrame:
    """Feature matrix for ``stays`` (index: patientunitstayid), NaN where unmeasured."""
    demo = demographics(tables["patient"])
    parts = [
        demo,
        creatinine_features(cohort.serum_creatinine(tables["lab"])),
        lab_features(tables["lab"]),
        vital_features(tables["vitalAperiodic"]),
        periodic_vital_features(tables["vitalPeriodic"]),
        admission_context(tables["patient"], tables["admissionDx"]),
        urine_features(tables["intakeOutput"], demo["weight_kg"]),
        comorbidities(tables["pastHistory"]),
    ]
    X = pd.concat(parts, axis=1).reindex(pd.Index(stays, name="patientunitstayid"))
    return X


def load_feature_inputs() -> dict[str, pd.DataFrame]:
    return {
        "patient": load("patient"),
        "lab": load("lab", usecols=["patientunitstayid", "labresultoffset", "labname",
                                    "labresult"]),
        "vitalAperiodic": load("vitalAperiodic", usecols=["patientunitstayid",
                                                          "observationoffset",
                                                          "noninvasivesystolic",
                                                          "noninvasivemean"]),
        "intakeOutput": load("intakeOutput", usecols=["patientunitstayid",
                                                      "intakeoutputoffset", "cellpath",
                                                      "cellvaluenumeric"]),
        "pastHistory": load("pastHistory", usecols=["patientunitstayid",
                                                    "pasthistoryenteredoffset",
                                                    "pasthistorypath"]),
        "vitalPeriodic": load("vitalPeriodic", usecols=["patientunitstayid", "observationoffset",
                                                        "heartrate", "respiration", "sao2",
                                                        "temperature"]),
        "admissionDx": load("admissionDx", usecols=["patientunitstayid", "admitdxenteredoffset",
                                                    "admitdxpath"]),
    }


def dictionary(X: pd.DataFrame, primary: pd.Index) -> str:
    lines = ["# Clinical features at T (generated by `python -m polysafe.features`)", "",
             "All features use only data with offset <= T = 24 h; pre-ICU measured values "
             "look back at most 7 days (D3.7). Missing = not measured by T; it is never "
             "filled with a normal value here. APACHE scores are excluded (D3.1). "
             "Left out of the logistic regression only (D3.6, D4.2-D4.3): "
             f"{', '.join(LR_EXCLUDED)}.", "",
             f"Cohort: baseline A, all sites n = {len(X):,}; primary sites n = {len(primary):,}.",
             "", "| Feature | Missing (all sites) | Missing (primary) | Median | IQR |",
             "|---|---:|---:|---:|---|"]
    for c in X.columns:
        s = X[c]
        lines.append(f"| `{c}` | {s.isna().mean():.1%} | {s.loc[primary].isna().mean():.1%} | "
                     f"{s.median():.3g} | {s.quantile(.25):.3g}-{s.quantile(.75):.3g} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build clinical features at T.")
    p.add_argument("--labels", type=Path, default=Path("results") / "labels_A.csv")
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    lab = pd.read_csv(args.labels)
    X = build(load_feature_inputs(), lab["patientunitstayid"])
    primary = pd.Index(lab.loc[lab["primary_site"], "patientunitstayid"])
    args.out.mkdir(parents=True, exist_ok=True)
    X.to_csv(args.out / "features_clinical.csv")
    # D3.4: ethnicity is reported by subgroup, never used as a predictor.
    eth = load("patient", usecols=["patientunitstayid", "ethnicity"]).set_index("patientunitstayid")
    eth.reindex(X.index).to_csv(args.out / "subgroups.csv")
    md = dictionary(X, primary)
    (args.out / "features_clinical.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
