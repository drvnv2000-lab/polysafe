"""Clinical features at T (PREREGISTRATION section 8, clinical-only covariates).

Rule: **only data with offset <= T** (section 3, "no exception"). Every
time-stamped table is cut at T by ``_at_or_before_t`` before anything else is
computed, and ``tests/test_features.py`` rewrites every row after T to prove
the features do not move.

Missing stays missing (NaN). Absence of a record is never coded as normal or
as "condition absent", except for comorbidity flags of stays whose past
history was taken (see ``comorbidities``).

APACHE scores are NOT included: eICU's APACHE day-1 values use lab results
drawn after T for 8.5% of the cohort (up to 7.3 h after T), so they would
leak. Pending a PI decision.

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
    w = _at_or_before_t(cr, "offset")
    g = w.groupby("patientunitstayid")["value"]
    return pd.DataFrame({
        "cr_baseline": cohort.baseline_a(cr)["value"],  # first draw in [0, T]
        "cr_pre_icu": cohort.baseline_c(cr)["value"],  # last draw before admission
        "cr_last": g.last(),
        "cr_max": g.max(),
    })


def lab_features(lab: pd.DataFrame) -> pd.DataFrame:
    w = _at_or_before_t(lab[lab["labname"].isin(LAB_PANEL) & lab["labresult"].notna()],
                        "labresultoffset")
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
    }


def dictionary(X: pd.DataFrame, primary: pd.Index) -> str:
    lines = ["# Clinical features at T (generated by `python -m polysafe.features`)", "",
             "All features use only data with offset <= T = 24 h. Missing = not measured "
             "by T; it is never filled with a normal value here. APACHE scores are "
             "excluded (post-T leakage, see `features.py`).", "",
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
    md = dictionary(X, primary)
    (args.out / "features_clinical.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
