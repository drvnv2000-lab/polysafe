"""Cohort construction with leakage-controlled exclusions (PREREGISTRATION sections 3-5, D1.3).

Prediction time T = 24 h after ICU admission. Outcome window (T, end], where
end = min(T + 7 days, ICU discharge or death). Exclusions, applied in order:

1. Stay shorter than T            (``unitdischargeoffset < T``)
2. Chronic dialysis / ESRD        (AKI is not definable for these patients)
3. AKI already present by T       (KDIGO stage >= 1 on creatinine drawn <= T)
4. Acute RRT started by T         (prevalent KDIGO stage 3)
5. No baseline creatinine         (D1.3: baseline must be observable)
6. No observable outcome          (D1.3: no serum creatinine and no RRT start in the window)

The KDIGO rules are evaluated once per creatinine draw (``kdigo_hits``). The
exclusion at T and the outcome after T both read from that one table, so they
cannot drift apart: a draw at or before T can only exclude, a draw after T can
only label.

The primary estimand is further restricted at the *site* level to hospitals
that record medication administration (``recording_hospitals``). That flag is a
population filter only and must never be used as a predictor: it is a hospital
proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from polysafe.tables import load

T_MIN = 24 * 60  # prediction time, minutes after unit admission
OUTCOME_WINDOW_MIN = 7 * 24 * 60  # outcome window (T, T + 7 d], censored at ICU discharge
RISE_WINDOW_MIN = 48 * 60  # KDIGO: >= 0.3 mg/dL rise within 48 h
RATIO_WINDOW_MIN = 7 * 24 * 60  # KDIGO: >= 1.5x baseline within 7 days
RISE_MG_DL = 0.3
RATIO = 1.5
EPS = 1e-9  # creatinine is stored as floats; 2.3 - 2.0 == 0.2999999999999998

# PI DECISION (confirmed): chronic markers only. Dialysis in `treatment` that is
# not marked chronic is acute renal replacement therapy, handled by RRT below.
PAST_HISTORY_DIALYSIS = r"renal failure - hemodialysis|renal failure - peritoneal dialysis"
DIAGNOSIS_ESRD = r"ESRD \(end stage renal disease\)|chronic kidney disease\|Stage 5"
TREATMENT_CHRONIC_DIALYSIS = r"hemodialysis\|for chronic renal failure"
# ESRD markers are used regardless of charting time (ESRD predates admission).
# Measured: no stay >= 24 h is flagged only by a marker charted after T.

# PI DECISION (confirmed): pre-ICU creatinine (offset < 0) counts toward the
# 48 h rise rule. KDIGO uses any 48 h window.
INCLUDE_PRE_ICU_IN_RISE = True

# PI DECISION (confirmed): dialysis or CVVH(D) started at or before T is
# prevalent KDIGO stage 3 and excluded. Catheter / shunt insertion is not RRT.
RRT = r"renal\|dialysis\|(?:hemodialysis|C V V H)|treatment of hyperkalemia\|dialysis"


@dataclass(frozen=True)
class Step:
    name: str
    reason: str
    removed: frozenset = field(repr=False)
    remaining: frozenset = field(repr=False)


# ---------------------------------------------------------------- creatinine


def serum_creatinine(lab: pd.DataFrame) -> pd.DataFrame:
    """Serum creatinine rows with a value: stay, offset, value; sorted by stay and time.

    Matches ``labname == "creatinine"`` exactly, which excludes ``urinary creatinine``.
    """
    cr = lab.loc[(lab["labname"] == "creatinine") & lab["labresult"].notna(),
                 ["patientunitstayid", "labresultoffset", "labresult"]]
    cr = cr.rename(columns={"labresultoffset": "offset", "labresult": "value"})
    return cr.sort_values(["patientunitstayid", "offset"], kind="stable").reset_index(drop=True)


def baseline_a(cr: pd.DataFrame) -> pd.DataFrame:
    """Baseline A: first serum creatinine with 0 <= offset <= T. Index: stay."""
    w = cr[(cr["offset"] >= 0) & (cr["offset"] <= T_MIN)]
    return w.groupby("patientunitstayid")[["offset", "value"]].first()


def baseline_c(cr: pd.DataFrame) -> pd.DataFrame:
    """Baseline C: most recent serum creatinine before ICU admission (offset < 0)."""
    w = cr[cr["offset"] < 0]
    return w.groupby("patientunitstayid")[["offset", "value"]].last()


BASELINES = {"A": baseline_a, "C": baseline_c}


def _rise_flags(cr: pd.DataFrame, include_pre_icu: bool) -> pd.Series:
    """True for each draw that is >= 0.3 mg/dL above some draw in the preceding 48 h."""
    hit = []
    w = cr if include_pre_icu else cr[cr["offset"] >= 0]
    for _, g in w.groupby("patientunitstayid", sort=False):
        t = g["offset"].to_numpy()
        v = g["value"].to_numpy()
        for j, idx in enumerate(g.index):
            earlier = (t < t[j]) & (t >= t[j] - RISE_WINDOW_MIN)
            if earlier.any() and v[j] - v[earlier].min() >= RISE_MG_DL - EPS:
                hit.append(idx)
    return pd.Series(cr.index.isin(hit), index=cr.index)


def _ratio_flags(cr: pd.DataFrame, baseline: pd.DataFrame) -> pd.Series:
    """True for each draw >= 1.5x baseline, drawn after the baseline and within 7 days of it."""
    b = cr[["patientunitstayid"]].join(baseline, on="patientunitstayid")
    return ((cr["offset"] > b["offset"])
            & (cr["offset"] <= b["offset"] + RATIO_WINDOW_MIN)
            & (cr["value"] >= RATIO * b["value"] - EPS))


def kdigo_hits(cr: pd.DataFrame, baseline: pd.DataFrame,
               include_pre_icu: bool = INCLUDE_PRE_ICU_IN_RISE) -> pd.DataFrame:
    """Every creatinine draw that meets a KDIGO stage >= 1 criterion, with which one.

    Whether a draw qualifies depends only on that draw, earlier draws and the
    baseline, never on later draws.
    """
    rise = _rise_flags(cr, include_pre_icu)
    ratio = _ratio_flags(cr, baseline)
    hits = cr.assign(rise=rise, ratio=ratio)
    return hits[hits["rise"] | hits["ratio"]].reset_index(drop=True)


def rise_by(cr: pd.DataFrame, until: int, include_pre_icu: bool = INCLUDE_PRE_ICU_IN_RISE) -> set:
    """Stays with a >= 0.3 mg/dL rise within 48 h completed by a draw at or before ``until``."""
    h = cr[_rise_flags(cr, include_pre_icu) & (cr["offset"] <= until)]
    return set(h["patientunitstayid"])


def ratio_by(cr: pd.DataFrame, baseline: pd.DataFrame, until: int) -> set:
    """Stays with a draw >= 1.5x baseline within 7 days, at or before ``until``."""
    h = cr[_ratio_flags(cr, baseline) & (cr["offset"] <= until)]
    return set(h["patientunitstayid"])


def aki_by(cr: pd.DataFrame, baseline: pd.DataFrame, until: int = T_MIN) -> set:
    """Stays meeting KDIGO stage >= 1 using only creatinine drawn at or before ``until``."""
    h = kdigo_hits(cr, baseline)
    return set(h.loc[h["offset"] <= until, "patientunitstayid"])


# ---------------------------------------------------------------- ESRD / RRT


def esrd_markers(past_history: pd.DataFrame, diagnosis: pd.DataFrame,
                 treatment: pd.DataFrame, aps: pd.DataFrame) -> pd.DataFrame:
    """One row per (stay, source) that marks chronic dialysis / ESRD, with first charting offset."""
    parts = [
        past_history.loc[
            past_history["pasthistorypath"].str.contains(PAST_HISTORY_DIALYSIS, case=False,
                                                         na=False),
            ["patientunitstayid", "pasthistoryoffset"],
        ].rename(columns={"pasthistoryoffset": "offset"}).assign(source="pastHistory dialysis"),
        diagnosis.loc[
            diagnosis["diagnosisstring"].str.contains(DIAGNOSIS_ESRD, na=False),
            ["patientunitstayid", "diagnosisoffset"],
        ].rename(columns={"diagnosisoffset": "offset"}).assign(source="diagnosis ESRD / CKD 5"),
        treatment.loc[
            treatment["treatmentstring"].str.contains(TREATMENT_CHRONIC_DIALYSIS, na=False),
            ["patientunitstayid", "treatmentoffset"],
        ].rename(columns={"treatmentoffset": "offset"}).assign(source="treatment chronic HD"),
        # APACHE IV "dialysis" = chronic dialysis before admission; it is a day-1 score input.
        aps.loc[aps["dialysis"] == 1, ["patientunitstayid"]]
        .assign(offset=np.nan, source="APACHE dialysis"),
    ]
    rows = pd.concat(parts, ignore_index=True)
    return rows.groupby(["patientunitstayid", "source"], as_index=False)["offset"].min()


def rrt_start(treatment: pd.DataFrame) -> pd.Series:
    """First charted dialysis / CVVH(D) per stay, in minutes. Index: stay."""
    rows = treatment[treatment["treatmentstring"].str.contains(RRT, na=False)]
    return rows.groupby("patientunitstayid")["treatmentoffset"].min()


def outcome_end(patient: pd.DataFrame) -> pd.Series:
    """End of the outcome window per stay: min(T + 7 d, ICU discharge or death). Index: stay."""
    p = patient.set_index("patientunitstayid")["unitdischargeoffset"]
    return p.clip(upper=T_MIN + OUTCOME_WINDOW_MIN)


def in_window(offsets: pd.Series, end: pd.Series) -> pd.Series:
    """True where T < offset <= end for that stay. ``offsets`` is indexed by stay."""
    e = end.reindex(offsets.index)
    return (offsets > T_MIN) & (offsets <= e)


def recording_hospitals(patient: pd.DataFrame, medication: pd.DataFrame) -> set:
    """Hospitals with at least one medication row for any stay (site-level, all stays).

    A population filter for the primary estimand. Never a predictor.
    """
    stays = set(medication["patientunitstayid"])
    has = patient["patientunitstayid"].isin(stays)
    return set(patient.loc[has, "hospitalid"])


# ---------------------------------------------------------------- cohort


def build_cohort(tables: dict[str, pd.DataFrame] | None = None,
                 baseline: str = "A") -> tuple[pd.DataFrame, list[Step]]:
    """Apply the exclusions in order. Returns (cohort patient rows, flow steps)."""
    tables = tables or load_inputs()
    patient = tables["patient"]
    cr = serum_creatinine(tables["lab"])
    base = BASELINES[baseline](cr)

    remaining = frozenset(patient["patientunitstayid"])
    flow = [Step("Start", "all ICU unit stays in patient.csv.gz", frozenset(), remaining)]

    def drop(name: str, reason: str, ids) -> None:
        nonlocal remaining
        removed = remaining & frozenset(ids)
        remaining = remaining - removed
        flow.append(Step(name, reason, removed, remaining))

    drop("Stay < 24 h", "ICU discharge or death before T, so no prediction time exists",
         patient.loc[patient["unitdischargeoffset"] < T_MIN, "patientunitstayid"])
    esrd = esrd_markers(tables["pastHistory"], tables["diagnosis"],
                        tables["treatment"], tables["apacheApsVar"])
    drop("Chronic dialysis / ESRD", "AKI is not definable on chronic dialysis",
         esrd["patientunitstayid"])
    drop(f"AKI by T (baseline {baseline})",
         "met KDIGO stage >= 1 on creatinine drawn <= T; nothing left to predict",
         aki_by(cr, base))
    rrt = rrt_start(tables["treatment"])
    drop("Acute RRT by T", "dialysis / CVVH started <= T: prevalent KDIGO stage 3",
         rrt[rrt <= T_MIN].index)
    drop(f"No baseline {baseline}",
         {"A": "no serum creatinine in the first 24 h",
          "C": "no serum creatinine before ICU admission"}[baseline],
         remaining - frozenset(base.index))
    end = outcome_end(patient)
    cr_seen = cr.loc[in_window(cr.set_index("patientunitstayid")["offset"], end).to_numpy(),
                     "patientunitstayid"]
    rrt_seen = rrt[in_window(rrt, end)].index
    drop("No observable outcome",
         "no serum creatinine and no RRT start in (T, min(T + 7 d, ICU discharge)]",
         remaining - frozenset(cr_seen) - frozenset(rrt_seen))

    cohort = patient[patient["patientunitstayid"].isin(remaining)]
    return cohort.reset_index(drop=True), flow


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "patient": load("patient"),
        "lab": load("lab", usecols=["patientunitstayid", "labresultoffset", "labname",
                                    "labresult"]),
        "pastHistory": load("pastHistory"),
        "diagnosis": load("diagnosis"),
        "treatment": load("treatment"),
        "apacheApsVar": load("apacheApsVar", usecols=["patientunitstayid", "dialysis"]),
        "medication": load("medication", usecols=["patientunitstayid"]),
    }
