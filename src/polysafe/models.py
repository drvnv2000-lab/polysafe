"""Prediction models and the paired dAUROC (PREREGISTRATION section 8, D2.1a, D7.1, D7.2).

Every choice below is fixed by the protocol before the first fit:

* primary: L2 logistic regression on D4.4's reduced feature set, clinical-only against
  clinical + medication; secondary: LightGBM on every feature; sensitivity: the full
  penalised logistic regression (every clinical feature except ``features.LR_EXCLUDED``);
* medication features: the five of D6.0/D6.0c (not ondansetron, D6.0d; aminoglycosides
  contribute no variable), exposed by T under D6.4 in the HICL + gtc strategy, primary
  variant; missing for stays with no medication record of their own (D7.1);
* imputation and standardisation fitted on the training fold (D3.6, D4.4, D7.1, D7.2);
  medication features of record-less stays filled with the training fold's share exposed
  in the logistic regression, left missing in LightGBM, no indicator (D7.1);
* folds: ``cv.patient_folds`` with random states 0-9, identical for both models of a pair;
  tuning in an inner patient-grouped split by log loss (D7.2);
* each stay's 10 out-of-fold predictions are averaged and every metric is computed on that
  one prediction per stay; intervals are D2.1a BCa intervals.

Usage:
    python -m polysafe.models    # writes results/model_*.csv and results/models.md
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss

from polysafe import ascertain, bootstrap, features, power
from polysafe.cv import N_REPEATS, N_SPLITS, patient_folds
from polysafe.tables import load

RESULTS = Path("results")
PRIMARY_COHORT = "n=671 (primary sites)"
ARM, VARIANT = "hicl_gtc", "primary"  # section 6, D7.2
# D6.0, D6.0c, D6.0d: exposure name -> feature column
MEDICATION = {"vancomycin": "med_vancomycin",
              "piperacillin-tazobactam": "med_piperacillin_tazobactam",
              "furosemide": "med_furosemide",
              "NSAIDs (aspirin excluded, D4.1)": "med_nsaids",
              "iodinated contrast": "med_contrast"}
MED_COLS = tuple(MEDICATION.values())
C_GRID = tuple(10.0 ** np.arange(-3, 3.01, 0.5))  # D7.2: 13 values, ascending (ties -> smaller)
LGBM_FIXED = {"objective": "binary", "learning_rate": 0.05, "subsample": 1.0,
              "subsample_freq": 0, "colsample_bytree": 1.0, "reg_alpha": 0.0,
              "reg_lambda": 0.0, "random_state": power.SEED, "deterministic": True,
              "n_jobs": 1, "verbose": -1}
# D7.2 grid, in tie-break order: fewer leaves, then more samples per leaf, then fewer trees
LGBM_GRID = tuple({"num_leaves": nl, "min_child_samples": mc, "n_estimators": ne}
                  for nl in (4, 8, 16) for mc in (40, 20, 10) for ne in (100, 300))
CLIP = 1e-6  # D7.2: predictions clipped before any logit
DCA_THRESHOLDS = tuple(np.round(np.arange(0.02, 0.4001, 0.01), 2))  # D7.2


@dataclass(frozen=True)
class Spec:
    name: str
    kind: str  # "lr" or "lgbm"
    clinical: tuple[str, ...]
    indicators: tuple[tuple[str, ...], ...] | None  # None: one per column with a gap


def specs(clinical_columns) -> tuple[Spec, ...]:
    cols = tuple(clinical_columns)
    return (Spec("lr_reduced", "lr", features.PRIMARY_LR_FEATURES,
                 features.PRIMARY_LR_INDICATORS),
            Spec("lgbm", "lgbm", cols, None),
            Spec("lr_full", "lr", tuple(c for c in cols if c not in features.LR_EXCLUDED),
                 None))


# ---------------------------------------------------------------- design matrices


def lr_design(train: pd.DataFrame, test: pd.DataFrame, spec: Spec,
              medication: bool) -> tuple[np.ndarray, np.ndarray]:
    """D3.6 / D4.4 imputation, D7.1 medication filling, D7.2 standardisation, all fitted
    on ``train``."""
    cols = list(spec.clinical)
    tr, te = train[cols].astype(float), test[cols].astype(float)
    groups = spec.indicators if spec.indicators is not None else tuple(
        (c,) for c in cols if tr[c].isna().any())
    ind_tr = [tr[list(g)].isna().any(axis=1).astype(float) for g in groups]
    ind_te = [te[list(g)].isna().any(axis=1).astype(float) for g in groups]
    median = tr.median()
    if median.isna().any():
        raise ValueError(f"no training values for {list(median[median.isna()].index)}")
    parts_tr, parts_te = [tr.fillna(median), *ind_tr], [te.fillna(median), *ind_te]
    if medication:
        m_tr, m_te = train[list(MED_COLS)].astype(float), test[list(MED_COLS)].astype(float)
        share = m_tr.mean()  # share exposed among training stays with a record (D7.1)
        parts_tr.append(m_tr.fillna(share))
        parts_te.append(m_te.fillna(share))
    x_tr = pd.concat(parts_tr, axis=1).to_numpy(float)
    x_te = pd.concat(parts_te, axis=1).to_numpy(float)
    mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0)
    ok = sd > 0
    z_tr, z_te = np.zeros_like(x_tr), np.zeros_like(x_te)
    z_tr[:, ok] = (x_tr[:, ok] - mu[ok]) / sd[ok]
    z_te[:, ok] = (x_te[:, ok] - mu[ok]) / sd[ok]
    return z_tr, z_te


def lgbm_design(train: pd.DataFrame, test: pd.DataFrame, spec: Spec,
                medication: bool) -> tuple[np.ndarray, np.ndarray]:
    cols = list(spec.clinical) + (list(MED_COLS) if medication else [])
    return train[cols].to_numpy(float), test[cols].to_numpy(float)


def fit_predict(spec: Spec, medication: bool, train: pd.DataFrame, y: np.ndarray,
                test: pd.DataFrame, param) -> np.ndarray:
    if spec.kind == "lr":
        x_tr, x_te = lr_design(train, test, spec, medication)
        model = LogisticRegression(C=param, solver="lbfgs", max_iter=10_000)
    else:
        from lightgbm import LGBMClassifier
        x_tr, x_te = lgbm_design(train, test, spec, medication)
        model = LGBMClassifier(**LGBM_FIXED, **param)
    return model.fit(x_tr, y).predict_proba(x_te)[:, 1]


def choose(spec: Spec, medication: bool, train: pd.DataFrame, y: np.ndarray,
           groups: np.ndarray, seed: int) -> tuple[object, list[float]]:
    """Lowest mean inner log loss; ties go to the earliest candidate (D7.2 order)."""
    inner = patient_folds(y, groups, n_repeats=1, seed=seed)[0]
    candidates = C_GRID if spec.kind == "lr" else LGBM_GRID
    losses = []
    for cand in candidates:
        fold = [log_loss(y[inner == k], fit_predict(spec, medication, train[inner != k],
                                                    y[inner != k], train[inner == k], cand),
                         labels=[0, 1]) for k in range(N_SPLITS)]
        losses.append(float(np.mean(fold)))
    return candidates[int(np.argmin(losses))], losses


# ---------------------------------------------------------------- cross-validation

_DATA: dict = {}


def _init(data: dict) -> None:
    _DATA.update(data)


def outer_task(task: tuple[int, int, str]) -> dict:
    """One outer fold, one spec: both models of the pair on the same split."""
    r, k, name = task
    X, y, g, folds = _DATA["X"], _DATA["y"], _DATA["groups"], _DATA["folds"]
    spec = next(s for s in _DATA["specs"] if s.name == name)
    te = folds[r] == k
    out = {"r": r, "k": k, "spec": name, "test": np.flatnonzero(te)}
    for med in (False, True):
        param, _ = choose(spec, med, X[~te], y[~te], g[~te], seed=1000 + 5 * r + k)
        out[med] = (fit_predict(spec, med, X[~te], y[~te], X[te], param), param)
    return out


def cross_validate(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, spec_list,
                   repeats: int = N_REPEATS, workers: int = 1) -> tuple[dict, pd.DataFrame]:
    """Pooled out-of-fold predictions {(spec, medication): averaged prediction per stay}
    and the tuning choice of every outer fit."""
    folds = patient_folds(y, groups, n_repeats=repeats, seed=0)  # random states 0..9
    tasks = [(r, k, s.name) for s in spec_list for r in range(repeats)
             for k in range(N_SPLITS)]
    data = {"X": X, "y": y, "groups": groups, "folds": folds, "specs": tuple(spec_list)}
    if workers > 1:
        with Pool(workers, initializer=_init, initargs=(data,)) as pool:
            done = list(pool.imap_unordered(outer_task, tasks))
    else:
        _init(data)
        done = [outer_task(t) for t in tasks]
    oof = {(s.name, m): np.full((repeats, len(y)), np.nan) for s in spec_list
           for m in (False, True)}
    tuning = []
    for d in done:
        for med in (False, True):
            pred, param = d[med]
            oof[(d["spec"], med)][d["r"], d["test"]] = pred
            tuning.append({"spec": d["spec"], "medication": med, "repeat": d["r"],
                           "fold": d["k"], "param": str(param)})
    if any(np.isnan(v).any() for v in oof.values()):
        raise RuntimeError("a stay is missing an out-of-fold prediction")
    pooled = {key: v.mean(axis=0) for key, v in oof.items()}
    return pooled, pd.DataFrame(tuning).sort_values(["spec", "medication", "repeat", "fold"])


# ---------------------------------------------------------------- metrics


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p))


def _newton(x: np.ndarray, y: np.ndarray, offset: np.ndarray, iters: int = 100) -> np.ndarray:
    """Unpenalised logistic regression of y on design x with an offset (Newton-Raphson)."""
    beta = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(x @ beta + offset)))
        step = np.linalg.solve(x.T @ (x * (p * (1 - p))[:, None]), x.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            return beta
    raise RuntimeError("calibration fit did not converge")


def calibration(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """D7.2: intercept with the logit as offset; slope of AKI on the logit."""
    lp, y = logit(p), np.asarray(y, float)
    ones = np.ones((len(y), 1))
    intercept = float(_newton(ones, y, lp)[0])
    slope = float(_newton(np.column_stack([ones, lp]), y, np.zeros(len(y)))[1])
    return intercept, slope


def calibration_curve(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    bins = pd.qcut(p, 10, labels=False, duplicates="drop")
    return (pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin")
            .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean"))
            .reset_index())


def net_benefit(p: np.ndarray, y: np.ndarray, t: float) -> float:
    treat, y = np.asarray(p) >= t, np.asarray(y).astype(bool)
    n = len(y)
    return float((treat & y).sum() / n - (treat & ~y).sum() / n * t / (1 - t))


def decision_curve(preds: dict, y: np.ndarray) -> pd.DataFrame:
    prev = float(np.mean(y))
    rows = []
    for t in DCA_THRESHOLDS:
        row = {"threshold": t, "treat_all": prev - (1 - prev) * t / (1 - t), "treat_none": 0.0}
        row.update({name: net_benefit(p, y, t) for name, p in preds.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def auprc(score: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    return float("nan") if y.all() or not y.any() else float(average_precision_score(y, score))


def interval(stat, groups, n_boot: int) -> dict:
    try:
        ci = bootstrap.bca(stat, groups, n_boot=n_boot)
    except bootstrap.BCaUndefined as e:
        return {"lo": np.nan, "hi": np.nan, "bca_failed": str(e)}
    return {"lo": ci.low, "hi": ci.high, "n_attempted": ci.n_attempted, "n_used": ci.n_used,
            "n_dropped": ci.n_dropped, "bca_failed": ""}


def nearest(value: float, grid) -> float:
    grid = np.asarray(sorted(grid), float)
    return float(grid[int(np.argmin(np.abs(grid - value)))])


def matched_cell(cov: pd.DataFrame, mde: pd.DataFrame, prevalence: float, a0: float,
                 rho: float, delta: float, cohort: str = PRIMARY_COHORT) -> dict:
    """D2.1a: the measured-prevalence cell of ``cohort`` nearest the realised values
    (the primary cohort unless a D8.1 analysis names another)."""
    sel = cov[(cov["cohort"] == cohort) & np.isclose(cov["prevalence"], prevalence)]
    if sel.empty:
        raise ValueError(f"no coverage cells for {cohort} at its measured prevalence")
    cell = {"a0": nearest(a0, sel["a0"].unique()), "rho": nearest(rho, sel["rho"].unique()),
            "delta": nearest(delta, sel["delta"].unique())}
    row = sel[(sel["a0"] == cell["a0"]) & (sel["rho"] == cell["rho"])
              & (sel["delta"] == cell["delta"])].iloc[0]
    m = mde[(mde["cohort"] == cohort) & np.isclose(mde["prevalence"], prevalence)
            & (mde["a0"] == cell["a0"]) & (mde["rho"] == cell["rho"])].iloc[0]
    return {f"cell_{k}": v for k, v in cell.items()} | {
        "cov_dauroc": row["bca_coverage"], "cov_auroc_clin": row["clin_coverage"],
        "cov_auroc_med": row["comb_coverage"],
        "cov_dauroc_min": sel["bca_coverage"].min(), "cov_dauroc_max": sel["bca_coverage"].max(),
        "cov_clin_min": sel["clin_coverage"].min(), "cov_clin_max": sel["clin_coverage"].max(),
        "cov_med_min": sel["comb_coverage"].min(), "cov_med_max": sel["comb_coverage"].max(),
        "mde_bca": m["mde_bca"], "mde_pct": m["mde_pct"]}


def evaluate(pooled: dict, y: np.ndarray, groups: np.ndarray, spec_list, cov: pd.DataFrame,
             mde: pd.DataFrame, prevalence: float, n_boot: int = bootstrap.N_BOOT):
    """Per-model metrics, per-pair comparison, calibration curves and decision curves."""
    metrics, comparison, curves = [], [], []
    for s in spec_list:
        p0, p1 = pooled[(s.name, False)], pooled[(s.name, True)]
        for med, p in ((False, p0), (True, p1)):
            ci_roc = interval(lambda i, p=p: bootstrap.auroc(p[i], y[i]), groups, n_boot)
            ci_prc = interval(lambda i, p=p: auprc(p[i], y[i]), groups, n_boot)
            icpt, slope = calibration(p, y)
            metrics.append({"spec": s.name, "medication": med,
                            "auroc": bootstrap.auroc(p, y), "auroc_lo": ci_roc["lo"],
                            "auroc_hi": ci_roc["hi"], "auroc_bca_failed": ci_roc["bca_failed"],
                            "auroc_n_attempted": ci_roc.get("n_attempted"),
                            "auroc_n_used": ci_roc.get("n_used"),
                            "auroc_n_dropped": ci_roc.get("n_dropped"),
                            "auprc": auprc(p, y), "auprc_lo": ci_prc["lo"],
                            "auprc_hi": ci_prc["hi"], "auprc_bca_failed": ci_prc["bca_failed"],
                            "auprc_n_attempted": ci_prc.get("n_attempted"),
                            "auprc_n_used": ci_prc.get("n_used"),
                            "auprc_n_dropped": ci_prc.get("n_dropped"),
                            "auprc_coverage": "coverage not measured",
                            "cal_intercept": icpt, "cal_slope": slope})
            curves.append(calibration_curve(p, y).assign(spec=s.name, medication=med))
        d = bootstrap.auroc(p1, y) - bootstrap.auroc(p0, y)
        ci = interval(bootstrap.delta_auroc(p1, p0, y), groups, n_boot)
        rho_logit = float(np.corrcoef(logit(p0), logit(p1))[0, 1])
        rho_prob = float(np.corrcoef(p0, p1)[0, 1])
        comparison.append({"spec": s.name, "dauroc": d, "lo": ci["lo"], "hi": ci["hi"],
                           "n_attempted": ci.get("n_attempted"), "n_used": ci.get("n_used"),
                           "n_dropped": ci.get("n_dropped"), "bca_failed": ci["bca_failed"],
                           "rho_logit": rho_logit, "rho_prob": rho_prob}
                          | matched_cell(cov, mde, prevalence, bootstrap.auroc(p0, y),
                                         rho_logit, d))
    dca = decision_curve({f"{s.name}_{m}": pooled[(s.name, m == 'medication')]
                          for s in spec_list for m in ("clinical", "medication")}, y)
    return (pd.DataFrame(metrics), pd.DataFrame(comparison), pd.concat(curves), dca)


# ---------------------------------------------------------------- data


def medication_features(stays: pd.Index, med: pd.DataFrame) -> pd.DataFrame:
    """D7.2 features (0/1) from the HICL + gtc arm, primary variant; NaN for stays with no
    medication record of their own (D7.1)."""
    tables, _ = ascertain.ascertain(med, stays)
    m = tables[(ARM, VARIANT)][list(MEDICATION)].astype(float).rename(columns=MEDICATION)
    m.loc[~stays.isin(med["patientunitstayid"].unique())] = np.nan
    return m


def load_data(results: Path = RESULTS) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    lab = pd.read_csv(results / "labels_A.csv")
    lab = lab[lab["primary_site"]].set_index("patientunitstayid").sort_index()
    stays = pd.Index(lab.index, name="patientunitstayid")
    if not stays.equals(ascertain.primary_stays()):
        raise ValueError("labels_A primary stays differ from the section 6 primary cohort")
    clinical = pd.read_csv(results / "features_clinical.csv").set_index("patientunitstayid")
    X = clinical.loc[stays].join(medication_features(stays, load("medication")))
    return X, lab["aki"].to_numpy(int), lab["uniquepid"].to_numpy()


# ---------------------------------------------------------------- report


def report(metrics: pd.DataFrame, comparison: pd.DataFrame, X: pd.DataFrame, y: np.ndarray,
           tuning: pd.DataFrame) -> str:
    def f(v: float) -> str:
        return "n/a" if pd.isna(v) else f"{v:.3f}"

    def pct(v: float) -> str:
        return "n/a" if pd.isna(v) else f"{100 * v:.1f}%"

    label = {"lr_reduced": "Primary: reduced L2 logistic regression (D4.4)",
             "lgbm": "Secondary: LightGBM", "lr_full": "Sensitivity: full penalised LR"}
    missing = int(X[list(MED_COLS)].isna().all(axis=1).sum())
    lines = ["# Models (generated by `python -m polysafe.models`)", "",
             "PREREGISTRATION section 8, D2.1a, D7.1, D7.2. Primary cohort: "
             f"{len(y):,} stays, {int(y.sum())} with AKI; {missing} stays without a medication "
             "record (medication features filled with the training fold's share exposed in "
             "the logistic regressions, missing in LightGBM; D7.1). Each stay's "
             f"{tuning['repeat'].nunique()} out-of-fold predictions are averaged; intervals "
             f"are patient-level BCa with {bootstrap.N_BOOT:,} resamples (D2.1a).", ""]
    for s in comparison.itertuples():
        m = metrics[metrics["spec"] == s.spec].set_index("medication")
        lines += [f"## {label.get(s.spec, s.spec)}", "",
                  "| Model | AUROC (95% BCa) | AUPRC (95% BCa) | Calibration intercept | "
                  "Calibration slope |", "|---|---|---|---:|---:|"]
        for med, name in ((False, "Clinical only"), (True, "Clinical + medication")):
            r = m.loc[med]
            lines.append(f"| {name} | {f(r['auroc'])} ({f(r['auroc_lo'])}-{f(r['auroc_hi'])}) | "
                         f"{f(r['auprc'])} ({f(r['auprc_lo'])}-{f(r['auprc_hi'])}) | "
                         f"{f(r['cal_intercept'])} | {f(r['cal_slope'])} |")
        failed = f" BCa failed: {s.bca_failed}." if s.bca_failed else ""
        drops = "; ".join(
            f"{name} AUROC {m.loc[med, 'auroc_n_used']} / {m.loc[med, 'auroc_n_dropped']}, "
            f"AUPRC {m.loc[med, 'auprc_n_used']} / {m.loc[med, 'auprc_n_dropped']}"
            for med, name in ((False, "clinical only"), (True, "clinical + medication")))
        lines += ["", f"Resamples used / dropped (D2.1a): {drops}."]
        lines += ["", f"**Paired dAUROC (medication - clinical): {s.dauroc:+.3f} "
                  f"(95% BCa {f(s.lo)} to {f(s.hi)})**; resamples attempted / used / dropped "
                  f"{s.n_attempted} / {s.n_used} / {s.n_dropped}.{failed}", "",
                  f"Realised correlation of the two models' predictions (D2.4, D7.2): "
                  f"{f(s.rho_logit)} on the logit scale, {f(s.rho_prob)} on the probability "
                  "scale.", "",
                  f"Simulated coverage (D2.1a), cell A0 = {s.cell_a0:g}, rho = {s.cell_rho:g}, "
                  f"dAUROC = {s.cell_delta:g}: dAUROC interval {pct(s.cov_dauroc)} (range over "
                  f"the cohort's cells {pct(s.cov_dauroc_min)}-{pct(s.cov_dauroc_max)}); "
                  f"clinical-only AUROC {pct(s.cov_auroc_clin)} ({pct(s.cov_clin_min)}-"
                  f"{pct(s.cov_clin_max)}); clinical + medication AUROC "
                  f"{pct(s.cov_auroc_med)} ({pct(s.cov_med_min)}-{pct(s.cov_med_max)}). "
                  "AUPRC intervals: coverage not measured.", "",
                  f"80%-power threshold in that cell (section 9 null statement, D2.1a): BCa "
                  f"{f(s.mde_bca)}; section 9.1 percentile {f(s.mde_pct)}.", ""]
    lines += ["Calibration curves, decision curves and the tuning choice of every fit are in "
              "`results/model_calibration.csv`, `results/model_decision_curve.csv` and "
              "`results/model_tuning.csv`; pooled predictions in "
              "`results/model_predictions.csv`."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Section 8 models and the paired dAUROC.")
    p.add_argument("--out", type=Path, default=RESULTS)
    p.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    args = p.parse_args(argv)
    cov_csv, mde_csv = args.out / "coverage_simulation.csv", args.out / "coverage_mde.csv"
    if not (cov_csv.exists() and mde_csv.exists()):
        raise SystemExit("D2.1a: the BCa coverage re-run must exist before any model is fitted")
    X, y, groups = load_data(args.out)
    clinical = [c for c in X.columns if c not in MED_COLS]
    spec_list = specs(clinical)
    pooled, tuning = cross_validate(X, y, groups, spec_list, workers=args.workers)
    prevalence = float(pd.read_csv(args.out / "power_cohorts.csv").set_index("cohort")
                       .loc[PRIMARY_COHORT, "measured_prevalence"])
    metrics, comparison, curves, dca = evaluate(pooled, y, groups, spec_list,
                                                pd.read_csv(cov_csv), pd.read_csv(mde_csv),
                                                prevalence)
    preds = pd.DataFrame({f"p_{s}_{'medication' if m else 'clinical'}": v
                          for (s, m), v in pooled.items()}, index=X.index)
    preds.to_csv(args.out / "model_predictions.csv")
    metrics.to_csv(args.out / "model_metrics.csv", index=False)
    comparison.to_csv(args.out / "model_comparison.csv", index=False)
    curves.to_csv(args.out / "model_calibration.csv", index=False)
    dca.to_csv(args.out / "model_decision_curve.csv", index=False)
    tuning.to_csv(args.out / "model_tuning.csv", index=False)
    md = report(metrics, comparison, X, y, tuning)
    (args.out / "models.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
