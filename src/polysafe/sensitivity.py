"""Sensitivity analyses of the primary comparison (PREREGISTRATION D8.1, D9.2).

Each analysis repeats only the primary comparison of D7.2: the clinical-only and
clinical + medication reduced L2 logistic regressions, every D7.2 setting unchanged
(``models.cross_validate`` with the ``lr_reduced`` spec). It reports both AUROCs, the
paired dAUROC with its D2.1a BCa interval and the D2.4 correlation. LightGBM is not
repeated. Which analyses are run was fixed before any model was fitted; how each one is
run and reported (D8.1) was fixed after the primary result was seen (D9.2).

* ascertainment: the five arms x four variants of section 6 on the primary cohort
  (20 analyses; HICL + gtc, primary variant is the primary analysis itself);
* full cohort: the 868 baseline-A stays at all hospitals;
* follow-up: the primary cohort restricted to stays with >= 48 h of post-T follow-up;
* baseline C: cohort and labels under baseline C at medication-recording hospitals;
* hospital clusters: stays linked by a shared hospital or patient form one cluster, used
  for the outer folds, the inner tuning splits and the BCa resampling;
* ethnicity: descriptive, on the primary analysis's pooled predictions.

Usage:
    python -m polysafe.sensitivity               # writes results/sensitivity*.csv and .md
    python -m polysafe.sensitivity --smoke       # 1 repeat, 200 resamples, to a scratch dir
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from polysafe import ascertain, bootstrap, features, models
from polysafe.cv import N_REPEATS, N_SPLITS, patient_folds
from polysafe.tables import load

RESULTS = Path("results")
FULL_COHORT = "n=868 (all sites)"
EXPECTED = {"full": 868, "full_record_less": 217, "followup": 314, "baseline_c": 497}  # D8.1
FOLLOWUP_H = 48.0
MIN_EVENTS_FOR_CI = 10  # D8.1 ethnicity rule
ARM_LABEL = {"generic_string": "Generic string", "generic_brand": "Generic + brand",
             "hicl": "HICL-based", "hicl_gtc": "HICL + gtc",
             "hicl_reviewed": "Review-corrected HICL (exploratory, D6.3)"}
VARIANT_LABEL = {"primary": "primary", "route_unknown": "systemic = unknown included (D6.2)",
                 "unsure": "unsure keys included (D5.13)",
                 "named_only": "unnamed rows excluded (D5.12)"}
NOT_MEASURED = "not measured"


# ---------------------------------------------------------------- inputs


def medication_features(tables: dict, arm: str, variant: str, stays: pd.Index,
                        recorded: set) -> pd.DataFrame:
    """D7.2 features from one ascertainment arm and variant; NaN for stays with no
    medication record of their own (D7.1)."""
    m = (tables[(arm, variant)].reindex(stays)[list(models.MEDICATION)].astype(float)
         .rename(columns=models.MEDICATION))
    m.loc[~stays.isin(list(recorded))] = np.nan
    return m


def hospital_clusters(hospital, patient) -> np.ndarray:
    """Connected components of stays linked by a shared hospital or a shared patient."""
    parent: dict = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for h, p in zip(hospital, patient, strict=True):
        a, b = ("h", h), ("p", p)
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = [find(("h", h)) for h in hospital]
    _, ids = np.unique([f"{r[0]}:{r[1]}" for r in roots], return_inverse=True)
    return ids


def folds_without_event(y: np.ndarray, groups: np.ndarray, repeats: int) -> list[str]:
    folds = patient_folds(y, groups, n_repeats=repeats, seed=0)
    return [f"repeat {r} fold {k}" for r in range(repeats) for k in range(N_SPLITS)
            if not y[folds[r] == k].any()]


# ---------------------------------------------------------------- one comparison


def compare(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, *, repeats: int,
            n_boot: int, workers: int) -> tuple[dict, dict]:
    """The D7.2 primary comparison on (X, y), folds and resampling grouped by ``groups``."""
    clinical = [c for c in X.columns if c not in models.MED_COLS]
    spec = models.specs(clinical)[0]
    if spec.name != "lr_reduced":
        raise RuntimeError("the primary spec is no longer first in models.specs")
    pooled, _ = models.cross_validate(X, y, groups, [spec], repeats=repeats, workers=workers)
    p0, p1 = pooled[(spec.name, False)], pooled[(spec.name, True)]
    a0, a1 = bootstrap.auroc(p0, y), bootstrap.auroc(p1, y)
    ci = models.interval(bootstrap.delta_auroc(p1, p0, y), groups, n_boot)
    row = {"n": len(y), "events": int(y.sum()), "clusters": len(np.unique(groups)),
           "auroc_clinical": a0, "auroc_medication": a1, "dauroc": a1 - a0,
           "lo": ci["lo"], "hi": ci["hi"], "n_attempted": ci.get("n_attempted"),
           "n_used": ci.get("n_used"), "n_dropped": ci.get("n_dropped"),
           "bca_failed": ci["bca_failed"],
           "rho_logit": float(np.corrcoef(models.logit(p0), models.logit(p1))[0, 1]),
           "rho_prob": float(np.corrcoef(p0, p1)[0, 1])}
    return row, {"p_clinical": p0, "p_medication": p1}


def coverage(row: dict, cov: pd.DataFrame, mde: pd.DataFrame, cohort: str,
             prevalence: float) -> dict:
    cell = models.matched_cell(cov, mde, prevalence, row["auroc_clinical"], row["rho_logit"],
                               row["dauroc"], cohort=cohort)
    return {"coverage_cohort": cohort, "cell_a0": cell["cell_a0"],
            "cell_rho": cell["cell_rho"], "cell_delta": cell["cell_delta"],
            "cov_dauroc": cell["cov_dauroc"], "mde_bca": cell["mde_bca"]}


# ---------------------------------------------------------------- ethnicity (D3.4)


def ethnicity_table(preds: pd.DataFrame, y: pd.Series, groups: pd.Series, eth: pd.Series,
                    n_boot: int) -> pd.DataFrame:
    """Within each group: stays, events, both AUROCs and the dAUROC. A BCa interval only
    with >= 10 events; no AUROC without an event. Descriptive; no test between groups."""
    eth = eth.reindex(preds.index).fillna("missing").replace("", "missing")
    rows = []
    for name, idx in eth.groupby(eth).groups.items():
        yy = y.loc[idx].to_numpy(int)
        p0 = preds.loc[idx, "p_lr_reduced_clinical"].to_numpy(float)
        p1 = preds.loc[idx, "p_lr_reduced_medication"].to_numpy(float)
        row = {"ethnicity": name, "n": len(yy), "events": int(yy.sum()),
               "auroc_clinical": np.nan, "auroc_medication": np.nan, "dauroc": np.nan,
               "lo": np.nan, "hi": np.nan, "interval": "no event: no AUROC"}
        if yy.any() and not yy.all():
            row.update(auroc_clinical=bootstrap.auroc(p0, yy),
                       auroc_medication=bootstrap.auroc(p1, yy))
            row["dauroc"] = row["auroc_medication"] - row["auroc_clinical"]
            row["interval"] = f"fewer than {MIN_EVENTS_FOR_CI} events: point estimate only"
            if yy.sum() >= MIN_EVENTS_FOR_CI:
                ci = models.interval(bootstrap.delta_auroc(p1, p0, yy),
                                     groups.loc[idx].to_numpy(), n_boot)
                row.update(lo=ci["lo"], hi=ci["hi"],
                           interval=f"BCa failed: {ci['bca_failed']}" if ci["bca_failed"]
                           else "BCa")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n", "ethnicity"], ascending=[False, True])


# ---------------------------------------------------------------- run


def run(out: Path, results: Path, repeats: int, n_boot: int, workers: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    lab_a = pd.read_csv(results / "labels_A.csv").set_index("patientunitstayid").sort_index()
    lab_c = pd.read_csv(results / "labels_C.csv").set_index("patientunitstayid").sort_index()
    clinical = pd.read_csv(results / "features_clinical.csv").set_index("patientunitstayid")
    med = load("medication")
    recorded = set(med["patientunitstayid"].unique())
    primary = lab_a[lab_a["primary_site"]]
    if not pd.Index(primary.index).equals(ascertain.primary_stays()):
        raise ValueError("labels_A primary stays differ from the section 6 primary cohort")
    followup = primary[primary["followup_h"] >= FOLLOWUP_H]
    base_c = lab_c[lab_c["primary_site"]]
    record_less = int((~lab_a.index.isin(list(recorded))).sum())
    found = {"full": len(lab_a), "full_record_less": record_less, "followup": len(followup),
             "baseline_c": len(base_c)}
    if found != EXPECTED:
        raise ValueError(f"D8.1 cohort sizes differ: expected {EXPECTED}, found {found}")

    all_stays = pd.Index(sorted(set(lab_a.index) | set(base_c.index)),
                         name="patientunitstayid")
    tables, _ = ascertain.ascertain(med, all_stays)
    missing_c = base_c.index.difference(clinical.index)
    if len(missing_c):
        extra = features.build(features.load_feature_inputs(), missing_c)
        clinical = pd.concat([clinical, extra[clinical.columns]])
    cov = pd.read_csv(results / "coverage_simulation.csv")
    mde = pd.read_csv(results / "coverage_mde.csv")
    prev = pd.read_csv(results / "power_cohorts.csv").set_index("cohort")["measured_prevalence"]

    def design(labels: pd.DataFrame, arm=models.ARM, variant=models.VARIANT):
        stays = pd.Index(labels.index, name="patientunitstayid")
        X = clinical.loc[stays].join(medication_features(tables, arm, variant, stays,
                                                         recorded))
        return X, labels["aki"].to_numpy(int)

    partial = out / "sensitivity_partial.csv"
    done = pd.read_csv(partial) if partial.exists() else pd.DataFrame(columns=["analysis"])
    rows = done.to_dict("records")

    def record(row: dict) -> None:
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial, index=False)
        print(f"done {row['analysis']}: dAUROC {row.get('dauroc', float('nan')):+.3f}",
              flush=True)

    kw = {"repeats": repeats, "n_boot": n_boot, "workers": workers}
    for arm in ascertain.ARMS:
        for variant in ascertain.VARIANTS:
            key = f"ascertainment:{arm}:{variant}"
            if key in set(done["analysis"]):
                continue
            X, y = design(primary, arm, variant)
            row, _ = compare(X, y, primary["uniquepid"].to_numpy(), **kw)
            row |= coverage(row, cov, mde, models.PRIMARY_COHORT,
                            float(prev[models.PRIMARY_COHORT]))
            record({"analysis": key, "group": "ascertainment", "arm": arm,
                    "variant": variant} | row)

    others = [("full_cohort", lab_a, lab_a["uniquepid"].to_numpy(), FULL_COHORT),
              ("followup_48h", followup, followup["uniquepid"].to_numpy(), None),
              ("baseline_c", base_c, base_c["uniquepid"].to_numpy(), None),
              ("hospital_clusters", primary,
               hospital_clusters(primary["hospitalid"], primary["uniquepid"]), None)]
    for key, labels, groups, cohort in others:
        if key in set(done["analysis"]):
            continue
        X, y = design(labels)
        base = {"analysis": key, "group": key, "arm": models.ARM, "variant": models.VARIANT}
        if key == "hospital_clusters":
            empty = folds_without_event(y, groups, repeats)
            if empty:  # D8.1: stop; the PI decides
                record(base | {"n": len(y), "events": int(y.sum()),
                               "clusters": len(np.unique(groups)),
                               "stopped": "outer test fold without an AKI event: "
                               + ", ".join(empty)})
                continue
        row, _ = compare(X, y, groups, **kw)
        if cohort is not None:
            row |= coverage(row, cov, mde, cohort, float(prev[cohort]))
        record(base | row)

    table = pd.DataFrame(rows)
    table.to_csv(out / "sensitivity.csv", index=False)
    preds = pd.read_csv(results / "model_predictions.csv").set_index("patientunitstayid")
    eth = pd.read_csv(results / "subgroups.csv").set_index("patientunitstayid")["ethnicity"]
    eth_table = ethnicity_table(preds, primary["aki"].astype(int), primary["uniquepid"], eth,
                                n_boot)
    eth_table.to_csv(out / "sensitivity_ethnicity.csv", index=False)
    primary_row = pd.read_csv(results / "model_comparison.csv").set_index("spec").loc[
        "lr_reduced"]
    md = report(table, eth_table, primary_row, repeats, n_boot)
    (out / "sensitivity.md").write_text(md, encoding="utf-8")
    partial.unlink()
    print(md)
    return 0


# ---------------------------------------------------------------- report


def _f(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:.3f}"


def _ci(r) -> str:
    return "n/a" if pd.isna(r["lo"]) else f"{_f(r['lo'])} to {_f(r['hi'])}"


def _excl(r) -> str:
    if pd.isna(r["lo"]):
        return "n/a"
    return "yes" if r["lo"] > 0 or r["hi"] < 0 else "no"


def _cov(r) -> str:
    if "cov_dauroc" not in r or pd.isna(r.get("cov_dauroc")):
        return NOT_MEASURED
    return f"{100 * r['cov_dauroc']:.1f}% ({r['coverage_cohort']})"


def report(table: pd.DataFrame, eth: pd.DataFrame, primary: pd.Series, repeats: int,
           n_boot: int) -> str:
    ref = float(primary["dauroc"])
    lines = ["# Sensitivity analyses (generated by `python -m polysafe.sensitivity`)", "",
             "PREREGISTRATION D8.1 and D9.2. Each analysis repeats only the primary comparison "
             "of D7.2 (reduced L2 logistic regression, clinical-only against clinical + "
             f"medication; {N_SPLITS} folds x {repeats} repeats; BCa with {n_boot:,} "
             "resamples). LightGBM is not repeated. **Which analyses are run was fixed before "
             "any model was fitted; how each is run and reported was fixed after the primary "
             "result was seen (D9.2).** No sensitivity analysis replaces the primary result; "
             "no test between analyses.", "",
             f"Primary result (D7.2, `results/models.md`): dAUROC {ref:+.3f} "
             f"(95% BCa {_f(primary['lo'])} to {_f(primary['hi'])}).", ""]
    asc = table[table["group"] == "ascertainment"]
    lines += ["## Exposure ascertainment (section 6, D5.9, D6.3)", "",
              "| Arm | Variant | AUROC clinical | AUROC +medication | dAUROC (95% BCa) | "
              "Excludes 0 | Change vs primary | rho (logit) | dAUROC coverage |",
              "|---|---|---:|---:|---|---|---:|---:|---|"]
    for _, r in asc.iterrows():
        mark = " **(primary analysis)**" if (r["arm"], r["variant"]) == (
            models.ARM, models.VARIANT) else ""
        lines.append(f"| {ARM_LABEL[r['arm']]}{mark} | {VARIANT_LABEL[r['variant']]} | "
                     f"{_f(r['auroc_clinical'])} | {_f(r['auroc_medication'])} | "
                     f"{r['dauroc']:+.3f} ({_ci(r)}) | {_excl(r)} | "
                     f"{r['dauroc'] - ref:+.3f} | {_f(r['rho_logit'])} | {_cov(r)} |")
    same = asc[(asc["arm"] == models.ARM) & (asc["variant"] == models.VARIANT)]
    if len(same):
        diff = abs(float(same["dauroc"].iloc[0]) - ref)
        lines += ["", "Reproduction check: the HICL + gtc, primary-variant row "
                  + ("reproduces" if diff < 1e-9 else f"DIFFERS from (by {diff:.2e})")
                  + " the primary dAUROC."]
    names = {"full_cohort": "Full cohort, all hospitals (section 6)",
             "followup_48h": f">= {FOLLOWUP_H:g} h post-T follow-up (section 3, D2.2)",
             "baseline_c": "Baseline C, medication-recording hospitals (section 4)",
             "hospital_clusters": "Hospital clusters (section 7)"}
    lines += ["", "## Cohort, follow-up, baseline and grouping", "",
              "| Analysis | Stays | Events | Groups | AUROC clinical | AUROC +medication | "
              "dAUROC (95% BCa) | Excludes 0 | Change vs primary | rho (logit) | "
              "dAUROC coverage |", "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|"]
    for _, r in table[table["group"] != "ascertainment"].iterrows():
        if isinstance(r.get("stopped"), str) and r["stopped"]:
            lines.append(f"| {names[r['analysis']]} | {r['n']:,} | {r['events']} | "
                         f"{r['clusters']} | stopped: {r['stopped']} | | | | | | |")
            continue
        lines.append(f"| {names[r['analysis']]} | {int(r['n']):,} | {int(r['events'])} | "
                     f"{int(r['clusters'])} | {_f(r['auroc_clinical'])} | "
                     f"{_f(r['auroc_medication'])} | {r['dauroc']:+.3f} ({_ci(r)}) | "
                     f"{_excl(r)} | {r['dauroc'] - ref:+.3f} | {_f(r['rho_logit'])} | "
                     f"{_cov(r)} |")
    lines += ["", "Groups: patients, except hospital clusters (stays linked by a shared "
              "hospital or patient). Coverage is read from the D2.1a simulation only where "
              "D8.1 names a simulated cohort (primary cohort; full cohort n = 868); "
              "otherwise not measured.", "",
              "## Ethnicity (D3.4, descriptive)", "",
              "On the primary analysis's pooled predictions. BCa only for groups with at least "
              f"{MIN_EVENTS_FOR_CI} AKI events; no test between groups.", "",
              "| Ethnicity | Stays | Events | AUROC clinical | AUROC +medication | dAUROC | "
              "95% BCa | Interval |", "|---|---:|---:|---:|---:|---:|---|---|"]
    for _, r in eth.iterrows():
        d = "n/a" if pd.isna(r["dauroc"]) else f"{r['dauroc']:+.3f}"
        lines.append(f"| {r['ethnicity']} | {r['n']} | {r['events']} | "
                     f"{_f(r['auroc_clinical'])} | {_f(r['auroc_medication'])} | {d} | "
                     f"{_ci(r)} | {r['interval']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="D8.1 sensitivity analyses.")
    p.add_argument("--out", type=Path, default=RESULTS)
    p.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    p.add_argument("--smoke", action="store_true",
                   help="1 repeat and 200 resamples, written to --out (use a scratch dir)")
    args = p.parse_args(argv)
    if args.smoke and args.out.resolve() == RESULTS.resolve():
        raise SystemExit("--smoke must not write to results/")
    repeats, n_boot = (1, 200) if args.smoke else (N_REPEATS, bootstrap.N_BOOT)
    return run(args.out, RESULTS, repeats, n_boot, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
