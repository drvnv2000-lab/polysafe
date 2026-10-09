"""BCa coverage re-run of the section 9.1 simulation (PREREGISTRATION D2.1, D2.1a).

Same cohorts, grid, seed and simulated studies as ``polysafe.power``. Each simulated
study draws its labels, folds, percentile-bootstrap weights and score noise from the same
random stream, in the same order, as ``power.one_study``; tau2 and the measured
prevalences are read from the committed ``results/power_cohorts.csv``. The percentile
bounds and estimates therefore reproduce ``results/power_simulation_raw.csv.gz``, and the
run checks that they do before writing anything.

After those draws each study takes ``bootstrap.N_BOOT_SIM`` further patient-weight
resamples and the leave-one-patient-out jackknife, and computes BCa intervals with
``bootstrap.bca_from`` (the formula used for the real intervals) for:

* the dAUROC (true value: the grid delta);
* the AUROC of the clinical-only score (true value: A0);
* the AUROC of the combined score (true value: A0 + delta).

Predictions are fixed out-of-fold predictions; nothing is refitted inside the bootstrap.
A BCa interval that does not exist (``BCaUndefined``) is recorded as a failure: it is
left out of coverage, counts as "not detected" for power and type I error, and the number
of failures is reported.

Usage:
    python -m polysafe.coverage   # writes results/coverage_simulation*.csv(.gz) and .md
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from polysafe import bootstrap, power
from polysafe.cv import patient_folds

RESULTS = Path("results")
KEYS = ["cohort", "prevalence", "a0", "delta", "rho", "sim"]


def _bca(theta: float, boot: np.ndarray, jack: np.ndarray) -> tuple[float, float, int, bool]:
    try:
        ci = bootstrap.bca_from(theta, boot, jack)
    except bootstrap.BCaUndefined:
        return np.nan, np.nan, int(np.isnan(boot).sum()), False
    return ci.low, ci.high, ci.n_dropped, True


def one_study(task: tuple) -> list[dict]:
    """``power.one_study`` plus BCa intervals, on the identical simulated study."""
    name, prevalence, tau2, j = task
    c = power._COHORT[name]
    pat, n_pat, n = c["pat"], c["n_pat"], len(c["pat"])
    rng = np.random.default_rng([power.SEED, j, int(prevalence * 1e4), n])
    L, y = power.draw_labels(pat, n_pat, tau2, round(prevalence * n), rng)
    folds = patient_folds(y, pat, seed=1000 * j)
    w = rng.multinomial(n_pat, np.full(n_pat, 1 / n_pat), size=power.N_BOOT)[:, pat].astype(float)
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    # --- everything above is power.one_study's stream; the BCa draws come after it
    wb = rng.multinomial(n_pat, np.full(n_pat, 1 / n_pat),
                         size=bootstrap.N_BOOT_SIM)[:, pat].astype(float)
    jw = (pat[None, :] != np.arange(n_pat)[:, None]).astype(float)  # leave one patient out
    rows = []
    for a0 in power.A0_GRID:
        s1 = power.sigma_for(a0, prevalence)
        p1 = power.cv_predict(L + s1 * z1, y, folds)
        auc1, pct1 = power.auc(p1, y), power.weighted_auc(p1, y, w)
        b1, j1 = power.weighted_auc(p1, y, wb), power.weighted_auc(p1, y, jw)
        clin = _bca(auc1, b1, j1)
        for delta in power.DELTA_GRID:
            s2 = power.sigma_for(round(a0 + delta, 4), prevalence)
            for rho in power.RHO_GRID:
                kap = power.kappa_for(rho, s1, s2)
                row = {"cohort": name, "prevalence": prevalence, "a0": a0, "delta": delta,
                       "rho": rho, "sim": j}
                if kap is None:
                    rows.append({**row, "feasible": False})
                    continue
                p2 = power.cv_predict(L + s2 * (kap * z1 + np.sqrt(1 - kap * kap) * z2), y,
                                      folds)
                auc2 = power.auc(p2, y)
                lo, hi = np.percentile(power.weighted_auc(p2, y, w) - pct1, [2.5, 97.5])
                b2, j2 = power.weighted_auc(p2, y, wb), power.weighted_auc(p2, y, jw)
                d = _bca(auc2 - auc1, b2 - b1, j2 - j1)
                comb = _bca(auc2, b2, j2)
                rows.append({**row, "feasible": True, "est": auc2 - auc1, "auc_clin": auc1,
                             "lo": lo, "hi": hi, "auc_comb": auc2,
                             "bca_lo": d[0], "bca_hi": d[1], "bca_dropped": d[2],
                             "bca_ok": d[3],
                             "clin_lo": clin[0], "clin_hi": clin[1], "clin_ok": clin[3],
                             "comb_lo": comb[0], "comb_hi": comb[1], "comb_ok": comb[3]})
    return rows


def tasks_from(cohorts_csv: Path, sims: int) -> list[tuple]:
    """The 9.1 task list: each cohort at its measured prevalence and D1.2's two."""
    c = pd.read_csv(cohorts_csv)
    return [(r.cohort, prev, float(r.tau2), j) for r in c.itertuples()
            for prev in (float(r.measured_prevalence), *power.PREVALENCES)
            for j in range(sims)]


def check_reproduces(raw: pd.DataFrame, committed: pd.DataFrame) -> float:
    """Max absolute difference from the committed 9.1 raw file; raises if not exact."""
    cols = ["est", "auc_clin", "lo", "hi"]
    a = raw[raw["feasible"]].set_index(KEYS)[cols].sort_index()
    b = committed[committed["feasible"]].set_index(KEYS)[cols].sort_index()
    if not a.index.equals(b.index):
        raise ValueError("re-run cells differ from the committed section 9.1 cells")
    diff = float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy())))
    if diff > 1e-12:
        raise ValueError(f"re-run does not reproduce section 9.1 (max difference {diff:g})")
    return diff


def covered(lo: pd.Series, hi: pd.Series, true: pd.Series) -> pd.Series:
    """1.0 if the interval contains the true value, 0.0 if not, NaN if it does not exist
    (failed intervals are counted separately, not as misses)."""
    return ((lo <= true) & (true <= hi)).astype(float).where(lo.notna() & hi.notna())


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    ok = raw[raw["feasible"]].copy()
    ok["bca_cov"] = covered(ok["bca_lo"], ok["bca_hi"], ok["delta"])
    ok["pct_cov"] = covered(ok["lo"], ok["hi"], ok["delta"])
    ok["clin_cov"] = covered(ok["clin_lo"], ok["clin_hi"], ok["a0"])
    ok["comb_cov"] = covered(ok["comb_lo"], ok["comb_hi"], (ok["a0"] + ok["delta"]).round(4))
    ok["bca_det"] = ok["bca_lo"] > 0
    ok["pct_det"] = ok["lo"] > 0
    ok["bca_width"] = ok["bca_hi"] - ok["bca_lo"]
    return (ok.groupby(["cohort", "prevalence", "a0", "rho", "delta"])
            .agg(n_sims=("sim", "size"),
                 bca_failed=("bca_ok", lambda s: int((~s.astype(bool)).sum())),
                 clin_failed=("clin_ok", lambda s: int((~s.astype(bool)).sum())),
                 comb_failed=("comb_ok", lambda s: int((~s.astype(bool)).sum())),
                 dropped_resamples=("bca_dropped", "sum"),
                 bca_coverage=("bca_cov", "mean"), pct_coverage=("pct_cov", "mean"),
                 clin_coverage=("clin_cov", "mean"), comb_coverage=("comb_cov", "mean"),
                 bca_power=("bca_det", "mean"), pct_power=("pct_det", "mean"),
                 bca_width=("bca_width", "mean"), mean_est=("est", "mean"))
            .reset_index())


def mde_table(summary: pd.DataFrame) -> pd.DataFrame:
    """80%-power threshold per cohort, prevalence, A0 and rho (``power.mde``), BCa and
    percentile, from the same simulated studies."""
    out = []
    for keys, g in summary.groupby(["cohort", "prevalence", "a0", "rho"]):
        out.append({**dict(zip(["cohort", "prevalence", "a0", "rho"], keys, strict=True)),
                    "mde_bca": power.mde(g.rename(columns={"bca_power": "power"})),
                    "mde_pct": power.mde(g.rename(columns={"pct_power": "power"}))})
    return pd.DataFrame(out)


def report(summary: pd.DataFrame, mde: pd.DataFrame, cohorts: pd.DataFrame,
           diff: float) -> str:
    prim = cohorts[cohorts["cohort"].str.startswith("n=671")].iloc[0]
    sel = summary[(summary["cohort"] == prim["cohort"])
                  & (summary["prevalence"] == prim["measured_prevalence"])]
    mid = sel[sel["a0"] == 0.75]
    rhos = sorted(sel["rho"].unique())
    null = summary[summary["delta"] == 0]
    fails = summary[["bca_failed", "clin_failed", "comb_failed"]].sum()

    def pct(v: float) -> str:
        return f"{100 * v:.1f}%"

    def y(v: float) -> str:
        return "> 0.10" if np.isnan(v) else f"{v:.3f}"

    lines = [
        "# BCa coverage re-run of the section 9.1 simulation "
        "(generated by `python -m polysafe.coverage`)", "",
        "PREREGISTRATION D2.1, D2.1a. Same cohorts, grid, seed and simulated studies as "
        f"section 9.1, BCa with {bootstrap.N_BOOT_SIM:,} patient-level resamples per "
        "simulated study and the leave-one-patient-out jackknife; fixed out-of-fold "
        "predictions, no refitting. Apart from the reproduction check, every number below "
        "is read from `results/coverage_simulation.csv` and `results/coverage_mde.csv`.", "",
        "**Same studies as 9.1.** The re-run's percentile bounds, estimates and "
        "clinical-only AUROCs reproduce `results/power_simulation_raw.csv.gz` (max absolute "
        f"difference {diff:g}, checked during the run).", "",
        f"**Undefined intervals.** BCa failed (D2.1a: no interval) for {int(fails['bca_failed'])} "
        f"dAUROC, {int(fails['clin_failed'])} clinical-only AUROC and "
        f"{int(fails['comb_failed'])} combined AUROC intervals; "
        f"{int(summary['dropped_resamples'].sum()):,} resamples were dropped as undefined.", "",
        "**Type I error at a true dAUROC of 0** (share of studies whose lower bound is above "
        "0; nominal one-sided 2.5%): BCa {} on average (max {} in any cell); percentile {} "
        "(max {}) on the same studies.".format(
            pct(null["bca_power"].mean()), pct(null["bca_power"].max()),
            pct(null["pct_power"].mean()), pct(null["pct_power"].max())), "",
        "**Coverage of the 95% dAUROC interval** over all cells: BCa {}-{} (mean {}), "
        "percentile {}-{} (mean {}). Clinical-only AUROC (true A0): BCa {}-{}. Combined "
        "AUROC (true A0 + dAUROC): BCa {}-{}.".format(
            pct(summary["bca_coverage"].min()), pct(summary["bca_coverage"].max()),
            pct(summary["bca_coverage"].mean()), pct(summary["pct_coverage"].min()),
            pct(summary["pct_coverage"].max()), pct(summary["pct_coverage"].mean()),
            pct(summary["clin_coverage"].min()), pct(summary["clin_coverage"].max()),
            pct(summary["comb_coverage"].min()), pct(summary["comb_coverage"].max())), "",
        "**Primary cohort, measured prevalence {:.1%}, A0 = 0.75:**".format(
            prim["measured_prevalence"]), "",
        "| rho | BCa dAUROC coverage (min-max over dAUROC) | type I BCa | type I percentile "
        "| 80%-power threshold BCa | percentile |",
        "|---:|---|---:|---:|---:|---:|"]
    m = mde[(mde["cohort"] == prim["cohort"])
            & (mde["prevalence"] == prim["measured_prevalence"]) & (mde["a0"] == 0.75)]
    for r in rhos:
        g = mid[mid["rho"] == r]
        z = g[g["delta"] == 0].iloc[0]
        t = m[m["rho"] == r].iloc[0]
        lines.append(f"| {r:g} | {pct(g['bca_coverage'].min())}-{pct(g['bca_coverage'].max())}"
                     f" | {pct(z['bca_power'])} | {pct(z['pct_power'])} | {y(t['mde_bca'])} "
                     f"| {y(t['mde_pct'])} |")
    lines += ["", f"Coverage over all cells of this cohort and prevalence (the D2.1a range): "
              f"dAUROC BCa {pct(sel['bca_coverage'].min())}-{pct(sel['bca_coverage'].max())}; "
              f"clinical-only AUROC {pct(sel['clin_coverage'].min())}-"
              f"{pct(sel['clin_coverage'].max())}; combined AUROC "
              f"{pct(sel['comb_coverage'].min())}-{pct(sel['comb_coverage'].max())}. Per-cell "
              "values are in `results/coverage_simulation.csv`, thresholds for every A0 in "
              "`results/coverage_mde.csv`."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="BCa coverage re-run of the 9.1 simulation.")
    p.add_argument("--labels", type=Path, default=RESULTS / "labels_A.csv")
    p.add_argument("--out", type=Path, default=RESULTS)
    p.add_argument("--sims", type=int, default=power.N_SIMS)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args(argv)

    cohorts_csv = args.out / "power_cohorts.csv"
    cohorts = power.load_cohorts(args.labels)
    tasks = tasks_from(cohorts_csv, args.sims)
    prevs = sorted({t[1] for t in tasks})
    sigmas = {(round(a + d, 4), pv): power.sigma_for(round(a + d, 4), pv)
              for pv in prevs for a in power.A0_GRID for d in power.DELTA_GRID}
    with Pool(args.workers, initializer=power._init, initargs=(cohorts, sigmas)) as pool:
        raw = pd.DataFrame([r for rows in pool.imap_unordered(one_study, tasks, chunksize=2)
                            for r in rows])
    committed = pd.read_csv(args.out / "power_simulation_raw.csv.gz")
    if args.sims != power.N_SIMS:
        committed = committed[committed["sim"] < args.sims]
    diff = check_reproduces(raw, committed)
    summary = summarise(raw)
    mde = mde_table(summary)
    raw.to_csv(args.out / "coverage_simulation_raw.csv.gz", index=False)
    summary.to_csv(args.out / "coverage_simulation.csv", index=False)
    mde.to_csv(args.out / "coverage_mde.csv", index=False)
    md = report(summary, mde, pd.read_csv(cohorts_csv), diff)
    (args.out / "coverage_simulation.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
