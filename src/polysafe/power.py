"""Power simulation for the paired dAUROC (PREREGISTRATION section 9, amended by D1.4).

Mirrors the pre-specified primary analysis on the real analysis cohorts:

* the real stays and real patient grouping (``uniquepid``) of the n = 671
  primary cohort and the n = 868 all-sites cohort (baseline A);
* the real fold structure (``cv.patient_folds``: 5-fold StratifiedGroupKFold
  grouped by patient, 10 repeats);
* the pre-specified estimate: dAUROC of out-of-fold predictions averaged over
  the repeats, with a patient-level bootstrap 95% CI. "Detected" means the
  CI's lower bound is above 0.

Data-generating model, one simulated study:

    L = b_patient + e_stay, b ~ N(0, tau2), e ~ N(0, 1 - tau2)    latent AKI liability
    AKI = the k stays with the highest L, k = round(prevalence x n)
    s_clin = L + sigma1 * z1                                      AUROC = A0
    s_comb = L + sigma2 * (kappa * z1 + sqrt(1 - kappa^2) * z2)   AUROC = A0 + delta

kappa makes the Pearson correlation between the two scores equal rho. tau2 is
calibrated so simulated AKI clusters within patients as much as the real labels
do. Each score is recalibrated by logistic regression on the training folds and
predicted on the held-out fold, as a fitted model would be.

Not simulated: the extra variance of fitting a many-feature model. rho < 1
stands for all disagreement between the two models.

Usage:
    python -m polysafe.power      # writes results/power_simulation.{csv,md}
"""

from __future__ import annotations

import argparse
import os
from functools import cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from polysafe.cv import N_SPLITS, patient_folds

SEED = 20261004
N_SIMS = 200
N_BOOT = 1000
TARGET_POWER = 0.80
A0_GRID = (0.70, 0.75, 0.80)
DELTA_GRID = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10)
RHO_GRID = (0.6, 0.75, 0.9, 0.95)
PREVALENCES = (0.129, 0.161)  # D1.2; the measured prevalence of each cohort is added
TAU2_GRID = tuple(round(x, 1) for x in np.arange(0.0, 0.95, 0.1))


# ---------------------------------------------------------------- AUROC


def weighted_auc(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """AUROC per row of weights ``w`` (shape (B, n)); ties count 1/2."""
    order = np.argsort(p, kind="mergesort")
    ps, ys, ws = p[order], y[order].astype(bool), w[:, order]
    starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])
    pos = np.add.reduceat(np.where(ys, ws, 0.0), starts, axis=1)
    neg = np.add.reduceat(np.where(ys, 0.0, ws), starts, axis=1)
    neg_below = np.cumsum(neg, axis=1) - neg
    return (pos * (neg_below + 0.5 * neg)).sum(axis=1) / (pos.sum(axis=1) * neg.sum(axis=1))


def auc(p: np.ndarray, y: np.ndarray) -> float:
    return float(weighted_auc(p, y, np.ones((1, len(p))))[0])


# ---------------------------------------------------------------- calibration


@cache
def sigma_for(target_auc: float, prevalence: float) -> float:
    if (target_auc, prevalence) in _SIGMAS:
        return _SIGMAS[(target_auc, prevalence)]
    """Noise sd so that AUROC(L + sigma z) = target when AKI = top ``prevalence`` of L."""
    rng = np.random.default_rng(SEED)
    L = rng.standard_normal(200_000)
    z = rng.standard_normal(200_000)
    y = L > np.quantile(L, 1 - prevalence)
    return brentq(lambda s: auc(L + s * z, y) - target_auc, 1e-3, 20, xtol=1e-5)


def kappa_for(rho: float, s1: float, s2: float) -> float | None:
    """Noise correlation giving corr(s_clin, s_comb) = rho; None if unreachable."""
    k = (rho * np.sqrt((1 + s1 * s1) * (1 + s2 * s2)) - 1) / (s1 * s2)
    return float(k) if -1 <= k <= 1 else None


def draw_labels(pat: np.ndarray, n_pat: int, tau2: float, k: int, rng):
    L = rng.normal(0, np.sqrt(tau2), n_pat)[pat] + rng.normal(0, np.sqrt(1 - tau2), len(pat))
    y = np.zeros(len(pat), dtype=np.int8)
    y[np.argsort(-L)[:k]] = 1
    return L, y


def pair_concordance(y: np.ndarray, pat: np.ndarray) -> float:
    """Mean over repeat patients of the share of their stay pairs where both have AKI."""
    vals = []
    for idx in pd.Series(np.arange(len(pat))).groupby(pat).groups.values():
        m = len(idx)
        if m > 1:
            s = int(y[np.asarray(idx)].sum())
            vals.append(s * (s - 1) / (m * (m - 1)))
    return float(np.mean(vals))


def calibrate_tau2(y_real: np.ndarray, pat: np.ndarray, draws: int = 300) -> tuple[float, dict]:
    target = pair_concordance(y_real, pat)
    n_pat, k = pat.max() + 1, int(y_real.sum())
    rng = np.random.default_rng(SEED)
    sim = {t: float(np.mean([pair_concordance(draw_labels(pat, n_pat, t, k, rng)[1], pat)
                             for _ in range(draws)])) for t in TAU2_GRID}
    best = min(sim, key=lambda t: abs(sim[t] - target))
    return best, {"observed": target, **{f"tau2={t}": v for t, v in sim.items()}}


# ---------------------------------------------------------------- one study


def _logistic(x_tr, y_tr, x_te, iters: int = 30) -> np.ndarray:
    """Two-parameter logistic regression by Newton-Raphson; returns P(AKI) for x_te."""
    mu, sd = x_tr.mean(), x_tr.std()
    x, xt = (x_tr - mu) / sd, (x_te - mu) / sd
    a, b = np.log(y_tr.mean() / (1 - y_tr.mean())), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + b * x)))
        r, w = y_tr - p, p * (1 - p)
        g0, g1 = r.sum(), (r * x).sum()
        h00, h01, h11 = w.sum(), (w * x).sum(), (w * x * x).sum()
        det = h00 * h11 - h01 * h01
        da, db = (h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det
        a, b = a + da, b + db
        if abs(da) + abs(db) < 1e-10:
            break
    return 1 / (1 + np.exp(-(a + b * xt)))


def cv_predict(s: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold P(AKI) averaged over repeats, the pre-specified pooling."""
    out = np.zeros(len(s))
    for r in range(folds.shape[0]):
        for k in range(N_SPLITS):
            te = folds[r] == k
            out[te] += _logistic(s[~te], y[~te], s[te])
    return out / folds.shape[0]


_COHORT: dict = {}
_SIGMAS: dict = {}


def _init(cohort: dict, sigmas: dict) -> None:
    _COHORT.update(cohort)
    _SIGMAS.update(sigmas)


def one_study(task: tuple) -> list[dict]:
    """Simulate one study for every (A0, delta, rho) cell, with common random numbers."""
    name, prevalence, tau2, j = task
    c = _COHORT[name]
    pat, n_pat, n = c["pat"], c["n_pat"], len(c["pat"])
    rng = np.random.default_rng([SEED, j, int(prevalence * 1e4), n])
    L, y = draw_labels(pat, n_pat, tau2, round(prevalence * n), rng)
    folds = patient_folds(y, pat, seed=1000 * j)
    w = rng.multinomial(n_pat, np.full(n_pat, 1 / n_pat), size=N_BOOT)[:, pat].astype(float)
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    rows = []
    for a0 in A0_GRID:
        s1 = sigma_for(a0, prevalence)
        p1 = cv_predict(L + s1 * z1, y, folds)
        auc1, boot1 = auc(p1, y), weighted_auc(p1, y, w)
        for delta in DELTA_GRID:
            s2 = sigma_for(round(a0 + delta, 4), prevalence)
            for rho in RHO_GRID:
                kap = kappa_for(rho, s1, s2)
                row = {"cohort": name, "prevalence": prevalence, "a0": a0, "delta": delta,
                       "rho": rho, "sim": j}
                if kap is None:
                    rows.append({**row, "feasible": False})
                    continue
                p2 = cv_predict(L + s2 * (kap * z1 + np.sqrt(1 - kap * kap) * z2), y, folds)
                d = weighted_auc(p2, y, w) - boot1
                lo, hi = np.percentile(d, [2.5, 97.5])
                rows.append({**row, "feasible": True, "est": auc(p2, y) - auc1,
                             "auc_clin": auc1, "lo": lo, "hi": hi, "detected": lo > 0})
    return rows


# ---------------------------------------------------------------- summary


def mde(g: pd.DataFrame) -> float:
    """Smallest true dAUROC with power >= TARGET_POWER, linear between grid points."""
    g = g.sort_values("delta")
    d, pw = g["delta"].to_numpy(), g["power"].to_numpy()
    for i in range(1, len(d)):
        if pw[i] >= TARGET_POWER > pw[i - 1]:
            return float(d[i - 1] + (TARGET_POWER - pw[i - 1]) * (d[i] - d[i - 1])
                         / (pw[i] - pw[i - 1]))
    return float("nan") if pw[-1] < TARGET_POWER else float(d[0])


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    ok = raw[raw["feasible"]].assign(width=lambda d: d["hi"] - d["lo"])
    return (ok.groupby(["cohort", "prevalence", "a0", "rho", "delta"])
            .agg(power=("detected", "mean"), mean_est=("est", "mean"),
                 ci_width=("width", "mean"), n_sims=("sim", "size"))
            .reset_index())


def load_cohorts(labels_csv: Path) -> dict:
    lab = pd.read_csv(labels_csv)
    cohorts = {}
    for name, df in {"n=671 (primary sites)": lab[lab["primary_site"]],
                     "n=868 (all sites)": lab}.items():
        pat = pd.factorize(df["uniquepid"])[0]
        cohorts[name] = {"pat": pat, "n_pat": int(pat.max() + 1),
                         "y": df["aki"].to_numpy().astype(np.int8)}
    return cohorts


def report(out: Path) -> str:
    """The section 9 appendix, built only from the saved simulation outputs."""
    s = pd.read_csv(out / "power_simulation.csv")
    m = pd.read_csv(out / "power_mde.csv")
    c = pd.read_csv(out / "power_cohorts.csv")
    measured = dict(zip(c["cohort"], c["measured_prevalence"], strict=True))
    prim = c[c["cohort"].str.startswith("n=671")].iloc[0]

    def y(v: float) -> str:
        return "> 0.10" if np.isnan(v) else f"{v:.3f}"

    def prev_label(cohort: str, p: float) -> str:
        return f"{p:.3f} (measured)" if p == measured[cohort] else f"{p:.3f}"

    mid = m[m["a0"] == 0.75]
    rhos = sorted(m["rho"].unique())
    lines = [
        "### 9.1 Simulation result (appended before any model was trained)", "",
        "Generated by `python -m polysafe.power` (`results/power_simulation.csv`, "
        "`results/power_mde.csv`). Every number below is read from those files.", "",
        "**Design simulated (identical to the analysis):** the real baseline-A cohorts "
        "(n = 671 stays / {:,} patients at medication-recording sites, {:,} events; "
        "n = 868 all sites) with their real `uniquepid` grouping; 5-fold "
        "`StratifiedGroupKFold` by patient x 10 repeats (`polysafe.cv.patient_folds`); "
        "out-of-fold predictions averaged over repeats; dAUROC with a patient-level "
        "bootstrap ({:,} resamples) percentile 95% CI. *Detected* = lower bound > 0."
        .format(int(prim["patients"]), int(prim["events"]), N_BOOT), "",
        "**Assumptions X:** AKI from a latent liability with a patient random effect "
        "calibrated to the real within-patient clustering of AKI "
        "(selected tau2 = " + ", ".join(f"{v:g}" for v in c["tau2"]) + " for the two cohorts: "
        "the real labels show almost no clustering of AKI within patients); "
        "clinical-only AUROC A0 in {0.70, 0.75, 0.80}; Pearson correlation rho "
        "between the two models' risk scores in {0.6, 0.75, 0.9, 0.95}; true dAUROC "
        "0-0.10; prevalence 0.129 and 0.161 (D1.2) plus each cohort's measured "
        f"prevalence; {int(s['n_sims'].min())} simulated studies per cell (Monte Carlo SE "
        f"of an 80% power estimate ~ {100 * np.sqrt(0.16 / s['n_sims'].min()):.1f} points); "
        f"seed {SEED}. Not simulated: the extra variance of fitting a many-feature "
        "model; rho < 1 stands for all disagreement between the two models.", "",
        "**Minimum dAUROC detected with 80% probability** (A0 = 0.75; linear "
        "interpolation between grid points):", "",
        "| Cohort | Prevalence | " + " | ".join(f"rho = {r:g}" for r in rhos) + " |",
        "|---|---|" + "---:|" * len(rhos)]
    for (cohort, p), g in mid.groupby(["cohort", "prevalence"], sort=False):
        vals = g.set_index("rho")["mde"]
        lines.append(f"| {cohort} | {prev_label(cohort, p)} | "
                     + " | ".join(y(vals.get(r, np.nan)) for r in rhos) + " |")
    spread = m.groupby(["cohort", "prevalence", "rho"])["mde"].agg(["min", "max"])
    lines += ["", "Across A0 = 0.70-0.80 the same cells range "
              f"{y(spread['min'].min())}-{y(spread['max'].max())}; per-A0 values are in "
              "`results/power_mde.csv`.", ""]

    null = s[s["delta"] == 0]
    bias = s[s["delta"] > 0].assign(b=lambda d: (d["mean_est"] - d["delta"]).round(3) + 0.0)
    false_rate = null["power"].mean()
    caveat = (" The pre-specified percentile bootstrap CI is therefore anti-conservative at "
              "this sample size (more false detections than nominal), so the thresholds "
              "above are, if anything, optimistic." if false_rate > 0.025 else "")
    lines += [f"**Validity checks.** At a true dAUROC of 0 the CI excluded 0 in "
              f"{100 * false_rate:.1f}% of studies on average (max "
              f"{100 * null['power'].max():.1f}% in any cell; nominal one-sided 2.5%).{caveat} "
              f"Mean estimate minus true dAUROC ranged {bias['b'].min():+.3f} to "
              f"{bias['b'].max():+.3f}, i.e. the point estimate is unbiased.", ""]

    pm = s[(s["cohort"] == prim["cohort"]) & (s["prevalence"] == prim["measured_prevalence"])
           & (s["a0"] == 0.75)]
    width = pm[pm["delta"] == 0.02].set_index("rho")["ci_width"]
    lines += ["**Precision of the primary estimate** (n = 671, measured prevalence, A0 = 0.75, "
              "true dAUROC = 0.02): mean 95% CI width " + ", ".join(
                  f"{width[r]:.3f} at rho = {r:g}" for r in rhos) + ".", ""]

    pmde = mid[(mid["cohort"] == prim["cohort"])
               & (mid["prevalence"] == prim["measured_prevalence"])].set_index("rho")["mde"]
    finite = pmde.dropna()
    inside = finite[finite <= 0.03]
    if inside.empty:
        plausible = ("A dAUROC of 0.01-0.03, the size D1.4 calls plausible, is below every "
                     "one of these thresholds.")
    else:
        below_all = [v for v in (0.02, 0.01) if v < finite.min()]
        plausible = ("Only at rho = {} does the threshold reach the 0.01-0.03 range D1.4 calls "
                     "plausible{}.".format(
                         " / ".join(f"{r:g} ({v:.3f})" for r, v in inside.items()),
                         f"; a dAUROC of {below_all[0]:.2f} or less is below every threshold"
                         if below_all else ""))
    lines += ["**Statement.** Under assumptions X, with n = 671 stays ({:,} patients) at "
              "medication-recording sites, measured prevalence {:.1%} and a clinical-only "
              "AUROC of 0.75, this design detects dAUROC >= {} with 80% probability "
              "(rho = {}). {}".format(
                  int(prim["patients"]), prim["measured_prevalence"],
                  " / ".join(y(pmde[r]) for r in rhos), " / ".join(f"{r:g}" for r in rhos),
                  plausible)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Power simulation for the paired dAUROC.")
    p.add_argument("--labels", type=Path, default=Path("results") / "labels_A.csv")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--sims", type=int, default=N_SIMS)
    p.add_argument("--report-only", action="store_true",
                   help="rebuild results/power_simulation.md from saved outputs")
    args = p.parse_args(argv)

    if args.report_only:
        md = report(args.out)
        (args.out / "power_simulation.md").write_text(md, encoding="utf-8")
        print(md)
        return 0

    cohorts = load_cohorts(args.labels)
    tasks, meta = [], []
    for name, c in cohorts.items():
        tau2, diag = calibrate_tau2(c["y"], c["pat"])
        measured = round(float(c["y"].mean()), 4)
        meta.append({"cohort": name, "n": len(c["y"]), "patients": c["n_pat"],
                     "events": int(c["y"].sum()), "measured_prevalence": measured,
                     "tau2": tau2, **diag})
        for prev in (measured, *PREVALENCES):
            tasks += [(name, prev, tau2, j) for j in range(args.sims)]

    prevs = sorted({t[1] for t in tasks})
    sigmas = {(round(a + d, 4), pv): sigma_for(round(a + d, 4), pv)
              for pv in prevs for a in A0_GRID for d in DELTA_GRID}
    workers = max(1, (os.cpu_count() or 2) - 2)
    with Pool(workers, initializer=_init, initargs=(cohorts, sigmas)) as pool:
        raw = pd.DataFrame([r for rows in pool.imap_unordered(one_study, tasks, chunksize=4)
                            for r in rows])
    summary = summarise(raw)
    m = (summary.groupby(["cohort", "prevalence", "a0", "rho"]).apply(mde, include_groups=False)
         .rename("mde").reset_index())
    args.out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.out / "power_simulation_raw.csv.gz", index=False)
    summary.to_csv(args.out / "power_simulation.csv", index=False)
    m.to_csv(args.out / "power_mde.csv", index=False)
    pd.DataFrame(meta).to_csv(args.out / "power_cohorts.csv", index=False)
    md = report(args.out)
    (args.out / "power_simulation.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
