# PolySafe — HANDOFF

Medication-aware acute kidney injury (AKI) prediction on eICU-CRD Demo v2.0.1.
Read `DATA_REPORT.md` first: it has the measured counts this spec is built on.

> **`PREREGISTRATION.md` is the analysis plan of record** (committed alone as
> `291760d`, 2026-10-04, before any model was trained). Where this file and the
> pre-registration differ, the pre-registration wins. It may only change through
> its append-only Deviations section.

**Analyst and drug-classification reviewer: Dr. Varad Vaidya (PharmD).** He
built the project, adjudicated the HICL mapping (171 review-packet keys,
2026-10-06) and made every decision the pre-registration attributes to "the
PI" (D9.1). He decides clinical definitions, drug classifications and final claims.

## State as of 2026-10-10 (end of day)

Rebuilt from the full git history, PREREGISTRATION D1-D9, and the complete
Claude Code transcript of 2026-10-04 to 2026-10-09 (session `efe2395d`, 4,686
records, recovered from `C:\Users\drvnv\.claude\projects`). The planning
chat that drafted the instructions (claude.ai) is not recoverable; every
instruction it produced was pasted into that transcript verbatim, and every
decision it produced is in the Deviations.

### How the work has been run
- A planner (Claude chat) drafted each change; the files were staged in
  `.git/claude-*/` and an executor (Claude Code in AntiGravity) applied them
  with a fixed checklist: STOP if anything is staged; copy the prepared file;
  check the staged blob hash and `git diff --cached --stat`; verify every
  quote an entry cites against the committed text; commit with the given
  message; run `pytest` and `ruff check .`; **push only if both are green**;
  delete the staging folder; report.
- **Forbidden, always:** amend, rebase, reset, cherry-pick, revert,
  force-push, `git add -A`. Pre-registration entries are never edited after
  drafting; a wrong entry gets a correction entry.
- From 2026-10-10: Claude edits files directly and hands commands to a job
  runner (`.jobs/runner.ps1`, started once in AntiGravity's terminal; `.jobs/`
  is git-excluded). Same checklist and rules.
- Remote: `origin` = github.com/drvnv2000-lab/polysafe (private), branch `main`.

### Environment
- Python 3.12.10 venv **shared with the ADMET project**: `..\.venv` (i.e.
  `V:\1. Work\Project 1\.venv`). lightgbm 4.7.0, scikit-learn 1.9.1,
  numpy 1.26.4, pandas 2.3.3.
- Model runs: `python -m polysafe.models --workers 6` with
  `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. ~9 min. The
  laptop has only ~2-3 GB of free commit memory; do not raise workers.
- Test suite: 224 tests (pytest + ruff green at `0241278`). CI
  (`.github/workflows`) runs only the data-free append-only and citation tests.
- Guard tests that bind every change: `tests/test_preregistration.py`
  (every commit since BASELINE `9635d73` only appends to PREREGISTRATION.md)
  and `tests/test_citations.py` (every quote of protocol text in committed
  files must exist verbatim).

### Primary result (`results/models.md`, `a713fd2`, 2026-10-09 15:24)
Primary cohort: 671 stays, 78 AKI, medication-recording hospitals, baseline A,
EPV 4.6 (D4.4). Repeated patient-grouped CV (5 folds x 10 repeats, states
0-9); each stay's 10 out-of-fold predictions averaged; patient-level BCa,
10,000 resamples (D2.1a).

| Spec | AUROC clinical | AUROC +medication | Paired dAUROC (95% BCa) |
|---|---|---|---|
| Reduced L2 LR (**primary**, D7.2) | 0.641 | 0.677 | **+0.036 (0.006 to 0.071)** |
| LightGBM (secondary) | 0.657 | 0.659 | +0.002 (-0.009 to 0.015) |
| Full penalised LR (sensitivity) | 0.624 | 0.640 | +0.016 (0.001 to 0.034) |

80%-power threshold (section 9, D2.1a): dAUROC 0.031. Calibration slopes
(clinical / +med): LR 1.05 / 1.12, LightGBM 0.81 / 0.81. Medication features
(D7.2): vancomycin, piperacillin-tazobactam, furosemide, NSAIDs (aspirin
excluded), iodinated contrast; HICL + gtc arm, primary variant.
**The LR-vs-LightGBM disagreement is the first thing a reviewer will raise;
the README reports both.**

### Done on 2026-10-10
- D9.1 (attribution), D9.2 (D8.1 was finalised after the primary result was seen).
- Step 7: D8.1 sensitivity analyses (`results/sensitivity.md`); the primary arm
  reproduces +0.036 exactly. Figures in `results/figures/` (`polysafe.figures`).
- D10.1 exploratory record-process check (`results/exploratory_d10.md`): E1 -0.001,
  E2 +0.035; marker present in 89% of stays, so the check is weak.
- Step 8: D11.1 secondary analysis (`results/secondary.md`).
- Step 9: Research Lab sections, FastAPI (`polysafe.app`), Dockerfile, CI lint + image
  check, one-command `polysafe.pipeline`, README with every number tested
  (`tests/test_readme.py`).
- Two independent ChatGPT reviews; their edits adopted in the README.

### Open, not blocking
- D12.1 final model and calibration CIs done (`results/final_model.md`). CI green on all
  three jobs (protocol, lint, docker build + serve); Docker Desktop is not running on the
  laptop, so the image is verified in CI only.
- Failure Injection (DEMO_SCOPE) not built; its section stays hidden by design.

## The research question

> Among ICU stays at hospitals that record medication administration, does adding
> medication exposure to a clinical AKI prediction model improve discrimination
> for AKI occurring after hour 24?

Everything serves that comparison. A model that only predicts AKI well is not
the deliverable; the **paired clinical-only vs clinical+medication** difference
is. Drug-specific associations are a strictly secondary, non-causal analysis
(pre-registration section 10).

## Non-negotiables

1. **Never fabricate a result.** Every number in the README comes from a run
   that happened. If it has not run, the README says "not run yet".
2. **No leakage.** Features come only from data with offset <= T. This is
   the single easiest way to ruin this project, and the easiest thing for a
   reviewer to catch.
3. **Fully automated.** One command: load -> label -> features -> train ->
   evaluate -> write results.
4. Simplest implementation that proves the point. No overengineering.
5. **No model is trained before the power simulation (step 2c) is appended to
   PREREGISTRATION.md section 9 and committed.**

## Data (already in `data/`, SHA-256 verified)

`patient, lab, medication, infusiondrug, admissiondrug, diagnosis,
apachePatientResult, treatment` plus `intakeOutput, vitalAperiodic, pastHistory,
apacheApsVar, apachePredVar, admissionDx`.

`hospital.csv.gz` is **empty** in the demo — use `hospitalid` from
`patient.csv.gz` for any hospital-level grouping. Do not download
`vitalPeriodic` or `nurseCharting`; v1 does not need them.

## Temporal design and outcome — decided by the PI, do not change

```
ICU admission ──► baseline window ──► T = 24 h ──► outcome window ──► end
   offset 0        (<= T, features)   prediction   (T, T+7d], censored at
                                                   ICU discharge or death
```

- **T = 24 h** after ICU admission. Features: only data with offset <= T.
- **Outcome window: (T, T + 7 days], censored at ICU discharge or death**
  (`unitdischargeoffset`). Creatinine drawn after ICU discharge does not count.
  Report the distribution of post-T follow-up time. Sensitivity analysis:
  restrict to >= 48 h of post-T follow-up.
- **AKI = KDIGO stage >= 1**, met by either:
  1. **Creatinine:** a rise of >= 0.3 mg/dL within any 48 h window, or >= 1.5x
     baseline within 7 days. Serum `creatinine` only — never `urinary
     creatinine`. Floating-point tolerance on the 0.3 threshold.
  2. **Acute RRT** (dialysis / CVVH(D)) **initiated after T and within the
     outcome window** counts as AKI stage 3, whether or not the creatinine
     criterion is observed.
- **Baseline A (primary):** first serum creatinine in the first 24 h (2,092
  stays). **Baseline C (sensitivity):** most recent creatinine before ICU
  admission, `labresultoffset < 0` (1,687 stays). Report both; if they disagree
  materially, say so. **"Lowest value during the stay" is prohibited.**

## Cohort exclusions (pre-registration section 5)

| Exclusion | Rationale |
|---|---|
| Stay < 24 h | No prediction time exists |
| Chronic dialysis / ESRD | AKI is undefinable. Chronic markers only: past history of hemo/peritoneal dialysis, diagnosis ESRD or CKD stage 5, treatment "hemodialysis for chronic renal failure", APACHE chronic-dialysis flag. Acute CVVH/HD is an *outcome*, not an exclusion |
| AKI already met by T | Prevalent, not incident. Pre-ICU creatinine draws count toward this |
| Acute RRT initiated at or before T | Prevalent KDIGO stage 3 (3 stays: 826668, 1770612, 2860223) |

Report the cohort flow — starting stays -> each exclusion -> final n, with the
event count at each step — as `results/cohort_flow.md`.

## Medication analysis — site-level restriction

- **The primary analysis is restricted at the SITE level, not the patient
  level**: only hospitals that record medication administration. 42 of 186
  hospitals record no medications for any stay (521 stays); 95 record them for
  every stay; 49 are mixed. ~79% of stays without medication rows come from the
  42 non-recording hospitals, so missingness is a site characteristic. State the
  estimand as *"among ICU stays at hospitals with medication capture"*.
- **`medication_data_available` must NOT be used as a predictor.** It is a
  hospital proxy: it would let the model learn site-level AKI differences.
- Within included sites, genuinely missing values stay missing (LightGBM handles
  NaN). Absence of a record is never coded as "unexposed".
- Sensitivity: the full cohort with missingness preserved.
- The clinical-only and clinical+medication models are always compared on the
  **same** stays.

## Validation design

- **Grouping unit is the patient (`uniquepid`) — MANDATORY.** 2,520 stays map
  to 1,841 patients; 416 patients have more than one stay; 1,095 stays (43%)
  belong to a repeat patient, and one patient has 24 stays. No patient may
  appear in more than one fold. `patienthealthsystemstayid` is nested within
  patient (verified), so it is protected too.
- **Repeated stratified grouped cross-validation** over the full analysis cohort
  (`StratifiedGroupKFold`, groups = `uniquepid`, stratified on the AKI label,
  repeated over several seeds). **No single train/test split** — the earlier
  70/30 plan is dropped: at this sample size one split is noise.
- **Hospital generalization is a secondary robustness analysis only**: grouped
  CV over clusters of hospitals. A single held-out hospital is impossible (the
  largest has 40 stays, none has 50+). 19 patients have stays at more than one
  hospital, so hospitals sharing a patient must be merged into one cluster.
- Report AUROC and AUPRC with bootstrap CIs, calibration curve and
  slope/intercept, and decision-curve analysis. Never a bare AUROC.

## Build order

| Step | Deliverable |
|---|---|
| 1 | **Done.** Loaders + cohort builder. Tests on real rows. |
| 2a | **Done.** Cohort with the RRT-at-T exclusion; report the flow. |
| 2b | **Done** (`results/cohort_flow.md`, `results/aki_labels.md`). AKI labels for baselines A and C under the pre-registered outcome (7-day window censored at ICU discharge, RRT rule). Incidence for both, post-T follow-up distribution, comparison with DATA_REPORT's provisional 22.6%. |
| 2c | **Done** (PREREGISTRATION 9.1, deviations D2-D3). **Power simulation** (pre-registration section 9) on the real cohort, real patient grouping and real fold structure. Append the result to PREREGISTRATION.md section 9 and commit — **before any model is trained.** |
| 3 | **Done.** HICL draft (`mapping/`); PI adjudication of the 171 review-packet keys (2026-10-06); brand alias list (D5.14b); review-corrected arm (D6.3). |
| 4 | **Done.** Clinical features (74; APACHE excluded per D3.1). Medication exposure by `polysafe.ascertain`: five ascertainment arms (D6.2-D6.7), `results/exposure_prevalence.csv`. |
| 5 | **Done** (2026-10-09). Clinical-only reduced L2 LR (D7.2), LightGBM, full penalised LR; patient-level BCa (D2.1a) and its coverage re-run. |
| 6 | **Done** (2026-10-09, `results/models.md`). Paired dAUROC above. **Still open:** re-point `tests/test_review.py::test_end_to_end_decision_to_stay_level_exposure` at `polysafe.ascertain` so a builder that bypasses `review.rows_counted` fails it. |
| 7 | **Specified, not run** (D8.1). Calibration and decision-curve data exist for the section 8 models; sensitivity analyses do not. |
| 8 | Secondary: pre-specified agents (vancomycin, piperacillin-tazobactam, furosemide; ondansetron as negative control), propensity-adjusted, associational language only. |
| 9 | FastAPI + Docker + README with real numbers. |

## Drug normalization (step 3) — the PI's work, not yours

`medication.csv.gz` has 75,604 rows. See DATA_REPORT.md for the measured detail.

**Normalize on `drughiclseqno` (HICL), not `drugname`.** There are 1,027 HICL
codes against 1,401 raw name strings, and 40.6% of rows have no `drugname` at
all — but 94.6% of those still carry a HICL, and 69.7% carry a HICL that appears
elsewhere *with* a name. Use `drugname` as the human-readable label and `gtc` as
the fallback for the remaining ~5%.

**Brand names are separate strings.** Raw generic-name matching undercounts
badly: furosemide 442 -> 558 once Lasix is included, piperacillin 85 -> 163 with
Zosyn. Every raw count is a lower bound until HICL normalization is done.

Your job: produce a frequency-ranked list of HICL codes with their observed
names and counts, plus a first-pass automated mapping to ingredients. **The PI
reviews and corrects it**, and that reviewed mapping is committed as the
project's own artifact. Never quietly accept a fuzzy match as correct.

## The bar

This is a portfolio project competing against a candidate whose strongest repos
have real results but **no external validation and no public benchmark anchor**.
PolySafe wins on: real multi-centre data, a pre-registered and leakage-controlled
design, patient-grouped validation, honest confidence intervals, a sensitivity
analysis across two baseline definitions, and pharmacist-authored drug
classification. It must also match on tests, CI, Docker and a live demo.

Nothing here is impressive if the leakage controls are sloppy.

## Required visual deliverables

A clinical study is judged in thirty seconds by two pictures. Both are
mandatory, and both are built from real output, never hand-drawn.

**1. The cohort flow diagram (CONSORT-style).** Starting stays, each exclusion
as a labelled branch with its count and reason, the final analysis cohort, and
the event count. Generated from `results/cohort_flow.md` so it can never drift
from the code. This is the most recognisable artifact in clinical research:
any clinician reading the README knows from this one image whether the study
design is sound.

**2. The calibration curve**, with the calibration slope and intercept stated
on the plot, for both the clinical-only and clinical+medication models.
Discrimination alone (AUROC) says nothing about whether predicted risks are
trustworthy; calibration is the first thing a clinical reviewer asks for and
almost no portfolio project shows it.

Supporting figure: the paired ΔAUROC across CV folds with its confidence
interval — one plot that answers the primary question.

README order: cohort flow, then the primary result, then calibration. The
reader should reach the headline number without scrolling.
