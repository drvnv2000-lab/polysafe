# PolySafe — Pre-registration

**Status: committed before any model is trained.** The git commit date of this
file is the claim. Nothing in it may be changed after modelling begins except
through the Deviations section at the end, which is append-only.

PI: Vinay (Pharm D). All clinical definitions below are his decisions.

---

## 1. Research questions

**Primary (predictive).**
Among ICU stays at hospitals that record medication administration, does adding
medication exposure to a clinical prediction model improve discrimination for
acute kidney injury occurring after hour 24?

**Secondary (associational, explicitly not causal).**
For a small pre-specified set of nephrotoxic agents, what is the adjusted
association between exposure before hour 24 and subsequent AKI?

No causal language will be used for the secondary analysis in any output.

## 2. Data

eICU Collaborative Research Database Demo v2.0.1 (PhysioNet, ODbL). All 14
tables SHA-256 verified against the distribution's own `SHA256SUMS.txt`.
No credentialed data is used. See `DATA_REPORT.md` for measured counts.

## 3. Temporal design

```
ICU admission ──► baseline window ──► T = 24 h ──► outcome window ──► end
   offset 0        (≤ T, features)    prediction    (T, T+7d] or ICU
                                       made here     discharge, whichever
                                                     comes first
```

- **Prediction time T = 24 h** after ICU admission. Verified optimal against
  T = 6 h (723 usable stays) and T = 12 h (1,091); T = 24 h yields 1,238 because
  shorter horizons strip the baseline creatinine draw.
- **Features: only data with offset ≤ T.** No exception.
- **Outcome window: (T, T + 7 days], censored at ICU discharge or death.**
  The distribution of post-T follow-up time will be reported. Stays are not
  excluded for short follow-up in the primary analysis; a sensitivity analysis
  restricts to ≥ 48 h of post-T follow-up.

## 4. Outcome definition

**AKI = KDIGO stage ≥ 1**, met by either:

1. **Creatinine:** a rise of ≥ 0.3 mg/dL within any 48 h window, or ≥ 1.5 ×
   baseline within 7 days. Serum `creatinine` only; `urinary creatinine` is a
   different analyte and is excluded everywhere.
2. **Renal replacement therapy:** initiation of *acute* RRT after T and within
   the outcome window counts as AKI (KDIGO stage 3), regardless of whether the
   creatinine criterion is subsequently observed.

Floating-point tolerance is applied to the 0.3 mg/dL threshold (2.3 − 2.0
evaluates below 0.3 in IEEE-754 and would silently drop true events).

### Baselines

- **Baseline A (primary):** first serum creatinine in the first 24 h. Available
  for 2,092 stays.
- **Baseline C (sensitivity):** most recent creatinine drawn before ICU
  admission (`labresultoffset < 0`). Available for 1,687 stays.
- **"Lowest value during stay" is prohibited** as a baseline: it peeks at
  post-T data and inflates incidence to ~45% against a real-world 20–30%.

Both baselines are reported. If they disagree materially, both are stated.

## 5. Cohort and exclusions

Starting population: all 2,520 ICU unit stays.

| Exclusion | Rationale |
|---|---|
| Stay < 24 h | No prediction time exists |
| Chronic dialysis / ESRD | AKI is undefinable. Chronic markers only — acute CVVH/HD is an *outcome*, not an exclusion |
| AKI already met by T | Prevalent, not incident. Pre-ICU creatinine draws count toward this determination |
| Acute RRT initiated at or before T | Prevalent KDIGO stage 3 |

Measured flow (baseline A, before the RRT-at-T rule): 2,520 → −893 → 1,627 →
−62 → 1,565 → −168 → **1,397**. The full flow with event counts at each step
will be published as `results/cohort_flow.md`.

## 6. Medication exposure and the estimand

**Measured problem.** 40.6% of medication rows have no `drugname`; 26% of stays
have no medication record at all. Critically, **missingness is a site
characteristic, not a patient one**: 42 hospitals record no medications for any
stay (521 stays), 95 record them for every stay, 49 are mixed. ~79% of missing
stays come from non-recording hospitals.

**Therefore:**

- The **primary analysis is restricted at the site level** to hospitals that
  record medication administration. The estimand is stated explicitly: *among
  ICU stays at hospitals with medication capture*. This is a site-selected, not
  patient-selected, population and will be described as such.
- **`medication_data_available` will NOT be used as a predictor.** It is a
  hospital proxy; including it would let the model learn site-level AKI
  differences and would align almost perfectly with hospital-grouped folds.
- Within included sites, genuinely missing values are **preserved as missing**
  (LightGBM handles NaN natively). Absence of a record is never coded as
  "unexposed".
- **Sensitivity:** the full cohort with missingness preserved, to show whether
  the conclusion moves.

**Normalization.** Exposure is derived from `drughiclseqno` (HICL), not
`drugname`: 1,027 HICL codes vs 1,401 raw strings, and 94.6% of nameless rows
carry a HICL (69.7% a HICL seen elsewhere with a name). `gtc` is the fallback.
The HICL→ingredient mapping is reviewed and corrected by the PI and committed
as a project artifact.

**Ascertainment sensitivity analysis.** The primary analysis is repeated under
four exposure-ascertainment strategies — generic-string only, generic + brand,
HICL-based, HICL + gtc — and the movement in the conclusion is reported. Raw
string matching is known to undercount badly (furosemide 442 → 558 once Lasix is
included; piperacillin 85 → 163 with Zosyn).

## 7. Validation design

- **Grouping unit is the patient (`uniquepid`), mandatory.** 2,520 stays map to
  1,841 patients; 416 patients have more than one stay; 1,095 stays (43%) belong
  to a repeat patient, and one patient contributes 24 stays. No patient may
  appear in more than one fold. `patienthealthsystemstayid` is nested within
  patient and is therefore also protected.
- **Repeated stratified grouped cross-validation** over the full analysis
  cohort. No single train/test split: at this sample size one split is noise.
- **Hospital generalization is a separate, secondary robustness analysis**
  using grouped CV over clusters of hospitals. A single held-out hospital is
  impossible — the largest has 40 stays and none has 50+.

## 8. Models and metrics

- **Clinical-only model** (pre-specified covariates: demographics, APACHE
  severity, baseline creatinine, labs ≤ T, vitals ≤ T, urine output ≤ T,
  comorbidities from `pastHistory`).
- **Clinical + medication model** — identical except for medication features.

Both: logistic regression first, then gradient boosting. The headline is the
**paired difference** between them, not either model's absolute score.

Reported for every model: **AUROC and AUPRC with bootstrap confidence
intervals, calibration curve and calibration slope/intercept, and decision-curve
analysis.** Discrimination alone is not sufficient for a clinical model.

## 9. Power — to be established by simulation before modelling

An analytic approximation (Hanley–McNeil + DeLong, n = 1,037, 22–25% prevalence)
suggested a minimum detectable paired ΔAUROC of 0.018–0.035 depending on the
correlation between nested models. **This is treated as a planning estimate
only and is explicitly not relied upon**, because it assumes independent
observations, which patient-level grouping with 43% repeat stays violates. The
true effective sample size is lower and the real detectable effect is therefore
larger.

**Before any model is trained**, a simulation will be run using the actual
cohort: real prevalence, real patient grouping, the proposed fold structure, a
range of plausible clinical/medication prediction correlations, and a range of
true ΔAUROC values. The resulting statement — "under assumptions X, this design
detects ΔAUROC ≥ Y with Z% probability" — is committed here as an appendix
before results are examined.

**What counts as a null.** If the observed ΔAUROC is smaller than the simulated
detectable effect, the conclusion is *"this design could not detect a difference
of this size"*, **not** *"medication information does not help"*. That
distinction will be stated in the abstract of the README.

## 10. Secondary epidemiological analysis

Strictly secondary. Pre-specified here, before any outcome is inspected.

- **Exposures (fixed now, not chosen by looking at results):** vancomycin,
  piperacillin-tazobactam, furosemide. Selected on prior clinical plausibility
  and measured exposure counts only.
- **Negative control exposure:** ondansetron — commonly administered, no
  plausible renal mechanism. If it shows an association of similar magnitude,
  the adjustment is inadequate and that will be reported as such.
- **Method:** propensity-score adjustment for baseline illness severity, with
  overlap/positivity diagnostics reported. Adjusted odds ratios with 95% CIs.
- **Excluded from drug-specific reporting:** any agent below the exposure
  threshold needed for a stable estimate (ketorolac, ibuprofen, losartan at
  current counts). These are pooled into classes or not reported individually.
- Output language is associational throughout. The most likely honest finding is
  attenuation of a crude association after adjustment for severity; that is
  reported as the result, not buried.

## 11. Deviations

Any departure from this plan is appended below with the date, the reason, and
whether it was decided before or after seeing results. An empty section at
publication means none occurred.

### D1 — 2026-10-04, before any model was trained

**D1.1 — Section 3, "1,238 usable stays": definition was ambiguous, not wrong.**
The figure reproduces exactly under this definition, which is now stated:
ICU LOS >= 24 h **and** a serum creatinine in [0, T] **and** any serum creatinine
after T — computed *before* the ESRD and AKI-by-T exclusions and *without* the
7-day / ICU-discharge censoring. For comparison, applying the censored outcome
window gives 1,124 with a post-T value and 1,056 meeting both conditions.

**D1.2 — Section 9, prevalence 22-25%: this was wrong.** It was carried over
from the provisional DATA_REPORT estimate and general ICU literature rather than
measured. Measured incidence under the project's own labelling is **16.1%**
(baseline A, uncensored) and **12.9%** with ICU-discharge censoring. All power
statements below use the measured values. The earlier figure made the design
look better powered than it is.

**D1.3 — Analysis cohort is smaller than section 9 assumed.** Requiring both an
observable baseline and an observable outcome leaves **868 stays**, of which
**671** are at medication-recording sites (the primary estimand). Section 9
assumed ~1,037.

**D1.4 — Consequence: the primary endpoint is re-specified.** Recomputed with
the measured prevalence and cohort size, the minimum detectable paired dAUROC
is:

| cohort | prevalence | r=0.75 | r=0.90 | r=0.95 |
|---|---|---|---|---|
| n=671 (primary) | 12.9% | 0.063 | 0.040 | 0.028 |
| n=671 (primary) | 16.1% | 0.056 | 0.036 | 0.025 |
| n=868 (all sites) | 12.9% | 0.055 | 0.035 | 0.025 |

Adding medication features to an ICU AKI model plausibly moves AUROC by
0.01-0.03. **This design cannot reliably detect an effect of that size.**
Rather than run a test that is predestined to be uninformative and then
rationalise the result, the endpoint structure is changed here, before any
model is trained:

- **Primary becomes an estimation, not a test.** Report dAUROC with a
  patient-level bootstrap 95% CI. The pre-specified conclusion is the interval
  itself. A CI containing 0 is reported as "this design cannot resolve an effect
  of this magnitude", never as evidence of no effect.
- **Co-primary: the exposure-ascertainment sensitivity analysis** (section 6).
  Comparing four ascertainment strategies on the same patients is a
  within-sample methods comparison and does not suffer the small-effect power
  problem. It is also the component that depends on pharmacological knowledge.
- **Drug-specific associations (section 10) remain secondary but are the
  best-powered substantive question**: minimum detectable OR ~1.6-2.0 against
  published nephrotoxin associations of roughly 2-3.

This deviation was decided from power arithmetic alone. No outcome data,
model output or association estimate had been examined.
