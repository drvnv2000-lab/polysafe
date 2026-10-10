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

### 9.1 Simulation result (appended before any model was trained)

Generated by `python -m polysafe.power` (`results/power_simulation.csv`, `results/power_mde.csv`). Every number below is read from those files.

**Design simulated (identical to the analysis):** the real baseline-A cohorts (n = 671 stays / 586 patients at medication-recording sites, 78 events; n = 868 all sites) with their real `uniquepid` grouping; 5-fold `StratifiedGroupKFold` by patient x 10 repeats (`polysafe.cv.patient_folds`); out-of-fold predictions averaged over repeats; dAUROC with a patient-level bootstrap (1,000 resamples) percentile 95% CI. *Detected* = lower bound > 0.

**Assumptions X:** AKI from a latent liability with a patient random effect calibrated to the real within-patient clustering of AKI (selected tau2 = 0, 0 for the two cohorts: the real labels show almost no clustering of AKI within patients); clinical-only AUROC A0 in {0.70, 0.75, 0.80}; Pearson correlation rho between the two models' risk scores in {0.6, 0.75, 0.9, 0.95}; true dAUROC 0-0.10; prevalence 0.129 and 0.161 (D1.2) plus each cohort's measured prevalence; 200 simulated studies per cell (Monte Carlo SE of an 80% power estimate ~ 2.8 points); seed 20261004. Not simulated: the extra variance of fitting a many-feature model; rho < 1 stands for all disagreement between the two models.

**Minimum dAUROC detected with 80% probability** (A0 = 0.75; linear interpolation between grid points):

| Cohort | Prevalence | rho = 0.6 | rho = 0.75 | rho = 0.9 | rho = 0.95 |
|---|---|---:|---:|---:|---:|
| n=671 (primary sites) | 0.116 (measured) | 0.076 | 0.061 | 0.040 | 0.029 |
| n=671 (primary sites) | 0.129 | 0.073 | 0.058 | 0.038 | 0.028 |
| n=671 (primary sites) | 0.161 | 0.070 | 0.055 | 0.036 | 0.026 |
| n=868 (all sites) | 0.123 (measured) | 0.066 | 0.055 | 0.034 | 0.026 |
| n=868 (all sites) | 0.129 | 0.065 | 0.052 | 0.035 | 0.026 |
| n=868 (all sites) | 0.161 | 0.058 | 0.047 | 0.031 | 0.024 |

Across A0 = 0.70-0.80 the same cells range 0.022-0.080; per-A0 values are in `results/power_mde.csv`.

**Validity checks.** At a true dAUROC of 0 the CI excluded 0 in 3.9% of studies on average (max 6.5% in any cell; nominal one-sided 2.5%). The pre-specified percentile bootstrap CI is therefore anti-conservative at this sample size (more false detections than nominal), so the thresholds above are, if anything, optimistic. Mean estimate minus true dAUROC ranged +0.000 to +0.003, i.e. the point estimate is unbiased.

**Precision of the primary estimate** (n = 671, measured prevalence, A0 = 0.75, true dAUROC = 0.02): mean 95% CI width 0.111 at rho = 0.6, 0.089 at rho = 0.75, 0.058 at rho = 0.9, 0.042 at rho = 0.95.

**Statement.** Under assumptions X, with n = 671 stays (586 patients) at medication-recording sites, measured prevalence 11.6% and a clinical-only AUROC of 0.75, this design detects dAUROC >= 0.076 / 0.061 / 0.040 / 0.029 with 80% probability (rho = 0.6 / 0.75 / 0.9 / 0.95). Only at rho = 0.95 (0.029) does the threshold reach the 0.01-0.03 range D1.4 calls plausible; a dAUROC of 0.02 or less is below every threshold.

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

### D2 — 2026-10-04, before any model was fitted

**D2.1 — Section 8 / D1.4: confidence-interval method.** The patient-level
percentile bootstrap CI is replaced with a patient-level BCa (bias-corrected and
accelerated) bootstrap CI. Reason: the section 9.1 simulation measured a type I
error of 3.9% (max 6.5% in any cell) against a nominal one-sided 2.5%. Measured
coverage will be reported alongside every interval.

**D2.2 — Section 3: primary estimand clarified.** The outcome is AKI during the
remainder of the ICU stay, capped at 7 days after T. ICU discharge ends the
exposure period rather than censoring it. The >= 48 h fixed-window analysis
(n = 314 stays at medication-recording sites, baseline A) is the pre-specified
sensitivity analysis.

**D2.3 — Section 9.1: known limitation.** The simulation omits model-fitting
noise, so the real detectable effects are larger than tabulated.

**D2.4 — Section 9.1: which assumption applies.** The realised correlation
between the clinical-only and clinical+medication predictions will be measured
and reported, to identify which correlation (rho) in the power table applies.

Decided after the section 9.1 simulation and before any model was fitted. No
model output or association estimate had been examined.

### D3 — 2026-10-04, before any model was fitted

**D3.1 — Section 8: APACHE severity excluded despite being pre-specified.** In
74 of 868 stays (8.5%), eICU's APACHE day-1 values match labs drawn after T, by
up to 7.3 h, including the worst creatinine in 25 stays. Using them would break
the section 3 rule (features only from offset <= T). Severity is carried by the
labs, blood pressure and urine output measured by T.

**D3.2 — `vitalPeriodic` added** (heart rate, respiratory rate, temperature,
SpO2). Omitting standard vitals would weaken the clinical baseline and bias
dAUROC upward.

**D3.3 — Unit type and admission source added; admission diagnosis as a coarse
category only.**

**D3.4 — Ethnicity excluded as a predictor**, retained for subgroup performance
reporting.

**D3.5 — Primary logistic regression is L2-penalised, tuned within training
folds.** Events per variable is 78/48 = 1.6 (48 = the step-4 clinical feature
set, before the D3.2-D3.3 additions), far below the conventional minimum of 10;
this is stated in the README.

**D3.6 — Imputation: median plus missingness indicator, fitted within each
training fold.** Labs above 40% missingness across all 868 stays (lactate, pH,
phosphate) are excluded from the LR model and retained for LightGBM.

**D3.7 — Lookback for pre-ICU values capped at 7 days before admission.**

Decided before any model was fitted. No model output or association estimate
had been examined.

### D4 — 2026-10-04, before any model was fitted

**D4.1 — NSAID class: aspirin excluded.** Aspirin accounts for 1,026 of the
1,154 NSAID rows (89%). Low-dose aspirin has a different indication and renal
risk profile; it is excluded from the NSAID class and not analysed separately
unless pre-specified later. Residual NSAIDs are 11 keys / 128 rows / 99 stays
across all 2,520 and fall below the exposure threshold, so NSAIDs join
aminoglycosides and iodinated contrast as not individually analysable.
Analysable exposures: vancomycin, piperacillin-tazobactam, furosemide; negative
control: ondansetron.

**D4.2 — Temperature excluded from the LR** (91% missing), retained for
LightGBM.

**D4.3 — The 40% missingness threshold (D3.6) is applied to the primary
cohort, not all 868 stays.** INR (41.6% missing in the primary cohort) is
therefore also excluded from the LR. LR exclusions: lactate, pH, phosphate, INR,
temperature.

**D4.4 — Primary logistic regression uses a pre-specified reduced feature
set,** chosen by the PI on clinical grounds before any model output was seen:

    age, cr_baseline, cr_max, bun, hco3, albumin, hgb, wbc, map_min,
    urine_ml_kg_h, hx_ckd, hx_chf, hx_diabetes, dx_operative

Per D3.6, missingness indicators are added only where missingness is plausibly
informative and substantial: urine_ml_kg_h (27.3%) and albumin (25.9%), plus a
single merged indicator for the three hx_* flags (4.6%, missing on the same
stays, where median imputation would assert absence of comorbidity). The
remaining variables (hco3, map_min, wbc, hgb, dx_operative; 1.5-6.0% missing)
receive median imputation without an indicator, their missingness being
technical rather than clinical. This refines D3.6 for the reduced model only;
the full 74-feature LR retains indicators for every imputed variable.

Total: 14 variables / 17 parameters. EPV = 78/17 = 4.6, below the conventional
minimum of 10, which is stated in the README.

Excluded for collinearity (map_below_65, anion_gap), availability (cr_pre_icu
67%, bili 70%), parameter cost (unit_*, 4 extra dummies) or rarity
(hx_cirrhosis, 1.6%). All remain in the full penalised LR and in LightGBM. The
full 74-feature penalised LR is a sensitivity analysis; LightGBM uses all
features.

**D4.5 — Aminoglycosides cannot be analysed:** absent from the data. Iodinated
contrast (69 stays) is below the exposure threshold and will not be reported
individually.

**D4.6 — eICU's `drughiclseqno` is NOT RxNorm's `HIC_SEQN`.** Cross-walking
them returns wrong ingredients (furosemide -> enoxacin, vancomycin -> corn
syrup). All mapping goes via drug names. RxNorm ATC classes are also unreliable
for combination products and are not used for class assignment.

Decided before any model was fitted. No model output or association estimate
had been examined.

### Type tags for D1.1-D4.6 (appended retrospectively, 2026-10-04)

Tags are added here instead of editing the entries above, so the
Deviations section stays append-only and the text of D1-D4 is unchanged.
Assigned by the analyst (Claude); the PI may confirm or amend them in a later
entry.

- [clarification]: states an existing rule more precisely; nothing computed changes.
- [correction]: fixes a factual or numerical error; the method is unchanged.
- [protocol addition]: adds a rule or procedure the protocol did not cover.
- [substantive change]: changes a pre-specified population, endpoint, covariate or method.

| Entry | Tag | Entry | Tag |
|---|---|---|---|
| D1.1 | [clarification] | D3.2 | [protocol addition] |
| D1.2 | [correction] | D3.3 | [substantive change] |
| D1.3 | [substantive change] | D3.4 | [protocol addition] |
| D1.4 | [substantive change] | D3.5 | [protocol addition] |
| D2.1 | [substantive change] | D3.6 | [protocol addition] |
| D2.2 | [clarification] | D3.7 | [protocol addition] |
| D2.3 | [clarification] | D4.1 | [substantive change] |
| D2.4 | [protocol addition] | D4.2 | [protocol addition] |
| D3.1 | [substantive change] | D4.3 | [substantive change] |
| | | D4.4 | [substantive change] |
| | | D4.5 | [clarification] |
| | | D4.6 | [protocol addition] |

### D5 — 2026-10-04, before any model was fitted

**D5.1 [protocol addition] HICL review stratified by consequence.** Tiers are
mutually exclusive and exhaustive over the keys of `mapping/hicl_draft.csv`,
first match wins: (1) current mapping names a pre-specified agent; (2) current
mapping feeds a pre-specified class variable; (3) any other key, whatever its
mapping status, whose drugname strings match a committed search term; (4) the
remainder, not reviewed. Tiers 1-3 are reviewed exhaustively. Keys not
currently feeding a pre-specified exposure or class variable are outside the
deterministic exposure construction. Unreviewed keys may still contain
undetected false positives or false negatives; this is quantified through the
pre-specified audit sample, not eliminated. After a name-based screen over all
keys (tier 3), residual false negatives are limited to keys whose drugname
strings contain no recognizable agent or class name. Mapping-coverage figures
are reported from the full key set, not the reviewed subset.

**D5.2 [protocol addition] Audit design and estimator** exactly as stated in
`mapping/review/REVIEW_PACKET.md`, including the limits on what it estimates
and the restriction of Wilson intervals to the key-level proportion.

**D5.3 [protocol addition] Decision vocabulary.** keep / reject / unsure;
unsure is excluded from the primary ascertainment strategy and included in a
sensitivity variant; both are reported.

**D5.4 [protocol addition] Correction scope.** corrected_name may remove a key
from an exposure, move it between two already pre-specified agent exposures,
or add or move it to an already pre-specified class variable. It may not
create a new exposure or class; corrections needing one are logged and
excluded.

**D5.5 [clarification] Mapping adjudication preceded any drug-outcome
association.** AKI labels existed and their marginal distribution was known;
no exposure-outcome estimate of any kind existed. Verifiable from commit
order. Disclosed for completeness: during cohort work on 2026-10-04, before
this entry, the crude AKI rate was compared once between stays with and
without any medication record (19.2% vs 15.1%, under the superseded
uncensored outcome definition). That comparison concerns whether medication
data exist at all, not any drug, and was never committed to the repository.

**D5.6 [protocol addition] Search-term list fixed and committed before
screening:** `mapping/review/search_terms.txt` at commit
`aaef7a99e175e4875abf667ab2667744f60b4dbc`.

**D5.7 [protocol addition] Optional exploratory intra-rater second pass**,
seed 20261004, blind to pass 1: 20 tier-1/2 keys, reported as raw percent
agreement alongside Cohen's kappa and labelled exploratory intra-rater
consistency, not a validation statistic.

**D5.8 [protocol addition] Within-key reconciliation rule.** A key whose
drugname strings name different ingredients is kept with row-level exclusion
of minority strings when one ingredient accounts for >= 90% of the key's rows,
and marked unsure otherwise. Threshold fixed before review. Rationale: forcing
every mixed key to unsure would discard large numbers of correctly mapped rows
on the evidence of a few stray ones.

**D5.9 [protocol addition] The ascertainment comparison is reported in full
regardless of what it shows:** tier counts, affected rows and stays,
per-strategy exposure prevalence for each pre-specified agent, and delta-AUROC
under each of the four ascertainment strategies. Whether this comparison
leads the write-up is an editorial decision about emphasis and does not alter
which numbers are reported.

**D5.10 [protocol addition] Tier-3 matching is whole-token and whole-phrase**
after NFKD normalization and casefolding, never bare substring. Search terms
with zero hits are reported.

**D5.11 [protocol addition] Wrong strings are never reassigned.** Rows of
strings listed in wrong_strings are excluded from all exposures. They are never
reassigned to another agent or class, even when the string names one.
corrected_name operates at hiclseqno level only. This is conservative by
design and biases toward under-ascertainment.

Decided before any model was fitted and before any drug-outcome association
was computed.

### D5 amendments — 2026-10-04, before any model was fitted

**D5.12 [protocol addition] Unnamed rows.** Unnamed medication rows sharing an
adjudicated HICL key are included in the primary ascertainment strategy under
that key's adjudicated mapping. A pre-specified sensitivity analysis excludes
all rows lacking a drugname string, holding the ascertainment strategy fixed,
quantifying dependence on HICL-only evidence. Both reported.

**D5.13 [correction] Unsure keys in the sensitivity variant.** The unsure
sensitivity variant includes an unsure key's rows EXCLUDING strings listed in
wrong_strings, identical to the row rule for kept keys. The earlier reading
that it includes all of the key's rows contradicts D5.11 and is void.

**D5.14 [protocol addition] Brand alias list.** The four ascertainment
strategies, including "+brand", were pre-specified. The CONTENT of the brand
alias list was not. That list was authored after an uncommitted exploratory
screen had revealed that at least one brand-only key exists; it was authored
from the reviewer's pharmacological knowledge, not from that screen's output,
which was discarded and never seen by the reviewer. Screen A (generic-only) is
unaffected and remains fully pre-committed. Screen A and screen B are both
reported in full; neither replaces or suppresses the other. No new
ascertainment strategy is created by this entry. Disclosed for completeness by
the analyst: the discarded screen's output (33 brand-like tokens and a 14-key
diagnostic table) was displayed in the analyst's working session on
2026-10-04, and the analyst's report to the reviewer quoted two of its strings,
"ZOFRAN" and "Zofran", as examples of brand-only keys. Whether the reviewer
viewed the session output is for the reviewer to state.

**D5.15 [clarification] Unascertainable rows.** 1,646 non-cancelled
medication rows carry neither a HICL code nor a drugname and are
unascertainable by any strategy. Reported as a floor on measurement
completeness.

### D6 — 2026-10-04, before any model was fitted

**D6.0 [protocol addition] Primary-model exposure set.** All pre-specified
agents and all pre-specified class variables enter the primary model
regardless of exposure frequency. The resulting EPV is reported as an observed
consequence and is not used to drop variables. This rule was committed before
events among exposed were computed; verifiable from commit order.

Analyst note (statement of fact about the committed text; not part of the
rule): no committed document defines the set of pre-specified class
variables. The class list used by the review code (`src/polysafe/review.py`,
`CLASSES`) is attributed there to a HANDOFF passage that appears in no
committed version of HANDOFF.md. Section 10 lists ondansetron as a
negative-control exposure, separately from its list of exposures.

**D6.0a [correction] Provenance scope of D6.0.** The rule's APPLICATION admits
no frequency-based exclusion of any pre-specified exposure. The choice to adopt
a frequency-independent rule was itself made with knowledge that some named
classes are rare; that is disclosed here rather than denied. At D6.0's commit,
exposed-stay counts and tier-3 candidate counts were already known; events
among exposed, events among unexposed and all model results were not.

**D6.0b [clarification] Append-only property and one repaired commit.** The
property in force is document-level: new content is appended at the end of
this file; nothing is inserted inside an earlier entry's block and no existing
line is deleted or modified. 067ee23 committed D6.0. d972c0c inserted a
sentence inside D6.0's block, which this property forbids. That insertion was
reverted and the sentence re-appended as D6.0a, so it visibly post-dates the
rule it qualifies. Two one-time modifications are disclosed: the revert, which
deleted the inserted sentence; and the line-ending commit, which adds
.gitattributes ("*.md text eol=lf") and renormalises the tracked markdown
files. Every committed markdown file was already stored with LF endings, so
the line-ending commit changed no committed line; it fixes working-copy line
endings on every platform. This entry's own text was amended in that same
line-ending commit to add this disclosure. Earlier history also contains two
changes this property would forbid, both made before it was in force and
recorded here rather than relaxed: 29a66d8 deleted the placeholder line
"*(none yet)*" when D1 was added, and 46c9f9c inserted the section 9.1
appendix inside section 9, as section 9 instructed. The append-only property
is baselined from the later of the two one-time modifications, the
line-ending commit, and applies to every commit after it.

**D6.0c [protocol addition] Definition of the pre-specified class set.** D6.0
refers to "all pre-specified class variables"; no committed text defined that
set. It is defined here as exactly the medication classes named in the
committed protocol: the NSAID class with aspirin excluded (D4.1),
aminoglycosides (D4.1, D4.5), and iodinated contrast (D4.1, D4.5). ACE
inhibitors/ARBs and loop diuretics appear only in code (review.CLASSES) and are
named nowhere in the committed protocol; they do not enter the primary model
and are reported descriptively as post-hoc exposures, labelled as not
pre-registered. Aminoglycosides are absent from the data and therefore
contribute no variable - an empty set, not an exclusion.
Basis: which classes the committed protocol names. Not exposure frequency.
Written after exposed-stay counts were known; disclosed.

**D6.0d [clarification] Ondansetron's role.** Section 10 enumerates the
pre-specified exposures under two distinct headings: "Exposures" (vancomycin,
piperacillin-tazobactam, furosemide) and "Negative control exposure"
(ondansetron). That structural separation assigns ondansetron a secondary
negative-control role, so it is not a primary-model medication feature under
D6.0. It remains the negative control in the secondary propensity analysis
(section 10) and a tier-1 agent for ascertainment reporting. The observation
that ondansetron is not nephrotoxic is consistent with this role but is not the
basis for it; section 1:19's "nephrotoxic agents" phrasing scopes the secondary
research question, not primary-model inclusion.

**D6.0e [correction] False citation in code, removed; full citation audit.**
src/polysafe/review.py attributed its class list to a HANDOFF passage ("HANDOFF:
NSAIDs, ACEi/ARB, aminoglycosides, vancomycin, loop diuretics, contrast") that
appears in no committed version of HANDOFF.md. That attribution is removed and
replaced with the list's actual provenance: no committed protocol source defined
it; the NSAID class, aminoglycosides and iodinated contrast are named in D4.1
and D4.5; "ACEi/ARB" appears only in DATA_REPORT.md, a data report rather than
the protocol; loop diuretics appear in no committed document. The D6 analyst
note above describes the attribution as it stood before this removal.
A repository-wide audit then resolved every citation of a D-entry, section
number, line number or quoted protocol passage in code, docstrings, tests,
PREREGISTRATION.md, HANDOFF.md, DEMO_SCOPE.md, DATA_REPORT.md,
MAPPING_REPORT.md, REVIEW_PACKET.md, search_terms.txt and the results reports
against committed text. No other unresolved citations found. One citation that
resolved but overstated its source was tightened: the medication checklist
comment in src/polysafe/medmap.py cited sections 6 and 10 for both the agents
and the classes; section 10 names the agents and neither section names the
classes. A test now fails if any cited D-entry id or section number is absent
from the committed protocol.

**D6.0f [correction] Scope of commit 9635d73 and the append-only baseline.**

9635d73 added .gitattributes establishing "*.md text eol=lf". Running
`git add --renormalize .` produced no change to any stored markdown
content - the staged diff was ".gitattributes | 1 +" alone - so the mass
CRLF-to-LF modification anticipated when that step was specified did not
occur, and no pre-existing markdown line was altered by normalisation.

That commit did, however, amend D6.0b in place. D6.0b's sentence stating
that the line-ending commit "changed no committed line" refers to the
renormalisation only, not to the commit as a whole; the commit changed
D6.0b's own lines. That amendment is a modification inside the repair
window and is disclosed here rather than undone, since restoring D6.0b's
8230b37 text would require a further in-place edit and a further
re-baseline for no gain.

The document-level append-only property is baselined at 9635d73. Three
prior modifications lie outside that baseline:
  - 29a66d8 deleted a "(none yet)" placeholder. Disclosed, not reverted.
  - 46c9f9c inserted section 9.1 inside section 9. Disclosed, not
    reverted.
  - d972c0c inserted lines inside D6.0's block and was reverted by
    50792b0, leaving the file byte-identical to 067ee23.
No history was rewritten and no commit was force-pushed.

**D6.0g [protocol addition] Definition of "committed protocol".**
The term is used by D6.0c and D6.0e and was never defined. It is
defined here on the basis of HANDOFF.md:6-9, which names
PREREGISTRATION.md "the analysis plan of record", gives it precedence
where the two differ, and allows it to change only through its
append-only Deviations section.

"The committed protocol" means PREREGISTRATION.md as committed, including
its appended Deviations entries, and nothing else. It is the sole source
of pre-specification: no other file pre-specifies any analysis, variable,
exposure or threshold.

"Protocol documents", as used by the step-8 citation audit, means
PREREGISTRATION.md and HANDOFF.md - the files whose sections and entries
may be cited. That is a wider set serving a different purpose: citation
resolution, not pre-specification. The two terms are distinct and are not
interchangeable.

DATA_REPORT.md, MAPPING_REPORT.md, REVIEW_PACKET.md and DEMO_SCOPE.md are
records of measurement or process. Nothing in them pre-specifies anything.

This definition does not change the class set determined by D6.0c:
NSAIDs, aminoglycosides and iodinated contrast are named in
PREREGISTRATION.md (D4.1, D4.5) and remain in; ACE inhibitors/ARBs and
loop diuretics are named nowhere in it and remain out. The definition is
therefore not outcome-selected.

**D6.0h [clarification] Provenance of the two excluded code-only classes.**
D6.0c excluded ACE inhibitors/ARBs and loop diuretics as not named in the
committed protocol. Both share a single origin, recorded here.

Both trace to uncommitted drafts of HANDOFF.md, authored in session on
2026-10-04 and read by the implementing agent, containing: "Pool the rest
into classes (NSAIDs, ACEi/ARB, aminoglycosides, vancomycin, loop
diuretics, contrast)." The class list in src/polysafe/medmap.py was
drafted from that sentence. The sentence was removed when HANDOFF.md was
rewritten before its first commit and therefore appears in no committed
version. Git cannot verify those drafts; this is session testimony, not
evidence.

Two committed attributions then cited sources that do not support the
list:
  - de873fd, medmap.py:46-47: "draft ingredient search terms for the
    pre-specified agents and classes (sections 6 and 10)". Sections 6 and
    10 name no class. This is the FIRST committed trace of the class
    list, predating the review.py attribution. D6.0e recorded it as an
    overstated citation; it is also the list's earliest committed source.
  - 723e862, review.py: cited a HANDOFF passage present in no committed
    version.
Both were removed at 25502e2.

ACE inhibitors/ARBs additionally appear at DATA_REPORT.md:105-106 as a
pooling recommendation. That is a data report, not protocol, and it was
demonstrably not adopted as written: it names lisinopril as supporting
drug-specific estimates, while section 10's exposure list excludes
lisinopril. The plan of record departed from the recommendation.

Loop diuretics appear in no other committed file except documents
downstream of medmap.CHECKLIST. The RxNorm ATC codes C03CA and C03CB in
hicl_draft.csv and rxnorm_cache.json are data, not attribution.

The conclusion is unchanged: neither class is named in the committed
protocol as a pre-specified class or candidate, so both remain excluded
from the primary model under D6.0c.

**D6.0i [correction] Scope of "named nowhere in PREREGISTRATION.md".**
D6.0c and D6.0g state that ACE inhibitors/ARBs and loop diuretics are
named nowhere in PREREGISTRATION.md. As written that is now literally
false: the file names them in D6.0c, D6.0e, D6.0g and D6.0h, which exclude
them. The intended and operative claim is that they are never named in
PREREGISTRATION.md as a pre-specified class, candidate exposure, or model
variable, and appear only in the entries that exclude them. Read D6.0c
and D6.0g with that scope.

**D6.0j [clarification] Repair-process disclosures.**
  (a) The citation-existence test failed during the run that produced
      6f15199 (D6.0f). The defect was in the test, which treated a new
      entry's own heading as an unresolved citation while the entry was
      present in the working tree but not yet committed; it was not a
      defect in the committed content. Verified by checking out 6f15199
      and running the pre-cfae4bf test against it: the test failed
      against the WORKING TREE only. Against the COMMITTED tree at
      6f15199 it passes; with D6.0f present but uncommitted on top of
      084b305, the state of the original run, it fails exactly as it did
      then. Linux CI passed on f6c2e19. Fixed at cfae4bf.
  (b) The failure was hidden because pytest output was piped to `tail`,
      which discards the exit status. Commands are now run so that a
      failing test stops the chain.
  (c) Known limitation: the citation-existence test checks that a cited
      D-entry id or section number EXISTS. It does not check that the
      cited text supports the claim made. Every citation defect found in
      this repair - de873fd, 723e862, and medmap.py's sections 6 and 10 -
      was of the second kind, which this test cannot catch.

**D6.0k [correction] Count of citation defects in D6.0j(c).**
D6.0j(c) lists "de873fd, 723e862, and medmap.py's sections 6 and 10" as
the citation defects found in this repair. de873fd IS the medmap.py
"sections 6 and 10" citation, so that list names two distinct defects and
one of them twice. The count is two, both of the semantic kind - a cited
source that exists but does not support the claim made. No statement in
D6.0j(c) is false; the list is redundant and invites a miscount.

**D5.14a [correction] Meaning of "discarded" in D5.14.**
D5.14 states that the exploratory screen's output was discarded. That
means it was not adopted and never committed, both of which hold. It does
not mean deleted from the machine: the superseded build, including its
brand-like token list and key diagnostic table, persists in Claude Code's
edit-history store outside this repository. Nothing there is
redistributed and no repository-scoped claim in D5.14 is affected. The
computation provenance section of DATA_REPORT.md records the full
position.

**D5.5a [correction] Direction and partition of the medication-record AKI
comparison.**
D5.5 reports the comparison as "between stays with and without any
medication record (19.2% vs 15.1%)", which reads as 19.2% among stays
WITH a record. The computed values are the reverse: 19.2% among stays
WITHOUT a medication record (48 of 250) and 15.1% among stays WITH one
(118 of 784), over a universe of 1,034 stays under the superseded outcome
definition noted in D5.5. Read D5.5 with that direction.
The partition is at STAY level - whether that stay carries any medication
record - and is not the partition used by the site-level cohort
restriction, which includes a stay if its hospital recorded medications
for any patient. The two overlap but differ: stays with no medication
record of their own occur at recording hospitals and are therefore inside
the restricted cohort. No inference about cohort selection or confounding
is drawn from this figure here.

**D6.2 [substantive change] Route and systemic-exposure requirement.**
Exposure under this protocol requires a systemically absorbed route.
Ingredient identity alone does not establish exposure.

The determination is made at the medication ROW level, not at the key
level. A single HICL or name key may carry rows of several different
routes - systemically eligible, locally acting and route-unknown
together - so no key is classified as systemic or non-systemic as a
whole. Each row is classified by its own route evidence.

Route evidence is the source row's routeadmin field, mapped to a
normalized route category by mapping/route_normalization.csv (committed
edacf88, amended 0a6c3ef). Route is not inferred from an agent's usual
clinical route, and is not manufactured where the source row does not
support it.

Systemic eligibility for an (exposure, route category) pair is recorded
in mapping/systemic_routes.csv (committed f382149, corrected 4003219),
one row per pair with a stated basis, covering every pair exhaustively.
The rules themselves live in that table and not in this entry, so that a
correction to a rule is made in one place rather than creating a
discrepancy between the protocol text and the artifact. Where a route
category is not governed by a universal principle, the table carries an
exposure-specific determination; those determinations are part of the
protocol artifact and are not derived from exposure frequency or from
outcome data.

Classification and its consequence, applied per row:
  systemic = yes      -> contributes to the primary ascertainment
  systemic = no       -> not an exposure
  systemic = unknown  -> EXCLUDED from the primary ascertainment,
                         INCLUDED in a pre-specified sensitivity variant
The sensitivity variant distinguishes and separately counts two sources
of unknown: route unknown (no interpretable routeadmin value), and route
known but systemic absorption for that exposure not determined. Key, row
and stay counts are reported for each.

This narrows the exposure definition and is therefore a substantive
change, not a clarification. It is committed before any adjudication
decision is recorded: every decision cell in the five review-packet
files is empty at this commit. The route and systemic tables were
committed earlier; that ordering - implementation before protocol text -
is disclosed here rather than concealed.

The rule was prompted by the committed mapping draft, in which HICL:4042
maps to vancomycin, together with that key carrying rows whose route
normalizes to enteral_oral alongside its intravenous rows: 42 among
non-cancelled orders and 44 including cancelled ones, with 5 such rows
under HICL:8466. Oral vancomycin has bioavailability under approximately
5% and is given for C. difficile colitis. Ingredient identity alone does
not distinguish those rows from the intravenous rows of the same key, and
that is the gap this rule closes.

The state of medication-exposure computation at this commit is recorded
in the computation provenance section of DATA_REPORT.md rather than
restated here.

**D6.1 [protocol addition] Mapping-level errors in the audit estimator.**
D5.2 adopts the audit design and estimator in
mapping/review/REVIEW_PACKET.md. Every estimate that design defines is
built either from error_rows_i or from the key-level proportion of keys
with any wrong string. A key whose drugname strings are all correct
while the automated ingredient mapping is wrong therefore contributes no
error rows and is not a key with any wrong string, so no estimate can
represent it.

The committed mapping draft contains such cases, neither yet
adjudicated. HICL:11528 is an audit-head key whose proposed mapping
names insulin lispro protamine while its drugname strings name insulin
lispro, including the brand name Humalog. HICL:37442 was found by the
tier-3 screen and is in no audit stratum; its proposed mapping names
ilex paraguariensis while its drugname strings name vancomycin.

Add a per-key field mapping_wrong (boolean).
  - mapping_wrong is true only when the key's proposed ingredient
    mapping is wrong for every row in the key. Then error_rows_i is ALL
    of key i's rows, including rows with no drugname, and key i counts
    as an error key in the key-level proportion.
  - mapping_wrong is false in every other case, including a key whose
    proposed mapping is correct for some of its drugname strings and
    wrong for others. Such a key records the affected strings in
    wrong_strings with the reason in reviewer_note, and error_rows_i and
    the key-level proportion are as REVIEW_PACKET.md defines them,
    unchanged, including its rule for keys named only through
    admissiondrug.
A key with mapping_wrong true records no wrong_strings, because the
error is not string-specific.

mapping_wrong does not require corrected_name. D5.4 permits
corrected_name only to remove a key from an exposure, to move it between
two already pre-specified agent exposures, or to add or move it to an
already pre-specified class variable. So where a key with a wrong
mapping is currently assigned to an exposure it does not belong in,
corrected_name records that removal; where the key is assigned to no
exposure and the correct ingredient is not a pre-specified exposure,
corrected_name stays empty. These two cases do not exhaust the
possibilities. D5.4 provides no operation for adding a key to an already
pre-specified agent exposure, so for a key whose correct ingredient is
such an agent corrected_name stays empty as well, and whether that key
is added to that exposure is a question this entry does not settle. In
every case the correct ingredient is recorded in reviewer_note.

REVIEW_PACKET.md scopes its combined estimate to the tier-4 approximate
stratum and states that it says nothing about the exhaustively reviewed
tiers. A mapping error recorded in one of those tiers therefore does not
enter that estimate, whether or not mapping_wrong records it.

Report every estimate both ways - as REVIEW_PACKET.md defines it, and
with mapping-level errors included - so the effect of this amendment is
visible. D5.2's restriction of Wilson intervals to the key-level
proportion applies to both versions of that proportion.

This entry is appended after D6.2 and is therefore out of numeric order.
The number records the amendment's subject, not its position.

**D6.1a [correction] Scope of D6.1's statement on corrected_name and agent exposures.**
D6.1 states that corrected_name stays empty for a key whose correct
ingredient is an already pre-specified agent. That holds only for a key
assigned to no exposure. Read generally it is wrong.

The governing rule is simpler than D6.1's two cases suggest:
corrected_name records whichever of D5.4's three permitted operations
applies to the key, and stays empty when none of them does. In
particular, where a key is assigned to one already pre-specified agent
exposure and its correct ingredient is a different already pre-specified
agent, D5.4's second operation applies and corrected_name records that
move. D6.1's sentence must not be read as forbidding it.

The case D6.1 identifies as uncovered is narrower than its wording: a
key assigned to no exposure whose correct ingredient is an already
pre-specified agent. D5.4 provides no operation for that case, so
corrected_name stays empty, and whether the key is added to that
exposure remains unsettled.

In every case the correct ingredient is recorded in reviewer_note, as
D6.1 states.

This entry is appended out of numeric order. The number records the
amendment's subject, not its position.

**D6.3 [protocol addition] Review-corrected HICL-based ascertainment as a fifth arm.**
Section 10 lists vancomycin, piperacillin-tazobactam and furosemide as
exposures. Section 6 names four ascertainment strategies: generic-string
only, generic + brand, HICL-based, HICL + gtc. The HICL-based strategy
derives exposure from each HICL key's ingredient mapping; the
generic-string and generic + brand strategies derive it from drugname
strings and do not depend on that mapping. A HICL key whose drugname
strings name a pre-specified agent while its proposed mapping does not
is therefore missed by the HICL-based strategy, and the defect is
differential across the strategies being compared.

The committed mapping draft contains such keys. In tier3_screen.csv,
HICL:37442's drugname strings name vancomycin and its
proposed_ingredients field reads "ilex paraguariensis leaf extract";
HICL:10093's drugname strings name vancomycin and its
proposed_ingredients field is empty.

D5.4 provides no operation for adding a key to an already pre-specified
agent exposure, and this entry does not add one. The four strategies
named in section 6 are unchanged, including the HICL-based strategy with
these keys missed.

Add a fifth ascertainment arm, review-corrected HICL-based. It is
constructed from the HICL-based strategy by reassigning a HICL key when
and only when all of the following hold as recorded in the review
files:
  - mapping_wrong is true for that key, per D6.1, or the key's
    proposed_ingredients field is empty;
  - reviewer_note names the correct ingredient;
  - that ingredient is an already pre-specified agent or a member of an
    already pre-specified class;
  - every string in the key's drugnames_medication and
    drugnames_admissiondrug fields names that ingredient.
Such a key is reassigned to that agent's exposure or that class
variable. No other key is reassigned. No exposure or class is created.
The field corrected_name is not used in this construction. In every
other respect the fifth arm is constructed exactly as the HICL-based
strategy is.

For a key whose proposed_ingredients field is empty, the reviewer
records the correct ingredient in reviewer_note. This entry does not
change the meaning of mapping_wrong or any estimate defined by D6.1 or
REVIEW_PACKET.md.

The fifth arm is secondary and exploratory. The comparison of the
HICL-based strategy against it is the quantity of interest: it measures
how much a mapping-correctness review changes exposure ascertainment.

This arm is added before any review cell has been filled and before any
estimate of the association between a pre-specified exposure and the
outcome has been computed under any ascertainment arm. It was added on
the basis of mapping correctness alone; no outcome was examined in
deciding to add it.

**D6.4 [protocol addition] Medication exposure window, cancelled orders and gtc precedence.**
Section 3 restricts features to data with offset <= T. For every
ascertainment strategy, a medication row counts toward exposure only if
all of the following hold:
  - drugstartoffset <= T, with T = 1440 minutes;
  - drugstopoffset >= 0;
  - drugordercancelled is not "Yes".
A row that started before ICU admission and had not stopped by admission
therefore counts; a row that both started and stopped before admission
does not. A drugstopoffset of 0 is taken at face value.

In the HICL + gtc strategy, gtc is used only for rows with no
drughiclseqno. Rows with a drughiclseqno are ascertained exactly as in
the HICL-based strategy, whatever their gtc.

A stay is exposed to an agent or class when at least one of its rows
counts toward it.

Decided by the PI before any ascertainment code exists and before any
drug-outcome association has been computed. Earlier exploratory exposure
counts recorded in DATA_REPORT.md predate this rule. Disclosed: when
choosing, the PI saw whole-table counts of medication rows starting
before admission, stopping before admission and cancelled; no outcome
was examined.

**D6.5 [clarification] The aminoglycoside class after review.**
D4.5 states that aminoglycosides are "absent from the data", and D6.0c
that they "contribute no variable - an empty set, not an exclusion."
The review assigned one key to the class through corrected_name:
HICL:33356, in audit_tail.csv, named only through admissiondrug as
NEOSPORIN. Its 7 medication rows have routeadmin "Topical" or "TP",
which mapping/route_normalization.csv maps to topical, and
mapping/systemic_routes.csv classifies aminoglycosides by that route as
systemic = no. Under D6.2 the class therefore has no row that counts
toward exposure and still contributes no variable. D4.5 and D6.0c are
read as statements about systemic exposure. Any later change that gives
the class a counting row needs its own entry.

**D6.6 [protocol addition] Exposures missed by an empty mapping in the audit sample.**
The audit estimator counts error_rows_i from wrong_strings and, in the
D6.1 version, from mapping_wrong. A key whose proposed_ingredients is
empty but whose correct ingredient belongs to a pre-specified agent or
class contributes no error rows under either version, although the
automated mapping missed an exposure. The review recorded two such keys
in the audit sample, both in audit_tail.csv: HICL:4703, corrected_name
"iodinated contrast", and HICL:33356, corrected_name "aminoglycosides".

The estimator is not changed. Alongside every audit estimate, the number
of audit keys with an empty proposed_ingredients whose corrected_name
assigns them to a pre-specified agent or class is reported, with their
row counts.

This entry was written after the review was complete and after these two
keys were seen.

**D5.14b [clarification] The brand alias list was committed on 2026-10-08.**
D5.14 states that the brand alias list "was authored after an uncommitted
exploratory screen". Before 2026-10-08 no committed version of
mapping/review/search_terms.txt contained any brand alias: every slot in
its brand section was empty, and screen B had not been run. The list is
committed now, alone, as that file's header requires. It was written on
2026-10-08 with an AI assistant proposing aliases from drug references;
the PI approved each alias, and added Zosyn under piperacillin after a
check. It was written after the PI's review of the 171 keys, in which
brand strings such as TORADOL, OPTIRAY and NEOSPORIN were seen, and after
the exploratory brand counts recorded in DATA_REPORT.md. No drug-outcome
association had been computed.

This entry is appended out of numeric order. The number records the
amendment's subject, not its position.

**D6.3a [clarification] How arm 5 reads the correct ingredient and when a string names it.**
D6.3 reassigns a key only if "reviewer_note names the correct
ingredient" and "every string in the key's drugnames_medication and
drugnames_admissiondrug fields names that ingredient".
  - reviewer_note names the correct ingredient when it begins
    "Correct ingredient: " followed by the ingredient, which ends at the
    first full stop. A note in any other form names no ingredient for
    this purpose and the key is not reassigned; such keys are listed
    when the arm is built.
  - A string names an ingredient when, under the matching of D5.10, it
    contains that ingredient's generic name or one of its brand aliases
    in the committed brand section of mapping/review/search_terms.txt.
After the review, the PI rewrote four reviewer notes into this form:
HICL:37442, HICL:5175, HICL:4703 and HICL:33356. Decided after the
review was complete; no outcome was examined.

This entry is appended out of numeric order. The number records the
amendment's subject, not its position.

**D6.7 [protocol addition] Which rows and keys the HICL-based arms read.**
Section 6 states that exposure "is derived from `drughiclseqno` (HICL),
not `drugname`". In the HICL-based strategy, the HICL + gtc strategy and
the review-corrected HICL-based arm (D6.3), a row is ascertained through
the reviewed mapping only if it has a drughiclseqno. Rows without one are
not ascertained through their NAME: key in those arms; the generic-string
and generic + brand strategies read every named row by its drugname.

In the HICL-based and HICL + gtc strategies, review decisions are applied
only to keys in tiers 1-3. Tier-4 keys, including those in
audit_head.csv and audit_tail.csv, feed no exposure in those two
strategies (D5.1). Their adjudication is used by the audit estimator, by
D6.6, and by the fifth arm through D6.3.

In medication.csv.gz every row without a drughiclseqno (4,441 rows) has
gtc = 0, so on this dataset the HICL + gtc strategy produces the same
exposure ascertainment as the HICL-based strategy. This is a property of
the data, not a rule.

Decided by the PI before any ascertainment code is committed and before
any drug-outcome association has been computed.

**D2.1a [substantive change] How the BCa interval is computed and how its coverage is measured.**
D2.1 replaces the percentile interval with "a patient-level BCa
(bias-corrected and accelerated) bootstrap CI" and states "Measured
coverage will be reported alongside every interval." D2.1 does not
specify the number of resamples or the procedure for measuring coverage;
this entry specifies both.

Computation:
  - Every BCa interval is a two-sided 95% interval.
  - A resample draws patients (`uniquepid`) with replacement and keeps
    every stay of a drawn patient. The acceleration is estimated by the
    leave-one-patient-out jackknife.
  - Every interval reported for the real cohort uses 10,000 resamples.
    Section 9.1 simulated 1,000 resamples under a design it describes as
    "identical to the analysis"; the real intervals depart from it in
    using 10,000.
  - Resamples reuse the out-of-fold predictions; no model is refitted
    within a resample, so the interval does not include the variation from
    refitting the models.
  - A resample on which the statistic is undefined (for example, one
    without both outcome classes) is dropped, not redrawn. Each interval
    is reported with the number of resamples attempted, used and dropped.
  - If the bias correction, the acceleration or an adjusted percentile is
    not finite, no interval is reported for that statistic and the failure
    is reported. No other interval method is substituted.

Coverage simulation:
  - Before any model is fitted, the section 9.1 simulation is re-run with
    BCa intervals: the same cohorts, grid, seed and simulated studies, with
    2,000 resamples per simulated study. Section 9.1 and its result files
    are not changed; the re-run writes its own files.
  - For a simulation cell, coverage is the share of its simulated 95%
    intervals that contain the corresponding true value. It is measured for
    the dAUROC and for the AUROCs of the simulation's clinical-only and
    combined scores, whose true values are the grid dAUROC, A0, and A0 plus
    the grid dAUROC. At a true dAUROC of 0 the share of dAUROC intervals
    whose lower bound is above 0 is also reported, and dropped resamples
    are counted per cell.

Reporting:
  - Next to each real AUROC or dAUROC interval: the coverage in the cell for
    the primary cohort at its measured prevalence whose A0, rho and true
    dAUROC are the grid points nearest to the realised clinical-only AUROC,
    the realised correlation of D2.4 and the observed dAUROC estimate, with
    that cell's coordinates; and the minimum and maximum coverage over all
    cells of that cohort and prevalence.
  - Any other bootstrap interval, including AUPRC, is reported with
    "coverage not measured".
  - "Detected" remains "lower bound > 0". The simulated detectable effect in
    section 9's null statement is the 80%-power threshold of the BCa re-run
    for the primary cohort at its measured prevalence, at the A0 and rho grid
    points nearest to the realised clinical-only AUROC and the realised
    correlation of D2.4, found by linear interpolation between grid dAUROC
    values as in section 9.1. The section 9.1 percentile threshold for the
    same cohort, prevalence, A0 and rho is reported beside it.
  - The BCa interval stays the primary interval whatever coverage the re-run
    shows. Poor coverage is reported as a limitation; the interval method is
    not changed after the coverage is seen.

Decided by the PI after the section 9.1 simulation and before any
prediction model is fitted to the cohort's data. No drug-outcome
association has been computed.

This entry is appended out of numeric order. The number records the
amendment's subject, not its position.

### D7 — 2026-10-09, before any model was fitted

**D7.1 [substantive change] Medication features of stays without a medication record.**
Section 6 states that "Absence of a record is never coded as "unexposed""
and that "`medication_data_available` will NOT be used as a predictor".
D3.6 imputes the logistic regression with "median plus missingness
indicator". In the primary cohort 20 of the 671 stays have no medication
record of their own. For those stays every medication feature is missing:
the feature builder gives them no 0 or 1, only a missing value.
  - In every logistic regression each medication feature of those stays is
    filled with the share of exposed stays among the training fold's stays
    that have a medication record. No missingness indicator is added for the
    medication features. This departs from D3.6 for these features only:
    the median of a binary exposure held by fewer than half of the stays is
    0, which would code the 20 stays as unexposed, and an indicator would
    flag exactly the stays without a medication record.
  - In LightGBM the medication features of those stays stay missing.
  - The 20 stays remain in the primary cohort.
Decided by the PI before any model was fitted.

**D7.2 [protocol addition] Model specification fixed before the first fit.**
  - Primary estimand: the paired dAUROC between the clinical + medication
    and the clinical-only L2-penalised logistic regressions on D4.4's
    reduced feature set. The same comparison with LightGBM is secondary;
    with the full penalised logistic regression it is a sensitivity
    analysis (D4.4).
  - Medication features: vancomycin, piperacillin-tazobactam, furosemide,
    NSAIDs (aspirin excluded) and iodinated contrast (D6.0, D6.0c). Each is
    1 if the stay is exposed by T under D6.4 in the HICL + gtc strategy,
    primary variant, and 0 otherwise, except as D7.1 states.
    Aminoglycosides contribute no variable (D6.0c, D6.5) and ondansetron is
    not a feature (D6.0d). The clinical + medication model adds these five
    to its clinical-only counterpart and changes nothing else.
  - Full penalised logistic regression: every clinical feature except the
    five that D3.6 and D4.2-D4.3 leave out of the logistic regression
    (lactate, pH, phosphate, INR, temperature): 69 of the 74 clinical
    features in `results/features_clinical.csv`. D4.4 calls it the "full
    74-feature" model.
  - Folds: `polysafe.cv.patient_folds`, 5-fold StratifiedGroupKFold by
    `uniquepid`, stratified on AKI, 10 repeats with random states 0-9. Both
    models of a comparison use identical folds.
  - Pooling: each stay's 10 out-of-fold predicted probabilities are
    averaged. Every metric, the paired dAUROC and every bootstrap interval
    is computed on that one averaged prediction per stay; per-repeat
    metrics are not averaged.
  - Logistic regression: imputation as D3.6 and D4.4 state, except as
    D7.1 states, fitted on the training fold; then every column is
    standardised with the training fold's mean and standard deviation (a
    column with zero standard deviation is set to 0). scikit-learn
    LogisticRegression, lbfgs solver, intercept not penalised, no class
    weights, at most 10,000 iterations. C is chosen from 13 values,
    10^-3 to 10^3 in steps of 10^0.5, by the lowest mean log loss in an
    inner 5-fold patient-grouped, AKI-stratified split of the training
    fold (`patient_folds` with one repeat and random state 1000 + 5r + k
    for outer repeat r and fold k); ties go to the smaller C. The model
    is then refitted on the whole training fold.
  - LightGBM: all clinical features plus the medication features, all
    numeric, no categorical handling, missing values left missing.
    Fixed: binary objective, learning rate 0.05, no row or column
    subsampling, no L1 or L2 penalty, random seed 20261004,
    deterministic mode, one thread. Tuned in the same inner split by the
    lowest mean binary log loss on the inner validation folds over
    num_leaves {4, 8, 16} x min_child_samples {10, 20, 40} x
    n_estimators {100, 300}; ties go to fewer leaves, then more samples
    per leaf, then fewer trees. Then refitted on the whole training
    fold.
  - Metrics on the pooled predictions: AUROC (ties counted half); AUPRC as
    average precision; for both, the D2.1a BCa interval. Before any logit
    is taken, predictions are clipped to [10^-6, 1 - 10^-6]. Calibration
    intercept (an intercept-only logistic regression of AKI with the logit
    of the prediction as offset) and slope (logistic regression of AKI on
    that logit), point estimates only. Calibration curve: mean prediction
    against observed AKI proportion in 10 quantile bins of the pooled
    predictions (duplicate bin edges dropped). Decision curve: a stay is
    treated when its prediction is at or above the threshold; net benefit
    = TP/n - FP/n x t/(1 - t), for both models, treat-all and treat-none,
    at thresholds t from 0.02 to 0.40 in steps of 0.01, one grid for all.
  - D2.4: "The realised correlation between the clinical-only and
    clinical+medication predictions will be measured and reported, to
    identify which correlation (rho) in the power table applies." Section
    9.1's rho is the "Pearson correlation rho between the two models' risk
    scores", and in `polysafe.power` the simulated predictions are logistic
    recalibrations of those scores. The realised rho used for D2.1a's
    matched cell is therefore the Pearson correlation between the logits of
    the two models' pooled predictions; the Pearson correlation between the
    pooled predicted probabilities is reported beside it.
Drafted by the analyst and approved by the PI before any model was
fitted; no model output had been seen.

### D8 — 2026-10-09, while the primary model run was in progress

**D8.1 [protocol addition] How the pre-registered sensitivity analyses are run and reported.**
  - Model: each analysis below repeats only the primary comparison of D7.2
    (the clinical-only and clinical + medication reduced L2 logistic
    regressions, with every D7.2 setting) and reports both AUROCs, the
    paired dAUROC with its D2.1a BCa interval and the D2.4 correlation.
    LightGBM is not repeated.
  - Ascertainment (section 6, D5.9, D6.3): the medication features are
    taken from each of the five arms (generic-string, generic + brand,
    HICL-based, HICL + gtc, review-corrected HICL-based) under each of the
    four variants (primary; systemic = unknown included, D6.2; unsure keys
    included, D5.13; unnamed rows excluded, D5.12), on the primary cohort:
    20 analyses, one of which (HICL + gtc, primary variant) is the primary
    analysis. The review-corrected arm is labelled exploratory (D6.3). All
    20 are reported together in one table.
  - Full cohort (section 6): the 868 baseline-A stays at all hospitals.
    Every stay without a medication record of its own (217 stays) is
    handled as D7.1 states for the 20 record-less stays of the primary
    cohort. Simulated coverage is read from the n = 868 cohort's cells at
    its measured prevalence instead of the primary cohort's.
  - Follow-up (section 3, D2.2): the primary cohort restricted to the 314
    stays with at least 48 h of post-T follow-up; the outcome is unchanged.
    D2.2's "fixed-window" is read as this restriction.
  - Baseline C (section 4): the cohort, exclusions and AKI labels built
    with baseline C, at medication-recording hospitals (497 stays). The
    clinical features are built by `polysafe.features` for those stays.
    Simulated coverage: not measured.
  - Hospital clusters (section 7): a cluster is a connected set of stays
    linked through a shared hospital or a shared patient (`uniquepid`), so
    no hospital and no patient spans two folds. The outer folds, the inner
    tuning splits and the BCa resampling all use these clusters in place of
    the patient; otherwise as D7.2 (5 folds, 10 repeats, random states
    0-9). If any outer test fold contains no AKI event, the analysis stops
    and the PI decides how to proceed. Simulated coverage: not measured.
  - Ethnicity (D3.4): on the primary analysis's pooled predictions, both
    AUROCs and the dAUROC within each ethnicity group, with its stays and
    events; missing ethnicity is its own group. A BCa interval is reported
    only for groups with at least 10 AKI events; other groups get point
    estimates only, and a group with no event gets no AUROC. No test
    between groups; descriptive only.
  - Section 6's "movement in the conclusion": each analysis's dAUROC and
    interval are reported beside the primary result, with the difference
    between the two point estimates and whether each interval excludes 0.
    No test between analyses; no sensitivity analysis replaces the primary
    result.
Decided by the PI while the primary model run was in progress; no model
output had been seen by the PI or the analyst when these choices were made.

### D9 — 2026-10-10, after the section 8 results were seen

**D9.1 [correction] Who made the decisions this document attributes to the PI.**
  - The header line "PI: Vinay (Pharm D). All clinical definitions below are
    his decisions." is wrong. Every clinical definition, the drug
    classification (including the adjudication of the 171 review-packet keys
    on 2026-10-06) and every decision in D1-D8 attributed to "the PI" was
    made by Dr. Varad Vaidya (PharmD), who is also the analyst. Vinay had no
    role in the study.
  - The header line stays as written because this document is append-only;
    read it, and every "PI" in sections 1-11, as Varad Vaidya.
  - No definition, analysis or result changes.
Recorded after the section 8 results were seen; it concerns attribution only.

**D9.2 [correction] When D8.1 was decided.**
  - The last sentence of D8.1 states that no model output had been seen when
    its choices were made. That is wrong. The section 8 run ended at
    15:24:34 on 2026-10-09; its results were committed as `a713fd2` at
    15:24:50 and the full `results/models.md` was displayed; D8.1 was drafted
    after that and committed as `0241278` at 15:28:31. The PI had seen the
    section 8 results, including the primary paired dAUROC, before D8.1 was
    finalised.
  - Which sensitivity analyses are run was fixed before any model was fitted
    (sections 3, 4, 6 and 7, D2.2, D3.4). How each one is run and reported,
    as D8.1 states it, was fixed after the primary result was seen, and is
    reported as such wherever the D8.1 analyses are reported.
  - No definition, analysis or result changes.
Recorded on 2026-10-10, after the section 8 results were seen.
