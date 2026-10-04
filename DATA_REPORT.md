# PolySafe — eICU-CRD Demo v2.0.1 data reality check

Measured from the files in `data/`, all 14 verified against the official
`SHA256SUMS.txt`. Every number here came from running code, not from
documentation.

> **Revision 2 (2026-10-04).** Revision 1 had three errors, found by an
> independent audit and confirmed by re-running: creatinine counts mixed
> "all rows" with "rows having a value"; the coverage table wrongly included
> urinary creatinine; and drug exposure counts missed brand names. All are
> corrected below, and the medication-completeness section is new.

## Cohort

| Measure | Value |
|---|---|
| Unit stays (`patient.csv.gz`) | **2,520** |
| Distinct `hospitalid` | 186 |
| Largest single hospital | 40 stays |
| Hospitals with >= 20 stays | 11 |
| Hospitals with >= 50 stays | 0 |

**Design consequence:** no hospital is large enough to hold out alone.
Hospital-level generalization needs *grouped* cross-validation over clusters of
hospitals. (PhysioNet's description says "20 of the larger hospitals"; the file
has 186 distinct ids. The measured value is authoritative — say so in the README.)

## Creatinine coverage (`lab.csv.gz`) — serum only

Rows with a value: **14,603 serum `creatinine`** + 123 `urinary creatinine`
= 14,726 total. Urinary creatinine is a different analyte and is excluded
everywhere below and in all labelling.

| Measure | Serum only |
|---|---|
| Stays with >= 1 | 2,373 (94%) |
| Stays with >= 2 | **2,058** (82%) |
| Stays with >= 3 | 1,699 (67%) |
| Stays with >= 5 | 1,113 (44%) |
| Median per stay | 4 |

## Baseline creatinine availability

| Baseline | Stays | Note |
|---|---|---|
| **A (primary)** first serum creatinine in first 24 h | 2,092 (83%) | PI's primary choice |
| **C (sensitivity)** most recent value before ICU admission (`labresultoffset < 0`) | 1,687 (67%) | 2,979 pre-admission rows |

## Provisional AKI labelling (2,058 stays with >= 2 timed serum values)

| Criterion | Stays | % |
|---|---|---|
| >= 0.3 mg/dL rise within 48 h | 465 | 22.6 |
| >= 1.5x lowest value in stay | 801 | 38.9 |
| Either | 928 | 45.1 |

**Provisional only.** These used "lowest value during the stay" as baseline,
which peeks at future values and counts ordinary fluctuation as AKI — hence 45%
against a real-world 20-30%. The PI has since fixed baselines A and C; expect
the final incidence to land nearer the 22.6% figure. These rows exist to be
superseded, not cited.

## Medications (`medication.csv.gz`) — read this before building features

75,604 rows, 1,401 distinct `drugname` strings. Three problems, in order of size:

**1. 40.6% of rows have no `drugname` at all** (30,661 rows). Largely
recoverable:

| Field present on blank-name rows | Share |
|---|---|
| `gtc` (generic therapeutic class) | 100% |
| `routeadmin` | 100% |
| `drughiclseqno` (HICL code) | 94.6% |
| HICL that appears elsewhere *with* a name | **69.7%** |

**`drughiclseqno` is the better join key than `drugname`** — 1,027 distinct HICL
codes versus 1,401 raw name strings, i.e. it is already partly normalized. Build
a HICL -> ingredient map, use `drugname` as a human-readable label, and fall back
to `gtc` for the remaining ~5%.

**2. Only 1,857 of 2,520 stays (74%) have any medication row**; 1,765 have a
named one. **Missing medication data is not evidence of no exposure.** Treating
absence as "unexposed" would bias every drug effect toward the null. Either
restrict the medication analysis to the 1,857 stays with records, or model
missingness explicitly — the PI decides which.

**3. Names are unnormalized and brand names are recorded separately.**
`potassium chloride`, `POTASSIUM CHLORIDE` and
`POTASSIUM CHLORIDE CRYS ER 20 MEQ PO TBCR` are three strings for one ingredient.

Exposure after adding hand-picked brand names (still raw text matching, so
treat every figure as a **lower bound**):

| Agent | Generic only | + brand names |
|---|---|---|
| furosemide | 442 | **558** (+ Lasix) |
| vancomycin | 299 | **315** (+ Vancocin) |
| lisinopril | 235 | **269** (+ Zestril, Prinivil) |
| piperacillin | 85 | **163** (+ Zosyn) |
| ketorolac | 54 | **75** (+ Toradol) |
| losartan | 17 | **29** (+ Cozaar) |
| ibuprofen | 17 | **23** (+ Motrin, Advil) |

Only furosemide, vancomycin, lisinopril and piperacillin support drug-specific
estimates. Pool the rest into classes (NSAIDs, ACEi/ARB, aminoglycosides).
Proper HICL-based normalization will raise all of these.

## Data provenance

eICU Collaborative Research Database Demo v2.0.1, PhysioNet, open access under
the Open Database License (ODbL). All 14 tables verified by SHA-256 against the
distribution's own `SHA256SUMS.txt`. The full 130 MB zip was not used: the
needed tables were fetched individually because PhysioNet's on-the-fly zip
endpoint served at ~22 KB/s against ~95 KB/s for static file URLs.
`hospital.csv.gz` is empty in the demo. `vitalPeriodic` and `nurseCharting`
were deliberately not downloaded.
