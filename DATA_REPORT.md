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
`hospital.csv.gz` is empty in the demo. `nurseCharting` was deliberately not
downloaded. `vitalPeriodic` was not downloaded for this report; it was fetched
later, when PREREGISTRATION D3.2 added it, and is verified with the others under
Integrity.

## Integrity

SHA-256 of each eICU table in use, as recorded in the distribution's own
`SHA256SUMS.txt` (eICU-CRD Demo v2.0.1). `SHA256SUMS.txt` lives with the tables in
the git-ignored `data/` folder and is not redistributed, so the values below are
this repository's record of it. Every local file was re-hashed on 2026-10-05 and
matched its recorded value.

Medication table (`medication.csv.gz`):
`480a251570a27932110452f66f248061f712c306663c274f78f90875648cb720`

| Table | SHA-256 |
|---|---|
| `admissionDx.csv.gz` | `607020180f1c7fbbc994a23552d6a4fdc70b86d4abc8e1b36f3ee28e96f15c42` |
| `admissiondrug.csv.gz` | `fcf0af0de0efe18ec932199d2fcb308143886df6d50daea869bea48464ea6d68` |
| `apacheApsVar.csv.gz` | `2dc6d63f9807c7e663ffa253998691605d527b702ac1db8cc5dda92308061e78` |
| `apachePatientResult.csv.gz` | `ce799692737f7a712c59aa8fbb9a302cee7c719887f0638caed3ae144b166821` |
| `apachePredVar.csv.gz` | `99d5a4ab2c5cd07b6ee0a540c86a576e5975ed3b93df8525e3d287d5babf3eae` |
| `diagnosis.csv.gz` | `c99770c25fdeaee1eb34e490f982bd17459bc8902f951a2ae23adef65e7b1b52` |
| `infusiondrug.csv.gz` | `3b9f024d28682884a820763ac516ae68b247ebf17903fee4f38240c04886d527` |
| `intakeOutput.csv.gz` | `45db8ee0e64a4e3745284bde5828f740175770c45d00ea0c27daff789d18106e` |
| `lab.csv.gz` | `e06852850abeb65211bc22ac43f17121ba41b14656fac862396449af84d8273a` |
| `medication.csv.gz` | `480a251570a27932110452f66f248061f712c306663c274f78f90875648cb720` |
| `pastHistory.csv.gz` | `3af6e3101c037347365d81daf73ea487ba12e14a6a10f0b2ed53beed2e30c7d5` |
| `patient.csv.gz` | `15e1eb52169828e3cfff3b67132a087298de4d0365bbb31bb294f6043bffd474` |
| `treatment.csv.gz` | `daefee030fa1ad29593dce880ad86bdfbdbb1e33fcb3e78fa21d71cb01a08628` |
| `vitalAperiodic.csv.gz` | `c294f46647990cb648b3ebfa589e83f330bdc0c2c922b5e8f23aa0ef99954635` |
| `vitalPeriodic.csv.gz` | `04dc093a2e77cd4c52ac7f2a47a8356fbb7456c7f15deb20d5abb1ca4303231d` |

The first fourteen are the tables this report was measured from. `vitalPeriodic`
was fetched later, when PREREGISTRATION D3.2 added it.

## Computation provenance

As of 2026-10-05, at the commit that adds this section (parent 4003219). Recorded
from a read-only search of the repository (every commit, untracked and ignored
files, the GitHub remote), the analysis scratch and temporary folders, the local
drives and user folders, editor histories and the working-session transcripts.

- **No medication exposure feature exists.** No stay-level or row-level medication
  exposure has been constructed by the pipeline or persisted anywhere. The only
  stay-level exposure computation is `review.stay_exposure`, which runs in memory
  inside `tests/test_review.py` on synthetic fixtures of 12 and 13 stays and is
  never written out. Report-only scripts assigned real medication rows to
  exposures in memory to produce the aggregate counts below; none wrote a row- or
  stay-level file.
- **Aggregate exposure counts do exist** (keys, rows or stays per key, agent, class
  or route; no stay identifiers, no outcome).
  - Committed: `mapping/MAPPING_REPORT.md`, `mapping/hicl_draft.csv`, the
    `n_stays`/`n_rows` columns of the review-packet CSVs,
    `mapping/review/REVIEW_PACKET.md`, this report, prose counts in
    `PREREGISTRATION.md` and `HANDOFF.md`, and the 12 cells of
    `mapping/systemic_routes.csv` whose basis reads "no determination made; no
    rows present in the data".
  - Outside the repository: analysis scratch files, including `cells_after.txt`
    and `cells_after2.txt` (rows and distinct stays per exposure and route
    category; aggregate counts only, no stay identifiers, no AKI);
    `CONTEXT_BRIEF.md`; pytest temporary copies of the review-packet CSVs; and
    `medmap.log`, an uncommitted rendering of the mapping report.
- **No drug-specific or class-specific exposure-outcome estimate exists anywhere.**
- **Two outcome comparisons exist; neither concerns exposure to any drug.**
  1. The medication-record availability comparison disclosed in PREREGISTRATION
     D5.5: crude AKI 19.2% in stays without any medication record and 15.1% in
     stays with one, under the superseded uncensored outcome definition.
  2. The pre-registered site-level restriction, which compares the whole cohort
     with its medication-recording-site subset: baseline A 12.3% (107/868) vs
     11.6% (78/671), baseline C 10.8% (71/660) vs 9.9% (49/497), in
     `results/cohort_flow.md` and `results/aki_labels.md`, and restated in other
     results files and in `PREREGISTRATION.md`. The complement - baseline A stays
     at non-recording sites, 29 events in 197 stays - was printed in no file
     before this section; it follows from those figures by subtraction.
- **The stay-level and hospital-level partitions differ** (added 2026-10-05,
  parent b35d556; counted from the patient and medication tables, no outcome
  read). A hospital is a recording hospital if it has a medication row for any
  stay: 186 hospitals in total, 144 recording, 42 non-recording. Stays with no
  medication record of their own: 663 of all 2,520 demo stays, of which 142 are
  at recording hospitals and 521 at non-recording ones. Within the baseline-A
  cohort of 868: 217 with no record, of which 20 fall inside the restricted 671
  and 197 are at non-recording hospitals. The 1,034-stay universe behind D5.5's
  figure used superseded labels and its exact split cannot be reconstructed
  without re-running the old labelling; a note made at the time put 78.6% of
  record-less stays at non-recording hospitals, consistent with 521 of 663.
- **On generalizability:** the site-level comparison, not D5.5's stay-level
  figure, is what shows the restricted cohort has lower crude AKI than the
  excluded stays - 11.6% in the 671 against 29 events in the 197
  non-recording-hospital stays (14.7%). D5.5's figure does not establish this on
  its own.
- **Open question, recorded not resolved:** 20 stays inside the restricted 671
  have no medication record of their own. Section 6 states that absence of a
  record is never coded as "unexposed" and that genuinely missing values are
  preserved as missing. How those 20 stays' medication features are handled is
  not yet specified and must be settled before the feature builder runs.
- **Editor and tool caches outside the repository retain superseded content.**
  - Claude Code's edit-history store under the Windows temporary folder holds the
    discarded exploratory review build (PREREGISTRATION D5.14), including its
    brand-like token list and key diagnostic table, together with earlier versions
    of `PREREGISTRATION.md`, which carry the committed D5.5 text.
  - Claude Code's file-history store in the user profile holds a copy of a
    working-tree `HANDOFF.md` that contained the D5.5 figure for about 39 minutes
    on 2026-10-04 (16:22 to 17:01 IST). That file was rewritten without the figure
    before `HANDOFF.md` was first committed (ec81281, 17:16 IST).
  - Nothing in these caches is redistributed, and none of it alters any
    repository-scoped claim.

## Review adjudication (2026-10-06)

- **What was reviewed:** all 171 keys of the five review-packet files
  (tier1_agents 33, tier2_classes 36, tier3_screen 7, audit_head 45,
  audit_tail 50), merged in commit 50ceede.
- **How:** Claude built a workbook from the committed packet files, one tab per
  file, with the five decision columns as inputs. The PI filled it with an AI
  assistant proposing values; the PI approved or changed each value. The
  adjudication is the PI's; it is AI-assisted and is described as such.
- **Checks before merging:** every value was validated against the committed
  rules (allowed vocabulary, mapping_wrong on every decided key, wrong_strings
  present among the key's strings, corrected_name a pre-specified agent or
  class, reviewer_note required where the mapping is wrong or empty, cited
  entries exist). Seven corrected_name cells added a key to an agent exposure,
  which D5.4 and D6.1a do not allow (HICL:10093, HICL:37442, four vancomycin
  `NAME:` keys and `NAME:ZOSYN 3.375 GRAM in NS 100 mL IVPB`); the PI chose to
  blank them. HICL:10093 and HICL:37442 reach vancomycin only through arm 5
  (D6.3).
- **Decisions:** tier1 33 keep; tier2 36 keep; tier3 7 keep; audit_head 45
  reject; audit_tail 3 keep, 46 reject, 1 unsure. mapping_wrong is true for
  HICL:37442 and HICL:11528 only. corrected_name adds HICL:5175 to NSAIDs,
  HICL:4703 to iodinated contrast and HICL:33356 to aminoglycosides (see D6.5).
- **Audit misses (D6.6):** 2 of 95 audit keys (HICL:4703, HICL:33356) had an
  empty proposed_ingredients while their correct ingredient belongs to a
  pre-specified class; the estimator does not count them.
- **Reliability:** no second pass has been done; D5.7's optional intra-rater
  pass remains optional and unreported.
- **Note rewrite (2026-10-08):** the PI rewrote the reviewer_note of
  HICL:37442, HICL:5175, HICL:4703 and HICL:33356 to begin "Correct
  ingredient: <name>." so that arm 5 can read them (D6.3a).

## Brand alias list (2026-10-08)

The brand section of `mapping/review/search_terms.txt` was empty in every
committed version until 2026-10-08 (D5.14b). The list committed that day was
proposed by an AI assistant from drug references and approved alias by alias by
the PI, who added Zosyn under piperacillin. It contains single-product brands
plus Neosporin, a combination product; the rows of HICL:33356 (NEOSPORIN) are
topical and not systemic under D6.2 (D6.5). Ophthalmic (Acular, Acuvail) and oral-contrast (Gastrografin) aliases are
included; D6.2 decides by route whether their rows count.
