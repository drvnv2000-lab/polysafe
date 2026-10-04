# PolySafe Research Lab — demo scope (locked 2026-10-04)

The demo is one app with four capabilities. Each appears only once the
artifact behind it exists on disk; nothing is shown as a placeholder.

| Capability | When it appears | Backed by |
|---|---|---|
| **Research Integrity** | now | `results/integrity.json` |
| **Measurement Lab** | after model fitting | the ascertainment-strategy comparison (PREREGISTRATION D5.9) |
| **Conclusion Robustness** | after model fitting | sensitivity analyses, including a distribution of delta-predicted-risk between the clinical-only and clinical+medication models |
| **Failure Injection** | after model fitting | deliberate perturbations of inputs and labels |

## Dropped

- **Provenance graph.** Redundant with the Measurement Lab, which already shows
  how each exposure count was built from review decisions.
- **Per-patient prediction card.** Replaced by the delta-predicted-risk
  distribution view inside Conclusion Robustness. Individual risk estimates from
  a model with 78 events and EPV 4.6 (PREREGISTRATION D4.4) are not defensible,
  so the demo does not present any single patient's predicted risk.

## Shell rule

Navigation is derived, not declared: a section is in the navigation only when
its backing artifact exists. No "coming soon", no greyed-out cards, no empty
charts. "Not fitted" is stated plainly where the pipeline has not reached a
stage.
