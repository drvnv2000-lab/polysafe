"""The public PolySafe page (GitHub Pages): ``docs/index.html``.

One self-contained page. Every number on it is read here from a committed results file
and embedded as JSON; the page computes nothing of its own. No individual patient's
predicted risk is shown (DEMO_SCOPE.md).

Usage:
    python -m polysafe.site    # writes docs/index.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from polysafe import figures

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
REPO = "https://github.com/drvnv2000-lab/polysafe"


def _f(v):
    return None if pd.isna(v) else round(float(v), 6)


def payload(results: Path = RESULTS) -> dict:
    flow = figures.flow_table((results / "cohort_flow.md").read_text(encoding="utf-8"))
    comp = pd.read_csv(results / "model_comparison.csv").set_index("spec")
    metrics = pd.read_csv(results / "model_metrics.csv")
    sens = pd.read_csv(results / "sensitivity.csv").set_index("analysis")
    sec = pd.read_csv(results / "secondary.csv")
    expl = pd.read_csv(results / "exploratory_d10.csv").set_index("analysis")
    share_md = (results / "exploratory_d10.md").read_text(encoding="utf-8")
    share_rows = [ln.strip("|").split("|") for ln in
                  share_md.split("## Exposed stays")[-1].splitlines() if ln.startswith("| ")][1:]
    integ = json.loads((results / "integrity.json").read_text(encoding="utf-8"))

    def auroc(spec, med):
        m = metrics[(metrics["spec"] == spec) & (metrics["medication"] == med)].iloc[0]
        return _f(m["auroc"])

    def row(r):
        return {"d": _f(r["dauroc"]), "lo": _f(r["lo"]), "hi": _f(r["hi"])}

    models = [{"key": k, "name": n, "a0": auroc(k, False), "a1": auroc(k, True)}
              | row(comp.loc[k]) for k, n in (("lr_reduced", "Logistic regression (primary)"),
                                             ("lgbm", "LightGBM (secondary)"),
                                             ("lr_full", "Full logistic regression"))]
    arms = [("ascertainment:generic_string:primary", "Drug-name text"),
            ("ascertainment:generic_brand:primary", "Name + brand names"),
            ("ascertainment:hicl_gtc:primary", "HICL codes (primary)"),
            ("ascertainment:hicl_gtc:named_only", "HICL, named rows only"),
            ("ascertainment:hicl_reviewed:primary", "HICL, review-corrected")]
    other = [("hospital_clusters", "Hospital-cluster CV", "No hospital or patient in two folds"),
             ("full_cohort", "All 186 hospitals", "868 stays, including sites without "
              "medication capture"),
             ("followup_48h", "At least 48 h follow-up", "314 stays observed for 2+ days after T"),
             ("baseline_c", "Pre-ICU creatinine baseline", "497 stays, baseline C")]
    devs = [{"id": d["id"], "tag": d["tag"], "title": d["title"],
             "utc": d["introduced_in"].get("utc", "") if isinstance(d["introduced_in"], dict)
             else ""} for d in integ["preregistration"]["deviations"]]
    tests = integ.get("tests", {})
    return {
        "flow": [{"step": r.step, "removed": None if pd.isna(r.removed) else int(r.removed),
                  "stays": int(r.stays), "patients": int(r.patients), "events": int(r.events),
                  "reason": r.reason} for r in flow.itertuples()],
        "models": models,
        "mde": _f(comp.loc["lr_reduced", "mde_bca"]),
        "arms": [{"name": n} | row(sens.loc[k]) for k, n in arms],
        "hidden": [{"drug": c[0].strip(), "exposed": int(c[1]), "only": int(c[2])}
                   for c in share_rows],
        "other": [{"name": n, "note": note, "n": int(sens.loc[k, "n"]),
                   "events": int(sens.loc[k, "events"])} | row(sens.loc[k])
                  for k, n, note in other],
        "expl": {k: row(expl.loc[k]) for k in ("E1", "E2")},
        "secondary": [{"agent": r.agent, "role": r.role, "exposed": int(r.exposed),
                       "events": int(r.events_exposed), "crude": _f(r.crude_or),
                       "or": _f(r.adjusted_or), "lo": _f(r.adjusted_lo), "hi": _f(r.adjusted_hi)}
                      for r in sec.itertuples()],
        "devs": devs,
        "results_utc": "2026-10-09T09:54:50+00:00",  # a713fd2, the section 8 results commit
        "tests": tests.get("passed") if isinstance(tests, dict) else None,
        "repo": REPO,
    }


def build(results: Path = RESULTS) -> str:
    data = json.dumps(payload(results), separators=(",", ":"))
    return TEMPLATE.replace("__DATA__", data.replace("</", "<\\/"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the public PolySafe page.")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "index.html")
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(), encoding="utf-8")
    (args.out.parent / ".nojekyll").write_text("", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


TEMPLATE = (Path(__file__).with_name("site_template.html")).read_text(encoding="utf-8") \
    if (Path(__file__).with_name("site_template.html")).exists() else "__DATA__"

if __name__ == "__main__":
    raise SystemExit(main())
