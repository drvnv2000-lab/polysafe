"""Research Integrity report: ``results/integrity.json`` built ONLY from artifacts.

Sources: PREREGISTRATION.md, git history, a real pytest run, a real ruff run,
the HICL review packet and any filled review CSVs, and the cohort/label/power
reports. Any value that cannot be found is ``{"status": "not_available",
"path": <where it looked>}``; nothing is ever defaulted to 0 or "".

Usage:
    python -m polysafe.integrity     # runs pytest + ruff, writes results/integrity.json
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREREG = "PREREGISTRATION.md"
JUNIT = "results/integrity_pytest.xml"

# Each control: the entry that established it and the tests that enforce it.
# A control with no test is reported as "asserted, not tested", never as passing.
CONTROLS = [
    ("T = 24 h feature cutoff (offset <= T)", "section 3",
     ["tests/test_features.py::test_nothing_after_T_can_change_a_feature",
      "tests/test_features.py::test_value_at_T_counts_and_one_minute_later_does_not",
      "tests/test_cohort.py::test_creatinine_after_T_cannot_change_the_cohort"]),
    ("APACHE excluded", "D3.1", ["tests/test_features.py::test_shape_and_no_apache"]),
    ("Patient-level grouping on uniquepid", "section 7",
     ["tests/test_power.py::test_no_patient_spans_two_folds_on_the_real_cohort"]),
    ("Repeated stratified grouped CV", "section 7",
     ["tests/test_power.py::test_no_patient_spans_two_folds_on_the_real_cohort",
      "tests/test_power.py::test_folds_are_stratified"]),
    ("String-level wrong_strings exclusion", "D5.8, D5.11",
     ["tests/test_review.py::test_string_level_filtering_drops_only_wrong_strings",
      "tests/test_review.py::test_end_to_end_decision_to_stay_level_exposure"]),
    ("Unnamed-row sensitivity", "D5.12",
     ["tests/test_review.py::test_named_only_sensitivity_drops_rows_without_a_string"]),
    ("Unsure handling", "D5.13",
     ["tests/test_review.py::"
      "test_unsure_only_in_the_sensitivity_variant_and_wrong_strings_still_excluded"]),
    ("Serum creatinine only (urinary excluded)", "section 4",
     ["tests/test_cohort.py::test_serum_creatinine_excludes_urinary"]),
    ("Floating-point tolerance on 0.3 mg/dL", "section 4",
     ["tests/test_cohort.py::test_rise_rule"]),
    ("Outcome window censored at ICU discharge", "section 3, D2.2",
     ["tests/test_labels.py::test_creatinine_after_icu_discharge_is_censored",
      "tests/test_labels.py::test_every_onset_is_inside_the_window"]),
    ("Exclusions: <24 h, ESRD, AKI by T, RRT by T", "section 5",
     ["tests/test_cohort.py::test_final_cohort_respects_every_exclusion"]),
    ("Ethnicity not a predictor", "D3.4",
     ["tests/test_features.py::test_ethnicity_is_not_a_feature"]),
    ("Pre-ICU lookback capped at 7 days", "D3.7",
     ["tests/test_features.py::test_pre_icu_lookback_is_seven_days"]),
    ("Primary LR: 14 variables / 17 parameters, LR exclusions", "D4.2-D4.4",
     ["tests/test_features.py::test_primary_lr_spec_matches_d4"]),
    ("No outcome or prediction column in review CSVs", "D5.5",
     ["tests/test_review.py::test_no_outcome_or_prediction_column_in_any_review_csv"]),
    ("Audit draw reproducible from seed 20261004", "D5.2",
     ["tests/test_review.py::test_audit_draw_is_reproducible"]),
    ("Whole-token search matching", "D5.10",
     ["tests/test_review.py::test_token_match_rejects_substring_and_accepts_whole_token"]),
    ("Site-level restriction; medication_data_available never a predictor", "section 6", []),
    ("Search terms committed before screening", "D5.6", []),
    ("BCa bootstrap intervals for dAUROC", "D2.1", []),
    ("Pre-registration is append-only after modelling begins", "section 11", []),
]

# Figures an entry corrected. Each string must appear verbatim in the file, in the
# named place, or the correction is reported not_available rather than asserted.
CORRECTIONS = [
    {"figure": "Expected AKI prevalence", "original": "22–25% prevalence", "original_in": "§9",
     "corrected_by": "D1.2", "corrected_to": "16.1%",
     "current": "measured prevalence 11.6%", "current_in": "§9.1"},
    {"figure": "Analysis cohort size", "original": "n = 1,037", "original_in": "§9",
     "corrected_by": "D1.3", "corrected_to": "**868 stays**",
     "current": "**671** are at medication-recording sites", "current_in": "D1.3"},
    {"figure": "Events per variable, primary LR", "original": "78/48 = 1.6", "original_in": "D3.5",
     "corrected_by": "D4.4", "corrected_to": "EPV = 78/17 = 4.6",
     "current": "EPV = 78/17 = 4.6", "current_in": "D4.4"},
    {"figure": "Basis of the 40% LR missingness threshold", "original": "across all 868 stays",
     "original_in": "D3.6", "corrected_by": "D4.3", "corrected_to": "applied to the primary",
     "current": "LR exclusions: lactate, pH, phosphate, INR,", "current_in": "D4.3"},
    {"figure": "APACHE severity covariate", "original": "APACHE", "original_in": "§8",
     "corrected_by": "D3.1", "corrected_to": "APACHE severity excluded",
     "current": "APACHE severity excluded", "current_in": "D3.1"},
    {"figure": "Rows of an unsure key in the sensitivity variant",
     "original": None, "original_in": "analyst's report (not in the pre-registration)",
     "corrected_by": "D5.13", "corrected_to": "EXCLUDING strings listed in",
     "current": "EXCLUDING strings listed in", "current_in": "D5.13"},
]

STAGES = [
    ("Pre-registration", [PREREG]),
    ("Cohort", ["results/cohort_flow.md"]),
    ("Outcome", ["results/aki_labels.md", "results/labels_A.csv"]),
    ("Leakage audit", [JUNIT]),
    ("Feature specification", ["results/features_clinical.md"]),
    ("Medication QC", ["mapping/review/REVIEW_PACKET.md"]),
    ("Model fitting", ["results/models"]),
    ("Validation", ["results/validation.json"]),
    ("Robustness", ["results/robustness.json"]),
]
LEAKAGE_TESTS = CONTROLS[0][2] + CONTROLS[1][2]


def na(path) -> dict:
    return {"status": "not_available", "path": str(path)}


def available(v) -> bool:
    return not (isinstance(v, dict) and v.get("status") == "not_available")


def _num(s: str) -> float:
    return float(s.replace(",", ""))


# ---------------------------------------------------------------- pre-registration


def _entry_texts(text: str) -> dict[str, str]:
    """{'D1.1': its text, ...} for every bold D-entry."""
    marks = list(re.finditer(r"^\*\*(D\d+\.\d+)\b", text, re.M))
    out = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        nxt = re.search(r"^#{2,3} ", text[m.end():end], re.M)
        out[m.group(1)] = text[m.start():m.end() + nxt.start() if nxt else end]
    return out


def _section_text(text: str, label: str) -> str | None:
    """'§9' -> section 9 body (stops at the next '## '); '§9.1' -> the 9.1 appendix."""
    num = label.lstrip("§")
    pat = rf"^##+ {re.escape(num)}[. ]" if "." not in num else rf"^### {re.escape(num)} "
    m = re.search(pat, text, re.M)
    if not m:
        return None
    rest = text[m.end():]
    stop = re.search(r"^## ", rest, re.M)
    return rest[:stop.start()] if stop else rest


def parse_preregistration(root: Path, git=None) -> dict:
    path = root / PREREG
    if not path.exists():
        return na(path)
    text = path.read_text(encoding="utf-8")
    sections = re.findall(r"^## (.+)$", text, re.M)
    tags = dict(re.findall(r"\| (D\d+\.\d+) \| (\[[a-z ]+\])", text))
    entries, date = [], None
    for line in text.splitlines():
        h = re.match(r"^### D\d+.*? — (\d{4}-\d{2}-\d{2})", line)
        if h:
            date = h.group(1)
        m = re.match(r"^\*\*(D\d+\.\d+)\b\s*(\[[a-z ]+\])?\s*(?:—\s*)?(.*)", line)
        if m:
            eid = m.group(1)
            tag = m.group(2) or tags.get(eid)
            entries.append({"id": eid, "tag": tag.strip("[]") if tag else na(f"{path} tag"),
                            "date": date or na(f"{path} heading date"),
                            "title": re.sub(r"\*\*.*$", "", m.group(3)).strip(" .") or eid,
                            "introduced_in": git(f"**{eid}") if git else na("git")})
    return {"path": PREREG, "sections": sections, "deviations": entries,
            "_entry_texts": _entry_texts(text), "_text": text}


def corrections(prereg: dict) -> list[dict]:
    if not available(prereg):
        return [dict(c, status="not_available", path=PREREG) for c in CORRECTIONS]
    text, entries = prereg["_text"], prereg["_entry_texts"]
    out = []
    for c in CORRECTIONS:
        def where(label):
            return entries.get(label) if label.startswith("D") else _section_text(text, label)
        checks = [c["corrected_to"] in (entries.get(c["corrected_by"]) or ""),
                  c["current"] in (where(c["current_in"]) or "")]
        if c["original"] is not None:
            checks.append(c["original"] in (where(c["original_in"]) or ""))
        out.append(dict(c, status="verified" if all(checks) else "not_available",
                        **({} if all(checks) else {"path": PREREG})))
    return out


# ---------------------------------------------------------------- git, pytest, ruff


def git_runner(root: Path):
    def run(*args) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                                  check=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def introduced(marker: str):
        out = run("log", "--reverse", "--format=%H%x1f%ct", "-S", marker, "--", PREREG)
        if not out or not out.strip():
            return na(f"git log -S {marker!r} -- {PREREG}")
        sha, ts = out.splitlines()[0].split("\x1f")
        return {"sha": sha, "utc": datetime.fromtimestamp(int(ts), UTC).isoformat()}

    def commits():
        out = run("log", "--format=%H%x1f%ct%x1f%s", "--", PREREG, "mapping/")
        if out is None:
            return na(f"git log -- {PREREG} mapping/ (in {root})")
        rows = [ln.split("\x1f") for ln in out.splitlines() if ln]
        return [{"sha": s, "utc": datetime.fromtimestamp(int(t), UTC).isoformat(), "subject": m}
                for s, t, m in rows]

    return introduced, commits


def run_pytest(root: Path) -> Path:
    junit = root / JUNIT
    junit.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    f"--junitxml={junit}"], cwd=root, capture_output=True, text=True)
    return junit


def parse_junit(path: Path) -> dict:
    if not path.exists():
        return na(path)
    suite = ET.parse(path).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    cases = {}
    for tc in suite.iter("testcase"):
        node = tc.get("classname", "").replace(".", "/") + ".py::" + tc.get("name", "")
        node = re.sub(r"\[.*\]$", "", node)
        failed = tc.find("failure") is not None or tc.find("error") is not None
        skipped = tc.find("skipped") is not None
        state = "failed" if failed else "skipped" if skipped else "passed"
        prev = cases.get(node)
        cases[node] = "failed" if "failed" in (prev, state) else (prev or state)
    n = int(suite.get("tests", 0))
    bad = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    skip = int(suite.get("skipped", 0))
    return {"collected": n, "passed": n - bad - skip, "failed": bad, "skipped": skip,
            "junit": str(path.relative_to(path.parents[1])) if len(path.parents) > 1 else
            str(path), "_cases": cases}


def run_ruff(root: Path) -> dict:
    r = subprocess.run([sys.executable, "-m", "ruff", "check", "."], cwd=root,
                       capture_output=True, text=True)
    return {"clean": r.returncode == 0, "summary": (r.stdout.strip().splitlines() or [""])[-1]}


def control_rows(tests: dict) -> list[dict]:
    cases = tests.get("_cases", {}) if available(tests) else {}
    rows = []
    for name, entry, nodes in CONTROLS:
        if not nodes:
            status = "asserted, not tested"
        elif not available(tests):
            status = "not_available"
        else:
            states = [cases.get(n) for n in nodes]
            status = ("test not found" if any(s is None for s in states) else
                      "failing" if "failed" in states else
                      "skipped" if "skipped" in states else "enforced (passed)")
        rows.append({"control": name, "established_by": entry, "tests": nodes, "status": status})
    return rows


# ---------------------------------------------------------------- review, cohort, power


def parse_review_packet(root: Path) -> dict:
    path = root / "mapping/review/REVIEW_PACKET.md"
    design_path = root / "mapping/review/audit_design.json"
    if not path.exists():
        return na(path)
    text = path.read_text(encoding="utf-8")
    tiers = {int(t): {"keys": int(_num(k)), "rows": int(_num(r))} for t, k, r in
             re.findall(r"^\| ([1-4]) \| `[^`]+` \| ([\d,]+) \| ([\d,]+) \|", text, re.M)}
    orphan = re.search(r"The remaining ([\d,]+) rows have neither", text)
    sha = re.search(r"at commit `([0-9a-f]{40})`", text)
    design = json.loads(design_path.read_text()) if design_path.exists() else None
    pick = (lambda k: design[k]) if design else (lambda k: na(design_path))
    return {"tiers": tiers or na(f"{path} tier table"),
            "audit_head_keys": pick("head_keys"), "audit_tail_keys": pick("n"),
            "audit_tail_mode": pick("tail_mode"), "seed": pick("seed"),
            "unmappable_rows": int(_num(orphan.group(1))) if orphan else na(f"{path} rows line"),
            "search_terms_sha": sha.group(1) if sha else na(f"{path} commit line")}


def parse_review_decisions(root: Path) -> dict:
    """'not_started' when no decision is filled; never reported as zero counts."""
    d = root / "mapping/review"
    files = ["tier1_agents.csv", "tier2_classes.csv", "tier3_screen.csv", "audit_head.csv",
             "audit_tail.csv"]
    missing = [f for f in files if not (d / f).exists()]
    if missing:
        return na(d / missing[0])
    frames = [pd.read_csv(d / f, dtype=str, keep_default_na=False) for f in files]
    allrows = pd.concat(frames, ignore_index=True)
    filled = allrows[allrows["decision"].str.strip() != ""]
    if filled.empty:
        return {"status": "not_started", "keys_to_review": int(len(allrows))}
    counts = filled["decision"].value_counts().to_dict()
    return {"status": "complete" if len(filled) == len(allrows) else "in_progress",
            "keys_to_review": int(len(allrows)), "adjudicated": int(len(filled)),
            "keep": int(counts.get("keep", 0)), "reject": int(counts.get("reject", 0)),
            "unsure": int(counts.get("unsure", 0))}


def parse_cohort(root: Path, prereg: dict) -> dict:
    flow, labels, power = (root / "results/cohort_flow.md", root / "results/aki_labels.md",
                           root / "results/power_simulation.md")
    out = {}
    if flow.exists():
        a = flow.read_text(encoding="utf-8").split("## Baseline C")[0]
        m = re.search(r"\| \*\*Primary estimand[^|]*\| [\d,]+ \| \*\*([\d,]+)\*\* \| ([\d,]+) "
                      r"\| ([\d,]+) \| ([\d.]+)% \|", a)
        f = re.search(r"\| No observable outcome \| [\d,]+ \| ([\d,]+) \| ([\d,]+) \| ([\d,]+) "
                      r"\| ([\d.]+)% \|", a)
        out["primary"] = ({"stays": int(_num(m[1])), "patients": int(_num(m[2])),
                           "events": int(_num(m[3])), "prevalence_pct": float(m[4])}
                          if m else na(f"{flow} primary row"))
        out["all_sites"] = ({"stays": int(_num(f[1])), "patients": int(_num(f[2])),
                             "events": int(_num(f[3])), "prevalence_pct": float(f[4])}
                            if f else na(f"{flow} final row"))
    else:
        out["primary"] = out["all_sites"] = na(flow)
    if labels.exists():
        m = re.search(r"^\| A \| .*?\| \d+ / \d+ = \*\*([\d.]+)%\*\* \(([\d.]+)-([\d.]+)%\) \|",
                      labels.read_text(encoding="utf-8"), re.M)
        out["primary_prevalence_ci_pct"] = ([float(m[2]), float(m[3])] if m
                                            else na(f"{labels} baseline A row"))
    else:
        out["primary_prevalence_ci_pct"] = na(labels)
    # The latest statement wins: deviations are append-only, so a revision comes later.
    epvs = re.findall(r"EPV = (\d+)/(\d+) = ([\d.]+)", prereg.get("_text", "")) \
        if available(prereg) else []
    epv = epvs[-1] if epvs else None
    out["epv_primary_lr"] = ({"events": int(epv[0]), "parameters": int(epv[1]),
                              "epv": float(epv[2]), "source": "PREREGISTRATION D4.4"}
                             if epv else na(f"{root / PREREG} EPV statement"))
    if power.exists():
        m = re.search(r"detects dAUROC >= ([\d. /]+) with 80% probability \(rho = ([\d. /]+)\)",
                      power.read_text(encoding="utf-8"))
        out["power_mde"] = ({"mde": [float(x) for x in m[1].split(" / ")],
                             "rho": [float(x) for x in m[2].split(" / ")]}
                            if m else na(f"{power} statement"))
    else:
        out["power_mde"] = na(power)
    return out


def timeline(root: Path, tests: dict, decisions: dict) -> list[dict]:
    cases = tests.get("_cases", {}) if available(tests) else {}
    rows = []
    for stage, paths in STAGES:
        present = all((root / p).exists() for p in paths)
        if stage == "Leakage audit":
            states = [cases.get(n) for n in LEAKAGE_TESTS]
            status = ("done" if present and all(s == "passed" for s in states) else
                      "failing" if "failed" in states else "not run")
        elif stage == "Medication QC" and present:
            s = decisions.get("status") if available(decisions) else "not_available"
            status = {"not_started": "packet built; review not started",
                      "in_progress": "review in progress", "complete": "review complete"
                      }.get(s, "packet built; review status not available")
        elif stage == "Model fitting":
            status = "fitted" if present else "not fitted"
        else:
            status = "done" if present else "not started"
        rows.append({"stage": stage, "status": status, "artifacts": paths})
    return rows


# ---------------------------------------------------------------- build


def build_integrity_report(root: Path = ROOT, out: Path | None = None, tests: dict | None = None,
                           ruff: dict | None = None, use_git: bool = True) -> dict:
    """Assemble the report. ``tests``/``ruff`` default to REAL runs; tests inject fakes."""
    root = Path(root)
    introduced, commits = git_runner(root) if use_git else (None, None)
    prereg = parse_preregistration(root, introduced)
    tests = tests if tests is not None else parse_junit(run_pytest(root))
    ruff = ruff if ruff is not None else run_ruff(root)
    decisions = parse_review_decisions(root)
    report = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "preregistration": ({k: v for k, v in prereg.items() if not k.startswith("_")}
                            if available(prereg) else prereg),
        "corrections": corrections(prereg),
        "commits": commits() if commits else na("git"),
        "tests": {k: v for k, v in tests.items() if not k.startswith("_")},
        "ruff": ruff,
        "controls": control_rows(tests),
        "review_packet": parse_review_packet(root),
        "review_decisions": decisions,
        "cohort": parse_cohort(root, prereg),
        "timeline": timeline(root, tests, decisions),
    }
    out = Path(out or root / "results/integrity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main() -> int:
    r = build_integrity_report()
    print(json.dumps({"tests": r["tests"], "ruff": r["ruff"],
                      "untested_controls": [c["control"] for c in r["controls"]
                                            if c["status"] != "enforced (passed)"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
