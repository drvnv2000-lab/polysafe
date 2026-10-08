import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from polysafe import integrity, lab, review

ROOT = integrity.ROOT
ARTIFACTS = ["PREREGISTRATION.md", "results/cohort_flow.md", "results/aki_labels.md",
             "results/labels_A.csv", "results/power_simulation.md",
             "results/features_clinical.md", "mapping/review/REVIEW_PACKET.md",
             "mapping/review/audit_design.json", "mapping/review/tier1_agents.csv",
             "mapping/review/tier2_classes.csv", "mapping/review/tier3_screen.csv",
             "mapping/review/audit_head.csv", "mapping/review/audit_tail.csv"]
FAKE_TESTS = {"collected": 3, "passed": 3, "failed": 0, "skipped": 0, "junit": "x",
              "_cases": {n: "passed" for _, _, nodes in integrity.CONTROLS for n in nodes}}


@pytest.fixture
def fixture_root(tmp_path):
    for a in ARTIFACTS:
        (tmp_path / a).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / a, tmp_path / a)
    return tmp_path


REVIEW_FILES = ("tier1_agents.csv", "tier2_classes.csv", "tier3_screen.csv", "audit_head.csv",
                "audit_tail.csv")


def _blank_review(root):
    """Reset the copied review files to the pre-review state these tests describe."""
    for name in REVIEW_FILES:
        f = root / "mapping/review" / name
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        df[[c for c in review.DECISION_COLUMNS if c in df]] = ""
        df.to_csv(f, index=False)


def _report(root):
    return integrity.build_integrity_report(root, tests=FAKE_TESTS,
                                            ruff={"clean": True, "summary": ""}, use_git=False)


# ---------------------------------------------------------------- parsers on real artifacts


def test_preregistration_parser():
    p = integrity.parse_preregistration(ROOT)
    assert len(p["sections"]) == 11 and p["sections"][0].startswith("1. Research questions")
    ids = [d["id"] for d in p["deviations"]]
    assert ids[0] == "D1.1" and "D5.15" in ids and len(ids) == len(set(ids))
    tags = {d["id"]: d["tag"] for d in p["deviations"]}
    assert tags["D1.2"] == "correction" and tags["D5.13"] == "correction"
    assert set(tags.values()) <= {"clarification", "correction", "protocol addition",
                                  "substantive change"}


def test_every_correction_verifies_against_the_text():
    rows = integrity.corrections(integrity.parse_preregistration(ROOT))
    assert rows and all(r["status"] == "verified" for r in rows), \
        [r["figure"] for r in rows if r["status"] != "verified"]


def test_review_packet_parser():
    p = integrity.parse_review_packet(ROOT)
    assert p["tiers"][1]["keys"] == 33 and p["unmappable_rows"] == 1646
    assert p["seed"] == 20261004 and len(p["search_terms_sha"]) == 40


def test_cohort_parser():
    c = integrity.parse_cohort(ROOT, integrity.parse_preregistration(ROOT))
    assert c["primary"] == {"stays": 671, "patients": 586, "events": 78, "prevalence_pct": 11.6}
    assert c["all_sites"]["stays"] == 868 and c["primary_prevalence_ci_pct"] == [9.4, 14.3]
    assert c["epv_primary_lr"]["epv"] == 4.6 and len(c["power_mde"]["mde"]) == 4


def test_junit_parser_and_control_status(tmp_path):
    xml = tmp_path / "j.xml"
    xml.write_text('<testsuites><testsuite tests="3" failures="1" errors="0" skipped="0">'
                   '<testcase classname="tests.test_a" name="test_ok"/>'
                   '<testcase classname="tests.test_a" name="test_bad"><failure/></testcase>'
                   '<testcase classname="tests.test_a" name="test_p[1]"/>'
                   "</testsuite></testsuites>")
    j = integrity.parse_junit(xml)
    assert (j["collected"], j["passed"], j["failed"]) == (3, 2, 1)
    assert j["_cases"]["tests/test_a.py::test_bad"] == "failed"
    assert j["_cases"]["tests/test_a.py::test_p"] == "passed"  # parametrised ids collapse
    rows = {r["control"]: r["status"] for r in integrity.control_rows(j)}
    assert rows["Search terms committed before screening"] == "asserted, not tested"
    assert rows["APACHE excluded"] == "test not found"  # not in this run: never a tick


def test_a_failing_enforcing_test_is_shown_as_failing():
    t = dict(FAKE_TESTS, _cases={**FAKE_TESTS["_cases"],
                                 "tests/test_features.py::test_shape_and_no_apache": "failed"})
    rows = {r["control"]: r["status"] for r in integrity.control_rows(t)}
    assert rows["APACHE excluded"] == "failing"


# ---------------------------------------------------------------- not_available behaviour


@pytest.mark.parametrize("artifact, field", [
    ("results/cohort_flow.md", ("cohort", "primary")),
    ("results/aki_labels.md", ("cohort", "primary_prevalence_ci_pct")),
    ("results/power_simulation.md", ("cohort", "power_mde")),
    ("mapping/review/REVIEW_PACKET.md", ("review_packet",)),
    ("mapping/review/tier1_agents.csv", ("review_decisions",)),
    ("mapping/review/audit_design.json", ("review_packet", "seed")),
])
def test_missing_artifact_is_not_available_never_a_default(fixture_root, artifact, field):
    (fixture_root / artifact).unlink()
    value = _report(fixture_root)
    for k in field:
        value = value[k]
    assert value["status"] == "not_available"
    assert Path(artifact).name in value["path"]


def test_missing_preregistration_is_not_available(fixture_root):
    (fixture_root / "PREREGISTRATION.md").unlink()
    r = _report(fixture_root)
    assert r["preregistration"]["status"] == "not_available"
    assert r["cohort"]["epv_primary_lr"]["status"] == "not_available"
    assert all(c["status"] == "not_available" for c in r["corrections"])


def test_review_not_started_is_not_zero_unsure(fixture_root):
    _blank_review(fixture_root)
    r = _report(fixture_root)
    assert r["review_decisions"]["status"] == "not_started"
    assert "unsure" not in r["review_decisions"]
    for name in REVIEW_FILES:
        f = fixture_root / "mapping/review" / name
        pd.read_csv(f, dtype=str, keep_default_na=False).assign(decision="keep").to_csv(
            f, index=False)
    done = _report(fixture_root)["review_decisions"]
    assert done["status"] == "complete" and done["unsure"] == 0


def test_timeline_states_not_fitted_plainly(fixture_root):
    _blank_review(fixture_root)
    stages = {t["stage"]: t["status"] for t in _report(fixture_root)["timeline"]}
    assert stages["Model fitting"] == "not fitted"
    assert stages["Medication QC"] == "packet built; review not started"
    assert stages["Cohort"] == "done"


# ---------------------------------------------------------------- the shell


def test_nav_is_derived_from_artifacts(fixture_root):
    assert lab.available_sections(fixture_root) == []  # no integrity.json yet
    assert "<section" not in lab.render(fixture_root)
    _report(fixture_root)
    assert [s.title for s in lab.available_sections(fixture_root)] == ["Overview",
                                                                       "Research Integrity"]


def test_section_without_its_artifact_is_not_rendered(fixture_root):
    _report(fixture_root)
    ghost = lab.Section("ghost", "Measurement Lab", "results/no_such_artifact.json",
                        lambda r: "SHOULD NOT APPEAR")
    page = lab.render(fixture_root, [*lab.SECTIONS, ghost])
    assert "Measurement Lab" not in page and "SHOULD NOT APPEAR" not in page
    assert 'href="#integrity"' in page


def test_page_has_no_placeholders_and_shows_gaps_plainly(fixture_root):
    _blank_review(fixture_root)
    _report(fixture_root)
    page = lab.render(fixture_root).lower()
    for banned in ("coming soon", "placeholder", "lorem", "todo"):
        assert banned not in page
    assert "not fitted" in page and "asserted, not tested" in page
    assert "review not started" in page


def test_not_available_renders_as_text_not_zero(fixture_root):
    (fixture_root / "results/power_simulation.md").unlink()
    _report(fixture_root)
    page = lab.render(fixture_root)
    assert "not available (looked for" in page and "power_simulation.md" in page


def test_report_is_valid_json_without_private_fields(fixture_root):
    _report(fixture_root)
    text = (fixture_root / "results/integrity.json").read_text(encoding="utf-8")
    assert "_text" not in text and "_cases" not in text and "_entry_texts" not in text
    json.loads(text)
