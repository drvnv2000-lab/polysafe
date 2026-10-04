"""Every cited D-entry id and section number must exist in the COMMITTED protocol.

Deliberately simple (PREREGISTRATION D6.0e): existence of ids and sections, not
quotation matching. The protocol is read from git HEAD, so a citation to an
entry that has not been committed fails. tests/ is not scanned: test fixtures
use synthetic ids on purpose.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ["src/polysafe/*.py", "PREREGISTRATION.md", "HANDOFF.md", "DEMO_SCOPE.md",
           "DATA_REPORT.md", "mapping/MAPPING_REPORT.md", "mapping/review/REVIEW_PACKET.md",
           "mapping/review/search_terms.txt", "results/*.md"]
ENTRY = re.compile(r"\bD\d+\.\d+[a-z]?\b")
SERIES = re.compile(r"\bD\d+\b(?![.\d])")
NUMBER = r"\d+(?:\.\d+)?"
SECTION = re.compile(rf"(?:[Ss]ections?|§)\s*({NUMBER}(?:\s*(?:-|–|,|and)\s*{NUMBER})*)")


def committed_protocol() -> str:
    return subprocess.run(["git", "show", "HEAD:PREREGISTRATION.md"], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8",
                          check=True).stdout.replace("\r\n", "\n")


def defined(protocol: str) -> tuple[set, set, set]:
    ids = set(re.findall(r"^\*\*(D\d+\.\d+[a-z]?)\b", protocol, re.M))
    series = set(re.findall(r"^### (D\d+)\b", protocol, re.M))
    sections = (set(re.findall(r"^## (\d+)\.", protocol, re.M))
                | set(re.findall(r"^### (\d+\.\d+) ", protocol, re.M)))
    return ids, series, sections


def citations(text: str):
    for n, line in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
        # An entry's own defining heading ("**D6.0f [...]") is not a citation of it.
        defining = re.match(r"\*\*(D\d+\.\d+[a-z]?)\b", line)
        for m in ENTRY.finditer(line):
            if defining and m.start() == 2:
                continue
            yield n, "entry", m.group(0)
        for m in SERIES.finditer(line):
            yield n, "series", m.group(0)
        for m in SECTION.finditer(line):
            for num in re.findall(NUMBER, m.group(1)):
                yield n, "section", num


def unresolved(files, protocol: str, root: Path = ROOT) -> list[str]:
    ids, series, sections = defined(protocol)
    known = {"entry": ids, "series": series, "section": sections}
    return [f"{f.relative_to(root).as_posix()}:{n} cites {kind} {ref}"
            for f in files for n, kind, ref in citations(f.read_text(encoding="utf-8"))
            if ref not in known[kind]]


def test_every_citation_resolves_in_the_committed_protocol():
    files = sorted({p for pattern in SCANNED for p in ROOT.glob(pattern)})
    assert files, "nothing scanned"
    bad = unresolved(files, committed_protocol())
    assert not bad, "unresolved protocol citations:\n" + "\n".join(bad)


def test_the_check_catches_a_missing_entry_series_and_section(tmp_path):
    protocol = "## 1. One\n\n### D1 — date\n\n**D1.1 — x.** text\n"
    f = tmp_path / "x.py"
    f.write_text("# see D1.1, D1.2, D1, D7 and section 1, sections 4 and 1, §9.1\n",
                 encoding="utf-8")
    assert sorted(unresolved([f], protocol, root=tmp_path)) == sorted([
        "x.py:1 cites entry D1.2", "x.py:1 cites series D7",
        "x.py:1 cites section 4", "x.py:1 cites section 9.1"])


def test_a_new_entrys_own_heading_is_not_a_citation_but_its_references_are(tmp_path):
    protocol = "## 1. One\n\n### D1 — date\n\n**D1.1 — x.** text\n"
    f = tmp_path / "PREREGISTRATION.md"
    f.write_text("**D1.2 [correction] New.** Refers to D1.1 and D1.9.\n", encoding="utf-8")
    assert unresolved([f], protocol, root=tmp_path) == ["PREREGISTRATION.md:1 cites entry D1.9"]
