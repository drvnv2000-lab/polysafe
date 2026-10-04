"""PREREGISTRATION.md is append-only at DOCUMENT level (PREREGISTRATION D6.0b).

New content may only be added at the end of the file. Inserting text inside an
earlier entry's block fails, as do deleting or modifying a line. Both sides of every
comparison are read as git blobs (`git show <sha>:PREREGISTRATION.md`), never from the
working tree, so checkout line-ending translation cannot affect the result.

BASELINE is the later of the two one-time modifications D6.0b discloses (the
line-ending commit); the property is checked for every commit after it. The
pre-baseline history is recorded here, not relaxed away. Under this rule it
contains:
  29a66d8  deleted the placeholder line "*(none yet)*" when D1 was added
  46c9f9c  inserted the section 9.1 appendix inside section 9, as section 9 instructed
  d972c0c  inserted a sentence inside D6.0's block
  50792b0  the revert of d972c0c (disclosed deletion)
  9635d73  BASELINE: .gitattributes + renormalisation; amended D6.0b's last sentence
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = "PREREGISTRATION.md"
BASELINE = "9635d732055cd925548f4f16ee066c4d64d4c015"  # line-ending commit (D6.0b)
PRE_BASELINE_CHANGES = ["29a66d8", "46c9f9c", "d972c0c", "50792b0", BASELINE]


def appended_only(old: str, new: str) -> str | None:
    """None if ``new`` is ``old`` with text added at the end only, else the reason."""
    old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
    base = old.rstrip("\n")
    if new.startswith(base) and (len(new) == len(base) or new[len(base)] == "\n"):
        return None
    i = next((k for k in range(min(len(base), len(new))) if base[k] != new[k]),
             min(len(base), len(new)))
    return f"existing text changed at line {base[:i].count(chr(10)) + 1} of the earlier version"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def _version(rev: str) -> str:
    return _git("show", f"{rev}:{DOC}")


# ---------------------------------------------------------------- the rule itself


def test_rule_rejects_an_insertion_inside_an_earlier_block():
    old = "### D1\n\n**D1.1** first.\n\n**D1.2** second.\n"
    new = "### D1\n\n**D1.1** first.\nInserted sentence.\n\n**D1.2** second.\n"
    assert appended_only(old, new) is not None


def test_rule_rejects_deletion_and_modification():
    old = "a\nb\nc\n"
    assert appended_only(old, "a\nc\n") is not None
    assert appended_only(old, "a\nB\nc\n") is not None
    assert appended_only(old, "a\nb\nc but longer\n") is not None  # last line extended


def test_rule_accepts_appending_at_the_end_only():
    old = "a\nb\nc\n"
    assert appended_only(old, old) is None
    assert appended_only(old, old + "\n**D9.9** new entry.\n") is None
    assert appended_only("a\nb\nc", "a\nb\nc\n\nd\n") is None  # missing final newline
    assert appended_only(old.replace("\n", "\r\n"), old + "d\n") is None  # CRLF vs LF


def test_reproduces_the_known_insertion_at_d972c0c():
    assert appended_only(_version("d972c0c^"), _version("d972c0c")) is not None


# ---------------------------------------------------------------- the history


def test_baseline_is_the_line_ending_commit():
    _git("cat-file", "-e", f"{BASELINE}^{{commit}}")  # must exist; never skipped
    attrs = _git("show", f"{BASELINE}:.gitattributes")
    assert "*.md text eol=lf" in attrs


def test_every_commit_since_baseline_only_appends():
    commits = _git("log", "--reverse", "--format=%H", f"{BASELINE}..HEAD", "--", DOC).split()
    bad = {c[:7]: why for c in commits
           if (why := appended_only(_version(f"{c}^"), _version(c))) is not None}
    assert not bad, f"PREREGISTRATION.md changed other than by appending: {bad}"


@pytest.mark.parametrize("rev", PRE_BASELINE_CHANGES)
def test_pre_baseline_changes_are_recorded_and_still_detected(rev):
    assert appended_only(_version(f"{rev}^"), _version(rev)) is not None
