"""HICL review packet, stratified by consequence (PREREGISTRATION D5).

Stage 0  ``mapping/review/search_terms.txt``: generic names of the pre-specified
         agents and class members, plus an owner-filled brand section. It must be
         committed before the screen runs; the packet records its commit SHA and
         refuses to build if the file has uncommitted changes.
Stage 1  Partition of every key in ``mapping/hicl_draft.csv``, first match wins:
           tier1_agents.csv     current mapping names a pre-specified agent
           tier2_classes.csv    current mapping feeds a pre-specified class variable
           tier3_screen.csv     ANY other key with a drugname string matching a search
                                term (whole-token / whole-phrase, D5.10)
           tier4_unreviewed.csv the remainder, not reviewed
Stage 2  Audit overlay on tier-4 approximate keys: a head census and a tail
         (census if fewer than 75 remain, else a seeded random 50).

Also here: the within-key reconciliation rule (D5.8) and the string-level row
filter that every medication feature must use (``rows_counted``).

No outcome, label or model output is read or written here.

Usage:
    python -m polysafe.review --terms        # stage 0: write search_terms.txt (then commit it)
    python -m polysafe.review                # stages 1-2: build the packet
    python -m polysafe.review --reliability  # after pass 1 is filled: freeze it, draw pass 2
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from polysafe.labels import wilson
from polysafe.medmap import CHECKLIST, MAPPING_DIR
from polysafe.tables import load

SEED = 20261004
REVIEW_DIR = MAPPING_DIR / "review"
DECISIONS = ("keep", "reject", "unsure")
MAPPING_WRONG_VALUES = ("true", "false")  # D6.1; required on every reviewed key
DECISION_COLUMNS = ("decision", "wrong_strings", "corrected_name", "reviewer_note", "mapping_wrong")
RECONCILE_SHARE = 0.90  # D5.8, fixed before review
HEAD_COVERAGE = 0.80  # a convention, not a derived threshold
HEAD_CAP = 60
CENSUS_BELOW = 75
TAIL_SIZE = 50
BOOTSTRAP = 2000
RELIABILITY_KEYS = 20
FORBIDDEN_COLUMN = re.compile(r"aki|onset|outcome|label|pred|prob|auroc|auc|model|risk|event",
                              re.I)

# Section 10 / D4.1: the analysable agents and the negative control.
AGENTS = {
    "vancomycin": ("vancomycin",),
    "piperacillin-tazobactam": ("piperacillin", "tazobactam"),
    "furosemide": ("furosemide",),
    "ondansetron": ("ondansetron",),
}
# Class variables reviewed in tier 2. Provenance of this list: no committed protocol source
# defined it (PREREGISTRATION D6.0c). The NSAID class (aspirin excluded), aminoglycosides and
# iodinated contrast are named in PREREGISTRATION D4.1 and D4.5; under D6.0c these are the
# pre-specified class set. "ACEi/ARB" appears only in DATA_REPORT.md, a data report, not the
# protocol; loop diuretics appear in no committed document. Both are reviewed here but are
# post-hoc exposures under D6.0c, outside the primary model. Member ingredients are the
# analyst's draft (medmap.CHECKLIST). Reviewing more keys is the safe side.
CLASSES = {
    "loop diuretics": CHECKLIST["loop diuretics"],
    "ACE inhibitors / ARBs": CHECKLIST["ACE inhibitors"] + CHECKLIST["ARBs"],
    "NSAIDs (aspirin excluded, D4.1)": CHECKLIST["NSAIDs (aspirin excluded, D4.1)"],
    "aminoglycosides": CHECKLIST["aminoglycosides"],
    "iodinated contrast": CHECKLIST["iodinated contrast"],
}

BRAND_START = "# == BRAND NAMES (owner) =="
BRAND_END = "# == END BRAND NAMES =="


# ---------------------------------------------------------------- token matching (D5.10)


def normalize(text: str) -> str:
    """NFKD, drop combining marks, casefold, every non-alphanumeric run -> one space, strip."""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    s = "".join(ch if ch.isalnum() else " " for ch in s)
    return " ".join(s.split())


def tokens(text: str) -> tuple[str, ...]:
    return tuple(normalize(text).split(" ")) if normalize(text) else ()


def term_matches(term: str, text: str) -> bool:
    """Whole-token match for a one-word term, contiguous whole-token run for a phrase."""
    t, s = tokens(term), tokens(text)
    if not t:
        return False
    return any(s[i:i + len(t)] == t for i in range(len(s) - len(t) + 1))


# ---------------------------------------------------------------- stage 0: search terms


def search_terms_text() -> str:
    lines = ["# PolySafe HICL review: tier-3 search terms (PREREGISTRATION D5.6).",
             "# Fixed and committed BEFORE the screen runs. One term per line; '#' = comment.",
             "# Matching (D5.10): NFKD + casefold, punctuation -> space, whole-token for one",
             "# word, contiguous whole tokens for a phrase. Never bare substring.",
             "# piperacillin-tazobactam is listed as its two components so that",
             "# 'piperacillin/tazobactam', 'piperacillin and tazobactam' etc. all match.", "",
             "# == Generic names: pre-specified agents (section 10, D4.1) =="]
    seen: list[str] = []  # ordered, de-duplicated
    for group in (AGENTS, CLASSES):
        if group is CLASSES:
            lines += ["", "# == Generic names: pre-specified class members =="]
        for name, terms in group.items():
            lines.append(f"# {name}")
            for t in terms:
                if t not in seen:
                    lines.append(t)
                    seen.append(t)
    lines += ["", BRAND_START,
              "# Template for the owner (D5.14). One slot per generic; write brand aliases",
              "# after '=', separated by ';'. Leave a slot empty if there is none.",
              "# Commit the completed file ALONE before running screen B.",
              "# Format:  <generic> = <alias>; <alias>"]
    lines += [f"{t} =" for t in seen]
    lines += [BRAND_END]
    return "\n".join(lines) + "\n"


def read_brand_aliases(path: Path) -> dict[str, str]:
    """{alias: generic it was written against} from the brand section ('' if unattributed)."""
    aliases, in_brands = {}, False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == BRAND_START:
            in_brands = True
        elif line == BRAND_END:
            in_brands = False
        elif in_brands and line and not line.startswith("#"):
            generic, _, rest = line.partition("=") if "=" in line else ("", "", line)
            for alias in (a.strip() for a in rest.split(";")):
                if alias:
                    aliases[alias] = generic.strip()
    return aliases


def read_search_terms(path: Path) -> tuple[list[str], list[str]]:
    """(generic terms, brand aliases) from search_terms.txt, as written."""
    generics, in_brands = [], False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == BRAND_START:
            in_brands = True
        elif line == BRAND_END:
            in_brands = False
        elif line and not line.startswith("#") and not in_brands:
            generics.append(line)
    return generics, list(read_brand_aliases(path))


def term_groups(path: Path) -> dict[str, str]:
    """Search term (normalised) -> agent or class it stands for; aliases via their generic."""
    groups = {}
    for group in (CLASSES, AGENTS):  # agents last, so they win for furosemide
        for name, terms in group.items():
            for t in terms:
                groups[normalize(t)] = name
    for alias, generic in read_brand_aliases(path).items():
        groups[normalize(alias)] = groups.get(normalize(generic), "")
    return groups


def committed_sha(path: Path) -> str:
    """SHA of the last commit touching ``path``; refuses if the file is not clean."""
    repo = path.resolve().parent
    dirty = subprocess.run(["git", "status", "--porcelain", "--", path.name], cwd=repo,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"{path} has uncommitted changes; commit it before screening")
    sha = subprocess.run(["git", "log", "-1", "--format=%H", "--", path.name], cwd=repo,
                         capture_output=True, text=True, check=True).stdout.strip()
    if not sha:
        raise RuntimeError(f"{path} has never been committed; commit it before screening")
    return sha


# ---------------------------------------------------------------- stage 1: partition


def _ingredient_sets(draft: pd.DataFrame) -> pd.Series:
    return draft["ingredients"].fillna("").map(lambda s: frozenset(x for x in s.split(" + ") if x))


def _feeds(sets: pd.Series, groups: dict) -> pd.Series:
    """Name of the first agent/class whose terms meet the key's ingredients, else ''."""
    return sets.map(lambda s: next((g for g, terms in groups.items() if s & set(terms)), ""))


def _counts(series: pd.Series) -> str:
    vc = series.dropna().value_counts()
    return " | ".join(f"{n} ({c})" for n, c in vc.items())


def name_strings(draft: pd.DataFrame, med: pd.DataFrame, adm: pd.DataFrame) -> pd.DataFrame:
    """Every distinct drugname string per key, each with its own row count."""
    m = med[med["drugordercancelled"] != "Yes"]
    by_med = m.dropna(subset=["drughiclseqno"]).groupby("drughiclseqno")
    med_names = by_med["drugname"].apply(_counts)
    unnamed = by_med["drugname"].apply(lambda s: int(s.isna().sum()))
    adm_names = adm.dropna(subset=["drughiclseqno"]).groupby("drughiclseqno")["drugname"] \
        .apply(_counts)
    rows = []
    for _, r in draft.iterrows():
        if r["key_type"] == "HICL":
            h = float(r["key"])
            rows.append({"drugnames_medication": med_names.get(h, ""),
                         "drugnames_admissiondrug": adm_names.get(h, ""),
                         "rows_without_name": int(unnamed.get(h, 0))})
        else:
            rows.append({"drugnames_medication": f"{r['key']} ({int(r['n_rows'])})",
                         "drugnames_admissiondrug": "", "rows_without_name": 0})
    return pd.DataFrame(rows, index=draft.index)


def parse_counts(cell: str) -> dict[str, int]:
    """'NAME A (3) | NAME B (12)' -> {'NAME A': 3, 'NAME B': 12}."""
    out = {}
    for part in str(cell or "").split(" | "):
        m = re.match(r"^(.*) \((\d+)\)$", part.strip())
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def screen(names: pd.DataFrame, terms: list[str]) -> pd.Series:
    """Per key: [(term, original drugname string), ...] for every whole-token hit."""
    def hits(row) -> list[tuple[str, str]]:
        strings = list(parse_counts(row["drugnames_medication"])) \
            + list(parse_counts(row["drugnames_admissiondrug"]))
        return [(t, s) for s in strings for t in terms if term_matches(t, s)]
    return names.apply(hits, axis=1)


def partition(draft: pd.DataFrame, names: pd.DataFrame, terms: list[str]):
    """Tier per key (first match wins), agent/class fed, hits, tier-3 source, zero-hit terms."""
    sets = _ingredient_sets(draft)
    agent, klass = _feeds(sets, AGENTS), _feeds(sets, CLASSES)
    hits = screen(names, terms)
    tier = pd.Series(4, index=draft.index)
    tier[hits.map(len) > 0] = 3
    tier[klass != ""] = 2
    tier[agent != ""] = 1
    source = pd.Series("", index=draft.index)
    t3 = tier == 3
    source[t3] = np.select(
        [draft.loc[t3, "match"] == "unmapped", draft.loc[t3, "name_source"] == "admissiondrug"],
        ["unmapped", "admissiondrug-only"], "mapped to something else")
    used = {t for hs in hits for t, _ in hs}
    zero_hit = [t for t in terms if t not in used]
    return tier, agent, klass, hits, source, zero_hit


# ---------------------------------------------------------------- reconciliation (D5.8)


def reconcile(strings: dict[str, tuple[int, str]], share: float = RECONCILE_SHARE) -> dict:
    """Within-key rule. ``strings``: {drugname string: (rows, ingredient it names)}.

    One ingredient >= ``share`` of the key's rows: keep, minority strings are wrong.
    Otherwise: unsure, every string is listed as conflicting.
    """
    by_ing: dict[str, int] = {}
    for rows, ing in strings.values():
        by_ing[ing] = by_ing.get(ing, 0) + rows
    total = sum(by_ing.values())
    top, top_rows = max(by_ing.items(), key=lambda kv: kv[1])
    if len(by_ing) == 1 or top_rows / total >= share:
        wrong = [s for s, (_, ing) in strings.items() if ing != top]
        return {"decision": "keep", "ingredient": top, "wrong_strings": wrong,
                "error_rows": sum(strings[s][0] for s in wrong)}
    return {"decision": "unsure", "ingredient": None, "wrong_strings": list(strings),
            "error_rows": total}


def _wrong(cell) -> list[str]:
    return [w.strip() for w in str(cell or "").split(" | ") if w.strip()]


def _row_keys(med: pd.DataFrame) -> pd.Series:
    key = med["drughiclseqno"].map(lambda v: f"HICL:{int(v)}" if pd.notna(v) else None)
    return key.fillna("NAME:" + med["drugname"].fillna(""))


def rows_counted(med: pd.DataFrame, reviewed: pd.DataFrame, include_unsure: bool = False,
                 named_only: bool = False) -> pd.Series:
    """Which medication rows count toward a reviewed key's exposure. Filters at the
    drugname-STRING level, not the hiclseqno level (D5.8).

    keep   -> the key's rows, minus rows whose drugname is listed in wrong_strings.
              Rows with no drugname follow the key (D5.12) unless ``named_only``
              (the D5.12 sensitivity analysis), which drops every row lacking a string.
    unsure -> no rows in the primary strategy; with ``include_unsure`` the same row
              rule as keep, i.e. wrong strings still excluded (D5.13)
    reject -> no rows
    A key named only through admissiondrug has no medication strings: any wrong string
    there removes all of its rows.
    Rows of wrong strings are excluded from ALL exposures and never reassigned (D5.11).
    """
    key = _row_keys(med)
    r = reviewed.set_index("key")
    decision = key.map(r["decision"]).fillna("")
    wrong = r["wrong_strings"].map(_wrong)
    adm_only = r["name_source"].eq("admissiondrug") & wrong.map(bool)
    row_wrong = [isinstance(n, str) and n in wrong.get(k, []) for k, n in
                 zip(key, med["drugname"], strict=True)]
    usable = ~pd.Series(row_wrong, index=med.index) \
        & ~key.map(adm_only).fillna(False).astype(bool)
    if named_only:
        usable &= med["drugname"].notna()
    accepted = decision.isin(("keep", "unsure") if include_unsure else ("keep",))
    return accepted & usable


def exposure_rows(med: pd.DataFrame, reviewed: pd.DataFrame, include_unsure: bool = False,
                  named_only: bool = False) -> pd.DataFrame:
    """Medication rows that count, each labelled with the ONE exposure its key feeds.

    ``reviewed`` columns: key, name_source, decision, wrong_strings, corrected_name and
    ``exposure`` (the agent/class from the packet). ``corrected_name`` acts at key
    (hiclseqno) level only and must name an already pre-specified agent or class
    (D5.4); rows of wrong strings are never reassigned (D5.11).
    """
    allowed = set(AGENTS) | set(CLASSES)
    r = reviewed.set_index("key")
    corrected = r["corrected_name"].fillna("").str.strip()
    new = sorted(set(corrected[corrected != ""]) - allowed)
    if new:
        raise ValueError(f"corrected_name must be a pre-specified agent or class: {new}")
    target = corrected.where(corrected != "", r["exposure"].fillna(""))
    counted = rows_counted(med, reviewed, include_unsure, named_only)
    rows = med[counted].assign(exposure=_row_keys(med)[counted].map(target))
    return rows[rows["exposure"].fillna("") != ""]


def stay_exposure(rows: pd.DataFrame, stays) -> pd.DataFrame:
    """Ascertainment dataset: one row per stay, one boolean column per exposure."""
    hit = pd.crosstab(rows["patientunitstayid"], rows["exposure"]) > 0
    return hit.reindex(pd.Index(stays, name="patientunitstayid"), fill_value=False)


# ---------------------------------------------------------------- stage 2: audit


def audit_sample(t4: pd.DataFrame):
    """Head census (80% of rows, cap 60) and tail (census if < 75 remain, else random 50)."""
    s = t4[t4["match"] == "approximate"].copy()
    s["_k"] = s["key_type"] + ":" + s["key"].astype(str)
    s = s.sort_values(["n_rows", "_k"], ascending=[False, True])
    total = int(s["n_rows"].sum())
    cum = s["n_rows"].cumsum().to_numpy()
    needed = int((cum < HEAD_COVERAGE * total).sum()) + 1 if total else 0
    head_n = min(needed, HEAD_CAP)
    head, rest = s.iloc[:head_n], s.iloc[head_n:]
    census = len(rest) < CENSUS_BELOW
    if census:
        tail = rest
    else:
        pick = np.random.default_rng(SEED).choice(len(rest), size=TAIL_SIZE, replace=False)
        tail = rest.iloc[np.sort(pick)]
    design = {"stratum_keys": len(s), "R": total, "needed_for_80": needed,
              "head_keys": head_n, "head_rows": int(head["n_rows"].sum()),
              "N": len(rest), "tail_population_rows": int(rest["n_rows"].sum()),
              "n": len(tail), "tail_mode": "census" if census else "sample",
              "rows_min": int(s["n_rows"].min()) if len(s) else 0,
              "rows_median": float(s["n_rows"].median()) if len(s) else 0.0,
              "rows_max": int(s["n_rows"].max()) if len(s) else 0, "seed": SEED}
    return head.index, tail.index, design


def mapping_wrong(row: pd.Series) -> bool:
    """D6.1 whole-key mapping verdict: ``true`` or ``false`` (any case), never blank."""
    value = str(row.get("mapping_wrong", "")).strip().lower()
    if value not in MAPPING_WRONG_VALUES:
        raise ValueError(f"{row['key']}: mapping_wrong must be true or false, "
                         f"got {row.get('mapping_wrong', '')!r}")
    if value == "true" and _wrong(row.get("wrong_strings", "")):
        raise ValueError(f"{row['key']}: a key with mapping_wrong true lists no "
                         "wrong_strings (D6.1)")
    return value == "true"


def error_rows(row: pd.Series, mapping_errors: bool = False) -> int:
    """Rows of the drugname strings within the key that the reviewer marked wrong.

    A key named only through admissiondrug has no per-string medication rows: all of its
    rows carry the borrowed name, so any wrong string makes all of them error rows.

    With ``mapping_errors`` (D6.1), a key whose mapping_wrong is true contributes ALL of
    its rows, including rows with no drugname.
    """
    if mapping_errors and mapping_wrong(row):
        return int(row["n_rows"])
    wrong = _wrong(row.get("wrong_strings", ""))
    if not wrong:
        return 0
    med = parse_counts(row["drugnames_medication"])
    adm = parse_counts(row["drugnames_admissiondrug"])
    unknown = [w for w in wrong if w not in med and w not in adm]
    if unknown:
        raise ValueError(f"{row['key']}: wrong_strings not among its drugnames: {unknown}")
    if not med:
        return int(row["n_rows"])
    return sum(med.get(w, 0) for w in wrong)


def is_error_key(row: pd.Series, mapping_errors: bool = False) -> bool:
    """Key-level error: any wrong string (REVIEW_PACKET.md); with ``mapping_errors``,
    also mapping_wrong true (D6.1)."""
    return bool(_wrong(row.get("wrong_strings", ""))) or (mapping_errors and mapping_wrong(row))


def audit_estimate(head_errors, tail_errors, N: int, R: int, seed: int = SEED,
                   tail_error_keys=None) -> dict:
    """Combined row-level error rate over the tier-4 approximate stratum (REVIEW_PACKET.md).

    ``tail_error_keys`` flags the tail keys that count in the key-level proportion
    ("keys with any wrong string"); by default, keys with any error rows.
    """
    head_errors = np.asarray(head_errors, dtype=float)
    tail_errors = np.asarray(tail_errors, dtype=float)
    n = len(tail_errors)
    e_head = float(head_errors.sum())
    e_tail = float(N / n * tail_errors.sum()) if n else 0.0
    rate = (e_head + e_tail) / R
    if n and N > n:
        idx = np.random.default_rng(seed).integers(0, n, size=(BOOTSTRAP, n))
        boot = (e_head + N / n * tail_errors[idx].sum(axis=1)) / R
        ci = tuple(float(x) for x in np.percentile(boot, [2.5, 97.5]))
    else:
        ci = (rate, rate)  # census: exact, no sampling error
    flags = tail_errors > 0 if tail_error_keys is None else np.asarray(tail_error_keys, bool)
    k = int(flags.sum())
    return {"E_head": e_head, "E_tail": e_tail, "N": N, "n": n, "R": R, "rate": rate,
            "rate_ci": ci, "census": N == n,
            "tail_keys_with_error": k, "tail_key_proportion": k / n if n else float("nan"),
            "tail_key_wilson": wilson(k, n)}


def estimate_from_review(review_dir: Path = REVIEW_DIR) -> dict:
    """Apply the estimator to the filled audit CSVs, using the frozen design.

    Reported both ways (D6.1): ``as_defined`` is the estimator as REVIEW_PACKET.md
    defines it; ``with_mapping_errors`` adds keys whose mapping_wrong is true.
    mapping_wrong must be filled on every audit key.
    """
    design = json.loads((review_dir / "audit_design.json").read_text())
    head = pd.read_csv(review_dir / "audit_head.csv", dtype=str, keep_default_na=False)
    tail = pd.read_csv(review_dir / "audit_tail.csv", dtype=str, keep_default_na=False)
    for _, r in pd.concat([head, tail]).iterrows():
        mapping_wrong(r)
    return {version: audit_estimate(
                [error_rows(r, flag) for _, r in head.iterrows()],
                [error_rows(r, flag) for _, r in tail.iterrows()],
                N=design["N"], R=design["R"],
                tail_error_keys=[is_error_key(r, flag) for _, r in tail.iterrows()])
            for version, flag in (("as_defined", False), ("with_mapping_errors", True))}


# ---------------------------------------------------------------- packet


def _review_frame(draft, names, idx, extra: dict | None = None, decisions: bool = True):
    d = draft.loc[idx]
    out = pd.DataFrame({
        "key": d["key_type"] + ":" + d["key"].astype(str),
        "hiclseqno": np.where(d["key_type"] == "HICL", d["key"], ""),
        "name_source": d["name_source"],
        "drugnames_medication": names.loc[idx, "drugnames_medication"],
        "drugnames_admissiondrug": names.loc[idx, "drugnames_admissiondrug"],
        "rows_without_name": names.loc[idx, "rows_without_name"],
        "n_stays": d["n_stays"],
        "n_rows": d["n_rows"],
        "gtc": d["gtc"],
        "route": d["route"],
        "proposed_ingredients": d["ingredients"].fillna(""),
        "proposed_rxnorm_name": d["rxnorm_name"].fillna(""),
        "match": d["match"],
    })
    for col, values in (extra or {}).items():
        out[col] = values
    if decisions:
        for col in DECISION_COLUMNS:
            out[col] = ""
    return out.reset_index(drop=True)


def build_review_packet(draft_csv: Path | None = None, out_dir: Path | None = None,
                        terms_file: Path | None = None, med: pd.DataFrame | None = None,
                        adm: pd.DataFrame | None = None, require_committed: bool = True) -> dict:
    out = Path(out_dir or REVIEW_DIR)
    terms_file = Path(terms_file or out / "search_terms.txt")
    sha = committed_sha(terms_file) if require_committed else "(not checked)"
    generics, brands = read_search_terms(terms_file)
    draft = pd.read_csv(draft_csv or MAPPING_DIR / "hicl_draft.csv", dtype={"key": str})
    med = med if med is not None else load("medication")
    adm = adm if adm is not None else load("admissiondrug",
                                           usecols=["drughiclseqno", "drugname"])
    names = name_strings(draft, med, adm)
    tier, agent, klass, hits, source, zero_hit = partition(draft, names, generics + brands)

    out.mkdir(parents=True, exist_ok=True)
    t3 = tier == 3
    frames = {
        "tier1_agents.csv": _review_frame(draft, names, tier.index[tier == 1],
                                          {"agent": agent[tier == 1]}),
        "tier2_classes.csv": _review_frame(draft, names, tier.index[tier == 2],
                                           {"class": klass[tier == 2]}),
        "tier3_screen.csv": _review_frame(draft, names, tier.index[t3], {
            "hit_terms": hits[t3].map(lambda h: " | ".join(dict.fromkeys(t for t, _ in h))),
            "hit_strings": hits[t3].map(lambda h: " || ".join(f"{t} <= {s}" for t, s in h)),
            "screen_source": source[t3]}),
        "tier4_unreviewed.csv": _review_frame(draft, names, tier.index[tier == 4],
                                              decisions=False),
    }
    head_idx, tail_idx, design = audit_sample(draft.loc[tier == 4])
    frames["audit_head.csv"] = _review_frame(draft, names, head_idx)
    frames["audit_tail.csv"] = _review_frame(draft, names, tail_idx)
    for name, f in frames.items():
        f.to_csv(out / name, index=False)
    design["search_terms_sha"] = sha
    (out / "audit_design.json").write_text(json.dumps(design, indent=1))

    live = med[med["drugordercancelled"] != "Yes"]
    orphan = int((live["drughiclseqno"].isna() & live["drugname"].isna()).sum())
    md = packet_markdown(draft, tier, source, generics, brands, zero_hit, design, sha,
                         len(live), orphan)
    (out / "REVIEW_PACKET.md").write_text(md, encoding="utf-8")
    return {"tiers": tier.value_counts().sort_index().to_dict(),
            "tier3_sources": source[t3].value_counts().to_dict(),
            "zero_hit_terms": zero_hit, "design": design, "sha": sha}


ZERO_HIT_HEADING = "### terms with no hits - absent from eICU or misspelled"


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def read_zero_hit(packet_md: Path) -> list[str]:
    """Zero-hit terms as printed in a REVIEW_PACKET.md."""
    lines = packet_md.read_text(encoding="utf-8").splitlines()
    i = lines.index(ZERO_HIT_HEADING)
    body = next(x for x in lines[i + 1:] if x.strip())
    return [] if body.strip() == "(none)" else re.findall(r"`([^`]+)`", body)


def compare_screens(a_dir: Path, b_dir: Path, med: pd.DataFrame, terms_file: Path) -> dict:
    """Per agent/class: tier-3 candidate keys, rows, distinct stays in A and B, and B - A.

    Rows are non-cancelled medication rows; stays are counted distinctly per group,
    never summed over keys. Screen A's files are read as they are, never rebuilt.
    """
    groups = term_groups(terms_file)
    live = med[med["drugordercancelled"] != "Yes"]
    row_key = _row_keys(live)

    def by_group(d: Path) -> dict[str, set]:
        t3 = pd.read_csv(d / "tier3_screen.csv", dtype=str, keep_default_na=False)
        out: dict[str, set] = {}
        for _, r in t3.iterrows():
            for term in r["hit_terms"].split(" | "):
                g = groups.get(normalize(term), "")
                out.setdefault(g, set()).add(r["key"])
        return out

    a, b = by_group(a_dir), by_group(b_dir)
    rows = []
    for g in [*AGENTS, *CLASSES, ""]:
        ka, kb = a.get(g, set()), b.get(g, set())
        ra, rb = live[row_key.isin(ka)], live[row_key.isin(kb)]
        sa, sb = set(ra["patientunitstayid"]), set(rb["patientunitstayid"])
        rows.append({"group": g or "(unattributed alias)", "A_keys": len(ka), "A_rows": len(ra),
                     "A_stays": len(sa), "B_keys": len(kb), "B_rows": len(rb),
                     "B_stays": len(sb), "added_keys": len(kb - ka),
                     "added_rows": len(live[row_key.isin(kb - ka)]),
                     "added_stays": len(sb - sa)})
    return {"per_group": pd.DataFrame(rows),
            "zero_hit_A": read_zero_hit(a_dir / "REVIEW_PACKET.md"),
            "zero_hit_B": read_zero_hit(b_dir / "REVIEW_PACKET.md")}


def build_screen_b(review_dir: Path = REVIEW_DIR) -> dict:
    """Screen B: the committed term file with the owner's brand aliases. Screen A is kept."""
    terms_file = review_dir / "search_terms.txt"
    if not read_brand_aliases(terms_file):
        raise RuntimeError("no brand aliases in search_terms.txt; screen B would equal screen A")
    b_dir = review_dir / "screen_b"
    med = load("medication")
    result = build_review_packet(out_dir=b_dir, terms_file=terms_file, med=med)
    cmp = compare_screens(review_dir, b_dir, med, terms_file)
    md = ["# Screen A vs screen B (generated by `python -m polysafe.review --screen-b`)", "",
          f"Screen A: generic names only (`{review_dir.name}/`). Screen B: generic names plus "
          f"the owner's brand aliases at commit `{result['sha']}` (`screen_b/`). Both are "
          "reported in full; neither replaces the other (D5.14). Tier-3 candidates only; "
          "nothing is recovered until adjudication accepts it.", "",
          _md_table(cmp["per_group"]), "",
          "Zero-hit terms, screen A: " + (", ".join(cmp["zero_hit_A"]) or "(none)"), "",
          "Zero-hit terms, screen B: " + (", ".join(cmp["zero_hit_B"]) or "(none)")]
    (b_dir / "SCREEN_COMPARISON.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return cmp


ESTIMATOR = """\
error_rows_i = total rows of the drugname strings within key i that the
               reviewer marked wrong (NOT all rows of key i).
E_head  = sum of error_rows over head keys. Exact; the head is a census.
N       = tail population size, n = keys sampled.
E_tail  = (N / n) * sum of error_rows_i over sampled keys.
          Horvitz-Thompson total under SRS of keys; inclusion
          probability n/N, weight N/n.
R       = total rows across all tier-4 approximate keys (head + tail
          population).
Combined row-level error rate = (E_head + E_tail) / R.
CI: nonparametric bootstrap, 2000 resamples of the n sampled keys with
    replacement, recomputing E_tail each time with E_head and R fixed;
    percentile interval; default_rng(20261004)."""


def packet_markdown(draft, tier, source, generics, brands, zero_hit, d, sha,
                    live_rows: int, orphan_rows: int) -> str:
    files = {1: "tier1_agents.csv", 2: "tier2_classes.csv", 3: "tier3_screen.csv",
             4: "tier4_unreviewed.csv"}
    lines = ["# HICL review packet (generated by `python -m polysafe.review`)", "",
             "PREREGISTRATION D5. Source: `mapping/hicl_draft.csv`. No outcome, label or "
             "model output was read to build this packet.", "",
             f"**Search terms:** `mapping/review/search_terms.txt` at commit `{sha}`, "
             f"committed before this screen ran. {len(generics)} generic terms; "
             f"{len(brands)} brand terms"
             + (" (owner section empty: generic names only)." if not brands else "."), "",
             "Provenance: one earlier build under a superseded specification (auto-derived "
             "brand tokens, substring matching, tier 3 limited to unmapped/admissiondrug "
             "keys) ran before the term list existed and was discarded uncommitted.", "",
             "## Tiers (mutually exclusive, exhaustive; first match wins)", "",
             "| Tier | File | Keys | Rows | Reviewed |", "|---|---|---:|---:|---|"]
    for t in (1, 2, 3, 4):
        sel = draft[tier == t]
        lines.append(f"| {t} | `{files[t]}` | {len(sel):,} | {int(sel['n_rows'].sum()):,} | "
                     f"{'exhaustively' if t < 4 else 'no (audit overlay only)'} |")
    lines += [f"| | **total** | **{len(draft):,}** | **{int(draft['n_rows'].sum()):,}** | |",
              "", f"Non-cancelled medication rows: {live_rows:,}. The remaining "
              f"{orphan_rows:,} rows have neither a HICL code nor a drugname, belong to no "
              "key and cannot be mapped or reviewed.",
              "", "1. **Agents** (section 10, D4.1): current mapping names "
              + ", ".join(AGENTS) + ".",
              "2. **Classes**: current mapping feeds a pre-specified class variable: "
              + "; ".join(f"{c} ({', '.join(t)})" for c, t in CLASSES.items()) + ".",
              "3. **Screen**: any other key, whatever its mapping status, with a drugname "
              "string matching a search term. Keys mapped to an unrelated ingredient are "
              "included on purpose. Each row records the original string and the term that "
              "hit (`hit_strings`: `term <= original string`).",
              "4. **Unreviewed**: the remainder.", "",
              "**Matching (D5.10).** Both string and term: Unicode NFKD, combining marks "
              "dropped, casefold, every run of non-alphanumeric characters (punctuation, "
              "symbols, whitespace) replaced by one space, stripped, split on spaces. A "
              "one-word term matches a whole token only; a multi-word term matches a "
              "contiguous run of whole tokens only. No bare substring matching.", "",
              "Tier-3 keys by source:", "", "| Source | Keys |", "|---|---:|"]
    for s in ("unmapped", "admissiondrug-only", "mapped to something else"):
        lines.append(f"| {s} | {int((source == s).sum())} |")
    lines += ["", ZERO_HIT_HEADING, ""]
    lines += [", ".join(f"`{t}`" for t in zero_hit) if zero_hit else "(none)"]
    never = int(((draft["key_type"] == "HICL") & (draft["name_source"] == "none")).sum())
    lines += ["", f"**Limit:** {never:,} HICL keys have no drugname anywhere in eICU; no name "
              "screen can reach them. Residual false negatives are keys whose strings contain "
              "no recognisable agent or class name.", "",
              "Mapping-coverage figures are reported from the full key set "
              "(`mapping/MAPPING_REPORT.md`), never from the reviewed subset.", "",
              "## Audit overlay on tier 4 (not a tier)", "",
              f"Stratum: tier-4 keys with match = approximate: {d['stratum_keys']:,} keys, "
              f"R = {d['R']:,} rows. Rows per key: min {d['rows_min']}, median "
              f"{d['rows_median']:g}, max {d['rows_max']}.", "",
              f"- `audit_head.csv`: largest keys by rows until 80% of R (a convention, not a "
              f"derived threshold), cap {HEAD_CAP}. 80% needs {d['needed_for_80']} keys; the "
              f"head uses **{d['head_keys']} keys = {d['head_rows']:,} rows = "
              f"{d['head_rows'] / d['R']:.1%} of R**"
              + (" (cap reached)." if d["needed_for_80"] > HEAD_CAP else "."),
              f"- `audit_tail.csv`: tail population N = {d['N']:,} keys "
              f"({d['tail_population_rows']:,} rows). "
              + (f"N < {CENSUS_BELOW}: **census**, all {d['n']} keys; the tail total is an "
                 "exact count, no estimator needed." if d["tail_mode"] == "census" else
                 f"N >= {CENSUS_BELOW}: **random sample** of n = {d['n']} keys, "
                 f"`numpy.random.default_rng({SEED})`."),
              "- `audit_design.json` freezes N, n and R for the estimator.", "",
              "### Estimator (verbatim)", "", "```", ESTIMATOR, "```", "",
              "Operational note: for keys named only through `admissiondrug` the medication "
              "rows carry no string of their own; a wrong string makes all of that key's rows "
              "error rows. Rows with no drugname in a medication-named key are not error rows "
              "(no string was reviewed).", "",
              "Reported separately: the key-level error proportion in the tail sample (keys "
              "with any wrong string / n) with a Wilson 95% interval.", "",
              "**Wilson applies ONLY to that key-level proportion. It is NOT valid for the "
              "combined row-weighted estimate. The combined estimate covers the tier-4 "
              "approximate stratum only and says nothing about the exhaustively reviewed "
              "tiers.**", "",
              "## Columns to fill (every reviewable CSV)", "",
              "- `decision`: `keep` | `reject` | `unsure` (below).",
              "- `wrong_strings`: the exact drugname strings that are wrong, separated by "
              "` | `, copied from `drugnames_medication` / `drugnames_admissiondrug` "
              "without the row count.",
              "- `corrected_name`, `reviewer_note`: free text.",
              "- `mapping_wrong`: `true` or `false`, required on every key given a "
              "decision (D6.1). `true` only when the proposed ingredient mapping is wrong "
              "for every row of the key; a key marked `true` lists no `wrong_strings`. When "
              "`mapping_wrong` is `true`, or `proposed_ingredients` is empty, record the "
              "correct ingredient in `reviewer_note` (D6.1, D6.3). Every audit estimate is "
              "reported both ways: as defined above, and with each `true` key contributing "
              "all of its rows, including rows with no drugname.", "",
              "## Decisions", "",
              "### Reconciliation (D5.8)", "",
              "Adjudication is at HICL-key level; the review never invents a row-level mapping "
              "scheme mid-review. Where distinct drugname strings under one key name different "
              "ingredients:", "",
              f"- If a single ingredient accounts for >= {RECONCILE_SHARE:.0%} of that key's "
              "rows, the key is reconcilable: `decision = keep`, with the minority strings "
              "named in `wrong_strings`. Their rows are excluded from the exposure.",
              "- Otherwise, the key is not reconcilable: `decision = unsure`, with all "
              "conflicting strings in `wrong_strings`.", "",
              f"The {RECONCILE_SHARE:.0%} threshold was fixed before review began.", "",
              "Rows of strings listed in `wrong_strings` are excluded from all exposures. "
              "They are never reassigned to another agent or class, even when the string "
              "names one. `corrected_name` operates at hiclseqno level only. This is "
              "conservative by design and biases toward under-ascertainment (D5.11).", "",
              "### Vocabulary (fixed before review)", "",
              "| decision | meaning |", "|---|---|",
              "| `keep` | the rows of this key's CORRECT strings contribute to its exposure; "
              "rows of strings listed in `wrong_strings` do not. `keep` does NOT mean all "
              "rows of the key count |",
              "| `reject` | contributes to no exposure |",
              "| `unsure` | EXCLUDED from the primary ascertainment strategy, INCLUDED in a "
              "pre-specified sensitivity variant; both reported |", "",
              "Rows with no drugname in a kept key have no string that could be wrong; they "
              "follow the key. Medication features filter at drugname-string level "
              "(`polysafe.review.rows_counted`), never at hiclseqno level.", "",
              "`corrected_name` may: remove a key from an exposure; move it between two "
              "ALREADY pre-specified agent exposures; add or move it to an ALREADY "
              "pre-specified class variable. It may NOT create a new exposure or class. "
              "Corrections requiring a new variable are logged below and excluded.", "",
              "### Corrections requiring a new variable (logged, excluded)", "",
              "| Key | Proposed new variable | Reviewer note |", "|---|---|---|", "| | | |", "",
              "## Reliability (optional, exploratory)", "",
              "`python -m polysafe.review --reliability`, run after the filled CSVs return, "
              "freezes pass 1 under `mapping/review/pass1/` and writes "
              f"`reliability_pass2.csv`: {RELIABILITY_KEYS} keys drawn at random from tiers "
              f"1-2 (`default_rng({SEED})`) with the decision columns blanked, blind to pass 1. "
              "`reliability_report()` gives raw percent agreement alongside Cohen's kappa, "
              "labelled exploratory intra-rater consistency, not a validation statistic: at "
              "n = 20 with imbalanced decisions kappa is unstable."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- reliability


def _tiers_1_2(review: Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(review / f, dtype=str, keep_default_na=False).assign(tier=t)
                      for f, t in (("tier1_agents.csv", "1"), ("tier2_classes.csv", "2"))],
                     ignore_index=True)


def draw_reliability_sample(review_dir: Path = REVIEW_DIR) -> Path:
    """Freeze the filled pass 1 and emit a blind pass-2 sample of tier-1/2 keys."""
    review = Path(review_dir)
    frozen = review / "pass1"
    if frozen.exists():
        raise FileExistsError(f"{frozen} already exists; pass 1 is frozen once")
    t12 = _tiers_1_2(review)
    bad = t12[~t12["decision"].isin(DECISIONS)]
    if len(bad):
        raise ValueError(f"pass 1 is not complete: {len(bad)} tier-1/2 keys lack a valid "
                         f"decision ({', '.join(DECISIONS)}), e.g. {bad['key'].iloc[0]}")
    frozen.mkdir(parents=True)
    for f in review.glob("*"):
        if f.is_file():
            shutil.copy2(f, frozen / f.name)
    pick = np.random.default_rng(SEED).choice(len(t12), size=min(RELIABILITY_KEYS, len(t12)),
                                              replace=False)
    sample = t12.iloc[pick].copy()
    for col in DECISION_COLUMNS:
        sample[col] = ""
    path = review / "reliability_pass2.csv"
    sample.to_csv(path, index=False)
    return path


def reliability_report(review_dir: Path = REVIEW_DIR) -> dict:
    """Exploratory intra-rater consistency: raw agreement and Cohen's kappa on `decision`."""
    from sklearn.metrics import cohen_kappa_score

    review = Path(review_dir)
    p1 = _tiers_1_2(review / "pass1").set_index("key")["decision"]
    p2 = pd.read_csv(review / "reliability_pass2.csv", dtype=str,
                     keep_default_na=False).set_index("key")["decision"]
    if not p2.isin(DECISIONS).all():
        raise ValueError("pass 2 is not complete")
    a = p1.loc[p2.index]
    return {"label": "exploratory intra-rater consistency (not a validation statistic)",
            "n": len(p2), "percent_agreement": float((a == p2).mean()),
            "cohen_kappa": float(cohen_kappa_score(a, p2, labels=list(DECISIONS)))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HICL review packet (PREREGISTRATION D5).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--terms", action="store_true", help="stage 0: write search_terms.txt")
    g.add_argument("--reliability", action="store_true",
                   help="freeze filled pass 1 and draw the blind pass-2 sample")
    g.add_argument("--screen-b", action="store_true",
                   help="run screen B with the owner's committed brand aliases")
    args = p.parse_args(argv)
    if args.screen_b:
        print(build_screen_b()["per_group"].to_string(index=False))
        return 0
    if args.terms:
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        path = REVIEW_DIR / "search_terms.txt"
        path.write_text(search_terms_text(), encoding="utf-8")
        print(path)
    elif args.reliability:
        print(draw_reliability_sample())
    else:
        print(json.dumps(build_review_packet(), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
