"""Exposure ascertainment: five arms -> one stay x exposure table per arm and variant.

PREREGISTRATION section 6 (four strategies), D6.3 (fifth arm), D6.3a, D6.4 (window,
cancelled orders, gtc), D6.7 (which rows and keys the HICL arms read), D6.2 (route),
D5.3/D5.13 (unsure keys), D5.11 (wrong strings), D5.12 (unnamed rows). Exposures are the
pre-specified set only (``routes.EXPOSURES``: the agents and the D6.0c classes).

Arms
  generic_string  drugname matched against the committed generic terms (D5.10)
  generic_brand   the same, plus the committed brand aliases (D5.14, D5.14b)
  hicl            reviewed mapping of tiers 1-3, rows with a drughiclseqno only (D6.7)
  hicl_gtc        hicl; gtc is read only for rows without a drughiclseqno (D6.4), and in
                  this data every such row has gtc = 0, which is asserted, not assumed
  hicl_reviewed   hicl, then the D6.3 reassignments read from all five review files

Variants (the same four for every arm)
  primary         systemic = yes rows only (D6.2); unsure keys excluded (D5.3)
  route_unknown   systemic = yes or unknown (D6.2 sensitivity)
  unsure          unsure keys included, wrong strings still excluded (D5.13)
  named_only      rows without a drugname dropped (D5.12 sensitivity)
For the string arms, ``unsure`` and ``named_only`` equal ``primary`` by construction:
they read no review decision and only named rows.

A stay is exposed when at least one of its rows counts (D6.4). No AKI label, outcome
column or model output is read here. The cohort's stays come from ``cohort.build_cohort``,
whose section 5 exclusions read creatinine and RRT to decide which stays enter.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from polysafe import cohort, review, routes
from polysafe.medmap import MAPPING_DIR
from polysafe.tables import load

ARMS = ("generic_string", "generic_brand", "hicl", "hicl_gtc", "hicl_reviewed")
VARIANTS = ("primary", "route_unknown", "unsure", "named_only")
EXPOSURES = routes.EXPOSURES
TIERS_1_3 = review.REVIEW_FILES[:3]
NOTE = re.compile(r"^Correct ingredient: ([^.]+)\.")  # D6.3a
RESULTS = MAPPING_DIR.parent / "results"


# ---------------------------------------------------------------- rows that count (D6.4)


def counting(med: pd.DataFrame) -> pd.DataFrame:
    """Rows active in [0, T]: drugstartoffset <= T, drugstopoffset >= 0, not cancelled."""
    keep = ((med["drugstartoffset"] <= cohort.T_MIN) & (med["drugstopoffset"] >= 0)
            & (med["drugordercancelled"] != "Yes"))
    return med[keep]


# ---------------------------------------------------------------- string arms (1, 2)


def string_rows(med: pd.DataFrame, terms_file: Path, with_brands: bool) -> pd.DataFrame:
    """One row per (medication row, exposure) whose drugname contains a committed term,
    whole-token / whole-phrase (D5.10). A row may hit more than one exposure."""
    generics, brands = review.read_search_terms(terms_file)
    groups = review.term_groups(terms_file)
    named = med[med["drugname"].notna()]
    padded = " " + named["drugname"].map(review.normalize) + " "
    parts = []
    for term in generics + (brands if with_brands else []):
        exposure = groups.get(review.normalize(term), "")
        if exposure not in EXPOSURES:
            continue
        hit = padded.str.contains(" " + review.normalize(term) + " ", regex=False)
        parts.append(named[hit].assign(exposure=exposure, key=review._row_keys(named[hit])))
    if not parts:
        return named.iloc[0:0].assign(exposure=pd.Series(dtype=str), key=pd.Series(dtype=str))
    return pd.concat(parts).drop_duplicates(["medicationid", "exposure"])


# ---------------------------------------------------------------- HICL arms (3, 4, 5)


def read_review(review_dir: Path, files=review.REVIEW_FILES) -> pd.DataFrame:
    """The committed review files, with the exposure each key feeds as built (tier 1 agent,
    tier 2 class, otherwise none). A key may appear in only one file."""
    frames = []
    for name in files:
        f = pd.read_csv(review_dir / name, dtype=str, keep_default_na=False)
        exposure = f["agent"] if "agent" in f else f["class"] if "class" in f else ""
        frames.append(f.assign(exposure=exposure, file=name))
    r = pd.concat(frames, ignore_index=True)
    dup = sorted(r.loc[r["key"].duplicated(), "key"])
    if dup:
        raise ValueError(f"keys in more than one review file: {dup}")
    return r


def _hicl_only(med: pd.DataFrame) -> pd.DataFrame:
    return med[med["drughiclseqno"].notna()]  # D6.7: NAME: keys never enter arms 3-5


def hicl_rows(med: pd.DataFrame, review_dir: Path, include_unsure: bool = False,
              named_only: bool = False) -> pd.DataFrame:
    """Arm 3: tier 1-3 decisions applied to HICL rows (corrected_name, D5.11, D5.12, D5.13)."""
    reviewed = read_review(review_dir, TIERS_1_3)
    hicl = _hicl_only(med)
    rows = review.exposure_rows(hicl, reviewed, include_unsure, named_only)
    rows = rows.assign(key=review._row_keys(rows))
    return rows[rows["exposure"].isin(EXPOSURES)]


def check_gtc(med: pd.DataFrame) -> None:
    """D6.4 reads gtc only for rows without a drughiclseqno. No gtc -> exposure map is
    pre-specified, so the HICL + gtc arm is defined only while every such row has gtc 0."""
    no_hicl = med[med["drughiclseqno"].isna()]
    bad = no_hicl["gtc"].fillna(0) != 0
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows without a drughiclseqno carry a gtc; "
                         "the HICL + gtc arm needs a pre-specified gtc map")


@dataclass
class Reassignment:
    exposures: dict  # key -> exposure, D6.3 reassignments
    unreadable: list  # eligible keys whose note is not in the D6.3a form


def _names_ingredient(ingredient: str, terms_file: Path) -> list[str]:
    """The generic and every committed brand alias written against it (D6.3a)."""
    aliases = review.read_brand_aliases(terms_file)
    target = review.normalize(ingredient)
    return [ingredient] + [a for a, g in aliases.items() if review.normalize(g) == target]


def reassignments(review_dir: Path, terms_file: Path) -> Reassignment:
    """D6.3 + D6.3a over all five review files: a HICL key with mapping_wrong true or an
    empty proposed_ingredients, whose note reads 'Correct ingredient: <x>.', where <x> is a
    generic of a pre-specified agent or class and every drug string of the key names it."""
    r = read_review(review_dir)
    generics = {review.normalize(g) for g in review.read_search_terms(terms_file)[0]}
    groups = review.term_groups(terms_file)
    out, unreadable = {}, []
    for _, k in r[r["key"].str.startswith("HICL:")].iterrows():
        if not (k["mapping_wrong"].strip().lower() == "true"
                or not k["proposed_ingredients"].strip()):
            continue
        m = NOTE.match(k["reviewer_note"].strip())
        if not m:
            unreadable.append(k["key"])
            continue
        ingredient = m.group(1).strip()
        exposure = groups.get(review.normalize(ingredient), "")
        if review.normalize(ingredient) not in generics or exposure not in EXPOSURES:
            continue
        strings = list(review.parse_counts(k["drugnames_medication"])) \
            + list(review.parse_counts(k["drugnames_admissiondrug"]))
        names = _names_ingredient(ingredient, terms_file)
        if strings and all(any(review.term_matches(n, s) for n in names) for s in strings):
            if k["decision"] != "keep":
                raise ValueError(f"{k['key']} meets D6.3 but its decision is "
                                 f"{k['decision']!r}; the protocol does not say what then")
            out[k["key"]] = exposure
    return Reassignment(out, sorted(unreadable))


def hicl_reviewed_rows(med: pd.DataFrame, review_dir: Path, terms_file: Path,
                       include_unsure: bool = False, named_only: bool = False,
                       reassigned: Reassignment | None = None) -> pd.DataFrame:
    """Arm 5: arm 3, with each reassigned key's rows assigned to its D6.3 exposure.
    The rows are taken exactly as arm 3 takes rows (D5.11 wrong strings out, D5.12)."""
    reassigned = reassigned or reassignments(review_dir, terms_file)
    base = hicl_rows(med, review_dir, include_unsure, named_only)
    moved = list(reassigned.exposures)  # the reassigned keys
    base = base[~base["key"].isin(moved)]
    keys = read_review(review_dir)
    keys = keys[keys["key"].isin(moved)].copy()
    keys["exposure"], keys["corrected_name"] = keys["key"].map(reassigned.exposures), ""
    hicl = _hicl_only(med)
    extra = review.exposure_rows(hicl, keys, include_unsure, named_only)
    extra = extra.assign(key=review._row_keys(extra))
    return pd.concat([base, extra])


# ---------------------------------------------------------------- route (D6.2) and stays


def apply_route(rows: pd.DataFrame, include_unknown: bool,
                table: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = routes.route_rows(rows)
    status = routes.systemic_status(rows, table)
    keep = status.isin(("yes", "unknown") if include_unknown else ("yes",))
    return rows[keep].assign(systemic=status[keep])


def stay_table(rows: pd.DataFrame, stays) -> pd.DataFrame:
    """One row per cohort stay, one boolean column per pre-specified exposure."""
    rows = rows[rows["patientunitstayid"].isin(set(stays))]
    if rows.empty:
        return pd.DataFrame(False, index=pd.Index(stays, name="patientunitstayid"),
                            columns=list(EXPOSURES))
    t = review.stay_exposure(rows, stays)
    return t.reindex(columns=list(EXPOSURES), fill_value=False).astype(bool)


def arm_rows(arm: str, variant: str, med: pd.DataFrame, review_dir: Path, terms_file: Path,
             reassigned: Reassignment | None = None) -> pd.DataFrame:
    """Counted rows of one arm and variant, after D6.4 and D6.2."""
    if arm not in ARMS or variant not in VARIANTS:
        raise ValueError(f"unknown arm/variant {arm}/{variant}")
    med = counting(med)
    unsure, named_only = variant == "unsure", variant == "named_only"
    if arm in ("generic_string", "generic_brand"):
        rows = string_rows(med, terms_file, with_brands=arm == "generic_brand")
    elif arm == "hicl":
        rows = hicl_rows(med, review_dir, unsure, named_only)
    elif arm == "hicl_gtc":
        check_gtc(med)
        rows = hicl_rows(med, review_dir, unsure, named_only)
    else:
        rows = hicl_reviewed_rows(med, review_dir, terms_file, unsure, named_only, reassigned)
    return apply_route(rows, include_unknown=variant == "route_unknown")


def primary_stays(tables: dict | None = None) -> pd.Index:
    """Section 6 primary estimand: analysis-cohort stays at medication-recording hospitals."""
    tables = tables or cohort.load_inputs()
    stays, _ = cohort.build_cohort(tables)
    hospitals = cohort.recording_hospitals(tables["patient"], tables["medication"])
    keep = stays[stays["hospitalid"].isin(hospitals)]
    return pd.Index(sorted(keep["patientunitstayid"]), name="patientunitstayid")


def ascertain(med: pd.DataFrame, stays, review_dir: Path = review.REVIEW_DIR,
              terms_file: Path | None = None) -> tuple[dict, Reassignment]:
    """{(arm, variant): stay x exposure table} over ``stays``, and the arm-5 reassignments."""
    terms_file = terms_file or review_dir / "search_terms.txt"
    reassigned = reassignments(review_dir, terms_file)
    tables = {(a, v): stay_table(arm_rows(a, v, med, review_dir, terms_file, reassigned), stays)
              for a in ARMS for v in VARIANTS}
    return tables, reassigned


# ---------------------------------------------------------------- report (D5.9, D6.6)


def prevalence(tables: dict) -> pd.DataFrame:
    rows = []
    for (arm, variant), t in tables.items():
        for exposure in EXPOSURES:
            rows.append({"arm": arm, "variant": variant, "exposure": exposure,
                         "exposed_stays": int(t[exposure].sum()), "cohort_stays": len(t)})
    return pd.DataFrame(rows)


def audit_misses(review_dir: Path) -> pd.DataFrame:
    """D6.6: audit keys with an empty proposed_ingredients whose corrected_name assigns them
    to a pre-specified agent or class."""
    r = read_review(review_dir, review.REVIEW_FILES[3:])
    hit = (r["proposed_ingredients"].str.strip() == "") & r["corrected_name"].isin(EXPOSURES)
    return r.loc[hit, ["file", "key", "corrected_name", "n_rows"]].reset_index(drop=True)


def report(prev: pd.DataFrame, reassigned: Reassignment, misses: pd.DataFrame) -> str:
    n = int(prev["cohort_stays"].iloc[0])
    lines = ["# Exposure ascertainment (generated by `python -m polysafe.ascertain`)", "",
             "PREREGISTRATION section 6, D5.9, D6.2-D6.7. Denominator: the section 6 primary "
             f"cohort, {n:,} stays. A stay is exposed when one of its rows counts (D6.4). "
             "No AKI label, outcome column or model output was read; the cohort comes from "
             "`cohort.build_cohort`.", "",
             "## Exposed stays by arm (primary variant)", ""]
    p = prev[prev["variant"] == "primary"].pivot(index="exposure", columns="arm",
                                                 values="exposed_stays")
    p = p.reindex(index=list(EXPOSURES), columns=list(ARMS))
    lines += ["| exposure | " + " | ".join(ARMS) + " |", "|---|" + "---:|" * len(ARMS)]
    lines += [f"| {e} | " + " | ".join(f"{int(v):,} ({v / n:.1%})" for v in p.loc[e]) + " |"
              for e in p.index]
    lines += ["", "Every arm and variant is in `results/exposure_prevalence.csv`. For the "
              "string arms the `unsure` and `named_only` variants equal `primary` by "
              "construction.", "",
              "## Arm 5 reassignments (D6.3, D6.3a)", ""]
    lines += [f"- `{k}` -> {e}" for k, e in sorted(reassigned.exposures.items())] or ["- none"]
    lines += ["", f"Eligible keys whose reviewer_note is not in the D6.3a form, so not "
              f"reassigned: {len(reassigned.unreadable)}"
              + (" (" + ", ".join(f"`{k}`" for k in reassigned.unreadable) + ")."
                 if reassigned.unreadable else "."), "",
              "## Audit-sample exposures missed by an empty mapping (D6.6)", "",
              f"{len(misses)} audit keys: "
              + ", ".join(f"`{r.key}` ({r.corrected_name}, {r.n_rows} rows)"
                          for r in misses.itertuples()) + "."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Exposure ascertainment, five arms.").parse_args(argv)
    tables_in = cohort.load_inputs()
    stays = primary_stays(tables_in)
    tables, reassigned = ascertain(load("medication"), stays)
    prev = prevalence(tables)
    RESULTS.mkdir(exist_ok=True)
    prev.to_csv(RESULTS / "exposure_prevalence.csv", index=False)
    md = report(prev, reassigned, audit_misses(review.REVIEW_DIR))
    (RESULTS / "ascertainment.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
