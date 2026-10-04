"""Step 3: first-pass medication mapping for PI review (PREREGISTRATION section 6).

Builds a frequency-ranked draft that maps each medication key to RxNorm
ingredients and ATC classes:

* key = HICL code (``drughiclseqno``); its name comes from ``medication``, or,
  if never named there, from ``admissiondrug`` (same coding; flagged);
* key = drug name, for rows without a HICL code.

Match status, never hidden:
  exact        the cleaned name matched an RxNorm string exactly (normalised)
  approximate  RxNorm's approximate-term search; MUST be reviewed by the PI
  unmapped     no name anywhere, or RxNorm returned nothing; only ``gtc`` remains

Not used: RxNorm's ``HIC_SEQN`` identifier. It is a different FDB code system
from eICU's ``drughiclseqno`` (HICL_SEQNO): looking eICU codes up as HIC_SEQN
maps furosemide to enoxacin, vancomycin to corn syrup, ondansetron to
etoposide phosphate.

Cancelled orders are not counted. No outcome data is read.

Every RxNorm response is cached in ``mapping/rxnorm_cache.json`` so the draft
is reproducible offline.

Usage:
    python -m polysafe.medmap    # writes mapping/hicl_draft.csv and mapping/MAPPING_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from polysafe.tables import load

RXNAV = "https://rxnav.nlm.nih.gov/REST"
MAPPING_DIR = Path(__file__).resolve().parents[2] / "mapping"

# Review checklist only: draft ingredient search terms for the pre-specified agents and
# classes (sections 6 and 10). NOT a classification; the PI classifies. ATC from RxClass
# is not used here because it includes the classes of combination products containing an
# ingredient (e.g. diphenhydramine is returned under M01A NSAIDs via ibuprofen combinations).
CHECKLIST = {
    "vancomycin": ("vancomycin",),
    "piperacillin-tazobactam": ("piperacillin", "tazobactam"),
    "furosemide": ("furosemide",),
    "ondansetron (negative control)": ("ondansetron",),
    "NSAIDs": ("ibuprofen", "ketorolac", "naproxen", "meloxicam", "celecoxib", "indomethacin",
               "diclofenac", "aspirin", "nabumetone", "etodolac", "sulindac", "piroxicam"),
    "ACE inhibitors": ("lisinopril", "enalapril", "enalaprilat", "captopril", "ramipril",
                       "benazepril", "quinapril", "fosinopril", "trandolapril", "moexipril",
                       "perindopril"),
    "ARBs": ("losartan", "valsartan", "irbesartan", "olmesartan", "telmisartan", "candesartan",
             "azilsartan", "eprosartan"),
    "aminoglycosides": ("gentamicin", "tobramycin", "amikacin", "streptomycin", "neomycin"),
    "loop diuretics": ("furosemide", "bumetanide", "torsemide", "ethacrynic acid"),
    "iodinated contrast": ("iohexol", "iopamidol", "iodixanol", "ioversol", "iopromide",
                           "ioxaglate", "diatrizoate"),
}

_FORM_WORDS = (
    "tab tabs tablet tablets cap caps capsule capsules inj injection ij soln solution susp "
    "suspension syrup elixir po iv ivpb im sq sc subq top topical er xr sr ec dr cr odt sdv mdv "
    "pf flex cont container bag vial syringe premix lvp mbp piggyback ud unit dose liquid oral "
    "cream oint ointment gel patch neb nebulizer inh inhaler mdi dpi spray nasal ophth opth "
    "drop drops supp suppository chewable chew packet pwd powder kit "
    "intravenous"
).split()
_UNIT = r"(?:mg|mcg|g|gm|gram|ml|l|meq|mmol|unit|units|u|%|mg/ml|mcg/ml|unit/ml|units/ml|mg/kg)"


def clean_name(name: str) -> str:
    """'1000 ML FLEX CONT : SODIUM CHLORIDE 0.9 % IV SOLN' -> 'sodium chloride'."""
    s = str(name)
    if " : " in s:
        s = s.split(" : ", 1)[1]
    s = re.sub(r"^\s*[\d.,]+\s*ml\s+-\s+", "", s, flags=re.I)
    s = s.lower()
    s = re.sub(rf"\b[\d.,/]+\s*{_UNIT}(?:\s*/\s*[\d.,]*\s*{_UNIT})*\b", " ", s)
    s = re.sub(r"[\d.,]+", " ", s)
    s = re.sub(r"[^a-z/\- ]", " ", s)
    words = [w for w in s.split() if w not in _FORM_WORDS and w not in {"-", "/"}]
    return " ".join(words).strip(" -/")


class RxNorm:
    """Tiny cached client for the four RxNav calls the draft needs."""

    def __init__(self, cache_path: Path, offline: bool = False):
        self.path = cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self.offline = offline

    def get(self, url: str) -> dict:
        if url not in self.cache:
            if self.offline:
                raise KeyError(f"not cached and offline: {url}")
            r = requests.get(RXNAV + url, timeout=30)
            r.raise_for_status()
            self.cache[url] = r.json()
            time.sleep(0.06)  # RxNav asks for <= 20 requests per second
        return self.cache[url]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache, indent=0, sort_keys=True))

    def lookup(self, query: str) -> dict:
        """Exact (normalised) match first, then approximate. Returns match info."""
        if not query:
            return {"match": "unmapped"}
        ids = self.get(f"/rxcui.json?name={quote(query)}&search=2")
        ids = (ids.get("idGroup") or {}).get("rxnormId") or []
        match, score = "exact", None
        if not ids:
            cands = self.get(f"/approximateTerm.json?term={quote(query)}&maxEntries=1")
            cands = (cands.get("approximateGroup") or {}).get("candidate") or []
            if not cands:
                return {"match": "unmapped"}
            ids, match, score = [cands[0]["rxcui"]], "approximate", float(cands[0]["score"])
        rxcui = ids[0]
        props = self.get(f"/rxcui/{rxcui}/properties.json").get("properties") or {}
        if props.get("tty") == "IN":
            ingredients = [(rxcui, props.get("name", ""))]
        else:
            rel = self.get(f"/rxcui/{rxcui}/related.json?tty=IN")
            groups = (rel.get("relatedGroup") or {}).get("conceptGroup") or []
            ingredients = [(c["rxcui"], c["name"]) for g in groups
                           for c in g.get("conceptProperties", [])]
        atc = {}
        for cui, _ in ingredients:
            cls = self.get(f"/rxclass/class/byRxcui.json?rxcui={cui}&relaSource=ATC")
            for item in (cls.get("rxclassDrugInfoList") or {}).get("rxclassDrugInfo", []):
                c = item["rxclassMinConceptItem"]
                atc[c["classId"]] = c["className"]
        return {"match": match, "score": score, "rxcui": rxcui,
                "rxnorm_name": props.get("name", ""),
                "ingredients": " + ".join(sorted({n.lower() for _, n in ingredients})),
                "atc": " | ".join(sorted(atc)),
                "atc_names": " | ".join(atc[k] for k in sorted(atc))}


def _top_names(s: pd.Series, k: int = 3) -> str:
    vc = s.dropna().value_counts().head(k)
    return " | ".join(f"{n} ({c})" for n, c in vc.items())


def medication_keys(med: pd.DataFrame, admission_drug: pd.DataFrame) -> pd.DataFrame:
    """One row per mapping key with frequencies, observed names and gtc, ranked by stays."""
    m = med[med["drugordercancelled"] != "Yes"]
    ad_names = admission_drug.dropna(subset=["drughiclseqno", "drugname"])
    rows = []
    for hicl, g in m.dropna(subset=["drughiclseqno"]).groupby("drughiclseqno"):
        names = g["drugname"].dropna()
        source = "medication"
        if names.empty:
            names = ad_names.loc[ad_names["drughiclseqno"] == hicl, "drugname"]
            source = "admissiondrug" if not names.empty else "none"
        rows.append({"key_type": "HICL", "key": str(int(hicl)), "n_stays": g["patientunitstayid"]
                     .nunique(), "n_rows": len(g), "pct_rows_named": g["drugname"].notna().mean(),
                     "name_source": source, "top_names": _top_names(names),
                     "query": clean_name(names.mode().iloc[0]) if not names.empty else "",
                     "gtc": g["gtc"].mode().iloc[0], "route": g["routeadmin"].mode().iloc[0]
                     if g["routeadmin"].notna().any() else ""})
    no_hicl = m[m["drughiclseqno"].isna() & m["drugname"].notna()]
    for name, g in no_hicl.groupby("drugname"):
        rows.append({"key_type": "NAME", "key": name, "n_stays": g["patientunitstayid"].nunique(),
                     "n_rows": len(g), "pct_rows_named": 1.0, "name_source": "medication",
                     "top_names": f"{name} ({len(g)})", "query": clean_name(name),
                     "gtc": g["gtc"].mode().iloc[0], "route": g["routeadmin"].mode().iloc[0]
                     if g["routeadmin"].notna().any() else ""})
    keys = pd.DataFrame(rows).sort_values(["n_stays", "n_rows"], ascending=False)
    return keys.reset_index(drop=True)


def draft(keys: pd.DataFrame, rx: RxNorm) -> pd.DataFrame:
    found = {q: rx.lookup(q) for q in keys["query"].unique()}
    cols = ["match", "score", "rxcui", "rxnorm_name", "ingredients", "atc", "atc_names"]
    out = keys.join(pd.DataFrame([found[q] for q in keys["query"]], index=keys.index,
                                 columns=cols))
    out["match"] = out["match"].fillna("unmapped")
    first_word = out["ingredients"].fillna("").str.split(" ").str[0]
    out["ingredient_in_name"] = [bool(w) and w in str(n).lower()
                                 for w, n in zip(first_word, out["top_names"], strict=True)]
    for col in ("pi_ingredient", "pi_class", "pi_reviewed", "pi_notes"):
        out[col] = ""
    return out


def report(d: pd.DataFrame, med: pd.DataFrame) -> str:
    m = med[med["drugordercancelled"] != "Yes"]
    key_of = m["drughiclseqno"].map(lambda v: f"HICL:{int(v)}" if pd.notna(v) else None)
    key_of = key_of.fillna("NAME:" + m["drugname"].fillna(""))
    status = dict(zip(d["key_type"] + ":" + d["key"], d["match"], strict=True))
    row_status = key_of.map(status).fillna("unmapped")  # no HICL and no name
    lines = ["# Medication mapping draft (generated by `python -m polysafe.medmap`)", "",
             "**Status: DRAFT for PI review.** Nothing here is accepted until the PI fills "
             "`pi_ingredient` / `pi_class` / `pi_reviewed` in `mapping/hicl_draft.csv`.", "",
             "Method: one key per HICL code (or per drug name when the row has no HICL). "
             "The most frequent observed name is cleaned of strengths and dose forms and "
             "looked up in RxNorm (exact normalised match, else approximate search), then "
             "mapped to ingredients and ATC classes. RxNorm's `HIC_SEQN` is NOT used: it is "
             "a different code system (eICU's furosemide code resolves to enoxacin there).", "",
             f"Non-cancelled medication rows: {len(m):,}; keys: {len(d):,} "
             f"({(d['key_type'] == 'HICL').sum():,} HICL, {(d['key_type'] == 'NAME').sum():,} "
             "name-only).", "",
             "| Match | Keys | Rows | Share of rows |", "|---|---:|---:|---:|"]
    for s in ("exact", "approximate", "unmapped"):
        k, r = int((d["match"] == s).sum()), int((row_status == s).sum())
        lines.append(f"| {s} | {k:,} | {r:,} | {r / len(m):.1%} |")
    named_via_ad = d[d["name_source"] == "admissiondrug"]
    n_ad = int(named_via_ad["n_rows"].sum())
    lines += ["", f"{len(named_via_ad):,} HICL keys ({n_ad:,} rows) are named only through "
              "`admissiondrug` (home medications); check those names apply.",
              "", "**The `atc` columns are hints only.** RxClass returns every ATC class whose "
              "products contain an ingredient, combinations included (diphenhydramine comes "
              "back under M01A NSAIDs via ibuprofen combinations). Do not read them as a "
              "drug's own class.", "",
              "## Pre-specified agents and classes: keys to review", "",
              "Draft ingredient search terms, not a classification. Rows and stays are summed "
              "over keys (a stay can appear under several keys).", "",
              "| Agent / class | Keys | Rows | Stays (sum) | Matches | Ingredients found |",
              "|---|---:|---:|---:|---|---|"]
    ing_sets = d["ingredients"].fillna("").str.split(r" \+ ")
    for label, terms in CHECKLIST.items():
        sel = d[ing_sets.map(lambda xs, t=terms: any(x in t for x in xs))]
        found = sorted({x for xs in ing_sets[sel.index] for x in xs if x in terms})
        lines.append(f"| {label} | {len(sel)} | {int(sel['n_rows'].sum()):,} | "
                     f"{int(sel['n_stays'].sum()):,} | {', '.join(sorted(set(sel['match'])))} | "
                     f"{', '.join(found) or 'none in the data'} |")
    lines += ["", "## Top 40 keys by stays", "",
              "| Key | Stays | Rows | Observed names | Match | Ingredients | ATC |",
              "|---|---:|---:|---|---|---|---|"]
    for _, r in d.head(40).iterrows():
        lines.append(f"| {r['key_type']} {r['key']} | {r['n_stays']} | {r['n_rows']} | "
                     f"{str(r['top_names']).replace('|', '/')} | {r['match']} | "
                     f"{r['ingredients'] if pd.notna(r['ingredients']) else ''} | "
                     f"{r['atc'] if pd.notna(r['atc']) else ''} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Draft HICL -> ingredient mapping for PI review.")
    p.add_argument("--out", type=Path, default=MAPPING_DIR)
    p.add_argument("--offline", action="store_true", help="use only the cached RxNorm responses")
    args = p.parse_args(argv)

    med = load("medication")
    keys = medication_keys(med, load("admissiondrug", usecols=["drughiclseqno", "drugname"]))
    rx = RxNorm(args.out / "rxnorm_cache.json", offline=args.offline)
    try:
        d = draft(keys, rx)
    finally:
        rx.save()
    args.out.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out / "hicl_draft.csv", index=False)
    md = report(d, med)
    (args.out / "MAPPING_REPORT.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
