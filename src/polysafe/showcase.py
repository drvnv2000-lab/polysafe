"""Data for the public page's "pharmacist view" (``results/showcase.json``).

For each primary-model exposure, on the primary cohort: how many exposed stays each way of
counting finds, and the most frequent order strings behind the HICL + gtc exposures, with
whether a plain drug-name text search would find them. Order strings and code numbers
only; no patient-level field leaves this module.

Usage:
    python -m polysafe.showcase
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polysafe import ascertain, models, review
from polysafe.tables import load

RESULTS = Path("results")
TOP = 6


def build() -> dict:
    med = load("medication")
    stays = set(ascertain.primary_stays())
    terms = review.REVIEW_DIR / "search_terms.txt"
    arms = {a: ascertain.arm_rows(a, "primary", med, review.REVIEW_DIR, terms)
            for a in ("generic_string", "generic_brand", models.ARM)}
    arms = {a: r[r["patientunitstayid"].isin(stays)] for a, r in arms.items()}
    out = []
    for exposure in models.MEDICATION:
        sel = {a: r[r["exposure"] == exposure] for a, r in arms.items()}
        hicl, text = sel[models.ARM], sel["generic_string"]
        text_ids = set(text["medicationid"])
        label = hicl["drugname"].where(hicl["drugname"].notna(), "No drug name, HICL code "
                                       + hicl["drughiclseqno"].astype("Int64").astype(str))
        g = hicl.assign(label=label, found=hicl["medicationid"].isin(text_ids)).groupby("label")
        top = (g.agg(rows=("medicationid", "size"), found=("found", "mean"))
               .sort_values("rows", ascending=False).head(TOP))
        out.append({"exposure": exposure.split(" (")[0],
                    "stays": {a: int(r["patientunitstayid"].nunique()) for a, r in sel.items()},
                    "strings": [{"label": k, "rows": int(v.rows),
                                 "text_finds": bool(v.found > .5)} for k, v in top.iterrows()]})
    return {"cohort_stays": len(stays), "exposures": out}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Pharmacist view for the public page.").parse_args(argv)
    data = build()
    (RESULTS / "showcase.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(json.dumps(data, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
