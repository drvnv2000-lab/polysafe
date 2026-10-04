"""Route normalization for ``medication.routeadmin`` (``mapping/route_normalization.csv``).

A lookup only: each raw routeadmin spelling maps to one route category. The table covers
every value in the non-cancelled rows of the 171 reviewed keys; each raw spelling is its own
row. A value is matched with its trailing whitespace stripped. A value missing from the
table raises; nothing defaults to ``unknown``.

No systemic determination is made here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from polysafe.medmap import MAPPING_DIR

ROUTE_TABLE = MAPPING_DIR / "route_normalization.csv"
COLUMNS = ("raw_value", "route_category", "row_count", "note")
CATEGORIES = (
    "parenteral_iv", "parenteral_im", "parenteral_sc", "parenteral_other",
    "parenteral_unspecified", "enteral_oral", "enteral_tube", "sublingual", "intranasal",
    "topical", "transdermal", "ophthalmic", "otic", "inhaled", "rectal", "oral_local", "unknown",
)


def read_table(path: Path | None = None) -> pd.DataFrame:
    """The table as committed: raw spellings (trailing spaces kept), category, rows, note."""
    t = pd.read_csv(path or ROUTE_TABLE, dtype=str, keep_default_na=False)
    if tuple(t.columns) != COLUMNS:
        raise ValueError(f"route table columns must be {COLUMNS}, got {tuple(t.columns)}")
    bad = sorted(set(t["route_category"]) - set(CATEGORIES))
    if bad:
        raise ValueError(f"route categories outside the vocabulary: {bad}")
    return t.assign(row_count=t["row_count"].astype(int))


def load_routes(path: Path | None = None) -> dict[str, str]:
    """Stripped raw value -> route category. Raises if two spellings that strip to the
    same value carry different categories."""
    t = read_table(path)
    key = t["raw_value"].str.rstrip()
    cats = t.groupby(key)["route_category"].unique()
    clash = {k: sorted(v) for k, v in cats.items() if len(v) > 1}
    if clash:
        raise ValueError(f"raw values that strip alike map to different categories: {clash}")
    return {k: v[0] for k, v in cats.items()}


def route_category(values: pd.Series, routes: dict[str, str] | None = None) -> pd.Series:
    """Route category of each raw routeadmin value. Raises on any value not in the table,
    including a missing one."""
    routes = load_routes() if routes is None else routes
    key = values.map(lambda v: v.rstrip() if isinstance(v, str) else v)
    out = key.map(routes)
    if out.isna().any():
        unseen = sorted({repr(v) for v in values[out.isna()].unique()})
        raise ValueError(f"routeadmin values not in {ROUTE_TABLE.name}: {', '.join(unseen)}")
    return out


def route_rows(med: pd.DataFrame, routes: dict[str, str] | None = None) -> pd.DataFrame:
    """``med`` with ``route_category`` added. ``medicationid`` is required and kept, so every
    downstream exposure row traces back to its source row."""
    if "medicationid" not in med.columns:
        raise ValueError("medicationid is required for provenance")
    return med.assign(route_category=route_category(med["routeadmin"], routes))
