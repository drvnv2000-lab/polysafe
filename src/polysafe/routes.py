"""Route normalization for ``medication.routeadmin`` (``mapping/route_normalization.csv``).

A lookup only: each raw routeadmin spelling maps to one route category. The table covers
every value in the non-cancelled rows of the 171 reviewed keys; each raw spelling is its own
row. A value is matched with its trailing whitespace stripped. A value missing from the
table raises; nothing defaults to ``unknown``.

The systemic-route table (``mapping/systemic_routes.csv``) gives a systemic status
(yes | no | unknown) for every (exposure, route category) pair, with its basis. The
determinations are the project owner's; this module only loads and validates the table,
and refuses one that is incomplete, so a missing pair cannot be read as "no".
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

from polysafe.medmap import MAPPING_DIR
from polysafe.review import AGENTS

ROUTE_TABLE = MAPPING_DIR / "route_normalization.csv"
COLUMNS = ("raw_value", "route_category", "row_count", "note")
CATEGORIES = (
    "parenteral_iv", "parenteral_im", "parenteral_sc", "parenteral_other",
    "parenteral_unspecified", "enteral_oral", "enteral_tube", "sublingual", "intranasal",
    "topical", "transdermal", "ophthalmic", "otic", "inhaled", "rectal", "oral_local", "unknown",
)

SYSTEMIC_TABLE = MAPPING_DIR / "systemic_routes.csv"
SYSTEMIC_COLUMNS = ("exposure", "route_category", "systemic_status", "basis")
STATUSES = ("yes", "no", "unknown")
# The pre-specified class set (D6.0c), labelled as in the review packet (review.CLASSES).
PRESPECIFIED_CLASSES = ("NSAIDs (aspirin excluded, D4.1)", "iodinated contrast",
                        "aminoglycosides")
EXPOSURES = (*AGENTS, *PRESPECIFIED_CLASSES)


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


def read_systemic(path: Path | None = None) -> pd.DataFrame:
    """The systemic-route table, validated: exactly one row for every (exposure, route
    category) pair, a status from STATUSES and a non-empty basis on each."""
    t = pd.read_csv(path or SYSTEMIC_TABLE, dtype=str, keep_default_na=False)
    if tuple(t.columns) != SYSTEMIC_COLUMNS:
        raise ValueError(f"systemic table columns must be {SYSTEMIC_COLUMNS}, "
                         f"got {tuple(t.columns)}")
    bad = sorted(set(t["systemic_status"]) - set(STATUSES))
    if bad:
        raise ValueError(f"systemic_status outside {STATUSES}: {bad}")
    pairs = list(zip(t["exposure"], t["route_category"], strict=True))
    expected = set(product(EXPOSURES, CATEGORIES))
    extra = sorted(set(pairs) - expected)
    if extra:
        raise ValueError(f"pairs outside the exposures x categories grid: {extra}")
    dup = sorted({p for p in pairs if pairs.count(p) > 1})
    if dup:
        raise ValueError(f"pairs listed more than once: {dup}")
    missing = sorted(expected - set(pairs))
    if missing:
        raise ValueError(f"pairs missing from the systemic table: {missing}")
    if (t["basis"].str.strip() == "").any():
        raise ValueError("every row needs a basis")
    return t


def systemic_status(rows: pd.DataFrame, table: pd.DataFrame | None = None) -> pd.Series:
    """Systemic status of each row from its ``exposure`` and ``route_category``. Raises on a
    pair the table does not hold."""
    t = read_systemic() if table is None else table
    status = t.set_index(["exposure", "route_category"])["systemic_status"]
    idx = pd.MultiIndex.from_arrays([rows["exposure"], rows["route_category"]])
    out = pd.Series(status.reindex(idx).to_numpy(), index=rows.index)
    if out.isna().any():
        unseen = sorted(set(idx[out.isna().to_numpy()]))
        raise ValueError(f"(exposure, route_category) pairs not in the systemic table: {unseen}")
    return out
