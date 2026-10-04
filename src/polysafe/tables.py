"""Loaders for the eICU-CRD Demo v2.0.1 tables in ``polysafe/data``.

One function, one table per call. All offsets in eICU are minutes relative to
ICU unit admission (negative = before admission).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

TABLES = (
    "patient",
    "lab",
    "medication",
    "infusiondrug",
    "admissiondrug",
    "diagnosis",
    "apachePatientResult",
    "treatment",
    "intakeOutput",
    "vitalAperiodic",
    "pastHistory",
    "apacheApsVar",
    "apachePredVar",
    "admissionDx",
    "vitalPeriodic",  # added by PREREGISTRATION D3.2
)


def load(table: str, usecols: list[str] | None = None,
         data_dir: Path | None = None) -> pd.DataFrame:
    """Read one eICU table, e.g. ``load("lab", usecols=["patientunitstayid", "labname"])``."""
    if table not in TABLES:
        raise ValueError(f"Unknown table {table!r}. Known: {', '.join(TABLES)}")
    path = Path(data_dir or DATA_DIR) / f"{table}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; see DATA_REPORT.md for how it was fetched.")
    return pd.read_csv(path, usecols=usecols, low_memory=False)
