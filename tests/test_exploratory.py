"""D10.1 marker. Synthetic data only."""

import numpy as np
import pandas as pd

from polysafe import exploratory


def test_marker_counts_only_unnamed_hicl_rows_active_by_t():
    med = pd.DataFrame({
        "patientunitstayid": [1, 2, 2, 3, 4],
        "drugname": [np.nan, "FUROSEMIDE", "", np.nan, np.nan],
        "drughiclseqno": [3660.0, 3660.0, 3660.0, np.nan, 3660.0],
        "drugstartoffset": [10, 10, 2000, 10, 10],  # stay 2's unnamed row starts after T
        "drugstopoffset": [100, 100, 3000, 100, 100],
        "drugordercancelled": ["No", "No", "No", "No", "Yes"],
    })
    u = exploratory.marker(med, pd.Index([1, 2, 3, 4, 5], name="patientunitstayid"))
    assert u.loc[1] == 1.0  # unnamed, HICL, active by T
    assert u.loc[2] == 0.0  # named row only by T
    assert u.loc[3] == 0.0  # unnamed but no HICL
    assert u.loc[4] == 0.0  # cancelled
    assert np.isnan(u.loc[5])  # no medication record: missing, never 0


def test_md_table():
    t = exploratory.md_table(pd.DataFrame({"a": [1], "b": ["x"]}))
    assert t.splitlines() == ["| a | b |", "|---|---|", "| 1 | x |"]
