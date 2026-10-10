"""The pharmacist view agrees with the committed exposure prevalence."""

import json

import pandas as pd

from polysafe import models

RES = models.RESULTS


def test_showcase_counts_match_exposure_prevalence():
    show = json.loads((RES / "showcase.json").read_text(encoding="utf-8"))
    prev = pd.read_csv(RES / "exposure_prevalence.csv")
    for e in show["exposures"]:
        for arm, n in e["stays"].items():
            row = prev[(prev["arm"] == arm) & (prev["variant"] == "primary")
                       & prev["exposure"].str.startswith(e["exposure"])]
            assert int(row["exposed_stays"].iloc[0]) == n, (arm, e["exposure"])
