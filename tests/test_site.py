"""The public page embeds the committed results, never typed numbers."""

import json
import re

import pandas as pd

from polysafe import site


def test_page_data_is_the_committed_results():
    html = site.build()
    data = json.loads(re.search(r"const D = (\{.*?\});\n", html, re.S).group(1))
    comp = pd.read_csv(site.RESULTS / "model_comparison.csv").set_index("spec")
    assert abs(data["models"][0]["d"] - comp.loc["lr_reduced", "dauroc"]) < 1e-6
    assert data["flow"][0]["stays"] == 2520 and data["flow"][-1]["stays"] == 671
    assert "__DATA__" not in html and "p_lr_reduced" not in html  # no per-stay predictions
