"""README figures are built from the committed results, never typed in."""

import pandas as pd

from polysafe import figures


def test_cohort_flow_matches_cohort_flow_md():
    md = (figures.RESULTS / "cohort_flow.md").read_text(encoding="utf-8")
    t = figures.flow_table(md)
    assert t["stays"].iloc[0] == 2520 and t["stays"].iloc[-1] == 671
    assert t["events"].iloc[-1] == 78
    # every step's removal is the drop in stays
    assert (t["stays"].shift(1) - t["stays"]).iloc[1:].tolist() == t["removed"].iloc[1:].tolist()
    svg = figures.cohort_flow(t)
    assert "Primary analysis cohort: 671 stays" in svg and "78 AKI events" in svg


def test_figures_carry_the_committed_numbers():
    out = figures.build()
    comp = pd.read_csv(figures.RESULTS / "model_comparison.csv").set_index("spec")
    d = comp.loc["lr_reduced"]
    assert f"{d['dauroc']:+.3f} ({d['lo']:+.3f} to {d['hi']:+.3f})" in out["dauroc.svg"]
    m = pd.read_csv(figures.RESULTS / "model_metrics.csv")
    r = m[(m["spec"] == "lr_reduced") & m["medication"]].iloc[0]
    assert f"slope {r['cal_slope']:.2f}" in out["calibration.svg"]
    assert all(svg.startswith("<svg") for svg in out.values())
