"""Every headline number in README.md is the committed result, never typed from memory."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
RES = ROOT / "results"


def test_primary_and_secondary_model_differences():
    comp = pd.read_csv(RES / "model_comparison.csv").set_index("spec")
    for spec in ("lr_reduced", "lgbm", "lr_full"):
        r = comp.loc[spec]
        assert f"{r['dauroc']:+.3f} ({r['lo']:.3f} to {r['hi']:.3f})" in README, spec


def test_sensitivity_numbers():
    s = pd.read_csv(RES / "sensitivity.csv").set_index("analysis")
    for key in ("hospital_clusters", "full_cohort", "baseline_c"):
        r = s.loc[key]
        assert f"{r['dauroc']:+.3f} ({r['lo']:.3f} to {r['hi']:.3f})" in README, key


def test_secondary_odds_ratios():
    t = pd.read_csv(RES / "secondary.csv").set_index("agent")
    for agent, r in t.iterrows():
        assert (f"{r['adjusted_or']:.2f} ({r['adjusted_lo']:.2f}-{r['adjusted_hi']:.2f})"
                in README), agent
