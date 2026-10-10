"""D12.1 final model. Synthetic data only."""

import numpy as np
import pandas as pd

from polysafe import final_model, models


def test_chosen_c_is_the_mode_with_ties_to_the_smaller():
    t = pd.DataFrame({"spec": "lr_reduced", "medication": False,
                      "param": ["0.1", "0.01", "0.1", "0.01", "1.0"]})
    assert final_model.chosen_c(t, False) == 0.01


def test_final_fit_names_every_design_column():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    X.loc[::7, "b"] = np.nan
    for c in models.MED_COLS:
        X[c] = (rng.random(n) < 0.2).astype(float)
    X.loc[:4, list(models.MED_COLS)] = np.nan  # record-less stays (D7.1)
    y = (rng.random(n) < 0.3).astype(int)
    spec = models.Spec("toy", "lr", ("a", "b"), (("b",),))
    t = final_model.fit_final(X, y, spec, True, 1.0)
    assert list(t["term"]) == ["(intercept)", "a", "b", "missing(b)", *models.MED_COLS]
    assert t["coefficient"].notna().all()
    assert np.isclose(t.loc[t["term"] == "a", "sd"].iloc[0], X["a"].std(ddof=0))
