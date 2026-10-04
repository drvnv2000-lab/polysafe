"""The validation fold structure (PREREGISTRATION section 7).

Repeated stratified grouped k-fold with the patient (``uniquepid``) as the
group: no patient ever appears in both a training and a test fold. The power
simulation (section 9) and every model use this one function, so the
simulated design and the analysed design are the same design.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

N_SPLITS = 5
N_REPEATS = 10


def patient_folds(y, groups, n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS,
                  seed: int = 0) -> np.ndarray:
    """Fold id per stay for each repeat: array of shape (n_repeats, n_stays)."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    out = np.empty((n_repeats, len(y)), dtype=np.int8)
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (_, test) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            out[r, test] = k
    return out
