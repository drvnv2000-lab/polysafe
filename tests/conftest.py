"""Tests run against the real eICU demo tables in polysafe/data (SHA-256 verified)."""

from __future__ import annotations

import pytest

from polysafe import cohort


@pytest.fixture(scope="session")
def tables():
    return cohort.load_inputs()


@pytest.fixture(scope="session")
def cr(tables):
    return cohort.serum_creatinine(tables["lab"])
