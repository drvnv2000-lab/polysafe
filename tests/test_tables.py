import pytest

from polysafe.tables import TABLES, load


@pytest.mark.parametrize("name", TABLES)
def test_every_table_loads_with_stay_ids(name):
    df = load(name, usecols=["patientunitstayid"])
    assert len(df) > 0
    assert df["patientunitstayid"].notna().all()


def test_patient_has_one_row_per_stay(tables):
    p = tables["patient"]
    assert len(p) == 2520
    assert p["patientunitstayid"].is_unique


def test_unknown_table_is_rejected():
    # hospital.csv.gz is empty in the demo and deliberately not loadable.
    with pytest.raises(ValueError, match="Unknown table"):
        load("hospital")
