import pandas as pd
import pytest

from polysafe import routes
from polysafe.medmap import MAPPING_DIR
from polysafe.review import _row_keys

REVIEWABLE = ("tier1_agents.csv", "tier2_classes.csv", "tier3_screen.csv",
              "audit_head.csv", "audit_tail.csv")


@pytest.fixture(scope="module")
def reviewed_rows():
    """Non-cancelled medication rows of the 171 reviewable keys."""
    from polysafe.tables import load
    keys = set(pd.concat([pd.read_csv(MAPPING_DIR / "review" / f, dtype=str,
                                      keep_default_na=False)["key"] for f in REVIEWABLE]))
    assert len(keys) == 171
    med = load("medication")
    live = med[med["drugordercancelled"] != "Yes"]
    return live[_row_keys(live).isin(keys)]


def test_every_value_in_the_reviewed_rows_resolves_to_exactly_one_category(reviewed_rows):
    table = routes.read_table()
    mapping = routes.load_routes()
    cats = routes.route_category(reviewed_rows["routeadmin"], mapping)
    assert cats.isin(routes.CATEGORIES).all()
    assert len(cats) == len(reviewed_rows) == table["row_count"].sum()
    # one row per raw spelling, and the recorded counts are the data's counts
    assert table["raw_value"].is_unique
    observed = reviewed_rows["routeadmin"].value_counts()
    assert dict(zip(table["raw_value"], table["row_count"], strict=True)) == observed.to_dict()


def test_an_unrecognised_value_raises_rather_than_defaulting_to_unknown():
    mapping = routes.load_routes()
    with pytest.raises(ValueError, match="'Not a route'"):
        routes.route_category(pd.Series(["IV", "Not a route"]), mapping)
    with pytest.raises(ValueError, match="not in route_normalization.csv"):  # missing value
        routes.route_category(pd.Series(["IV", float("nan")]), mapping)
    with pytest.raises(ValueError, match="'iv'"):  # matching is case-sensitive
        routes.route_category(pd.Series(["iv"]), mapping)


def test_trailing_space_entries_match_after_stripping():
    table = routes.read_table()
    padded = table[table["raw_value"] != table["raw_value"].str.rstrip()]
    assert len(padded) == 9
    assert padded["note"].str.contains("matched after stripping").all()
    got = routes.route_category(padded["raw_value"])
    assert got.tolist() == padded["route_category"].tolist()
    stripped = routes.route_category(padded["raw_value"].str.rstrip())
    assert stripped.tolist() == padded["route_category"].tolist()


def test_tf_and_ft_resolve_to_the_same_category():
    got = routes.route_category(pd.Series(["TF", "FT"]))
    assert got.nunique() == 1


def test_the_table_rejects_a_category_outside_the_vocabulary(tmp_path):
    bad = routes.read_table().head(2).assign(route_category=["parenteral_iv", "systemic"])
    bad.to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(ValueError, match="systemic"):
        routes.read_table(tmp_path / "t.csv")


def test_spellings_that_strip_alike_must_share_a_category(tmp_path):
    t = pd.DataFrame({"raw_value": ["IV", "IV  "], "route_category": ["parenteral_iv", "unknown"],
                      "row_count": [1, 1], "note": ["", ""]})
    t.to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(ValueError, match="different categories"):
        routes.load_routes(tmp_path / "t.csv")


def test_route_rows_keeps_medicationid_for_provenance(reviewed_rows):
    sample = reviewed_rows.head(50)
    out = routes.route_rows(sample)
    assert out["medicationid"].tolist() == sample["medicationid"].tolist()
    with pytest.raises(ValueError, match="medicationid"):
        routes.route_rows(sample.drop(columns="medicationid"))
