import pandas as pd
import pytest

from polysafe import review, routes
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


def test_radinj_resolves_as_inj_and_injection_do():
    got = routes.route_category(pd.Series(["RADINJ", "INJ", "Injection"]))
    assert got.tolist() == ["parenteral_unspecified"] * 3


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


# ---------------------------------------------------------------- systemic-route table

P1_CATS = ("parenteral_iv", "parenteral_im", "parenteral_sc", "parenteral_other",
           "parenteral_unspecified")
P2_CATS = ("topical", "transdermal", "ophthalmic", "otic", "oral_local", "inhaled", "intranasal")
NO_DETERMINATION = "no determination made; no rows present in the data"
NSAID = "NSAIDs (aspirin excluded, D4.1)"
P4 = ("P4 - enteral_tube inherits the enteral_oral determination for the same exposure. Tube "
      "delivery places drug in the same gastrointestinal tract as oral delivery; the absorption "
      "pathway is identical and the delivery device does not alter it. This is not a separate "
      "determination.")
# The project owner's determinations, recorded verbatim (2026-10-05).
OWNER = {
    ("vancomycin", "enteral_oral"): ("no", "bioavailability <5%; the oral product treats "
                                           "C. difficile colitis, not systemic infection"),
    ("vancomycin", "enteral_tube"): ("no", "enteral delivery, same absorption as oral"),
    ("furosemide", "enteral_oral"): ("yes", "oral bioavailability ~50%, variable"),
    ("furosemide", "enteral_tube"): ("yes", "enteral absorption as oral"),
    ("ondansetron", "enteral_oral"): ("yes", "oral bioavailability ~60%"),
    ("ondansetron", "enteral_tube"): ("yes", "enteral absorption as oral"),
    ("ondansetron", "sublingual"): ("yes", "systemic transmucosal route, bypasses first pass"),
    (NSAID, "enteral_oral"): ("yes", "well absorbed orally"),
    (NSAID, "rectal"): ("yes", "rectal administration produces systemic NSAID exposure"),
    ("iodinated contrast", "enteral_oral"): ("no", "oral contrast is for GI imaging, "
                                                   "minimally absorbed"),
    ("iodinated contrast", "enteral_tube"): ("no", "enteral delivery, as oral"),
    ("piperacillin-tazobactam", "enteral_oral"): ("no", "no oral formulation exists"),
    ("piperacillin-tazobactam", "enteral_tube"): ("no", "no oral formulation exists"),
    ("aminoglycosides", "enteral_oral"): ("no", "oral aminoglycosides (neomycin, paromomycin) "
                                                "are minimally absorbed and used for gut "
                                                "decontamination"),
    ("aminoglycosides", "enteral_tube"): ("no", "enteral delivery, as oral"),
}


def test_the_systemic_table_is_complete_without_duplicates_and_statuses_are_valid():
    t = routes.read_systemic()
    assert len(t) == 7 * 17 == 119
    pairs = list(zip(t["exposure"], t["route_category"], strict=True))
    assert len(set(pairs)) == len(pairs)
    assert set(pairs) == {(e, c) for e in routes.EXPOSURES for c in routes.CATEGORIES}
    assert set(t["systemic_status"]) <= {"yes", "no", "unknown"}


def test_exposures_are_the_review_packet_labels():
    assert routes.EXPOSURES[:4] == tuple(review.AGENTS)
    assert set(routes.PRESPECIFIED_CLASSES) <= set(review.CLASSES)


def test_cells_follow_the_principles_and_the_owner_determinations():
    t = routes.read_systemic()
    for r in t.itertuples():
        pair = (r.exposure, r.route_category)
        if r.route_category == "enteral_tube":
            continue  # P4, tested below
        if pair in OWNER:
            assert (r.systemic_status, r.basis) == OWNER[pair], pair
        elif r.route_category in P1_CATS:
            assert (r.systemic_status, r.basis[:4]) == ("yes", "P1: "), pair
        elif r.route_category in P2_CATS:
            assert (r.systemic_status, r.basis[:4]) == ("no", "P2: "), pair
        elif r.route_category == "unknown":
            assert (r.systemic_status, r.basis[:4]) == ("unknown", "P3: "), pair
        else:
            assert (r.systemic_status, r.basis) == ("unknown", NO_DETERMINATION), pair
    assert (t["basis"] == NO_DETERMINATION).sum() == 12


def test_enteral_tube_inherits_enteral_oral_by_p4():
    t = routes.read_systemic().set_index(["exposure", "route_category"])
    for e in routes.EXPOSURES:
        tube, oral = t.loc[(e, "enteral_tube")], t.loc[(e, "enteral_oral")]
        assert tube["systemic_status"] == oral["systemic_status"], e
        assert tube["basis"].startswith(P4), e
        if (e, "enteral_tube") in OWNER:  # the owner's first basis is kept after P4
            assert tube["basis"] == f"{P4} Basis as first recorded: {OWNER[(e, 'enteral_tube')][1]}"
    assert t.loc[(NSAID, "enteral_tube"), "systemic_status"] == "yes"


def test_the_p1_basis_matches_the_route_table():
    t = routes.read_systemic()
    assert t.loc[t["route_category"].isin(P1_CATS), "basis"].nunique() == 1
    table = routes.read_table().assign(raw=lambda d: d["raw_value"].str.rstrip())
    category = table.set_index("raw")["route_category"]
    # "every raw value it contains (INJ, Injection, INF, IMV, IM-V, RADINJ)"
    assert set(table.loc[table["route_category"] == "parenteral_unspecified", "raw"]) == {
        "INJ", "Injection", "INF", "IMV", "IM-V", "RADINJ"}
    # "I-pleural, Intraductal, Intracatheter, Dwell, IC Dwell - and are classified unknown"
    for raw in ("I-pleural", "Intraductal", "Intracatheter", "Dwell", "IC Dwell"):
        assert category[raw] == "unknown", raw


@pytest.mark.parametrize("damage, message", [
    (lambda t: t.iloc[1:], "missing"),
    (lambda t: pd.concat([t, t.iloc[:1]]), "more than once"),
    (lambda t: t.assign(systemic_status=t["systemic_status"].replace("no", "No")), "outside"),
    (lambda t: t.assign(basis=t["basis"].where(t.index != 0, " ")), "basis"),
])
def test_read_systemic_refuses_a_damaged_table(tmp_path, damage, message):
    damage(routes.read_systemic()).to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(ValueError, match=message):
        routes.read_systemic(tmp_path / "t.csv")


def test_systemic_status_looks_up_each_row_and_raises_on_an_unseen_pair():
    rows = pd.DataFrame({"exposure": ["vancomycin", "vancomycin", "furosemide"],
                         "route_category": ["parenteral_iv", "enteral_oral", "unknown"]})
    assert routes.systemic_status(rows).tolist() == ["yes", "no", "unknown"]
    with pytest.raises(ValueError, match="loop diuretics"):
        routes.systemic_status(pd.DataFrame({"exposure": ["loop diuretics"],
                                             "route_category": ["parenteral_iv"]}))
