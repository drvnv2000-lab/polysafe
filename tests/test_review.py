import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polysafe import review
from polysafe.medmap import MAPPING_DIR

REVIEWABLE = ("tier1_agents.csv", "tier2_classes.csv", "tier3_screen.csv",
              "audit_head.csv", "audit_tail.csv")


@pytest.fixture(scope="module")
def tables():
    from polysafe.tables import load
    return load("medication"), load("admissiondrug", usecols=["drughiclseqno", "drugname"])


def _build(tmp: Path, tables) -> dict:
    shutil.copy(MAPPING_DIR / "review" / "search_terms.txt", tmp / "search_terms.txt")
    return review.build_review_packet(out_dir=tmp, med=tables[0], adm=tables[1],
                                      require_committed=False)


@pytest.fixture(scope="module")
def packet(tmp_path_factory, tables):
    out = tmp_path_factory.mktemp("packet")
    return out, _build(out, tables)


def _csv(out: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(out / name, dtype=str, keep_default_na=False)


# ---------------------------------------------------------------- stage 0


def test_committed_generic_terms_match_the_definitions():
    committed, _ = review.read_search_terms(MAPPING_DIR / "review" / "search_terms.txt")
    expected = [t for terms in (*review.AGENTS.values(), *review.CLASSES.values()) for t in terms]
    assert committed == list(dict.fromkeys(expected))


# ---------------------------------------------------------------- partition


def test_tiers_partition_the_source_exactly(packet):
    out, _ = packet
    draft = pd.read_csv(MAPPING_DIR / "hicl_draft.csv", dtype={"key": str})
    source = set(draft["key_type"] + ":" + draft["key"])
    tiers = [set(_csv(out, f)["key"]) for f in ("tier1_agents.csv", "tier2_classes.csv",
                                                "tier3_screen.csv", "tier4_unreviewed.csv")]
    assert sum(len(t) for t in tiers) == len(draft) == len(source)
    assert set().union(*tiers) == source
    for i in range(4):
        for j in range(i + 1, 4):
            assert not tiers[i] & tiers[j]


def test_every_prespecified_agent_is_in_tier1(packet):
    agents = set(_csv(packet[0], "tier1_agents.csv")["agent"])
    for agent in review.AGENTS:
        assert agent in agents, f"pre-specified agent missing from tier 1: {agent}"


def test_tier3_catches_keys_mapped_to_something_unrelated(packet):
    t3 = _csv(packet[0], "tier3_screen.csv")
    mismatched = t3[(t3["match"] != "unmapped") & ~t3.apply(
        lambda r: any(review.term_matches(t, r["proposed_ingredients"])
                      for t in r["hit_terms"].split(" | ")), axis=1)]
    if mismatched.empty:
        pytest.skip("no tier-3 key has a current mapping that disagrees with its drugname")
    assert len(mismatched) >= 1


def test_no_outcome_or_prediction_column_in_any_review_csv(packet):
    for f in packet[0].glob("*.csv"):
        bad = [c for c in pd.read_csv(f, nrows=0).columns if review.FORBIDDEN_COLUMN.search(c)]
        assert not bad, f"{f.name}: {bad}"


def test_decision_columns_ship_empty(packet):
    for name in REVIEWABLE:
        df = _csv(packet[0], name)
        for col in review.DECISION_COLUMNS:
            assert col in df.columns and (df[col] == "").all(), f"{name}:{col}"
    assert not set(review.DECISION_COLUMNS) & set(_csv(packet[0], "tier4_unreviewed.csv"))


# ---------------------------------------------------------------- matching (D5.10)


def test_token_match_rejects_substring_and_accepts_whole_token():
    assert not review.term_matches("vancomycin", "VANCOMYCINHCL 1 G")
    assert not review.term_matches("lasix", "PLASIXAN")
    assert review.term_matches("vancomycin", "VANCOmycin 1 GM in NS 250 mL IVPB")
    assert review.term_matches("tazobactam", "piperacillin-tazobactam 3.375 g")


def test_multiword_term_matches_contiguous_tokens_only():
    assert review.term_matches("ethacrynic acid", "ETHACRYNIC ACID 25 MG PO TABS")
    assert review.term_matches("ethacrynic acid", "ethacrynic-acid")
    assert not review.term_matches("ethacrynic acid", "ETHACRYNIC SODIUM ACID")
    assert not review.term_matches("ethacrynic acid", "ACID ETHACRYNIC")


def test_normalization_handles_unicode_and_case():
    assert review.normalize("Vancomycín  HCl/1g") == "vancomycin hcl 1g"


def test_zero_hit_terms_are_listed():
    draft = pd.DataFrame({"key_type": ["HICL"], "key": ["1"], "ingredients": ["furosemide"],
                          "match": ["exact"], "name_source": ["medication"]})
    names = pd.DataFrame({"drugnames_medication": ["FUROSEMIDE 40 MG INJ (5)"],
                          "drugnames_admissiondrug": [""], "rows_without_name": [0]})
    *_, zero = review.partition(draft, names, ["furosemide", "zzqdrugnotreal"])
    assert zero == ["zzqdrugnotreal"]


# ---------------------------------------------------------------- reconciliation (D5.8)


def test_reconcile_majority_keeps_with_minority_excluded():
    r = review.reconcile({"FUROSEMIDE 40 MG": (500, "furosemide"),
                          "ENALAPRIL 5 MG": (3, "enalapril")})
    assert r["decision"] == "keep" and r["error_rows"] == 3
    assert r["wrong_strings"] == ["ENALAPRIL 5 MG"]


def test_reconcile_split_key_is_unsure():
    r = review.reconcile({"A": (60, "furosemide"), "B": (40, "bumetanide")})
    assert r["decision"] == "unsure"


def _chain_fixture():
    # One HICL, two strings: 500 furosemide rows over stays 1-10 and 3 enalapril rows on
    # stays 11-12, which have no furosemide rows at all.
    furo = pd.DataFrame({"patientunitstayid": np.repeat(np.arange(1, 11), 50),
                         "drughiclseqno": 3660.0, "drugname": "FUROSEMIDE 40 MG INJ"})
    enal = pd.DataFrame({"patientunitstayid": [11, 11, 12], "drughiclseqno": 3660.0,
                         "drugname": "ENALAPRIL 5 MG TAB"})
    med = pd.concat([furo, enal], ignore_index=True)
    reviewed = pd.DataFrame({"key": ["HICL:3660"], "name_source": ["medication"],
                             "decision": ["keep"], "wrong_strings": ["ENALAPRIL 5 MG TAB"],
                             "corrected_name": [""], "exposure": ["furosemide"]})
    return med, reviewed


def test_string_level_filtering_drops_only_wrong_strings():
    med, reviewed = _chain_fixture()
    counted = review.rows_counted(med, reviewed)
    assert counted.sum() == 500
    assert not counted[med["drugname"] == "ENALAPRIL 5 MG TAB"].any()


def test_end_to_end_decision_to_stay_level_exposure():
    """review decision -> string-level inclusion -> exposure rows -> stay exposure -> dataset."""
    med, reviewed = _chain_fixture()
    rows = review.exposure_rows(med, reviewed)
    assert (rows["exposure"] == "furosemide").sum() == 500
    data = review.stay_exposure(rows, stays=range(1, 13))
    assert data.loc[1:10, "furosemide"].all()
    assert not data.loc[[11, 12], "furosemide"].any()  # the enalapril-only stays
    # A key-level builder would get this fixture wrong at stay level, not just in rows:
    assert len(med) == 503 and med["patientunitstayid"].nunique() == 12


def test_corrected_name_cannot_create_a_new_variable():
    med, reviewed = _chain_fixture()
    with pytest.raises(ValueError, match="pre-specified"):
        review.exposure_rows(med, reviewed.assign(corrected_name="enalapril"))
    moved = review.exposure_rows(med, reviewed.assign(corrected_name="loop diuretics"))
    assert set(moved["exposure"]) == {"loop diuretics"}


def test_unsure_only_in_the_sensitivity_variant():
    med, reviewed = _chain_fixture()
    unsure = reviewed.assign(decision="unsure")
    assert review.rows_counted(med, unsure).sum() == 0
    assert review.rows_counted(med, unsure, include_unsure=True).sum() == 503


# ---------------------------------------------------------------- audit and estimator


def test_audit_is_a_subset_of_tier4_approximate(packet):
    out, _ = packet
    t4 = _csv(out, "tier4_unreviewed.csv")
    stratum = set(t4.loc[t4["match"] == "approximate", "key"])
    head, tail = set(_csv(out, "audit_head.csv")["key"]), set(_csv(out, "audit_tail.csv")["key"])
    assert head <= stratum and tail <= stratum and not head & tail


def test_audit_draw_is_reproducible(packet, tmp_path, tables):
    _build(tmp_path, tables)
    for name in ("audit_head.csv", "audit_tail.csv", "audit_design.json"):
        assert (tmp_path / name).read_text() == (packet[0] / name).read_text()


def _stratum(n_keys: int) -> pd.DataFrame:
    return pd.DataFrame({"key_type": "HICL", "key": [str(i) for i in range(n_keys)],
                         "n_rows": np.arange(n_keys, 0, -1), "match": "approximate"})


def test_tail_is_a_census_below_75_and_a_sample_above():
    _, tail, d = review.audit_sample(_stratum(40))
    assert d["tail_mode"] == "census" and d["n"] == d["N"] == len(tail)
    _, tail, d = review.audit_sample(_stratum(300))
    assert d["tail_mode"] == "sample" and d["n"] == len(tail) == review.TAIL_SIZE
    assert d["head_keys"] <= review.HEAD_CAP


def test_estimator_reproduces_a_hand_computed_answer():
    # E_head = 10 + 0 = 10; E_tail = (4 / 2) * (3 + 0) = 6; rate = (10 + 6) / 100 = 0.16
    r = review.audit_estimate(head_errors=[10, 0], tail_errors=[3, 0], N=4, R=100)
    assert (r["E_head"], r["E_tail"], r["rate"]) == (10.0, 6.0, 0.16)
    assert r["tail_keys_with_error"] == 1 and r["tail_key_proportion"] == 0.5
    lo, hi = r["rate_ci"]
    assert 0.10 <= lo <= 0.16 <= hi <= 0.22  # tail resamples give E_tail in {0, 6, 12}
    assert r == review.audit_estimate([10, 0], [3, 0], N=4, R=100)  # seeded


def test_census_estimate_is_exact():
    r = review.audit_estimate(head_errors=[5], tail_errors=[1, 2], N=2, R=50)
    assert r["rate"] == pytest.approx(8 / 50) and r["rate_ci"] == (r["rate"], r["rate"])


def test_error_rows_count_only_wrong_strings():
    row = pd.Series({"key": "HICL:1", "n_rows": 510, "wrong_strings": "B",
                     "drugnames_medication": "A (500) | B (3)", "drugnames_admissiondrug": ""})
    assert review.error_rows(row) == 3
    with pytest.raises(ValueError):
        review.error_rows(row.copy().replace({"B": "C"}))
    adm_only = pd.Series({"key": "HICL:2", "n_rows": 40, "wrong_strings": "X",
                          "drugnames_medication": "", "drugnames_admissiondrug": "X (7)"})
    assert review.error_rows(adm_only) == 40


# ---------------------------------------------------------------- reliability hook


def test_reliability_hook(packet, tmp_path):
    for f in packet[0].glob("*"):
        shutil.copy(f, tmp_path / f.name)
    with pytest.raises(ValueError, match="not complete"):
        review.draw_reliability_sample(tmp_path)
    for name in ("tier1_agents.csv", "tier2_classes.csv"):
        df = _csv(tmp_path, name)
        df["decision"] = "keep"
        df.to_csv(tmp_path / name, index=False)
    path = review.draw_reliability_sample(tmp_path)
    p2 = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert len(p2) == review.RELIABILITY_KEYS and p2["key"].is_unique
    assert (p2[list(review.DECISION_COLUMNS)] == "").all().all()
    assert (tmp_path / "pass1" / "tier1_agents.csv").exists()
    with pytest.raises(FileExistsError):
        review.draw_reliability_sample(tmp_path)
    p2["decision"] = ["keep"] * 18 + ["reject"] * 2
    p2.to_csv(path, index=False)
    rep = review.reliability_report(tmp_path)
    assert rep["percent_agreement"] == 0.9 and "exploratory" in rep["label"]
