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


def test_end_to_end_decision_to_stay_level_exposure(monkeypatch, tmp_path):
    """review decision -> string-level inclusion -> exposure rows -> stay exposure -> dataset,
    through the builder the models use (``polysafe.ascertain``: the HICL arms' row builder
    and the stay table). A builder that bypasses review.rows_counted fails this test: it
    would mark the enalapril-only stays 11-12 as exposed to furosemide."""
    from polysafe import ascertain

    med, reviewed = _chain_fixture()
    monkeypatch.setattr(ascertain, "read_review", lambda *a, **k: reviewed)
    rows = ascertain.hicl_rows(med, tmp_path)
    assert (rows["exposure"] == "furosemide").sum() == 500
    data = ascertain.stay_table(rows, stays=range(1, 13))
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


def test_unsure_only_in_the_sensitivity_variant_and_wrong_strings_still_excluded():
    """D5.13: in the sensitivity variant an unsure key follows the keep row rule."""
    med, reviewed = _chain_fixture()
    unsure = reviewed.assign(decision="unsure")
    assert review.rows_counted(med, unsure).sum() == 0
    assert review.rows_counted(med, unsure, include_unsure=True).sum() == 500


def test_named_only_sensitivity_drops_rows_without_a_string():
    """D5.12: unnamed rows follow the key in the primary strategy, dropped in the sensitivity."""
    med, reviewed = _chain_fixture()
    unnamed = pd.DataFrame({"patientunitstayid": [13, 13], "drughiclseqno": 3660.0,
                            "drugname": [None, None]})
    med = pd.concat([med, unnamed], ignore_index=True)
    assert review.rows_counted(med, reviewed).sum() == 502
    assert review.rows_counted(med, reviewed, named_only=True).sum() == 500
    data = review.stay_exposure(review.exposure_rows(med, reviewed, named_only=True),
                                stays=range(1, 14))
    assert not data.loc[13, "furosemide"]


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


# ---------------------------------------------------------------- mapping_wrong (D6.1)


def _audit_row(**kw) -> pd.Series:
    row = {"key": "HICL:1", "n_rows": 60, "wrong_strings": "", "mapping_wrong": "false",
           "drugnames_medication": "A (20) | B (5)", "drugnames_admissiondrug": ""}
    row.update(kw)
    return pd.Series(row)


def test_mapping_wrong_must_be_true_or_false():
    for bad in ("", " ", "yes", "1", "t"):
        with pytest.raises(ValueError, match="mapping_wrong must be true or false"):
            review.mapping_wrong(_audit_row(mapping_wrong=bad))
    for good, expected in (("true", True), ("TRUE", True), (" false ", False),
                           ("FALSE", False)):
        assert review.mapping_wrong(_audit_row(mapping_wrong=good)) is expected


def test_mapping_wrong_true_lists_no_wrong_strings():
    with pytest.raises(ValueError, match="D6.1"):
        review.mapping_wrong(_audit_row(mapping_wrong="true", wrong_strings="B"))


def test_mapping_error_counts_all_rows_only_in_the_d61_version():
    whole = _audit_row(mapping_wrong="true")
    assert review.error_rows(whole) == 0
    assert review.error_rows(whole, mapping_errors=True) == 60  # includes unnamed rows
    partial = _audit_row(wrong_strings="B")
    assert review.error_rows(partial) == review.error_rows(partial, mapping_errors=True) == 5


def test_estimate_from_review_reports_both_versions(tmp_path):
    df = pd.DataFrame({"key": ["HICL:1", "HICL:2"], "n_rows": ["60", "10"],
                       "drugnames_medication": ["A (20) | B (5)", "C (10)"],
                       "drugnames_admissiondrug": ["", ""], "wrong_strings": ["", "C"],
                       "mapping_wrong": ["true", "false"]})
    df.iloc[[0]].to_csv(tmp_path / "audit_head.csv", index=False)
    df.iloc[[1]].to_csv(tmp_path / "audit_tail.csv", index=False)
    (tmp_path / "audit_design.json").write_text('{"N": 1, "R": 100}')
    r = review.estimate_from_review(tmp_path)
    assert r["as_defined"]["E_head"] == 0
    assert r["with_mapping_errors"]["E_head"] == 60
    assert r["as_defined"]["E_tail"] == r["with_mapping_errors"]["E_tail"] == 10
    assert r["with_mapping_errors"]["rate"] == pytest.approx(70 / 100)


def test_estimate_from_review_rejects_a_blank_mapping_wrong(tmp_path):
    df = pd.DataFrame({"key": ["HICL:1"], "n_rows": ["10"], "drugnames_medication": ["C (10)"],
                       "drugnames_admissiondrug": [""], "wrong_strings": [""],
                       "mapping_wrong": [""]})
    df.to_csv(tmp_path / "audit_head.csv", index=False)
    df.to_csv(tmp_path / "audit_tail.csv", index=False)
    (tmp_path / "audit_design.json").write_text('{"N": 1, "R": 10}')
    with pytest.raises(ValueError, match="mapping_wrong must be true or false"):
        review.estimate_from_review(tmp_path)


def test_key_level_proportion_counts_any_wrong_string(tmp_path):
    # tail key 2: its only wrong string is an admission-drug name with 0 medication rows
    df = pd.DataFrame({"key": ["HICL:1", "HICL:2", "HICL:3"], "n_rows": ["10", "10", "10"],
                       "drugnames_medication": ["C (10)", "C (10)", "C (10)"],
                       "drugnames_admissiondrug": ["", "X (2)", ""],
                       "wrong_strings": ["", "X", ""],
                       "mapping_wrong": ["false", "false", "true"]})
    df.iloc[[0]].to_csv(tmp_path / "audit_head.csv", index=False)
    df.iloc[[1, 2]].to_csv(tmp_path / "audit_tail.csv", index=False)
    (tmp_path / "audit_design.json").write_text('{"N": 2, "R": 30}')
    r = review.estimate_from_review(tmp_path)
    assert r["as_defined"]["E_tail"] == 0
    assert r["as_defined"]["tail_keys_with_error"] == 1
    assert r["with_mapping_errors"]["tail_keys_with_error"] == 2


# ---------------------------------------------------------------- merge (Excel-edited copies)

_COMMITTED = ("key,drugnames_medication,decision,wrong_strings,corrected_name,"
              "reviewer_note,mapping_wrong\n"
              'HICL:1,"VANC 1 G, IV (3)",,,,,\n'
              "HICL:2,ONDANSETRON (5),,,,,\n")


def test_merge_copies_only_decision_columns(tmp_path):
    target = tmp_path / "audit_head.csv"
    target.write_text(_COMMITTED, encoding="utf-8")
    edited = tmp_path / "edited.csv"
    # what Excel produces: BOM, CRLF, rows re-sorted, a source cell mangled, TRUE
    edited.write_bytes(("\ufeffkey,drugnames_medication,decision,wrong_strings,"
                        "corrected_name,reviewer_note,mapping_wrong\r\n"
                        "HICL:2,MANGLED,keep,,,,FALSE\r\n"
                        "HICL:1,MANGLED, reject ,,,ilex not vancomycin; per D6.1,TRUE\r\n"
                        ).encode("utf-8"))
    assert review.merge_decisions(edited, target) == 2
    assert target.read_text(encoding="utf-8") == (
        "key,drugnames_medication,decision,wrong_strings,corrected_name,"
        "reviewer_note,mapping_wrong\n"
        'HICL:1,"VANC 1 G, IV (3)",reject,,,ilex not vancomycin; per D6.1,true\n'
        "HICL:2,ONDANSETRON (5),keep,,,,false\n")


def test_merge_rejects_changed_keys_and_bad_values(tmp_path):
    target = tmp_path / "audit_head.csv"
    head = ("key,drugnames_medication,decision,wrong_strings,corrected_name,"
            "reviewer_note,mapping_wrong\n")
    for body, match in (("HICL:1,x,,,,,\n", "keys differ"),
                        ("HICL:1,x,,,,,\nHICL:3,x,,,,,\n", "keys differ"),
                        ("HICL:1,x,maybe,,,,\nHICL:2,x,,,,,\n", "decision"),
                        ("HICL:1,x,keep,,,,yes\nHICL:2,x,,,,,\n", "mapping_wrong")):
        target.write_text(_COMMITTED, encoding="utf-8")
        edited = tmp_path / "edited.csv"
        edited.write_text(head + body, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            review.merge_decisions(edited, target)
        assert target.read_text(encoding="utf-8") == _COMMITTED  # untouched on error


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


# ---------------------------------------------------------------- brand template, screen B


def test_brand_template_has_one_empty_slot_per_generic(tmp_path):
    f = tmp_path / "search_terms.txt"
    f.write_text(review.search_terms_text(), encoding="utf-8")
    generics, brands = review.read_search_terms(f)
    assert brands == [] and review.read_brand_aliases(f) == {}
    body = f.read_text(encoding="utf-8")
    section = body[body.index(review.BRAND_START):body.index(review.BRAND_END)]
    slots = [ln[:-2] for ln in section.splitlines() if ln.endswith(" =")]
    assert slots == generics


def test_filled_aliases_are_read_with_their_generic(tmp_path):
    f = tmp_path / "search_terms.txt"
    text = review.search_terms_text().replace("furosemide =\n",
                                              "furosemide = FakeBrandA; FakeB\n")
    f.write_text(text, encoding="utf-8")
    assert review.read_brand_aliases(f) == {"FakeBrandA": "furosemide", "FakeB": "furosemide"}
    assert review.term_groups(f)[review.normalize("FakeBrandA")] == "furosemide"
    assert review.term_groups(f)["bumetanide"] == "loop diuretics"


def _screen_dir(d, rows, zero):
    d.mkdir()
    pd.DataFrame(rows, columns=["key", "hit_terms"]).to_csv(d / "tier3_screen.csv", index=False)
    (d / "REVIEW_PACKET.md").write_text(f"x\n\n{review.ZERO_HIT_HEADING}\n\n{zero}\n",
                                        encoding="utf-8")


def test_compare_screens_counts_distinct_stays_per_agent(tmp_path):
    f = tmp_path / "search_terms.txt"
    f.write_text(review.search_terms_text().replace("furosemide =\n", "furosemide = FakeLas\n"),
                 encoding="utf-8")
    _screen_dir(tmp_path / "a", [("HICL:1", "furosemide")], "`zzq`")
    _screen_dir(tmp_path / "b", [("HICL:1", "furosemide"), ("NAME:FAKELAS", "FakeLas")], "(none)")
    med = pd.DataFrame({"patientunitstayid": [1, 1, 2, 2, 3], "drugordercancelled": "No",
                        "drughiclseqno": [1.0, 1.0, None, None, None],
                        "drugname": ["X", "X", "FAKELAS", "FAKELAS", "FAKELAS"]})
    c = review.compare_screens(tmp_path / "a", tmp_path / "b", med, f)
    row = c["per_group"].set_index("group").loc["furosemide"]
    assert (row.A_keys, row.A_rows, row.A_stays) == (1, 2, 1)
    assert (row.B_keys, row.B_rows, row.B_stays) == (2, 5, 3)
    assert (row.added_keys, row.added_rows, row.added_stays) == (1, 3, 2)
    assert c["zero_hit_A"] == ["zzq"] and c["zero_hit_B"] == []


def test_screen_b_refuses_without_aliases(tmp_path):
    (tmp_path / "search_terms.txt").write_text(review.search_terms_text(), encoding="utf-8")
    with pytest.raises(RuntimeError, match="no brand aliases"):
        review.build_screen_b(tmp_path)
