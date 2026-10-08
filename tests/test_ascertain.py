"""Exposure ascertainment (section 6, D5.11-D5.13, D6.2-D6.4, D6.3a, D6.7)."""

import pandas as pd
import pytest

from polysafe import ascertain, review
from polysafe.tables import load

TERMS = review.REVIEW_DIR / "search_terms.txt"


def _med(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    base = {"medicationid": range(1, n + 1), "patientunitstayid": [1] * n,
            "drughiclseqno": [None] * n, "drugname": ["X"] * n, "gtc": [0] * n,
            "drugordercancelled": ["No"] * n, "drugstartoffset": [0] * n,
            "drugstopoffset": [10] * n, "routeadmin": ["IV"] * n}
    base.update(cols)
    return pd.DataFrame(base)


# ---------------------------------------------------------------- rules on synthetic rows


def test_counting_window_is_active_in_0_to_T_and_drops_cancelled():
    med = _med(drugstartoffset=[1440, 1441, -500, -500, 0],
               drugstopoffset=[2000, 2000, -1, 0, 5],
               drugordercancelled=["No", "No", "No", "No", "Yes"])
    assert list(ascertain.counting(med)["medicationid"]) == [1, 4]


def test_string_arms_match_whole_tokens_and_brands_only_in_arm_2():
    med = _med(drugname=["VANCOMYCIN 1 G IVPB", "ZOSYN 3.375 GRAM", "vancomycinx", None])
    arm1 = ascertain.string_rows(med, TERMS, with_brands=False)
    arm2 = ascertain.string_rows(med, TERMS, with_brands=True)
    assert set(zip(arm1["medicationid"], arm1["exposure"], strict=True)) == {(1, "vancomycin")}
    assert set(zip(arm2["medicationid"], arm2["exposure"], strict=True)) == {
        (1, "vancomycin"), (2, "piperacillin-tazobactam")}


def test_hicl_gtc_arm_refuses_a_row_without_hicl_that_carries_a_gtc():
    ascertain.check_gtc(_med(gtc=[0, 0]))
    with pytest.raises(ValueError, match="gtc map"):
        ascertain.check_gtc(_med(gtc=[0, 12]))


# ---------------------------------------------------------------- the real data


@pytest.fixture(scope="module")
def med():
    return load("medication")


@pytest.fixture(scope="module")
def reassigned():
    return ascertain.reassignments(review.REVIEW_DIR, TERMS)


@pytest.fixture(scope="module")
def rows(med, reassigned):
    return {(a, v): ascertain.arm_rows(a, v, med, review.REVIEW_DIR, TERMS, reassigned)
            for a in ascertain.ARMS for v in ascertain.VARIANTS}


def test_review_keys_are_unique_across_the_five_files():
    assert len(ascertain.read_review(review.REVIEW_DIR)) == 171


def test_no_counted_row_lies_outside_the_d64_window(rows):
    for (arm, variant), r in rows.items():
        assert (r["drugstartoffset"] <= 1440).all(), (arm, variant)
        assert (r["drugstopoffset"] >= 0).all() and (r["drugordercancelled"] != "Yes").all()


def test_no_row_counts_twice_for_the_same_exposure(rows):
    for key, r in rows.items():
        assert not r.duplicated(["medicationid", "exposure"]).any(), key


def test_only_prespecified_exposures_and_route_rule(rows):
    for (arm, variant), r in rows.items():
        assert r["exposure"].isin(ascertain.EXPOSURES).all()
        allowed = {"yes", "unknown"} if variant == "route_unknown" else {"yes"}
        assert set(r["systemic"]) <= allowed, (arm, variant)


def test_hicl_gtc_equals_hicl_exactly(med, rows):
    ascertain.check_gtc(med)
    for v in ascertain.VARIANTS:
        a = rows[("hicl", v)][["medicationid", "exposure"]].sort_values(["medicationid",
                                                                         "exposure"])
        b = rows[("hicl_gtc", v)][["medicationid", "exposure"]].sort_values(["medicationid",
                                                                             "exposure"])
        assert a.reset_index(drop=True).equals(b.reset_index(drop=True)), v


def test_hicl_arms_read_no_name_keys(rows):
    for (arm, variant), r in rows.items():
        if arm.startswith("hicl"):
            assert r["key"].str.startswith("HICL:").all(), (arm, variant)


def test_hicl_11528_never_counts(rows):
    for (arm, variant), r in rows.items():
        assert "HICL:11528" not in set(r["key"]), (arm, variant)


def test_37442_and_10093_reach_vancomycin_only_in_arm_5(rows, reassigned):
    for k in ("HICL:37442", "HICL:10093"):
        assert reassigned.exposures[k] == "vancomycin"
        assert k not in set(rows[("hicl", "primary")]["key"])
        r5 = rows[("hicl_reviewed", "primary")]
        assert set(r5.loc[r5["key"] == k, "exposure"]) == {"vancomycin"}


def test_audit_key_4703_changes_only_arm_5(rows, reassigned):
    assert reassigned.exposures["HICL:4703"] == "iodinated contrast"
    for arm in ("hicl", "hicl_gtc"):
        assert "HICL:4703" not in set(rows[(arm, "primary")]["key"])
    assert "HICL:4703" in set(rows[("hicl_reviewed", "primary")]["key"])


def test_33356_is_reassigned_to_aminoglycosides_but_no_row_counts(med, rows, reassigned):
    assert reassigned.exposures["HICL:33356"] == "aminoglycosides"
    pre_route = ascertain.hicl_reviewed_rows(ascertain.counting(med), review.REVIEW_DIR,
                                             TERMS, reassigned=reassigned)
    pre_route = pre_route[pre_route["key"] == "HICL:33356"]
    assert set(pre_route["exposure"]) <= {"aminoglycosides"}
    for variant in ascertain.VARIANTS:
        assert "HICL:33356" not in set(rows[("hicl_reviewed", variant)]["key"])


def test_arm_5_differs_from_arm_3_only_on_reassigned_keys(rows, reassigned):
    for v in ascertain.VARIANTS:
        a = set(zip(rows[("hicl", v)]["medicationid"], rows[("hicl", v)]["exposure"],
                    strict=True))
        r5 = rows[("hicl_reviewed", v)]
        b = set(zip(r5["medicationid"], r5["exposure"], strict=True))
        changed = {m for m, _ in a ^ b}
        keys = set(rows[("hicl", v)].loc[rows[("hicl", v)]["medicationid"].isin(changed),
                                         "key"]) | set(r5.loc[r5["medicationid"].isin(changed),
                                                              "key"])
        assert keys <= set(reassigned.exposures), v


def test_unreadable_notes_are_listed_not_reassigned(reassigned):
    assert "HICL:11528" in reassigned.unreadable  # mapping_wrong, note not in D6.3a form
    assert not set(reassigned.unreadable) & set(reassigned.exposures)


def test_tables_cover_exactly_the_primary_cohort(med, tables):
    stays = ascertain.primary_stays(tables)
    assert len(stays) == 671  # results/cohort_flow.md, baseline A, recording sites
    out, _ = ascertain.ascertain(med, stays)
    for key, t in out.items():
        assert list(t.index) == list(stays) and list(t.columns) == list(ascertain.EXPOSURES)
        assert t.dtypes.eq(bool).all(), key
    assert (out[("hicl_gtc", "primary")] == out[("hicl", "primary")]).all().all()
    assert (out[("hicl", "route_unknown")] >= out[("hicl", "primary")]).all().all()
