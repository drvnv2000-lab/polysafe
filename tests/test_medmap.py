import pandas as pd
import pytest

from polysafe import medmap


@pytest.mark.parametrize("raw, expected", [
    ("1000 ML FLEX CONT : SODIUM CHLORIDE 0.9 % IV SOLN", "sodium chloride"),
    ("ACETAMINOPHEN 325 MG PO TABS", "acetaminophen"),
    ("2 ML  -  METOCLOPRAMIDE HCL 5 MG/ML IJ SOLN", "metoclopramide hcl"),
    ("ONDANSETRON 2 MG/1 ML 2ML SDV INJ", "ondansetron"),
    ("Dextrose 50%", "dextrose"),
    ("piperacillin-tazobactam 3.375 g IVPB", "piperacillin-tazobactam"),
    ("LASIX", "lasix"),
])
def test_clean_name(raw, expected):
    assert medmap.clean_name(raw) == expected


def _offline(tmp_path, cache):
    rx = medmap.RxNorm(tmp_path / "c.json", offline=True)
    rx.cache = cache
    return rx


def test_exact_match_on_an_ingredient(tmp_path):
    rx = _offline(tmp_path, {
        "/rxcui.json?name=furosemide&search=2": {"idGroup": {"rxnormId": ["4603"]}},
        "/rxcui/4603/properties.json": {"properties": {"tty": "IN", "name": "furosemide"}},
        "/rxclass/class/byRxcui.json?rxcui=4603&relaSource=ATC": {"rxclassDrugInfoList": {
            "rxclassDrugInfo": [{"rxclassMinConceptItem": {"classId": "C03CA",
                                                           "className": "Sulfonamides, plain"}}]}},
    })
    r = rx.lookup("furosemide")
    assert (r["match"], r["ingredients"], r["atc"]) == ("exact", "furosemide", "C03CA")


def test_brand_name_is_approximate_and_resolves_to_ingredient(tmp_path):
    rx = _offline(tmp_path, {
        "/rxcui.json?name=lasix&search=2": {"idGroup": {}},
        "/approximateTerm.json?term=lasix&maxEntries=1": {"approximateGroup": {"candidate": [
            {"rxcui": "202991", "score": "13.9"}]}},
        "/rxcui/202991/properties.json": {"properties": {"tty": "BN", "name": "Lasix"}},
        "/rxcui/202991/related.json?tty=IN": {"relatedGroup": {"conceptGroup": [
            {"conceptProperties": [{"rxcui": "4603", "name": "furosemide"}]}]}},
        "/rxclass/class/byRxcui.json?rxcui=4603&relaSource=ATC": {},
    })
    r = rx.lookup("lasix")
    assert r["match"] == "approximate" and r["ingredients"] == "furosemide"


def test_offline_never_calls_the_network(tmp_path):
    with pytest.raises(KeyError):
        _offline(tmp_path, {}).lookup("vancomycin")


def test_keys_skip_cancelled_orders_and_flag_name_sources():
    med = pd.DataFrame({
        "patientunitstayid": [1, 1, 2, 3, 4],
        "drughiclseqno": [10.0, 10.0, 20.0, None, 30.0],
        "drugname": ["FUROSEMIDE 40 MG INJ", None, None, "vancomycin", None],
        "drugordercancelled": ["No", "No", "No", "No", "Yes"],
        "gtc": [5, 5, 7, 9, 1], "routeadmin": ["IV", "IV", "PO", "IV", "PO"],
    })
    adm = pd.DataFrame({"drughiclseqno": [20.0], "drugname": ["LISINOPRIL"]})
    k = medmap.medication_keys(med, adm).set_index(["key_type", "key"])
    assert ("HICL", "30") not in k.index  # its only order was cancelled
    assert k.loc[("HICL", "10"), "query"] == "furosemide"
    assert k.loc[("HICL", "10"), "pct_rows_named"] == 0.5
    assert k.loc[("HICL", "20"), "name_source"] == "admissiondrug"
    assert k.loc[("NAME", "vancomycin"), "n_rows"] == 1


def test_unnamed_key_is_unmapped(tmp_path):
    keys = pd.DataFrame({"key_type": ["HICL"], "key": ["99"], "query": [""],
                         "top_names": [""], "n_rows": [3], "n_stays": [2]})
    d = medmap.draft(keys, _offline(tmp_path, {}))
    assert d.loc[0, "match"] == "unmapped" and d.loc[0, "pi_reviewed"] == ""
