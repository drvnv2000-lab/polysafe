"""The web service serves committed results only and no per-patient risk."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from polysafe.app import app  # noqa: E402

client = TestClient(app)


def test_page_and_health():
    r = client.get("/")
    assert r.status_code == 200 and "PolySafe Research Lab" in r.text
    h = client.get("/health").json()
    assert h["status"] == "ok" and "Primary result" in h["sections"]


def test_results_are_the_committed_csv():
    rows = client.get("/api/results/comparison").json()
    assert {r["spec"] for r in rows} == {"lr_reduced", "lgbm", "lr_full"}
    assert client.get("/api/results/predictions").status_code == 404  # never served
