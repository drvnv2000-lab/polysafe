"""PolySafe Research Lab as a web service (HANDOFF step 9, DEMO_SCOPE.md).

Serves the Research Lab page and the committed results. It never fits a model and never
returns a single patient's predicted risk (EPV 4.6; DEMO_SCOPE.md). No patient-level data
is needed or served.

Usage:
    uvicorn polysafe.app:app --port 8000
"""

from __future__ import annotations

import csv
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from polysafe import lab

ROOT = lab.ROOT
RESULTS = {"comparison": "results/model_comparison.csv",
           "metrics": "results/model_metrics.csv",
           "sensitivity": "results/sensitivity.csv",
           "exploratory": "results/exploratory_d10.csv",
           "secondary": "results/secondary.csv"}

app = FastAPI(title="PolySafe Research Lab",
              description="Medication-aware AKI prediction on eICU-CRD Demo v2.0.1: "
                          "pre-registered results, read from committed files.")


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return lab.render(ROOT)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "sections": [s.title for s in lab.available_sections(ROOT)]}


@app.get("/api/results")
def results_index() -> dict:
    return {"available": [k for k, v in RESULTS.items() if (Path(ROOT) / v).exists()]}


@app.get("/api/results/{name}")
def result(name: str) -> list[dict]:
    if name not in RESULTS or not (Path(ROOT) / RESULTS[name]).exists():
        raise HTTPException(404, f"no such result: {name}")
    with (Path(ROOT) / RESULTS[name]).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
