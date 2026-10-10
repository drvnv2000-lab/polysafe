"""README figures, drawn as SVG from committed results only (HANDOFF, required visuals).

* ``cohort_flow.svg``: CONSORT-style flow parsed from ``results/cohort_flow.md``;
* ``calibration.svg``: decile calibration of both models, slope and intercept on the plot,
  for the primary logistic regression and LightGBM (``model_calibration.csv``,
  ``model_metrics.csv``);
* ``dauroc.svg``: the paired dAUROC with its BCa interval for the section 8 models and the
  D8.1 sensitivity analyses (drawn only once ``sensitivity.csv`` exists).

No numbers are typed in: every label is read from a results file. No plotting library.

Usage:
    python -m polysafe.figures    # writes results/figures/*.svg
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import pandas as pd

RESULTS = Path("results")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, BORDER = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#d9d8d2"
CLINICAL, MEDICATION = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated)
FONT = "font-family='Inter, Segoe UI, Helvetica, Arial, sans-serif'"


def _t(x, y, s, size=12, fill=INK, anchor="start", weight="normal") -> str:
    return (f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size}' fill='{fill}' "
            f"text-anchor='{anchor}' font-weight='{weight}'>{escape(str(s))}</text>")


def _svg(w: int, h: int, body: list[str], title: str) -> str:
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' "
            f"viewBox='0 0 {w} {h}' role='img' aria-label='{escape(title)}'>"
            f"<title>{escape(title)}</title>"
            f"<rect width='{w}' height='{h}' fill='{SURFACE}'/>" + "".join(body) + "</svg>\n")


# ---------------------------------------------------------------- cohort flow


def flow_table(md: str, section: str = "Baseline A") -> pd.DataFrame:
    block = md.split(f"## {section}", 1)[1].split("\n## ", 1)[0]
    rows = [r for r in block.splitlines() if r.startswith("|")][2:]
    cells = [[c.strip().strip("*") for c in r.strip("|").split("|")] for r in rows]
    t = pd.DataFrame(cells, columns=["step", "removed", "stays", "patients", "events",
                                     "rate", "reason"])
    for c in ("removed", "stays", "patients", "events"):
        t[c] = pd.to_numeric(t[c].str.replace(",", ""), errors="coerce")
    return t


def cohort_flow(t: pd.DataFrame) -> str:
    w, box_w, box_h, gap = 860, 300, 54, 34
    x0, side_x = 40, 420
    body, y = [], 56
    body.append(_t(x0, 30, "Cohort flow: eICU-CRD Demo v2.0.1, prediction at T = 24 h, "
                   "baseline A", 15, INK, weight="600"))
    for i, r in t.iterrows():
        last = i == len(t) - 1
        if i > 0:
            ex_y = y - gap / 2 - box_h / 2 - 6
            body.append(f"<line x1='{x0 + box_w / 2}' y1='{ex_y + box_h / 2 - 1}' "
                        f"x2='{side_x}' y2='{ex_y + box_h / 2 - 1}' stroke='{AXIS}' "
                        "stroke-width='1.5'/>")
            body.append(f"<rect x='{side_x}' y='{ex_y + 6}' width='{w - side_x - 30}' "
                        f"height='{box_h - 12}' rx='6' fill='white' stroke='{BORDER}'/>")
            label = (f"Excluded: {r['step']} (n = {int(r['removed']):,})" if not last
                     else f"Not in primary estimand: other sites (n = {int(r['removed']):,})")
            body.append(_t(side_x + 10, ex_y + 23, label, 12, INK, weight="600"))
            body.append(_t(side_x + 10, ex_y + 38, r["reason"][:84], 10.5, INK2))
        fill = "#eef4fc" if last else "white"
        stroke = CLINICAL if last else BORDER
        body.append(f"<rect x='{x0}' y='{y}' width='{box_w}' height='{box_h}' rx='8' "
                    f"fill='{fill}' stroke='{stroke}' stroke-width='{2 if last else 1}'/>")
        head = "Primary analysis cohort" if last else ("All ICU stays" if i == 0 else
                                                       "Remaining")
        body.append(_t(x0 + 12, y + 21, f"{head}: {int(r['stays']):,} stays", 13, INK,
                       weight="600"))
        body.append(_t(x0 + 12, y + 40, f"{int(r['patients']):,} patients; "
                       f"{int(r['events'])} AKI events ({r['rate']})", 11.5, INK2))
        if not last:
            body.append(f"<line x1='{x0 + box_w / 2}' y1='{y + box_h}' x2='{x0 + box_w / 2}' "
                        f"y2='{y + box_h + gap}' stroke='{AXIS}' stroke-width='1.5'/>")
        y += box_h + gap
    body.append(_t(x0, y + 4, "AKI: KDIGO stage >= 1 in (T, min(T + 7 d, ICU discharge)], "
                   "creatinine or acute RRT. Source: results/cohort_flow.md.", 10.5, MUTED))
    return _svg(w, int(y + 20), body, "Cohort flow diagram")


# ---------------------------------------------------------------- calibration


def calibration(curves: pd.DataFrame, metrics: pd.DataFrame) -> str:
    panels = (("lr_reduced", "Primary: reduced logistic regression"),
              ("lgbm", "Secondary: LightGBM"))
    size, pad_l, pad_t, gap = 300, 56, 70, 90
    w = pad_l + 2 * size + gap + 30
    h = pad_t + size + 120
    hi = float(max(curves["mean_predicted"].max(), curves["observed"].max()))
    top = max(0.1, round(hi + 0.05, 1))
    body = [_t(pad_l, 26, "Calibration (deciles of pooled out-of-fold risk)", 15, INK,
               weight="600")]
    for j, (spec, name) in enumerate(panels):
        ox = pad_l + j * (size + gap)
        def sx(v, ox=ox):
            return ox + v / top * size
        def sy(v):
            return pad_t + size - v / top * size
        body.append(_t(ox, pad_t - 22, name, 13, INK, weight="600"))
        for k in range(int(round(top / 0.1)) + 1):
            v = k * 0.1
            body.append(f"<line x1='{sx(0)}' y1='{sy(v)}' x2='{sx(top)}' y2='{sy(v)}' "
                        f"stroke='{GRID}' stroke-width='1'/>")
            body.append(_t(sx(0) - 6, sy(v) + 4, f"{v:.1f}", 10, MUTED, "end"))
            body.append(_t(sx(v), sy(0) + 16, f"{v:.1f}", 10, MUTED, "middle"))
        body.append(f"<line x1='{sx(0)}' y1='{sy(0)}' x2='{sx(top)}' y2='{sy(top)}' "
                    f"stroke='{AXIS}' stroke-width='1.5' stroke-dasharray='5 4'/>")
        body.append(_t(ox + size / 2, sy(0) + 34, "Mean predicted risk", 11, INK2, "middle"))
        if j == 0:
            body.append(f"<text transform='translate({ox - 40},{pad_t + size / 2}) "
                        f"rotate(-90)' {FONT} font-size='11' fill='{INK2}' "
                        "text-anchor='middle'>Observed AKI rate</text>")
        for k, (med, colour, label) in enumerate(((False, CLINICAL, "Clinical only"),
                                                  (True, MEDICATION, "Clinical + medication"))):
            c = curves[(curves["spec"] == spec) & (curves["medication"] == med)]
            pts = " ".join(f"{sx(p):.1f},{sy(o):.1f}"
                           for p, o in zip(c["mean_predicted"], c["observed"], strict=True))
            body.append(f"<polyline points='{pts}' fill='none' stroke='{colour}' "
                        "stroke-width='2'/>")
            body += [f"<circle cx='{sx(p):.1f}' cy='{sy(o):.1f}' r='4' fill='{colour}' "
                     f"stroke='{SURFACE}' stroke-width='2'/>"
                     for p, o in zip(c["mean_predicted"], c["observed"], strict=True)]
            m = metrics[(metrics["spec"] == spec) & (metrics["medication"] == med)].iloc[0]
            ly = sy(0) + 58 + 17 * k
            body.append(f"<rect x='{ox}' y='{ly - 9}' width='12' height='3' rx='1.5' "
                        f"fill='{colour}'/>")
            body.append(_t(ox + 18, ly - 3, f"{label}: slope {m['cal_slope']:.2f}, "
                           f"intercept {m['cal_intercept']:+.2f}", 11, INK))
    body.append(_t(pad_l, h - 8, "Dashed line: perfect calibration. Source: "
                   "results/model_calibration.csv, results/model_metrics.csv.", 10.5, MUTED))
    return _svg(w, h, body, "Calibration curves")


# ---------------------------------------------------------------- paired dAUROC


SENS_ROWS = (("ascertainment:generic_string:primary", "Ascertainment: generic string"),
             ("ascertainment:generic_brand:primary", "Ascertainment: generic + brand"),
             ("ascertainment:hicl:primary", "Ascertainment: HICL-based"),
             ("ascertainment:hicl_gtc:named_only", "Primary arm, unnamed rows excluded"),
             ("ascertainment:hicl_reviewed:primary", "Ascertainment: review-corrected (expl.)"),
             ("full_cohort", "Full cohort, all hospitals"),
             ("followup_48h", ">= 48 h post-T follow-up"),
             ("baseline_c", "Baseline C"),
             ("hospital_clusters", "Hospital-cluster CV"))


def dauroc(comparison: pd.DataFrame, sens: pd.DataFrame | None, mde: float) -> str:
    rows = [("Primary: reduced logistic regression", comparison.loc["lr_reduced"], True),
            ("Secondary: LightGBM", comparison.loc["lgbm"], False),
            ("Sensitivity: full penalised LR", comparison.loc["lr_full"], False)]
    if sens is not None:
        s = sens.set_index("analysis")
        rows += [(name, s.loc[key], False) for key, name in SENS_ROWS
                 if key in s.index and pd.notna(s.loc[key].get("dauroc"))]
    lo = min(-0.05, min(float(r["lo"]) for _, r, _ in rows) - 0.01)
    hi = max(0.10, max(float(r["hi"]) for _, r, _ in rows) + 0.01)
    lab_w, plot_w, val_w, row_h, top = 290, 360, 190, 30, 64
    w = lab_w + plot_w + val_w + 20
    h = top + row_h * len(rows) + 70
    def sx(v):
        return lab_w + (v - lo) / (hi - lo) * plot_w
    body = [_t(20, 26, "Paired dAUROC, clinical + medication minus clinical only "
               "(95% patient-level BCa)", 15, INK, weight="600")]
    tick = -0.05
    while tick <= hi + 1e-9:
        if tick >= lo - 1e-9:
            body.append(f"<line x1='{sx(tick)}' y1='{top - 8}' x2='{sx(tick)}' "
                        f"y2='{h - 56}' stroke='{GRID}'/>")
            body.append(_t(sx(tick), h - 40, f"{tick:+.2f}" if tick else "0", 10, MUTED,
                           "middle"))
        tick = round(tick + 0.05, 2)
    body.append(f"<line x1='{sx(0)}' y1='{top - 8}' x2='{sx(0)}' y2='{h - 56}' "
                f"stroke='{INK2}' stroke-width='1.5'/>")
    body.append(f"<line x1='{sx(mde)}' y1='{top - 8}' x2='{sx(mde)}' y2='{h - 56}' "
                f"stroke='{AXIS}' stroke-width='1.5' stroke-dasharray='5 4'/>")
    body.append(_t(sx(mde) + 4, top - 12, f"80%-power threshold {mde:.3f}", 10, MUTED))
    sep = 3
    for i, (name, r, primary) in enumerate(rows):
        y = top + row_h * i + row_h / 2 + (8 if i >= sep else 0)
        colour = MEDICATION if primary else INK2
        body.append(_t(20, y + 4, name, 12, INK, weight="600" if primary else "normal"))
        body.append(f"<line x1='{sx(float(r['lo'])):.1f}' y1='{y}' "
                    f"x2='{sx(float(r['hi'])):.1f}' y2='{y}' stroke='{colour}' "
                    "stroke-width='2' stroke-linecap='round'/>")
        body.append(f"<circle cx='{sx(float(r['dauroc'])):.1f}' cy='{y}' "
                    f"r='{6 if primary else 5}' fill='{colour}' stroke='{SURFACE}' "
                    "stroke-width='2'/>")
        body.append(_t(lab_w + plot_w + 14, y + 4, f"{float(r['dauroc']):+.3f} "
                       f"({float(r['lo']):+.3f} to {float(r['hi']):+.3f})", 11.5, INK,
                       weight="600" if primary else "normal"))
        if i == sep - 1 and len(rows) > sep:
            yy = y + row_h / 2 + 4
            body.append(f"<line x1='20' y1='{yy}' x2='{w - 20}' y2='{yy}' stroke='{GRID}'/>")
    body.append(_t(20, h - 14, "Rows below the line: D8.1 sensitivity analyses of the "
                   "primary comparison; their run and reporting were fixed after the primary "
                   "result was seen (D9.2).", 10.5, MUTED))
    return _svg(w, h, body, "Paired dAUROC")


# ---------------------------------------------------------------- main


def build(results: Path = RESULTS) -> dict[str, str]:
    out = {"cohort_flow.svg": cohort_flow(flow_table(
               (results / "cohort_flow.md").read_text(encoding="utf-8"))),
           "calibration.svg": calibration(pd.read_csv(results / "model_calibration.csv"),
                                          pd.read_csv(results / "model_metrics.csv"))}
    comparison = pd.read_csv(results / "model_comparison.csv").set_index("spec")
    sens_csv = results / "sensitivity.csv"
    sens = pd.read_csv(sens_csv) if sens_csv.exists() else None
    out["dauroc.svg"] = dauroc(comparison, sens, float(comparison.loc["lr_reduced",
                                                                       "mde_bca"]))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="README figures from committed results.")
    p.add_argument("--results", type=Path, default=RESULTS)
    args = p.parse_args(argv)
    folder = args.results / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for name, svg in build(args.results).items():
        (folder / name).write_text(svg, encoding="utf-8")
        print(f"wrote {folder / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


