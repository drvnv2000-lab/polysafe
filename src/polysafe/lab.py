"""PolySafe Research Lab: one self-contained HTML page, ``results/lab/index.html``.

Navigation is DERIVED: a section is rendered (in the nav and on the page) only
when the artifact backing it exists on disk. There are no placeholder sections.
Today that is Overview and Research Integrity, both backed by
``results/integrity.json`` (``python -m polysafe.integrity``). Later capabilities
(DEMO_SCOPE.md) are added here with their own artifact when they exist.

Usage:
    python -m polysafe.lab
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Section:
    id: str
    title: str
    artifact: str  # relative to the repo root
    render: Callable[[dict], str]


def _e(x) -> str:
    return html.escape(str(x))


def _v(value, fmt: str = "{}") -> str:
    """A value, or a plain 'not available' naming where it was looked for. Never 0 or ''."""
    if isinstance(value, dict) and value.get("status") == "not_available":
        where = _e(value["path"])
        return f'<span class="na">not available (looked for <code>{where}</code>)</span>'
    return _e(fmt.format(value))


def _chip(status: str) -> str:
    kind = ("ok" if status in ("done", "enforced (passed)", "verified", "fitted", "review complete")
            else "bad" if status in ("failing", "test not found") else "warn")
    return f'<span class="chip {kind}">{_e(status)}</span>'


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def _review_status(d: dict) -> str:
    if isinstance(d, dict) and d.get("status") == "not_available":
        return _v(d)
    if d["status"] == "not_started":
        return f"Review not started ({d['keys_to_review']:,} keys to review)"
    return (f"{d['status'].replace('_', ' ')}: {d['adjudicated']:,} of {d['keys_to_review']:,} "
            f"keys adjudicated ({d['keep']} keep, {d['reject']} reject, {d['unsure']} unsure)")


def render_overview(r: dict) -> str:
    c = r["cohort"]
    p, a = c["primary"], c["all_sites"]
    ci = c["primary_prevalence_ci_pct"]
    epv = c["epv_primary_lr"]
    pw = c["power_mde"]
    model = next(t["status"] for t in r["timeline"] if t["stage"] == "Model fitting")

    def stat(label, value, note=""):
        return (f'<div class="stat"><div class="label">{_e(label)}</div>'
                f'<div class="value">{value}</div><div class="note">{note}</div></div>')

    prev = (f"{_e(p['prevalence_pct'])}%" if isinstance(p, dict) and "stays" in p else _v(p))
    ci_txt = (f"95% CI {ci[0]}–{ci[1]}%" if isinstance(ci, list) else _v(ci))
    stats = "".join([
        stat("Primary cohort", f"{p['stays']:,} stays" if "stays" in p else _v(p),
             f"{p['patients']:,} patients · medication-recording sites" if "stays" in p else ""),
        stat("AKI events", f"{p['events']:,}" if "events" in p else _v(p), f"{prev} · {ci_txt}"),
        stat("Events per variable", _v(epv["epv"]) if "epv" in epv else _v(epv),
             f"{epv['events']} / {epv['parameters']} parameters, primary LR" if "epv" in epv
             else ""),
        stat("Medication review", _review_status(r["review_decisions"])),
        stat("Model", _e(model.capitalize()), "No performance figure exists yet"),
    ])
    pw_txt = (" / ".join(f"{m:.3f}" for m in pw["mde"]) + " at rho = "
              + " / ".join(f"{x:g}" for x in pw["rho"])) if "mde" in pw else _v(pw)
    all_txt = (f"{a['stays']:,} stays, {a['events']:,} events ({a['prevalence_pct']}%)"
               if "stays" in a else _v(a))
    stages = "".join(f'<li><span class="stage">{_e(t["stage"])}</span>{_chip(t["status"])}</li>'
                     for t in r["timeline"])
    return (f'<div class="stats">{stats}</div>'
            f"<p>All sites (sensitivity population): {all_txt}. Smallest ΔAUROC this design "
            f"detects with 80% probability (pre-registration §9.1): {pw_txt}.</p>"
            f'<h3>Pipeline</h3><ol class="timeline">{stages}</ol>')


def render_integrity(r: dict) -> str:
    t, ru = r["tests"], r["ruff"]
    if "collected" in t:
        tests = f"{t['passed']:,} of {t['collected']:,} tests passed"
        if t["failed"]:
            tests += f", {t['failed']:,} failed"
    else:
        tests = _v(t)
    lint = ("ruff clean" if ru.get("clean") else f"ruff not clean: {_e(ru.get('summary', ''))}"
            ) if "clean" in ru else _v(ru)
    controls = _table(["Control", "Established by", "Enforcing test", "Status"], [
        [_e(c["control"]), _e(c["established_by"]),
         "<br>".join(f"<code>{_e(n.split('::')[-1])}</code>" for n in c["tests"]) or "none",
         _chip(c["status"])] for c in r["controls"]])

    pre = r["preregistration"]
    if "deviations" in pre:
        devs = sorted(pre["deviations"],
                      key=lambda d: d["introduced_in"].get("utc", "") if isinstance(
                          d["introduced_in"], dict) else "")
        dev_rows = [[f'<a id="{_e(d["id"])}"></a><b>{_e(d["id"])}</b>', _v(d["tag"]),
                     _e(d["title"]), _v(d["date"]),
                     (f'<code>{_e(d["introduced_in"]["sha"][:7])}</code> '
                      f'{_e(d["introduced_in"]["utc"][:16].replace("T", " "))} UTC')
                     if "sha" in d["introduced_in"] else _v(d["introduced_in"])]
                    for d in devs]
        deviations = _table(["Entry", "Type", "Title", "Date", "Committed"], dev_rows)
    else:
        deviations = _v(pre)
    corr = _table(["Figure", "Originally", "Corrected by", "Current value", "Verified"], [
        [_e(c["figure"]), f'{_e(c["original"] or "")} <span class="muted">({_e(c["original_in"])})'
         "</span>", f'<a href="#{_e(c["corrected_by"])}">{_e(c["corrected_by"])}</a>',
         f'{_e(c["current"].replace("**", ""))} <span class="muted">({_e(c["current_in"])})'
         "</span>",
         _chip(c["status"])] for c in r["corrections"]])

    pk = r["review_packet"]
    if "tiers" in pk and isinstance(pk["tiers"], dict) and "status" not in pk["tiers"]:
        tiers = ", ".join(f"tier {k}: {v['keys']:,} keys / {v['rows']:,} rows"
                          for k, v in sorted(pk["tiers"].items()))
        packet = (f"<p>{_e(tiers)}. Audit: head {_v(pk['audit_head_keys'])} keys, tail "
                  f"{_v(pk['audit_tail_keys'])} keys ({_v(pk['audit_tail_mode'])}), seed "
                  f"{_v(pk['seed'])}. Unmappable rows: {_v(pk['unmappable_rows'], '{:,}')}. "
                  f"Search terms committed at <code>{_v(pk['search_terms_sha'])}</code> before "
                  f"screening. {_review_status(r['review_decisions'])}.</p>")
    else:
        packet = f"<p>{_v(pk)}</p>"
    commits = (_table(["Commit", "UTC", "Subject"], [
        [f"<code>{_e(c['sha'][:7])}</code>", _e(c["utc"][:16].replace("T", " ")), _e(c["subject"])]
        for c in r["commits"]]) if isinstance(r["commits"], list) else _v(r["commits"]))
    return (f'<p class="checks">{tests} · {lint} · report generated '
            f'{_e(r["generated_utc"])}</p>'
            "<h3>Methodological controls</h3><p class=\"muted\">Status comes from each "
            "control's own test in the last real run. A control with no test is shown as "
            "asserted, not tested.</p>" + controls +
            "<h3>Corrected figures</h3><p class=\"muted\">Every value below is matched verbatim "
            "against PREREGISTRATION.md.</p>" + corr +
            "<h3>Deviations, in commit order</h3>" + deviations +
            "<h3>Medication review</h3>" + packet +
            "<h3>Pre-registration and mapping commits</h3>" + commits)


SECTIONS = [
    Section("overview", "Overview", "results/integrity.json", render_overview),
    Section("integrity", "Research Integrity", "results/integrity.json", render_integrity),
]

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--card:#ffffff;
--ok:#1f7a4d;--okbg:#e3f3ea;--warn:#8a5a00;--warnbg:#fbf0d9;--bad:#a12a2a;--badbg:#f8e1e1;
--accent:#2d5f8a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151614;--fg:#e9e7e1;
--muted:#9c9a93;--line:#2c2d2a;--card:#1c1d1b;--ok:#7fd1a5;--okbg:#173326;--warn:#f0c46a;
--warnbg:#3a2e14;--bad:#f19a9a;--badbg:#3d1c1c;--accent:#8db8de}}
:root[data-theme="dark"]{--bg:#151614;--fg:#e9e7e1;--muted:#9c9a93;--line:#2c2d2a;--card:#1c1d1b;
--ok:#7fd1a5;--okbg:#173326;--warn:#f0c46a;--warnbg:#3a2e14;--bad:#f19a9a;--badbg:#3d1c1c;
--accent:#8db8de}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:20px 16px 0;max-width:1080px;margin:auto}h1{font-size:22px;margin:0}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);z-index:1}
nav div{max-width:1080px;margin:auto;padding:0 16px;display:flex;gap:20px;overflow-x:auto}
nav a{color:var(--muted);text-decoration:none;padding:12px 0;white-space:nowrap}
nav a:hover{color:var(--fg)}main{max-width:1080px;margin:auto;padding:0 16px 48px}
section{padding-top:24px}h2{font-size:19px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:15px;margin:22px 0 8px}.muted,.note{color:var(--muted);font-size:13px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.value{font-size:20px;font-weight:600;margin:2px 0}
.tw{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}code{font-size:12px}a{color:var(--accent)}
.chip{display:inline-block;border-radius:999px;padding:1px 9px;font-size:12px;white-space:nowrap}
.ok{background:var(--okbg);color:var(--ok)}.warn{background:var(--warnbg);color:var(--warn)}
.bad{background:var(--badbg);color:var(--bad)}.na{color:var(--warn)}
.timeline{list-style:none;padding:0;margin:0}
.timeline li{display:flex;justify-content:space-between;gap:12px;padding:7px 0;
border-bottom:1px solid var(--line)}.stage{font-weight:500}
.checks{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
"""


def available_sections(root: Path, sections=SECTIONS) -> list[Section]:
    """Only the sections whose backing artifact exists. This IS the navigation."""
    return [s for s in sections if (Path(root) / s.artifact).exists()]


def render(root: Path = ROOT, sections=SECTIONS) -> str:
    present = available_sections(root, sections)
    cache: dict[str, dict] = {}
    for s in present:
        cache.setdefault(s.artifact, json.loads((Path(root) / s.artifact).read_text("utf-8")))
    nav = "".join(f'<a href="#{s.id}">{_e(s.title)}</a>' for s in present)
    body = "".join(f'<section id="{s.id}"><h2>{_e(s.title)}</h2>{s.render(cache[s.artifact])}'
                   "</section>" for s in present)
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>PolySafe Research Lab</title><style>{CSS}</style></head><body>"
            "<header><h1>PolySafe Research Lab</h1><p class=\"muted\">Medication-aware AKI "
            "prediction on eICU-CRD Demo v2.0.1. Every value is read from a committed artifact."
            f"</p></header><nav><div>{nav}</div></nav><main>{body}</main></body></html>")


def build_lab(root: Path = ROOT, out: Path | None = None, sections=SECTIONS) -> Path:
    out = Path(out or Path(root) / "results/lab/index.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(root, sections), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build_lab())
