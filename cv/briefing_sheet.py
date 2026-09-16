"""
cv/briefing_sheet.py — one-page printable match-day sheet for the Opposition Analysis tab (app.py tab 6).

The dashboard is dark and scrolls; the dressing room needs one white A4 sheet. These pure builders render the
same briefing card and pitch (or the highlight dossier) as a light, low-ink layout inside .vcc-sheet, which is
hidden on screen. PRINT_CSS only takes over when the "Export Briefing" action adds the vcc-printing class to the
page: everything except the sheet is then hidden and page breaks inside the sheet are avoided. Printing any other
tab is unaffected.
"""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path

from cv import coach_brief as cb

CREST_MAX_B = 300_000

PRINT_CSS = """
<style>
.vcc-sheet { display:none; width:100%; background:#FFFFFF; color:#111418; font:9pt/1.35 'Segoe UI',Inter,Arial,sans-serif;
  -webkit-print-color-adjust:exact; print-color-adjust:exact; }
.vcc-sheet * { box-sizing:border-box; }
.vcc-sheet .sh-head { display:flex; align-items:center; gap:5mm; border-bottom:1.2mm solid #1F6F3A; padding-bottom:3mm; margin-bottom:3mm; break-inside:avoid; }
.vcc-sheet .sh-crest { width:16mm; height:16mm; object-fit:contain; }
.vcc-sheet .sh-title { flex:1; }
.vcc-sheet .sh-eyebrow { font:700 7pt Bahnschrift,'Arial Narrow',Arial,sans-serif; letter-spacing:.18em; text-transform:uppercase; color:#1F6F3A; }
.vcc-sheet .sh-opp { font:700 20pt/1.05 Bahnschrift,'Arial Narrow',Arial,sans-serif; text-transform:uppercase; color:#0B0F14; }
.vcc-sheet .sh-sub { font:600 9pt 'Segoe UI',Arial,sans-serif; color:#3A4450; margin-top:1mm; }
.vcc-sheet .sh-badges { display:flex; flex-direction:column; align-items:flex-end; gap:1.5mm; }
.vcc-sheet .sh-struct { font:700 10pt Bahnschrift,Arial,sans-serif; letter-spacing:.12em; border:0.5mm solid #0B0F14; padding:1.5mm 3mm; color:#0B0F14; }
.vcc-sheet .sh-conf { font:600 8pt 'Segoe UI',Arial,sans-serif; padding:1mm 2.5mm; border-radius:3mm; border:0.3mm solid currentColor; }
.vcc-sheet .sh-conf.green { color:#1B6E2E; } .vcc-sheet .sh-conf.amber { color:#8A5A00; }
.vcc-sheet .sh-conf.red { color:#B3261E; } .vcc-sheet .sh-conf.grey { color:#57606A; }
.vcc-sheet .sh-metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:2.5mm; margin-bottom:3mm; break-inside:avoid; }
.vcc-sheet .sh-metrics > div { border:0.3mm solid #C9D1D9; border-top:1mm solid #1F6F3A; padding:2mm 2.5mm; }
.vcc-sheet .sh-metrics b, .vcc-sheet .sh-quick b, .vcc-sheet .sh-item b { display:block; font:700 6.5pt Bahnschrift,Arial,sans-serif;
  letter-spacing:.14em; text-transform:uppercase; color:#57606A; }
.vcc-sheet .sh-metrics span { display:block; font:700 12pt Bahnschrift,Arial,sans-serif; color:#0B0F14; margin-top:.8mm; }
.vcc-sheet .sh-metrics em { font-style:normal; font-size:8pt; color:#1B6E2E; }
.vcc-sheet .sh-body { display:grid; grid-template-columns:1.25fr 1fr; gap:4mm; break-inside:avoid; }
.vcc-sheet section { break-inside:avoid; margin-bottom:2.5mm; border-left:1mm solid #1F6F3A; padding-left:2.5mm; }
.vcc-sheet section.p1 { border-left-color:#1F6FEB; } .vcc-sheet section.p3 { border-left-color:#B7791F; }
.vcc-sheet h3 { margin:0 0 1mm; font:700 9.5pt Bahnschrift,Arial,sans-serif; text-transform:uppercase; letter-spacing:.05em; color:#0B0F14; }
.vcc-sheet h3 small { font:600 7.5pt 'Segoe UI',Arial,sans-serif; text-transform:none; letter-spacing:0; color:#57606A; margin-left:2mm; }
.vcc-sheet ul { margin:0; padding-left:4mm; } .vcc-sheet li { margin:0 0 .7mm; font-size:8.6pt; color:#111418; }
.vcc-sheet .sh-pitch svg { width:100%; height:auto; display:block; }
.vcc-sheet .sh-legend { display:flex; flex-wrap:wrap; gap:3mm; font-size:7.5pt; color:#3A4450; margin-top:1.5mm; }
.vcc-sheet .sh-legend i { display:inline-block; vertical-align:middle; margin-right:1.2mm; }
.vcc-sheet .sh-legend .def { width:6mm; height:0.8mm; background:#1F6FEB; }
.vcc-sheet .sh-legend .press { width:6mm; height:0; border-top:0.8mm dashed #D1242F; }
.vcc-sheet .sh-legend .corr { width:4mm; height:3mm; background:rgba(31,111,235,.22); border:0.3mm solid #1F6FEB; }
.vcc-sheet .sh-legend .flank { width:4mm; height:3mm; background:rgba(46,160,67,.22); border:0.3mm dashed #B7791F; }
.vcc-sheet .sh-note { border:0.3mm solid #C9D1D9; padding:3mm; font-size:10pt; color:#8A5A00; }
.vcc-sheet .sh-quick { display:grid; grid-template-columns:1fr 1fr; gap:3mm; margin-bottom:3mm; break-inside:avoid; }
.vcc-sheet .sh-quick > div { border:0.3mm solid #C9D1D9; border-left:1.2mm solid #D1242F; padding:2mm 3mm; }
.vcc-sheet .sh-quick > div.weak { border-left-color:#1B6E2E; }
.vcc-sheet .sh-quick span { display:block; font:700 11pt Bahnschrift,Arial,sans-serif; color:#0B0F14; margin-top:.8mm; }
.vcc-sheet .sh-patterns { border:0.3mm solid #C9D1D9; padding:2mm 3mm 0; margin-bottom:3mm; break-inside:avoid; }
.vcc-sheet .sh-patterns h3 { margin:0 0 1.5mm; font:700 8pt Bahnschrift,Arial,sans-serif; letter-spacing:.12em; text-transform:uppercase; color:#0B0F14; }
.vcc-sheet .sh-pgrid { display:grid; grid-template-columns:repeat(3,1fr); gap:4mm; }
.vcc-sheet .sh-sections { display:grid; grid-template-columns:repeat(3,1fr); gap:4mm; break-inside:avoid; }
.vcc-sheet .sh-item { margin-bottom:2mm; break-inside:avoid; }
.vcc-sheet .sh-item p { margin:.5mm 0 0; font-size:8.6pt; color:#111418; }
.vcc-sheet .sh-item p.lever { color:#1B6E2E; font-weight:600; } .vcc-sheet .sh-item p.empty { color:#8C959F; }
.vcc-sheet .sh-foot { display:flex; justify-content:space-between; gap:4mm; margin-top:3mm; padding-top:1.5mm;
  border-top:0.3mm solid #C9D1D9; font-size:7pt; color:#57606A; }
@media print {
  html.vcc-printing, html.vcc-printing body { background:#FFFFFF !important; }
  html.vcc-printing body *:not(.vcc-sheet):not(.vcc-sheet *):not(:has(.vcc-sheet)) { display:none !important; }
  html.vcc-printing *:has(.vcc-sheet) { background:transparent !important; border:0 !important; box-shadow:none !important;
    padding:0 !important; margin:0 !important; height:auto !important; min-height:0 !important; max-width:none !important;
    overflow:visible !important; position:static !important; transform:none !important; }
  html.vcc-printing .vcc-sheet { display:block !important; break-inside:avoid; }
}
</style>
"""

LEGEND = ('<div class="sh-legend"><span><i class="def"></i>Defensive line</span><span><i class="press"></i>Press trigger</span>'
          '<span><i class="corr"></i>Compact corridor</span><span><i class="flank"></i>Space on flanks</span></div>')


def crest_data_uri(path: str | Path | None) -> str | None:
    """Embed a cached crest so the printed sheet needs no network; None if missing or oversized."""
    if not path:
        return None
    file = Path(path)
    if not file.is_file() or file.stat().st_size > CREST_MAX_B:
        return None
    return "data:image/png;base64," + base64.b64encode(file.read_bytes()).decode("ascii")


def _head(eyebrow: str, opponent: str, subline: str, badge: str | None, confidence: dict, crest_uri: str | None) -> str:
    crest = f'<img class="sh-crest" src="{crest_uri}" alt="">' if crest_uri else ""
    return (f'<div class="sh-head">{crest}<div class="sh-title"><div class="sh-eyebrow">{escape(eyebrow)}</div>'
            f'<div class="sh-opp">{escape(opponent)}</div>' + (f'<div class="sh-sub">{escape(subline)}</div>' if subline else "")
            + '</div><div class="sh-badges">' + (f'<span class="sh-struct">{escape(badge)}</span>' if badge else "")
            + f'<span class="sh-conf {confidence["tone"]}">{escape(confidence["text"])}</span></div></div>')


def _foot(generated: str) -> str:
    return (f'<div class="sh-foot"><span>PitchPulse · Tiverton Town FC opposition analysis</span>'
            f'<span>Prepared {escape(generated)}</span></div>')


def full_sheet(brief: dict, geo: dict | None, crest_uri: str | None = None, generated: str = "") -> str:
    """Briefing card, three phases and the print-theme pitch on one A4 sheet (single line, Markdown-safe)."""
    head = _head(f"Tiverton Town FC · Match-day briefing · {brief['date']}", brief["opponent"],
                 f"Team shape: {brief['shape']}" if brief["shape"] else "", brief["badge"], brief["confidence"], crest_uri)
    if not brief["available"]:
        return f'<div class="vcc-sheet">{head}<div class="sh-note">{escape(brief["unavailable"])}</div>{_foot(generated)}</div>'
    metrics = "".join(f'<div><b>{escape(m["label"])}</b><span>{escape(m["value"])}</span><em>{escape(m["detail"])}</em></div>'
                      for m in brief["metrics"])
    phases = []
    for i, phase in enumerate(brief["phases"], start=1):
        points = "".join(f"<li>{escape(point)}</li>" for point in phase["points"])
        phases.append(f'<section class="p{i}"><h3>{i}. {escape(phase["title"])}<small>{escape(phase["subtitle"])}</small></h3>'
                      f"<ul>{points}</ul></section>")
    pitch = cb.pitch_svg(geo, theme="print") + LEGEND if geo else ""
    return (f'<div class="vcc-sheet">{head}<div class="sh-metrics">{metrics}</div>'
            f'<div class="sh-body"><div class="sh-phases">{"".join(phases)}</div><div class="sh-pitch">{pitch}</div></div>'
            f"{_foot(generated)}</div>")


def dossier_sheet(dossier: dict, crest_uri: str | None = None, generated: str = "") -> str:
    """Highlight & Tendency Dossier as one A4 sheet: quick read, then the three sections side by side."""
    head = _head("Tiverton Town FC · Opposition scouting dossier", dossier["opponent"], dossier["note"],
                 dossier["footage"].upper(), dossier["confidence"], crest_uri)
    quick = dossier["quick_read"]
    quick_read = (f'<div class="sh-quick"><div><b>Primary threat</b><span>{escape(quick["threat"])}</span></div>'
                  f'<div class="weak"><b>Primary vulnerability</b><span>{escape(quick["vulnerability"])}</span></div></div>')
    patterns = dossier["patterns"]
    pattern_row = (f'<div class="sh-patterns"><h3>{escape(patterns["title"])}</h3>'
                   f'<div class="sh-pgrid">{"".join(_sheet_item(item) for item in patterns["items"])}</div></div>')
    sections = [f'<section class="p{i}"><h3>{i}. {escape(section["title"])}</h3>{"".join(_sheet_item(item) for item in section["items"])}</section>'
                for i, section in enumerate(dossier["sections"], start=1)]
    return (f'<div class="vcc-sheet">{head}{quick_read}{pattern_row}<div class="sh-sections">{"".join(sections)}</div>'
            f"{_foot(generated)}</div>")


def _sheet_item(item: dict) -> str:
    read_class = "" if item["lever"] else ' class="empty"'
    lever = f'<p class="lever">→ {escape(item["lever"])}</p>' if item["lever"] else ""
    return f'<div class="sh-item"><b>{escape(item["label"])}</b><p{read_class}>{escape(item["read"])}</p>{lever}</div>'
