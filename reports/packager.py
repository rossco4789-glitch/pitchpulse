"""
reports/packager.py
Self-contained HTML matchday dossier packager for Tiverton Town FC.

Converts a dossier markdown string (produced by agents/synthesis.py) into a
fully offline HTML file with:
  - Base64-embedded pitch-plot PNGs (graceful placeholder if missing/empty)
  - OLED dark theme (#09090b) with Tivvy gold (#f59e0b) headers
  - 2-column CSS Grid layout (55 % text / 45 % plot) per moment section
  - @media print — white A4, break-inside: avoid on cards, break-before: page
    per moment section

Public API
----------
    from reports.packager import build_html
    html_path = build_html(dossier_md, plots_dir, out_path)

CLI usage
---------
    python reports/packager.py [--dossier PATH] [--plots-dir DIR] [--out PATH]
"""

import base64
import html as _html
import re
import sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
PROC_DIR  = ROOT / "data" / "processed"
PLOTS_DIR = PROC_DIR / "plots"
HTML_OUT  = PROC_DIR / "tivvy_tactical_dossier.html"

# ── Section → plot mapping ─────────────────────────────────────────────────────
# Each tuple: (keyword_in_section_h2, png_filename, alt_text)
_SECTION_PLOT_MAP: list[tuple[str, str | None, str | None]] = [
    ("MOMENT 1",    "shot_map.png",       "Shot Map"),
    ("MOMENT 2",    "transition_map.png", "Transition Map"),
    ("SET PIECE",   "zonal_heatmap.png",  "Zonal Heatmap"),
    ("NON-LEAGUE",  None,                 None),
    ("DATA QUALITY", None,                None),
]


# ── PNG inlining ───────────────────────────────────────────────────────────────

def _encode_png(path: Path) -> str | None:
    """Base64-encode a PNG file. Returns None if the file is missing or empty."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


def _plot_img(plots_dir: Path, filename: str, alt: str) -> str:
    """Return an <img> with base64 src, or a styled placeholder <div>."""
    data = _encode_png(plots_dir / filename)
    if data:
        return (
            f'<img src="data:image/png;base64,{data}" '
            f'alt="{_html.escape(alt)}" class="pitch-plot">'
        )
    return (
        f'<div class="plot-placeholder">'
        f'<span>{_html.escape(alt)}</span>'
        f'<small>Plot not available — run visualiser first</small>'
        f'</div>'
    )


# ── Markdown-to-HTML converter (regex-only, no external deps) ─────────────────

def _inline(s: str) -> str:
    """Apply inline Markdown formatting after HTML-escaping."""
    s = _html.escape(s)
    # **bold**
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    # _italic_
    s = re.sub(r'_(.+?)_', r'<em>\1</em>', s)
    return s


def _section_to_html(text: str) -> str:
    """Convert one dossier section (Markdown subset) to an HTML fragment."""
    out: list[str] = []
    bar_lines: list[str] = []

    def flush_bars() -> None:
        if bar_lines:
            out.append('<div class="bar-block">')
            for bl in bar_lines:
                out.append(
                    f'<div class="bar-row"><code>{_html.escape(bl)}</code></div>'
                )
            out.append('</div>')
            bar_lines.clear()

    for raw in text.splitlines():
        line = raw.strip()

        # Block-character bar lines — collect and emit as a monospace block
        if '█' in raw:
            bar_lines.append(raw)
            continue
        else:
            flush_bars()

        # Skip separators and blank lines (spacing handled by CSS margin)
        if not line or line == '---':
            continue

        # H1 — document title (title section only)
        if line.startswith('# ') and not line.startswith('## '):
            out.append(f'<h1 class="doc-title">{_inline(line[2:])}</h1>')

        # H2 — moment / section header
        elif line.startswith('## '):
            out.append(
                f'<h2 class="moment-header">{_inline(line[3:])}</h2>'
            )

        # Blockquote — manager note
        elif line.startswith('> '):
            out.append(
                f'<blockquote class="mgr-note">{_inline(line[2:])}</blockquote>'
            )

        # Alert: critical 🚨
        elif '🚨' in line[:20]:
            content = re.sub(r'^[\*\s🚨]+', '', line).rstrip('*')
            out.append(
                f'<div class="alert alert-critical">'
                f'<span class="alert-icon">🚨</span>{_inline(content)}</div>'
            )

        # Alert: QS / positive ⚡
        elif '⚡' in line[:20]:
            content = re.sub(r'^[\*\s⚡]+', '', line).rstrip('*')
            out.append(
                f'<div class="alert alert-qs">'
                f'<span class="alert-icon">⚡</span>{_inline(content)}</div>'
            )

        # Alert: warning ⚠
        elif line.startswith('⚠') or re.match(r'^\*+⚠', line):
            content = re.sub(r'^[\*\s⚠]+', '', line).rstrip('*')
            out.append(
                f'<div class="alert alert-warn">'
                f'<span class="alert-icon">⚠</span>{_inline(content)}</div>'
            )

        # Regular paragraph
        elif line:
            out.append(f'<p>{_inline(line)}</p>')

    flush_bars()
    return '\n'.join(out)


# ── Section rendering ──────────────────────────────────────────────────────────

def _find_plot(section_text: str) -> tuple[str | None, str | None]:
    """Return (png_filename, alt_text) for this section's paired plot."""
    h2 = ''
    for ln in section_text.splitlines():
        if ln.startswith('## '):
            h2 = ln[3:].upper()
            break
    for keyword, filename, alt in _SECTION_PLOT_MAP:
        if keyword in h2:
            return filename, alt
    return None, None


def _render_section(section_text: str, plots_dir: Path, idx: int) -> str:
    """Render one parsed section as an HTML <section> or structural element."""
    upper = section_text.upper()
    is_title   = idx == 0
    is_quality = 'DATA QUALITY' in upper[:40]

    body_html = _section_to_html(section_text)
    filename, alt = _find_plot(section_text)
    has_plot = bool(filename)

    # ── Title / document header ────────────────────────────────────────────────
    if is_title:
        return f'<header class="dossier-header">\n{body_html}\n</header>\n'

    # ── Data Quality footer ────────────────────────────────────────────────────
    if is_quality:
        return (
            f'<footer class="dossier-footer">\n'
            f'<hr class="footer-rule">\n'
            f'{body_html}\n'
            f'</footer>\n'
        )

    section_id = f'section-{idx}'

    # ── Moment with plot (2-column grid) ───────────────────────────────────────
    if has_plot:
        plot_html = _plot_img(plots_dir, filename, alt)  # type: ignore[arg-type]
        return (
            f'<section class="moment-section" id="{section_id}">\n'
            f'  <div class="moment-grid">\n'
            f'    <div class="moment-text">{body_html}</div>\n'
            f'    <div class="moment-plot">{plot_html}</div>\n'
            f'  </div>\n'
            f'</section>\n'
        )

    # ── Full-width section (Non-League Physics, etc.) ──────────────────────────
    return (
        f'<section class="moment-section full-width" id="{section_id}">\n'
        f'  <div class="moment-text">{body_html}</div>\n'
        f'</section>\n'
    )


# ── CSS ────────────────────────────────────────────────────────────────────────

_CSS = """
/* ── Reset & base ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:         #09090b;
  --surface:    #111113;
  --border:     rgba(255,255,255,.09);
  --gold:       #f59e0b;
  --gold-dim:   rgba(245,158,11,.12);
  --text:       rgba(255,255,255,.87);
  --text-muted: rgba(255,255,255,.45);
  --green:      #34d399;
  --green-dim:  rgba(52,211,153,.11);
  --amber:      #fbbf24;
  --amber-dim:  rgba(251,191,36,.11);
  --red:        #f87171;
  --red-dim:    rgba(248,113,113,.11);
  --font:       system-ui, -apple-system, 'Segoe UI', sans-serif;
  --mono:       'Cascadia Code', 'Fira Code', Consolas, monospace;
  --radius:     10px;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 14px;
  line-height: 1.65;
  padding: 2rem 1.5rem;
  max-width: 1100px;
  margin: 0 auto;
}

/* ── Document header ───────────────────────────────────────────── */
.dossier-header {
  border-bottom: 2px solid var(--gold);
  padding-bottom: 1.25rem;
  margin-bottom: 2rem;
}
.doc-title {
  font-size: 1.1rem;
  font-weight: 900;
  letter-spacing: .18em;
  color: var(--gold);
  text-transform: uppercase;
  margin-bottom: .6rem;
}
.dossier-header p  { color: var(--text-muted); font-size: .82rem; margin: .15rem 0; }
.dossier-header strong { color: var(--text); }

/* ── Moment sections ───────────────────────────────────────────── */
.moment-section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem;
  margin-bottom: 1.75rem;
}

/* ── 2-column grid ─────────────────────────────────────────────── */
.moment-grid {
  display: grid;
  grid-template-columns: 55fr 45fr;
  gap: 1.5rem;
  align-items: start;
}
.full-width .moment-text { max-width: 100%; }

/* ── Mobile: collapse to single column (manager's phone) ────────── */
@media (max-width: 640px) {
  .moment-grid { grid-template-columns: 1fr; }
  .moment-plot { order: -1; }           /* plot above text on narrow screens */
  body { padding: 1rem; font-size: 13px; }
  .moment-section { padding: 1rem; }
}

/* ── Section headers ───────────────────────────────────────────── */
.moment-header {
  font-size: .76rem;
  font-weight: 900;
  letter-spacing: .22em;
  color: var(--gold);
  text-transform: uppercase;
  border-bottom: 1px solid var(--gold-dim);
  padding-bottom: .5rem;
  margin-bottom: 1rem;
}

/* ── Body text ─────────────────────────────────────────────────── */
p { margin: .45rem 0; }
p strong { color: #fff; }

/* ── Manager note blockquote ───────────────────────────────────── */
blockquote.mgr-note {
  border-left: 3px solid var(--amber);
  background: var(--amber-dim);
  border-radius: 0 6px 6px 0;
  padding: .5rem .8rem;
  margin: .75rem 0;
  font-size: .82rem;
  color: var(--amber);
  font-style: italic;
}

/* ── Alert boxes ───────────────────────────────────────────────── */
.alert {
  display: flex;
  align-items: flex-start;
  gap: .5rem;
  border-radius: 7px;
  padding: .55rem .8rem;
  margin: .6rem 0;
  font-size: .82rem;
  line-height: 1.5;
}
.alert-icon { flex-shrink: 0; font-size: 1rem; margin-top: .05rem; }
.alert strong { color: inherit; }

.alert-warn     { background: var(--amber-dim); border: 1px solid rgba(251,191,36,.25); color: var(--amber); }
.alert-critical { background: var(--red-dim);   border: 1px solid rgba(248,113,113,.25); color: var(--red); }
.alert-qs       { background: var(--green-dim); border: 1px solid rgba(52,211,153,.25); color: var(--green); }

/* ── Bar chart rows ────────────────────────────────────────────── */
.bar-block {
  background: rgba(255,255,255,.025);
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: .45rem .7rem;
  margin: .5rem 0;
  overflow-x: auto;
}
.bar-row code {
  display: block;
  font-family: var(--mono);
  font-size: .76rem;
  color: rgba(255,255,255,.68);
  white-space: pre;
  padding: .12rem 0;
}

/* ── Pitch plot column ─────────────────────────────────────────── */
.moment-plot {
  display: flex;
  align-items: flex-start;
  justify-content: center;
}
img.pitch-plot {
  width: 100%;
  max-width: 100%;
  border-radius: 8px;
  border: 1px solid var(--border);
  display: block;
}
.plot-placeholder {
  width: 100%;
  min-height: 180px;
  border: 1px dashed rgba(255,255,255,.14);
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: .4rem;
  color: var(--text-muted);
  font-size: .78rem;
  text-align: center;
  padding: 1rem;
}
.plot-placeholder span { font-weight: 700; font-size: .85rem; }

/* ── Footer ────────────────────────────────────────────────────── */
.dossier-footer { margin-top: 2rem; padding-top: 1rem; }
.footer-rule    { border: none; border-top: 1px solid var(--border); margin-bottom: 1rem; }
.dossier-footer p, .dossier-footer em { color: var(--text-muted); font-size: .78rem; }
.dossier-footer strong { color: var(--text); }

/* ── Print / A4 stylesheet ─────────────────────────────────────── */
@media print {
  @page { size: A4; margin: 15mm 12mm; }

  :root {
    --bg:         #ffffff;
    --surface:    #f7f7f7;
    --border:     #ddd;
    --gold:       #92600a;
    --gold-dim:   #fef3c7;
    --text:       #1a1a1a;
    --text-muted: #555;
    --green:      #065f46;
    --green-dim:  #d1fae5;
    --amber:      #92600a;
    --amber-dim:  #fef3c7;
    --red:        #991b1b;
    --red-dim:    #fee2e2;
  }

  body { background: #fff; color: #1a1a1a; padding: 0; max-width: 100%; font-size: 11pt; }

  /* Each moment section starts on a fresh page */
  .moment-section { page-break-before: always; break-before: page; break-inside: avoid; box-shadow: none; }

  /* Header on the opening page — do not force a break before it */
  .dossier-header { break-before: avoid; break-after: avoid; }

  /* Sub-elements that must not split across pages */
  .moment-grid, .bar-block, .alert, blockquote.mgr-note { break-inside: avoid; }

  img.pitch-plot { border-radius: 0; width: 100%; }

  .moment-header { color: #92600a; border-color: #fde68a; }
  .doc-title     { color: #92600a; }

  .dossier-footer { break-before: avoid; }

  /* Suppress browser-added URL annotations on links */
  a[href]::after { content: none !important; }
}
"""

# ── HTML template ──────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tiverton Town FC — Tactical Dossier{match_suffix}</title>
  <style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


# ── Public API ─────────────────────────────────────────────────────────────────

def build_html(
    dossier_md: str,
    plots_dir:  Path = PLOTS_DIR,
    out_path:   Path = HTML_OUT,
) -> Path:
    """
    Convert a dossier markdown string to a self-contained HTML report.

    Parameters
    ----------
    dossier_md : str
        Full dossier text as produced by ``agents/synthesis.py``.
    plots_dir  : Path
        Directory containing shot_map.png, transition_map.png, zonal_heatmap.png.
    out_path   : Path
        Output path for the HTML file.

    Returns
    -------
    Path
        The path of the written HTML file.
    """
    # Split on horizontal rules (any number of dashes, any surrounding newlines)
    raw_sections = re.split(r'\n+---+\n+', dossier_md.strip())
    sections     = [s.strip() for s in raw_sections if s.strip()]

    body_parts: list[str] = []
    for idx, section in enumerate(sections):
        body_parts.append(_render_section(section, plots_dir, idx))

    # Extract match date for the <title> tag (graceful: empty string on miss)
    match_suffix = ''
    title_section = sections[0] if sections else ''
    for ln in title_section.splitlines():
        if ln.startswith('**Match Date:**'):
            match_suffix = ' — ' + ln.replace('**Match Date:**', '').strip()
            break

    html = _HTML_TEMPLATE.format(
        match_suffix=_html.escape(match_suffix),
        css=_CSS,
        body='\n'.join(body_parts),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding='utf-8')
    return out_path


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description='PitchPulse HTML dossier packager — Tiverton Town FC'
    )
    parser.add_argument(
        '--dossier', default=None,
        help='Path to dossier .md file (default: most-recent reports/dossier_*.md)',
    )
    parser.add_argument(
        '--plots-dir', default=str(PLOTS_DIR),
        help=f'Directory containing plot PNGs (default: {PLOTS_DIR.relative_to(ROOT)})',
    )
    parser.add_argument(
        '--out', default=str(HTML_OUT),
        help=f'Output HTML path (default: {HTML_OUT.relative_to(ROOT)})',
    )
    args = parser.parse_args()

    # Locate dossier markdown
    if args.dossier:
        md_path = Path(args.dossier)
    else:
        candidates = sorted(
            (ROOT / 'reports').glob('dossier_*.md'),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(
                '  [PACK] ERROR — no dossier_*.md found in reports/. '
                'Run run_matchday.py first.'
            )
            sys.exit(1)
        md_path = candidates[0]

    print(f'\n  [PACK] Reading dossier: {md_path}')
    dossier_md = md_path.read_text(encoding='utf-8')

    plots_dir = Path(args.plots_dir)
    out_path  = Path(args.out)

    out = build_html(dossier_md, plots_dir, out_path)
    size_kb = round(out.stat().st_size / 1024, 1)
    print(f'  [PACK] HTML written → {out}  ({size_kb} KB)')
    print(f'  [PACK] Open: file:///{out.as_posix()}\n')


if __name__ == '__main__':
    main()
