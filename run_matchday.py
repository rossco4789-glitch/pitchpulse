#!/usr/bin/env python3
"""
run_matchday.py
Master matchday orchestration pipeline for Tiverton Town FC.

Connects the three pipeline stages in sequence:
  Step 1 — reconcile/sync.py       : ingest tagger + feed → match_ledger.json
  Step 2 — reports/visualizer.py   : render shot map, transition map, heatmap → plots/
  Step 3 — agents/synthesis.py     : run three tactical agents → CLI gate → dossier_*.md

Usage:
  python run_matchday.py                          # auto-detect newest ledger, run all steps
  python run_matchday.py --latest                 # explicit latest-ledger mode
  python run_matchday.py --skip-reconcile         # skip Step 1 (use existing ledger)
  python run_matchday.py --skip-visuals           # skip Step 2 (agents only)
  python run_matchday.py --skip-reconcile --skip-visuals   # agents only, fastest path

Error policy:
  Step 1 failure → hard stop  (no ledger = no downstream work)
  Step 2 failure → warn only  (visuals are non-blocking for the dossier)
  Step 3 failure → hard stop  (dossier is the matchday deliverable)
"""

from __future__ import annotations

import argparse
import sys
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── Project root on sys.path ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────────
PROC_DIR    = ROOT / "data" / "processed"
LEDGER_PATH = PROC_DIR / "match_ledger.json"

_COL = 72


# ══════════════════════════════════════════════════════════════════════════════
# Terminal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _rule(char: str = "═") -> None:
    print(char * _COL)


def _step_header(n: int, label: str, skipped: bool = False) -> None:
    print()
    _rule()
    status = "  [SKIP]" if skipped else f"  STEP {n}"
    print(f"{status}  {label}")
    _rule()
    print()


def _ok(msg: str)   -> None: print(f"  ✓  {msg}")
def _warn(msg: str) -> None: print(f"  ⚠  {msg}")
def _err(msg: str)  -> None: print(f"  ✗  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Reconciliation
# ══════════════════════════════════════════════════════════════════════════════

def step_reconcile() -> Path:
    """
    Import and run reconcile.sync inline.
    Hard-stops on failure — no ledger means no pipeline.
    Returns the Path of the written ledger.
    """
    try:
        from reconcile.sync import (
            ensure_dirs,
            load_tag_events,
            load_club_feed,
            extract_club_events,
            reconcile_events,
            build_ledger,
            write_ledger,
            print_audit,
            RAW_DIR,
            LEDGER_OUT,
            RECON_WINDOW_S,
        )
    except ImportError as exc:
        _err(f"Cannot import reconcile.sync: {exc}")
        sys.exit(1)

    ensure_dirs()

    tag_events = load_tag_events(RAW_DIR)
    feed_data, feed_type = load_club_feed(RAW_DIR)

    club_events = []
    if feed_type != "none" and feed_data is not None:
        club_events = extract_club_events(feed_data, feed_type)

    matched, unmatched_tags, unmatched_club = reconcile_events(
        tag_events, club_events, window=RECON_WINDOW_S
    )

    ledger = build_ledger(matched, unmatched_tags, unmatched_club)
    write_ledger(ledger, LEDGER_OUT)
    print_audit(ledger)

    _ok(f"Ledger written → {LEDGER_OUT.relative_to(ROOT)}")
    return LEDGER_OUT


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Visualisations
# ══════════════════════════════════════════════════════════════════════════════

def step_visuals(ledger_path: Path) -> bool:
    """
    Import reports.visualizer and render all three plots from the ledger.
    Non-blocking — warns on failure rather than hard-stopping.
    Returns True if all plots succeeded, False otherwise.
    """
    try:
        import reports.visualizer as viz
    except ImportError as exc:
        _warn(f"Cannot import reports.visualizer: {exc}")
        _warn("Visuals skipped — continuing to agents.")
        return False

    with open(ledger_path, encoding="utf-8") as f:
        ledger = json.load(f)

    # Flatten all tag events from matched + unmatched for the visualizer.
    # v2 events carry real x_m, y_m and zone_id from reconcile/sync.py.
    # Legacy v1 events (no spatial data) render at zone centroids if zone_id
    # is present, or are omitted from position-sensitive plots.
    all_tags: list[dict] = []
    for m in ledger.get("matched", []):
        all_tags.append(m["tag"])
    all_tags.extend(ledger.get("unmatched_tags", []))

    plots_dir = PROC_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    success = True
    for plot_fn, label, filename in [
        (viz.plot_shot_map,       "Shot Map",       "shot_map.png"),
        (viz.plot_transition_map, "Transition Map", "transition_map.png"),
        (viz.plot_zonal_heatmap,  "Zonal Heatmap",  "zonal_heatmap.png"),
    ]:
        try:
            out = plot_fn(all_tags, output_path=plots_dir / filename)
            _ok(f"{label} → {Path(out).relative_to(ROOT)}")
        except Exception:
            _warn(f"{label} failed:")
            traceback.print_exc()
            success = False

    return success


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Tactical agents + approval gate
# ══════════════════════════════════════════════════════════════════════════════

def step_agents(ledger_path: Path) -> "Path | None":
    """
    Import agents.synthesis and run the full agent pipeline with CLI approval gate.
    Hard-stops on import failure.
    Returns the Path of the written dossier, or None if not saved.
    """
    try:
        from agents.synthesis import run_approval_gate
    except ImportError as exc:
        _err(f"Cannot import agents.synthesis: {exc}")
        sys.exit(1)

    with open(ledger_path, encoding="utf-8") as f:
        ledger = json.load(f)

    return run_approval_gate(ledger)


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — HTML dossier packager
# ══════════════════════════════════════════════════════════════════════════════

def step_package(dossier_md_path: Path, plots_dir: Path) -> Path:
    """
    Package the approved dossier markdown into a self-contained HTML report
    with embedded pitch-plot PNGs (base64), OLED dark theme, and A4 print CSS.
    Returns the path of the written HTML file.
    """
    try:
        from reports.packager import build_html
    except ImportError as exc:
        _err(f"Cannot import reports.packager: {exc}")
        sys.exit(1)

    dossier_md = dossier_md_path.read_text(encoding="utf-8")
    html_out   = PROC_DIR / "tivvy_tactical_dossier.html"
    out        = build_html(dossier_md, plots_dir, html_out)
    size_kb    = round(out.stat().st_size / 1024, 1)
    _ok(f"HTML dossier → {out.relative_to(ROOT)}  ({size_kb} KB)")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PitchPulse matchday pipeline — Tiverton Town FC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Use the most recently modified match_ledger.json (default behaviour)",
    )
    parser.add_argument(
        "--skip-reconcile", action="store_true",
        help="Skip Step 1 — use the existing match_ledger.json on disk",
    )
    parser.add_argument(
        "--skip-visuals", action="store_true",
        help="Skip Step 2 — bypass visualiser and go straight to agents",
    )
    args = parser.parse_args()

    print()
    _rule("═")
    print("  PITCHPULSE — MATCHDAY PIPELINE")
    print("  Tiverton Town FC | UEFA Pro Licence Standard")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    _rule("═")

    # ── Step 1: Reconciliation ─────────────────────────────────────────────────
    _step_header(1, "POST-MATCH RECONCILIATION", skipped=args.skip_reconcile)

    if args.skip_reconcile:
        if not LEDGER_PATH.exists():
            _err(
                f"--skip-reconcile specified but no ledger found at "
                f"{LEDGER_PATH.relative_to(ROOT)}\n"
                "  Run without --skip-reconcile to generate it first."
            )
            sys.exit(1)
        ledger_path = LEDGER_PATH
        _ok(f"Using existing ledger → {ledger_path.relative_to(ROOT)}")
    else:
        ledger_path = step_reconcile()

    # ── Step 2: Visualisations ─────────────────────────────────────────────────
    _step_header(2, "PITCH VISUALISATIONS", skipped=args.skip_visuals)

    if args.skip_visuals:
        _warn("Visuals skipped by flag — plots will not be updated.")
    else:
        vis_ok = step_visuals(ledger_path)
        if not vis_ok:
            _warn("One or more plots failed. Proceeding to agents.")

    # ── Step 3: Tactical agents ────────────────────────────────────────────────
    _step_header(3, "TACTICAL ANALYSIS AGENTS + DOSSIER APPROVAL")

    dossier_path = step_agents(ledger_path)

    # ── Step 4: HTML packager ──────────────────────────────────────────────────
    if dossier_path:
        _step_header(4, "HTML DOSSIER PACKAGER")
        plots_dir = PROC_DIR / "plots"
        step_package(dossier_path, plots_dir)

    # ── Pipeline summary ───────────────────────────────────────────────────────
    print()
    _rule("═")
    print("  PIPELINE COMPLETE")
    if dossier_path:
        _ok("Dossier approved, written, and packaged as HTML.")
    else:
        _warn("Dossier was not saved (quit or pipeline error).")
    _rule("═")
    print()

    sys.exit(0 if dossier_path else 1)


if __name__ == "__main__":
    main()
