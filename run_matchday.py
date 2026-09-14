#!/usr/bin/env python3
"""
run_matchday.py
Master matchday orchestration pipeline for Tiverton Town FC.

Connects the pipeline stages in sequence:
  Step 1 — reconcile/sync.py       : ingest tagger + feed → match_ledger.json
  Step 2 — reports/visualizer.py   : render shot map, transition map, heatmap → plots/
  Step 3 — agents/synthesis.py     : run four tactical agents → CLI gate → dossier_*.md
  Step 4 — reports/packager.py     : HTML dossier with embedded PNGs
  Step 5 — reports/dof_card.py     : 1080×1920 DoF match card PNG (WhatsApp-ready)

Usage:
  python run_matchday.py                          # auto-detect newest ledger, run all steps
  python run_matchday.py --latest                 # explicit latest-ledger mode
  python run_matchday.py --skip-reconcile         # skip Step 1 (use existing ledger)
  python run_matchday.py --skip-visuals           # skip Step 2 (agents only)
  python run_matchday.py --skip-reconcile --skip-visuals   # agents only, fastest path
  python run_matchday.py --date 2026-09-19 --opponent "Willand Rovers"   # run_id 2026-09-19_willand_rovers
  python run_matchday.py --run-id 2026-09-19_willand_rovers              # explicit run_id wins

Run ID (eval ledger key shared by tagger_sanity and the packager gate):
  --run-id given              → used verbatim
  --date and/or --opponent    → {YYYY-MM-DD}_{opponent_slug}  (missing date = today, missing opponent = matchday)
  neither                     → {today}_matchday

Error policy:
  Step 1 failure → hard stop  (no ledger = no downstream work)
  Sanity ERRORs  → warn here, block at Step 4 via the eval gate
  Step 2 failure → warn only  (visuals are non-blocking for the dossier)
  Step 3 failure → hard stop  (dossier is the matchday deliverable)
"""

from __future__ import annotations

import argparse
import re
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


def _match_date(value: str) -> str:
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got '{value}'")


def resolve_run_id(run_id: str | None, date: str | None, opponent: str | None) -> str:
    """Explicit --run-id wins; else {date}_{opponent_slug}; else {today}_matchday."""
    if run_id:
        return run_id
    day  = date or datetime.now().strftime("%Y-%m-%d")
    slug = re.sub(r"[^a-z0-9]+", "_", (opponent or "").lower()).strip("_") or "matchday"
    return f"{day}_{slug}"


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Reconciliation
# ══════════════════════════════════════════════════════════════════════════════

def step_reconcile(run_id: str) -> Path:
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
    ledger["run_id"] = run_id
    write_ledger(ledger, LEDGER_OUT)
    print_audit(ledger)

    _ok(f"Ledger written → {LEDGER_OUT.relative_to(ROOT)}  [run_id {run_id}]")
    return LEDGER_OUT


# ══════════════════════════════════════════════════════════════════════════════
# Step 1b — Tagger sanity audit → eval ledger
# ══════════════════════════════════════════════════════════════════════════════

def step_sanity(ledger_path: Path, run_id: str) -> int:
    """
    Run tools/tagger_sanity checks inline and log findings under run_id.
    Non-blocking here — unresolved ERRORs stop Step 4 via the eval gate.
    Returns the number of ERROR findings.
    """
    tools_dir = str(ROOT / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import tagger_sanity

    with open(ledger_path, encoding="utf-8") as f:
        events = tagger_sanity.extract_events(json.load(f))

    findings = tagger_sanity.log_findings(tagger_sanity.run_checks(events), events, run_id)
    tagger_sanity.print_report(findings, len(events))
    errors = sum(1 for f in findings if f[0] == "ERROR")
    _ok(f"{len(findings)} finding(s) logged to eval ledger under run_id '{run_id}'")
    if errors:
        _warn(f"{errors} ERROR(s) — Step 4 packaging will be blocked until resolved.")
    return errors


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

def step_package(dossier_md_path: Path, plots_dir: Path, run_id: str) -> Path:
    """
    Package the approved dossier markdown into a self-contained HTML report
    with embedded pitch-plot PNGs (base64), OLED dark theme, and A4 print CSS.
    Exits 1 via the eval gate if run_id has unresolved ERRORs.
    Returns the path of the written HTML file.
    """
    try:
        from reports.packager import build_html, check_gate
    except ImportError as exc:
        _err(f"Cannot import reports.packager: {exc}")
        sys.exit(1)

    check_gate(run_id)

    dossier_md = dossier_md_path.read_text(encoding="utf-8")
    html_out   = PROC_DIR / "tivvy_tactical_dossier.html"
    out        = build_html(dossier_md, plots_dir, html_out)
    size_kb    = round(out.stat().st_size / 1024, 1)
    _ok(f"HTML dossier → {out.relative_to(ROOT)}  ({size_kb} KB)")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — DoF match card
# ══════════════════════════════════════════════════════════════════════════════

def step_dof_card(ledger_path: Path, plots_dir: Path) -> Path | None:
    """
    Render the 1080×1920 DoF summary card PNG.
    Non-blocking — warns on failure.
    Returns the output Path or None on failure.
    """
    try:
        from reports.dof_card import build_dof_card, CONTEXT_PATH, OUT_PATH
    except ImportError as exc:
        _warn(f"Cannot import reports.dof_card: {exc}")
        return None

    with open(ledger_path, encoding="utf-8") as f:
        ledger = json.load(f)

    ctx: dict = {}
    if CONTEXT_PATH.exists():
        with open(CONTEXT_PATH, encoding="utf-8") as f:
            ctx = json.load(f)
    else:
        _warn("match_context.json not found — DoF card will render without match context.")

    try:
        out      = build_dof_card(ledger, ctx, plots_dir, OUT_PATH)
        size_kb  = round(out.stat().st_size / 1024, 1)
        _ok(f"DoF card → {out.relative_to(ROOT)}  ({size_kb} KB  |  1080×1920 px)")
        return out
    except Exception:
        _warn("DoF card render failed:")
        traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # UTF-8 output so banners (═ → ✓) survive redirection on Windows cp1252 consoles
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except AttributeError:
            pass

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
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Eval ledger run id shared by tagger_sanity and the packager gate",
    )
    parser.add_argument(
        "--date", type=_match_date, default=None,
        help="Match date YYYY-MM-DD (run_id fallback component)",
    )
    parser.add_argument(
        "--opponent", type=str, default=None,
        help='Opponent name, e.g. "Willand Rovers" (run_id fallback component)',
    )
    args = parser.parse_args()
    run_id = resolve_run_id(args.run_id, args.date, args.opponent)

    print()
    _rule("═")
    print("  PITCHPULSE — MATCHDAY PIPELINE")
    print("  Tiverton Town FC | Tactical Performance Intelligence")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Run ID : {run_id}")
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
        ledger_path = step_reconcile(run_id)

    # ── Step 1b: Tagger sanity → eval ledger ───────────────────────────────────
    _step_header(1, "TAGGER SANITY AUDIT  (eval ledger)")
    step_sanity(ledger_path, run_id)

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

    plots_dir = PROC_DIR / "plots"

    # ── Step 4: HTML packager ──────────────────────────────────────────────────
    if dossier_path:
        _step_header(4, "HTML DOSSIER PACKAGER")
        step_package(dossier_path, plots_dir, run_id)

    # ── Step 5: DoF match card ─────────────────────────────────────────────────
    _step_header(5, "DOF MATCH CARD  (1080×1920 WhatsApp PNG)")
    step_dof_card(ledger_path, plots_dir)

    # ── Pipeline summary ───────────────────────────────────────────────────────
    print()
    _rule("═")
    print("  PIPELINE COMPLETE")
    if dossier_path:
        _ok("Dossier approved, packaged as HTML, and DoF card rendered.")
    else:
        _warn("Dossier was not saved (quit or pipeline error). DoF card still rendered.")
    _rule("═")
    print()

    sys.exit(0 if dossier_path else 1)


if __name__ == "__main__":
    main()
