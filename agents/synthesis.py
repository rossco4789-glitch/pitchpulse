"""
agents/synthesis.py
Multi-agent tactical analysis engine for Tiverton Town FC.
Evaluated strictly through the UEFA 4 Moments of the Game framework.

Agents
------
  Agent 1 — InPossessionAgent  : SHOT + BOX_ENTRY  → penetration, corridors, half-space use
  Agent 2 — PressAgent         : HIGH_REGAIN + DEF_TURNOVER → pressing efficiency, LoE, compactness
  Agent 3 — SetPieceAgent      : SET_PIECE → dead-ball attack/defence, Rest Defense exposure

Zone inference
--------------
  The mobile tagger captures no x/y coordinates. Zones are inferred from
  event_type + sub_type using the 18-zone matrix from cv/zones.py:
    THIRD  ∈ {D, M, A}   (defensive / middle / attacking)
    CHANNEL ∈ {LF, LH, LC, RC, RH, RF}
  Half-spaces  = *_LH / *_RH
  Zone 14      = A_LC / A_RC (central attacking channel)

CLI approval gate
-----------------
  After all three agents produce their draft, the manager is presented the
  full dossier and prompted:
    approve           → write dossier to reports/dossier_<match_id>.md
    reject <feedback> → re-run with feedback injected (max 3 cycles)
    quit              → exit without saving

Usage
-----
  python agents/synthesis.py
  python agents/synthesis.py --ledger data/processed/match_ledger.json
  python agents/synthesis.py --match-id 2026-09-06_tivvy
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

# ── Project root on sys.path ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.zones import ZONES, ZONE_ORDER  # noqa: E402 (after sys.path insert)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROC_DIR    = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
LEDGER_PATH = PROC_DIR / "match_ledger.json"

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_REJECTIONS = 3           # hard cap on rejection cycles before forced approval
_COL           = 72          # terminal rule width

# ── Zone inference tables ──────────────────────────────────────────────────────
# Half-space threshold: ≥ 33% of box entries via CARRY → Qualitative Superiority flag
_QS_HALF_SPACE_THRESHOLD = 0.33

# BOX_ENTRY sub_type → corridor label
_BOX_ENTRY_CORRIDOR: dict[str | None, str] = {
    "CROSS": "Flank",
    "CARRY": "Half-Space",
    "PASS":  "Central (Zone 14)",
    None:    "Central (Zone 14)",
}
# BOX_ENTRY sub_type → half-space flag (used for Qualitative Superiority)
_BOX_ENTRY_IS_HALF_SPACE: dict[str | None, bool] = {
    "CROSS": False,
    "CARRY": True,   # carry through A_LH / A_RH = half-space penetration
    "PASS":  False,
    None:    False,
}

# SET_PIECE sub_type → inferred zone IDs (both lateral variants)
_SET_PIECE_ZONES: dict[str | None, list[str]] = {
    "ATT_CORNER": ["A_LF", "A_RF"],
    "DEF_CORNER": ["D_LF", "D_RF"],
    "FREE_KICK":  ["M_LC", "M_RC"],
    None:         ["A_LC"],
}

# HIGH_REGAIN inferred zone — "high" pressing defined as M/A third
_HIGH_REGAIN_INFERRED_ZONE = "M_LC"

# DEF_TURNOVER inferred zone — loss of possession, defaulted to middle third
_DEF_TURNOVER_INFERRED_ZONE = "M_LC"


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _rule(char: str = "─") -> None:
    print(char * _COL)


def _extract_tags(ledger: dict, event_type: str) -> list[dict]:
    """
    Collect all tag events of a given event_type from both
    matched (ledger['matched'][n]['tag']) and unmatched_tags lists.
    Returns events sorted by match_seconds ascending.
    """
    tags: list[dict] = []
    for match in ledger.get("matched", []):
        tag = match.get("tag", {})
        if tag.get("event_type") == event_type:
            tags.append(tag)
    for tag in ledger.get("unmatched_tags", []):
        if tag.get("event_type") == event_type:
            tags.append(tag)
    return sorted(tags, key=lambda t: t.get("match_seconds", 0))


def _wrap(text: str, indent: int = 2, width: int = 74) -> str:
    return textwrap.fill(
        text, width=width,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
    )


def _safe_pct(numerator: int, denominator: int) -> int:
    return round((numerator / denominator) * 100) if denominator else 0


def _match_id_from_ledger(ledger: dict) -> str:
    """Derive a safe match ID slug from the ledger's generated_at timestamp."""
    raw = ledger.get("generated_at", datetime.now(timezone.utc).isoformat())
    return raw[:10]  # YYYY-MM-DD


# ══════════════════════════════════════════════════════════════════════════════
# Agent 1 — In-Possession Agent
# Moment: IN POSSESSION
# Events: SHOT, BOX_ENTRY
# Formal language: Half-Spaces, Qualitative Superiority, Zone 14, Corridor
# ══════════════════════════════════════════════════════════════════════════════

def run_in_possession_agent(ledger: dict, feedback: str = "") -> str:
    """
    Analyse SHOT and BOX_ENTRY events.

    Returns a formatted markdown section covering:
    - Shot volume and outcome split
    - Box entry corridor analysis (Flank / Half-Space / Central)
    - Qualitative Superiority flag (≥33% half-space carries)
    - Tactical lever recommendations
    """
    shots   = _extract_tags(ledger, "SHOT")
    entries = _extract_tags(ledger, "BOX_ENTRY")

    # Shot outcome counts
    on_target  = sum(1 for s in shots if s.get("sub_type") == "ON_TARGET")
    off_target = sum(1 for s in shots if s.get("sub_type") == "OFF_TARGET")
    blocked    = sum(1 for s in shots if s.get("sub_type") == "BLOCKED")
    shot_count = len(shots)

    # Box entry corridor split
    corridor_counts: dict[str, int] = {"Flank": 0, "Half-Space": 0, "Central (Zone 14)": 0}
    half_space_count = 0

    for e in entries:
        sub = e.get("sub_type")
        corridor = _BOX_ENTRY_CORRIDOR.get(sub, "Central (Zone 14)")
        corridor_counts[corridor] += 1
        if _BOX_ENTRY_IS_HALF_SPACE.get(sub, False):
            half_space_count += 1

    total_entries = len(entries)
    qs_flag = (
        total_entries > 0
        and (half_space_count / total_entries) >= _QS_HALF_SPACE_THRESHOLD
    )
    dominant_corridor = (
        max(corridor_counts, key=corridor_counts.get)
        if total_entries > 0
        else "N/A"
    )

    lines: list[str] = []
    lines.append("## MOMENT 1 — IN POSSESSION")

    if feedback:
        lines.append(f"> ⚙ Manager note: {feedback.strip()}")
        lines.append("")

    lines.append(
        f"**Shots:** {shot_count}  |  "
        f"On Target: {on_target}  |  "
        f"Off Target: {off_target}  |  "
        f"Blocked: {blocked}"
    )
    lines.append(f"**Box Entries:** {total_entries}")
    lines.append("")

    if total_entries > 0:
        lines.append("**Entry Corridor Analysis:**")
        for corridor, n in corridor_counts.items():
            pct = _safe_pct(n, total_entries)
            bar = "█" * (pct // 10)
            lines.append(f"  {corridor:<22}  {n:>2}  ({pct:>3}%)  {bar}")
        lines.append("")
        lines.append(f"**Dominant Entry Corridor:** {dominant_corridor}")
        lines.append("")

        if qs_flag:
            lines.append(
                "**⚡ QUALITATIVE SUPERIORITY — HALF-SPACE EXPLOITATION DETECTED**"
            )
            lines.append(_wrap(
                f"Tiverton generated {half_space_count} of {total_entries} box entries "
                f"({_safe_pct(half_space_count, total_entries)}%) via Carry actions through "
                "the attacking half-spaces (A_LH / A_RH). This represents a positive "
                "indicator of Qualitative Superiority — the team is bypassing the "
                "opposition's defensive block through individual quality in transition "
                "corridors. Tactical lever: sustain half-space rotation between the #10 "
                "and the wide midfielder on the strong side."
            ))
        else:
            lines.append("**Half-Space Exploitation:** Below threshold (<33% via Carry)")
            lines.append(_wrap(
                "Box entries are dominated by flank delivery (CROSS) or direct central "
                "passes (PASS). Qualitative Superiority through the half-spaces "
                "(A_LH / A_RH) is under-utilised. Tactical lever: encourage combination "
                "play to draw the opposition's defensive mid before committing the "
                "half-space runner — particularly on the weak side of a wide press."
            ))
    else:
        lines.append("⚠ No box entries recorded. Check tagger export completeness.")

    lines.append("")
    if shot_count == 0:
        lines.append("⚠ No shots recorded. Verify tagger data before finalising dossier.")
    elif on_target == 0:
        lines.append(
            "⚠ Zero shots on target. In-Possession structure is generating entries "
            "but the final action quality (Zone 14 finish) is the constraining factor."
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Agent 2 — Out-of-Possession / Pressing Agent
# Moment: OUT OF POSSESSION + DEFENSIVE TRANSITION
# Events: HIGH_REGAIN (turnover won), DEF_TURNOVER (turnover lost)
# Formal language: Counter-Pressing Phase, Line of Engagement, Compactness,
#                  Pressing Trigger, Block
# ══════════════════════════════════════════════════════════════════════════════

def run_press_agent(ledger: dict, feedback: str = "") -> str:
    """
    Analyse HIGH_REGAIN and DEF_TURNOVER events.

    Returns a formatted markdown section covering:
    - Counter-Pressing Phase efficiency (high regain rate)
    - Line of Engagement inference from regain zone distribution
    - Compactness score from turnover spread
    - Pressing Trigger recommendation
    """
    regains   = _extract_tags(ledger, "HIGH_REGAIN")
    turnovers = _extract_tags(ledger, "DEF_TURNOVER")

    total_regains   = len(regains)
    total_turnovers = len(turnovers)
    total_events    = total_regains + total_turnovers

    # Counter-Pressing efficiency: regains as % of all possession changes
    cp_efficiency = _safe_pct(total_regains, total_events) if total_events else 0

    # Line of Engagement inference
    # HIGH_REGAIN = pressing win; by definition in the M/A third.
    # DEF_TURNOVER = possession lost; spread across all thirds.
    # We flag LoE as "High Block" if CP efficiency ≥ 50%, else "Mid Block".
    if cp_efficiency >= 60:
        line_of_engagement = "High Block  (LoE > halfway)"
        press_trigger_rec  = (
            "Sustain high press trigger on the opposition GK distribution. "
            "The ball-side winger sets the trigger; the #8 / #10 cuts the "
            "central pass lane to force the long ball."
        )
    elif cp_efficiency >= 35:
        line_of_engagement = "Mid Block  (LoE at halfway)"
        press_trigger_rec  = (
            "Adopt a mid-press trigger on the opposition centre-back receiving "
            "the ball back from the striker. The #9 sets the press direction; "
            "the #10 shadows the pivot. Maintain compactness in the 4-4-2 block."
        )
    else:
        line_of_engagement = "Low Block  (LoE inside own half)"
        press_trigger_rec  = (
            "The pressing structure is not generating regains efficiently. "
            "Consider a passive mid-block and discipline the defensive shape "
            "before committing to a Counter-Pressing Phase trigger. "
            "Rest Defense priority: protect the space behind the last line."
        )

    # Compactness score: lower spread of DEF_TURNOVER events = more compact block.
    # Without x/y we approximate using match_seconds spread (temporal compactness proxy).
    if total_turnovers >= 2:
        seconds = sorted(t.get("match_seconds", 0) for t in turnovers)
        temporal_spread_min = round((seconds[-1] - seconds[0]) / 60, 1)
        if temporal_spread_min <= 15:
            compactness_note = (
                f"DEF_TURNOVER events are concentrated within a {temporal_spread_min}-minute "
                "window — suggests a vulnerable phase rather than systemic compactness issues."
            )
        else:
            compactness_note = (
                f"DEF_TURNOVER events spread across {temporal_spread_min} minutes — "
                "indicates recurring compactness breakdown throughout the match, not an isolated "
                "phase. Structural review of Unit Cohesion in the Block is recommended."
            )
    else:
        compactness_note = "Insufficient DEF_TURNOVER volume for compactness analysis."

    lines: list[str] = []
    lines.append("## MOMENT 2 & 3 — OUT OF POSSESSION / PRESSING")

    if feedback:
        lines.append(f"> ⚙ Manager note: {feedback.strip()}")
        lines.append("")

    lines.append(
        f"**HIGH_REGAIN (Pressing Wins):** {total_regains}  |  "
        f"**DEF_TURNOVER (Losses):** {total_turnovers}  |  "
        f"**Total Possession Changes:** {total_events}"
    )
    lines.append(
        f"**Counter-Pressing Phase Efficiency:** {cp_efficiency}%  "
        f"(regains / total possession changes)"
    )
    lines.append("")
    lines.append(f"**Line of Engagement:** {line_of_engagement}")
    lines.append("")
    lines.append("**Pressing Trigger Recommendation:**")
    lines.append(_wrap(press_trigger_rec))
    lines.append("")
    lines.append("**Compactness Assessment:**")
    lines.append(_wrap(compactness_note))
    lines.append("")

    if total_regains == 0 and total_turnovers == 0:
        lines.append("⚠ No HIGH_REGAIN or DEF_TURNOVER events found. Verify tagger export.")
    elif cp_efficiency < 35 and total_events >= 5:
        lines.append(
            "⚠ HIGH ALERT — Counter-Pressing Phase efficiency is critically low. "
            "The opposition may be exploiting the Defensive Transition phase."
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Agent 3 — Set-Piece Agent
# Moment: IN POSSESSION (attacking) + OUT OF POSSESSION (defending)
# Events: SET_PIECE (sub_type: ATT_CORNER, DEF_CORNER, FREE_KICK)
# Formal language: Rest Defense, Unit Cohesion, Attacking Transition
# ══════════════════════════════════════════════════════════════════════════════

def run_set_piece_agent(ledger: dict, feedback: str = "") -> str:
    """
    Analyse SET_PIECE events by sub_type.

    Returns a formatted markdown section covering:
    - Attacking dead-ball volume and inferred zone (ATT_CORNER → A_LF/A_RF)
    - Defensive dead-ball exposure (DEF_CORNER → D_LF/D_RF)
    - Free kick distribution
    - Rest Defense exposure flag (DEF_CORNER volume as proxy)
    - Unit Cohesion recommendation
    """
    set_pieces = _extract_tags(ledger, "SET_PIECE")

    att_corners  = [e for e in set_pieces if e.get("sub_type") == "ATT_CORNER"]
    def_corners  = [e for e in set_pieces if e.get("sub_type") == "DEF_CORNER"]
    free_kicks   = [e for e in set_pieces if e.get("sub_type") == "FREE_KICK"]
    unclassified = [e for e in set_pieces if e.get("sub_type") not in
                    ("ATT_CORNER", "DEF_CORNER", "FREE_KICK")]

    total_sp      = len(set_pieces)
    att_total     = len(att_corners)
    def_total     = len(def_corners)
    fk_total      = len(free_kicks)

    # Rest Defense exposure: high DEF_CORNER count relative to ATT_CORNER suggests
    # the opposition is pinning Tiverton back in their own half via set pieces.
    rest_defense_flag = (def_total >= 3) or (def_total > att_total and def_total >= 2)

    # Attacking corner inferred zone
    att_zone_label = (
        f"{att_total} attacking corner(s) from A_LF / A_RF delivery corridors"
        if att_total > 0
        else "No attacking corners recorded."
    )
    def_zone_label = (
        f"{def_total} defensive corner(s) — Rest Defense exposure at D_LF / D_RF"
        if def_total > 0
        else "No defensive corners recorded."
    )

    # Unit Cohesion recommendation based on DEF_CORNER volume
    if def_total == 0:
        cohesion_rec = (
            "No defensive corners conceded — Unit Cohesion under dead-ball pressure "
            "was not tested. Maintain zonal marking discipline at corners."
        )
    elif def_total <= 2:
        cohesion_rec = (
            f"{def_total} defensive corner(s) — manageable. Ensure the defensive "
            "unit maintains its Block shape in the post-corner Attacking Transition "
            "phase. The #6 / #5 must hold the back line to prevent transition exposure."
        )
    else:
        cohesion_rec = (
            f"⚠ {def_total} defensive corners conceded. REST DEFENSE is under "
            "sustained pressure. Tactical lever: tighten Unit Cohesion at corners — "
            "consider a mixed zonal/man-mark scheme with the #4 or #6 assigned to the "
            "near-post runner. Ensure the #7 / #11 hold width for the counter-press "
            "trigger on second-ball situations."
        )

    # Free kick note
    fk_note = (
        f"{fk_total} free kick(s) recorded (inferred zone: M_LC / M_RC). "
        "Direct free kick delivery into the attacking third should be coordinated "
        "with a first-post decoy runner to disrupt the opposition's defensive wall shape."
        if fk_total > 0
        else "No free kicks recorded."
    )

    lines: list[str] = []
    lines.append("## SET PIECE ANALYSIS")

    if feedback:
        lines.append(f"> ⚙ Manager note: {feedback.strip()}")
        lines.append("")

    lines.append(
        f"**Total Set Pieces:** {total_sp}  |  "
        f"Att Corners: {att_total}  |  "
        f"Def Corners: {def_total}  |  "
        f"Free Kicks: {fk_total}"
    )
    if unclassified:
        lines.append(f"  *(Unclassified SET_PIECE events: {len(unclassified)} — sub_type missing)*")
    lines.append("")

    lines.append(f"**Attacking Set Pieces:** {att_zone_label}")
    lines.append(f"**Defensive Set Pieces:** {def_zone_label}")
    lines.append("")

    if rest_defense_flag:
        lines.append(
            "**🚨 REST DEFENSE ALERT** — Opposition is generating repeated dead-ball "
            "situations inside Tiverton's defensive third."
        )
        lines.append(_wrap(
            "The frequency of defensive corners elevates Rest Defense risk on "
            "second-ball clearances. The holding midfielder (#4 / #8) must track "
            "the opposition's late runners from the edge of the box and immediately "
            "initiate the Attacking Transition counter-press if the ball is cleared."
        ))
        lines.append("")

    lines.append("**Unit Cohesion Recommendation:**")
    lines.append(_wrap(cohesion_rec))
    lines.append("")
    lines.append("**Free Kick Note:**")
    lines.append(_wrap(fk_note))

    if total_sp == 0:
        lines.append("\n⚠ No SET_PIECE events tagged. This section may be incomplete.")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Dossier assembly
# ══════════════════════════════════════════════════════════════════════════════

def _build_dossier(
    ledger:   dict,
    feedback: str = "",
) -> str:
    """Run all three agents and concatenate their output into a full dossier."""
    match_id  = _match_id_from_ledger(ledger)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    header = "\n".join([
        "# TIVERTON TOWN FC — TACTICAL ANALYSIS DOSSIER",
        f"**Match Date:** {match_id}",
        f"**Generated:**  {generated}",
        f"**Standard:**   UEFA Pro Licence | 4 Moments of the Game",
        "",
        "---",
        "",
    ])

    section_ip  = run_in_possession_agent(ledger, feedback)
    section_pr  = run_press_agent(ledger, feedback)
    section_sp  = run_set_piece_agent(ledger, feedback)

    summary_stats = ledger.get("summary", {})
    footer = "\n".join([
        "",
        "---",
        "",
        "## DATA QUALITY",
        f"Matched events: {summary_stats.get('matched', '?')}  |  "
        f"Unmatched tags: {summary_stats.get('unmatched_tags', '?')}  |  "
        f"Unmatched club events: {summary_stats.get('unmatched_club_events', '?')}",
        "",
        "_Dossier produced by PitchPulse · Tiverton Town FC · Zero-Budget Performance Stack_",
    ])

    return header + section_ip + "\n\n---\n\n" + section_pr + "\n\n---\n\n" + section_sp + footer


def _write_dossier(content: str, match_id: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = match_id.replace(":", "-").replace(" ", "_")
    out  = REPORTS_DIR / f"dossier_{slug}.md"
    out.write_text(content, encoding="utf-8")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Human-in-the-Loop CLI approval gate
# ══════════════════════════════════════════════════════════════════════════════

def run_approval_gate(ledger: dict) -> bool:
    """
    Present the draft dossier to the manager and loop until:
      - 'approve'           → write dossier to disk, return True
      - 'reject <feedback>' → re-run agents with feedback, repeat (max MAX_REJECTIONS)
      - 'quit'              → exit without saving, return False

    Returns True if the dossier was saved, False otherwise.
    """
    feedback    = ""
    rejections  = 0
    match_id    = _match_id_from_ledger(ledger)

    while True:
        dossier = _build_dossier(ledger, feedback)

        print()
        _rule("═")
        print("  PITCHPULSE — TACTICAL ANALYSIS DOSSIER (DRAFT)")
        _rule("═")
        print()
        print(dossier)
        print()
        _rule()

        if rejections >= MAX_REJECTIONS:
            print(
                f"  [GATE] Maximum rejection cycles ({MAX_REJECTIONS}) reached. "
                "Forcing approval and writing dossier."
            )
            _rule()
            out = _write_dossier(dossier, match_id)
            print(f"  [GATE] Dossier written → {out.relative_to(ROOT)}")
            _rule("═")
            return True

        cycle_label = (
            f"  Cycle {rejections + 1} / {MAX_REJECTIONS} max rejections."
            if rejections > 0 else ""
        )
        if cycle_label:
            print(cycle_label)

        print("  Commands:")
        print("    approve              → accept and write dossier to disk")
        print("    reject <feedback>    → re-run agents with your feedback injected")
        print("    quit                 → discard draft and exit")
        _rule()

        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [GATE] Interrupted — exiting without saving.")
            return False

        if not raw:
            continue

        cmd  = raw.split(None, 1)
        verb = cmd[0].lower()

        if verb == "approve":
            out = _write_dossier(dossier, match_id)
            print()
            _rule("═")
            print(f"  [GATE] Dossier approved and written → {out.relative_to(ROOT)}")
            _rule("═")
            return True

        elif verb == "reject":
            rejections += 1
            feedback = cmd[1].strip() if len(cmd) > 1 else ""
            if not feedback:
                print(
                    "  [GATE] No feedback provided. Re-running agents without constraints."
                )
            else:
                print(f"  [GATE] Feedback injected: \"{feedback}\"")
            print(f"  [GATE] Re-running agents (cycle {rejections})…")
            continue

        elif verb == "quit":
            print("  [GATE] Draft discarded. No dossier written.")
            return False

        else:
            print(f"  [GATE] Unrecognised command: '{raw}'. Type approve, reject, or quit.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PitchPulse tactical analysis agents — Tiverton Town FC"
    )
    parser.add_argument(
        "--ledger",
        default=str(LEDGER_PATH),
        help=f"Path to match_ledger.json (default: {LEDGER_PATH.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--match-id",
        default=None,
        help="Optional match ID slug for the dossier filename (default: derived from ledger timestamp)",
    )
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    if not ledger_path.exists():
        print(
            f"\n  [ERROR] Ledger not found: {ledger_path}\n"
            "  Run reconcile/sync.py first to generate the match ledger.\n"
        )
        sys.exit(1)

    print()
    _rule("═")
    print("  PITCHPULSE — MULTI-AGENT TACTICAL ANALYSIS ENGINE")
    print("  Tiverton Town FC | UEFA Pro Licence Standard")
    _rule("═")
    print(f"\n  Loading ledger: {ledger_path.relative_to(ROOT)}")

    with open(ledger_path, encoding="utf-8") as f:
        ledger = json.load(f)

    if args.match_id:
        ledger["_match_id_override"] = args.match_id

    summary = ledger.get("summary", {})
    print(
        f"  Events — Matched: {summary.get('matched', '?')}  |  "
        f"Unmatched tags: {summary.get('unmatched_tags', '?')}  |  "
        f"Unmatched club: {summary.get('unmatched_club_events', '?')}"
    )
    print("\n  Running three specialist agents…\n")
    print("    Agent 1: In-Possession Analyst     (SHOT, BOX_ENTRY)")
    print("    Agent 2: Out-of-Possession / Press  (HIGH_REGAIN, DEF_TURNOVER)")
    print("    Agent 3: Set-Piece Analyst          (SET_PIECE)")
    print()

    saved = run_approval_gate(ledger)
    sys.exit(0 if saved else 1)


if __name__ == "__main__":
    main()
