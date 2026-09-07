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

from cv.zones import ZONES, ZONE_ORDER, zones_for_third  # noqa: E402 (after sys.path insert)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROC_DIR      = ROOT / "data" / "processed"
REPORTS_DIR   = ROOT / "reports"
LEDGER_PATH   = PROC_DIR / "match_ledger.json"
CONTEXT_PATH  = ROOT / "data" / "raw" / "match_context.json"

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_REJECTIONS = 3           # hard cap on rejection cycles before forced approval
_COL           = 72          # terminal rule width

# ── Spatial constants — real zone IDs, no sub-type inference ──────────────────
# Half-space zones (attacking + middle third)
_HALF_SPACE_ZONES  = {"A_LH", "A_RH", "M_LH", "M_RH"}
# Zone 14: central attacking channel
_ZONE_14           = {"A_LC", "A_RC"}
# Attacking third
_ATTACKING_THIRD   = {z for z in ZONES if z.startswith("A_")}
# Middle + attacking = credible "high" pressing zone
_HIGH_PRESS_ZONES  = {z for z in ZONES if z.startswith("A_") or z.startswith("M_")}
# Defensive third
_DEFENSIVE_THIRD   = {z for z in ZONES if z.startswith("D_")}
# Flank channels
_FLANK_CHANNELS    = {"LF", "RF"}

# Qualitative Superiority: ≥ 30% of spatially-confirmed box entries in half-space zones.
# This is now derived from real zone_id, not sub-type.
_QS_HALF_SPACE_THRESHOLD = 0.30


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
# Match context helpers — load, score timeline, player resolution
# ══════════════════════════════════════════════════════════════════════════════

def _load_match_context() -> dict:
    """
    Load data/raw/match_context.json produced by data/parse_report.py.
    Returns an empty dict with a diagnostic notice if the file is absent.
    """
    if not CONTEXT_PATH.exists():
        print(
            f"  [CTX] match_context.json not found at {CONTEXT_PATH.relative_to(ROOT)}\n"
            "  [CTX] Run: python data/parse_report.py <report.docx>\n"
            "  [CTX] Falling back to ledger-only analysis (no squad/score-state context)."
        )
        return {}
    with open(CONTEXT_PATH, encoding="utf-8") as f:
        ctx = json.load(f)
    opponent = ctx.get("opponent", "Unknown")
    result   = ctx.get("result", "?")
    score    = ctx.get("score", {})
    print(
        f"  [CTX] Loaded match context: {result} | {opponent} "
        f"{score.get('opponent','?')}–{score.get('tiverton','?')} Tiverton"
    )
    return ctx


def _build_score_timeline(ctx: dict) -> list:
    """
    Build a sorted list of score-state checkpoints from ctx['scorers'].
    Each entry: {"minute_second": int, "tiverton": int, "opponent": int}
    Timeline starts implicitly at 0-0.
    """
    events = []
    scorers = ctx.get("scorers", {})
    for s in scorers.get("tiverton", []):
        events.append({"minute_second": s["minute"] * 60, "team": "tiverton"})
    for s in scorers.get("opponent", []):
        events.append({"minute_second": s["minute"] * 60, "team": "opponent"})
    events.sort(key=lambda e: e["minute_second"])

    timeline = []
    tiv, opp = 0, 0
    for e in events:
        if e["team"] == "tiverton":
            tiv += 1
        else:
            opp += 1
        timeline.append({"minute_second": e["minute_second"], "tiverton": tiv, "opponent": opp})
    return timeline


def _score_at_second(timeline: list, match_seconds: int) -> dict:
    """Return the live score at a given match_seconds from the score timeline."""
    tiv, opp = 0, 0
    for checkpoint in timeline:
        if checkpoint["minute_second"] <= match_seconds:
            tiv = checkpoint["tiverton"]
            opp = checkpoint["opponent"]
        else:
            break
    return {"tiverton": tiv, "opponent": opp}


def _format_lineup_header(ctx: dict) -> str:
    """Format a one-line XI string from match_context lineup."""
    lineup = ctx.get("lineup", [])
    if not lineup:
        return "Lineup data unavailable — run parse_report.py"
    parts = []
    for p in lineup:
        name = p["player"]
        ann  = []
        if p.get("role") == "GK":
            ann.append("GK")
        if p.get("captain"):
            ann.append("C")
        if p.get("subbed_off"):
            ann.append(f"off {p['subbed_off']}'")
        parts.append(f"{name}({', '.join(ann)})" if ann else name)
    subs = [s for s in ctx.get("subs", []) if s.get("used")]
    subs_str = "  ·  Subs: " + " · ".join(
        f"{s['player']}({s['minute']}')" for s in subs
    ) if subs else ""
    return " · ".join(parts) + subs_str


# ══════════════════════════════════════════════════════════════════════════════
# Agent 1 — In-Possession Agent
# Moment: IN POSSESSION
# Events: SHOT, BOX_ENTRY
# Formal language: Half-Spaces, Qualitative Superiority, Zone 14, Corridor
# ══════════════════════════════════════════════════════════════════════════════

def run_in_possession_agent(ledger: dict, feedback: str = "", ctx: dict | None = None) -> str:
    """
    Analyse SHOT and BOX_ENTRY events using real zone_id from pitch tap coordinates.

    Zone classification is derived from zone_id stamped by reconcile/sync.py
    via cv.zones.get_zone_by_coords(x_m, y_m). No sub-type inference is used.

    Returns a formatted markdown section covering:
    - Shot volume, outcome split, and Zone 14 shot proportion
    - Box entry corridor analysis by actual zone (Flank / Half-Space / Central)
    - Qualitative Superiority flag (≥30% confirmed half-space entries)
    - Spatial coverage note when zone data is missing (legacy v1 exports)
    """
    shots   = _extract_tags(ledger, "SHOT")
    entries = _extract_tags(ledger, "BOX_ENTRY")

    # Shot outcomes
    on_target  = sum(1 for s in shots if s.get("sub_type") == "ON_TARGET")
    off_target = sum(1 for s in shots if s.get("sub_type") == "OFF_TARGET")
    blocked    = sum(1 for s in shots if s.get("sub_type") == "BLOCKED")
    shot_count = len(shots)

    # Zone 14 shots: zone_id in {A_LC, A_RC}
    zone14_shots = sum(1 for s in shots if s.get("zone_id") in _ZONE_14)

    # Box entry corridor split — from real zone_id
    corridor_counts: dict[str, int] = {"Flank": 0, "Half-Space": 0, "Central (Zone 14)": 0, "Unknown": 0}
    half_space_count = 0
    zoned_entries    = 0

    for e in entries:
        zid = e.get("zone_id")
        if zid is None:
            corridor_counts["Unknown"] += 1
            continue
        zoned_entries += 1
        channel = ZONES[zid]["channel"] if zid in ZONES else ""
        if channel in _FLANK_CHANNELS:
            corridor_counts["Flank"] += 1
        elif zid in _HALF_SPACE_ZONES:
            corridor_counts["Half-Space"] += 1
            half_space_count += 1
        else:
            corridor_counts["Central (Zone 14)"] += 1

    total_entries  = len(entries)
    qs_flag = (
        zoned_entries > 0
        and (half_space_count / zoned_entries) >= _QS_HALF_SPACE_THRESHOLD
    )
    dominant_corridor = (
        max((k for k in corridor_counts if k != "Unknown"), key=lambda k: corridor_counts[k])
        if zoned_entries > 0
        else "Insufficient spatial data"
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
        f"Blocked: {blocked}  |  "
        f"Zone 14: {zone14_shots}"
    )
    lines.append(f"**Box Entries:** {total_entries}  (spatially confirmed: {zoned_entries})")
    lines.append("")

    if corridor_counts["Unknown"] > 0:
        lines.append(
            f"⚠ {corridor_counts['Unknown']} entry event(s) have no zone data "
            "(legacy v1 export — no pitch tap recorded). Spatial analysis below "
            "reflects confirmed events only."
        )
        lines.append("")

    if zoned_entries > 0:
        lines.append("**Entry Corridor Analysis (real pitch coordinates):**")
        for corridor in ("Flank", "Half-Space", "Central (Zone 14)"):
            n   = corridor_counts[corridor]
            pct = _safe_pct(n, zoned_entries)
            bar = "█" * (pct // 10)
            lines.append(f"  {corridor:<22}  {n:>2}  ({pct:>3}%)  {bar}")
        lines.append("")
        lines.append(f"**Dominant Entry Corridor:** {dominant_corridor}")
        lines.append("")

        if qs_flag:
            lines.append(
                "**⚡ QUALITATIVE SUPERIORITY — HALF-SPACE EXPLOITATION CONFIRMED**"
            )
            lines.append(_wrap(
                f"Tiverton achieved {half_space_count} of {zoned_entries} spatially-confirmed "
                f"box entries ({_safe_pct(half_space_count, zoned_entries)}%) in the attacking "
                "half-spaces (A_LH / A_RH) — confirmed from real pitch coordinates, not "
                "sub-type inference. This is genuine Qualitative Superiority: individual "
                "actions bypassing the opposition's defensive block in the most dangerous "
                "pre-box corridors. Tactical lever: sustain half-space rotation between "
                "the #10 and the strong-side wide midfielder."
            ))
        else:
            pct_hs = _safe_pct(half_space_count, zoned_entries) if zoned_entries else 0
            lines.append(f"**Half-Space Exploitation:** {pct_hs}% (threshold: {int(_QS_HALF_SPACE_THRESHOLD*100)}%)")
            lines.append(_wrap(
                "Box entries are concentrated on flanks or through the central channel. "
                "Qualitative Superiority through A_LH / A_RH is under-exploited. "
                "Tactical lever: draw the opposition defensive mid with a central "
                "dummy run before releasing the half-space runner, particularly on "
                "the weak side when the opposition is overloading the ball."
            ))
    else:
        lines.append("⚠ No spatially-confirmed box entries. All events lack zone data.")

    lines.append("")
    if shot_count == 0:
        lines.append("⚠ No shots recorded. Verify tagger export completeness.")
    elif on_target == 0:
        lines.append(
            "⚠ Zero shots on target — Zone 14 penetration is occurring but "
            "the final-action quality is the constraining factor."
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Agent 2 — Out-of-Possession / Pressing Agent
# Moment: OUT OF POSSESSION + DEFENSIVE TRANSITION
# Events: HIGH_REGAIN (turnover won), DEF_TURNOVER (turnover lost)
# Formal language: Counter-Pressing Phase, Line of Engagement, Compactness,
#                  Pressing Trigger, Block
# ══════════════════════════════════════════════════════════════════════════════

def run_press_agent(ledger: dict, feedback: str = "", ctx: dict | None = None) -> str:
    """
    Analyse HIGH_REGAIN and DEF_TURNOVER events using real zone_id.

    Line of Engagement is now derived from the spatial distribution of
    HIGH_REGAIN events — where on the pitch Tiverton actually wins the ball —
    rather than from a volumetric efficiency ratio (which is a different metric).

    Returns a formatted markdown section covering:
    - Counter-Pressing Phase efficiency (regains / total possession changes)
    - Line of Engagement: median third of HIGH_REGAIN events (spatial)
    - Compactness: spread of DEF_TURNOVER events across channels (spatial)
    - Pressing Trigger recommendation anchored to real zone data
    """
    regains   = _extract_tags(ledger, "HIGH_REGAIN")
    turnovers = _extract_tags(ledger, "DEF_TURNOVER")

    total_regains   = len(regains)
    total_turnovers = len(turnovers)
    total_events    = total_regains + total_turnovers

    # Counter-Pressing Phase efficiency (volume metric, not LoE)
    cp_efficiency = _safe_pct(total_regains, total_events) if total_events else 0

    # ── Line of Engagement — derived from spatial distribution of HIGH_REGAIN ──
    # Where Tiverton wins the ball tells us where the press is active.
    regain_thirds: dict[str, int] = {"A": 0, "M": 0, "D": 0, "unknown": 0}
    zoned_regains = 0
    for r in regains:
        zid = r.get("zone_id")
        if zid and zid in ZONES:
            regain_thirds[ZONES[zid]["third"]] += 1
            zoned_regains += 1
        else:
            regain_thirds["unknown"] += 1

    # LoE = third where the majority of regains occur
    if zoned_regains > 0:
        dominant_regain_third = max(("A", "M", "D"), key=lambda t: regain_thirds[t])
        loe_map = {
            "A": "High Block  (LoE in opposition's Attacking Third — A_* zones)",
            "M": "Mid Block   (LoE at Halfway — M_* zones)",
            "D": "Low Block   (LoE in Defensive Third — D_* zones: reactive)",
        }
        line_of_engagement = loe_map[dominant_regain_third]
        high_pct = _safe_pct(regain_thirds["A"] + regain_thirds["M"], zoned_regains)
    else:
        line_of_engagement  = "Insufficient spatial data — no zone_id on HIGH_REGAIN events"
        dominant_regain_third = None
        high_pct = 0

    # Pressing Trigger recommendation based on confirmed LoE
    if dominant_regain_third == "A":
        press_trigger_rec = (
            "High press confirmed in the A_* zones. Trigger: ball to opposition "
            "CB under back-pass pressure. Ball-side winger sets the trigger; "
            "#8 / #10 cuts the central pass lane to force the long ball over the top."
        )
    elif dominant_regain_third == "M":
        press_trigger_rec = (
            "Mid-block press confirmed at halfway. Trigger: opposition pivot "
            "receiving to feet. The #9 channels the press direction; the #10 "
            "shadows the pivot. Hold 4-4-2 Block compactness — do not over-commit."
        )
    elif dominant_regain_third == "D":
        press_trigger_rec = (
            "Regains concentrated in the Defensive Third — the press is disorganised "
            "or not being executed. The Block is reactive rather than proactive. "
            "Immediate lever: establish a clear press trigger and compact the mid-block "
            "to prevent the opposition playing through the M_* zones freely."
        )
    else:
        press_trigger_rec = (
            "Spatial data required for a precision Pressing Trigger recommendation. "
            "Ensure tagger exports use the v2 pitch tap (x_m, y_m populated)."
        )

    # ── Compactness — spatial spread of DEF_TURNOVER across channels ───────────
    to_channels: set[str] = set()
    to_thirds:   dict[str, int] = {"A": 0, "M": 0, "D": 0}
    zoned_to = 0
    for t in turnovers:
        zid = t.get("zone_id")
        if zid and zid in ZONES:
            to_channels.add(ZONES[zid]["channel"])
            to_thirds[ZONES[zid]["third"]] += 1
            zoned_to += 1

    if zoned_to >= 2:
        channel_spread = len(to_channels)
        if channel_spread <= 2:
            compactness_note = (
                f"DEF_TURNOVER events cluster in {channel_spread} channel(s) "
                f"({', '.join(sorted(to_channels))}) — block is spatially compact."
            )
        elif channel_spread <= 4:
            compactness_note = (
                f"DEF_TURNOVER events spread across {channel_spread} channels "
                f"({', '.join(sorted(to_channels))}) — moderate width in the Block. "
                "Monitor for exploitation of the wide channels in the next phase."
            )
        else:
            compactness_note = (
                f"DEF_TURNOVER events distributed across all {channel_spread} channels "
                f"({', '.join(sorted(to_channels))}) — the Block is not compact. "
                "Unit Cohesion is breaking down laterally; tighten horizontal "
                "distances between the defensive and midfield lines."
            )
    elif zoned_to == 1:
        compactness_note = "Single spatially-confirmed DEF_TURNOVER — insufficient for compactness analysis."
    else:
        compactness_note = "No spatially-confirmed DEF_TURNOVER events — compactness analysis unavailable."

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
    if zoned_regains > 0:
        lines.append(
            f"**Regain Zone Distribution:** "
            f"A_*={regain_thirds['A']}  M_*={regain_thirds['M']}  D_*={regain_thirds['D']}  "
            f"(High-press rate: {high_pct}%)"
        )
    lines.append("")
    lines.append(f"**Line of Engagement (spatial):** {line_of_engagement}")
    lines.append("")
    lines.append("**Pressing Trigger Recommendation:**")
    lines.append(_wrap(press_trigger_rec))
    lines.append("")
    lines.append("**Compactness Assessment (channel spread):**")
    lines.append(_wrap(compactness_note))
    lines.append("")

    if total_regains == 0 and total_turnovers == 0:
        lines.append("⚠ No HIGH_REGAIN or DEF_TURNOVER events found. Verify tagger export.")
    elif cp_efficiency < 35 and total_events >= 5:
        lines.append(
            "⚠ HIGH ALERT — Counter-Pressing Phase efficiency critically low. "
            "The opposition is exploiting the Defensive Transition phase."
        )

    # ── Score-state layer (requires match_context) ─────────────────────────────
    if ctx:
        timeline = _build_score_timeline(ctx)
        if timeline:
            lines.append("")
            lines.append("**Score-State Defensive Analysis:**")

            # Split events into leading / level / trailing phases
            leading_regains   = 0
            leading_turnovers = 0
            other_regains     = 0
            other_turnovers   = 0

            for r in regains:
                sc = _score_at_second(timeline, r.get("match_seconds", 0))
                if sc["tiverton"] > sc["opponent"]:
                    leading_regains += 1
                else:
                    other_regains += 1
            for t in turnovers:
                sc = _score_at_second(timeline, t.get("match_seconds", 0))
                if sc["tiverton"] > sc["opponent"]:
                    leading_turnovers += 1
                else:
                    other_turnovers += 1

            score = ctx.get("score", {})
            tiv_goals = score.get("tiverton", 0)
            opp_goals = score.get("opponent", 0)
            opponent  = ctx.get("opponent", "Opposition")

            # Narrate score phases from timeline
            phase_desc = []
            prev_sec = 0
            tiv_r, opp_r = 0, 0
            for cp in timeline:
                mins_from = prev_sec // 60
                mins_to   = cp["minute_second"] // 60
                tiv_r, opp_r = cp["tiverton"], cp["opponent"]
                phase_desc.append(f"{mins_from}'–{mins_to}' ({tiv_r-( 1 if cp['tiverton']>tiv_r else 0)}-{opp_r-(1 if cp['opponent']>opp_r else 0)}→{tiv_r}-{opp_r})")
                prev_sec = cp["minute_second"]

            lines.append(
                f"  Final: Tiverton {tiv_goals}–{opp_goals} {opponent}"
            )

            if leading_regains + leading_turnovers > 0:
                lead_total = leading_regains + leading_turnovers
                lead_eff   = _safe_pct(leading_regains, lead_total) if lead_total else 0
                lines.append(
                    f"  Whilst leading — Regains: {leading_regains}  |  Turnovers: {leading_turnovers}  "
                    f"|  Efficiency: {lead_eff}%"
                )
                if leading_turnovers > leading_regains:
                    lines.append(_wrap(
                        "⚠ More possession losses than regains whilst protecting a lead. "
                        "The Block is not holding compactness under reduced pressing intensity. "
                        "Tactical lever: drop the Line of Engagement 5–8m when leading; "
                        "prioritise Rest Defense shape over counter-pressing triggers."
                    ))
                else:
                    lines.append(_wrap(
                        "Pressing efficiency held whilst leading — the team maintained "
                        "Counter-Pressing Phase discipline without over-committing. "
                        "Sustain this balance: press the trigger, recover the Block quickly."
                    ))

            if other_regains + other_turnovers > 0:
                other_total = other_regains + other_turnovers
                other_eff   = _safe_pct(other_regains, other_total) if other_total else 0
                lines.append(
                    f"  Level/trailing phase — Regains: {other_regains}  |  Turnovers: {other_turnovers}  "
                    f"|  Efficiency: {other_eff}%"
                )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Agent 3 — Set-Piece Agent
# Moment: IN POSSESSION (attacking) + OUT OF POSSESSION (defending)
# Events: SET_PIECE (sub_type: ATT_CORNER, DEF_CORNER, FREE_KICK)
# Formal language: Rest Defense, Unit Cohesion, Attacking Transition
# ══════════════════════════════════════════════════════════════════════════════

def run_set_piece_agent(ledger: dict, feedback: str = "", ctx: dict | None = None) -> str:
    """
    Analyse SET_PIECE events using real zone_id from pitch tap coordinates.

    Zone data now replaces the previous inferred zone lookup table.
    DEF_CORNER events in defensive zones are flagged as Rest Defense exposure.
    ATT_CORNER delivery zones inform delivery corridor recommendations.

    Returns a formatted markdown section covering:
    - Attacking dead-ball zone distribution (real coordinates)
    - Defensive dead-ball exposure by zone
    - Rest Defense flag (defensive corners in D_LF / D_RF / D_LC / D_RC)
    - Unit Cohesion recommendation
    """
    set_pieces   = _extract_tags(ledger, "SET_PIECE")

    att_corners  = [e for e in set_pieces if e.get("sub_type") == "ATT_CORNER"]
    def_corners  = [e for e in set_pieces if e.get("sub_type") == "DEF_CORNER"]
    free_kicks   = [e for e in set_pieces if e.get("sub_type") == "FREE_KICK"]
    unclassified = [e for e in set_pieces if e.get("sub_type") not in
                    ("ATT_CORNER", "DEF_CORNER", "FREE_KICK")]

    total_sp = len(set_pieces)
    att_total = len(att_corners)
    def_total = len(def_corners)
    fk_total  = len(free_kicks)

    # Attacking corner zones from real coordinates
    att_zones = [e.get("zone_id") for e in att_corners if e.get("zone_id")]
    att_zone_str = (
        f"{att_total} attacking corner(s)  —  "
        f"zones: {', '.join(att_zones) if att_zones else 'no spatial data'}"
        if att_total > 0 else "No attacking corners recorded."
    )

    # Defensive corner zones — Rest Defense flag if in D_* zones
    def_zones = [e.get("zone_id") for e in def_corners if e.get("zone_id")]
    def_in_d_third = [z for z in def_zones if z and z.startswith("D_")]
    rest_defense_flag = (def_total >= 3) or (len(def_in_d_third) >= 2)

    def_zone_str = (
        f"{def_total} defensive corner(s)  —  "
        f"zones: {', '.join(def_zones) if def_zones else 'no spatial data'}"
        if def_total > 0 else "No defensive corners recorded."
    )

    # Free kick zones
    fk_zones = [e.get("zone_id") for e in free_kicks if e.get("zone_id")]
    fk_note = (
        f"{fk_total} free kick(s)  —  zones: {', '.join(fk_zones) if fk_zones else 'no spatial data'}. "
        "Coordinate free kick delivery with a first-post decoy run to disrupt the defensive wall."
        if fk_total > 0 else "No free kicks recorded."
    )

    # Unit Cohesion recommendation
    if def_total == 0:
        cohesion_rec = (
            "No defensive corners — Unit Cohesion under dead-ball pressure was not "
            "tested this match. Maintain zonal marking discipline in training."
        )
    elif def_total <= 2:
        cohesion_rec = (
            f"{def_total} defensive corner(s) — manageable. Ensure the Block "
            "holds shape in the post-corner Attacking Transition phase. The #5 / #6 "
            "must anchor the back line to prevent late-runner exploitation."
        )
    else:
        cohesion_rec = (
            f"⚠ {def_total} defensive corners conceded. REST DEFENSE under sustained "
            "pressure. Tactical lever: assign the #4 or #6 to the near-post runner in "
            "a hybrid zonal/man-mark scheme. The #7 / #11 must hold width for the "
            "second-ball counter-press trigger on clearances."
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
        lines.append(f"  *(Unclassified: {len(unclassified)} — sub_type missing)*")
    lines.append("")

    lines.append(f"**Attacking Set Pieces:** {att_zone_str}")
    lines.append(f"**Defensive Set Pieces:** {def_zone_str}")
    lines.append("")

    if rest_defense_flag:
        lines.append(
            "**🚨 REST DEFENSE ALERT** — Repeated defensive corners in the D_* "
            "zones. Second-ball clearances are elevating counter-attack exposure."
        )
        lines.append(_wrap(
            "The holding midfielder (#4 / #8) must track late opposition runners "
            "from the box edge and trigger the Attacking Transition counter-press "
            "immediately on clearance. Do not allow the opposition to recycle."
        ))
        lines.append("")

    lines.append("**Unit Cohesion Recommendation:**")
    lines.append(_wrap(cohesion_rec))
    lines.append("")
    lines.append("**Free Kick Note:**")
    lines.append(_wrap(fk_note))

    if total_sp == 0:
        lines.append("\n⚠ No SET_PIECE events tagged. This section may be incomplete.")

    # ── Opposition & competition context ───────────────────────────────────────
    if ctx:
        opponent    = ctx.get("opponent", "")
        competition = ctx.get("competition", "")
        keywords    = ctx.get("tactical_keywords", [])
        if opponent or competition:
            lines.append("")
            lines.append("**Match Context:**")
            if opponent and competition:
                lines.append(_wrap(
                    f"Set piece exposure assessed in the context of {competition} "
                    f"against {opponent}. "
                    + ("Direct play and aerial delivery are elevated in this competition; "
                       "anticipate increased dead-ball volume in the opposition half "
                       "and prioritise second-ball positioning on all deliveries."
                       if any(k in keywords for k in ["direct", "aerial", "cross"])
                       else "Standard set piece preparation applies for this fixture.")
                ))

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Agent 4 — Non-League Physics Agent
# Moment: ALL FOUR (aerial duels + second balls underpin every moment)
# Events: AERIAL_DUEL (WON/LOST), SECOND_BALL (WON/LOST)
# Formal language: Second Ball, Direct Play, Quantitative Superiority
# ══════════════════════════════════════════════════════════════════════════════

def run_nonleague_agent(ledger: dict, feedback: str = "", ctx: dict | None = None) -> str:
    """
    Analyse AERIAL_DUEL and SECOND_BALL events — the dominant possession-transition
    mechanism at non-league level. These events are absent from most analysis
    frameworks, which are calibrated for technical possession football.

    Returns a formatted markdown section covering:
    - Aerial duel win rate overall and by zone third (A / M / D)
    - Conditional tactical levers based on win rate and dominant third
    - Second ball recovery rate with dominated / contested / at-risk thresholds
    - Zone-specific coaching recommendations for each scenario

    NOTE: Sequence chaining (aerial-to-second-ball transition velocity) is a
    planned future feature and is NOT implemented in this version.
    """
    aerials     = _extract_tags(ledger, "AERIAL_DUEL")
    secondballs = _extract_tags(ledger, "SECOND_BALL")

    aerial_won  = [e for e in aerials     if e.get("sub_type") == "WON"]
    aerial_lost = [e for e in aerials     if e.get("sub_type") == "LOST"]
    sb_won      = [e for e in secondballs if e.get("sub_type") == "WON"]
    sb_lost     = [e for e in secondballs if e.get("sub_type") == "LOST"]

    total_aerial = len(aerials)
    total_sb     = len(secondballs)
    aerial_wr    = _safe_pct(len(aerial_won), total_aerial) if total_aerial else None
    sb_wr        = _safe_pct(len(sb_won),     total_sb)     if total_sb     else None

    # Aerial zone distribution (where the physical battle is being fought)
    aerial_thirds: dict[str, int] = {"A": 0, "M": 0, "D": 0}
    for e in aerials:
        zid = e.get("zone_id")
        if zid and zid in ZONES:
            aerial_thirds[ZONES[zid]["third"]] += 1

    zoned_aerials = sum(aerial_thirds.values())
    dominant_aerial_third = (
        max(aerial_thirds, key=aerial_thirds.get)
        if zoned_aerials > 0 and max(aerial_thirds.values()) > 0
        else None
    )

    # Second ball zone distribution
    sb_thirds: dict[str, int] = {"A": 0, "M": 0, "D": 0}
    for e in secondballs:
        zid = e.get("zone_id")
        if zid and zid in ZONES:
            sb_thirds[ZONES[zid]["third"]] += 1

    lines: list[str] = []
    lines.append("## NON-LEAGUE PHYSICS — AERIAL DUELS & SECOND BALLS")

    if feedback:
        lines.append(f"> ⚙ Manager note: {feedback.strip()}")
        lines.append("")

    # Aerial headline
    if total_aerial > 0:
        lines.append(
            f"**Aerial Duels:** {total_aerial} total  |  "
            f"Won: {len(aerial_won)}  |  Lost: {len(aerial_lost)}  |  "
            f"Win Rate: {aerial_wr}%"
        )
        if zoned_aerials > 0:
            lines.append(
                f"**Aerial Zone Distribution:** "
                f"A_*={aerial_thirds['A']}  M_*={aerial_thirds['M']}  D_*={aerial_thirds['D']}"
            )
        if dominant_aerial_third:
            third_name = {"A": "Attacking Third", "M": "Middle Third", "D": "Defensive Third"}
            lines.append(
                f"**Primary Aerial Battle Zone:** {third_name[dominant_aerial_third]}"
            )
            lines.append("")
            if dominant_aerial_third == "D" and aerial_wr is not None and aerial_wr < 50:
                lines.append(_wrap(
                    "⚠ Aerial battles concentrated in the Defensive Third with a sub-50% "
                    "win rate — Tiverton are conceding aerial Quantitative Superiority in "
                    "the most dangerous zone. Direct balls over the back line are a live "
                    "threat. Tactical lever: drop the defensive line 5m to contest second "
                    "balls in the D_LC / D_RC corridors, not on the penalty spot."
                ))
            elif dominant_aerial_third == "M" and aerial_wr is not None and aerial_wr >= 55:
                lines.append(_wrap(
                    "Aerial dominance confirmed in the Middle Third. Tiverton are winning "
                    "the physical battle at halfway — a strong platform for direct "
                    "Attacking Transition. Task the #10 to exploit space behind the "
                    "aerial contest immediately on second-ball recovery."
                ))
            elif dominant_aerial_third == "A":
                lines.append(_wrap(
                    f"Aerial contests concentrated in the Attacking Third. "
                    f"Win rate {aerial_wr}% — "
                    + ("attacking set piece delivery and long balls over the top are "
                       "creating Quantitative Superiority in the box." if aerial_wr and aerial_wr >= 50
                       else "the aerial delivery quality into the box is not generating "
                            "first-contact wins. Review set piece delivery height and timing.")
                ))
    else:
        lines.append("**Aerial Duels:** Not tagged this match.")
        lines.append(_wrap(
            "⚠ AERIAL_DUEL is a critical event type at non-league level. "
            "Ensure the v2 tagger is used and aerial contests are captured — "
            "this data drives the second-ball and direct-play analysis."
        ))

    lines.append("")

    # Second ball headline
    if total_sb > 0:
        lines.append(
            f"**Second Balls:** {total_sb} total  |  "
            f"Won: {len(sb_won)}  |  Lost: {len(sb_lost)}  |  "
            f"Recovery Rate: {sb_wr}%"
        )
        if sum(sb_thirds.values()) > 0:
            lines.append(
                f"**Second Ball Zones:** "
                f"A_*={sb_thirds['A']}  M_*={sb_thirds['M']}  D_*={sb_thirds['D']}"
            )
        lines.append("")

        if sb_wr is not None:
            if sb_wr >= 60:
                lines.append(_wrap(
                    f"Second ball recovery rate of {sb_wr}% — Tiverton are winning the "
                    "physical contest after first contacts. Transition opportunities from "
                    "second ball wins should be logged against shot attempts to build a "
                    "direct play index over time."
                ))
            elif sb_wr >= 40:
                lines.append(_wrap(
                    f"Second ball recovery rate of {sb_wr}% — contested. Neither side has "
                    "clear dominance on second balls. The midfield unit must identify the "
                    "primary delivery zone and pre-position the #8 / #10 ahead of the "
                    "aerial contest to improve recovery positioning."
                ))
            else:
                lines.append(_wrap(
                    f"⚠ Second ball recovery rate of {sb_wr}% — opposition is winning the "
                    "majority of second contacts. This is a systemic risk in direct-play "
                    "matches: the opposition can sustain pressure phases from second-ball "
                    "control. Lever: consider a shorter build-out phase to avoid launching "
                    "aerial balls into unfavourable contest zones."
                ))
    else:
        lines.append("**Second Balls:** Not tagged this match.")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Dossier assembly
# ══════════════════════════════════════════════════════════════════════════════

def _build_dossier(
    ledger:   dict,
    feedback: str  = "",
    ctx:      dict | None = None,
) -> str:
    """Run all four agents and concatenate their output into a full dossier."""
    if ctx is None:
        ctx = {}
    match_id  = _match_id_from_ledger(ledger)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Rich header from match context ─────────────────────────────────────────
    score   = ctx.get("score", {})
    tiv_g   = score.get("tiverton", "?")
    opp_g   = score.get("opponent",  "?")
    opponent    = ctx.get("opponent",    "")
    competition = ctx.get("competition", "")
    venue       = ctx.get("venue",       "")
    home_game   = ctx.get("home_game",   True)
    date_str    = ctx.get("date",        match_id)
    result      = ctx.get("result",      "")

    def _scorer_str(scorers: list) -> str:
        return "  ·  ".join(f"{s['player']} {s.get('minute_raw', s['minute'])}′" for s in scorers) or "none"

    tiv_scorers = _scorer_str(ctx.get("scorers", {}).get("tiverton", []))
    opp_scorers = _scorer_str(ctx.get("scorers", {}).get("opponent", []))

    if opponent:
        match_line   = f"Tiverton Town {tiv_g}–{opp_g} {opponent}"
        result_line  = (
            f"**Result:** {result}  |  **Competition:** {competition}  |  "
            f"**Venue:** {venue} ({'home' if home_game else 'away'})"
        )
        scorers_line = (
            f"**Tiverton Scorers:** {tiv_scorers}  |  "
            f"**{opponent} Scorers:** {opp_scorers}"
        )
        lineup_line  = f"**XI:** {_format_lineup_header(ctx)}"
    else:
        match_line   = f"Match Date: {match_id}"
        result_line  = f"**Generated:** {generated}  |  **Framework:** Tactical Analysis"
        scorers_line = ""
        lineup_line  = ""

    header_lines = [
        "# TIVERTON TOWN FC — TACTICAL ANALYSIS DOSSIER",
        f"**{match_line}**",
        f"**Date:** {date_str}  |  {result_line}",
    ]
    if scorers_line:
        header_lines.append(scorers_line)
    if lineup_line:
        header_lines.append(lineup_line)
    header_lines += [
        f"**Generated:** {generated}  |  Tactical Analysis Framework | 4 Moments of the Game",
        "",
        "---",
        "",
    ]
    header = "\n".join(header_lines)

    section_ip  = run_in_possession_agent(ledger, feedback, ctx=ctx)
    section_pr  = run_press_agent(ledger, feedback, ctx=ctx)
    section_sp  = run_set_piece_agent(ledger, feedback, ctx=ctx)
    section_nl  = run_nonleague_agent(ledger, feedback, ctx=ctx)

    summary_stats = ledger.get("summary", {})
    schema_v      = ledger.get("schema_version", 1)
    zoned_count   = summary_stats.get("events_with_zone", "—")
    footer = "\n".join([
        "",
        "---",
        "",
        "## DATA QUALITY",
        f"Schema version: v{schema_v}  |  "
        f"Matched: {summary_stats.get('matched', '?')}  |  "
        f"Unmatched tags: {summary_stats.get('unmatched_tags', '?')}  |  "
        f"Unmatched club events: {summary_stats.get('unmatched_club_events', '?')}  |  "
        f"Events with real zone: {zoned_count}",
        "",
        "_PitchPulse Tactical Intelligence · Tiverton Town FC Performance Analysis_",
    ])

    sep = "\n\n---\n\n"
    return header + section_ip + sep + section_pr + sep + section_sp + sep + section_nl + footer


def _write_dossier(content: str, match_id: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = match_id.replace(":", "-").replace(" ", "_")
    out  = REPORTS_DIR / f"dossier_{slug}.md"
    out.write_text(content, encoding="utf-8")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Human-in-the-Loop CLI approval gate
# ══════════════════════════════════════════════════════════════════════════════

def run_approval_gate(ledger: dict) -> "Path | None":
    """
    Present the draft dossier to the manager and loop until:
      - 'approve'           → write dossier to disk, return Path
      - 'reject <feedback>' → re-run agents with feedback, repeat (max MAX_REJECTIONS)
      - 'quit'              → exit without saving, return None

    Returns the Path of the written dossier, or None if not saved.
    """
    feedback    = ""
    rejections  = 0
    match_id    = _match_id_from_ledger(ledger)
    ctx         = _load_match_context()

    while True:
        dossier = _build_dossier(ledger, feedback, ctx=ctx)

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
            return out  # forced approval path

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
            return None

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
            return out

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
            return None

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
    print("  Tiverton Town FC | Tactical Performance Intelligence")
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
    print("\n  Running four specialist agents…\n")
    print("    Agent 1: In-Possession Analyst     (SHOT, BOX_ENTRY)")
    print("    Agent 2: Out-of-Possession / Press  (HIGH_REGAIN, DEF_TURNOVER)")
    print("    Agent 3: Set-Piece Analyst          (SET_PIECE)")
    print("    Agent 4: Non-League Physics         (AERIAL_DUEL, SECOND_BALL)")
    print()

    saved = run_approval_gate(ledger)
    sys.exit(0 if saved else 1)


if __name__ == "__main__":
    main()
