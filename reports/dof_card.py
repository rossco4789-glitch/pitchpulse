"""
reports/dof_card.py
Director of Football matchday summary card.

Produces a 1080×1920 px (9:16 portrait) OLED-dark PNG at:
  data/processed/plots/dof_match_card.png

Designed for immediate WhatsApp delivery to the DoF.
No CDN, no external fonts — fully offline.

Usage:
    python reports/dof_card.py
    python reports/dof_card.py --ledger data/processed/match_ledger.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────────
LEDGER_PATH  = ROOT / "data" / "processed" / "match_ledger.json"
CONTEXT_PATH = ROOT / "data" / "raw" / "match_context.json"
PLOTS_DIR    = ROOT / "data" / "processed" / "plots"
CREST_PATH   = ROOT / "assets" / "tivvy_crest.png"
OUT_PATH     = PLOTS_DIR / "dof_match_card.png"

# ── Palette ────────────────────────────────────────────────────────────────────
BG       = "#09090b"   # OLED black
CARD_BG  = "#18181b"   # tile panel
GOLD     = "#f59e0b"   # Tivvy amber
GOLD_DIM = "#78350f"   # muted amber for borders
WHITE    = "#f8fafc"
MUTED    = "#94a3b8"   # secondary text
GREEN    = "#22c55e"
AMBER    = "#f59e0b"
RED      = "#ef4444"

# ── Figure dimensions ──────────────────────────────────────────────────────────
FIG_W, FIG_H = 10.8, 19.2   # inches
DPI          = 100            # → 1080 × 1920 px

# ── Zone lookups (matches cv/zones.py channel definitions) ────────────────────
_HALF_SPACE_ZONES = {"A_LH", "A_RH", "M_LH", "M_RH"}
_FLANK_CHANNELS   = {"LF", "RF"}


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_tags(ledger: dict, event_type: str) -> list[dict]:
    tags: list[dict] = []
    for m in ledger.get("matched", []):
        t = m.get("tag", {})
        if t.get("event_type") == event_type:
            tags.append(t)
    for t in ledger.get("unmatched_tags", []):
        if t.get("event_type") == event_type:
            tags.append(t)
    return tags


def _safe_pct(num: int, denom: int) -> int:
    return round(num / denom * 100) if denom else 0


def _rag(value: int, green_thresh: int, red_thresh: int, higher_is_better: bool = True) -> str:
    """Return 'green', 'amber', or 'red' RAG status."""
    if higher_is_better:
        if value >= green_thresh:
            return "green"
        if value >= red_thresh:
            return "amber"
        return "red"
    else:
        if value <= green_thresh:
            return "green"
        if value <= red_thresh:
            return "amber"
        return "red"


def _load_image(path: Path):
    """Load image array or return None if file missing/corrupt."""
    try:
        if path.exists() and path.stat().st_size > 0:
            return mpimg.imread(str(path))
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# KPI extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_kpis(ledger: dict) -> dict:
    """Derive the four KPI values from the match ledger."""
    try:
        from cv.zones import ZONES
    except ImportError:
        ZONES = {}

    # ── Box Entry corridor ──
    entries = _get_tags(ledger, "BOX_ENTRY")
    flank_n = hs_n = central_n = 0
    zoned = 0
    for e in entries:
        zid = e.get("zone_id")
        if not zid or zid not in ZONES:
            continue
        zoned += 1
        channel = ZONES[zid].get("channel", "")
        if channel in _FLANK_CHANNELS:
            flank_n += 1
        elif zid in _HALF_SPACE_ZONES:
            hs_n += 1
        else:
            central_n += 1

    if zoned > 0:
        dom_label  = max([("Flank", flank_n), ("Half-Space", hs_n), ("Central", central_n)],
                         key=lambda x: x[1])[0]
        dom_pct    = _safe_pct(max(flank_n, hs_n, central_n), zoned)
        entry_val  = f"{dom_label}"
        entry_subtitle  = f"{dom_pct}% of {zoned} confirmed entries"
        entry_rag  = "amber"
        if dom_label == "Half-Space" and dom_pct >= 40:
            entry_rag = "green"
        elif dom_label == "Flank" and dom_pct >= 60:
            entry_rag = "amber"
    else:
        entry_val      = "—"
        entry_subtitle = "No spatial data (v1 export)"
        entry_rag      = "amber"

    # ── Press efficiency ──
    regains   = _get_tags(ledger, "HIGH_REGAIN")
    turnovers = _get_tags(ledger, "DEF_TURNOVER")
    total_press = len(regains) + len(turnovers)
    if total_press > 0:
        press_pct      = _safe_pct(len(regains), total_press)
        press_val      = f"{press_pct}%"
        press_subtitle = f"{len(regains)} regains / {total_press} changes"
        press_rag      = _rag(press_pct, 60, 40)
    else:
        press_val      = "—"
        press_subtitle = "No pressing events tagged"
        press_rag      = "amber"

    # ── Aerial win % ──
    aerials    = _get_tags(ledger, "AERIAL_DUEL")
    aerial_won = sum(1 for a in aerials if a.get("sub_type") == "WON")
    if aerials:
        awr            = _safe_pct(aerial_won, len(aerials))
        aerial_val     = f"{awr}%"
        aerial_subtitle = f"{aerial_won} won / {len(aerials)} contested"
        aerial_rag     = _rag(awr, 55, 45)
    else:
        aerial_val     = "—"
        aerial_subtitle = "Not tagged this match"
        aerial_rag     = "amber"

    # ── Second ball ──
    sbs    = _get_tags(ledger, "SECOND_BALL")
    sb_won = sum(1 for s in sbs if s.get("sub_type") == "WON")
    if sbs:
        sbr          = _safe_pct(sb_won, len(sbs))
        sb_val       = f"{sbr}%"
        sb_subtitle  = f"{sb_won} won / {len(sbs)} contested"
        sb_rag       = _rag(sbr, 55, 40)
    else:
        sb_val      = "—"
        sb_subtitle = "Not tagged this match"
        sb_rag      = "amber"

    return {
        "entry":  {"title": "BOX ENTRIES",  "value": entry_val,  "subtitle": entry_subtitle,  "rag": entry_rag},
        "press":  {"title": "PRESS EFF.",   "value": press_val,  "subtitle": press_subtitle,  "rag": press_rag},
        "aerial": {"title": "AERIAL WIN %", "value": aerial_val, "subtitle": aerial_subtitle, "rag": aerial_rag},
        "sb":     {"title": "2ND BALL REC.","value": sb_val,     "subtitle": sb_subtitle,     "rag": sb_rag},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Bullet generation (rule-based, no LLM)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_bullets(ledger: dict, ctx: dict) -> list[str]:
    shots     = _get_tags(ledger, "SHOT")
    on_target = sum(1 for s in shots if s.get("sub_type") == "ON_TARGET")
    regains   = _get_tags(ledger, "HIGH_REGAIN")
    turnovers = _get_tags(ledger, "DEF_TURNOVER")
    aerials   = _get_tags(ledger, "AERIAL_DUEL")
    aerial_won = sum(1 for a in aerials if a.get("sub_type") == "WON")

    result   = ctx.get("result", "")
    opponent = ctx.get("opponent", "opposition")
    score    = ctx.get("score", {})
    subs     = [s for s in ctx.get("subs", []) if s.get("used") and s.get("minute")]

    # Bullet 1 — In Possession
    shot_n = len(shots)
    if shot_n == 0:
        b1 = "IN POSSESSION: No shots recorded — confirm tagger completeness for this fixture."
    elif on_target == 0:
        b1 = (f"IN POSSESSION: {shot_n} shot(s), zero on target — "
              "penetration achieved but final-action quality is the constraint. "
              "Prioritise shooting-angle selection in Zone 14.")
    else:
        ot_pct = _safe_pct(on_target, shot_n)
        quality = "strong" if ot_pct >= 50 else "moderate"
        b1 = (f"IN POSSESSION: {shot_n} shots, {on_target} on target ({ot_pct}%) — "
              f"{quality} conversion efficiency. "
              + ("Sustain half-space delivery to maintain Zone 14 access."
                 if ot_pct >= 50
                 else "Improve first-touch quality in the attacking third."))

    # Bullet 2 — Pressing
    total_press = len(regains) + len(turnovers)
    if total_press == 0:
        b2 = ("PRESSING: No events recorded — ensure HIGH_REGAIN and "
              "DEF_TURNOVER are tagged on matchday for this analysis.")
    else:
        eff = _safe_pct(len(regains), total_press)
        if eff >= 60:
            b2 = (f"PRESSING: {eff}% regain efficiency — Counter-Pressing Phase "
                  "is decisive. LoE is creating ball-wins in productive zones. "
                  "Sustain press trigger discipline.")
        elif eff >= 40:
            b2 = (f"PRESSING: {eff}% regain efficiency — contested. Tighten press "
                  "triggers; the #9 must set the angle to channel opposition build-out "
                  "before the #10 commits to press.")
        else:
            b2 = (f"PRESSING: {eff}% regain efficiency — opposition exploiting "
                  "Defensive Transition. Drop Line of Engagement to M_* zones and "
                  "prioritise Block compactness.")

    # Bullet 3 — Priority lever from match result and context
    tiv_g = score.get("tiverton", 0)
    opp_g = score.get("opponent",  0)

    if result == "W":
        if subs:
            sub_names = " & ".join(s["player"] for s in subs[:2])
            sub_min   = subs[0]["minute"]
            b3 = (f"LEVER: Impact substitutions shaped the result — "
                  f"{sub_names} (from {sub_min}′) shifted tempo. "
                  "Map the optimal sub window against the next opponent's pressing pattern.")
        else:
            b3 = (f"LEVER: {tiv_g}–{opp_g} W vs {opponent} — the defensive "
                  "block held when protecting the lead. Drill 4-4-2 compactness "
                  "in the next training block to sustain Rest Defense discipline.")
    elif result == "L":
        b3 = (f"LEVER: Defeat vs {opponent} ({tiv_g}–{opp_g}) — identify the "
              "score-state moment where defensive shape broke. Address the "
              "Defensive Transition trigger in the next training block.")
    elif result == "D":
        b3 = (f"LEVER: Draw vs {opponent} — review the final 15-minute phase. "
              "Attacking Transition from second-ball regains must convert to shots; "
              "the closing-period shot count is the target metric.")
    elif aerials:
        awr = _safe_pct(aerial_won, len(aerials))
        if awr < 50:
            b3 = (f"LEVER: Aerial win rate {awr}% — opposition achieving Quantitative "
                  "Superiority in the physical contest. Review direct-ball delivery "
                  "zones and second-ball runner pre-positioning.")
        else:
            b3 = (f"LEVER: Aerial dominance at {awr}% — exploit this platform. "
                  "A direct-play runner in behind the aerial contest creates "
                  "the Attacking Transition goal opportunity.")
    else:
        b3 = ("LEVER: Increase aerial and second-ball tagging volume to unlock "
              "direct-play pattern analysis. This is the primary non-league "
              "possession mechanism and must be quantified match-to-match.")

    return [b1, b2, b3]


# ══════════════════════════════════════════════════════════════════════════════
# Drawing primitives
# ══════════════════════════════════════════════════════════════════════════════

def _draw_section_label(fig: plt.Figure, y: float, label: str) -> None:
    """Thin gold rule with a centred section label."""
    fig.text(0.5, y + 0.008, label, ha="center", va="center",
             fontsize=10, color=GOLD, fontweight="bold",
             fontfamily="DejaVu Sans")
    # Gold hairline rule
    line = mpatches.FancyArrowPatch(
        (0.04, y), (0.96, y),
        transform=fig.transFigure,
        arrowstyle="-",
        color=GOLD_DIM, linewidth=0.8,
    )
    fig.add_artist(line)


def _draw_banner(fig: plt.Figure, ctx: dict) -> None:
    """Top banner: crest + match context."""
    # Banner background
    ax = fig.add_axes([0.0, 0.820, 1.0, 0.178])
    ax.set_facecolor(CARD_BG)
    ax.axis("off")

    # Gold top stripe
    ax.axhline(y=0.975, color=GOLD, linewidth=4, xmin=0.0, xmax=1.0)

    score  = ctx.get("score", {})
    tiv_g  = score.get("tiverton", "?")
    opp_g  = score.get("opponent",  "?")
    opp    = ctx.get("opponent",    "")
    comp   = ctx.get("competition", "")
    venue  = ctx.get("venue",       "")
    home   = ctx.get("home_game",   True)
    date   = ctx.get("date",        "")
    result = ctx.get("result",      "")

    result_color = {"W": GREEN, "D": AMBER, "L": RED}.get(result, MUTED)

    if opp:
        # Scoreline
        scoreline = f"{tiv_g}  —  {opp_g}"
        ax.text(0.5, 0.82, scoreline, ha="center", va="top",
                fontsize=54, color=WHITE, fontweight="bold",
                transform=ax.transAxes)
        # Result badge
        ax.text(0.5, 0.56, f"{'WIN' if result == 'W' else result}",
                ha="center", va="top",
                fontsize=16, color=result_color, fontweight="bold",
                transform=ax.transAxes)
        # Teams
        match_line = f"Tiverton Town  vs  {opp}"
        ax.text(0.5, 0.42, match_line, ha="center", va="top",
                fontsize=14, color=GOLD, fontweight="bold",
                transform=ax.transAxes)
        # Competition & venue
        venue_tag = "home" if home else "away"
        detail = f"{comp}  ·  {venue} ({venue_tag})  ·  {date}"
        ax.text(0.5, 0.22, detail, ha="center", va="top",
                fontsize=10, color=MUTED,
                transform=ax.transAxes)
    else:
        ax.text(0.5, 0.6, "TIVERTON TOWN FC", ha="center", va="center",
                fontsize=22, color=GOLD, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.3, "Tactical Match Summary", ha="center", va="center",
                fontsize=13, color=MUTED, transform=ax.transAxes)

    # ── Club crest ────────────────────────────────────────────────────────────
    crest_img = _load_image(CREST_PATH)
    if crest_img is not None:
        # Small axes in the top-left of the banner
        cax = fig.add_axes([0.02, 0.855, 0.13, 0.135])
        cax.imshow(crest_img)
        cax.axis("off")
    else:
        # Fallback: amber circle with "TT"
        cax = fig.add_axes([0.02, 0.855, 0.13, 0.135])
        cax.set_facecolor(BG)
        cax.axis("off")
        circle = plt.Circle((0.5, 0.5), 0.45, color=GOLD, transform=cax.transAxes)
        cax.add_patch(circle)
        cax.text(0.5, 0.5, "TT", ha="center", va="center",
                 fontsize=20, color=BG, fontweight="bold",
                 transform=cax.transAxes)


def _draw_kpi_tile(
    fig: plt.Figure,
    left: float, bottom: float, width: float, height: float,
    title: str, value: str, subtitle: str, rag: str = "amber",
) -> None:
    """Single KPI tile with coloured value and gold top accent."""
    color = {"green": GREEN, "amber": GOLD, "red": RED}.get(rag, GOLD)

    ax = fig.add_axes([left, bottom, width, height])
    ax.set_facecolor(CARD_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Coloured top accent bar
    ax.add_patch(mpatches.Rectangle((0, 0.94), 1, 0.06, color=color,
                                     transform=ax.transAxes, clip_on=False))
    # Title
    ax.text(0.5, 0.84, title, ha="center", va="top",
            fontsize=11, color=MUTED, fontweight="bold",
            transform=ax.transAxes)
    # Value
    ax.text(0.5, 0.53, value, ha="center", va="center",
            fontsize=34 if len(value) <= 4 else 26,
            color=color, fontweight="bold",
            transform=ax.transAxes)
    # Subtitle
    ax.text(0.5, 0.13, subtitle, ha="center", va="bottom",
            fontsize=8.5, color=MUTED,
            transform=ax.transAxes, wrap=True)


def _draw_pitch_image(
    fig: plt.Figure,
    left: float, bottom: float, width: float, height: float,
    img_path: Path, label: str,
) -> None:
    """Inset pitch image with label. Placeholder if image unavailable."""
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_facecolor(CARD_BG)
    ax.axis("off")

    img = _load_image(img_path)
    if img is not None:
        ax.imshow(img, aspect="auto", extent=[0, 1, 0, 1],
                  transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No Plot\nAvailable", ha="center", va="center",
                fontsize=12, color=MUTED, transform=ax.transAxes)

    ax.text(0.5, -0.06, label, ha="center", va="top",
            fontsize=10, color=GOLD, fontweight="bold",
            transform=ax.transAxes)


def _draw_bullets(fig: plt.Figure, bullets: list[str]) -> None:
    """Three executive bullet points with gold bullet marks."""
    ax = fig.add_axes([0.04, 0.072, 0.92, 0.155])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bullet_mark = "▸"
    y_positions = [0.82, 0.50, 0.18]

    for y_pos, text in zip(y_positions, bullets):
        # Gold bullet
        ax.text(0.01, y_pos, bullet_mark, ha="left", va="center",
                fontsize=14, color=GOLD, fontweight="bold",
                transform=ax.transAxes)
        # Wrap text manually (matplotlib wrap unreliable in fixed axes)
        # Limit to ~88 chars per line
        words = text.split()
        lines = []
        line  = []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 84:
                lines.append(" ".join(line[:-1]))
                line = [w]
        if line:
            lines.append(" ".join(line))
        display = "\n".join(lines[:3])  # max 3 lines per bullet

        ax.text(0.06, y_pos, display, ha="left", va="center",
                fontsize=9.5, color=WHITE, linespacing=1.4,
                transform=ax.transAxes)


def _draw_footer(fig: plt.Figure, ledger: dict, ctx: dict) -> None:
    """Data quality row + footer stamp."""
    summary    = ledger.get("summary", {})
    schema_v   = ledger.get("schema_version", 1)
    matched    = summary.get("matched", "?")
    unmatched  = summary.get("unmatched_tags", "?")
    zoned      = summary.get("events_with_zone", "—")
    generated  = ctx.get("date", "")

    dq_text = (
        f"Data: schema v{schema_v}  ·  "
        f"Matched {matched}  ·  Unmatched {unmatched}  ·  "
        f"Zoned {zoned}"
    )
    fig.text(0.5, 0.048, dq_text, ha="center", va="center",
             fontsize=8.5, color=MUTED)

    fig.text(0.5, 0.012, "PitchPulse Tactical Intelligence  ·  Tiverton Town FC",
             ha="center", va="center", fontsize=8, color=GOLD_DIM)


# ══════════════════════════════════════════════════════════════════════════════
# Main card builder
# ══════════════════════════════════════════════════════════════════════════════

def build_dof_card(
    ledger:    dict,
    ctx:       dict,
    plots_dir: Path,
    out_path:  Path,
) -> Path:
    """
    Render the DoF match card and write to out_path.
    Returns out_path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(BG)

    # ── Top banner ─────────────────────────────────────────────────────────────
    _draw_banner(fig, ctx)

    # ── Section label: KPI Dashboard ──────────────────────────────────────────
    _draw_section_label(fig, 0.812, "MATCH PERFORMANCE INDICATORS")

    # ── KPI tiles (2×2 grid) ──────────────────────────────────────────────────
    kpis = _extract_kpis(ledger)
    tile_w, tile_h = 0.435, 0.155
    gap_x          = 0.085
    left_x         = 0.04
    right_x        = left_x + tile_w + gap_x

    # Top row
    _draw_kpi_tile(fig, left_x,  0.640, tile_w, tile_h, **kpis["entry"])
    _draw_kpi_tile(fig, right_x, 0.640, tile_w, tile_h, **kpis["press"])
    # Bottom row
    _draw_kpi_tile(fig, left_x,  0.470, tile_w, tile_h, **kpis["aerial"])
    _draw_kpi_tile(fig, right_x, 0.470, tile_w, tile_h, **kpis["sb"])

    # ── Section label: Pitch Analysis ─────────────────────────────────────────
    _draw_section_label(fig, 0.462, "PITCH ANALYSIS")

    # ── Pitch miniatures ──────────────────────────────────────────────────────
    img_w, img_h = 0.435, 0.190
    _draw_pitch_image(
        fig, left_x, 0.252, img_w, img_h,
        plots_dir / "shot_map.png", "ATTACKING ACTIONS",
    )
    _draw_pitch_image(
        fig, right_x, 0.252, img_w, img_h,
        plots_dir / "transition_map.png", "PRESS & TURNOVERS",
    )

    # ── Section label: Tactical Takeaways ─────────────────────────────────────
    _draw_section_label(fig, 0.240, "TACTICAL TAKEAWAYS")

    # ── Executive bullets ─────────────────────────────────────────────────────
    bullets = _generate_bullets(ledger, ctx)
    _draw_bullets(fig, bullets)

    # ── Footer ────────────────────────────────────────────────────────────────
    _draw_footer(fig, ledger, ctx)

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(str(out_path), dpi=DPI, facecolor=BG, bbox_inches=None)
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _load_json(path: Path, label: str) -> dict:
    if not path.exists():
        print(f"  [DOF] {label} not found: {path.relative_to(ROOT)}")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="PitchPulse DoF match card renderer — Tiverton Town FC"
    )
    ap.add_argument("--ledger",  type=Path, default=LEDGER_PATH)
    ap.add_argument("--context", type=Path, default=CONTEXT_PATH)
    ap.add_argument("--out",     type=Path, default=OUT_PATH)
    ap.add_argument("--plots",   type=Path, default=PLOTS_DIR)
    args = ap.parse_args()

    ledger = _load_json(args.ledger,  "Match ledger")
    ctx    = _load_json(args.context, "Match context")

    if not ledger:
        print("  [DOF] Cannot render without ledger. Run reconcile/sync.py first.")
        sys.exit(1)

    out = build_dof_card(ledger, ctx, args.plots, args.out)
    size_kb = round(out.stat().st_size / 1024, 1)
    print(f"  [DOF] ✓ Card rendered → {out.relative_to(ROOT)}  ({size_kb} KB  |  1080×1920 px)")


if __name__ == "__main__":
    main()
