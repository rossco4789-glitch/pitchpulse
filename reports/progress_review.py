"""
reports/progress_review.py
Longitudinal Tactical Progress Review engine for Tiverton Town FC.
Aggregates multiple match ledgers over a rolling 6–8 game window.

CLI:
    python reports/progress_review.py --last 6
    python reports/progress_review.py --from 15-08-2026 --to 06-09-2026

Output:
    data/processed/progress_review_DD-MM-YYYY.html
    data/processed/progress_review_DD-MM-YYYY_dof_card.html

Ledger discovery:
    Globs data/processed/ledger_DD-MM-YYYY.json (UK standard, e.g. ledger_15-08-2026.json).
    Also accepts legacy ledger_YYYY-MM-DD.json for backward compatibility.
    After each matchday: cp data/processed/match_ledger.json data/processed/ledger_$(date +%d-%m-%Y).json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from collections import namedtuple
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.zones import ZONES

# ── Paths ──────────────────────────────────────────────────────────────────────
PROC_DIR = ROOT / "data" / "processed"

# ── OLED colour palette ────────────────────────────────────────────────────────
OLED_BG  = "#09090b"
SURFACE  = "#18181b"
LINE     = "#3f3f46"
GOLD     = "#f59e0b"
CRIMSON  = "#f43f5e"
EMERALD  = "#10b981"
CYAN     = "#22d3ee"
BLUE     = "#3b82f6"
PURPLE   = "#a855f7"
ZINC     = "#71717a"
TEXT     = "#e4e4e7"
TEXT_DIM = "#52525b"

_HALF_SPACE   = {"A_LH", "A_RH"}
_ZONE_14      = {"A_LC", "A_RC"}
_QS_THRESHOLD = 0.30


# ══════════════════════════════════════════════════════════════════════════════
# MatchSnapshot — one per ledger file
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MatchSnapshot:
    # Identity
    match_date:    str  = ""
    opponent:      str  = "Unknown"
    competition:   str  = ""
    result:        str  = ""
    tiv_goals:     int  = 0
    opp_goals:     int  = 0
    # In Possession
    shots:               int = 0
    on_target:           int = 0
    zone14_shots:        int = 0
    box_entries:         int = 0
    zoned_entries:       int = 0
    half_space_entries:  int = 0
    # Pressing
    ball_wins:           int = 0
    possession_changes:  int = 0
    d_regains:           int = 0
    m_regains:           int = 0
    a_regains:           int = 0
    channel_spread:      int = 0
    # Set Pieces
    att_corners:          int  = 0
    def_corners:          int  = 0
    free_kicks:           int  = 0
    rest_defense_flagged: bool = False
    # Non-League Physics
    aerial_total:   int = 0
    aerial_won:     int = 0
    aerial_d_total: int = 0
    aerial_d_won:   int = 0
    aerial_m_total: int = 0
    aerial_m_won:   int = 0
    sb_total:       int = 0
    sb_won:         int = 0


# ══════════════════════════════════════════════════════════════════════════════
# Ledger helpers
# ══════════════════════════════════════════════════════════════════════════════

def _all_tags(ledger: dict) -> list[dict]:
    tags: list[dict] = []
    for m in ledger.get("matched", []):
        t = m.get("tag")
        if t:
            tags.append(t)
    tags.extend(ledger.get("unmatched_tags", []))
    return tags


def _by_type(tags: list[dict], ev: str) -> list[dict]:
    return [t for t in tags if t.get("event_type") == ev]


def _safe_pct(num: int, denom: int) -> float:
    return round(100 * num / denom, 1) if denom else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Ledger discovery
# ══════════════════════════════════════════════════════════════════════════════

def _parse_ledger_date(date_str: str) -> date:
    """Parse a date string from a ledger filename, accepting UK (DD-MM-YYYY) or ISO (YYYY-MM-DD)."""
    for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised ledger date format: {date_str!r}")


def _parse_filter_date(date_str: str) -> date:
    """Parse a filter date string supplied by the user, accepting DD-MM-YYYY or YYYY-MM-DD."""
    return _parse_ledger_date(date_str)


def discover_ledgers(
    proc_dir:  Path = PROC_DIR,
    last_n:    int  = 6,
    date_from: Optional[str] = None,
    date_to:   Optional[str] = None,
) -> list[Path]:
    """
    Find date-stamped ledger files in proc_dir.
    Accepts UK format ledger_DD-MM-YYYY.json (preferred) and legacy ledger_YYYY-MM-DD.json.
    Returns paths in chronological order.
    """
    candidates: list[tuple[date, Path]] = []
    for p in proc_dir.glob("ledger_*.json"):
        raw = p.stem.replace("ledger_", "")
        try:
            d = _parse_ledger_date(raw)
            candidates.append((d, p))
        except ValueError:
            pass  # skip non-date-stamped files (e.g. match_ledger.json backups)

    candidates.sort(key=lambda x: x[0], reverse=True)   # newest first

    if date_from or date_to:
        lo = _parse_filter_date(date_from) if date_from else date.min
        hi = _parse_filter_date(date_to)   if date_to   else date.max
        candidates = [(d, p) for d, p in candidates if lo <= d <= hi]
    else:
        candidates = candidates[:last_n]

    return [p for _, p in reversed(candidates)]          # chronological


# ══════════════════════════════════════════════════════════════════════════════
# Per-game metric extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_per_game_metrics(ledger_path: Path) -> MatchSnapshot:
    with open(ledger_path, encoding="utf-8") as f:
        ledger = json.load(f)

    snap = MatchSnapshot()
    snap.match_date  = ledger.get("match_date", ledger_path.stem.replace("ledger_", ""))
    ctx              = ledger.get("ctx", {})
    snap.opponent    = ctx.get("opponent",    "Unknown")
    snap.competition = ctx.get("competition", "")
    snap.result      = ctx.get("result",      "")
    score            = ctx.get("score", {})
    snap.tiv_goals   = int(score.get("tiverton", 0) or 0)
    snap.opp_goals   = int(score.get("opponent",  0) or 0)

    tags = _all_tags(ledger)

    # ── In Possession ─────────────────────────────────────────────────────────
    shots = _by_type(tags, "SHOT")
    snap.shots        = len(shots)
    snap.on_target    = sum(1 for s in shots if s.get("sub_type") == "ON_TARGET")
    snap.zone14_shots = sum(1 for s in shots if s.get("zone_id") in _ZONE_14)

    entries = _by_type(tags, "BOX_ENTRY")
    snap.box_entries        = len(entries)
    snap.zoned_entries      = sum(1 for e in entries if e.get("zone_id"))
    snap.half_space_entries = sum(1 for e in entries if e.get("zone_id") in _HALF_SPACE)

    # ── Pressing ──────────────────────────────────────────────────────────────
    regains   = _by_type(tags, "HIGH_REGAIN")
    turnovers = _by_type(tags, "DEF_TURNOVER")
    snap.ball_wins          = len(regains)
    snap.possession_changes = len(regains) + len(turnovers)

    for r in regains:
        zid = r.get("zone_id")
        if zid and zid in ZONES:
            t = ZONES[zid]["third"]
            if   t == "A": snap.a_regains += 1
            elif t == "M": snap.m_regains += 1
            elif t == "D": snap.d_regains += 1

    to_channels: set[str] = set()
    for t in turnovers:
        zid = t.get("zone_id")
        if zid and zid in ZONES:
            to_channels.add(ZONES[zid]["channel"])
    snap.channel_spread = len(to_channels)

    # ── Set Pieces ────────────────────────────────────────────────────────────
    sp    = _by_type(tags, "SET_PIECE")
    att_c = [e for e in sp if e.get("sub_type") == "ATT_CORNER"]
    def_c = [e for e in sp if e.get("sub_type") == "DEF_CORNER"]
    snap.att_corners = len(att_c)
    snap.def_corners = len(def_c)
    snap.free_kicks  = len([e for e in sp if e.get("sub_type") == "FREE_KICK"])
    def_d = [e for e in def_c if (e.get("zone_id") or "").startswith("D_")]
    snap.rest_defense_flagged = len(def_c) >= 3 or len(def_d) >= 2

    # ── Non-League Physics ────────────────────────────────────────────────────
    aerials = _by_type(tags, "AERIAL_DUEL")
    sbs     = _by_type(tags, "SECOND_BALL")
    snap.aerial_total = len(aerials)
    snap.aerial_won   = sum(1 for a in aerials if a.get("sub_type") == "WON")
    snap.sb_total     = len(sbs)
    snap.sb_won       = sum(1 for s in sbs if s.get("sub_type") == "WON")

    for a in aerials:
        zid = a.get("zone_id")
        won = a.get("sub_type") == "WON"
        if zid and zid in ZONES:
            t = ZONES[zid]["third"]
            if t == "D":
                snap.aerial_d_total += 1
                if won: snap.aerial_d_won += 1
            elif t == "M":
                snap.aerial_m_total += 1
                if won: snap.aerial_m_won += 1

    return snap


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation → DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_snapshots(snaps: list[MatchSnapshot]):
    """Convert MatchSnapshot list → pandas DataFrame (one row per game)."""
    import pandas as pd

    rows = []
    for s in snaps:
        zoned_r = s.a_regains + s.m_regains + s.d_regains
        loe_idx = round((s.a_regains * 2 + s.m_regains) / zoned_r, 2) if zoned_r else 0.0
        rows.append({
            "match_date":           s.match_date,
            "opponent":             s.opponent,
            "result":               s.result,
            "tiv_goals":            s.tiv_goals,
            "opp_goals":            s.opp_goals,
            # IP
            "shots":                s.shots,
            "on_target":            s.on_target,
            "on_target_pct":        _safe_pct(s.on_target,           s.shots),
            "zone14_pct":           _safe_pct(s.zone14_shots,        s.shots),
            "half_space_pct":       _safe_pct(s.half_space_entries,  s.zoned_entries),
            "box_entries":          s.box_entries,
            # Press
            "ball_wins":            s.ball_wins,
            "possession_changes":   s.possession_changes,
            "cp_efficiency":        _safe_pct(s.ball_wins,           s.possession_changes),
            "loe_index":            loe_idx,
            "a_regains":            s.a_regains,
            "m_regains":            s.m_regains,
            "d_regains":            s.d_regains,
            "channel_spread":       s.channel_spread,
            # SP
            "att_corners":          s.att_corners,
            "def_corners":          s.def_corners,
            "free_kicks":           s.free_kicks,
            "rest_defense_flagged": int(s.rest_defense_flagged),
            # NL
            "aerial_total":         s.aerial_total,
            "aerial_wr":            _safe_pct(s.aerial_won,    s.aerial_total),
            "aerial_wr_d":          _safe_pct(s.aerial_d_won,  s.aerial_d_total),
            "aerial_wr_m":          _safe_pct(s.aerial_m_won,  s.aerial_m_total),
            "sb_total":             s.sb_total,
            "sb_wr":                _safe_pct(s.sb_won,        s.sb_total),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("match_date").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Chart helpers
# ══════════════════════════════════════════════════════════════════════════════

def _dark_ax(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.patch.set_facecolor(OLED_BG)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=ZINC, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(LINE); sp.set_linewidth(0.6)
    ax.grid(axis="y", color=LINE, linewidth=0.4, linestyle="--", alpha=0.5)


def _game_labels(df) -> list[str]:
    return [
        f"{str(r['match_date'])[5:]}\n{str(r['opponent'])[:8]}"
        for _, r in df.iterrows()
    ]


def _ma(ax: plt.Axes, x, vals, color: str, w: int = 3) -> None:
    import pandas as pd
    ma = pd.Series(vals, dtype=float).rolling(w, min_periods=1, center=True).mean()
    ax.plot(x, ma.values, color=color, lw=1.6, ls="--", alpha=0.85, zorder=5, label=f"{w}-game MA")


def fig_in_possession(df) -> plt.Figure:
    n, x, labels = len(df), np.arange(len(df)), _game_labels(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.patch.set_facecolor(OLED_BG)
    fig.suptitle("In Possession Trends", color=TEXT, fontsize=10,
                 fontweight="bold", x=0.02, ha="left")

    _dark_ax(fig, ax1)
    ax1.bar(x, df["shots"],     color=ZINC,   alpha=0.55, label="Shots",    zorder=3)
    ax1.bar(x, df["on_target"], color=GOLD,   alpha=0.85, label="On Target",zorder=4)
    _ma(ax1, x, df["shots"], CRIMSON)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=6.5, color=ZINC)
    ax1.set_ylabel("Count", fontsize=7, color=ZINC)
    ax1.set_title("Shots per Game", color=TEXT_DIM, fontsize=8)
    ax1.legend(fontsize=6.5, facecolor=SURFACE, edgecolor=LINE, labelcolor=TEXT)

    _dark_ax(fig, ax2)
    ax2.bar(x - 0.2, df["half_space_pct"], width=0.38, color=BLUE,   alpha=0.85, label="Half-space %", zorder=4)
    ax2.bar(x + 0.2, df["zone14_pct"],     width=0.38, color=PURPLE, alpha=0.85, label="Zone 14 %",    zorder=4)
    ax2.axhline(30, color=GOLD, lw=0.9, ls=":", alpha=0.7, label="QS threshold")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=6.5, color=ZINC)
    ax2.set_ylabel("Rate (%)", fontsize=7, color=ZINC)
    ax2.set_title("Entry Corridor Quality", color=TEXT_DIM, fontsize=8)
    ax2.legend(fontsize=6.5, facecolor=SURFACE, edgecolor=LINE, labelcolor=TEXT)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


def fig_pressing(df) -> plt.Figure:
    x, labels = np.arange(len(df)), _game_labels(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.patch.set_facecolor(OLED_BG)
    fig.suptitle("Press & LoE Trends", color=TEXT, fontsize=10,
                 fontweight="bold", x=0.02, ha="left")

    _dark_ax(fig, ax1)
    ax1.bar(x, df["cp_efficiency"], color=EMERALD, alpha=0.78, label="CP Efficiency %", zorder=4)
    _ma(ax1, x, df["cp_efficiency"], CYAN)
    ax1.axhline(35, color=CRIMSON, lw=0.9, ls=":", alpha=0.7, label="Alert threshold")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=6.5, color=ZINC)
    ax1.set_ylabel("Efficiency (%)", fontsize=7, color=ZINC)
    ax1.set_title("Counter-Pressing Phase Efficiency", color=TEXT_DIM, fontsize=8)
    ax1.legend(fontsize=6.5, facecolor=SURFACE, edgecolor=LINE, labelcolor=TEXT)

    _dark_ax(fig, ax2)
    ax2.plot(x, df["loe_index"], color=BLUE, lw=1.8, marker="o", markersize=5,
             markerfacecolor=GOLD, markeredgecolor="white", markeredgewidth=0.5, zorder=5)
    ax2.axhline(1.0, color=ZINC, lw=0.7, ls="--", alpha=0.45)
    ax2.set_ylim(-0.1, 2.2)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["Low Block", "Mid Block", "High Block"], fontsize=6.5, color=ZINC)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=6.5, color=ZINC)
    ax2.set_title("Line of Engagement Height Index", color=TEXT_DIM, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


def fig_set_pieces(df) -> plt.Figure:
    x, labels = np.arange(len(df)), _game_labels(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.patch.set_facecolor(OLED_BG)
    fig.suptitle("Set Piece Trends", color=TEXT, fontsize=10,
                 fontweight="bold", x=0.02, ha="left")

    _dark_ax(fig, ax1)
    ax1.bar(x - 0.2, df["att_corners"], width=0.38, color=EMERALD, alpha=0.85, label="Att Corners", zorder=4)
    ax1.bar(x + 0.2, df["def_corners"], width=0.38, color=CRIMSON, alpha=0.85, label="Def Corners", zorder=4)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=6.5, color=ZINC)
    ax1.set_ylabel("Count", fontsize=7, color=ZINC)
    ax1.set_title("Corner Volume", color=TEXT_DIM, fontsize=8)
    ax1.legend(fontsize=6.5, facecolor=SURFACE, edgecolor=LINE, labelcolor=TEXT)

    _dark_ax(fig, ax2)
    colors = [CRIMSON if v else ZINC for v in df["rest_defense_flagged"]]
    ax2.bar(x, df["def_corners"], color=colors, alpha=0.8, zorder=4)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=6.5, color=ZINC)
    ax2.set_ylabel("Def Corners", fontsize=7, color=ZINC)
    ax2.set_title("Rest Defense Exposure  (red = flagged)", color=TEXT_DIM, fontsize=8)
    ax2.legend(handles=[
        mpatches.Patch(color=CRIMSON, alpha=0.8, label="Rest Defense Flagged"),
        mpatches.Patch(color=ZINC,    alpha=0.8, label="Clean"),
    ], fontsize=6.5, facecolor=SURFACE, edgecolor=LINE, labelcolor=TEXT)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


def fig_nonleague(df) -> plt.Figure:
    x, labels = np.arange(len(df)), _game_labels(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.patch.set_facecolor(OLED_BG)
    fig.suptitle("Non-League Physics Trends", color=TEXT, fontsize=10,
                 fontweight="bold", x=0.02, ha="left")

    _dark_ax(fig, ax1)
    ax1.plot(x, df["aerial_wr_d"], color=CRIMSON, lw=1.6, marker="s", markersize=4,
             markeredgecolor="white", markeredgewidth=0.4, label="Aerial WR D-third", zorder=5)
    ax1.plot(x, df["aerial_wr_m"], color=EMERALD, lw=1.6, marker="o", markersize=4,
             markeredgecolor="white", markeredgewidth=0.4, label="Aerial WR M-third", zorder=5)
    ax1.axhline(50, color=GOLD, lw=0.8, ls=":", alpha=0.6, label="50% parity")
    ax1.set_ylim(-5, 105)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=6.5, color=ZINC)
    ax1.set_ylabel("Win Rate (%)", fontsize=7, color=ZINC)
    ax1.set_title("Aerial Duel Win Rate by Zone", color=TEXT_DIM, fontsize=8)
    ax1.legend(fontsize=6.5, facecolor=SURFACE, edgecolor=LINE, labelcolor=TEXT)

    _dark_ax(fig, ax2)
    ax2.bar(x, df["sb_wr"], color=BLUE, alpha=0.82, label="Second Ball Recovery %", zorder=4)
    _ma(ax2, x, df["sb_wr"], CYAN)
    ax2.axhline(40, color=CRIMSON, lw=0.9, ls=":", alpha=0.7, label="Risk threshold")
    ax2.axhline(60, color=EMERALD, lw=0.9, ls=":", alpha=0.7, label="Dominance threshold")
    ax2.set_ylim(0, 110)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=6.5, color=ZINC)
    ax2.set_ylabel("Recovery Rate (%)", fontsize=7, color=ZINC)
    ax2.set_title("Second Ball Recovery", color=TEXT_DIM, fontsize=8)
    ax2.legend(fontsize=6.5, facecolor=SURFACE, edgecolor=LINE, labelcolor=TEXT)

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


def _fig_to_b64(fig: plt.Figure, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ══════════════════════════════════════════════════════════════════════════════
# Narrative generators
# ══════════════════════════════════════════════════════════════════════════════

def _narrate_ip(df) -> str:
    avg_hs  = df["half_space_pct"].mean()
    avg_z14 = df["zone14_pct"].mean()
    avg_sot = df["on_target_pct"].mean()
    trend   = df["half_space_pct"].iloc[-1] - df["half_space_pct"].iloc[0] if len(df) > 1 else 0.0
    status  = (
        "above the Qualitative Superiority threshold — inside channels are being consistently accessed"
        if avg_hs >= 30 else
        f"below the 30% QS threshold — inside-channel access must be a training priority"
    )
    trend_note = (
        f" Trend: {'rising' if trend > 3 else 'falling' if trend < -3 else 'stable'} "
        f"({'+' if trend >= 0 else ''}{trend:.0f}% across the window)."
        if len(df) > 1 else ""
    )
    return (
        f"Half-space penetration averaged **{avg_hs:.0f}%** — {status}.{trend_note} "
        f"Zone 14 shot share: **{avg_z14:.0f}%** (target ≥20%). "
        f"Shot-on-target rate: **{avg_sot:.0f}%**."
    )


def _narrate_press(df) -> str:
    avg_cp  = df["cp_efficiency"].mean()
    avg_loe = df["loe_index"].mean()
    loe_lbl = "High Block" if avg_loe >= 1.5 else "Mid Block" if avg_loe >= 0.8 else "Low Block (reactive)"
    alert   = " ⚠ CP efficiency critically low — Defensive Transition phase is being exploited." if avg_cp < 35 else ""
    var_loe = df["loe_index"].std() if len(df) > 1 else 0.0
    stability = "consistent block height" if var_loe < 0.3 else "variable block height — shape instability noted"
    return (
        f"Counter-Pressing Phase efficiency averaged **{avg_cp:.0f}%**.{alert} "
        f"LoE height index averaged **{avg_loe:.2f}** ({loe_lbl}) — {stability}."
    )


def _narrate_sp(df) -> str:
    flagged  = int(df["rest_defense_flagged"].sum())
    avg_def  = df["def_corners"].mean()
    flag_note = (
        f"⚠ Rest Defense flagged in **{flagged}/{len(df)} games** — dead-ball defensive shape is a priority coaching area."
        if flagged >= 2 else
        f"Rest Defense flagged in {flagged}/{len(df)} games — within acceptable range."
    )
    return (
        f"{flag_note} Defensive corner volume averaged **{avg_def:.1f} per game**. "
        "Second-ball clearance discipline and block shape after defensive set pieces should be monitored."
    )


def _narrate_nl(df) -> str:
    avg_d  = df["aerial_wr_d"].mean()
    avg_m  = df["aerial_wr_m"].mean()
    avg_sb = df["sb_wr"].mean()
    d_note = (
        "⚠ Sub-50% aerial win rate in the defensive third — direct balls over the back line are a live threat."
        if avg_d < 50 else f"Solid defensive-third aerial win rate ({avg_d:.0f}%)."
    )
    m_note = "strong platform for Attacking Transition" if avg_m >= 55 else "contested — midfield physical battle not being won"
    sb_note = "dominant" if avg_sb >= 60 else "contested" if avg_sb >= 40 else "⚠ below risk threshold — systemic second-ball problem"
    return (
        f"{d_note} Middle-third aerial dominance: **{avg_m:.0f}%** ({m_note}). "
        f"Second-ball recovery rate averaged **{avg_sb:.0f}%** ({sb_note})."
    )


# ══════════════════════════════════════════════════════════════════════════════
# HTML report
# ══════════════════════════════════════════════════════════════════════════════

_STYLE = """
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700;900&family=Inter:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#09090b;--sf:#18181b;--bd:rgba(255,255,255,.06);--am:#f59e0b;--tx:#e4e4e7;--dim:#71717a}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font-family:'Inter',system-ui,sans-serif;font-size:14px;line-height:1.6;padding:32px 24px;max-width:1100px;margin:0 auto}
h1{font-family:'Barlow Condensed',Impact,sans-serif;font-size:2rem;font-weight:900;color:var(--am);letter-spacing:.04em;text-transform:uppercase;margin-bottom:4px}
h2{font-family:'Barlow Condensed',Impact,sans-serif;font-size:1.1rem;font-weight:700;color:var(--tx);letter-spacing:.06em;text-transform:uppercase;margin:28px 0 10px;display:flex;align-items:center;gap:10px}
h2::before{content:'';display:block;width:3px;height:16px;background:var(--am);border-radius:2px;flex-shrink:0}
p{color:var(--dim);margin-bottom:12px;max-width:820px}strong{color:var(--tx)}
.meta{display:flex;gap:20px;flex-wrap:wrap;margin-top:10px}
.meta span{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--dim)}
.meta span b{color:var(--am)}
.chips{display:flex;gap:10px;margin:14px 0;flex-wrap:wrap}
.chip{font-family:'JetBrains Mono',monospace;font-size:.8rem;padding:4px 14px;border-radius:6px;font-weight:700}
.w{background:rgba(16,185,129,.15);color:#10b981;border:1px solid rgba(16,185,129,.3)}
.d{background:rgba(245,158,11,.12);color:var(--am);border:1px solid rgba(245,158,11,.25)}
.l{background:rgba(244,63,94,.12);color:#f43f5e;border:1px solid rgba(244,63,94,.25)}
.plot{width:100%;border-radius:8px;margin:10px 0;border:1px solid var(--bd)}
table{width:100%;border-collapse:collapse;font-size:.75rem;font-family:'JetBrains Mono',monospace;margin-top:10px}
th{color:var(--dim);text-transform:uppercase;letter-spacing:.08em;font-size:.63rem;padding:7px 10px;border-bottom:1px solid var(--bd);text-align:left}
td{padding:6px 10px;border-bottom:1px solid rgba(255,255,255,.025);color:var(--tx)}
tr:hover td{background:rgba(255,255,255,.02)}
footer{margin-top:40px;padding-top:14px;border-top:1px solid var(--bd);font-family:'JetBrains Mono',monospace;font-size:.62rem;color:var(--dim)}
</style>"""


def render_html_report(df, snaps: list[MatchSnapshot]) -> str:
    n      = len(df)
    wins   = sum(1 for s in snaps if s.result.upper().startswith("W"))
    draws  = sum(1 for s in snaps if s.result.upper().startswith("D"))
    losses = sum(1 for s in snaps if s.result.upper().startswith("L"))
    gf     = sum(s.tiv_goals for s in snaps)
    ga     = sum(s.opp_goals  for s in snaps)
    d0     = df["match_date"].iloc[0]  if not df.empty else "?"
    d1     = df["match_date"].iloc[-1] if not df.empty else "?"
    now    = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build figures and convert to b64 (closes figures in the process)
    b64_ip    = _fig_to_b64(fig_in_possession(df))
    b64_press = _fig_to_b64(fig_pressing(df))
    b64_sp    = _fig_to_b64(fig_set_pieces(df))
    b64_nl    = _fig_to_b64(fig_nonleague(df))

    rows = ""
    for _, r in df.iterrows():
        rd = "🚨" if r["rest_defense_flagged"] else "—"
        rows += (
            f"<tr><td>{r['match_date']}</td><td>{r['opponent']}</td>"
            f"<td>{r['result']}</td><td>{r['tiv_goals']}–{r['opp_goals']}</td>"
            f"<td>{r['loe_index']:.2f}</td><td>{r['cp_efficiency']:.0f}%</td>"
            f"<td>{r['half_space_pct']:.0f}%</td><td>{r['aerial_wr_d']:.0f}%</td>"
            f"<td>{r['sb_wr']:.0f}%</td><td>{rd}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PitchPulse Progress Review {d0} – {d1}</title>
{_STYLE}</head><body>
<div style="border-bottom:1px solid var(--bd);padding-bottom:16px;margin-bottom:24px">
  <h1>Tiverton Town FC — Tactical Progress Review</h1>
  <div class="meta">
    <span>Window <b>{d0}</b> → <b>{d1}</b></span>
    <span>Games <b>{n}</b></span>
    <span>Generated <b>{now}</b></span>
  </div>
  <div class="chips">
    <span class="chip w">W {wins}</span><span class="chip d">D {draws}</span>
    <span class="chip l">L {losses}</span>
    <span class="chip d">GF {gf} &nbsp; GA {ga}</span>
  </div>
</div>
<h2>In Possession</h2>
<p>{_narrate_ip(df)}</p>
<img src="data:image/png;base64,{b64_ip}" class="plot" alt="In Possession Trends">
<h2>Press &amp; Line of Engagement</h2>
<p>{_narrate_press(df)}</p>
<img src="data:image/png;base64,{b64_press}" class="plot" alt="Press Trends">
<h2>Set Pieces</h2>
<p>{_narrate_sp(df)}</p>
<img src="data:image/png;base64,{b64_sp}" class="plot" alt="Set Piece Trends">
<h2>Non-League Physics</h2>
<p>{_narrate_nl(df)}</p>
<img src="data:image/png;base64,{b64_nl}" class="plot" alt="Non-League Physics Trends">
<h2>Game-by-Game Results Log</h2>
<table>
<thead><tr><th>Date</th><th>Opponent</th><th>Result</th><th>Score</th>
<th>LoE Idx</th><th>CP%</th><th>½-Space%</th><th>Aerial WR(D)</th>
<th>2nd Ball%</th><th>RD</th></tr></thead>
<tbody>{rows}</tbody></table>
<footer>PitchPulse · Tiverton Town FC · UEFA 4 Moments Framework · LOCAL &amp; OFFLINE</footer>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# DoF executive card
# ══════════════════════════════════════════════════════════════════════════════

_DOF_STYLE = """
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700;900&family=Inter:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#09090b;color:#e4e4e7;font-family:'Inter',system-ui,sans-serif;
     min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
.card{width:100%;max-width:680px;background:#111116;border:1px solid rgba(255,255,255,.08);border-radius:16px;overflow:hidden}
.head{background:linear-gradient(135deg,#111116,#1a1610);padding:22px 26px;border-bottom:1px solid rgba(245,158,11,.15)}
.head h1{font-family:'Barlow Condensed',Impact,sans-serif;font-size:1.5rem;font-weight:900;color:#f59e0b;letter-spacing:.04em;text-transform:uppercase;margin-bottom:4px}
.sub{font-size:.72rem;color:#71717a;letter-spacing:.06em;text-transform:uppercase}
.chips{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
.chip{font-family:'JetBrains Mono',monospace;font-size:.78rem;padding:3px 12px;border-radius:5px;font-weight:700}
.w{background:rgba(16,185,129,.15);color:#10b981;border:1px solid rgba(16,185,129,.3)}
.d{background:rgba(245,158,11,.12);color:#f59e0b;border:1px solid rgba(245,158,11,.25)}
.l{background:rgba(244,63,94,.12);color:#f43f5e;border:1px solid rgba(244,63,94,.25)}
.grid{display:grid;grid-template-columns:repeat(3,1fr)}
.cell{padding:18px 20px;border-right:1px solid rgba(255,255,255,.06);border-bottom:1px solid rgba(255,255,255,.06)}
.cell:nth-child(3n){border-right:none}
.lbl{font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;color:#71717a;font-weight:600;margin-bottom:6px}
.val{font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#f59e0b;line-height:1}
.trend{font-size:.7rem;margin-top:5px}
.up{color:#10b981}.dn{color:#f43f5e}.warn{color:#f59e0b}.neutral{color:#71717a}
.finding{padding:18px 26px}
.finding-lbl{font-size:.6rem;letter-spacing:.12em;text-transform:uppercase;color:#f43f5e;font-weight:700;margin-bottom:7px}
.finding-txt{font-size:.8rem;color:#a1a1aa;line-height:1.55}
footer{padding:12px 26px;font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#3f3f46;border-top:1px solid rgba(255,255,255,.04)}
</style>"""


def render_dof_card(df, snaps: list[MatchSnapshot]) -> str:
    n      = len(df)
    wins   = sum(1 for s in snaps if s.result.upper().startswith("W"))
    draws  = sum(1 for s in snaps if s.result.upper().startswith("D"))
    losses = sum(1 for s in snaps if s.result.upper().startswith("L"))
    gf     = sum(s.tiv_goals for s in snaps)
    ga     = sum(s.opp_goals  for s in snaps)
    d0     = df["match_date"].iloc[0]  if not df.empty else "?"
    d1     = df["match_date"].iloc[-1] if not df.empty else "?"

    avg_hs   = df["half_space_pct"].mean()
    avg_cp   = df["cp_efficiency"].mean()
    avg_sb   = df["sb_wr"].mean()
    avg_loe  = df["loe_index"].mean()
    avg_wr_d = df["aerial_wr_d"].mean()
    flagged  = int(df["rest_defense_flagged"].sum())

    loe_lbl = "High Block" if avg_loe >= 1.5 else "Mid Block" if avg_loe >= 0.8 else "Low Block"

    findings: list[str] = []
    if flagged >= 2:
        findings.append(f"Set piece Rest Defense flagged in {flagged}/{n} games — dead-ball defensive shape requires a dedicated training block.")
    if avg_cp < 35:
        findings.append(f"Counter-Pressing efficiency at {avg_cp:.0f}% — critically low. Defensive Transition phase is being exploited.")
    if avg_wr_d < 50:
        findings.append(f"Defensive-third aerial win rate at {avg_wr_d:.0f}% — direct balls over the back line are a live threat.")
    if avg_sb < 40:
        findings.append(f"Second-ball recovery at {avg_sb:.0f}% — systemic weakness in direct-play phases.")
    if avg_hs < 25:
        findings.append(f"Half-space penetration at {avg_hs:.0f}% — inside channels consistently underexploited.")
    if not findings:
        findings.append("Tactical metrics within acceptable ranges across the window. Maintain current principles.")
    finding_html = " ".join(findings[:2])

    def cell(lbl, val_html, trend_html, trend_cls):
        return (f'<div class="cell"><div class="lbl">{lbl}</div>'
                f'<div class="val">{val_html}</div>'
                f'<div class="trend {trend_cls}">{trend_html}</div></div>')

    hs_cls   = "up" if avg_hs >= 30 else "warn" if avg_hs >= 22 else "dn"
    hs_trend = "↑ QS threshold met" if avg_hs >= 30 else "↓ below 30%" if avg_hs < 22 else "→ approaching"
    cp_cls   = "up" if avg_cp >= 50 else "warn" if avg_cp >= 35 else "dn"
    cp_trend = "↑ solid" if avg_cp >= 50 else "⚠ alert zone" if avg_cp < 35 else "→ stable"
    sb_cls   = "up" if avg_sb >= 60 else "warn" if avg_sb >= 40 else "dn"
    sb_trend = "↑ dominant" if avg_sb >= 60 else "⚠ risk zone" if avg_sb < 40 else "→ contested"
    rd_col   = "#f43f5e" if flagged >= 2 else "#f59e0b" if flagged == 1 else "#10b981"
    rd_cls   = "dn" if flagged >= 2 else "warn" if flagged == 1 else "up"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PitchPulse DoF Card</title>
{_DOF_STYLE}</head><body>
<div class="card">
  <div class="head">
    <h1>Tiverton Town FC — {n}-Game Review</h1>
    <div class="sub">{d0} → {d1}</div>
    <div class="chips">
      <span class="chip w">W {wins}</span><span class="chip d">D {draws}</span>
      <span class="chip l">L {losses}</span>
      <span class="chip d">GF {gf} &nbsp; GA {ga}</span>
    </div>
  </div>
  <div class="grid">
    {cell("Half-Space %",       f"{avg_hs:.0f}%",   hs_trend, hs_cls)}
    {cell("CP Efficiency",      f"{avg_cp:.0f}%",   cp_trend, cp_cls)}
    {cell("2nd Ball Recovery",  f"{avg_sb:.0f}%",   sb_trend, sb_cls)}
    {cell("LoE Index",          f"{avg_loe:.2f}",   loe_lbl,  "neutral")}
    {cell("Aerial WR D-third",  f"{avg_wr_d:.0f}%",
          "↑ solid" if avg_wr_d >= 50 else "↓ vulnerability",
          "up" if avg_wr_d >= 50 else "dn")}
    {cell("Rest Defense Flags",
          f'<span style="color:{rd_col}">{flagged}/{n}</span>',
          "⚠ priority" if flagged >= 2 else "within range",
          rd_cls)}
  </div>
  <div class="finding">
    <div class="finding-lbl">Key Finding &amp; Priority</div>
    <div class="finding-txt">{finding_html}</div>
  </div>
  <footer>PitchPulse · UEFA 4 Moments Framework · LOCAL &amp; OFFLINE · {datetime.now().strftime('%Y-%m-%d')}</footer>
</div>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# ReviewResult + generate_review
# ══════════════════════════════════════════════════════════════════════════════

ReviewResult = namedtuple("ReviewResult", [
    "df", "snaps", "html_report", "dof_card_html",
    "html_report_path", "dof_card_path",
])


def generate_review(
    ledger_paths: list[Path],
    proc_dir: Path = PROC_DIR,
) -> ReviewResult:
    """
    Build a full rolling review from a list of ledger Paths.
    Figures are generated inside render_html_report() and closed after b64 conversion.
    For Streamlit st.pyplot(), call fig_in_possession(df) etc. directly.
    """
    if not ledger_paths:
        raise ValueError(
            "No ledger files supplied. Expected pattern: ledger_DD-MM-YYYY.json (e.g. ledger_15-08-2026.json)"
        )

    snaps = [extract_per_game_metrics(p) for p in ledger_paths]
    df    = aggregate_snapshots(snaps)

    html_report   = render_html_report(df, snaps)
    dof_card_html = render_dof_card(df, snaps)

    today     = datetime.now().strftime("%Y-%m-%d")
    html_path = proc_dir / f"progress_review_{today}.html"
    dof_path  = proc_dir / f"progress_review_{today}_dof_card.html"

    proc_dir.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_report,   encoding="utf-8")
    dof_path.write_text(dof_card_html,  encoding="utf-8")

    print(f"  [REVIEW] {html_path.name}")
    print(f"  [REVIEW] {dof_path.name}")

    return ReviewResult(
        df=df, snaps=snaps,
        html_report=html_report, dof_card_html=dof_card_html,
        html_report_path=html_path, dof_card_path=dof_path,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _cli() -> None:
    parser = argparse.ArgumentParser(description="PitchPulse Tactical Progress Review")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--last", type=int, default=6, metavar="N",
                     help="N most recent ledger files (default: 6)")
    grp.add_argument("--from", dest="date_from", metavar="DD-MM-YYYY",
                     help="Start date in UK format, e.g. 15-08-2026")
    parser.add_argument("--to", dest="date_to",  metavar="DD-MM-YYYY",
                        help="End date in UK format, e.g. 06-09-2026")
    args = parser.parse_args()

    paths = discover_ledgers(PROC_DIR, last_n=args.last,
                             date_from=args.date_from, date_to=args.date_to)
    if not paths:
        print("⚠  No date-stamped ledger files found in data/processed/")
        print("   Expected: data/processed/ledger_DD-MM-YYYY.json (e.g. ledger_15-08-2026.json)")
        print("   Tip: after each matchday, copy match_ledger.json to a date-stamped name.")
        sys.exit(0)

    print(f"\n{'═'*56}")
    print("  PITCHPULSE — Tactical Progress Review")
    print(f"{'═'*56}")
    for p in paths:
        print(f"    {p.name}")
    print(f"{'─'*56}")

    result = generate_review(paths, PROC_DIR)

    print(f"{'─'*56}")
    if not result.df.empty:
        df = result.df
        wins   = sum(1 for s in result.snaps if s.result.upper().startswith("W"))
        draws  = sum(1 for s in result.snaps if s.result.upper().startswith("D"))
        losses = sum(1 for s in result.snaps if s.result.upper().startswith("L"))
        print(f"  Games    : {len(df)}")
        print(f"  Record   : W{wins} D{draws} L{losses}")
        print(f"  CP Eff   : {df['cp_efficiency'].mean():.0f}%")
        print(f"  ½-Space  : {df['half_space_pct'].mean():.0f}%")
        print(f"  2nd Ball : {df['sb_wr'].mean():.0f}%")
    print(f"{'═'*56}\n")


if __name__ == "__main__":
    _cli()
