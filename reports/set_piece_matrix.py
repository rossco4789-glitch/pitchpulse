"""
reports/set_piece_matrix.py
Set-Piece Target Matrix for Tiverton Town FC.

Extracts ATT_CORNER and DEF_CORNER events from a match ledger, infers
first-contact outcome and sequence result from the surrounding event
sequence, then renders a side-by-side penalty-area scatter plot.

Usage (standalone smoke test):
    python reports/set_piece_matrix.py

    from reports.set_piece_matrix import plot_set_piece_matrix, compute_kpis, extract_corners
    path = plot_set_piece_matrix(ledger)   # → data/processed/plots/set_piece_matrix.png
    kpis = compute_kpis(corners)

Coordinate contract:
    All coordinates in metres on a 105 × 68 m pitch.
    DEF corners are mirrored (x → 105 − x, y → 68 − y) for the display view
    so both panels share the same attacking-end orientation.

Output: data/processed/plots/set_piece_matrix.png @ 150 DPI
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLOTS_DIR = ROOT / "data" / "processed" / "plots"

# ── Colour palette (mirrors visualizer.py) ─────────────────────────────────
OLED_BG  = "#09090b"
SURFACE  = "#18181b"
LINE     = "#3f3f46"
GOLD     = "#f59e0b"
CRIMSON  = "#f43f5e"
ZINC     = "#71717a"
EMERALD  = "#10b981"
PURPLE   = "#a855f7"
CYAN     = "#22d3ee"
TEXT     = "#e4e4e7"
TEXT_DIM = "#52525b"

# ── Pitch constants (105 × 68 m FIFA) ──────────────────────────────────────
# 18-yard box (attacking end, x > 88.5)
BOX_X0, BOX_X1 = 88.5, 105.0
BOX_Y0, BOX_Y1 = 13.84, 54.16
# 6-yard box
SIX_X0 = 99.5
SIX_Y0, SIX_Y1 = 25.84, 42.16
# Penalty spot
PEN_X, PEN_Y = 93.0, 34.0
# Goal mouth
GOAL_Y0, GOAL_Y1 = 30.34, 37.66

# ── Delivery zone definitions (evaluated left-to-right; first match wins) ──
# All coordinates assume attacking direction = increasing x.
# fmt: off
_ZONES: list[tuple[str, object]] = [
    ("Near Post",            lambda x, y: x >= SIX_X0 and (y < 29.0 or y > 39.0)),
    ("Central / Six-Yard",   lambda x, y: x >= SIX_X0 and 29.0 <= y <= 39.0),
    ("Penalty Spot / 12-Yd", lambda x, y: 91.0 <= x < SIX_X0 and 26.5 <= y <= 41.5),
    ("Back Post",            lambda x, y: 91.0 <= x < SIX_X0 and (y < 26.5 or y > 41.5)),
    ("Edge / Cutback",       lambda x, y: 87.0 <= x < 91.0 and 18.0 <= y <= 50.0),
    ("Second Ball",          lambda x, y: True),  # catch-all
]
# fmt: on

# Zone display colours (subtle background fill)
_ZONE_FILL: dict[str, str] = {
    "Near Post":            "#f59e0b",  # amber
    "Central / Six-Yard":   "#a855f7",  # purple
    "Penalty Spot / 12-Yd": "#f43f5e",  # crimson
    "Back Post":            "#22d3ee",  # cyan
    "Edge / Cutback":       "#10b981",  # emerald
    "Second Ball":          "#71717a",  # zinc
}
_ZONE_ALPHA = 0.10

# ── First-contact outcome colours ──────────────────────────────────────────
_FC_COLOUR: dict[str, str] = {
    "Won (Tivvy)":               GOLD,
    "Lost (Opponent Clearance)": CRIMSON,
    "Flick-on / Uncontested":    ZINC,
}

# ── Sequence-outcome markers ────────────────────────────────────────────────
_SEQ_MARKER: dict[str, str] = {
    "Shot On Target":     "*",   # star
    "Shot Off Target":    "X",   # cross
    "Counter Conceded":   "v",   # down-triangle
    "Turnover / Recycled": "o",  # circle
}


# ══════════════════════════════════════════════════════════════════════════════
# Zone helpers
# ══════════════════════════════════════════════════════════════════════════════

def classify_delivery_zone(x_m: float | None, y_m: float | None) -> str:
    """Return the tactical delivery cluster for a coordinate pair."""
    if x_m is None or y_m is None:
        return "Second Ball"
    for name, test in _ZONES:
        if test(x_m, y_m):
            return name
    return "Second Ball"


# ══════════════════════════════════════════════════════════════════════════════
# Sequence inference helpers
# ══════════════════════════════════════════════════════════════════════════════

def _all_ledger_events(ledger: dict) -> list[dict]:
    """Flatten all tagged events from a match ledger, sorted by time."""
    events: list[dict] = []
    for m in ledger.get("matched", []):
        events.append(m.get("tag", {}))
    events += ledger.get("unmatched_tags", [])
    events += ledger.get("opponent_events", [])
    return sorted(events, key=lambda e: e.get("match_seconds", 0))


def _infer_first_contact(
    corner: dict,
    all_events: list[dict],
    window_s: int = 8,
) -> str:
    """
    Infer first-contact outcome from the next event in a time window.

    For ATT corners (Tivvy delivering):
        Next Tivvy SHOT/BOX_ENTRY/AERIAL_DUEL → "Won (Tivvy)"
        Next Opp HIGH_REGAIN/DEF_TURNOVER     → "Lost (Opponent Clearance)"
    For DEF corners (Opponent delivering):
        Next Tivvy HIGH_REGAIN/DEF_TURNOVER   → "Won (Tivvy)"
        Next Opp SHOT/BOX_ENTRY               → "Lost (Opponent Clearance)"
    """
    t0 = corner.get("match_seconds", 0)
    is_att = corner.get("_type") == "ATT"

    window = [
        e for e in all_events
        if t0 < e.get("match_seconds", 0) <= t0 + window_s
        and e.get("event_type") != "SET_PIECE"
    ]
    if not window:
        return "Flick-on / Uncontested"

    nxt = min(window, key=lambda e: e.get("match_seconds", 0))
    et = nxt.get("event_type", "")
    team = nxt.get("team", "tiverton")

    if is_att:
        if team == "tiverton" and et in ("SHOT", "BOX_ENTRY", "AERIAL_DUEL"):
            return "Won (Tivvy)"
        if team == "opponent" and et in ("HIGH_REGAIN", "DEF_TURNOVER"):
            return "Lost (Opponent Clearance)"
    else:
        if team == "tiverton" and et in ("HIGH_REGAIN", "DEF_TURNOVER", "AERIAL_DUEL"):
            return "Won (Tivvy)"
        if team == "opponent" and et in ("SHOT", "BOX_ENTRY"):
            return "Lost (Opponent Clearance)"

    return "Flick-on / Uncontested"


def _infer_sequence_outcome(
    corner: dict,
    all_events: list[dict],
    window_s: int = 12,
) -> str:
    """
    Infer what happened in the 12s sequence after the corner.

    For ATT corners: shot by Tivvy = good; opp regain = counter conceded.
    For DEF corners: opp shot = bad (returned as "Shot On/Off Target");
                     Tivvy regain → Turnover / Recycled (we cleared).
    """
    t0 = corner.get("match_seconds", 0)
    is_att = corner.get("_type") == "ATT"

    window = sorted(
        [e for e in all_events
         if t0 < e.get("match_seconds", 0) <= t0 + window_s
         and e.get("event_type") != "SET_PIECE"],
        key=lambda e: e.get("match_seconds", 0),
    )

    for ev in window:
        et = ev.get("event_type", "")
        st = ev.get("sub_type") or ""
        team = ev.get("team", "tiverton")

        if is_att:
            if et == "SHOT" and team == "tiverton":
                return "Shot On Target" if st == "ON_TARGET" else "Shot Off Target"
            if et in ("HIGH_REGAIN", "DEF_TURNOVER") and team == "opponent":
                return "Counter Conceded"
        else:
            if et == "SHOT" and team == "opponent":
                return "Shot On Target" if st == "ON_TARGET" else "Shot Off Target"
            if et in ("HIGH_REGAIN", "DEF_TURNOVER") and team == "tiverton":
                return "Turnover / Recycled"

    return "Turnover / Recycled"


# ══════════════════════════════════════════════════════════════════════════════
# Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_corners(ledger: dict) -> list[dict]:
    """
    Return a list of enriched corner dicts from the ledger.

    Each dict carries:
        _type          : "ATT" or "DEF"
        x_m, y_m      : display coordinates (DEF mirrored to attacking view)
        x_raw, y_raw  : original coordinates from the tagger
        match_seconds  : event time
        first_contact  : inferred first-contact outcome
        sequence_outcome: inferred sequence result
        delivery_zone  : one of the 6 tactical clusters
    """
    all_events = _all_ledger_events(ledger)
    corners: list[dict] = []

    for ev in all_events:
        if ev.get("event_type") != "SET_PIECE":
            continue
        sub = ev.get("sub_type", "")
        if sub not in ("ATT_CORNER", "DEF_CORNER"):
            continue

        is_att = sub == "ATT_CORNER"
        x_raw = ev.get("x_m")
        y_raw = ev.get("y_m")

        # Mirror DEF corners to attacking-end view
        if not is_att and x_raw is not None and y_raw is not None:
            x_disp = 105.0 - x_raw
            y_disp = 68.0 - y_raw
        else:
            x_disp = x_raw
            y_disp = y_raw

        c: dict = {
            "_type":    "ATT" if is_att else "DEF",
            "x_m":      x_disp,
            "y_m":      y_disp,
            "x_raw":    x_raw,
            "y_raw":    y_raw,
            "match_seconds": ev.get("match_seconds", 0),
            "delivery_zone": classify_delivery_zone(x_disp, y_disp),
        }
        c["first_contact"]    = _infer_first_contact(c, all_events)
        c["sequence_outcome"] = _infer_sequence_outcome(c, all_events)
        corners.append(c)

    return corners


# ══════════════════════════════════════════════════════════════════════════════
# KPI computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_kpis(corners: list[dict]) -> dict:
    """
    Return a summary KPI dict.

        att_corners          : int
        def_corners          : int
        att_fc_win_pct       : float  (% ATT corners where Tivvy won first contact)
        def_clearance_pct    : float  (% DEF corners where Tivvy cleared first)
        att_shots_generated  : int    (ATT corners that produced a Tivvy shot)
        att_shots_on_target  : int
        def_shots_conceded   : int    (DEF corners that produced an opp shot)
        att_box_entry_pct    : float  (proxy: % ATT with Won 1st contact)
    """
    att = [c for c in corners if c["_type"] == "ATT"]
    def_ = [c for c in corners if c["_type"] == "DEF"]

    def _pct(num: int, den: int) -> float:
        return round(num / den * 100, 1) if den else 0.0

    att_won = sum(1 for c in att if c["first_contact"] == "Won (Tivvy)")
    def_won = sum(1 for c in def_ if c["first_contact"] == "Won (Tivvy)")

    att_shots = sum(
        1 for c in att
        if c["sequence_outcome"] in ("Shot On Target", "Shot Off Target")
    )
    att_on_tgt = sum(
        1 for c in att if c["sequence_outcome"] == "Shot On Target"
    )
    def_shots = sum(
        1 for c in def_
        if c["sequence_outcome"] in ("Shot On Target", "Shot Off Target")
    )

    return {
        "att_corners":         len(att),
        "def_corners":         len(def_),
        "att_fc_win_pct":      _pct(att_won, len(att)),
        "def_clearance_pct":   _pct(def_won, len(def_)),
        "att_shots_generated": att_shots,
        "att_shots_on_target": att_on_tgt,
        "def_shots_conceded":  def_shots,
        "att_box_entry_pct":   _pct(att_won, len(att)),  # first-contact proxy
    }


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def _draw_penalty_area(ax: plt.Axes) -> None:
    """Draw goal, 18yd box, 6yd box, penalty spot on a zoomed axes."""
    kw = dict(linewidth=0.9, zorder=2)

    # 18-yard box
    ax.add_patch(mpatches.Rectangle(
        (BOX_X0, BOX_Y0), BOX_X1 - BOX_X0, BOX_Y1 - BOX_Y0,
        fill=False, edgecolor=LINE, **kw,
    ))
    # 6-yard box
    ax.add_patch(mpatches.Rectangle(
        (SIX_X0, SIX_Y0), BOX_X1 - SIX_X0, SIX_Y1 - SIX_Y0,
        fill=False, edgecolor=LINE, **kw,
    ))
    # Goal mouth
    ax.add_patch(mpatches.Rectangle(
        (BOX_X1, GOAL_Y0), 2.4, GOAL_Y1 - GOAL_Y0,
        fill=False, edgecolor=LINE, linewidth=1.4, zorder=2,
    ))
    # Penalty spot
    ax.scatter(PEN_X, PEN_Y, s=18, color=LINE, zorder=3)
    # Penalty arc (portion visible in zoomed view)
    arc = mpatches.Arc(
        (PEN_X, PEN_Y), 18.3, 18.3,
        angle=0, theta1=308, theta2=52,
        color=LINE, linewidth=0.9, zorder=2,
    )
    ax.add_patch(arc)


def _draw_zones(ax: plt.Axes) -> None:
    """Overlay semi-transparent zone rectangles with short labels."""
    zone_boxes: list[tuple[str, float, float, float, float]] = [
        # (name,      x0,    y0,    width, height)
        ("Near Post (L)",    SIX_X0, BOX_Y0, BOX_X1 - SIX_X0, SIX_Y0 - BOX_Y0),
        ("Near Post (R)",    SIX_X0, SIX_Y1, BOX_X1 - SIX_X0, BOX_Y1 - SIX_Y1),
        ("Central / 6-Yd",  SIX_X0, SIX_Y0, BOX_X1 - SIX_X0, SIX_Y1 - SIX_Y0),
        ("Pen Spot",         91.0,   26.5,   SIX_X0 - 91.0,   41.5 - 26.5),
        ("Back Post (L)",    91.0,   BOX_Y0, SIX_X0 - 91.0,   26.5 - BOX_Y0),
        ("Back Post (R)",    91.0,   41.5,   SIX_X0 - 91.0,   BOX_Y1 - 41.5),
        ("Edge / Cutback",   87.0,   18.0,   4.0,              32.0),
    ]
    zone_colour_map = {
        "Near Post (L)":   _ZONE_FILL["Near Post"],
        "Near Post (R)":   _ZONE_FILL["Near Post"],
        "Central / 6-Yd":  _ZONE_FILL["Central / Six-Yard"],
        "Pen Spot":        _ZONE_FILL["Penalty Spot / 12-Yd"],
        "Back Post (L)":   _ZONE_FILL["Back Post"],
        "Back Post (R)":   _ZONE_FILL["Back Post"],
        "Edge / Cutback":  _ZONE_FILL["Edge / Cutback"],
    }
    for name, x0, y0, w, h in zone_boxes:
        colour = zone_colour_map[name]
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), w, h,
            facecolor=colour, alpha=_ZONE_ALPHA,
            edgecolor="none", zorder=1,
        ))


def _zone_pct_labels(ax: plt.Axes, corners: list[dict]) -> None:
    """Annotate each zone with its percentage share."""
    n = len(corners)
    if n == 0:
        return
    from collections import Counter
    counts = Counter(c["delivery_zone"] for c in corners)
    label_positions: dict[str, tuple[float, float]] = {
        "Near Post":            (102.0, 22.0),
        "Central / Six-Yard":   (102.0, 34.0),
        "Penalty Spot / 12-Yd": (95.0,  34.0),
        "Back Post":            (95.0,  20.0),
        "Edge / Cutback":       (88.8,  34.0),
        "Second Ball":          (89.0,  56.0),
    }
    for zone, (lx, ly) in label_positions.items():
        pct = counts.get(zone, 0) / n * 100
        if pct < 1:
            continue
        fill = _ZONE_FILL.get(zone, ZINC)
        ax.text(
            lx, ly, f"{pct:.0f}%",
            ha="center", va="center",
            fontsize=7.5, fontweight="black",
            color=fill, alpha=0.85, zorder=7,
        )


def _scatter_corners(ax: plt.Axes, corners: list[dict]) -> None:
    """Plot corners as colour-coded (first contact) × shape-coded (outcome) points."""
    rng = np.random.default_rng(7)
    for c in corners:
        x, y = c.get("x_m"), c.get("y_m")
        if x is None or y is None:
            x, y = 93.0, 34.0  # penalty spot fallback
        # Small jitter to separate stacked points
        xj = float(x) + rng.uniform(-0.5, 0.5)
        yj = float(y) + rng.uniform(-0.5, 0.5)
        colour = _FC_COLOUR.get(c["first_contact"], ZINC)
        marker = _SEQ_MARKER.get(c["sequence_outcome"], "o")
        ax.scatter(
            xj, yj,
            marker=marker,
            color=colour,
            s=220 if marker == "*" else 150,
            edgecolors="white",
            linewidths=0.5,
            alpha=0.92,
            zorder=6,
        )


def _no_data_label(ax: plt.Axes, msg: str) -> None:
    ax.text(
        96.0, 34.0, msg,
        ha="center", va="center",
        fontsize=10, color=TEXT_DIM, alpha=0.45, fontstyle="italic",
    )


def _panel_legends(ax: plt.Axes) -> None:
    """Compact dual legend: first contact (colour) + sequence outcome (shape)."""
    fc_handles = [
        mlines.Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=col, markeredgecolor="white",
            markersize=8, markeredgewidth=0.5,
            label=label,
        )
        for label, col in _FC_COLOUR.items()
    ]
    seq_handles = [
        mlines.Line2D(
            [0], [0], marker=mk, color="none",
            markerfacecolor=TEXT_DIM, markeredgecolor="white",
            markersize=8 if mk == "*" else 7, markeredgewidth=0.5,
            label=label,
        )
        for label, mk in _SEQ_MARKER.items()
    ]
    all_handles = fc_handles + seq_handles
    leg = ax.legend(
        handles=all_handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        framealpha=0.30,
        facecolor=SURFACE,
        edgecolor=LINE,
        fontsize=6.5,
        ncol=1,
    )
    for txt in leg.get_texts():
        txt.set_color(TEXT)


def _brand(ax: plt.Axes, subtitle: str) -> None:
    ax.text(
        0.012, 0.995, "TIVERTON TOWN FC",
        transform=ax.transAxes,
        fontsize=6, fontweight="black", color=GOLD,
        ha="left", va="top", alpha=0.65, fontfamily="monospace",
    )
    ax.text(
        0.012, 0.963, subtitle,
        transform=ax.transAxes,
        fontsize=5.5, fontweight="bold", color=TEXT_DIM,
        ha="left", va="top", alpha=0.50, fontfamily="monospace",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main render function
# ══════════════════════════════════════════════════════════════════════════════

def plot_set_piece_matrix(
    ledger: dict,
    output_path: Path | None = None,
) -> tuple[Path, list[dict]]:
    """
    Render the Set-Piece Target Matrix and save to *output_path*.

    Returns (path, corners_list) so callers can chain compute_kpis().
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or PLOTS_DIR / "set_piece_matrix.png"

    corners = extract_corners(ledger)
    att = [c for c in corners if c["_type"] == "ATT"]
    def_ = [c for c in corners if c["_type"] == "DEF"]

    fig, (ax_att, ax_def) = plt.subplots(1, 2, figsize=(16, 7))
    fig.set_facecolor(OLED_BG)
    fig.suptitle(
        "Set-Piece Target Matrix  ·  Corner Delivery Analysis",
        color=TEXT, fontsize=12, fontweight="bold", y=0.97,
    )

    # ── Panel config ──────────────────────────────────────────────────────
    panels = [
        (ax_att, att,  "ATT CORNERS (Tivvy Delivering)", f"{len(att)} event{'s' if len(att) != 1 else ''}"),
        (ax_def, def_, "DEF CORNERS (Opponent Delivering)", f"{len(def_)} event{'s' if len(def_) != 1 else ''}"),
    ]

    for ax, group, title, count_label in panels:
        ax.set_facecolor(OLED_BG)

        # Pitch lines (manual — no mplsoccer dependency for sub-axes)
        _draw_penalty_area(ax)
        _draw_zones(ax)

        # Goal line
        ax.axvline(x=105, color=LINE, linewidth=1.2, zorder=2)

        # Pitch boundary lines in view
        ax.plot([84, 105], [0, 0],       color=LINE, linewidth=0.7, zorder=2)
        ax.plot([84, 105], [68, 68],     color=LINE, linewidth=0.7, zorder=2)
        ax.plot([84, 84],  [0, 68],      color=LINE, linewidth=0.7,
                linestyle="--", alpha=0.35, zorder=2)

        if group:
            _scatter_corners(ax, group)
            _zone_pct_labels(ax, group)
        else:
            _no_data_label(ax, "No corners tagged")

        _panel_legends(ax)
        _brand(ax, f"PitchPulse · Dead Ball · {count_label}")

        ax.set_xlim(84, 107.5)
        ax.set_ylim(-1, 69)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, color=TEXT, fontsize=9.5, fontweight="bold", pad=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=OLED_BG)
    plt.close(fig)
    kb = out.stat().st_size // 1024
    print(f"  [PLOT] {out.name:<35}  {kb} KB")
    return out, corners


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════════════

_MOCK_LEDGER: dict = {
    "matched": [],
    "unmatched_tags": [
        # ATT corners
        {"id": "c1", "event_type": "SET_PIECE", "sub_type": "ATT_CORNER",
         "team": "tiverton", "match_seconds": 300,
         "x_m": 102.0, "y_m": 31.0, "period": "1H"},
        {"id": "c1b", "event_type": "SHOT", "sub_type": "ON_TARGET",
         "team": "tiverton", "match_seconds": 305,
         "x_m": 96.0, "y_m": 34.0, "period": "1H"},
        {"id": "c2", "event_type": "SET_PIECE", "sub_type": "ATT_CORNER",
         "team": "tiverton", "match_seconds": 900,
         "x_m": 96.0, "y_m": 29.0, "period": "1H"},
        {"id": "c2b", "event_type": "HIGH_REGAIN", "sub_type": None,
         "team": "opponent", "match_seconds": 906,
         "x_m": 78.0, "y_m": 40.0, "period": "1H"},
        {"id": "c3", "event_type": "SET_PIECE", "sub_type": "ATT_CORNER",
         "team": "tiverton", "match_seconds": 1500,
         "x_m": 100.5, "y_m": 36.5, "period": "1H"},
        {"id": "c4", "event_type": "SET_PIECE", "sub_type": "ATT_CORNER",
         "team": "tiverton", "match_seconds": 2100,
         "x_m": 89.5, "y_m": 22.0, "period": "1H"},
        # DEF corners
        {"id": "c5", "event_type": "SET_PIECE", "sub_type": "DEF_CORNER",
         "team": "opponent", "match_seconds": 1200,
         "x_m": 3.0, "y_m": 14.0, "period": "1H"},
        {"id": "c5b", "event_type": "SHOT", "sub_type": "OFF_TARGET",
         "team": "opponent", "match_seconds": 1207,
         "x_m": 8.0, "y_m": 33.0, "period": "1H"},
        {"id": "c6", "event_type": "SET_PIECE", "sub_type": "DEF_CORNER",
         "team": "opponent", "match_seconds": 2700,
         "x_m": 5.0, "y_m": 55.0, "period": "1H"},
        {"id": "c6b", "event_type": "DEF_TURNOVER", "sub_type": None,
         "team": "tiverton", "match_seconds": 2704,
         "x_m": 20.0, "y_m": 34.0, "period": "1H"},
    ],
    "opponent_events": [],
    "substitutions": [],
}


def _smoke_test() -> None:
    col = 62
    print()
    print("═" * col)
    print("  SET-PIECE TARGET MATRIX — Smoke Test")
    print("═" * col)

    path, corners = plot_set_piece_matrix(_MOCK_LEDGER)
    kpis = compute_kpis(corners)

    att = [c for c in corners if c["_type"] == "ATT"]
    def_ = [c for c in corners if c["_type"] == "DEF"]
    print(f"  ATT corners : {len(att)}")
    print(f"  DEF corners : {len(def_)}")
    print(f"  ATT 1st-contact win rate : {kpis['att_fc_win_pct']}%")
    print(f"  DEF clearance rate       : {kpis['def_clearance_pct']}%")
    print(f"  Shots generated (ATT)    : {kpis['att_shots_generated']}")
    print(f"  Shots conceded  (DEF)    : {kpis['def_shots_conceded']}")
    print("─" * col)

    kb = path.stat().st_size // 1024
    ok = kb >= 10
    print(f"  {'✓' if ok else '⚠'} {path.name:<38} {kb:>4} KB")
    print()

    # Empty ledger should not crash
    path2, corners2 = plot_set_piece_matrix({}, output_path=path.parent / "sp_empty.png")
    assert len(corners2) == 0
    print(f"  ✓ Empty ledger handled gracefully ({path2.name})")
    path2.unlink(missing_ok=True)

    print("═" * col)
    print(f"  {'PASSED ✓' if ok else 'FAILED ⚠'}")
    print("═" * col)
    print()


if __name__ == "__main__":
    _smoke_test()
