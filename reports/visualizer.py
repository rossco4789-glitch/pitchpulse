"""
reports/visualizer.py
OLED-dark tactical pitch visualization engine for Tiverton Town FC.
Uses mplsoccer + matplotlib. Non-interactive Agg backend — safe for scripts.

Usage:
    python reports/visualizer.py          # smoke-test with 14 mock events

    import reports.visualizer as viz
    viz.plot_shot_map(events)
    viz.plot_transition_map(events)
    viz.plot_zonal_heatmap(events)

Coordinate contract:
    Events with 'x', 'y' keys are plotted precisely.
    Events with only 'zone_id' fall back to zone centroid coordinates.
    Missing / empty datasets produce a labelled blank figure, not an error.

Output: data/processed/plots/{shot_map,transition_map,zonal_heatmap}.png @ 150 DPI
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # non-interactive; must precede pyplot import

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch

# ── Project root on sys.path (allows direct script execution) ──────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.zones import ZONES, ZONE_ORDER, get_zone_by_coords, get_zone_centroid

# ── Output directory ───────────────────────────────────────────────────────────
PLOTS_DIR = ROOT / "data" / "processed" / "plots"

# ── OLED colour palette (matches PitchPulse tagger) ───────────────────────────
OLED_BG  = "#09090b"
SURFACE  = "#18181b"
LINE     = "#3f3f46"
GOLD     = "#f59e0b"
CRIMSON  = "#f43f5e"
ORANGE   = "#fb923c"
ZINC     = "#71717a"
EMERALD  = "#10b981"
CYAN     = "#22d3ee"
TEXT     = "#e4e4e7"
TEXT_DIM = "#52525b"

# ── Shared mplsoccer pitch keyword arguments ───────────────────────────────────
_PITCH_KW: dict = dict(
    pitch_type="custom",
    pitch_length=105,
    pitch_width=68,
    pitch_color=OLED_BG,
    line_color=LINE,
    line_zorder=2,
    goal_type="box",
)


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_plots_dir() -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR


def _resolve_coords(event: dict) -> tuple[float, float] | None:
    """
    Return (x, y) pitch coordinates for an event.
    Priority: explicit x/y → zone_id centroid → None.
    """
    x = event.get("x")
    y = event.get("y")
    if x is not None and y is not None:
        try:
            return float(x), float(y)
        except (TypeError, ValueError):
            pass
    zone_id = event.get("zone_id")
    if zone_id:
        return get_zone_centroid(zone_id)
    return None


def _dark_pitch(figsize: tuple[float, float] = (13, 8.1)) -> tuple:
    """
    Draw a dark-themed pitch.
    Returns (fig, ax, pitch_obj).
    """
    pitch = Pitch(**_PITCH_KW)
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor(OLED_BG)
    ax.set_facecolor(OLED_BG)
    return fig, ax, pitch


def _brand_stamp(ax: plt.Axes, subtitle: str) -> None:
    """Tivvy watermark in the top-left corner of an axes."""
    ax.text(
        0.012, 0.988, "TIVERTON TOWN FC",
        transform=ax.transAxes,
        fontsize=6.5, fontweight="black", color=GOLD,
        ha="left", va="top", alpha=0.65, fontfamily="monospace",
    )
    ax.text(
        0.012, 0.956, subtitle,
        transform=ax.transAxes,
        fontsize=5.8, fontweight="bold", color=TEXT_DIM,
        ha="left", va="top", alpha=0.55, fontfamily="monospace",
    )


def _no_data_label(ax: plt.Axes, msg: str = "No data") -> None:
    ax.text(
        52.5, 34, msg,
        ha="center", va="center",
        fontsize=13, color=TEXT_DIM, alpha=0.45, fontstyle="italic",
    )


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> Path:
    """Save figure at dpi, close it, and print confirmation."""
    fig.savefig(
        path, dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    kb = path.stat().st_size // 1024
    print(f"  [PLOT] {path.name:<30}  {kb} KB")
    return path


def _legend(ax: plt.Axes, handles: list) -> None:
    """Attach a dark-styled legend to an axes."""
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.92),
        framealpha=0.28,
        facecolor=SURFACE,
        edgecolor=LINE,
        fontsize=8,
    )
    for text in leg.get_texts():
        text.set_color(TEXT)


# ══════════════════════════════════════════════════════════════════════════════
# Shot map
# ══════════════════════════════════════════════════════════════════════════════

_SHOT_STYLES: dict[str, dict] = {
    "ON_TARGET":  dict(marker="*", color=CRIMSON, s=300, label="On Target",  zorder=6),
    "OFF_TARGET": dict(marker="X", color=ORANGE,  s=160, label="Off Target", zorder=6),
    "BLOCKED":    dict(marker="D", color=ZINC,    s=130, label="Blocked",    zorder=6),
}


def plot_shot_map(
    events:      list[dict],
    output_path: Path | None = None,
) -> Path:
    """
    Render a shot map: ★ on-target | ✕ off-target | ◆ blocked.
    Attacking direction is left → right (x increases).
    Returns the output path.
    """
    out = output_path or _ensure_plots_dir() / "shot_map.png"
    _ensure_plots_dir()

    shots = [e for e in events if e.get("event_type") == "SHOT"]
    fig, ax, _ = _dark_pitch()

    plotted_any = False
    for ev in shots:
        coords = _resolve_coords(ev)
        if coords is None:
            continue
        x, y = coords
        sub = ev.get("sub_type") or "OFF_TARGET"
        style = _SHOT_STYLES.get(sub, _SHOT_STYLES["OFF_TARGET"])
        ax.scatter(x, y, edgecolors="white", linewidths=0.4, alpha=0.92, **style)
        plotted_any = True

    if not plotted_any:
        _no_data_label(ax, "No shot data")

    handles = [
        mlines.Line2D(
            [0], [0],
            marker=v["marker"], color="none",
            markerfacecolor=v["color"], markersize=9,
            markeredgecolor="white", markeredgewidth=0.4,
            label=v["label"],
        )
        for v in _SHOT_STYLES.values()
    ]
    _legend(ax, handles)

    ax.set_title(
        f"Shot Map  ·  {len(shots)} event{'s' if len(shots) != 1 else ''}",
        color=TEXT, fontsize=11, fontweight="bold", pad=10,
    )
    _brand_stamp(ax, "PitchPulse · Shot Analysis")
    return _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
# Transition map
# ══════════════════════════════════════════════════════════════════════════════

def plot_transition_map(
    events:      list[dict],
    output_path: Path | None = None,
) -> Path:
    """
    Visualise High Regains (emerald ▲) and Defensive Turnovers (amber ▼).
    Small random jitter separates overlapping zone-centroid markers.
    Returns the output path.
    """
    out = output_path or _ensure_plots_dir() / "transition_map.png"
    _ensure_plots_dir()

    regains   = [e for e in events if e.get("event_type") == "HIGH_REGAIN"]
    turnovers = [e for e in events if e.get("event_type") == "DEF_TURNOVER"]
    fig, ax, _ = _dark_pitch()

    rng = np.random.default_rng(42)   # deterministic jitter

    def _scatter_group(group: list[dict], marker: str, color: str) -> None:
        for ev in group:
            coords = _resolve_coords(ev)
            if coords is None:
                continue
            x, y = coords
            xj = x + rng.uniform(-1.2, 1.2)
            yj = y + rng.uniform(-1.2, 1.2)
            ax.scatter(
                xj, yj,
                marker=marker, color=color, s=230,
                edgecolors="white", linewidths=0.5,
                alpha=0.90, zorder=6,
            )

    _scatter_group(regains,   "^", EMERALD)
    _scatter_group(turnovers, "v", GOLD)

    if not regains and not turnovers:
        _no_data_label(ax, "No transition data")

    handles = [
        mlines.Line2D(
            [0], [0], marker="^", color="none",
            markerfacecolor=EMERALD, markersize=10,
            markeredgecolor="white", markeredgewidth=0.4,
            label=f"High Regain ({len(regains)})",
        ),
        mlines.Line2D(
            [0], [0], marker="v", color="none",
            markerfacecolor=GOLD, markersize=10,
            markeredgecolor="white", markeredgewidth=0.4,
            label=f"Def Turnover ({len(turnovers)})",
        ),
    ]
    _legend(ax, handles)

    total = len(regains) + len(turnovers)
    ax.set_title(
        f"Transition Map  ·  {total} event{'s' if total != 1 else ''}",
        color=TEXT, fontsize=11, fontweight="bold", pad=10,
    )
    _brand_stamp(ax, "PitchPulse · Pressing & Rest-Defence")
    return _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
# Zonal heatmap
# ══════════════════════════════════════════════════════════════════════════════

# Gradient: OLED black → deep green → bright emerald → gold (high density)
_HEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "tivvy_heat",
    [OLED_BG, "#052e16", "#065f46", EMERALD, GOLD],
    N=256,
)


def plot_zonal_heatmap(
    events:      list[dict],
    output_path: Path | None = None,
) -> Path:
    """
    Shade all 18 tactical zones by event count density.
    Zone IDs and counts are labelled on each rectangle.
    Returns the output path.
    """
    out = output_path or _ensure_plots_dir() / "zonal_heatmap.png"
    _ensure_plots_dir()

    # Count events per zone
    counts: dict[str, int] = {zid: 0 for zid in ZONES}
    for ev in events:
        zid = ev.get("zone_id")
        if zid and zid in counts:
            counts[zid] += 1
            continue
        coords = _resolve_coords(ev)
        if coords:
            zid, _ = get_zone_by_coords(*coords)
            if zid and zid in counts:
                counts[zid] += 1

    max_count = max(counts.values(), default=0) or 1

    fig, ax, _ = _dark_pitch()

    for zone_id in ZONE_ORDER:
        zone = ZONES[zone_id]
        x_min, y_min, x_max, y_max = zone["bbox"]
        w = x_max - x_min
        h = y_max - y_min
        cx, cy = zone["centroid"]
        intensity = counts[zone_id] / max_count

        # Filled rectangle
        rect = mpatches.FancyBboxPatch(
            (x_min, y_min), w, h,
            boxstyle="square,pad=0",
            facecolor=_HEAT_CMAP(intensity),
            edgecolor=LINE,
            linewidth=0.7,
            alpha=0.75,
            zorder=3,
        )
        ax.add_patch(rect)

        # Count label (only when > 0)
        if counts[zone_id] > 0:
            ax.text(
                cx, cy, str(counts[zone_id]),
                ha="center", va="center",
                fontsize=10, fontweight="black",
                color="white", zorder=5,
            )

        # Faint zone-ID tag near bottom of each cell
        ax.text(
            cx, y_min + 1.4, zone_id,
            ha="center", va="bottom",
            fontsize=5, color=TEXT_DIM, alpha=0.5,
            fontfamily="monospace", zorder=5,
        )

    # Colour bar
    sm = plt.cm.ScalarMappable(
        cmap=_HEAT_CMAP,
        norm=mcolors.Normalize(vmin=0, vmax=max_count),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.02, aspect=28)
    cbar.set_label("Events", color=TEXT_DIM, fontsize=7, labelpad=4)
    cbar.ax.yaxis.set_tick_params(color=TEXT_DIM, labelcolor=TEXT_DIM, labelsize=6.5)
    cbar.outline.set_edgecolor(LINE)

    if not any(counts.values()):
        _no_data_label(ax, "No event data")

    total = sum(counts.values())
    ax.set_title(
        f"Zonal Heatmap  ·  {total} event{'s' if total != 1 else ''}  ·  18-Zone Matrix",
        color=TEXT, fontsize=11, fontweight="bold", pad=10,
    )
    _brand_stamp(ax, "PitchPulse · Zone Analysis")
    return _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════════════

_MOCK_EVENTS: list[dict] = [
    # Shots
    {"event_type": "SHOT", "sub_type": "ON_TARGET",  "x": 92.0, "y": 34.0, "player_num": 9},
    {"event_type": "SHOT", "sub_type": "ON_TARGET",  "x": 88.0, "y": 28.0, "player_num": 10},
    {"event_type": "SHOT", "sub_type": "ON_TARGET",  "x": 94.0, "y": 40.0, "player_num": 9},
    {"event_type": "SHOT", "sub_type": "OFF_TARGET", "x": 84.0, "y": 18.0, "player_num": 7},
    {"event_type": "SHOT", "sub_type": "OFF_TARGET", "x": 79.0, "y": 52.0, "player_num": 11},
    {"event_type": "SHOT", "sub_type": "BLOCKED",    "x": 96.0, "y": 38.0, "player_num": 9},
    {"event_type": "SHOT", "sub_type": "BLOCKED",    "x": 90.0, "y": 44.0, "player_num": 8},
    # High regains (attacking third — pressing success)
    {"event_type": "HIGH_REGAIN", "sub_type": None, "x": 78.0, "y": 12.0, "player_num": 7},
    {"event_type": "HIGH_REGAIN", "sub_type": None, "x": 82.0, "y": 56.0, "player_num": 11},
    {"event_type": "HIGH_REGAIN", "sub_type": None, "x": 74.0, "y": 34.0, "player_num": 8},
    # Defensive turnovers (own half — rest-defence vulnerability)
    {"event_type": "DEF_TURNOVER", "sub_type": None, "x": 22.0, "y": 60.0, "player_num": 3},
    {"event_type": "DEF_TURNOVER", "sub_type": None, "x": 18.0, "y":  8.0, "player_num": 2},
    {"event_type": "DEF_TURNOVER", "sub_type": None, "x": 46.0, "y": 42.0, "player_num": 6},
    # Box entries (for heatmap coverage)
    {"event_type": "BOX_ENTRY", "sub_type": "CROSS", "x": 85.0, "y":  5.0, "player_num": 7},
    {"event_type": "BOX_ENTRY", "sub_type": "PASS",  "x": 72.0, "y": 30.0, "player_num": 10},
]


def _smoke_test() -> None:
    col = 58
    print()
    print("═" * col)
    print("  TIVVY VISUALIZER — Smoke Test")
    print("═" * col)
    print(f"  Mock events : {len(_MOCK_EVENTS)}")
    print(f"  Output dir  : {PLOTS_DIR.relative_to(ROOT)}")
    print("─" * col)

    p1 = plot_shot_map(_MOCK_EVENTS)
    p2 = plot_transition_map(_MOCK_EVENTS)
    p3 = plot_zonal_heatmap(_MOCK_EVENTS)

    print("─" * col)
    all_ok = True
    for p in (p1, p2, p3):
        kb = p.stat().st_size // 1024
        ok = kb >= 10
        flag = "✓" if ok else "⚠ SUSPICIOUSLY SMALL"
        print(f"  {flag}  {p.name:<30} {kb:>4} KB")
        if not ok:
            all_ok = False

    print("═" * col)
    print(f"  {'PASSED ✓' if all_ok else 'FAILED ⚠'}")
    print("═" * col)
    print()


if __name__ == "__main__":
    _smoke_test()
