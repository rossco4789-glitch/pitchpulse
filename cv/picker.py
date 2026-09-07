"""
cv/picker.py
Tier 1 Computer-Assisted Coordinate Picker — Pixel → Pitch Conversion.

Converts camera pixel coordinates (u, v) to real-world pitch coordinates
(x_m, y_m) using the homography matrix H produced by cv/calibration.py.
Optionally appends the resulting event to match_ledger.json with provenance
field source="video_assisted".

Mathematics
-----------
  [x']   [h00 h01 h02]   [u]
  [y'] = [h10 h11 h12] × [v]
  [w']   [h20 h21 h22]   [1]

  x_m = x'/w'  →  clamped to [0.0, 105.0]
  y_m = y'/w'  →  clamped to [0.0,  68.0]

Spatial integrity contract (unchanged)
--------------------------------------
  zone_id is derived exclusively via cv.zones.get_zone_by_coords(x_m, y_m).
  The picker honours this: no zone is inferred from pixel position, only from
  the calibrated metric coordinate after projection.

Usage
-----
  # Single event — print result only:
  python cv/picker.py --pixel 540 320 --event-type SHOT --sub-type ON_TARGET --match-seconds 4620 --period 2H

  # Amend the ledger in place:
  python cv/picker.py --pixel 540 320 --event-type SHOT --sub-type ON_TARGET \\
    --match-seconds 4620 --period 2H --player 9 \\
    --ledger data/processed/match_ledger.json

  # Interactive — click a video frame to pick the pixel:
  python cv/picker.py --frame data/raw/frame.jpg --event-type BOX_ENTRY --sub-type CARRY \\
    --match-seconds 2830 --period 1H --ledger data/processed/match_ledger.json

This module does NOT touch run_matchday.py — it is a standalone midweek utility.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.calibration import load_calibration, DEFAULT_CALIB_PATH
from cv.zones import get_zone_by_coords, PITCH_LENGTH, PITCH_WIDTH

# ── Valid event vocabulary (mirrors tagger/index.html) ────────────────────────
VALID_EVENT_TYPES = {
    "SHOT":        ["ON_TARGET", "OFF_TARGET", "BLOCKED"],
    "BOX_ENTRY":   ["PASS", "CROSS", "CARRY"],
    "DEF_TURNOVER": [],
    "HIGH_REGAIN": [],
    "SET_PIECE":   ["ATT_CORNER", "DEF_CORNER", "FREE_KICK"],
    "AERIAL_DUEL": ["WON", "LOST"],
    "SECOND_BALL": ["WON", "LOST"],
}


# ══════════════════════════════════════════════════════════════════════════════
# Core conversion
# ══════════════════════════════════════════════════════════════════════════════

def pixel_to_pitch(u: float, v: float, H: np.ndarray) -> tuple[float, float]:
    """
    Project pixel coordinate (u, v) to real-world pitch coordinate (x_m, y_m)
    using homography matrix H.

    The result is clamped to the FIFA pitch bounds:
      x_m ∈ [0.0, 105.0]
      y_m ∈ [0.0,  68.0]

    Parameters
    ----------
    u, v : pixel column and row.
    H    : 3×3 float64 homography matrix from cv/calibration.py.

    Returns
    -------
    (x_m, y_m) : pitch coordinates in metres.
    """
    p_pixel = np.array([u, v, 1.0], dtype=np.float64)
    p_world = H @ p_pixel
    w = p_world[2]

    if abs(w) < 1e-10:
        # Degenerate: point at infinity — return pitch centre as fallback
        return (PITCH_LENGTH / 2.0, PITCH_WIDTH / 2.0)

    x_m = float(p_world[0] / w)
    y_m = float(p_world[1] / w)

    # Clamp to valid pitch bounds
    x_m = max(0.0, min(PITCH_LENGTH, x_m))
    y_m = max(0.0, min(PITCH_WIDTH,  y_m))

    return round(x_m, 4), round(y_m, 4)


def resolve_zone(x_m: float, y_m: float) -> tuple[Optional[str], Optional[str]]:
    """
    Return (zone_id, zone_name) for pitch coordinates via cv/zones.get_zone_by_coords().
    This is the sole zone-classification entry point — no sub-type inference.
    """
    return get_zone_by_coords(x_m, y_m)


def build_video_event(
    u: float,
    v: float,
    H: np.ndarray,
    event_type: str,
    match_seconds: int,
    period: str,
    sub_type: str = "",
    player_num: Optional[int] = None,
    video_timestamp_s: Optional[float] = None,
) -> dict:
    """
    Build a complete video-assisted event dict from a pixel coordinate.

    Parameters
    ----------
    u, v            : pixel coordinate clicked on the video frame.
    H               : homography matrix from cv/calibration.py.
    event_type      : one of VALID_EVENT_TYPES.
    match_seconds   : match time in seconds (e.g. 4620 = 77:00 in 2H).
    period          : "1H" or "2H".
    sub_type        : optional sub-type string (e.g. "ON_TARGET").
    player_num      : shirt number, if known.
    video_timestamp_s: raw video file timestamp in seconds (for traceability).

    Returns
    -------
    dict : v2-compatible event object with source="video_assisted".
    """
    x_m, y_m = pixel_to_pitch(u, v, H)
    zone_id, zone_name = resolve_zone(x_m, y_m)

    # Derive clock_display from match_seconds
    period_offset = 0 if period == "1H" else 2700  # 45 mins
    elapsed = match_seconds - period_offset
    mins = elapsed // 60
    secs = elapsed % 60
    clock_display = f"{mins}:{secs:02d}"

    # x_pct / y_pct for ledger compatibility with tagger schema
    x_pct = round(x_m / PITCH_LENGTH, 4)
    y_pct = round(y_m / PITCH_WIDTH,  4)

    event: dict = {
        "id":             str(uuid.uuid4()),
        "source":         "video_assisted",
        "period":         period,
        "match_seconds":  match_seconds,
        "clock_display":  clock_display,
        "event_type":     event_type,
        "sub_type":       sub_type or None,
        "x_pct":          x_pct,
        "y_pct":          y_pct,
        "x_m":            x_m,
        "y_m":            y_m,
        "zone_id":        zone_id,
        "zone_name":      zone_name,
        "pixel_u":        round(u, 1),
        "pixel_v":        round(v, 1),
        "timestamp_iso":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }

    if player_num is not None:
        event["player_num"] = player_num

    if video_timestamp_s is not None:
        event["video_timestamp_s"] = round(video_timestamp_s, 2)

    return event


# ══════════════════════════════════════════════════════════════════════════════
# Ledger amendment
# ══════════════════════════════════════════════════════════════════════════════

def append_video_event(ledger_path: Path, event: dict) -> Path:
    """
    Append a video-assisted event to the unmatched_tags list in match_ledger.json.

    The event lands in unmatched_tags (not matched) because it has no paired
    club-feed entry.  Downstream agents already handle unmatched_tags events —
    no pipeline changes required.

    Parameters
    ----------
    ledger_path : path to match_ledger.json (read-modify-write).
    event       : dict from build_video_event().

    Returns
    -------
    Path : the ledger path (unchanged) for chaining.
    """
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        raise FileNotFoundError(
            f"Ledger not found: {ledger_path}\n"
            "Run python run_matchday.py (or --skip-visuals) to generate it first."
        )

    with open(ledger_path, encoding="utf-8") as f:
        ledger = json.load(f)

    if "unmatched_tags" not in ledger:
        ledger["unmatched_tags"] = []

    ledger["unmatched_tags"].append(event)

    # Update summary counter
    summary = ledger.setdefault("summary", {})
    summary["video_assisted_events"] = summary.get("video_assisted_events", 0) + 1

    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)

    return ledger_path


# ══════════════════════════════════════════════════════════════════════════════
# Interactive frame picker (matplotlib click interface)
# ══════════════════════════════════════════════════════════════════════════════

def _run_frame_picker(frame_path: Path) -> tuple[float, float]:
    """
    Display a video frame in matplotlib and return (u, v) of the user's click.
    Blocks until the window is closed.
    """
    import cv2 as _cv2
    import matplotlib.pyplot as plt

    img = _cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read frame: {frame_path}")
    img_rgb = _cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result: list[tuple[float, float]] = []

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img_rgb)
    ax.set_title(
        "PITCHPULSE PICKER — Click the event location, then close this window",
        color="#f59e0b", fontsize=11,
    )
    fig.patch.set_facecolor("#09090b")
    ax.set_facecolor("#09090b")

    def _on_click(event) -> None:
        if event.inaxes is not ax:
            return
        if result:  # only first click counts
            return
        u, v = float(event.xdata), float(event.ydata)
        result.append((u, v))
        ax.plot(u, v, "o", color="#f59e0b", markersize=10, zorder=5)
        ax.set_title(
            f"✓ Selected pixel ({u:.0f}, {v:.0f}) — close this window to confirm",
            color="#22c55e", fontsize=11,
        )
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_click)
    plt.tight_layout()
    plt.show()

    if not result:
        raise RuntimeError("No pixel selected — window closed without clicking.")

    return result[0]


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="PitchPulse Tier 1 — Pixel-to-Pitch Coordinate Picker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    px_group = parser.add_mutually_exclusive_group(required=True)
    px_group.add_argument(
        "--pixel", nargs=2, type=float, metavar=("U", "V"),
        help="Pixel coordinate to convert (e.g. --pixel 540 320)",
    )
    px_group.add_argument(
        "--frame", metavar="IMAGE",
        help="Path to a still frame; click the event location interactively",
    )

    parser.add_argument("--event-type",    required=True, metavar="TYPE",  help="Event type (e.g. SHOT)")
    parser.add_argument("--sub-type",      default="",    metavar="SUB",   help="Sub-type (e.g. ON_TARGET)")
    parser.add_argument("--match-seconds", required=True, type=int,        help="Match time in seconds")
    parser.add_argument("--period",        required=True, choices=["1H", "2H"], help="Match period")
    parser.add_argument("--player",        type=int, default=None,         help="Shirt number (optional)")
    parser.add_argument("--video-ts",      type=float, default=None,       help="Video file timestamp in seconds")
    parser.add_argument(
        "--calib", default=str(DEFAULT_CALIB_PATH), metavar="FILE",
        help=f"Calibration JSON path (default: {DEFAULT_CALIB_PATH.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--ledger", default=None, metavar="FILE",
        help="match_ledger.json path — if given, appends event to unmatched_tags",
    )

    args = parser.parse_args(argv)

    # ── Validate event type ───────────────────────────────────────────────────
    event_type = args.event_type.upper()
    if event_type not in VALID_EVENT_TYPES:
        print(f"  ✗  Unknown event type '{event_type}'.  Valid: {list(VALID_EVENT_TYPES)}", file=sys.stderr)
        sys.exit(1)

    sub_type = args.sub_type.upper() if args.sub_type else ""
    valid_subs = VALID_EVENT_TYPES[event_type]
    if sub_type and valid_subs and sub_type not in valid_subs:
        print(f"  ✗  Sub-type '{sub_type}' not valid for {event_type}.  Valid: {valid_subs}", file=sys.stderr)
        sys.exit(1)

    # ── Load calibration ──────────────────────────────────────────────────────
    calib_path = Path(args.calib)
    try:
        H = load_calibration(calib_path)
    except FileNotFoundError as exc:
        print(f"  ✗  {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Get pixel coordinate ──────────────────────────────────────────────────
    if args.pixel:
        u, v = args.pixel
    else:
        u, v = _run_frame_picker(Path(args.frame))

    # ── Build event ───────────────────────────────────────────────────────────
    event = build_video_event(
        u=u, v=v, H=H,
        event_type=event_type,
        match_seconds=args.match_seconds,
        period=args.period,
        sub_type=sub_type,
        player_num=args.player,
        video_timestamp_s=args.video_ts,
    )

    # ── Print result ──────────────────────────────────────────────────────────
    x_m, y_m = event["x_m"], event["y_m"]
    zone_id   = event.get("zone_id", "—")
    zone_name = event.get("zone_name", "out of bounds")

    print()
    print(f"  Pixel         : ({u:.1f}, {v:.1f})")
    print(f"  Pitch coord   : ({x_m:.2f} m, {y_m:.2f} m)")
    print(f"  Zone          : {zone_id}  ·  {zone_name}")
    print(f"  Event         : {event_type}" + (f" / {sub_type}" if sub_type else ""))
    print(f"  Match time    : {event['clock_display']} ({args.period})")
    print(f"  Event ID      : {event['id']}")
    print()

    # ── Amend ledger (optional) ───────────────────────────────────────────────
    if args.ledger:
        ledger_path = Path(args.ledger)
        try:
            append_video_event(ledger_path, event)
            print(f"  ✓  Event appended → {ledger_path}")
        except FileNotFoundError as exc:
            print(f"  ✗  {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print("  (--ledger not specified — event printed only, not saved)")

    print()


if __name__ == "__main__":
    main()
