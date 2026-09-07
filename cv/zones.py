"""
cv/zones.py
18-Zone Tactical Matrix for Tiverton Town FC.

Pitch template : 105 × 68 metres (FIFA standard).
Origin         : bottom-left corner.
x-axis         : 0 (defensive goal line) → 105 (attacking goal line).
y-axis         : 0 (left touchline)      → 68  (right touchline).

Zone ID convention: {THIRD}_{CHANNEL}
  Thirds   : D (Defensive 0-35m) | M (Middle 35-70m) | A (Attacking 70-105m)
  Channels : LF | LH | LC | RC | RH | RF  (left → right, 6 equal bands of ≈11.33m)

The two central attacking zones A_LC + A_RC together form the Zone 14 corridor.
"""

from __future__ import annotations

from typing import TypedDict

# ── Pitch dimensions ───────────────────────────────────────────────────────────
PITCH_LENGTH: float = 105.0
PITCH_WIDTH:  float = 68.0

# ── Partition boundaries ───────────────────────────────────────────────────────
_W = PITCH_WIDTH / 6          # ≈ 11.333 m per lateral channel

_X_BOUNDS: list[float] = [0.0, 35.0, 70.0, 105.0]
_Y_BOUNDS: list[float] = [round(i * _W, 6) for i in range(7)]  # 0 … 68

_THIRDS:   list[str] = ["D",  "M",  "A"]
_CHANNELS: list[str] = ["LF", "LH", "LC", "RC", "RH", "RF"]

_THIRD_NAMES: dict[str, str] = {
    "D": "Defensive Third",
    "M": "Middle Third",
    "A": "Attacking Third",
}
_CHANNEL_NAMES: dict[str, str] = {
    "LF": "Left Flank",
    "LH": "Left Half-Space",
    "LC": "Left Central",
    "RC": "Right Central",
    "RH": "Right Half-Space",
    "RF": "Right Flank",
}


# ── Type definition ────────────────────────────────────────────────────────────
class ZoneSpec(TypedDict):
    id:       str
    name:     str
    third:    str
    channel:  str
    bbox:     tuple[float, float, float, float]   # x_min, y_min, x_max, y_max
    centroid: tuple[float, float]                 # (cx, cy)


# ── Zone construction ──────────────────────────────────────────────────────────
def _build_zones() -> dict[str, ZoneSpec]:
    zones: dict[str, ZoneSpec] = {}
    for t_idx, third in enumerate(_THIRDS):
        x_min = _X_BOUNDS[t_idx]
        x_max = _X_BOUNDS[t_idx + 1]
        for c_idx, channel in enumerate(_CHANNELS):
            y_min = _Y_BOUNDS[c_idx]
            y_max = _Y_BOUNDS[c_idx + 1]
            zone_id = f"{third}_{channel}"

            name = f"{_THIRD_NAMES[third]} · {_CHANNEL_NAMES[channel]}"
            if third == "A" and channel in ("LC", "RC"):
                name += " (Zone 14)"

            zones[zone_id] = ZoneSpec(
                id=zone_id,
                name=name,
                third=third,
                channel=channel,
                bbox=(x_min, y_min, x_max, y_max),
                centroid=(
                    round((x_min + x_max) / 2, 4),
                    round((y_min + y_max) / 2, 4),
                ),
            )
    return zones


# ── Public zone registry ───────────────────────────────────────────────────────
ZONES: dict[str, ZoneSpec] = _build_zones()

# Ordered zone IDs for consistent iteration (D→M→A, LF→RF)
ZONE_ORDER: list[str] = [
    f"{t}_{c}" for t in _THIRDS for c in _CHANNELS
]


# ── Public helpers ─────────────────────────────────────────────────────────────
def get_zone_by_coords(x: float, y: float) -> tuple[str | None, str | None]:
    """
    Return (zone_id, zone_name) for pitch coordinates (x, y) in metres.
    Returns (None, None) if the point lies outside all zone bounding boxes.

    Uses half-open intervals [x_min, x_max) to avoid duplicate assignment
    on shared boundaries, except the final boundary which is closed.
    """
    for zone_id in ZONE_ORDER:
        zone = ZONES[zone_id]
        x_min, y_min, x_max, y_max = zone["bbox"]
        in_x = x_min <= x < x_max if x_max < PITCH_LENGTH else x_min <= x <= x_max
        in_y = y_min <= y < y_max if y_max < PITCH_WIDTH  else y_min <= y <= y_max
        if in_x and in_y:
            return zone_id, zone["name"]
    return None, None


def get_zone_centroid(zone_id: str) -> tuple[float, float] | None:
    """
    Return (x, y) centroid for zone_id.
    Returns None if zone_id is not recognised.
    """
    zone = ZONES.get(zone_id)
    return zone["centroid"] if zone else None


def zones_for_third(third: str) -> list[ZoneSpec]:
    """Return all 6 ZoneSpec dicts for a given third ('D', 'M', or 'A')."""
    return [ZONES[f"{third}_{c}"] for c in _CHANNELS]


# ── Debug helper ───────────────────────────────────────────────────────────────
def list_zones() -> None:
    """Print a formatted table of all 18 zones (debug / CLI use)."""
    col = 76
    print("─" * col)
    print(f"  {'Zone ID':<10} {'Third':<18} {'Channel':<16} {'Centroid':>12}  BBox")
    print("─" * col)
    for zone_id in ZONE_ORDER:
        z  = ZONES[zone_id]
        cx, cy = z["centroid"]
        x0, y0, x1, y1 = z["bbox"]
        print(
            f"  {zone_id:<10}"
            f" {_THIRD_NAMES[z['third']]:<18}"
            f" {_CHANNEL_NAMES[z['channel']]:<16}"
            f" ({cx:5.1f},{cy:5.1f})"
            f"  [{x0:.0f}-{x1:.0f}, {y0:.1f}-{y1:.1f}]"
        )
    print("─" * col)
    print(f"  {len(ZONES)} zones total  |  Pitch: {PITCH_LENGTH}m × {PITCH_WIDTH}m")
    print("─" * col)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    list_zones()

    # Quick coordinate lookup test
    test_cases = [
        (92.0, 34.0, "A_RC — central attacking"),
        (20.0,  5.0, "D_LF — defensive left flank"),
        (52.5, 34.0, "M_RC — central midfield"),
        (105.0, 68.0, "A_RF — corner boundary"),
        (-1.0,  0.0, "out of bounds"),
    ]
    print("\n  Coordinate lookup tests:")
    for x, y, expected in test_cases:
        zid, zname = get_zone_by_coords(x, y)
        result = f"{zid} · {zname}" if zid else "None (out of bounds)"
        print(f"  ({x:5.1f}, {y:4.1f})  →  {result}")
        print(f"             expected ≈  {expected}")
