#!/usr/bin/env python3
"""
reconcile/sync.py
Post-match reconciliation engine for Tiverton Town FC.
Bridges mobile pitch-side tags with official club match communications.

Usage:
    python reconcile/sync.py

Reads:
    data/raw/tivvy_events_*.json   — tagger exports (one file per period)
    data/raw/tivvy_x_feed.json     — structured club feed (preferred)
    data/raw/tivvy_x_feed.txt      — plain-text club feed (fallback)

Writes:
    data/processed/match_ledger.json
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from glob import glob

# Zone derivation from real spatial coordinates (v2 tagger produces x_m, y_m).
# Import lazily inside build_ledger so sync.py remains usable standalone.
_ZONE_FN = None

def _get_zone_fn():
    """Lazy-load cv.zones.get_zone_by_coords once, cache it."""
    global _ZONE_FN
    if _ZONE_FN is None:
        try:
            _root = Path(__file__).resolve().parent.parent
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from cv.zones import get_zone_by_coords
            _ZONE_FN = get_zone_by_coords
        except ImportError:
            _ZONE_FN = False           # sentinel: cv.zones unavailable
    return _ZONE_FN if _ZONE_FN is not False else None

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
RAW_DIR    = ROOT / "data" / "raw"
PROC_DIR   = ROOT / "data" / "processed"
LEDGER_OUT = PROC_DIR / "match_ledger.json"

# ── Configuration ──────────────────────────────────────────────────────────────
RECON_WINDOW_S = 90   # ±90 second temporal reconciliation window

# ── Squad Roster: shirt number → player name ───────────────────────────────────
SQUAD: dict[int, str] = {
    1:  "S. Humphries",
    2:  "L. Cruwys",
    3:  "M. Stokes",
    4:  "R. Tilney",
    5:  "J. Manley",
    6:  "C. Burrows",
    7:  "T. Granger",
    8:  "A. Fewings",
    9:  "D. Waters",
    10: "B. Holloway",
    11: "K. Sercombe",
    12: "P. Jarvis",
    13: "O. Pearce",
    14: "N. Aves",
    15: "J. Sampson",
    16: "T. Hookway",
    17: "R. Luscombe",
    18: "F. Damerell",
}

# ── Club-feed keyword → normalised event type ──────────────────────────────────
KEYWORD_MAP: dict[str, str] = {
    "disallow":  "DISALLOWED_GOAL",   # must precede "goal"
    "own goal":  "OWN_GOAL",
    "own":       "OWN_GOAL",
    "goal":      "GOAL",
    "red":       "RED_CARD",
    "yellow":    "YELLOW_CARD",
    "substitut": "SUBSTITUTION",
    "sub":       "SUBSTITUTION",
    "penalty":   "PENALTY",
    "pen":       "PENALTY",
    "corner":    "CORNER",
    "offside":   "OFFSIDE",
    "free kick": "FREE_KICK",
    "free":      "FREE_KICK",
}

# Tagger event types eligible for club-event reconciliation
RECONCILABLE_TAGS = {
    "SHOT", "SET_PIECE", "HIGH_REGAIN", "BOX_ENTRY", "DEF_TURNOVER",
    "AERIAL_DUEL", "SECOND_BALL",           # non-league physics events (v2)
}

# ── Minute-string regex ────────────────────────────────────────────────────────
_MINUTE_RE = re.compile(r"(\d{1,3})(?:\+(\d{1,2}))?'")

# ── Column width for audit output ──────────────────────────────────────────────
_COL = 72


# ══════════════════════════════════════════════════════════════════════════════
# Directory initialisation
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dirs() -> None:
    """Create data/raw and data/processed if they do not exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Data ingestion
# ══════════════════════════════════════════════════════════════════════════════

def load_tag_events(raw_dir: Path) -> list[dict]:
    """
    Load all tivvy_events_*.json exports from raw_dir.
    Merges all periods into a single list sorted by match_seconds.
    """
    pattern = str(raw_dir / "tivvy_events_*.json")
    files   = sorted(glob(pattern))

    if not files:
        print(f"  [WARN] No tag export files found in {raw_dir.relative_to(ROOT)}")
        return []

    events: list[dict] = []
    for fp in files:
        with open(fp, encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, list):
            events.extend(data)
            print(f"  [TAG ] Loaded {len(data):>3} events ← {Path(fp).name}")
        else:
            print(f"  [WARN] Unexpected format in {Path(fp).name} — skipped.")

    events.sort(key=lambda e: e.get("match_seconds", 0))
    print(f"  [TAG ] {len(events)} total tag events after merge.")
    return events


def load_club_feed(raw_dir: Path) -> tuple[list | str | None, str]:
    """
    Try tivvy_x_feed.json (structured JSON) first.
    Fall back to tivvy_x_feed.txt (plain text, one post per line).
    Returns (data, feed_type) where feed_type ∈ {'json', 'txt', 'none'}.
    """
    json_path = raw_dir / "tivvy_x_feed.json"
    txt_path  = raw_dir / "tivvy_x_feed.txt"

    if json_path.exists():
        with open(json_path, encoding="utf-8-sig") as f:
            data = json.load(f)
        print(f"  [FEED] Structured JSON feed loaded ← {json_path.name}")
        return data, "json"

    if txt_path.exists():
        with open(txt_path, encoding="utf-8-sig") as f:
            text = f.read()
        print(f"  [FEED] Plain-text feed loaded ← {txt_path.name}")
        return text, "txt"

    print("  [WARN] No club feed found in data/raw/. All tags will be unmatched.")
    return None, "none"


# ══════════════════════════════════════════════════════════════════════════════
# Minute-string parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_minute_to_seconds(raw: str) -> int | None:
    """
    Convert a match-minute string to elapsed seconds from kick-off.

    Examples:
        "34'"    → 2040   (34 × 60)
        "45+2'"  → 2820   (47 × 60)
        "HT"     → 2700   (45 × 60)
        "FT"     → 5400   (90 × 60)

    Returns None if the string cannot be parsed.
    """
    token = raw.strip().upper()

    if token in ("HT", "HALF-TIME", "HALFTIME", "HALF TIME"):
        return 45 * 60
    if token in ("FT", "FULL-TIME", "FULLTIME", "FULL TIME"):
        return 90 * 60

    m = _MINUTE_RE.search(raw)
    if m:
        base  = int(m.group(1))
        extra = int(m.group(2)) if m.group(2) else 0
        return (base + extra) * 60

    return None


def _detect_event_type(text: str) -> str:
    """Return the normalised event type for the first matching keyword in text."""
    lower = text.lower()
    for kw, ev_type in KEYWORD_MAP.items():
        if kw in lower:
            return ev_type
    return "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════════════
# Club feed normalisation
# ══════════════════════════════════════════════════════════════════════════════

def extract_club_events(feed_data: list | str, feed_type: str) -> list[dict]:
    """
    Normalise a raw club feed into a flat, sorted list of:
        { elapsed_s: int, type: str, detail: str, raw_text: str }
    """
    club_events: list[dict] = []

    if feed_type == "json":
        for item in feed_data:
            raw_minute = str(item.get("minute", ""))
            elapsed_s  = parse_minute_to_seconds(raw_minute)
            if elapsed_s is None:
                continue
            club_events.append({
                "elapsed_s": elapsed_s,
                "type":      str(item.get("type", "UNKNOWN")).upper(),
                "detail":    item.get("detail", ""),
                "raw_text":  f"{raw_minute} {item.get('detail', '')}".strip(),
            })

    elif feed_type == "txt":
        for line in feed_data.splitlines():
            line = line.strip()
            if not line:
                continue

            # Attempt to parse a minute string from the line
            elapsed_s = None
            m = _MINUTE_RE.search(line)
            if m:
                elapsed_s = parse_minute_to_seconds(m.group(0))

            # Fallback: look for HT / FT tokens
            if elapsed_s is None:
                upper = line.upper()
                if any(t in upper for t in ("HT", "HALF-TIME", "HALF TIME")):
                    elapsed_s = 45 * 60
                elif any(t in upper for t in ("FT", "FULL-TIME", "FULL TIME")):
                    elapsed_s = 90 * 60

            if elapsed_s is None:
                continue  # line has no temporal anchor — skip

            club_events.append({
                "elapsed_s": elapsed_s,
                "type":      _detect_event_type(line),
                "detail":    line,
                "raw_text":  line,
            })

    club_events.sort(key=lambda e: e["elapsed_s"])
    return club_events


# ══════════════════════════════════════════════════════════════════════════════
# Player resolution
# ══════════════════════════════════════════════════════════════════════════════

def resolve_player(num: int | None) -> str:
    """Resolve a shirt number to a player name. Returns a safe fallback string."""
    if num is None:
        return "Team (unattributed)"
    try:
        key = int(num)
    except (TypeError, ValueError):
        return f"#{num} (invalid)"
    return SQUAD.get(key, f"#{key} (unknown)")


# ══════════════════════════════════════════════════════════════════════════════
# Temporal reconciliation
# ══════════════════════════════════════════════════════════════════════════════

def reconcile_events(
    tag_events:  list[dict],
    club_events: list[dict],
    window:      int = RECON_WINDOW_S,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Greedy nearest-match reconciliation.

    For each eligible tag event (sorted by match_seconds), find the closest
    club event within ±window seconds that has not already been claimed.
    One club event may be claimed by at most one tag.

    Returns:
        matched          — list of { tag, club_event, delta_s, player_name }
        unmatched_tags   — tag events with no club event within the window
        unmatched_club   — club events not claimed by any tag
    """
    matched:        list[dict] = []
    unmatched_tags: list[dict] = []
    used_indices:   set[int]   = set()

    for tag in tag_events:
        if tag.get("event_type") not in RECONCILABLE_TAGS:
            # Pass through — SUB and any future non-spatial event types go
            # straight to unmatched_tags; no club-feed matching attempted.
            unmatched_tags.append(tag)
            continue

        t_secs     = tag.get("match_seconds", 0)
        best_idx   = None
        best_delta: int | None = None

        for idx, club_ev in enumerate(club_events):
            if idx in used_indices:
                continue
            delta = abs(t_secs - club_ev["elapsed_s"])
            if delta <= window and (best_delta is None or delta < best_delta):
                best_delta = delta
                best_idx   = idx

        if best_idx is not None:
            used_indices.add(best_idx)
            matched.append({
                "tag":         tag,
                "club_event":  club_events[best_idx],
                "delta_s":     best_delta,
                "player_name": resolve_player(tag.get("player_num")),
            })
        else:
            unmatched_tags.append(tag)

    unmatched_club = [
        ev for i, ev in enumerate(club_events) if i not in used_indices
    ]

    return matched, unmatched_tags, unmatched_club


# ══════════════════════════════════════════════════════════════════════════════
# Ledger assembly & I/O
# ══════════════════════════════════════════════════════════════════════════════

def _stamp_zone(tag: dict) -> None:
    """
    Derive and stamp zone_id + zone_name onto a tag event in-place.

    Uses real x_m / y_m coordinates from the v2 tagger (pitch tap).
    Events without spatial data (legacy v1 exports) receive zone_id: null.
    This is the ONLY place zone IDs enter the ledger — no sub-type inference.
    """
    x_m = tag.get("x_m")
    y_m = tag.get("y_m")

    if x_m is None or y_m is None:
        tag["zone_id"]   = None
        tag["zone_name"] = None
        return

    fn = _get_zone_fn()
    if fn is None:
        tag["zone_id"]   = None
        tag["zone_name"] = "cv.zones unavailable"
        return

    zone_id, zone_name = fn(float(x_m), float(y_m))
    tag["zone_id"]   = zone_id
    tag["zone_name"] = zone_name


def build_ledger(
    matched:          list[dict],
    unmatched_tags:   list[dict],
    unmatched_club:   list[dict],
    opponent_events:  list[dict] | None = None,
) -> dict:
    opponent_events = opponent_events or []

    # ── Extract substitution events before zone-stamping ──────────────────
    # SUB events carry no spatial data; keep them in a dedicated top-level
    # list so synthesis.py can build a dynamic player-name timeline.
    substitutions = [t for t in unmatched_tags if t.get("event_type") == "SUB"]
    spatial_tags  = [t for t in unmatched_tags if t.get("event_type") != "SUB"]

    # Enrich each sub record with resolved player names
    for sub in substitutions:
        sub["player_off_name"] = resolve_player(sub.get("player_off"))
        sub["player_on_name"]  = resolve_player(sub.get("player_on"))

    # ── Stamp zone_id onto every spatial tag event ────────────────────────
    _zoned: set[int] = set()
    for m in matched:
        tag_id = id(m["tag"])
        if tag_id not in _zoned:
            _stamp_zone(m["tag"])
            _zoned.add(tag_id)
    for tag in spatial_tags:
        _stamp_zone(tag)

    zoned_count = sum(
        1 for m in matched if m["tag"].get("zone_id") is not None
    ) + sum(
        1 for t in spatial_tags if t.get("zone_id") is not None
    )

    return {
        "generated_at":            datetime.now(timezone.utc).isoformat(),
        "reconciliation_window_s": RECON_WINDOW_S,
        "schema_version":          2,
        "summary": {
            "matched":               len(matched),
            "unmatched_tags":        len(spatial_tags),
            "unmatched_club_events": len(unmatched_club),
            "events_with_zone":      zoned_count,
            "substitutions":         len(substitutions),
            "opponent_events":       len(opponent_events),
        },
        "matched":               matched,
        "unmatched_tags":        spatial_tags,
        "substitutions":         substitutions,
        "opponent_events":       opponent_events,     # ← opposition tracking (v2.3)
        "unmatched_club_events": unmatched_club,
    }


def write_ledger(ledger: dict, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# Terminal audit summary
# ══════════════════════════════════════════════════════════════════════════════

def _rule(char: str = "─") -> None:
    print(char * _COL)


def print_audit(ledger: dict) -> None:
    s = ledger["summary"]

    _rule("═")
    print("  TIVERTON TOWN FC — POST-MATCH RECONCILIATION AUDIT")
    print(f"  Generated : {ledger['generated_at']}")
    print(f"  Window    : ±{ledger['reconciliation_window_s']}s")
    _rule()
    print(f"  {'MATCHED':<32} {s['matched']:>4}  ✓")
    print(f"  {'UNMATCHED TAGS':<32} {s['unmatched_tags']:>4}  ⚠")
    print(f"  {'UNMATCHED CLUB EVENTS':<32} {s['unmatched_club_events']:>4}  ⚠")
    print(f"  {'EVENTS WITH REAL ZONE (v2)':<32} {s.get('events_with_zone', '—'):>4}  📍")
    print(f"  {'SUBSTITUTIONS':<32} {s.get('substitutions', 0):>4}  ↔")
    print(f"  {'OPPOSITION EVENTS':<32} {s.get('opponent_events', 0):>4}  🔴")
    _rule()

    if ledger["matched"]:
        print("  MATCHED EVENTS")
        _rule()
        hdr = (
            f"  {'Clock':<8} {'Per':<4} {'Tag Type':<16} {'Sub':<13}"
            f"{'Club Type':<16} {'Δs':>4}  Player"
        )
        print(hdr)
        _rule()
        for m in ledger["matched"]:
            tag = m["tag"]
            clb = m["club_event"]
            print(
                f"  {tag.get('clock_display','--:--'):<8}"
                f" {tag.get('period','?'):<4}"
                f" {tag.get('event_type',''):<16}"
                f" {str(tag.get('sub_type') or ''):<13}"
                f" {clb.get('type',''):<16}"
                f" {m['delta_s']:>3}s"
                f"  {m['player_name']}"
            )
        _rule()

    if ledger["unmatched_tags"]:
        print("\n  UNMATCHED TAGS  (no club event within window)")
        _rule()
        for tag in ledger["unmatched_tags"]:
            print(
                f"  {tag.get('clock_display','--:--'):<8}"
                f" {tag.get('period','?'):<4}"
                f" {tag.get('event_type',''):<16}"
                f" {str(tag.get('sub_type') or ''):<13}"
                f"  {resolve_player(tag.get('player_num'))}"
            )
        _rule()

    if ledger.get("opponent_events"):
        print("\n  OPPOSITION EVENTS  (zone-stamped, not reconciled against club feed)")
        _rule()
        for tag in ledger["opponent_events"]:
            print(
                f"  {tag.get('clock_display','--:--'):<8}"
                f" {tag.get('period','?'):<4}"
                f" {tag.get('event_type',''):<16}"
                f" {str(tag.get('sub_type') or ''):<13}"
                f"  zone: {tag.get('zone_id') or 'no coord'}"
            )
        _rule()

    if ledger["unmatched_club_events"]:
        print("\n  UNMATCHED CLUB EVENTS  (no tag within window)")
        _rule()
        for ev in ledger["unmatched_club_events"]:
            mm  = ev["elapsed_s"] // 60
            ss  = ev["elapsed_s"] % 60
            det = ev.get("detail", "")[:44]
            print(f"  {mm:02d}:{ss:02d}    {ev.get('type',''):<18}  {det}")
        _rule()

    _rule("═")
    print(f"  Ledger → {LEDGER_OUT.relative_to(ROOT)}")
    _rule("═")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print()
    _rule("═")
    print("  TIVVY SYNC — Post-Match Reconciliation Engine")
    _rule("═")

    ensure_dirs()

    print("\n[1/4] Ingesting tag exports …")
    tag_events = load_tag_events(RAW_DIR)

    print("\n[2/4] Loading club feed …")
    feed_data, feed_type = load_club_feed(RAW_DIR)
    club_events: list[dict] = []
    if feed_type != "none" and feed_data is not None:
        club_events = extract_club_events(feed_data, feed_type)
    print(f"  [FEED] {len(club_events)} club event(s) parsed.")

    # Separate opposition events before reconciliation — the club feed is
    # Tiverton-centric, so opposition observations are stored directly without
    # any attempt to match them against club-feed entries.
    opponent_events  = [e for e in tag_events if e.get("team") == "opponent"]
    tiverton_events  = [e for e in tag_events if e.get("team") != "opponent"]
    if opponent_events:
        print(f"  [OPP ] {len(opponent_events)} opposition event(s) split off before reconciliation.")

    print("\n[3/4] Reconciling …")
    matched, unmatched_tags, unmatched_club = reconcile_events(
        tiverton_events, club_events, window=RECON_WINDOW_S
    )
    print(
        f"  Matched: {len(matched)}  |  "
        f"Unmatched tags: {len(unmatched_tags)}  |  "
        f"Unmatched club: {len(unmatched_club)}"
    )

    # Zone-stamp opposition events using the same spatial logic
    for opp_ev in opponent_events:
        _stamp_zone(opp_ev)

    print("\n[4/4] Writing ledger …")
    ledger = build_ledger(matched, unmatched_tags, unmatched_club,
                          opponent_events=opponent_events)
    write_ledger(ledger, LEDGER_OUT)

    print()
    print_audit(ledger)


if __name__ == "__main__":
    main()
