#!/usr/bin/env python3
"""
tagger_sanity.py — PitchPulse post-match ledger auditor
==========================================================
Usage:
    python tools/tagger_sanity.py                          # default ledger
    python tools/tagger_sanity.py path/to/ledger.json     # explicit path
    python tools/tagger_sanity.py --test                   # run built-in unit tests

Input formats accepted:
    • Raw tagger export  — a JSON *array* of event objects
    • Match ledger       — the reconcile output with matched / unmatched_tags keys

Exit codes:
    0  — clean or warnings only
    1  — one or more CRITICAL ERRORS
"""

import io
import json
import sys
from collections import Counter
from pathlib import Path

# Force UTF-8 output so Unicode symbols render on Windows terminals
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── ANSI colours ──────────────────────────────────────────────────────────────
_USE_COLOUR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"{code}{text}\033[0m" if _USE_COLOUR else text

GRN  = "\033[92m"
YLW  = "\033[93m"
RED  = "\033[91m"
CYN  = "\033[96m"
BLD  = "\033[1m"

PASS_TAG = _c(GRN, "✓ PASS   ")
WARN_TAG = _c(YLW, "⚠ WARNING")
ERR_TAG  = _c(RED,  "✕ ERROR  ")

# ── Pitch constants ────────────────────────────────────────────────────────────
PITCH_X   = 105.0
PITCH_Y   = 68.0

# ── Audit thresholds ──────────────────────────────────────────────────────────
SPIKE_WINDOW_S  = 3      # seconds
SPIKE_THRESHOLD = 4      # events within SPIKE_WINDOW_S → warning
MIN_EVENTS      = 10     # below this is suspicious for a full match

# ── Events that must carry spatial coordinates ────────────────────────────────
SPATIAL_TYPES = {
    "SHOT", "BOX_ENTRY", "DEF_TURNOVER", "HIGH_REGAIN",
    "SET_PIECE", "AERIAL_DUEL", "SECOND_BALL",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Ledger parsing
# ═══════════════════════════════════════════════════════════════════════════════

def extract_events(data: dict | list) -> list[dict]:
    """
    Accept either:
      • A raw tagger array  → return as-is (sorted by match_seconds)
      • A match ledger dict → pull tag objects from matched[] + unmatched_tags[]
    """
    if isinstance(data, list):
        events = data
    else:
        events = [m["tag"] for m in data.get("matched", [])]
        events += data.get("unmatched_tags", [])

    return sorted(events, key=lambda e: e.get("match_seconds", 0))


# ═══════════════════════════════════════════════════════════════════════════════
# Individual checks
# ═══════════════════════════════════════════════════════════════════════════════

def _check_duplicates(events: list[dict]) -> list[tuple]:
    findings = []
    counts = Counter(e.get("id") for e in events)
    for id_, n in counts.items():
        if n > 1:
            findings.append((
                "ERROR", "DUPLICATE_ID",
                f"Event id '{id_}' appears {n} times — each event must be unique",
            ))
    return findings


def _check_chronology(events: list[dict]) -> list[tuple]:
    findings = []
    for i in range(1, len(events)):
        t_prev = events[i - 1].get("match_seconds", 0)
        t_curr = events[i].get("match_seconds", 0)
        if t_curr < t_prev:
            findings.append((
                "ERROR", "TIME_ORDER",
                f"idx {i}: {events[i].get('event_type')} at {t_curr}s "
                f"precedes previous event at {t_prev}s "
                f"(Δ = {t_curr - t_prev}s)  [{events[i].get('clock_display', '?')}]",
            ))
    return findings


def _check_pitch_bounds(events: list[dict]) -> list[tuple]:
    findings = []
    for i, ev in enumerate(events):
        et = ev.get("event_type", "")
        if et == "SUB":
            continue
        xm = ev.get("x_m")
        ym = ev.get("y_m")

        # Null coordinates on spatial events
        if et in SPATIAL_TYPES and (xm is None or ym is None):
            findings.append((
                "WARNING", "NULL_COORDS",
                f"idx {i}: {et} at {ev.get('clock_display', '?')} — "
                "no spatial coordinates recorded",
            ))
            continue

        if xm is None or ym is None:
            continue

        if not (0 <= xm <= PITCH_X):
            findings.append((
                "ERROR", "OUT_OF_BOUNDS",
                f"idx {i}: {et} at {ev.get('clock_display', '?')} — "
                f"x_m = {xm:.2f} is outside [0, {PITCH_X}]",
            ))
        if not (0 <= ym <= PITCH_Y):
            findings.append((
                "ERROR", "OUT_OF_BOUNDS",
                f"idx {i}: {et} at {ev.get('clock_display', '?')} — "
                f"y_m = {ym:.2f} is outside [0, {PITCH_Y}]",
            ))
    return findings


def _check_spatial_plausibility(events: list[dict]) -> list[tuple]:
    findings = []
    for i, ev in enumerate(events):
        et = ev.get("event_type", "")
        xm = ev.get("x_m")
        ym = ev.get("y_m")

        if xm is None or ym is None:
            continue

        # SHOT from own half — x_m is direction-aware (0=own goal, 105=opp goal)
        if et == "SHOT" and xm < 52.5:
            findings.append((
                "WARNING", "DEFENSIVE_SHOT",
                f"idx {i}: SHOT at {ev.get('clock_display', '?')} tagged in own half "
                f"(x_m = {xm:.1f} < 52.5) — verify attacking direction or retag",
            ))

        # HIGH_REGAIN in defensive third — semantically a DEF_TURNOVER, not a high press
        if et == "HIGH_REGAIN" and xm < 35:
            findings.append((
                "WARNING", "LOW_REGAIN",
                f"idx {i}: HIGH_REGAIN at {ev.get('clock_display', '?')} tagged in "
                f"defensive third (x_m = {xm:.1f} < 35) — "
                "should this be DEF_TURNOVER?",
            ))

    return findings


def _check_event_spikes(events: list[dict]) -> list[tuple]:
    """Flag clusters of > SPIKE_THRESHOLD events within SPIKE_WINDOW_S seconds."""
    findings = []
    reported_at: set[int] = set()

    for i, ev in enumerate(events):
        t0 = ev.get("match_seconds", 0)
        # Skip if we already reported a spike that covers this window
        if any(abs(t0 - rt) <= SPIKE_WINDOW_S for rt in reported_at):
            continue
        cluster = [e for e in events if abs(e.get("match_seconds", 0) - t0) <= SPIKE_WINDOW_S]
        if len(cluster) > SPIKE_THRESHOLD:
            reported_at.add(t0)
            type_list = ", ".join(e.get("event_type", "?") for e in cluster)
            findings.append((
                "WARNING", "EVENT_SPIKE",
                f"idx {i}: {len(cluster)} events within {SPIKE_WINDOW_S}s window "
                f"at ~{t0}s — possible multi-tap or clock error  "
                f"[{type_list}]",
            ))
    return findings


def _check_sub_integrity(events: list[dict]) -> list[tuple]:
    findings = []
    for i, ev in enumerate(events):
        if ev.get("event_type") != "SUB":
            continue
        missing = []
        if not ev.get("player_off"):
            missing.append("player_off")
        if not ev.get("player_on"):
            missing.append("player_on")
        if missing:
            findings.append((
                "WARNING", "SUB_INCOMPLETE",
                f"idx {i}: SUB at {ev.get('clock_display', '?')} — "
                f"missing field(s): {', '.join(missing)}",
            ))
    return findings


def _check_event_count(events: list[dict]) -> list[tuple]:
    if len(events) < MIN_EVENTS:
        return [(
            "WARNING", "LOW_COUNT",
            f"Only {len(events)} events total — minimum expected for a full match "
            f"is {MIN_EVENTS}. Possible incomplete session.",
        )]
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

_ALL_CHECKS = {
    "DUPLICATE_ID":    ("Duplicate event IDs",                       "ERROR"),
    "TIME_ORDER":      ("Chronological sequence",                     "ERROR"),
    "OUT_OF_BOUNDS":   ("Pitch boundary integrity",                   "ERROR"),
    "NULL_COORDS":     ("Null coordinates on spatial events",         "WARNING"),
    "DEFENSIVE_SHOT":  ("Shot plausibility (defensive half)",         "WARNING"),
    "LOW_REGAIN":      ("High-regain plausibility (def. third)",      "WARNING"),
    "EVENT_SPIKE":     (f"Event spike (> {SPIKE_THRESHOLD} in {SPIKE_WINDOW_S}s)", "WARNING"),
    "SUB_INCOMPLETE":  ("Substitution integrity",                     "WARNING"),
    "LOW_COUNT":       (f"Minimum event count (< {MIN_EVENTS})",      "WARNING"),
}


def run_checks(events: list[dict]) -> list[tuple]:
    findings = []
    findings += _check_duplicates(events)
    findings += _check_chronology(events)
    findings += _check_pitch_bounds(events)
    findings += _check_spatial_plausibility(events)
    findings += _check_event_spikes(events)
    findings += _check_sub_integrity(events)
    findings += _check_event_count(events)
    return findings


def print_report(findings: list[tuple], n_events: int) -> int:
    """Print terminal dashboard. Returns exit code (0 or 1)."""
    errors   = [f for f in findings if f[0] == "ERROR"]
    warnings = [f for f in findings if f[0] == "WARNING"]

    bar = "═" * 60
    thin = "─" * 60

    print(f"\n{_c(BLD + CYN, bar)}")
    print(f"  {_c(BLD, 'PitchPulse Tagger Sanity Report')}")
    print(f"  Events audited : {n_events}")
    print(f"{_c(CYN, thin)}")

    # Summary table
    for check_key, (label, default_sev) in _ALL_CHECKS.items():
        matched = [f for f in findings if f[1] == check_key]
        if not matched:
            print(f"  {PASS_TAG}  {label}")
        else:
            sev = matched[0][0]
            tag = ERR_TAG if sev == "ERROR" else WARN_TAG
            n   = len(matched)
            print(f"  {tag}  {label}  ({n} issue{'s' if n > 1 else ''})")

    # Diagnostics
    if findings:
        print(f"\n{_c(CYN, thin)}")
        print(f"  {_c(BLD, 'Diagnostics:')}")
        for sev, check, msg in findings:
            tag = ERR_TAG if sev == "ERROR" else WARN_TAG
            print(f"  {tag}  [{check}]  {msg}")

    # Verdict
    print(f"\n{_c(CYN, thin)}")
    if errors:
        verdict = _c(RED + BLD, f"RESULT: {len(errors)} CRITICAL ERROR(S) — fix before analysis")
        exit_code = 1
    elif warnings:
        verdict = _c(YLW + BLD, f"RESULT: {len(warnings)} WARNING(S) — review recommended")
        exit_code = 0
    else:
        verdict = _c(GRN + BLD, "RESULT: ALL CHECKS PASSED ✓")
        exit_code = 0
    print(f"  {verdict}")
    print(f"{_c(CYN, bar)}\n")

    return exit_code


# ═══════════════════════════════════════════════════════════════════════════════
# Built-in unit tests  (python tools/tagger_sanity.py --test)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_tests() -> None:
    import traceback

    PASS_T = _c(GRN, "  PASS")
    FAIL_T = _c(RED, "  FAIL")
    failures = 0

    def assert_check(label: str, events: list, expected_keys: set[str],
                     forbidden_keys: set[str] | None = None) -> None:
        nonlocal failures
        findings = run_checks(events)
        found    = {f[1] for f in findings}
        ok = True
        for k in expected_keys:
            if k not in found:
                print(f"{FAIL_T}  {label} — expected '{k}' not raised")
                ok = False
        for k in (forbidden_keys or set()):
            if k in found:
                print(f"{FAIL_T}  {label} — unexpected '{k}' was raised")
                ok = False
        if ok:
            print(f"{PASS_T}  {label}")
        else:
            failures += 1

    # ── shared valid base event ────────────────────────────────────────────────
    def _ev(i, t, et, x, y, **kw):
        return {
            "id":            str(i),
            "match_seconds": t,
            "clock_display": f"{t//60:02d}:{t%60:02d}",
            "event_type":    et,
            "sub_type":      None,
            "x_m":           x,
            "y_m":           y,
            "team":          "tiverton",
            **kw,
        }

    clean = [
        _ev(1, 300,  "SHOT",        88.0, 34.0),
        _ev(2, 600,  "BOX_ENTRY",   80.0, 12.0),
        _ev(3, 900,  "HIGH_REGAIN", 72.0, 34.0),
        _ev(4, 1200, "DEF_TURNOVER",30.0, 34.0),
        _ev(5, 1500, "AERIAL_DUEL", 52.0, 34.0),
        _ev(6, 1800, "SET_PIECE",   95.0, 2.0),
        _ev(7, 2100, "SECOND_BALL", 65.0, 40.0),
        _ev(8, 2400, "SHOT",        75.0, 34.0),
        _ev(9, 2700, "BOX_ENTRY",   82.0, 55.0),
        _ev(10,3000, "HIGH_REGAIN", 60.0, 34.0),
    ]

    div = "─" * 50
    print(f"\n  {div}")
    print(f"  {_c(BLD, 'PitchPulse Sanity — Unit Tests')}")
    print(f"  {div}")

    # 1. Clean data → no findings
    assert_check("Clean data produces zero findings",
                 clean, set(), {"DUPLICATE_ID","TIME_ORDER","OUT_OF_BOUNDS",
                                "DEFENSIVE_SHOT","LOW_REGAIN","EVENT_SPIKE",
                                "SUB_INCOMPLETE","LOW_COUNT"})

    # 2. Duplicate ID
    dup = clean[:] + [_ev(1, 3600, "SHOT", 80.0, 34.0)]  # id '1' reused
    assert_check("Duplicate ID is caught",
                 dup, {"DUPLICATE_ID"})

    # 3. Backwards time jump
    reversed_time = [
        _ev(1, 600,  "SHOT",     88.0, 34.0),
        _ev(2, 300,  "BOX_ENTRY",80.0, 12.0),  # jumps back
        _ev(3, 900,  "SHOT",     75.0, 34.0),
        _ev(4, 1200, "SHOT",     72.0, 34.0),
        _ev(5, 1500, "SHOT",     80.0, 34.0),
        _ev(6, 1800, "SHOT",     76.0, 34.0),
        _ev(7, 2100, "SHOT",     90.0, 34.0),
        _ev(8, 2400, "SHOT",     88.0, 20.0),
        _ev(9, 2700, "SHOT",     85.0, 45.0),
        _ev(10,3000, "SHOT",     78.0, 34.0),
    ]
    assert_check("Backwards time jump is caught",
                 reversed_time, {"TIME_ORDER"})

    # 4. Out-of-bounds coordinate
    oob = [dict(e) for e in clean]
    oob[0] = _ev(99, 300, "SHOT", 110.0, 34.0)  # x > 105
    assert_check("Out-of-bounds x coordinate caught",
                 oob, {"OUT_OF_BOUNDS"})

    oob2 = [dict(e) for e in clean]
    oob2[0] = _ev(99, 300, "SHOT", 80.0, -5.0)  # y < 0
    assert_check("Out-of-bounds negative y coordinate caught",
                 oob2, {"OUT_OF_BOUNDS"})

    # 5. Defensive SHOT
    def_shot = [dict(e) for e in clean]
    def_shot[0] = _ev(99, 300, "SHOT", 20.0, 34.0)  # own half
    assert_check("Defensive SHOT plausibility warning raised",
                 def_shot, {"DEFENSIVE_SHOT"})

    # 6. HIGH_REGAIN in defensive third
    low_regain = [dict(e) for e in clean]
    low_regain[2] = _ev(99, 900, "HIGH_REGAIN", 18.0, 34.0)
    assert_check("HIGH_REGAIN in defensive third raises LOW_REGAIN",
                 low_regain, {"LOW_REGAIN"})

    # 7. Event spike (5 events at t=1000±1s)
    spike_base = [
        _ev(1,  300, "SHOT",        88.0, 34.0),
        _ev(2,  999, "BOX_ENTRY",   80.0, 12.0),
        _ev(3, 1000, "SHOT",        75.0, 34.0),
        _ev(4, 1001, "HIGH_REGAIN", 70.0, 34.0),
        _ev(5, 1001, "DEF_TURNOVER",30.0, 34.0),
        _ev(6, 1002, "AERIAL_DUEL", 52.0, 34.0),  # 5 events in 3s window
        _ev(7, 2100, "SET_PIECE",   95.0, 2.0),
        _ev(8, 2400, "SECOND_BALL", 65.0, 40.0),
        _ev(9, 2700, "SHOT",        80.0, 34.0),
        _ev(10,3000, "SHOT",        88.0, 34.0),
    ]
    assert_check("Event spike (5 in 3s) raises EVENT_SPIKE",
                 spike_base, {"EVENT_SPIKE"})

    # 8. SUB with missing player_off
    sub_bad = clean + [{
        "id":            "sub1",
        "match_seconds": 3300,
        "clock_display": "55:00",
        "event_type":    "SUB",
        "player_on":     7,
        "player_off":    None,   # missing
        "x_m":           None,
        "y_m":           None,
        "team":          "tiverton",
    }]
    assert_check("SUB missing player_off raises SUB_INCOMPLETE",
                 sub_bad, {"SUB_INCOMPLETE"})

    # 9. Low event count
    tiny = [_ev(1, 300, "SHOT", 80.0, 34.0)]
    assert_check("Low event count (< 10) raises LOW_COUNT",
                 tiny, {"LOW_COUNT"})

    # 10. Null coordinates on spatial event
    null_coords = [dict(e) for e in clean]
    null_coords[1] = {**null_coords[1], "x_m": None, "y_m": None}
    assert_check("Null coords on spatial event raises NULL_COORDS",
                 null_coords, {"NULL_COORDS"})

    print(f"  {div}")
    if failures:
        print(f"\n  {_c(RED + BLD, f'{failures} test(s) FAILED')}\n")
        sys.exit(1)
    else:
        print(f"\n  {_c(GRN + BLD, 'All tests passed ✓')}\n")
        sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_LEDGER = Path("data/processed/match_ledger.json")


def main() -> None:
    args = sys.argv[1:]

    if args and args[0] == "--test":
        _run_tests()
        return  # _run_tests exits internally

    ledger_path = Path(args[0]) if args else DEFAULT_LEDGER

    if not ledger_path.exists():
        print(f"{ERR_TAG}  Ledger not found: {ledger_path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"{ERR_TAG}  Invalid JSON in {ledger_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    events   = extract_events(data)
    findings = run_checks(events)
    exit_code = print_report(findings, len(events))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
