"""
data/parse_report.py
Parse a Tiverton Town match report Word doc (.docx) into:
  - data/raw/tivvy_x_feed.json   (goal/sub events for reconcile/sync.py)
  - data/raw/match_context.json  (full tactical context for agents/synthesis.py)

Usage:
    python data/parse_report.py <path/to/report.docx>
    python data/parse_report.py <path/to/report.docx> --out-dir data/raw
"""

import argparse
import json
import re
import sys
from pathlib import Path

import docx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_minute(raw: str) -> int:
    """'45+3' -> 48, '77' -> 77. Returns 999 if unparseable."""
    m = re.match(r"(\d+)(?:\+(\d+))?", raw.strip())
    if not m:
        return 999
    return int(m.group(1)) + (int(m.group(2)) if m.group(2) else 0)


def _parse_scorers(text: str) -> list:
    """'Slough 77, Hall 85' -> [{'player':'Slough','minute':77,'minute_raw':'77'}, ...]"""
    results = []
    for part in text.split(","):
        part = part.strip()
        m = re.match(r"(.+?)\s+(\d+(?:\+\d+)?)\s*$", part)
        if m:
            results.append({
                "player": m.group(1).strip(),
                "minute": _parse_minute(m.group(2)),
                "minute_raw": m.group(2),
            })
    return results


def _parse_lineup(text: str) -> list:
    """
    'Tiverton: Smith(GK), Winter, Wood(65), Palmer(C), ...'
    Returns list of dicts with player, role, captain, subbed_off.
    """
    text = re.sub(r"^Tiverton\s*:\s*", "", text)
    starters = []
    for part in text.split(","):
        part = part.strip().rstrip(".")
        annotations = re.findall(r"\(([^)]+)\)", part)
        name = re.sub(r"\([^)]*\)", "", part).strip()
        if not name:
            continue
        entry = {"player": name, "role": None, "captain": False, "subbed_off": None}
        for ann in annotations:
            if ann == "GK":
                entry["role"] = "GK"
            elif ann == "C":
                entry["captain"] = True
            elif ann.isdigit():
                entry["subbed_off"] = int(ann)
        starters.append(entry)
    return starters


def _parse_subs(text: str) -> list:
    """'Horne(65), Coulibaly(65), Koerner(86), Pryce-Hall' -> list of dicts"""
    text = re.sub(r"^Subs\s*:\s*", "", text)
    subs = []
    for part in text.split(","):
        part = part.strip()
        m = re.match(r"(.+?)\((\d+)\)", part)
        if m:
            subs.append({"player": m.group(1).strip(), "minute": int(m.group(2)), "used": True})
        elif part:
            subs.append({"player": part, "minute": None, "used": False})
    return subs


def _extract_minute_events(paragraphs: list) -> list:
    """
    Scan narrative for minute references and return context snippets.
    Useful for agents to correlate spatial tagger events with prose.
    """
    pattern = re.compile(
        r"\b(?:(\d+)(?:st|nd|rd|th)\s+minute|(\d+)\s*\'|minute\s+(\d+))\b",
        re.IGNORECASE,
    )
    seen = set()
    events = []
    for para in paragraphs:
        for m in pattern.finditer(para):
            minute_str = m.group(1) or m.group(2) or m.group(3)
            minute = int(minute_str)
            if minute not in seen:
                seen.add(minute)
                events.append({"minute": minute, "context": para[:250]})
    return sorted(events, key=lambda x: x["minute"])


_TACTICAL_TERMS = [
    "press", "pressing", "counter", "transition", "compact",
    "half-space", "high line", "low block", "set piece", "corner",
    "free kick", "cross", "direct", "aerial", "second ball",
    "overlap", "overload", "channel", "wide", "flank",
    "block", "regain", "turnover", "tempo", "shape",
]


def _extract_tactical_keywords(paragraphs: list) -> list:
    full_text = " ".join(paragraphs).lower()
    return [t for t in _TACTICAL_TERMS if t in full_text]


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_report(docx_path: Path) -> dict:
    doc = docx.Document(str(docx_path))
    non_empty = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    # --- Header ---
    date        = non_empty[1] if len(non_empty) > 1 else ""
    competition = non_empty[2] if len(non_empty) > 2 else ""
    venue_line  = non_empty[3] if len(non_empty) > 3 else ""
    home_game   = not venue_line.lower().startswith("at ")
    venue       = re.sub(r"^[Aa]t\s+", "", venue_line) if not home_game else venue_line

    # --- Score block ---
    # Find "Tiverton Town N" line — guaranteed to exist
    tiverton_score_line = next(
        (p for p in non_empty if re.match(r"Tiverton Town\s+\d+\s*$", p)), ""
    )
    tiverton_m = re.match(r"Tiverton Town\s+(\d+)\s*$", tiverton_score_line)
    tiverton_score = int(tiverton_m.group(1)) if tiverton_m else 0

    tiverton_idx = non_empty.index(tiverton_score_line) if tiverton_score_line in non_empty else -1

    # Structure: [..., 'St Blazey 1', 'Redd 45+3', 'Tiverton Town 2', 'Slough 77, Hall 85', ...]
    # Opponent team+score is 2 lines before Tiverton; opponent scorers 1 line before
    opp_score_line      = non_empty[tiverton_idx - 2] if tiverton_idx >= 2 else ""
    opp_scorers_raw     = non_empty[tiverton_idx - 1] if tiverton_idx >= 1 else ""
    tiverton_scorers_raw = non_empty[tiverton_idx + 1] if tiverton_idx >= 0 and tiverton_idx + 1 < len(non_empty) else ""

    opp_m = re.match(r"(.+?)\s+(\d+)\s*$", opp_score_line)
    opponent       = opp_m.group(1).strip() if opp_m else ""
    opponent_score = int(opp_m.group(2))    if opp_m else 0

    result = (
        "W" if tiverton_score > opponent_score else
        "D" if tiverton_score == opponent_score else "L"
    )

    # --- Lineup & subs ---
    lineup_para = next((p for p in non_empty if p.startswith("Tiverton:")), "")
    subs_para   = next((p for p in non_empty if p.startswith("Subs:")), "")
    starters    = _parse_lineup(lineup_para) if lineup_para else []
    subs        = _parse_subs(subs_para)     if subs_para   else []

    # --- Narrative (between score block and lineup) ---
    lineup_idx = non_empty.index(lineup_para) if lineup_para in non_empty else len(non_empty)
    narrative_start = tiverton_idx + 2  # skip scorer line
    narrative_paras = non_empty[narrative_start:lineup_idx]

    return {
        "date":        date,
        "competition": competition,
        "venue":       venue,
        "home_game":   home_game,
        "opponent":    opponent,
        "score": {
            "tiverton": tiverton_score,
            "opponent": opponent_score,
        },
        "result": result,
        "scorers": {
            "tiverton": _parse_scorers(tiverton_scorers_raw),
            "opponent": _parse_scorers(opp_scorers_raw),
        },
        "lineup": starters,
        "subs":   subs,
        "minute_events":      _extract_minute_events(narrative_paras),
        "tactical_keywords":  _extract_tactical_keywords(narrative_paras),
        "narrative_paragraphs": narrative_paras,
    }


# ---------------------------------------------------------------------------
# Build x_feed.json (consumed by reconcile/sync.py)
# ---------------------------------------------------------------------------

def build_x_feed(ctx: dict) -> list:
    """Emit goal + sub events in the format reconcile/sync.py expects."""
    events = []

    for s in ctx["scorers"]["tiverton"]:
        tiv = ctx["score"]["tiverton"]
        opp = ctx["score"]["opponent"]
        events.append({
            "minute": f"{s['minute_raw']}'",
            "type":   "GOAL",
            "detail": f"GOAL! {s['player']} scores for Tiverton Town! ({tiv}-{opp})",
        })

    for s in ctx["scorers"]["opponent"]:
        events.append({
            "minute": f"{s['minute_raw']}'",
            "type":   "OPP_GOAL",
            "detail": f"GOAL! {s['player']} scores for {ctx['opponent']}.",
        })

    for sub in ctx["subs"]:
        if sub["used"] and sub["minute"]:
            events.append({
                "minute": f"{sub['minute']}'",
                "type":   "SUBSTITUTION",
                "detail": f"Sub: {sub['player']} comes on.",
            })

    events.sort(key=lambda e: _parse_minute(e["minute"].rstrip("'")))
    return events


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Parse Tiverton Town match report .docx")
    ap.add_argument("docx_path", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    if not args.docx_path.exists():
        print(f"ERROR: {args.docx_path} not found", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ctx    = parse_report(args.docx_path)
    x_feed = build_x_feed(ctx)

    ctx_path  = args.out_dir / "match_context.json"
    feed_path = args.out_dir / "tivvy_x_feed.json"

    ctx_path.write_text(json.dumps(ctx, indent=2, ensure_ascii=False), encoding="utf-8")
    feed_path.write_text(json.dumps(x_feed, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    scorers_str = ", ".join(f"{s['player']} {s['minute']}'" for s in ctx["scorers"]["tiverton"])
    subs_used   = sum(1 for s in ctx["subs"] if s["used"])
    print(f"✓ match_context.json → {ctx_path}")
    print(f"✓ tivvy_x_feed.json  → {feed_path}")
    print()
    print(f"  Result     : {ctx['result']} | {ctx['opponent']} {ctx['score']['opponent']}–{ctx['score']['tiverton']} Tiverton Town")
    print(f"  Competition: {ctx['competition']}")
    print(f"  Venue      : {ctx['venue']} ({'home' if ctx['home_game'] else 'away'})")
    print(f"  Scorers    : {scorers_str or 'none parsed'}")
    print(f"  Lineup     : {len(ctx['lineup'])} players | {subs_used} subs used")
    print(f"  Minute refs: {len(ctx['minute_events'])} events")
    print(f"  Keywords   : {ctx['tactical_keywords']}")


if __name__ == "__main__":
    main()
