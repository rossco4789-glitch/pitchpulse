#!/usr/bin/env python3
"""
highlight_finder.py — PitchPulse public highlight video finder
==============================================================
Finds recent public highlight packages for an opponent so the Opposition Scouting track can log moments
from the reel without leaving the app. Search runs through yt-dlp's YouTube search (metadata only, nothing
is downloaded); results are ranked and cached per club.

Usage:
    python tools/highlight_finder.py --opponent "Sholing"
    python tools/highlight_finder.py --opponent "Sholing" --refresh      # ignore the cache and search again

Ranking (title and channel only; YouTube search does not return upload dates):
    dropped   the opponent's name is in neither the title nor the channel name
    +10+i     the title names an opponent from data/scouting/evidence/<slug>.json results; i grows with recency
    +3        the title says "highlights"
    +1        uploaded by the opponent's own channel
    -3        pre-season, friendly, women's, youth or reserve side
    -2        shorter than 90 seconds or longer than 30 minutes

Output:
    data/scouting/sources/<slug>/video_links.json
    {"opponent", "query", "fetched_at", "videos": [{"title", "url", "channel", "duration_s", "score", "fixture"}],
     "warnings"}

Contract:
    find_highlights never raises. A failed search keeps the previous cache (or returns an empty list) and
    records the reason in "warnings".

Exit codes: 0 finished (read "warnings"), 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[1]
SOURCES     = ROOT / "data" / "scouting" / "sources"
EVIDENCE    = ROOT / "data" / "scouting" / "evidence"
LINKS_FILE  = "video_links.json"
SEARCH_SIZE = 20
NOT_A_MATCH = re.compile(r"(?i)pre[- ]?season|friendly|\bwomen|\bladies|\bu\d{2}s?\b|under[- ]\d{2}|youth|academy|reserves")


def slugify(name: str) -> str:
    """Same folder name as cv/club_assets.slugify."""
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).replace(" fc ", " ").split())


def links_path(sources: Path, opponent: str) -> Path:
    return Path(sources) / slugify(opponent) / LINKS_FILE


# ── Search ───────────────────────────────────────────────────────────────────

def youtube_search(query: str, limit: int = SEARCH_SIZE) -> list[dict]:
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    return [e for e in (info or {}).get("entries") or [] if e]


def recent_opponents(opponent: str, evidence: Path = EVIDENCE) -> list[dict]:
    """Played fixtures from the scout evidence, oldest first: [{"opponent", "date"}]."""
    try:
        ev = json.loads((Path(evidence) / f"{slugify(opponent)}.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [{"opponent": f["opponent"], "date": f["date"]} for f in ev.get("fixtures", []) if f.get("played")]


def rank_highlights(opponent: str, entries: list[dict], recent: list[dict] = ()) -> list[dict]:
    club = _norm(opponent)
    ranked = []
    for e in entries:
        title, channel = e.get("title") or "", e.get("channel") or e.get("uploader") or ""
        url = e.get("url") or (f"https://www.youtube.com/watch?v={e['id']}" if e.get("id") else None)
        if not url or club not in f" {_norm(title)} " and club not in f" {_norm(channel)} ":
            continue
        score, fixture = 0, None
        for i, fx in enumerate(recent):
            if _norm(fx["opponent"]) in _norm(title):
                score, fixture = 10 + i, f"{fx['date']} v {fx['opponent']}"   # later fixtures overwrite earlier
        if "highlight" in title.lower():
            score += 3
        if club in _norm(channel):
            score += 1
        if NOT_A_MATCH.search(title):
            score -= 3
        duration = e.get("duration")
        if duration and not 90 <= duration <= 1800:
            score -= 2
        ranked.append({"title": title, "url": url, "channel": channel,
                       "duration_s": int(duration) if duration else None, "score": score, "fixture": fixture})
    return sorted(ranked, key=lambda v: (-v["score"], -(v["duration_s"] or 0)))


# ── Cache ────────────────────────────────────────────────────────────────────

def cached_highlights(opponent: str, sources: Path = SOURCES) -> dict | None:
    try:
        data = json.loads(links_path(sources, opponent).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("videos"), list) else None


def find_highlights(opponent: str, sources: Path = SOURCES, refresh: bool = False, search=youtube_search,
                    evidence: Path = EVIDENCE) -> dict:
    name = " ".join((opponent or "").split())
    cached = cached_highlights(name, sources) if name else None
    if cached and not refresh:
        return cached
    query = f"{name} highlights"
    try:
        entries = search(query)
    except Exception as exc:  # yt-dlp raises its own error types; a failed search must not break the tab
        keep = cached or {"opponent": name, "query": query, "fetched_at": None, "videos": []}
        return {**keep, "warnings": [f"highlight search failed ({type(exc).__name__}); showing the last saved list"]}
    data = {"opponent": name, "query": query, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "videos": rank_highlights(name, entries, recent_opponents(name, evidence)), "warnings": []}
    path = links_path(sources, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except AttributeError:
            pass

    ap = argparse.ArgumentParser(description="Find and rank public highlight videos for an opponent.")
    ap.add_argument("--opponent", required=True, help='Club name, e.g. "Sholing"')
    ap.add_argument("--refresh", action="store_true", help="Search again even if a saved list exists")
    args = ap.parse_args(argv)
    if not slugify(args.opponent):
        print("  [ERROR] --opponent must contain letters or digits", file=sys.stderr)
        return 2

    data = find_highlights(args.opponent, refresh=args.refresh)
    print(f"  {data['opponent']} · {len(data['videos'])} videos · saved {links_path(SOURCES, args.opponent)}")
    for v in data["videos"][:8]:
        print(f"  {v['score']:>3}  {v['title']}  [{v['channel']}]  {v['fixture'] or ''}")
    for w in data.get("warnings", []):
        print(f"  [WARN] {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
