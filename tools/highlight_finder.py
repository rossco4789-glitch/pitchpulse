#!/usr/bin/env python3
"""
highlight_finder.py — PitchPulse match reel finder
==================================================
Finds one public highlight reel for each of an opponent's last completed fixtures so the Opposition Scouting
track can log moments match by match. Search runs through yt-dlp's YouTube search (metadata only, nothing is
downloaded).

Usage:
    python tools/highlight_finder.py --opponent "Sholing"
    python tools/highlight_finder.py --opponent "Sholing" --refresh      # ignore the saved reels and search again

Fixtures:
    The last 5 played fixtures in data/scouting/evidence/<slug>.json (run tools/scout_fetcher.py first), newest first.

Search, per fixture:
    '"<opponent>" "<rival>" highlights'; if no video names both clubs, retried once without quotes.

Pick, per fixture. YouTube search returns no upload dates and the same clubs meet most seasons, so a video
must prove it shows this fixture:
    required  both clubs named (in the title, or one in the title and the other as the channel)
    required  the fixture score in the title (either order, e.g. 1-3 or 3 – 1), or a date from the evidence season
    rejected  any other score in the title, a year outside the season, or a pre-season, friendly, women's,
              youth or reserve game
Among the videos left: +3 the title says "highlights", +2 uploaded by either club, -2 shorter than 90 seconds
or longer than 30 minutes.

Output:
    data/scouting/sources/<slug>/video_links.json — a list, newest fixture first:
    [{"match_date", "fixture_label", "score", "title", "url", "channel"}]

Contract:
    find_match_reels never raises. Saved reels are reused until the evidence file is newer (a new result was
    fetched) or refresh=True. A failed search keeps that fixture's previously saved reel and records the reason
    in "warnings"; fixtures with no reel are left out of the list.

Exit codes: 0 finished (read warnings), 1 no played fixtures in the evidence file, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[1]
SOURCES     = ROOT / "data" / "scouting" / "sources"
EVIDENCE    = ROOT / "data" / "scouting" / "evidence"
LINKS_FILE  = "video_links.json"
LAST        = 5
SEARCH_SIZE = 10
NOT_A_MATCH = re.compile(r"(?i)pre[- ]?season|friendly|\bwomen|\bladies|\bu\d{2}s?\b|under[- ]\d{2}|youth|academy|reserves")
CLUB_SUFFIX = re.compile(r"\s+(?:town|city|united|borough|rovers|athletic|fc|afc)$")
YEAR        = re.compile(r"\b(20\d\d)\b")
SCORE       = re.compile(r"(?<!\d)(\d{1,2})\s*[-–]\s*(\d{1,2})(?!\d)")
SHORT_DATE  = re.compile(r"\b\d{1,2}[./]\d{1,2}[./](\d{2})\b")


def slugify(name: str) -> str:
    """Same folder name as cv/club_assets.slugify."""
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")


def _norm(text: str) -> str:
    return " " + " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()) + " "


def _names(club: str) -> set[str]:
    """Full club name plus the short form fans use ('Bracknell Town' → 'Bracknell')."""
    full = _norm(club).strip()
    short = CLUB_SUFFIX.sub("", full)
    return {full, short} if len(short) >= 4 else {full}


def _mentions(text: str, club: str) -> bool:
    return any(f" {name} " in _norm(text) for name in _names(club))


def links_path(sources: Path, opponent: str) -> Path:
    return Path(sources) / slugify(opponent) / LINKS_FILE


# ── Fixtures ─────────────────────────────────────────────────────────────────

def last_fixtures(opponent: str, evidence: Path = EVIDENCE, last: int = LAST) -> tuple[list[dict], set[str]]:
    """Last `last` played fixtures, newest first, and the evidence season's years."""
    try:
        ev = json.loads((Path(evidence) / f"{slugify(opponent)}.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], set()
    played = [f for f in ev.get("fixtures", []) if f.get("played")][-last:] if last else []
    fixtures = [{"opponent": opponent, "rival": f["opponent"], "home": f.get("venue") == "home", "match_date": f["date"],
                 "goals_for": f.get("goals_for"), "goals_against": f.get("goals_against")} for f in reversed(played)]
    season = (ev.get("league") or {}).get("season") or ""
    return fixtures, set(YEAR.findall(season.replace("-", " ")))


def fixture_label(fx: dict) -> str:
    return f"{fx['opponent']} v {fx['rival']}" if fx["home"] else f"{fx['rival']} v {fx['opponent']}"


def score_line(fx: dict) -> str:
    """Home goals first, like the fixture label."""
    if fx.get("goals_for") is None or fx.get("goals_against") is None:
        return ""
    gf, ga = fx["goals_for"], fx["goals_against"]
    return f"{gf}-{ga}" if fx["home"] else f"{ga}-{gf}"


def fixture_query(fx: dict) -> str:
    return f'"{fx["opponent"]}" "{fx["rival"]}" highlights'


# ── Search and pick ──────────────────────────────────────────────────────────

def youtube_search(query: str, limit: int = SEARCH_SIZE) -> list[dict]:
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    return [e for e in (info or {}).get("entries") or [] if e]


def _proves_fixture(title: str, fx: dict, season_years: set[str]) -> bool:
    if NOT_A_MATCH.search(title) or any(year not in season_years for year in YEAR.findall(title)):
        return False
    scores = {(int(h), int(a)) for h, a in SCORE.findall(title)}
    gf, ga = fx.get("goals_for"), fx.get("goals_against")
    if gf is not None and ga is not None and scores:
        return bool(scores & {(gf, ga), (ga, gf)})          # a different score is a different meeting
    dated = any(year in season_years for year in YEAR.findall(title)) or any(
        f"20{yy}" in season_years for yy in SHORT_DATE.findall(title))
    return bool(season_years) and dated


def pick_reel(fx: dict, entries: list[dict], season_years: set[str] = frozenset()) -> dict | None:
    """Best video that proves it shows this fixture, or None."""
    best, best_points = None, None
    for e in entries:
        title, channel = e.get("title") or "", e.get("channel") or e.get("uploader") or ""
        url = e.get("url") or (f"https://www.youtube.com/watch?v={e['id']}" if e.get("id") else None)
        titled = [_mentions(title, club) for club in (fx["opponent"], fx["rival"])]
        by_club = [_mentions(channel, club) for club in (fx["opponent"], fx["rival"])]
        if not url or not (all(titled) or (titled[0] and by_club[1]) or (titled[1] and by_club[0])):
            continue
        if not _proves_fixture(title, fx, season_years):
            continue
        points = (3 if "highlight" in title.lower() else 0) + (2 if any(by_club) else 0)
        duration = e.get("duration")
        if duration and not 90 <= duration <= 1800:
            points -= 2
        if best_points is None or points > best_points:
            best_points = points
            best = {"match_date": fx["match_date"], "fixture_label": fixture_label(fx), "score": score_line(fx),
                    "title": title, "url": url, "channel": channel}
    return best


# ── Cache ────────────────────────────────────────────────────────────────────

def cached_reels(opponent: str, sources: Path = SOURCES) -> list[dict] | None:
    try:
        data = json.loads(links_path(sources, opponent).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, list) and all(isinstance(r, dict) and r.get("url") for r in data) else None


def find_match_reels(opponent: str, sources: Path = SOURCES, refresh: bool = False, search=youtube_search,
                     evidence: Path = EVIDENCE, last: int = LAST) -> dict:
    """{"reels": [...], "warnings": [...]}; the reels are saved to video_links.json."""
    name = " ".join((opponent or "").split())
    fixtures, season_years = last_fixtures(name, evidence, last)
    cached = cached_reels(name, sources)
    if not fixtures:
        return {"reels": cached or [], "warnings": [f"No played fixtures for {name} yet. Fetch their results first."]}
    links, evidence_file = links_path(sources, name), Path(evidence) / f"{slugify(name)}.json"
    if cached is not None and not refresh and links.stat().st_mtime >= evidence_file.stat().st_mtime:
        return {"reels": cached, "warnings": []}       # saved after the latest results: nothing new to search for
    saved = {(r.get("match_date"), r.get("fixture_label")): r for r in cached or []}
    wanted = [(fx["match_date"], fixture_label(fx)) for fx in fixtures]

    reels, warnings = [], []
    for fx, (date, label) in zip(fixtures, wanted):
        try:
            reel = pick_reel(fx, search(fixture_query(fx)), season_years)
            if reel is None:
                reel = pick_reel(fx, search(f"{fx['opponent']} {fx['rival']} highlights"), season_years)
        except Exception as exc:  # yt-dlp raises its own error types; one failed search must not lose the rest
            reel = saved.get((date, label))
            warnings.append(f"Search failed for {label} ({date}, {type(exc).__name__}); "
                            + ("kept the saved reel." if reel else "no reel saved."))
        else:
            if reel is None:
                warnings.append(f"No highlight reel found for {label} ({date}).")
        if reel:
            reels.append(reel)
    links.parent.mkdir(parents=True, exist_ok=True)
    links.write_text(json.dumps(reels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"reels": reels, "warnings": warnings}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except AttributeError:
            pass

    ap = argparse.ArgumentParser(description="Find one highlight reel for each of an opponent's last 5 fixtures.")
    ap.add_argument("--opponent", required=True, help='Club name, e.g. "Sholing"')
    ap.add_argument("--refresh", action="store_true", help="Search again even if reels are saved")
    args = ap.parse_args(argv)
    if not slugify(args.opponent):
        print("  [ERROR] --opponent must contain letters or digits", file=sys.stderr)
        return 2
    if not last_fixtures(args.opponent)[0]:
        print(f"  [ERROR] no played fixtures in {EVIDENCE / (slugify(args.opponent) + '.json')}; run scout_fetcher.py",
              file=sys.stderr)
        return 1

    data = find_match_reels(args.opponent, refresh=args.refresh)
    print(f"  {args.opponent} · {len(data['reels'])} match reels · saved {links_path(SOURCES, args.opponent)}")
    for r in data["reels"]:
        print(f"  {r['match_date']:<11} {r['fixture_label']} {r['score']:<5} → {r['title']}  [{r['channel']}]")
    for w in data["warnings"]:
        print(f"  [WARN] {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
