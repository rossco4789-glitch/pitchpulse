#!/usr/bin/env python3
"""
fixture_agent.py — PitchPulse next-fixture resolver and preview scaffolder
=========================================================================
Finds Tiverton Town's next match, writes a preview stub the scout brief builder can read, and warms the
opponent's crest, kit colours and public evidence before the analyst opens the app.

Usage:
    python tools/fixture_agent.py                 # resolve, scaffold the stub, sync assets and evidence
    python tools/fixture_agent.py --no-sync       # resolve and scaffold only (no Wikipedia or evidence fetch)

Sources, in order:
    data/scouting/fixtures.json   optional manual manifest: a list of fixture dicts in the schema below
                                  ("date" as YYYY-MM-DD); the earliest one on or after today wins
    footballwebpages.co.uk        <team>/fixtures-results; first unplayed, not-postponed row

Fixture schema:
    {"opponent": "Sholing", "home_away": "Home", "venue": "Ladysmead", "date": "2026-09-19",
     "date_str": "Sat 19 Sep 2026", "kickoff": "15:00", "competition": "FA Cup", "slug": "sholing"}

Contract:
    get_next_fixture returns None when no source lists an upcoming match; it never raises on network or
    parse failure. scaffold_preview_stub never overwrites a stub the manager has already filled in.

Exit codes: 0 fixture resolved, 1 no upcoming fixture found.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import scout_fetcher as sf  # noqa: E402
from scout_brief import MONTHS, _resolve_year, slugify  # noqa: E402

CLUB         = "Tiverton Town"
HOME_GROUND  = "Ladysmead"
MANIFEST     = ROOT / "data" / "scouting" / "fixtures.json"
PREVIEWS     = ROOT / "data" / "scouting" / "previews"
FETCHER_PATH = ROOT / "tools" / "scout_fetcher.py"
FWP_DATE     = re.compile(r"(?i)(\d{1,2})\s+([a-z]{3})")
FWP_KICKOFF  = re.compile(r"(?i)^(\d{1,2})(?:[.:](\d{2}))?\s*(am|pm)$")
POSTPONED    = re.compile(r"(?i)^\s*P\s*-\s*P\s*$")


# ── Normalisers ──────────────────────────────────────────────────────────────

def competition_name(raw: str) -> str:
    """'FA Cup 2Q' → 'FA Cup', 'FA Trophy 3Q' → 'FA Trophy', 'Southern South' → 'Southern League'; else raw."""
    text = (raw or "").strip()
    low = text.lower()
    if "fa cup" in low:
        return "FA Cup"
    if "fa trophy" in low:
        return "FA Trophy"
    if "southern" in low:
        return "Southern League"
    return text


def _kickoff(raw: str) -> str:
    m = FWP_KICKOFF.match((raw or "").strip())
    if not m:
        return "TBC"
    hh, mm = int(m.group(1)), int(m.group(2) or 0)
    if m.group(3).lower() == "pm" and hh < 12:
        hh += 12
    return f"{hh:02d}:{mm:02d}"


def _fixture(opponent: str, home: bool, when: date, kickoff: str, competition: str, venue: str | None) -> dict:
    return {
        "opponent": opponent,
        "home_away": "Home" if home else "Away",
        "venue": venue or (HOME_GROUND if home else f"{opponent} (away)"),
        "date": when.isoformat(),
        "date_str": when.strftime("%a %d %b %Y").replace(" 0", " "),
        "kickoff": kickoff,
        "competition": competition,
        "slug": slugify(opponent),
    }


# ── Sources ──────────────────────────────────────────────────────────────────

def from_manifest(path: Path, today: date) -> dict | None:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    upcoming = []
    for row in rows if isinstance(rows, list) else []:
        try:
            when = date.fromisoformat(row["date"])
            home = str(row["home_away"]).lower() == "home"
            fx = _fixture(row["opponent"], home, when, row.get("kickoff") or "TBC",
                          competition_name(row.get("competition", "")), row.get("venue"))
        except (KeyError, TypeError, ValueError):
            continue
        if when >= today:
            upcoming.append(fx)
    return min(upcoming, key=lambda f: f["date"]) if upcoming else None


def parse_next_fixture(page: str, today: date) -> dict | None:
    """First unplayed, not-postponed footballwebpages row dated on or after today."""
    rows = sf._safe(sf.Fetcher(offline=True), "fixtures", sf.parse_fixtures, page) or []
    for row in rows:
        if row["played"] or POSTPONED.match(row["raw_score"]):
            continue
        m = FWP_DATE.search(row["date"])
        if not m or m.group(2).lower() not in MONTHS:
            continue
        when = _resolve_year(MONTHS.index(m.group(2).lower()) + 1, int(m.group(1)), None, today)
        if when is None or when < today:
            continue
        return _fixture(row["opponent"], row["venue"] == "home", when, _kickoff(row["raw_score"]),
                        competition_name(row["competition"]), None)
    return None


def get_next_fixture(team_name: str = CLUB, *, today: date | None = None, manifest: Path = MANIFEST,
                     fetcher=None) -> dict | None:
    today = today or date.today()
    fx = from_manifest(manifest, today)
    if fx:
        return fx
    fetcher = fetcher or sf.Fetcher()
    page = fetcher.get(urljoin(sf.FWP, f"{sf._slug(team_name)}/fixtures-results"))
    return parse_next_fixture(page, today) if page else None


# ── Preview stub ─────────────────────────────────────────────────────────────

def preview_markdown(fixture: dict) -> str:
    home, away = (CLUB, fixture["opponent"]) if fixture["home_away"] == "Home" else (fixture["opponent"], CLUB)
    return (
        f"{home} v {away} — {fixture['date_str']}, {fixture['kickoff']}, {fixture['venue']}\n"
        f"Kick off {fixture['kickoff']}\n"
        f"At {fixture['venue']}\n"
        f"Competition: {fixture['competition']}\n"
        "\n## Formation\n- \n"
        "\n## Key Players\n- \n"
        "\n## Manager Notes\n- \n"
    )


def scaffold_preview_stub(fixture: dict, previews: Path = PREVIEWS) -> str:
    """Write previews/<slug>.md and return its path. An existing stub is kept so manager notes survive."""
    path = Path(previews) / f"{fixture['slug']}.md"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(preview_markdown(fixture), encoding="utf-8")
    return str(path)


# ── Sync ─────────────────────────────────────────────────────────────────────

def sync_fixture_to_scout(fixture: dict, *, resolve=None, run=subprocess.run) -> dict:
    """Cache crest and kit colours (any division) and dispatch scout_fetcher. Never raises."""
    if resolve is None:
        from cv.club_assets import resolve_club_assets as resolve
    assets = resolve(fixture["opponent"])
    cmd = [sys.executable, str(FETCHER_PATH), "--opponent", fixture["opponent"]]
    try:
        code = run(cmd, cwd=str(ROOT)).returncode
    except OSError as exc:
        print(f"  [WARN] scout_fetcher did not start ({exc})", file=sys.stderr)
        code = None
    return {"assets": assets, "fetcher_exit": code}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except AttributeError:
            pass

    ap =argparse.ArgumentParser(description="Resolve the next fixture and scaffold its preview stub.")
    ap.add_argument("--team", default=CLUB, help=f"Club to resolve (default {CLUB})")
    ap.add_argument("--no-sync", action="store_true", help="Skip crest/kit lookup and evidence fetch")
    args = ap.parse_args(argv)

    fx = get_next_fixture(args.team)
    if not fx:
        print("  [ERROR] no upcoming fixture in the manifest or the fixtures feed", file=sys.stderr)
        return 1
    print(f"  Next fixture: {fx['opponent']} ({fx['home_away']}) · {fx['competition']} · "
          f"{fx['date_str']} {fx['kickoff']} · {fx['venue']}")
    print(f"  Preview stub: {scaffold_preview_stub(fx)}")
    if not args.no_sync:
        res = sync_fixture_to_scout(fx)
        print(f"  Club assets: {res['assets'].get('kit_source')} · scout_fetcher exit {res['fetcher_exit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
