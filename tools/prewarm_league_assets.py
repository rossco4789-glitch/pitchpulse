#!/usr/bin/env python3
"""
tools/prewarm_league_assets.py — cache crest and kit colours for every league opponent before matchday
=====================================================================================================
Runs cv/club_assets.resolve_club_assets for the 21 Southern League Division One South opponents (Tiverton Town
excluded) so Match Setup never waits on Wikipedia. Clubs with a valid meta.json are skipped unless --force.
Uncached lookups are spaced by --delay seconds (default 2.0; each club makes 3-5 requests) for Wikimedia API
etiquette, and cv/club_assets honours HTTP 429 Retry-After. A club that still hits the limit shows OFFLINE; rerun.

Usage:
    python tools/prewarm_league_assets.py                              # fill the gaps
    python tools/prewarm_league_assets.py --force                      # refresh clubs not marked verified
    python tools/prewarm_league_assets.py --force --override-verified  # refresh everything, manual records included

Status:
    CACHED     meta.json was already valid; no network call
    PROTECTED  --force skipped a "verified": true record (pass --override-verified to refetch it)
    RESOLVED   club page found; crest and kit colours saved
    NOT_FOUND  Wikipedia has no matching club page; saved so it is not looked up again
    OFFLINE    the lookup failed on the network; nothing saved, retried on the next run

Exit codes: 0 finished (whatever the statuses), 2 bad arguments.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.club_assets import SOURCES, cached_assets, resolve_club_assets  # noqa: E402
from cv.league_roster import OWN_CLUB, SOUTHERN_LEAGUE_DIV_ONE_SOUTH  # noqa: E402

DEFAULT_DELAY_S = 2.0


def opponents() -> list[str]:
    return [club for club in SOUTHERN_LEAGUE_DIV_ONE_SOUTH if club != OWN_CLUB]


def prewarm(clubs: list[str], sources: Path = SOURCES, force: bool = False, delay: float = DEFAULT_DELAY_S,
            sleep=time.sleep, override_verified: bool = False) -> list[dict]:
    rows, lookups = [], 0
    for club in clubs:
        meta = cached_assets(club, sources)
        if meta and not force:
            status = "CACHED"
        elif meta and meta.get("verified") and not override_verified:
            status = "PROTECTED"
        else:
            if lookups and delay > 0:
                sleep(delay)
            lookups += 1
            outcome = {}
            meta = resolve_club_assets(club, sources=sources, refresh=force, report=outcome,
                                       override_verified=override_verified)
            status = "OFFLINE" if not outcome["saved"] else "RESOLVED" if meta["verified"] else "NOT_FOUND"
        rows.append({"club": club, "status": status, "kit_source": meta.get("kit_source", "default"),
                     "home": meta["home_kit"], "away": meta["away_kit"], "crest": bool(meta.get("badge_path")),
                     "page": meta["name"] if meta.get("verified") else ""})
    return rows


def format_table(rows: list[dict]) -> str:
    header = f"{'Club':<24} {'Status':<10} {'Kit source':<13} {'Home':<8} {'Away':<8} {'Crest':<6} Wikipedia page"
    lines = [header, "-" * len(header)]
    lines += [f"{r['club']:<24} {r['status']:<10} {r['kit_source']:<13} {r['home']:<8} {r['away']:<8} "
              f"{'yes' if r['crest'] else 'no':<6} {r['page']}" for r in rows]
    counts = Counter(r["status"] for r in rows)
    lines.append("-" * len(header))
    lines.append("  ".join(f"{s}: {counts[s]}" for s in ("CACHED", "PROTECTED", "RESOLVED", "NOT_FOUND", "OFFLINE"))
                 + f"  (of {len(rows)})")
    return "\n".join(lines)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description="Cache crest and kit colours for all league opponents.")
    ap.add_argument("--force", action="store_true", help="Refresh clubs that already have meta.json (verified records are kept)")
    ap.add_argument("--override-verified", action="store_true", help="With --force, also refetch records marked verified")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY_S, help="Seconds between uncached lookups (default 2.0)")
    ap.add_argument("--sources", type=Path, default=SOURCES, help="Scouting sources folder (default data/scouting/sources)")
    args = ap.parse_args(argv)
    if args.delay < 0:
        ap.error("--delay must be 0 or more")

    clubs = opponents()
    print(f"Pre-warming {len(clubs)} Southern League Division One South opponents ({OWN_CLUB} excluded)…", flush=True)
    print(format_table(prewarm(clubs, args.sources, args.force, args.delay, override_verified=args.override_verified)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
