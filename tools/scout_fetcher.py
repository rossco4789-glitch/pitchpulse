#!/usr/bin/env python3
"""
scout_fetcher.py — PitchPulse public match-evidence fetcher
============================================================
Off-pitch, pre-match tool. Pulls verifiable public facts about an opponent so the
scouting brief cites match evidence instead of staff assumptions.

Usage:
    python tools/scout_fetcher.py --opponent "Dorchester Town" \\
        --club-site https://www.dorchestertownfc.co.uk \\
        --players "Corby Moore,Ollie Haste,Will Spetch" --last 3
    python tools/scout_fetcher.py ... --offline        # cached pages only, no network

Sources (robots.txt checked, identified User-Agent, max 1 request/second):
    footballwebpages.co.uk   league table, fixtures/results, match pages (team sheets, events)
    --club-site              player profiles, post-match reaction articles

Output (both gitignored — they hold third-party text):
    data/scouting/evidence/<slug>.json
    data/scouting/cache/<host>/<hash>.html     raw page cache used by --offline and on fetch failure

Contract:
    HTTP errors, timeouts, network errors, robots.txt refusals and malformed pages never
    raise. Each is printed to stderr and recorded in evidence["warnings"]; the page falls
    back to its cached copy, else that section is left empty. The fetcher records only
    what a source states: formation, in-match positions, corner takers and aerial duel
    counts are not published by these sources and are listed under "not_published".

Exit codes:
    0  evidence written (read "warnings" for partial coverage)
    2  bad arguments
"""

import argparse
import hashlib
import html
import json
import re
import socket
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

ROOT       = Path(__file__).resolve().parents[1]
CACHE_DIR  = ROOT / "data" / "scouting" / "cache"
OUT_DIR    = ROOT / "data" / "scouting" / "evidence"
FWP        = "https://www.footballwebpages.co.uk/"
USER_AGENT = "PitchPulse-scout/0.1 (+Tiverton Town FC performance analysis)"
TIMEOUT_S  = 15
DELAY_S    = 1.0
NETWORK_ERRORS = (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError, ValueError)
VOID_TAGS  = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
NOT_PUBLISHED = ["formation", "in-match player positions", "corner and free-kick takers", "aerial duel counts"]

TACTICAL = re.compile(
    r"(?i)set[- ]?plays?|set[- ]?pieces?|\bcorners?\b|free[- ]?kicks?|\bheaders?\b|aerial|in the air|"
    r"counter|second[- ]phase|second balls?|back post|near post|far post|\bpress\w*|\bshape\b|formation|"
    r"dictate|tempo|marshal\w*|offside|edge of the box|clean sheet|defend\w*|territory|"
    r"\bcross(?:es|ed)?\b|cut[- ]?backs?|pulled? (?:it |the ball )?back|scrambl\w*|goalmouth|"
    r"on the (?:right|left)\b|down the (?:right|left)\b|\bwinner\b|equali[sz]\w*"
)
REPORT_LINK = re.compile(r'href="((?:https?://[^"/]+)?/(?:news/reaction|match-report)-[^"#?]+)"')
NEWS_PAGES  = 4
SCORE = re.compile(r"^(?:\((\d+)\))?\s*(\d+)\s*-\s*(\d+)\s*(?:\((\d+)\))?$")
EVENT = re.compile(r"^(\d{1,3}(?:\+\d+)?)'\s*(.+)$")
CAPTAIN = re.compile(r"\s*\((?:C|Captain)\)\s*$")
SCORES = re.compile(r"^(.+?) scores( \((?:pen|penalty)\))?$", re.I)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _clean(text: str) -> str:
    return " ".join(html.unescape(text).replace("‍", "").split())


def _visible_lines(page: str) -> list[str]:
    page = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", page, flags=re.S | re.I)
    return [ln for ln in (_clean(x) for x in re.sub(r"<[^>]+>", "\n", page).splitlines()) if ln]


def _warn(warnings: list, msg: str) -> None:
    warnings.append(msg)
    print(f"  [WARN] {msg}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# Fetching — polite, cache-backed, never raises
# ══════════════════════════════════════════════════════════════════════════════

class Fetcher:
    """get(url) returns page text or None. Every failure is a warning, never an exception."""

    def __init__(self, offline: bool = False, cache_dir: Path = CACHE_DIR, delay: float = DELAY_S):
        self.offline, self.cache_dir, self.delay = offline, cache_dir, delay
        self.warnings: list[str] = []
        self.sources: list[dict] = []
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._last = 0.0

    def cache_path(self, url: str) -> Path:
        host = urlparse(url).netloc or "local"
        return self.cache_dir / host / (hashlib.sha1(url.encode("utf-8")).hexdigest()[:16] + ".html")

    def _pause(self) -> None:
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _open(self, url: str) -> str:
        self._pause()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")

    def _allowed(self, url: str) -> bool:
        parts = urlparse(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._robots:
            rp: urllib.robotparser.RobotFileParser | None = urllib.robotparser.RobotFileParser(base + "/robots.txt")
            try:
                rp.parse(self._open(base + "/robots.txt").splitlines())
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500:
                    rp.parse([])  # no robots.txt published: crawling allowed
                else:
                    _warn(self.warnings, f"robots.txt HTTP {exc.code} for {parts.netloc}; skipping host")
                    rp = None
            except NETWORK_ERRORS as exc:
                _warn(self.warnings, f"robots.txt unreachable for {parts.netloc} ({exc}); skipping host")
                rp = None
            self._robots[base] = rp
        rp = self._robots[base]
        return bool(rp and rp.can_fetch(USER_AGENT, url))

    def _from_cache(self, url: str, reason: str) -> str | None:
        cached = self.cache_path(url)
        if cached.exists():
            ts = datetime.fromtimestamp(cached.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
            self.sources.append({"url": url, "status": "cache", "reason": reason, "cached_at": ts})
            return cached.read_text(encoding="utf-8", errors="replace")
        self.sources.append({"url": url, "status": "unavailable", "reason": reason})
        if reason == "offline mode":
            _warn(self.warnings, f"no cached copy of {url} (offline mode)")
        return None

    def get(self, url: str) -> str | None:
        if self.offline:
            return self._from_cache(url, "offline mode")
        if not self._allowed(url):
            _warn(self.warnings, f"not fetched (robots.txt disallows or host skipped): {url}")
            return self._from_cache(url, "robots")
        try:
            body = self._open(url)
        except urllib.error.HTTPError as exc:
            _warn(self.warnings, f"HTTP {exc.code} for {url}")
            return self._from_cache(url, f"HTTP {exc.code}")
        except NETWORK_ERRORS as exc:
            _warn(self.warnings, f"fetch failed for {url} ({exc})")
            return self._from_cache(url, "network error")
        cached = self.cache_path(url)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(body, encoding="utf-8")
        self.sources.append({"url": url, "status": "fetched", "at": _now()})
        return body


# ══════════════════════════════════════════════════════════════════════════════
# Parsers — footballwebpages.co.uk
# ══════════════════════════════════════════════════════════════════════════════

class _TableParser(HTMLParser):
    """Rows (cell text + hrefs) of the first <table>; <h1>/<h2> heading text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.headings: list[str] = []
        self.rows: list[dict] = []
        self._heading = None
        self._tables = 0
        self._row = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("h1", "h2"):
            self._heading = ""
        elif tag == "table":
            self._tables += 1
        elif tag == "tr" and self._tables == 1:
            self._row = {"cells": [], "hrefs": []}
        elif tag in ("td", "th") and self._row is not None:
            self._cell = ""
        elif tag == "a" and self._row is not None and a.get("href"):
            self._row["hrefs"].append(a["href"])

    def handle_endtag(self, tag):
        if tag in ("h1", "h2") and self._heading is not None:
            text = " ".join(self._heading.split())
            if text:
                self.headings.append(text)
            self._heading = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row["cells"].append(" ".join(self._cell.split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row["cells"]):
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._heading is not None:
            self._heading += data
        if self._cell is not None:
            self._cell += data


def parse_league_table(page: str, club: str) -> dict | None:
    """Heading 'Club – League Table – <division> – <season>' plus the full standings."""
    p = _TableParser()
    p.feed(page)
    heading = next((h for h in p.headings if "League Table" in h), "")
    parts = [x.strip() for x in re.split(r"\s+[–—]\s+", heading)]
    standings = []
    updated = None
    for r in p.rows:
        c = r["cells"]
        if c and c[0].startswith("Last updated"):
            updated = c[0].split(":", 1)[1].strip()
        elif len(c) >= 19 and c[1].isdigit():
            standings.append({
                "position": int(c[1]), "team": c[2], "played": int(c[11]), "won": int(c[12]),
                "drawn": int(c[13]), "lost": int(c[14]), "goals_for": int(c[15]),
                "goals_against": int(c[16]), "goal_difference": int(c[17]), "points": int(c[18]),
            })
    row = next((s for s in standings if s["team"].lower() == club.lower()), None)
    if row is None:
        return None
    return {
        "division": parts[2] if len(parts) >= 4 else None,
        "season": parts[3] if len(parts) >= 4 else None,
        "last_updated": updated,
        "teams": len(standings),
        "row": row,
        "standings": standings,
    }


def parse_fixtures(page: str) -> list[dict]:
    """Fixtures/results table: date, venue, opponent, competition, score (team first), match URL."""
    p = _TableParser()
    p.feed(page)
    out = []
    for r in p.rows:
        c = r["cells"]
        if len(c) < 5 or c[0] == "Date":
            continue
        fx = {
            "date": c[0],
            "venue": {"H": "home", "A": "away"}.get(c[1], c[1]),
            "opponent": c[2],
            "competition": c[3],
            "raw_score": c[4],
            "scorers": c[6] if len(c) > 6 else "",
            "url": next((urljoin(FWP, h) for h in r["hrefs"] if "match/" in h), None),
            "played": False,
        }
        m = SCORE.match(c[4])
        if m:
            gf, ga = int(m.group(2)), int(m.group(3))
            fx.update(played=True, goals_for=gf, goals_against=ga,
                      half_time=f"{m.group(1)}-{m.group(4)}" if m.group(1) is not None else None,
                      result="W" if gf > ga else "D" if gf == ga else "L")
        out.append(fx)
    return out


class _MatchParser(HTMLParser):
    """ul.match-events items and each side's ul.match-line-up entries, in document order."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lineups: list[list[dict]] = []
        self.events: list[str] = []
        self._pending_side = False
        self._in_lineup = False
        self._li = None
        self._shirt = False
        self._in_events = False
        self._event = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag == "div" and ("home-line-up" in cls or "away-line-up" in cls):
            self._pending_side = True
        elif tag == "ul" and "match-line-up" in cls and self._pending_side:
            self._in_lineup, self._pending_side = True, False
            self.lineups.append([])
        elif tag == "ul" and "match-events" in cls:
            self._in_events = True
        elif tag == "li" and self._in_lineup:
            self._li = {"name": None, "team_slug": None, "shirt": None, "on_pitch_at_end": "playing" in cls}
            self.lineups[-1].append(self._li)
        elif tag == "li" and self._in_events:
            self._event = ""
        elif tag == "a" and self._li is not None:
            m = re.match(r"([a-z0-9-]+)/appearances/", (a.get("href") or "").lstrip("/"))
            if m:
                self._li["team_slug"] = m.group(1)
            if a.get("title"):
                self._li["name"] = a["title"]
        elif tag == "span" and self._li is not None and "fa-layers-text" in cls:
            self._shirt = True

    def handle_endtag(self, tag):
        if tag == "ul" and self._in_lineup:
            self._in_lineup, self._li = False, None
        elif tag == "ul" and self._in_events:
            self._in_events = False
        elif tag == "li" and self._event is not None:
            text = " ".join(self._event.split())
            if text:
                self.events.append(text)
            self._event = None
        elif tag == "li":
            self._li = None
        elif tag == "span":
            self._shirt = False

    def handle_data(self, data):
        if self._shirt and self._li is not None and data.strip().isdigit():
            self._li["shirt"] = int(data.strip())
        if self._event is not None:
            self._event += " " + data


def parse_match(page: str, team_slug: str) -> dict:
    """Score lines, typed events and the team sheet. First 11 entries = starting XI."""
    p = _MatchParser()
    p.feed(page)
    lines = _visible_lines(page)
    events = []
    for raw in p.events:
        m = EVENT.match(raw)
        if not m:
            continue
        text = m.group(2)
        kind = ("own_goal" if "(og)" in text else "goal" if SCORES.match(text)
                else "sub" if " replaced " in text else "red" if "sent off" in text
                else "yellow" if "cautioned" in text else "other")
        events.append({"minute": m.group(1), "kind": kind, "text": text})
    sheet = next((side for side in p.lineups if any(x["team_slug"] == team_slug for x in side)), [])
    players = [{
        "shirt": x["shirt"],
        "name": CAPTAIN.sub("", x["name"] or "").strip(),
        "captain": bool(CAPTAIN.search(x["name"] or "")),
        "started": i < 11,
    } for i, x in enumerate(sheet)]
    names = {pl["name"] for pl in players}
    return {
        "full_time": next((ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("Full-time:")), None),
        "half_time": next((ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("Half-time:")), None),
        "lineup_published": len(players) >= 11,
        "starters": [pl for pl in players if pl["started"]],
        "bench": [pl for pl in players if not pl["started"]],
        "events": events,
        "team_goals": [{"minute": e["minute"], "scorer": g.group(1), "penalty": bool(g.group(2))}
                       for e in events if e["kind"] == "goal" and (g := SCORES.match(e["text"])) and g.group(1) in names],
    }


def involvement(match: dict, name: str) -> dict:
    """started | sub_on | unused_sub | not_in_squad | no_team_sheet, plus minutes, goals, cards."""
    events = match.get("events", [])
    sheet = match.get("starters", []) + match.get("bench", [])
    mine = next((pl for pl in sheet if pl["name"] == name), None)
    on = next((e["minute"] for e in events if e["kind"] == "sub" and e["text"].startswith(f"{name} replaced ")), None)
    off = next((e["minute"] for e in events if e["kind"] == "sub" and e["text"].endswith(f" replaced {name}")), None)
    if not match.get("lineup_published"):
        status = "no_team_sheet"
    elif mine and mine["started"]:
        status = "started"
    elif on:
        status = "sub_on"
    elif mine:
        status = "unused_sub"
    else:
        status = "not_in_squad"
    return {
        "status": status,
        "shirt": mine["shirt"] if mine else None,
        "captain": bool(mine and mine["captain"]),
        "on": on,
        "off": off,
        "goals": [e["minute"] for e in events
                  if e["kind"] == "goal" and (g := SCORES.match(e["text"])) and g.group(1) == name],
        "cards": [f"{e['kind']} {e['minute']}'" for e in events
                  if e["kind"] in ("yellow", "red") and e["text"].startswith(f"{name} ")],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Parsers — club website (articles, player profiles)
# ══════════════════════════════════════════════════════════════════════════════

class _ArticleParser(HTMLParser):
    """Title, date and body paragraphs. Webflow club sites: h1.post-heading, .post-date-text, .main-post-content.
    WordPress club sites (e.g. Sholing): h1.post-title, .timestamp, .post-summary lead then .post-body."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title, self.date, self.paragraphs = "", "", []
        self._mode = None
        self._depth = 0
        self._buf = ""

    def _flush(self):
        text = _clean(self._buf)
        if text:
            self.paragraphs.append(text)
        self._buf = ""

    def handle_starttag(self, tag, attrs):
        cls = (dict(attrs).get("class") or "").split()
        if self._mode == "body":
            if tag == "br":
                self._buf += " "
            if tag in VOID_TAGS:
                return
            self._depth += 1
            if tag in ("p", "li", "h2", "h3", "h4"):
                self._flush()
        elif {"main-post-content", "post-summary", "post-body"} & set(cls):
            self._mode, self._depth = "body", 1
        elif {"post-heading", "post-title"} & set(cls) and not self.title:
            self._mode = "title"
        elif {"post-date-text", "timestamp"} & set(cls) and not self.date:
            self._mode = "date"

    def handle_endtag(self, tag):
        if tag in VOID_TAGS:
            return
        if self._mode == "body":
            if tag in ("p", "li", "h2", "h3", "h4"):
                self._flush()
            self._depth -= 1
            if self._depth == 0:
                self._flush()
                self._mode = None
        elif self._mode in ("title", "date"):
            self._mode = None

    def handle_data(self, data):
        if self._mode == "title":
            self.title += data
        elif self._mode == "date":
            self.date += data
        elif self._mode == "body":
            self._buf += data


def parse_article(page: str) -> dict:
    p = _ArticleParser()
    p.feed(page)
    return {"title": _clean(p.title), "date": _clean(p.date), "paragraphs": p.paragraphs}


def parse_profile(page: str) -> dict | None:
    lines = _visible_lines(page)
    if "Position:" not in lines:
        return None
    i = lines.index("Position:")

    def after(label):
        k = lines.index(label) if label in lines else None
        return lines[k + 1] if k is not None and k + 1 < len(lines) else None

    start = lines.index("Previous Clubs:") + 2 if "Previous Clubs:" in lines else i + 2
    return {
        "name": lines[i - 1] if i else None,
        "position": after("Position:"),
        "joined": after("Joined the Club:"),
        "previous_clubs": after("Previous Clubs:"),
        "bio": max(lines[start:start + 6], key=len, default=""),
    }


def tactical_sentences(paragraphs: list[str]) -> list[str]:
    sentences = [s.strip() for p in paragraphs for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
    return [s for s in sentences if TACTICAL.search(s)]


def _parse_date(text: str) -> datetime | None:
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text, fmt)
        except (TypeError, ValueError):
            continue
    return None


def _season_start(season: str | None) -> datetime | None:
    m = re.match(r"(\d{4})", season or "")
    return datetime(int(m.group(1)), 7, 1) if m else None


# ══════════════════════════════════════════════════════════════════════════════
# Evidence assembly
# ══════════════════════════════════════════════════════════════════════════════

def _safe(fetcher: Fetcher, label: str, fn, *args):
    try:
        return fn(*args)
    except Exception as exc:  # contract: a malformed page must never crash the run
        _warn(fetcher.warnings, f"{label}: parse failed ({type(exc).__name__}: {exc})")
        return None


def build_evidence(opponent: str, fwp_slug: str, club_site: str | None,
                   players: list[str], last: int, fetcher: Fetcher) -> dict:
    ev = {
        "opponent": opponent,
        "generated_at": _now(),
        "tool": "tools/scout_fetcher.py",
        "league": None,
        "fixtures": [],
        "recent_competitive": [],
        "players": {name: {"profile": None, "involvement": []} for name in players},
        "statements": [],
        "not_published": NOT_PUBLISHED,
        "sources": fetcher.sources,
        "warnings": fetcher.warnings,
    }

    page = fetcher.get(urljoin(FWP, f"{fwp_slug}/league-table"))
    if page:
        ev["league"] = _safe(fetcher, "league table", parse_league_table, page, opponent)
        if ev["league"] is None:
            _warn(fetcher.warnings, f"league table: no row found for {opponent!r}")

    page = fetcher.get(urljoin(FWP, f"{fwp_slug}/fixtures-results"))
    if page:
        ev["fixtures"] = _safe(fetcher, "fixtures", parse_fixtures, page) or []
    played = [f for f in ev["fixtures"] if f["played"]]

    for fx in played:
        match_page = fetcher.get(fx["url"]) if fx.get("url") else None
        fx["match"] = (_safe(fetcher, f"match {fx['date']} v {fx['opponent']}", parse_match, match_page, fwp_slug)
                       if match_page else None)

    league = ev["league"]
    if league and played:
        counts = Counter(f["competition"] for f in ev["fixtures"])
        label = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        results = [f["result"] for f in played if f["competition"] == label]
        w, d, l = results.count("W"), results.count("D"), results.count("L")
        row = league["row"]
        league["check"] = {
            "league_label_in_fixtures": label,
            "results_counted": len(results),
            "record_from_results": f"W{w} D{d} L{l}",
            "points_from_results": 3 * w + d,
            "matches_table": (len(results), 3 * w + d) == (row["played"], row["points"]),
        }
        if not league["check"]["matches_table"]:
            _warn(fetcher.warnings, f"league record from results ({league['check']['record_from_results']}, "
                                    f"{3 * w + d} pts) does not match table (P{row['played']}, {row['points']} pts)")

    ev["recent_competitive"] = played[-last:] if last > 0 else []

    for name in players:
        for fx in played:
            base = {"date": fx["date"], "opponent": fx["opponent"], "competition": fx["competition"]}
            if fx.get("match"):
                ev["players"][name]["involvement"].append({**base, **involvement(fx["match"], name)})
            else:
                ev["players"][name]["involvement"].append({**base, "status": "no_match_page"})

    if club_site:
        site = club_site.rstrip("/")
        for name in players:
            url = f"{site}/players/{_slug(name)}"
            page = fetcher.get(url)
            profile = _safe(fetcher, f"profile {name}", parse_profile, page) if page else None
            if profile:
                profile.update(url=url, tactical_sentences=tactical_sentences([profile["bio"]]))
            ev["players"][name]["profile"] = profile

        news = site if site.endswith("/news") else f"{site}/news"
        index = fetcher.get(news) or ""
        for n in range(2, NEWS_PAGES + 1):                 # older archive pages, only when the site links them
            if not re.search(rf'href="[^"]*/news/page/{n}/?"', index):
                break
            index += fetcher.get(f"{news}/page/{n}/") or ""
        links = list(dict.fromkeys(urljoin(site + "/", h) for h in REPORT_LINK.findall(index)))
        season_start = _season_start(league["season"] if league else None)
        for url in links:
            page = fetcher.get(url)
            art = _safe(fetcher, f"article {url}", parse_article, page) if page else None
            if not art or not art["paragraphs"]:
                continue
            when = _parse_date(art["date"])
            if season_start and when and when < season_start:
                continue
            ev["statements"].append({
                "url": url, "title": art["title"], "date": art["date"],
                "intro": art["paragraphs"][0],
                "tactical_sentences": tactical_sentences(art["paragraphs"][1:]),
            })
    return ev


def print_summary(ev: dict, out: Path | None) -> None:
    print(f"\nPitchPulse · Scout Fetcher · {ev['opponent']}")
    league = ev.get("league")
    if league:
        r = league["row"]
        print(f"League     {league['division']} {league['season']} (updated {league['last_updated']})")
        print(f"Record     position {r['position']}/{league['teams']} · P{r['played']} W{r['won']} D{r['drawn']} "
              f"L{r['lost']} · F{r['goals_for']} A{r['goals_against']} · {r['points']} pts")
        c = league.get("check")
        if c:
            verdict = "matches table" if c["matches_table"] else "MISMATCH"
            print(f"Check      {c['results_counted']} '{c['league_label_in_fixtures']}' results → "
                  f"{c['record_from_results']}, {c['points_from_results']} pts → {verdict}")
    else:
        print("League     unavailable")
    for fx in ev["recent_competitive"]:
        m = fx.get("match") or {}
        xi = ", ".join(f"{p['shirt']} {p['name']}{' (C)' if p['captain'] else ''}" for p in m.get("starters", []))
        print(f"{fx['date']:<10} {fx['competition']:<14} {fx['venue']:<4} {fx['opponent']:<18} {fx['raw_score']:<12} "
              f"XI: {xi or 'no team sheet'}")
    for name, data in ev["players"].items():
        prof = data.get("profile") or {}
        print(f"{name} [{prof.get('position') or 'no profile'}]")
        for row in data["involvement"]:
            extra = []
            if row.get("on"):
                extra.append(f"on {row['on']}'")
            if row.get("off"):
                extra.append(f"off {row['off']}'")
            if row.get("goals"):
                extra.append("goal " + ", ".join(f"{g}'" for g in row["goals"]))
            extra += row.get("cards", [])
            print(f"    {row['date']:<10} {row['opponent']:<18} {row['status']:<13} {' · '.join(extra)}")
    print(f"Statements {len(ev['statements'])} articles · "
          f"{sum(len(s['tactical_sentences']) for s in ev['statements'])} tactical sentences")
    print(f"Sources    {dict(Counter(s['status'] for s in ev['sources']))} · warnings {len(ev['warnings'])}")
    if out:
        print(f"Evidence   {out}")


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except AttributeError:
            pass

    ap = argparse.ArgumentParser(description="Fetch verifiable public match evidence for an opponent.")
    ap.add_argument("--opponent", required=True, help='Club name as shown in league tables, e.g. "Dorchester Town"')
    ap.add_argument("--fwp-slug", help="footballwebpages.co.uk team slug (default: slug of --opponent)")
    ap.add_argument("--club-site", help="Club website base URL for player profiles and reaction articles")
    ap.add_argument("--players", default="", help="Comma-separated player names to track")
    ap.add_argument("--last", type=int, default=3, help="Recent played fixtures to report in full (default 3)")
    ap.add_argument("--offline", action="store_true", help="Use cached pages only; no network")
    ap.add_argument("--out", type=Path, help="Evidence JSON path (default data/scouting/evidence/<slug>.json)")
    args = ap.parse_args(argv)

    opponent = " ".join(args.opponent.split())
    if not _slug(opponent) or args.last < 0:
        print("  [ERROR] --opponent must contain letters or digits and --last must be ≥ 0", file=sys.stderr)
        return 2

    fetcher = Fetcher(offline=args.offline)
    players = [p.strip() for p in args.players.split(",") if p.strip()]
    try:
        ev = build_evidence(opponent, args.fwp_slug or _slug(opponent), args.club_site, players, args.last, fetcher)
    except Exception as exc:  # last-resort guard: still write what we have
        _warn(fetcher.warnings, f"evidence build aborted ({type(exc).__name__}: {exc})")
        ev = {"opponent": opponent, "generated_at": _now(), "league": None, "fixtures": [],
              "recent_competitive": [], "players": {}, "statements": [], "not_published": NOT_PUBLISHED,
              "sources": fetcher.sources, "warnings": fetcher.warnings}

    out = args.out or OUT_DIR / f"{_slug(opponent).replace('-', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_summary(ev, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
