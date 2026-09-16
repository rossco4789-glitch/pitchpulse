#!/usr/bin/env python3
"""
scout_brief.py — PitchPulse opposition briefing builder
=======================================================
Uploaded match preview (.docx/.txt/.md) → public match evidence → brief.md → mobile HTML.
Called by app.py Tab 1 (Opposition Briefing). No Streamlit import; fully offline-testable.

Pipeline (run):
    1. extract text from the upload buffer (python-docx for .docx, UTF-8 for .txt/.md)
    2. detect "<Home> v <Away>" (one side Tiverton), date, kick-off and venue from the header lines
    3. save the upload to data/scouting/sources/<slug>/preview.<ext>
    4. scout_fetcher.build_evidence → data/scouting/evidence/<slug>.json
    5. compose brief.md in the packager Markdown subset (4 moments + dead balls)
    6. reports.packager.build_html → data/processed/<slug>_scouting.html

Contract:
    Network or parse failure never raises: the brief degrades to the uploaded notes and the
    reason is returned in result["warnings"]. A hand-edited brief.md (no AUTO_MARKER) is never
    overwritten unless overwrite=True; the draft goes to brief_auto.md and the HTML is compiled
    from the hand-edited brief. Only an unreadable upload or a missing "X v Tiverton" line raises
    ValueError, because nothing trustworthy can be built without an opponent.
"""

import io
import json
import re
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scout_fetcher as sf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLUB = "Tiverton Town"
AUTO_MARKER = "Compiled automatically by the PitchPulse scout brief builder"

BANNED = re.compile(r"(?i)\b(testament|crucial|unlock\w*|delve\w*|pivotal|tapestry|seamless\w*|game[- ]changer)\b")
MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
DATE_DMY = re.compile(rf"(?i)\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{MONTH}(?:,?\s+(\d{{4}}))?\b")
DATE_MDY = re.compile(rf"(?i)\b{MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b")
DATE_NUM = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
VERSUS = re.compile(r"(?i)^\s*(.+?)\s+(?:v|vs\.?|versus)\s+(.+?)\s*$")
KICKOFF = re.compile(r"(?i)\b(\d{1,2})[.:](\d{2})\s*(am|pm)?")
VENUE = re.compile(r"(?i)^(?:at|venue:?)\s+(.+)$")
COMPETITION = re.compile(r"(?i)^competition:?\s+(.+)$")
CUP_NAME = re.compile(r"(?i)\b(FA (?:Cup|Trophy|Vase)(?:\s+\w*\d\w*)?)")
SIDE_META = re.compile(r"\s+[—–|]\s+.*$|,.*$|\s+\(.*$")          # "Sholing — Sat 19 Sep, 15:00, ground" → "Sholing"
SCORER = re.compile(r"^(.+?) scores(?: \((?:pen|penalty)\))?$", re.I)
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

ROUTES = {
    "possession": re.compile(r"(?i)tempo|dictat|distribut|creative|schemer|playmak|build[- ]up|possession"),
    "out_of_possession": re.compile(r"(?i)defensive (?:shield|shape|line|record)|clean sheet|\bshape\b(?! of)|\bpress(?:es|ed|ing)?\b|compact|\bblock\b|back (?:four|line)"),
    "transitions": re.compile(r"(?i)\bcounter|\bpace\b|\bbreak(?:s|ing)?\b|transition"),
    "dead_ball": re.compile(r"(?i)set[- ]?pieces?|set[- ]?plays?|corners?|free[- ]?kicks?|header|aerial|in the air|second[- ]phase|second balls?"),
    "adjustments": re.compile(r"(?i)suspend|injur|absent|without|unavailable|rotat"),
}


# ── Text helpers ──────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _tidy(text: str, limit: int = 280) -> str | None:
    """Packager-safe sentence: no underscores (italics) or bold markers; None if it breaks prose rules."""
    s = " ".join(text.replace("_", " ").replace("**", "").split())
    if not s or BANNED.search(s):
        return None
    if len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0] + "…"
    return s


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s]


def _route(sentences: list[str], key: str, cap: int = 3) -> list[str]:
    out = []
    for s in sentences:
        t = _tidy(s)
        if t and ROUTES[key].search(t) and t not in out:
            out.append(t)
        if len(out) == cap:
            break
    return out


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".docx":
        import docx
        try:
            doc = docx.Document(io.BytesIO(data))
        except Exception as exc:
            raise ValueError(f"could not read {filename} as a Word document ({type(exc).__name__})") from exc
        return "\n".join(p.text for p in doc.paragraphs)
    if suffix in (".txt", ".md"):
        return data.decode("utf-8-sig", errors="replace")
    raise ValueError(f"unsupported file type {suffix!r}: upload .docx, .txt or .md")


# ── Fixture detection ─────────────────────────────────────────────────────────

def _resolve_year(month: int, day: int, year: str | None, today: date) -> date | None:
    try:
        if year:
            y = int(year)
            return date(y + 2000 if y < 100 else y, month, day)
        d = date(today.year, month, day)
        return date(today.year + 1, month, day) if d < today - timedelta(days=30) else d
    except ValueError:
        return None


def detect_fixture(text: str, today: date | None = None) -> dict:
    today = today or date.today()
    head = [ln.strip() for ln in text.splitlines() if ln.strip()][:12]

    teams = next((m for m in map(VERSUS.match, head)
                  if m and any("tiverton" in side.lower() for side in m.groups())), None)
    if not teams:
        raise ValueError(f"no '<Opponent> v {CLUB}' line in the first 12 lines of the preview")
    home, away = (SIDE_META.sub("", side).strip() for side in teams.groups())
    home_game = "tiverton" in home.lower()

    found = None
    for ln in head:
        if m := DATE_DMY.search(ln):
            found = _resolve_year(MONTHS.index(m.group(2).lower()[:3]) + 1, int(m.group(1)), m.group(3), today)
        elif m := DATE_MDY.search(ln):
            found = _resolve_year(MONTHS.index(m.group(1).lower()[:3]) + 1, int(m.group(2)), m.group(3), today)
        elif m := DATE_NUM.search(ln):
            found = _resolve_year(int(m.group(2)), int(m.group(1)), m.group(3), today)
        if found:
            break

    kickoff = None
    for ln in head:
        if "kick" in ln.lower() or re.search(r"(?i)\d\s*(am|pm)\b|\bko\b", ln):
            if m := KICKOFF.search(ln):
                hh, mm = int(m.group(1)), int(m.group(2))
                if (m.group(3) or "").lower() == "pm" and hh < 12:
                    hh += 12
                kickoff = f"{hh:02d}:{mm:02d}"
                break

    venue = next((m.group(1).strip() for m in map(VENUE.match, head) if m), None)
    competition = next((m.group(1).strip() for m in map(COMPETITION.match, head) if m), None) or next(
        (m.group(1).strip() for m in map(CUP_NAME.search, head) if m), None)
    return {"home": home, "away": away, "opponent": away if home_game else home, "home_game": home_game,
            "date": found.isoformat() if found else None, "kickoff": kickoff, "venue": venue, "competition": competition}


# ── Evidence ──────────────────────────────────────────────────────────────────

def _played(ev: dict | None) -> list[dict]:
    return [f for f in (ev or {}).get("fixtures", []) if f.get("played")]


def _sheet_names(ev: dict | None) -> list[str]:
    names = []
    for fx in _played(ev):
        for pl in (fx.get("match") or {}).get("starters", []) + (fx.get("match") or {}).get("bench", []):
            if pl["name"] and pl["name"] not in names:
                names.append(pl["name"])
    return names


def named_players(text: str, ev: dict | None, cap: int = 8) -> list[str]:
    """Team-sheet names the preview mentions, in order of first mention ('Harvey-Joe Bertrand' ~ 'Harvey Bertrand')."""
    hits = []
    for name in _sheet_names(ev):
        parts = name.split()
        short = f"{parts[0].split('-')[0]} {parts[-1]}" if len(parts) > 1 else name
        pos = min((i for i in (text.find(name), text.find(short)) if i >= 0), default=-1)
        if pos >= 0:
            hits.append((pos, name))
    return [n for _, n in sorted(hits)][:cap]


def gather_evidence(opponent: str, text: str, club_site: str | None, fetcher, warnings: list) -> dict | None:
    try:
        ev = sf.build_evidence(opponent, sf._slug(opponent), club_site, [], 5, fetcher)
    except Exception as exc:  # contract: the app never sees a fetch exception
        warnings.append(f"public match data failed ({type(exc).__name__}: {exc})")
        return None
    if not ev.get("league") and not _played(ev):
        warnings.append("public match data unavailable (network, robots or no matching club); brief uses the uploaded notes only")
        return None

    names = named_players(text, ev)
    played = [fx for fx in _played(ev) if fx.get("match")]
    ev["players"] = {n: {"profile": None, "involvement": [sf.involvement(fx["match"], n) for fx in played]} for n in names}
    if club_site:
        site = club_site.rstrip("/")
        for n in names:
            try:
                page = fetcher.get(f"{site}/players/{sf._slug(n)}")
                profile = sf.parse_profile(page) if page else None
                if profile:
                    profile["tactical_sentences"] = sf.tactical_sentences([profile["bio"]])
                ev["players"][n]["profile"] = profile
            except Exception as exc:
                warnings.append(f"profile {n}: {type(exc).__name__}")
    return ev


# ── Brief composition ─────────────────────────────────────────────────────────

def _minute(m: str) -> int:
    return int(m.split("+")[0])


def _uk(iso: str | None) -> str:
    if not iso:
        return "not found in the preview"
    d = date.fromisoformat(iso)
    return f"{d:%A}, {d.day} {d:%B %Y}"


def _result_line(fx: dict) -> str:
    verb = {"W": "Won", "D": "Drew", "L": "Lost"}[fx["result"]]
    where = "at" if fx["venue"] == "away" else "against"
    return f"{verb} {fx['goals_for']}-{fx['goals_against']} {where} {fx['opponent']} ({fx['competition']})"


def _goals(match: dict) -> tuple[list[dict], list[dict]]:
    """(team goals, goals against) from event text, so penalties count: 'Jake Mccarthy scores (pen)' is a goal."""
    squad = {p["name"] for p in match.get("starters", []) + match.get("bench", [])}
    ours, theirs = [], []
    for e in match.get("events", []):
        m = SCORER.match(e.get("text", ""))
        if m:
            (ours if m.group(1) in squad else theirs).append({"minute": e["minute"], "scorer": m.group(1)})
    return ours, theirs


def _late_goals(played: list[dict]) -> tuple[int, int, int]:
    scored = conceded = sheets = 0
    for fx in played:
        m = fx.get("match")
        if not m or not m.get("events"):
            continue
        sheets += 1
        ours, theirs = _goals(m)
        scored += any(_minute(g["minute"]) >= 85 for g in ours)
        conceded += any(_minute(g["minute"]) >= 85 for g in theirs)
    return scored, conceded, sheets


def _names_and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


def _captaincy(sheets: list[dict]) -> str | None:
    """Recent armband first: a change of captain is news even when the old captain has more games."""
    worn = [next((p["name"] for p in f["match"]["starters"] if p["captain"]), None) for f in sheets]
    worn = [w for w in worn if w]
    if not worn:
        return None
    recent = worn[-1]
    streak = next((i for i, w in enumerate(reversed(worn)) if w != recent), len(worn))
    top, c = Counter(worn).most_common(1)[0]
    if top == recent or Counter(worn)[recent] == c:
        return (f"**Captain:** {recent} wore the armband in {Counter(worn)[recent]} of {len(worn)} team sheets"
                + (f", including the last {streak}." if streak < len(worn) else "."))
    return (f"**Captain:** {recent} has worn the armband in the last {streak} team sheet{'s' if streak > 1 else ''}. "
            f"{top} wore it in {c} of {len(worn)}.")


def _quotes(ev: dict | None, key: str, cap: int = 2) -> list[str]:
    out = []
    for st in (ev or {}).get("statements", []):
        for s in _route(st.get("tactical_sentences", []), key, cap):
            out.append(f"**{_tidy(st['title'], 80)} ({st['date']}):** {s}")
    return out[:cap]


def compose_brief(fx: dict, notes: str, ev: dict | None, generated: date | None = None) -> str:
    generated = generated or date.today()
    sents = _sentences("\n".join(ln for ln in notes.splitlines() if len(ln.split()) > 8))  # skip header lines
    league = (ev or {}).get("league")
    played = _played(ev)
    sheets = [f for f in played if (f.get("match") or {}).get("lineup_published")]
    opp = fx["opponent"]
    L: list[str] = []

    def section(title: str) -> None:
        L.extend(["", "---", "", f"## {title}", ""])

    def para(text: str) -> None:
        L.extend([text, ""])

    def preview(key: str) -> None:
        for s in _route(sents, key):
            para(f"**Club preview (unverified):** {s}")

    # Header
    para(f"# {fx['home']} v {fx['away']} — Pre-Match Briefing")
    para(f"**Match Date:** {_uk(fx['date'])}")
    if fx["kickoff"] or fx["venue"]:
        where = f" at {fx['venue']}" if fx["venue"] else ""
        para(f"**Kick-off:** {fx['kickoff'] or 'time not stated'}{where} ({'home' if fx['home_game'] else 'away'})")
    competition = fx.get("competition") or next(
        (f["competition"] for f in (ev or {}).get("fixtures", []) if not f.get("played") and "tiverton" in f["opponent"].lower()), None)
    if competition:
        para(f"**Competition:** {competition}")
        if league:
            para(f"**Opponent league:** {league['division']}, {league['season']}")
    else:
        para(f"**Competition:** {league['division']}, {league['season']}" if league
             else "**Competition:** not verified (public league data unavailable)")
    para(f"**Evidence:** uploaded club preview; " + (
        f"Football Web Pages table and {len(sheets)} team sheets" if ev else "no public match data"))
    para("⚠ Draft built automatically. A coach checks every line before it reaches the players.")
    if ev:
        para(f"⚠ No fetched source publishes {', '.join(sf.NOT_PUBLISHED)}. Nothing below infers them.")
    else:
        para("⚠ Public match data could not be fetched. Every claim below comes from the uploaded preview and is unverified.")

    # Executive summary
    section("Executive Summary & Form")
    if league:
        r = league["row"]
        para(f"**League position:** {r['position']} of {league['teams']}. Played {r['played']}, won {r['won']}, "
             f"drew {r['drawn']}, lost {r['lost']}, {r['goals_for']} scored, {r['goals_against']} conceded, {r['points']} points.")
        tiv = next((s for s in league["standings"] if "tiverton" in s["team"].lower()), None)
        if tiv:
            para(f"**Tiverton:** {tiv['position']} with {tiv['points']} points from {tiv['played']} games.")
    last5 = [f for f in played if f.get("result")][-5:]
    if last5:
        para(f"**Last {len(last5)} games ({' '.join(f['result'] for f in last5)}):** "
             + "; ".join(_result_line(f) for f in last5) + ".")
    scored, conceded, n = _late_goals(played)
    if n:
        para(f"**Late goals:** {opp} scored after the 85th minute in {scored} of {n} games and conceded after it in {conceded}.")
    if not ev:
        para("**Form:** not verified. Check the league table before the team meeting.")
    if n and max(scored, conceded) >= 2:
        para(f"> **Lever:** Keep two substitutions for the last 15 minutes. {opp} scored late in {scored} of {n} games and conceded late in {conceded}.")
    else:
        para("> **Lever:** Set our Line of Engagement from the evidence in this brief, not from the preview's reputation claims.")

    # In possession
    section("In Possession Phase")
    if sheets:
        xi = sheets[-1]["match"]["starters"]
        para(f"**Most recent XI ({sheets[-1]['date']} v {sheets[-1]['opponent']}):** "
             + ", ".join(f"{p['shirt']} {p['name']}{' (captain)' if p['captain'] else ''}" for p in xi) + ".")
    scorers = Counter(g["scorer"] for f in played if f.get("match") for g in _goals(f["match"])[0])
    finishers = [name for name, c in scorers.items() if c == max(scorers.values())] if scorers else []
    if scorers:
        para("**Goal threat:** " + ", ".join(f"{name} {c}" for name, c in scorers.most_common(3))
             + " (goals on team sheets, penalties included).")
    preview("possession")
    para("> **Lever:** Formation is not published. Read their midfield shape in the first 10 minutes and set the Pressing Trigger on their deepest midfielder's first touch.")

    # Out of possession
    section("Out of Possession Phase")
    scored_games = [f for f in played if f.get("goals_against") is not None]      # a result without a scoreline has no sample
    if scored_games:
        league_label = ((league or {}).get("check") or {}).get("league_label_in_fixtures")
        groups = ([("league game", [f for f in scored_games if f["competition"] == league_label]),
                   ("cup game", [f for f in scored_games if f["competition"] != league_label])] if league_label
                  else [("game", scored_games)])
        parts = []
        for noun, games in groups:
            if not games:
                continue
            against = [f["goals_against"] for f in games]
            sheets_kept = against.count(0)
            parts.append(f"{sum(against)} conceded in {len(games)} {noun}{'' if len(games) == 1 else 's'} "
                         f"({sheets_kept} clean sheet{'' if sheets_kept == 1 else 's'}, most in one game {max(against)})")
        para("**Defensive record:** " + "; ".join(parts) + ".")
    captain = _captaincy(sheets)
    if captain:
        para(captain)
    preview("out_of_possession")
    para("> **Lever:** Attack the Half-Spaces behind their full-backs until their Block height is confirmed from our own footage.")

    # Transitions
    section("Attacking & Defensive Transitions")
    for q in _quotes(ev, "transitions"):
        para(q)
    if finishers:
        top = max(scorers.values())
        para(f"**Primary finisher{'s' if len(finishers) > 1 else ''}:** {_names_and(finishers)}, "
             f"{top} goal{'' if top == 1 else 's'}{' each' if len(finishers) > 1 else ''}.")
    preview("transitions")
    para(f"> **Lever:** Hold Rest Defense of two centre-backs plus the holding midfielder whenever both full-backs advance"
         f"{f', with one of them tracking {_names_and(finishers)}' if finishers else ''}.")

    # Dead balls
    section("Dead-Ball Organization")
    for q in _quotes(ev, "dead_ball"):
        para(q)
    for name, data in (ev or {}).get("players", {}).items():
        prof = data.get("profile") or {}
        for s in _route(prof.get("tactical_sentences", []), "dead_ball", 1):
            para(f"**{name} (club profile):** {s}")
    preview("dead_ball")
    para("⚠ Corner and free-kick takers and delivery zones are not published.")
    para("> **Lever:** Tag every opposition corner and free kick in the first half so the half-time briefing names the taker and target zone.")

    # Profiles
    section("Key Opposition Profiles")
    for name, data in (ev or {}).get("players", {}).items():
        inv = data["involvement"]
        starts = sum(i["status"] == "started" for i in inv)
        subs = sum(i["status"] == "sub_on" for i in inv)
        goals = sum(len(i["goals"]) for i in inv)
        cards = [c for i in inv for c in i["cards"]]
        shirt = next((i["shirt"] for i in reversed(inv) if i["shirt"]), None)
        prof = data.get("profile") or {}
        line = (f"**{name}{f' (number {shirt})' if shirt else ''}:** {starts} starts, {subs} off the bench, "
                f"{goals} goal{'' if goals == 1 else 's'} in {len(inv)} games{f'; cards {len(cards)}' if cards else ''}.")
        if prof.get("position"):
            line += f" Club profile position: {prof['position']}."
        para(line)
    if not (ev or {}).get("players"):
        para("**Profiles:** no preview player could be matched to a published team sheet.")

    # Adjustments
    section("Tivvy Tactical Adjustments")
    if _route(sents, "adjustments"):
        preview("adjustments")
        para("> **Lever:** Name the cover for every absentee listed above before the team meeting.")
    else:
        para("**Absentees:** none named in the preview. Confirm their XI in the warm-up.")

    # Data quality
    section("Data Quality & Sources")
    if league:
        para(f"**League data:** Football Web Pages table updated {league['last_updated']}.")
        chk = league.get("check")
        if chk:
            para(f"**Table check:** results rebuild to {chk['record_from_results']}, {chk['points_from_results']} points; "
                 f"{'matches the table' if chk['matches_table'] else 'does NOT match the table'}.")
    if ev:
        para(f"**Match data:** {len(sheets)} team sheets from {len(played)} played games; "
             f"{len(ev.get('statements', []))} club reaction articles; "
             f"{sum(1 for d in ev.get('players', {}).values() if d.get('profile'))} player profiles.")
        warns = [w for w in (_tidy(x, 160) for x in ev.get("warnings", [])) if w]
        if warns:
            para(f"**Fetch warnings ({len(warns)}):** " + "; ".join(warns[:3]) + ".")
    para("**Not verified:** every line marked club preview. The preview is written by staff for supporters, not from match data.")
    para(f"**Build:** {AUTO_MARKER} on {generated.day} {generated:%B %Y}.")
    return "\n".join(L).rstrip() + "\n"


# ── Orchestration ─────────────────────────────────────────────────────────────

def run(filename: str, data: bytes, *, club_site: str | None = None, opponent: str | None = None,
        offline: bool = False, overwrite: bool = False, root: Path = ROOT, today: date | None = None,
        progress=print, fetcher=None) -> dict:
    today = today or date.today()
    warnings: list[str] = []

    progress("Reading the uploaded preview")
    text = extract_text(filename, data)
    fx = detect_fixture(text, today)
    if opponent:
        fx["opponent"] = opponent.strip()
    slug = slugify(fx["opponent"])

    src_dir = root / "data" / "scouting" / "sources" / slug
    src_dir.mkdir(parents=True, exist_ok=True)
    preview_path = src_dir / f"preview{Path(filename).suffix.lower()}"
    preview_path.write_bytes(data)
    progress(f"Saved preview for {fx['opponent']} ({_uk(fx['date'])})")

    progress("Fetching league table, results and team sheets")
    fetcher = fetcher or sf.Fetcher(offline=offline)
    ev = gather_evidence(fx["opponent"], text, club_site, fetcher, warnings)
    if ev:
        out = root / "data" / "scouting" / "evidence" / f"{slug}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")

    progress("Composing the 4-moments brief")
    md = compose_brief(fx, text, ev, today)
    brief_path = src_dir / "brief.md"
    kept_manual = brief_path.exists() and AUTO_MARKER not in brief_path.read_text(encoding="utf-8") and not overwrite
    if kept_manual:
        draft = src_dir / "brief_auto.md"
        draft.write_text(md, encoding="utf-8")
        warnings.append(f"kept the hand-edited brief.md; new draft saved as {draft.name} (tick overwrite to replace)")
        md = brief_path.read_text(encoding="utf-8")
    else:
        brief_path.write_text(md, encoding="utf-8")

    progress("Compiling the mobile HTML briefing")
    sys.path.insert(0, str(ROOT))
    from reports.packager import build_html
    html_path = build_html(md, root / "data" / "scouting" / "no_plots", root / "data" / "processed" / f"{slug}_scouting.html")

    league = (ev or {}).get("league")
    return {
        "opponent": fx["opponent"], "slug": slug, "fixture": fx,
        "division": f"{league['division']} {league['season']}" if league else None,
        "position": league["row"]["position"] if league else None,
        "teams": league["teams"] if league else None,
        "points": league["row"]["points"] if league else None,
        "degraded": ev is None, "kept_manual": kept_manual,
        "preview_path": preview_path, "brief_path": brief_path, "html_path": html_path,
        "warnings": warnings + ((ev or {}).get("warnings") or []),
    }
