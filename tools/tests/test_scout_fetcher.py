"""
tools/tests/test_scout_fetcher.py
Parser and failure-contract tests for the public match-evidence fetcher.
Offline only — synthetic HTML that mirrors the source page structures; urlopen is stubbed.

Run: python -m pytest tools/tests/test_scout_fetcher.py -v
"""

import email.message
import socket
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import scout_fetcher as sf  # noqa: E402


# ── Synthetic pages ───────────────────────────────────────────────────────────

def _table_row(pos, team, p, w, d, l, f, a, pts):
    cells = ["", pos, team, "0", "0", "0", "0", "0", "0", "0", "0", p, w, d, l, f, a, str(int(f) - int(a)), pts]
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


TABLE_HTML = (
    "<h1>Football Web Pages</h1>"
    "<h1>Dorchester Town – League Table – Southern League - South Division – 2026-2027</h1>"
    "<table><tr><td></td><td></td><td>Home</td><td>Away</td><td>Total</td></tr>"
    "<tr><th></th><th></th><th>P</th><th>W</th></tr>"
    + _table_row("4", "Dorchester Town", "4", "3", "0", "1", "7", "7", "9")
    + _table_row("12", "Tiverton Town", "4", "1", "1", "2", "3", "4", "4")
    + "<tr><td>Last updated: Saturday 12th September 2026 at 17:10:35</td></tr></table>"
)

FIXTURES_HTML = (
    "<table><tr><th>Date</th><th>H/A</th><th>Opponent</th><th>Competition</th><th>KO/Score</th><th>Attd</th><th>Scorers</th></tr>"
    '<tr><td>Sat 8 Aug</td><td>A</td><td>Weymouth</td><td>Southern South</td><td><a href="match/2026-2027/l/weymouth/dorchester-town/1">(1)1 - 5(2)</a></td><td>2,272</td><td>Burrows</td></tr>'
    '<tr><td>Tue 11 Aug</td><td>H</td><td>Paulton Rovers</td><td>Southern South</td><td><a href="match/2026-2027/l/dorchester-town/paulton-rovers/2">(1)2 - 0(0)</a></td><td>476</td><td>Spetch, Taylor</td></tr>'
    '<tr><td>Sat 22 Aug</td><td>H</td><td>Winchester City</td><td>FA Cup P</td><td><a href="match/2026-2027/c/dorchester-town/winchester-city/3">(1)3 - 1(0)</a></td><td>431</td><td></td></tr>'
    '<tr><td>Mon 31 Aug</td><td>H</td><td>Hungerford Town</td><td>Southern South</td><td><a href="match/2026-2027/l/dorchester-town/hungerford-town/4">1 - 0</a></td><td>442</td><td>Taylor</td></tr>'
    '<tr><td>Sat 5 Sep</td><td>A</td><td>Willand Rovers</td><td>Southern South</td><td><a href="match/2026-2027/l/willand-rovers/dorchester-town/5">(0)3 - 2(1)</a></td><td>269</td><td></td></tr>'
    '<tr><td>Tue 15 Sep</td><td>H</td><td>Tiverton Town</td><td>Southern South</td><td><a href="match/2026-2027/l/dorchester-town/tiverton-town/6">7.45pm</a></td><td></td><td></td></tr>'
    '<tr><td>Sat 19 Sep</td><td>A</td><td>Bishops Cleeve</td><td>Southern South</td><td>P - P</td><td></td><td></td></tr>'
    "</table>"
)


def _li(team, name, shirt, playing=True):
    cls = ' class="playing"' if playing else ""
    return (f'<li{cls}><a href="{team}/appearances/x/1" title="{name}"><span class="shirt-container">'
            f'<span class="fa-layers fa-fw shirt"><i class="fas fa-shirt"></i><span class="fa-layers-text">{shirt}</span>'
            f'</span></span><span class="player">{name}</span></a></li>')


DOR_XI = [("Ryan Hall", 1, True), ("Harvey-Joe Bertrand", 2, True), ("Ollie Haste", 3, True),
          ("Will Spetch (Captain)", 4, True), ("Tom Purrington", 6, True), ("Ethan Taylor", 9, True),
          ("Matt Buse", 10, False), ("Matty Burrows", 11, False), ("Jack Winsor", 14, True),
          ("Jayden Nielsen", 15, False), ("Jay Williams", 16, True)]
DOR_BENCH = [("Ieuan Turner", 5, False), ("Henry Spalding", 20, True)]

MATCH_HTML = (
    "<p>Full-time: Brixham AFC 2-3 Dorchester Town</p><p>Half-time: Brixham AFC 0-0 Dorchester Town</p>"
    '<ul class="match-events with-extra">'
    "<li><span>38'</span> <span>Charlie Johansen sent off</span></li>"
    "<li class=\"goal\"><span>57'</span> <span>Tom Purrington scores</span></li>"
    "<li><span>60'</span> <span>Henry Spalding replaced Matt Buse</span></li>"
    "<li><span>70'</span> <span>Ollie Haste cautioned</span></li>"
    "<li class=\"goal\"><span>71'</span> <span>James Moxon scores</span></li>"
    "</ul>"
    '<div class="col-12 home-line-up"><p class="match-heading">Brixham AFC</p><ul class="match-line-up">'
    + "".join(_li("brixham-afc", f"Home Player {i}", i) for i in range(1, 14))
    + '</ul></div><div class="col-12 away-line-up"><p class="match-heading">Dorchester Town</p><ul class="match-line-up">'
    + "".join(_li("dorchester-town", n, s, p) for n, s, p in DOR_XI + DOR_BENCH)
    + "</ul></div>"
)

ARTICLE_HTML = (
    '<h1 class="post-heading heading-5">Reaction: Brixham 2-3 The Magpies</h1>'
    '<div class="post-date-text">September 5, 2026</div>'
    '<div class="main-post-content w-richtext">'
    "<p>Post-match reaction from the captain:</p>"
    "<p>We got the job done.<br>We concede from lapses at set plays and second phase balls.</p>"
    '<div class="w-embed"><p>Embedded note about the pitch.</p></div>'
    "<p>&#x200d;</p>"
    "</div><p>Footer text outside the article body.</p>"
)

PROFILE_HTML = (
    "<nav>Menu</nav><h1>Will Spetch</h1><p>Position:</p><p>Defence</p><p>Joined the Club:</p><p>September 2023</p>"
    "<p>Previous Clubs:</p><p>Poole Town</p>"
    "<p>A hugely experienced centre back known for leadership and aerial prowess. His threat at set pieces brought eight goals.</p>"
)


# ── Parser tests ──────────────────────────────────────────────────────────────

def test_league_table_division_season_and_row():
    lg = sf.parse_league_table(TABLE_HTML, "Dorchester Town")
    assert lg["division"] == "Southern League - South Division"
    assert lg["season"] == "2026-2027"
    assert lg["last_updated"].startswith("Saturday 12th September 2026")
    assert lg["teams"] == 2
    assert lg["row"] == {"position": 4, "team": "Dorchester Town", "played": 4, "won": 3, "drawn": 0, "lost": 1,
                         "goals_for": 7, "goals_against": 7, "goal_difference": 0, "points": 9}


def test_league_table_missing_club_returns_none():
    assert sf.parse_league_table(TABLE_HTML, "Oxford City") is None


def test_fixtures_scores_are_team_first_and_unplayed_rows_flagged():
    fx = sf.parse_fixtures(FIXTURES_HTML)
    assert [f["played"] for f in fx] == [True, True, True, True, True, False, False]
    assert fx[0]["result"] == "L" and (fx[0]["goals_for"], fx[0]["goals_against"]) == (1, 5)
    assert fx[0]["half_time"] == "1-2"
    assert fx[3]["half_time"] is None and fx[3]["result"] == "W"
    assert fx[0]["url"] == "https://www.footballwebpages.co.uk/match/2026-2027/l/weymouth/dorchester-town/1"
    assert fx[6]["url"] is None and fx[6]["raw_score"] == "P - P"


def test_match_xi_is_first_eleven_not_the_playing_class():
    m = sf.parse_match(MATCH_HTML, "dorchester-town")
    assert m["lineup_published"]
    assert [p["name"] for p in m["starters"]][6:8] == ["Matt Buse", "Matty Burrows"]  # subbed off, no class
    assert [p["name"] for p in m["bench"]] == ["Ieuan Turner", "Henry Spalding"]
    spetch = next(p for p in m["starters"] if p["shirt"] == 4)
    assert spetch["name"] == "Will Spetch" and spetch["captain"]
    assert m["full_time"] == "Brixham AFC 2-3 Dorchester Town"
    assert [e["kind"] for e in m["events"]] == ["red", "goal", "sub", "yellow", "goal"]
    assert m["team_goals"] == [{"minute": "57", "scorer": "Tom Purrington", "penalty": False}]


def test_penalty_goals_count_for_the_team_and_the_scorer():
    page = MATCH_HTML.replace("<span>Tom Purrington scores</span>", "<span>Tom Purrington scores (pen)</span>").replace(
        "<span>James Moxon scores</span>", "<span>James Moxon scores (og)</span>")
    m = sf.parse_match(page, "dorchester-town")
    assert [e["kind"] for e in m["events"]] == ["red", "goal", "sub", "yellow", "own_goal"]
    assert m["team_goals"] == [{"minute": "57", "scorer": "Tom Purrington", "penalty": True}]
    assert sf.involvement(m, "Tom Purrington")["goals"] == ["57"]


def test_involvement_statuses_minutes_goals_cards():
    m = sf.parse_match(MATCH_HTML, "dorchester-town")
    assert sf.involvement(m, "Matt Buse") | {} == {**sf.involvement(m, "Matt Buse")}
    assert sf.involvement(m, "Matt Buse")["status"] == "started"
    assert sf.involvement(m, "Matt Buse")["off"] == "60"
    assert sf.involvement(m, "Henry Spalding")["status"] == "sub_on"
    assert sf.involvement(m, "Henry Spalding")["on"] == "60"
    assert sf.involvement(m, "Ieuan Turner")["status"] == "unused_sub"
    assert sf.involvement(m, "Corby Moore")["status"] == "not_in_squad"
    assert sf.involvement(m, "Tom Purrington")["goals"] == ["57"]
    assert sf.involvement(m, "Ollie Haste")["cards"] == ["yellow 70'"]
    assert sf.involvement(m, "Will Spetch")["captain"] is True


def test_involvement_without_team_sheet():
    m = sf.parse_match("<p>Full-time: A 1-0 B</p>", "dorchester-town")
    assert sf.involvement(m, "Ollie Haste")["status"] == "no_team_sheet"


def test_article_body_handles_void_tags_and_stops_at_body_end():
    art = sf.parse_article(ARTICLE_HTML)
    assert art["title"] == "Reaction: Brixham 2-3 The Magpies"
    assert art["date"] == "September 5, 2026"
    assert art["paragraphs"] == [
        "Post-match reaction from the captain:",
        "We got the job done. We concede from lapses at set plays and second phase balls.",
        "Embedded note about the pitch.",
    ]
    assert sf.tactical_sentences(art["paragraphs"][1:]) == [
        "We concede from lapses at set plays and second phase balls."]


def test_profile_fields_and_tactical_sentences():
    prof = sf.parse_profile(PROFILE_HTML)
    assert (prof["name"], prof["position"], prof["joined"], prof["previous_clubs"]) == \
        ("Will Spetch", "Defence", "September 2023", "Poole Town")
    assert len(sf.tactical_sentences([prof["bio"]])) == 2
    assert sf.parse_profile("<p>No profile here</p>") is None


# ── Failure contract: never raises ────────────────────────────────────────────

class _Resp:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")
        self.headers = email.message.Message()
        self.headers["Content-Type"] = "text/html; charset=utf-8"

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _stub_urlopen(monkeypatch, page_behaviour, robots="User-agent: *\nDisallow: /private\n"):
    calls = []

    def fake(req, timeout=None):
        url = req.full_url
        calls.append(url)
        if url.endswith("/robots.txt"):
            if isinstance(robots, Exception):
                raise robots
            return _Resp(robots)
        if isinstance(page_behaviour, Exception):
            raise page_behaviour
        return _Resp(page_behaviour)

    monkeypatch.setattr(sf.urllib.request, "urlopen", fake)
    return calls


@pytest.mark.parametrize("failure", [
    urllib.error.URLError("dns failure"),
    socket.timeout("timed out"),
    urllib.error.HTTPError("https://example.test/page", 503, "unavailable", hdrs=None, fp=None),
    ConnectionResetError("reset"),
])
def test_fetch_failure_returns_none_and_warns(monkeypatch, tmp_path, failure):
    _stub_urlopen(monkeypatch, failure)
    f = sf.Fetcher(cache_dir=tmp_path, delay=0)
    assert f.get("https://example.test/page") is None
    assert f.warnings and f.sources[-1]["status"] == "unavailable"


def test_fetch_failure_falls_back_to_cache(monkeypatch, tmp_path):
    _stub_urlopen(monkeypatch, socket.timeout("timed out"))
    f = sf.Fetcher(cache_dir=tmp_path, delay=0)
    url = "https://example.test/page"
    f.cache_path(url).parent.mkdir(parents=True)
    f.cache_path(url).write_text("<p>cached</p>", encoding="utf-8")
    assert f.get(url) == "<p>cached</p>"
    assert f.sources[-1]["status"] == "cache" and f.sources[-1]["reason"] == "network error"


def test_successful_fetch_writes_cache(monkeypatch, tmp_path):
    _stub_urlopen(monkeypatch, "<p>live</p>")
    f = sf.Fetcher(cache_dir=tmp_path, delay=0)
    assert f.get("https://example.test/page") == "<p>live</p>"
    assert f.cache_path("https://example.test/page").read_text(encoding="utf-8") == "<p>live</p>"
    assert f.sources[-1]["status"] == "fetched" and not f.warnings


def test_robots_disallow_blocks_page_request(monkeypatch, tmp_path):
    calls = _stub_urlopen(monkeypatch, "<p>secret</p>")
    f = sf.Fetcher(cache_dir=tmp_path, delay=0)
    assert f.get("https://example.test/private/page") is None
    assert calls == ["https://example.test/robots.txt"]
    assert any("robots.txt" in w for w in f.warnings)


def test_unreachable_robots_skips_host(monkeypatch, tmp_path):
    calls = _stub_urlopen(monkeypatch, "<p>page</p>", robots=urllib.error.URLError("down"))
    f = sf.Fetcher(cache_dir=tmp_path, delay=0)
    assert f.get("https://example.test/a") is None
    assert f.get("https://example.test/b") is None
    assert calls == ["https://example.test/robots.txt"]  # robots attempted once, pages never requested


def test_offline_mode_never_touches_network(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("network used in offline mode")

    monkeypatch.setattr(sf.urllib.request, "urlopen", boom)
    f = sf.Fetcher(offline=True, cache_dir=tmp_path, delay=0)
    assert f.get("https://example.test/page") is None
    assert "offline mode" in f.warnings[-1]


# ── Evidence assembly ─────────────────────────────────────────────────────────

class _StubFetcher(sf.Fetcher):
    def __init__(self, pages: dict):
        super().__init__(offline=True, delay=0)
        self.pages = pages

    def get(self, url):
        return self.pages.get(url)


def _pages(table=TABLE_HTML):
    base = "https://www.footballwebpages.co.uk/"
    pages = {base + "dorchester-town/league-table": table, base + "dorchester-town/fixtures-results": FIXTURES_HTML}
    for f in sf.parse_fixtures(FIXTURES_HTML):
        if f["url"]:
            pages[f["url"]] = MATCH_HTML
    return pages


def test_build_evidence_league_check_matches_table():
    fetcher = _StubFetcher(_pages())
    ev = sf.build_evidence("Dorchester Town", "dorchester-town", None, ["Ollie Haste", "Corby Moore"], 3, fetcher)
    check = ev["league"]["check"]
    assert check["league_label_in_fixtures"] == "Southern South"
    assert (check["results_counted"], check["record_from_results"], check["points_from_results"]) == (4, "W3 D0 L1", 9)
    assert check["matches_table"] and not fetcher.warnings
    assert [f["opponent"] for f in ev["recent_competitive"]] == ["Winchester City", "Hungerford Town", "Willand Rovers"]
    assert [r["status"] for r in ev["players"]["Corby Moore"]["involvement"]] == ["not_in_squad"] * 5
    assert ev["not_published"] == sf.NOT_PUBLISHED


def test_build_evidence_flags_table_mismatch():
    wrong = TABLE_HTML.replace("<td>9</td></tr>", "<td>12</td></tr>", 1)
    fetcher = _StubFetcher(_pages(wrong))
    ev = sf.build_evidence("Dorchester Town", "dorchester-town", None, [], 3, fetcher)
    assert ev["league"]["check"]["matches_table"] is False
    assert any("does not match table" in w for w in fetcher.warnings)


WP_REPORT_HTML = (
    '<h1 class="header-global-title">News</h1>'
    '<div class="post-header"><h1 class="post-title">Match Report: Andover New Street (H)</h1>'
    '<div class="post-meta"><span class="cat">Club News</span><span class="timestamp">5 September, 2026</span></div>'
    '<div class="post-summary"><p class="lead">Read all about today’s FA Cup win!</p></div></div>'
    '<div class="post-body prose"><p>Dotse got past his man on the right before sending a fizzing cross in.</p>'
    "<p>A foul led to a free kick on the right, with Olly Pendlebury curling it in.</p><p>Next up is Moneyfields.</p></div>"
    '<div class="post-footer"><h2>Share</h2></div>'
)


def test_wordpress_club_report_parses_title_date_and_body():
    art = sf.parse_article(WP_REPORT_HTML)
    assert (art["title"], art["date"]) == ("Match Report: Andover New Street (H)", "5 September, 2026")
    assert art["paragraphs"][0] == "Read all about today’s FA Cup win!" and "Share" not in art["paragraphs"]
    assert sf.tactical_sentences(art["paragraphs"][1:]) == [
        "Dotse got past his man on the right before sending a fizzing cross in.",
        "A foul led to a free kick on the right, with Olly Pendlebury curling it in."]
    assert sf._parse_date("5 September, 2026") == sf.datetime(2026, 9, 5)


def test_club_news_archive_pages_and_match_report_links():
    site = "https://club.test"
    first = ('<a href="https://club.test/match-report-plymouth-parkway-h/">r</a>'
             '<a href="https://club.test/match-preview-plymouth-parkway-h/">p</a><a href="https://club.test/news/page/2/">2</a>')
    second = '<a href="https://club.test/match-report-andover-new-street-h/">r</a>'
    old = WP_REPORT_HTML.replace("5 September, 2026", "21 July, 2025")
    pages = {**_pages(), f"{site}/news": first, f"{site}/news/page/2/": second,
             f"{site}/match-report-plymouth-parkway-h/": WP_REPORT_HTML, f"{site}/match-report-andover-new-street-h/": old}
    ev = sf.build_evidence("Dorchester Town", "dorchester-town", site + "/news/", [], 3, _StubFetcher(pages))
    assert [s["url"] for s in ev["statements"]] == [f"{site}/match-report-plymouth-parkway-h/"]    # last season's report dropped
    assert ev["statements"][0]["title"] == "Match Report: Andover New Street (H)"


def test_build_evidence_survives_total_outage():
    fetcher = _StubFetcher({})
    ev = sf.build_evidence("Dorchester Town", "dorchester-town", "https://club.test", ["Will Spetch"], 3, fetcher)
    assert ev["league"] is None and ev["fixtures"] == [] and ev["statements"] == []
    assert ev["players"]["Will Spetch"] == {"profile": None, "involvement": []}


def test_cli_bad_arguments_exit_2(capsys):
    assert sf.main(["--opponent", "!!!"]) == 2
