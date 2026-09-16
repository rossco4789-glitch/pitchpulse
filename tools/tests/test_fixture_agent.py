"""
tools/tests/test_fixture_agent.py
Fixture agent: feed parsing, home/away designation, manifest precedence, cup opponent asset fallback, preview stub.
Offline only — the fixtures page is an inline HTML table and Wikipedia / scout_fetcher are stubbed.

Run: python -m pytest tools/tests/test_fixture_agent.py -v
"""

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import fixture_agent as fa  # noqa: E402
import scout_brief as sb  # noqa: E402
from cv import club_assets as ca  # noqa: E402

TODAY = date(2026, 9, 16)
PAGE = """<table>
<tr><th>Date</th><th>H/A</th><th>Opponent</th><th>Competition</th><th>Score</th></tr>
<tr><td>Sat 12 Sep</td><td>A</td><td>Weymouth</td><td>Southern South</td><td>2-1</td></tr>
<tr><td>Tue 15 Sep</td><td>A</td><td>Dorchester Town</td><td>Southern South</td><td>P - P</td></tr>
<tr><td>Sat 19 Sep</td><td>H</td><td>Sholing</td><td>FA Cup 2Q</td><td>3pm</td></tr>
<tr><td>Sat 26 Sep</td><td>A</td><td>Swindon Supermarine</td><td>Southern South</td><td>7.45pm</td></tr>
</table>"""


class _Page:
    def __init__(self, body):
        self.body, self.urls = body, []

    def get(self, url):
        self.urls.append(url)
        return self.body


def test_feed_skips_played_and_postponed_rows():
    fx = fa.parse_next_fixture(PAGE, TODAY)
    assert fx == {"opponent": "Sholing", "home_away": "Home", "venue": "The Slee Blackwell Solicitors Stadium",
                  "date": "2026-09-19", "date_str": "Sat 19 Sep 2026", "kickoff": "15:00", "competition": "FA Cup", "slug": "sholing"}


def test_away_row_designation_and_league_competition():
    fx = fa.parse_next_fixture(PAGE, date(2026, 9, 20))
    assert (fx["opponent"], fx["home_away"], fx["venue"]) == ("Swindon Supermarine", "Away", "Swindon Supermarine (away)")
    assert (fx["competition"], fx["kickoff"]) == ("Southern League", "19:45")


def test_competition_names():
    assert fa.competition_name("FA Trophy 3Q") == "FA Trophy"
    assert fa.competition_name("Devon St Luke's Cup") == "Devon St Luke's Cup"


def test_get_next_fixture_uses_feed_when_no_manifest(tmp_path):
    page = _Page(PAGE)
    fx = fa.get_next_fixture(today=TODAY, manifest=tmp_path / "missing.json", fetcher=page)
    assert fx["opponent"] == "Sholing"
    assert page.urls == ["https://www.footballwebpages.co.uk/tiverton-town/fixtures-results"]


def test_manifest_takes_precedence_and_ignores_past_rows(tmp_path):
    manifest = tmp_path / "fixtures.json"
    manifest.write_text(json.dumps([
        {"opponent": "Weymouth", "home_away": "Away", "date": "2026-09-12", "competition": "Southern South"},
        {"opponent": "Gosport Borough", "home_away": "Home", "date": "2026-09-26", "kickoff": "15:00",
         "competition": "FA Trophy 3Q"},
    ]), encoding="utf-8")
    page = _Page(PAGE)
    fx = fa.get_next_fixture(today=TODAY, manifest=manifest, fetcher=page)
    assert (fx["opponent"], fx["home_away"], fx["competition"]) == ("Gosport Borough", "Home", "FA Trophy")
    assert page.urls == []


def test_no_feed_returns_none(tmp_path):
    assert fa.get_next_fixture(today=TODAY, manifest=tmp_path / "missing.json", fetcher=_Page(None)) is None


def test_stub_header_and_sections_read_back_by_scout_brief(tmp_path):
    fx = fa.parse_next_fixture(PAGE, TODAY)
    path = Path(fa.scaffold_preview_stub(fx, previews=tmp_path))
    text = path.read_text(encoding="utf-8")
    assert path.name == "sholing.md"
    assert text.splitlines()[0] == "Tiverton Town v Sholing — Sat 19 Sep 2026, 15:00, The Slee Blackwell Solicitors Stadium"
    for section in ("## Formation", "## Key Players", "## Manager Notes"):
        assert section in text
    detected = sb.detect_fixture(text, TODAY)
    assert (detected["home_game"], detected["date"], detected["kickoff"], detected["venue"]) == \
        (True, "2026-09-19", "15:00", "The Slee Blackwell Solicitors Stadium")


def test_away_stub_puts_opponent_first(tmp_path):
    fx = fa.parse_next_fixture(PAGE, date(2026, 9, 20))
    text = Path(fa.scaffold_preview_stub(fx, previews=tmp_path)).read_text(encoding="utf-8")
    assert text.startswith("Swindon Supermarine v Tiverton Town — Sat 26 Sep 2026, 19:45,")


def test_stub_keeps_manager_notes(tmp_path):
    fx = fa.parse_next_fixture(PAGE, TODAY)
    path = Path(fa.scaffold_preview_stub(fx, previews=tmp_path))
    path.write_text("edited by the manager", encoding="utf-8")
    fa.scaffold_preview_stub(fx, previews=tmp_path)
    assert path.read_text(encoding="utf-8") == "edited by the manager"


def test_cup_opponent_offline_falls_back_to_default_kits(tmp_path, monkeypatch):
    def offline(_name):
        raise ca.NETWORK_ERRORS[0]("offline")
    monkeypatch.setattr(ca, "find_club_page", offline)
    calls = []
    res = fa.sync_fixture_to_scout(
        fa.parse_next_fixture(PAGE, TODAY),
        resolve=lambda name: ca.resolve_club_assets(name, sources=tmp_path),
        run=lambda cmd, cwd: calls.append(cmd) or type("R", (), {"returncode": 0})(),
    )
    assert res["assets"]["slug"] == "sholing"
    assert (res["assets"]["home_kit"], res["assets"]["kit_source"]) == (ca.DEFAULT_HOME, "default")
    assert res["fetcher_exit"] == 0
    assert calls[0][1:] == [str(fa.FETCHER_PATH), "--opponent", "Sholing"]


def test_fetcher_that_fails_to_start_does_not_raise():
    def boom(cmd, cwd):
        raise OSError("no python")
    res = fa.sync_fixture_to_scout({"opponent": "Sholing"}, resolve=lambda n: {"slug": "sholing"}, run=boom)
    assert res["fetcher_exit"] is None
