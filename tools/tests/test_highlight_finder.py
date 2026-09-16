"""
tools/tests/test_highlight_finder.py
Match reel finder: last-5 fixtures from evidence, targeted per-fixture search, proof that a video shows that
fixture (score or season date), multi-match caching and failed-search fallback.
Offline only — the YouTube search is a stub returning canned yt-dlp entries per query.

Run: python -m pytest tools/tests/test_highlight_finder.py -v
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import highlight_finder as hf  # noqa: E402

# Evidence fixtures, oldest first, as scout_fetcher writes them
PLAYED = [
    ("Sat 8 Aug", "Bath City", "home", 1, 2),
    ("Sat 22 Aug", "Chertsey Town", "home", 2, 4),
    ("Sat 29 Aug", "Bracknell Town", "away", 1, 3),
    ("Mon 31 Aug", "Gosport Borough", "home", 1, 0),
    ("Sat 5 Sep", "Andover New Street", "home", 2, 0),
    ("Sat 12 Sep", "Plymouth Parkway", "home", 1, 3),
]

VIDEOS = {
    "Chertsey Town": [{"id": "chertsey", "title": "Highlights - Sholing 2-4 Chertsey Town", "channel": "Sholing Football Club", "duration": 594}],
    "Bracknell Town": [
        {"id": "old_bracknell", "title": "Match Highlights - Bracknell Town 4-2 Sholing", "channel": "Sholing Football Club", "duration": 156},
        {"id": "bracknell", "title": "Bracknell 3-1 Sholing | Highlights", "channel": "Bracknell Town FC", "duration": 300},
    ],
    "Gosport Borough": [
        {"id": "old_gosport", "title": "Highlights | Sholing v Gosport Borough | 26.08.24", "channel": "Sholing Football Club", "duration": 245},
        {"id": "gosport", "title": "MATCH HIGHLIGHTS | Gosport Borough vs Sholing | 31.08.26", "channel": "Gosport Borough FC", "duration": 400},
    ],
    "Andover New Street": [{"id": "debut", "title": "Micky Hubbard scores on his Sholing debut", "channel": "Keith Legg", "duration": 10}],
    "Plymouth Parkway": [
        {"id": "old_parkway", "title": "Sholing 2 - 1 Plymouth Parkway | Full Match Highlights", "channel": "Sholing Football Club", "duration": 399},
        {"id": "parkway", "title": "Plymouth Parkway 3 – 1 Sholing | Match Highlights", "channel": "Sholing Football Club", "duration": 421},
    ],
}


def _evidence(tmp_path):
    ev = tmp_path / "evidence"
    ev.mkdir(exist_ok=True)
    fixtures = [{"date": d, "opponent": o, "venue": v, "played": True, "goals_for": gf, "goals_against": ga}
                for d, o, v, gf, ga in PLAYED]
    fixtures.append({"date": "Sat 19 Sep", "opponent": "Tiverton Town", "venue": "away", "played": False})
    (ev / "sholing.json").write_text(json.dumps({"league": {"season": "2026-2027"}, "fixtures": fixtures}), encoding="utf-8")
    return ev


class _Search:
    def __init__(self, fail_for=()):
        self.queries, self.fail_for = [], fail_for

    def __call__(self, query):
        self.queries.append(query)
        rival = next((r for r in VIDEOS if r in query), None)
        if rival in self.fail_for:
            raise RuntimeError("offline")
        return VIDEOS.get(rival, [])


def _fixture(rival, home, gf, ga):
    return {"opponent": "Sholing", "rival": rival, "home": home, "match_date": "Sat 1 Aug", "goals_for": gf, "goals_against": ga}


# ── Fixtures and queries ─────────────────────────────────────────────────────

def test_last_five_played_fixtures_newest_first(tmp_path):
    fixtures, years = hf.last_fixtures("Sholing", _evidence(tmp_path))
    assert [f["rival"] for f in fixtures] == ["Plymouth Parkway", "Andover New Street", "Gosport Borough",
                                              "Bracknell Town", "Chertsey Town"]
    assert years == {"2026", "2027"}
    away = fixtures[3]
    assert (hf.fixture_label(away), hf.score_line(away)) == ("Bracknell Town v Sholing", "3-1")
    assert hf.fixture_query(fixtures[0]) == '"Sholing" "Plymouth Parkway" highlights'


def test_missing_evidence_means_no_fixtures(tmp_path):
    assert hf.last_fixtures("Sholing", tmp_path) == ([], set())


# ── Proof that a video shows this fixture ────────────────────────────────────

def test_score_in_either_order_proves_the_fixture_and_other_scores_are_rejected():
    fx = _fixture("Plymouth Parkway", True, 1, 3)
    reel = hf.pick_reel(fx, VIDEOS["Plymouth Parkway"], {"2026", "2027"})
    assert reel == {"match_date": "Sat 1 Aug", "fixture_label": "Sholing v Plymouth Parkway", "score": "1-3",
                    "title": "Plymouth Parkway 3 – 1 Sholing | Match Highlights",
                    "url": "https://www.youtube.com/watch?v=parkway", "channel": "Sholing Football Club"}
    assert hf.pick_reel(fx, VIDEOS["Plymouth Parkway"][:1], {"2026"}) is None      # 2-1 is an older meeting


def test_short_club_names_and_season_dates_prove_a_fixture():
    assert hf.pick_reel(_fixture("Bracknell Town", False, 1, 3), VIDEOS["Bracknell Town"], {"2026", "2027"})["url"].endswith("=bracknell")
    gosport = hf.pick_reel(_fixture("Gosport Borough", True, 1, 0), VIDEOS["Gosport Borough"], {"2026", "2027"})
    assert gosport["url"].endswith("=gosport")                                    # 31.08.26, not 26.08.24


def test_videos_must_name_both_clubs_and_show_the_first_team():
    fx = _fixture("Chertsey Town", True, 2, 4)
    women = {"id": "w", "title": "Sholing Women 2-4 Chertsey Town Women | Highlights", "channel": "Chertsey Town FC"}
    other = {"id": "o", "title": "Chertsey Town 2-4 Weymouth | Highlights", "channel": "Chertsey Town FC"}
    by_channel = {"id": "c", "title": "Highlights v Sholing (2-4)", "channel": "Chertsey Town FC"}
    assert hf.pick_reel(fx, [women, other], {"2026"}) is None
    assert hf.pick_reel(fx, [women, other, by_channel], {"2026"})["url"].endswith("=c")
    assert hf.pick_reel(_fixture("Andover New Street", True, 2, 0), VIDEOS["Andover New Street"], {"2026"}) is None


# ── Search and cache ─────────────────────────────────────────────────────────

def test_each_fixture_gets_a_targeted_search_and_the_reels_are_cached(tmp_path):
    search = _Search()
    result = hf.find_match_reels("Sholing", tmp_path / "sources", search=search, evidence=_evidence(tmp_path))
    assert search.queries == ['"Sholing" "Plymouth Parkway" highlights', '"Sholing" "Andover New Street" highlights',
                              "Sholing Andover New Street highlights",              # unquoted retry when nothing proves the match
                              '"Sholing" "Gosport Borough" highlights', '"Sholing" "Bracknell Town" highlights',
                              '"Sholing" "Chertsey Town" highlights']

    reels = result["reels"]
    assert [(r["match_date"], r["fixture_label"], r["score"]) for r in reels] == [
        ("Sat 12 Sep", "Sholing v Plymouth Parkway", "1-3"), ("Mon 31 Aug", "Sholing v Gosport Borough", "1-0"),
        ("Sat 29 Aug", "Bracknell Town v Sholing", "3-1"), ("Sat 22 Aug", "Sholing v Chertsey Town", "2-4")]
    assert all(set(r) == {"match_date", "fixture_label", "score", "title", "url", "channel"} for r in reels)
    assert result["warnings"] == ["No highlight reel found for Sholing v Andover New Street (Sat 5 Sep)."]

    saved = json.loads((tmp_path / "sources" / "sholing" / "video_links.json").read_text(encoding="utf-8"))
    assert saved == reels and hf.cached_reels("Sholing", tmp_path / "sources") == reels


def test_saved_reels_are_reused_until_new_results_are_fetched(tmp_path):
    sources, evidence = tmp_path / "sources", _evidence(tmp_path)
    first = hf.find_match_reels("Sholing", sources, search=_Search(), evidence=evidence)["reels"]

    search = _Search()
    assert hf.find_match_reels("Sholing", sources, search=search, evidence=evidence) == {"reels": first, "warnings": []}
    assert search.queries == []                                                   # a fixture with no reel does not force a search

    links = hf.links_path(sources, "Sholing")
    later = links.stat().st_mtime + 60
    os.utime(evidence / "sholing.json", (later, later))                           # scout_fetcher saved a new result
    hf.find_match_reels("Sholing", sources, search=search, evidence=evidence)
    assert len(search.queries) == 6

    search = _Search()
    hf.find_match_reels("Sholing", sources, refresh=True, search=search, evidence=evidence)
    assert len(search.queries) == 6


def test_failed_search_keeps_that_fixtures_saved_reel(tmp_path):
    sources, evidence = tmp_path / "sources", _evidence(tmp_path)
    first = hf.find_match_reels("Sholing", sources, search=_Search(), evidence=evidence)["reels"]
    result = hf.find_match_reels("Sholing", sources, refresh=True, evidence=evidence,
                                 search=_Search(fail_for=("Chertsey Town", "Andover New Street")))
    assert result["reels"] == first
    assert result["warnings"] == [
        "Search failed for Sholing v Andover New Street (Sat 5 Sep, RuntimeError); no reel saved.",
        "Search failed for Sholing v Chertsey Town (Sat 22 Aug, RuntimeError); kept the saved reel.",
    ]


def test_no_evidence_warns_and_searches_nothing(tmp_path):
    search = _Search()
    result = hf.find_match_reels("Sholing", tmp_path / "sources", search=search, evidence=tmp_path)
    assert result == {"reels": [], "warnings": ["No played fixtures for Sholing yet. Fetch their results first."]}
    assert search.queries == []
