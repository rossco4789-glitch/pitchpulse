"""
tools/tests/test_highlight_finder.py
Highlight finder: ranking by recent fixtures, non-first-team penalties, link caching, failed-search fallback.
Offline only — the YouTube search is a stub returning canned yt-dlp entries.

Run: python -m pytest tools/tests/test_highlight_finder.py -v
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import highlight_finder as hf  # noqa: E402

ENTRIES = [
    {"id": "a", "title": "Match Highlights - Sholing 1-2 Bath City", "channel": "Sholing Football Club", "duration": 724},
    {"id": "b", "title": "Highlights - Sholing 2-4 Chertsey Town", "channel": "Sholing Football Club", "duration": 594},
    {"id": "c", "title": "Match Highlights | Sholing FC vs Winchester City FC | Pre-Season", "channel": "Winchester City FC", "duration": 260},
    {"id": "d", "title": "Sholing Women 1-7 Farnham Town Women | Highlights", "channel": "Farnham Town FC", "duration": 300},
    {"id": "e", "title": "Sholing highlights", "channel": "Sholing Football Club", "duration": 696},
    {"id": "f", "title": "Bath City 3-0 Weymouth | Highlights", "channel": "Bath City FC", "duration": 400},
    {"title": "no link at all Sholing", "channel": "Somebody"},
]
RECENT = [{"opponent": "Bath City", "date": "Sat 8 Aug"}, {"opponent": "Chertsey Town", "date": "Sat 22 Aug"}]


def _evidence(tmp_path):
    ev = tmp_path / "evidence"
    ev.mkdir()
    fixtures = [{**r, "played": True} for r in RECENT] + [{"opponent": "Tiverton Town", "date": "Sat 19 Sep", "played": False}]
    (ev / "sholing.json").write_text(json.dumps({"fixtures": fixtures}), encoding="utf-8")
    return ev


class _Search:
    def __init__(self, entries=ENTRIES, error=None):
        self.entries, self.error, self.queries = entries, error, []

    def __call__(self, query):
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.entries


def test_most_recent_fixture_ranks_first_and_unrelated_clubs_are_dropped():
    ranked = hf.rank_highlights("Sholing", ENTRIES, RECENT)
    assert [v["url"][-1] for v in ranked] == ["b", "a", "e", "d", "c"]
    assert (ranked[0]["fixture"], ranked[0]["score"]) == ("Sat 22 Aug v Chertsey Town", 15)
    assert ranked[0]["url"] == "https://www.youtube.com/watch?v=b"


def test_pre_season_and_womens_reels_rank_below_first_team_reels():
    ranked = {v["url"][-1]: v["score"] for v in hf.rank_highlights("Sholing", ENTRIES)}
    assert ranked["c"] < ranked["e"] and ranked["d"] < ranked["e"]


def test_results_are_cached_and_reused(tmp_path):
    search = _Search()
    data = hf.find_highlights("Sholing", tmp_path / "sources", search=search, evidence=_evidence(tmp_path))
    path = tmp_path / "sources" / "sholing" / "video_links.json"
    assert json.loads(path.read_text(encoding="utf-8")) == data
    assert data["videos"][0]["title"] == "Highlights - Sholing 2-4 Chertsey Town"
    assert search.queries == ["Sholing highlights"]

    again = hf.find_highlights("Sholing", tmp_path / "sources", search=search)
    assert again == data and len(search.queries) == 1                      # cache hit, no second search
    assert hf.cached_highlights("Sholing", tmp_path / "sources") == data

    hf.find_highlights("Sholing", tmp_path / "sources", refresh=True, search=search, evidence=tmp_path / "evidence")
    assert len(search.queries) == 2


def test_failed_search_keeps_the_saved_list(tmp_path):
    saved = hf.find_highlights("Sholing", tmp_path, search=_Search(), evidence=tmp_path / "none")
    failed = hf.find_highlights("Sholing", tmp_path, refresh=True, search=_Search(error=RuntimeError("offline")))
    assert failed["videos"] == saved["videos"]
    assert failed["warnings"] == ["highlight search failed (RuntimeError); showing the last saved list"]


def test_failed_search_with_no_cache_returns_an_empty_list(tmp_path):
    data = hf.find_highlights("Sholing", tmp_path, search=_Search(error=OSError("offline")))
    assert data["videos"] == [] and data["warnings"]
    assert not (tmp_path / "sholing" / "video_links.json").exists()
