"""
tools/tests/test_scout_brief.py
Opposition briefing builder: fixture detection, degraded offline path, evidence path, brief protection.
Offline only — the fetcher is either an empty offline cache or a stubbed build_evidence.

Run: python -m pytest tools/tests/test_scout_brief.py -v
"""

import io
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import scout_brief as sb  # noqa: E402
import scout_fetcher as sf  # noqa: E402

TODAY = date(2026, 9, 14)
PREVIEW = (
    "Match Preview\nTuesday September 15th\nDorchester Town v Tiverton Town\n7.45 pm Kick Off\n"
    "At The Avenue Stadium\n"
    "Tiverton will be without the suspended Finn Roberts too. "
    "Defender Harvey Bertrand has returned to the club this term. "
    "His influence is a testament to that and he wins headers from corners. "
    "Midfielder Corby Moore is a schemer who controls the tempo in midfield.\n"
)


def _offline(tmp_path):
    return sf.Fetcher(offline=True, cache_dir=tmp_path / "cache", delay=0)


def _run(tmp_path, data=PREVIEW.encode(), name="preview.txt", **kw):
    return sb.run(name, data, root=tmp_path, today=TODAY, progress=lambda _m: None,
                  fetcher=_offline(tmp_path), **kw)


def _assert_packager_safe(md: str):
    assert "_" not in md, "underscores render as italics in the packager"
    assert not sb.BANNED.search(md)
    assert "SET PIECE" not in md.upper() and "MOMENT 1" not in md.upper()


# ── Detection ─────────────────────────────────────────────────────────────────

def test_detects_away_fixture_date_kickoff_venue():
    fx = sb.detect_fixture(PREVIEW, TODAY)
    assert fx == {"home": "Dorchester Town", "away": "Tiverton Town", "opponent": "Dorchester Town",
                  "home_game": False, "date": "2026-09-15", "kickoff": "19:45", "venue": "The Avenue Stadium",
                  "competition": None}


def test_detects_home_fixture_with_explicit_year():
    fx = sb.detect_fixture("Tiverton Town vs Willand Rovers\nSaturday 3rd October 2026\n", TODAY)
    assert (fx["opponent"], fx["home_game"], fx["date"]) == ("Willand Rovers", True, "2026-10-03")


def test_missing_versus_line_raises():
    with pytest.raises(ValueError, match="Tiverton"):
        sb.detect_fixture("Match Preview\nA big night under the lights\n", TODAY)


def test_docx_and_bom_text_extraction():
    import docx
    doc = docx.Document()
    doc.add_paragraph("Dorchester Town v Tiverton Town")
    buf = io.BytesIO()
    doc.save(buf)
    assert "Dorchester Town v Tiverton Town" in sb.extract_text("p.docx", buf.getvalue())
    assert sb.extract_text("p.md", "﻿hello".encode("utf-8")) == "hello"
    with pytest.raises(ValueError):
        sb.extract_text("p.pdf", b"%PDF")


# ── Degraded path ─────────────────────────────────────────────────────────────

def test_offline_run_degrades_to_notes_and_still_compiles(tmp_path):
    res = _run(tmp_path)
    assert res["degraded"] and res["division"] is None
    assert any("unavailable" in w for w in res["warnings"])
    md = res["brief_path"].read_text(encoding="utf-8")
    _assert_packager_safe(md)
    assert "Finn Roberts" in md and "Tuesday, 15 September 2026" in md
    assert "testament" not in md  # banned-vocabulary preview sentence dropped, not quoted
    assert "Match Preview" not in md.split("---", 1)[1]  # header lines are not quoted as claims
    html = res["html_path"].read_text(encoding="utf-8")
    assert res["html_path"].name == "dorchester_town_scouting.html" and "<img" not in html
    assert (tmp_path / "data/scouting/sources/dorchester_town/preview.txt").read_bytes() == PREVIEW.encode()


def test_route_ignores_impressive_and_shape_idiom():
    sents = ["He has an impressive reputation.", "They face us in the shape of the Magpies.", "They press high."]
    assert sb._route(sents, "out_of_possession") == ["They press high."]


def test_fetch_exception_never_escapes(tmp_path, monkeypatch):
    monkeypatch.setattr(sf, "build_evidence", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    res = _run(tmp_path)
    assert res["degraded"] and any("boom" in w for w in res["warnings"])


# ── Evidence path ─────────────────────────────────────────────────────────────

def _player(shirt, name, captain=False, started=True):
    return {"shirt": shirt, "name": name, "captain": captain, "started": started}


def _match(goals):
    starters = [_player(1, "Ryan Hall"), _player(4, "Will Spetch", captain=True),
                _player(2, "Harvey-Joe Bertrand"), _player(8, "Corby Moore")] + \
               [_player(20 + i, f"Squad Player{i}") for i in range(7)]
    events = [{"minute": m, "kind": "goal", "text": f"{s} scores"} for m, s in goals]
    names = {p["name"] for p in starters}
    return {"lineup_published": True, "starters": starters, "bench": [], "events": events,
            "team_goals": [{"minute": m, "scorer": s} for m, s in goals if s in names]}


def _evidence():
    fixtures = [
        {"date": "Sat 8 Aug", "venue": "away", "opponent": "Weymouth", "competition": "Southern South", "played": True,
         "goals_for": 1, "goals_against": 2, "result": "L", "match": _match([("30", "Corby Moore"), ("88", "Opp Nine")])},
        {"date": "Sat 15 Aug", "venue": "home", "opponent": "Paulton Rovers", "competition": "Southern South", "played": True,
         "goals_for": 2, "goals_against": 0, "result": "W", "match": _match([("12", "Will Spetch"), ("90+2", "Corby Moore")])},
    ]
    standings = [{"position": 4, "team": "Dorchester Town", "played": 2, "won": 1, "drawn": 0, "lost": 1, "goals_for": 3,
                  "goals_against": 2, "goal_difference": 1, "points": 3},
                 {"position": 12, "team": "Tiverton Town", "played": 2, "won": 0, "drawn": 1, "lost": 1, "goals_for": 1,
                  "goals_against": 2, "goal_difference": -1, "points": 1}]
    return {"league": {"division": "Southern League - South Division", "season": "2026-2027", "last_updated": "12 Sep",
                       "teams": 22, "row": standings[0], "standings": standings,
                       "check": {"record_from_results": "W1 D0 L1", "points_from_results": 3, "matches_table": True}},
            "fixtures": fixtures, "statements": [{"title": "Reaction: Brixham 2-3", "date": "September 5, 2026",
                                                   "tactical_sentences": ["We lose concentration at set plays."]}],
            "players": {}, "warnings": [], "sources": []}


def test_evidence_run_builds_verified_sections(tmp_path, monkeypatch):
    monkeypatch.setattr(sf, "build_evidence", lambda *a, **k: _evidence())
    res = _run(tmp_path)
    assert (res["degraded"], res["position"], res["teams"], res["points"]) == (False, 4, 22, 3)
    md = res["brief_path"].read_text(encoding="utf-8")
    _assert_packager_safe(md)
    for needle in ("**League position:** 4 of 22", "**Tiverton:** 12 with 1 points", "(L W)",
                   "1 Ryan Hall, 4 Will Spetch (captain)", "Corby Moore 2", "conceded after it in 1",
                   "We lose concentration at set plays.", "**Harvey-Joe Bertrand (number 2):** 2 starts",
                   "matches the table"):
        assert needle in md, needle
    assert (tmp_path / "data/scouting/evidence/dorchester_town.json").exists()


# ── Synthesis fixes ───────────────────────────────────────────────────────────

FIXTURE_STUB = ("Tiverton Town v Sholing — Sat 19 Sep 2026, 15:00, The Slee Blackwell Solicitors Stadium\n"
                "Kick off 15:00\nAt The Slee Blackwell Solicitors Stadium\nCompetition: FA Cup 2Q\n")


def _fx(**over):
    return {**sb.detect_fixture(FIXTURE_STUB, TODAY), **over}


def _sholing(sheets):
    """Evidence with one team sheet per (competition, goals_against, captain, events) tuple, oldest first."""
    squad = ["Jack Turner", "Harry Taylor", "Byron Mason", "Jake Mccarthy"] + [f"Squad Player{i}" for i in range(7)]
    fixtures = []
    for i, (competition, against, captain, events) in enumerate(sheets):
        starters = [_player(n + 1, name, captain=name == captain) for n, name in enumerate(squad)]
        match = {"lineup_published": True, "starters": starters, "bench": [],
                 "events": [{"minute": m, "kind": k, "text": t} for m, k, t in events]}
        fixtures.append({"date": f"Sat {i + 1} Aug", "venue": "home", "opponent": f"Rival {i}", "competition": competition,
                         "played": True, "goals_for": 1, "goals_against": against, "result": "L", "match": match})
    league = {"division": "Southern League - Premier South", "season": "2026-2027", "last_updated": "15 Sep", "teams": 22,
              "row": {"position": 16, "played": 3, "won": 0, "drawn": 0, "lost": 3, "goals_for": 3, "goals_against": 7, "points": 0},
              "standings": [], "check": {"league_label_in_fixtures": "Southern Prem South", "record_from_results": "W0 D0 L3",
                                         "points_from_results": 0, "matches_table": True}}
    return {"league": league, "fixtures": fixtures, "statements": [], "players": {}, "warnings": [], "sources": []}


PEN = ("45+2", "other", "Jake Mccarthy scores (pen)")
LEAGUE = "Southern Prem South"


def test_fixture_title_strips_header_metadata_and_reads_the_competition():
    fx = sb.detect_fixture(FIXTURE_STUB, TODAY)
    assert (fx["home"], fx["away"], fx["opponent"], fx["competition"]) == ("Tiverton Town", "Sholing", "Sholing", "FA Cup 2Q")
    assert (fx["date"], fx["kickoff"], fx["venue"]) == ("2026-09-19", "15:00", "The Slee Blackwell Solicitors Stadium")
    md = sb.compose_brief(fx, FIXTURE_STUB, None, TODAY)
    assert md.startswith("# Tiverton Town v Sholing — Pre-Match Briefing\n")
    assert sb.detect_fixture("Bishop's Cleeve v Tiverton Town, 7.45pm\n", TODAY)["opponent"] == "Bishop's Cleeve"
    assert sb.detect_fixture("Sholing v Tiverton Town\nEmirates FA Cup Second Qualifying Round\n", TODAY)["competition"] == "FA Cup"


def test_cup_competition_takes_precedence_over_the_opponent_league():
    ev = _sholing([(LEAGUE, 2, "Byron Mason", [])])
    md = sb.compose_brief(_fx(), "", ev, TODAY)
    assert "**Competition:** FA Cup 2Q" in md and "**Opponent league:** Southern League - Premier South, 2026-2027" in md
    ev["fixtures"].append({"date": "Sat 19 Sep", "venue": "away", "opponent": "Tiverton Town", "competition": "FA Cup 2Q", "played": False})
    assert "**Competition:** FA Cup 2Q" in sb.compose_brief(_fx(competition=None), "", ev, TODAY)       # from their fixture list
    assert "**Competition:** Southern League - Premier South, 2026-2027" in sb.compose_brief(
        _fx(competition=None), "", _sholing([(LEAGUE, 2, "Byron Mason", [])]), TODAY)


def test_penalties_count_towards_goal_tallies_and_ties_share_top_billing():
    ev = _sholing([(LEAGUE, 1, "Byron Mason", [("29", "goal", "Harry Taylor scores"), PEN]),
                   (LEAGUE, 0, "Byron Mason", [("24", "goal", "Jake Mccarthy scores"), ("80", "goal", "Harry Taylor scores"),
                                               ("88", "goal", "Opp Nine scores (pen)")])])
    md = sb.compose_brief(_fx(), "", ev, TODAY)
    assert "**Goal threat:** Harry Taylor 2, Jake Mccarthy 2" in md
    assert "**Primary finishers:** Harry Taylor and Jake Mccarthy, 2 goals each." in md
    assert "tracking Harry Taylor and Jake Mccarthy" in md
    assert "conceded after it in 1" in md                                     # the opponent's late penalty is a goal against


def test_defensive_denominator_uses_scored_games_split_by_competition():
    ev = _sholing([(LEAGUE, 4, "Byron Mason", []), (LEAGUE, 3, "Byron Mason", []), ("FA Cup 1Q", 0, "Harry Taylor", [])])
    ev["fixtures"].append({"date": "Sat 9 Sep", "venue": "away", "opponent": "Abandoned FC", "competition": LEAGUE,
                           "played": True, "goals_for": None, "goals_against": None})
    md = sb.compose_brief(_fx(), "", ev, TODAY)
    assert ("**Defensive record:** 7 conceded in 2 league games (0 clean sheets, most in one game 4); "
            "0 conceded in 1 cup game (1 clean sheet, most in one game 0).") in md


def test_captaincy_reports_the_recent_armband_first():
    ev = _sholing([(LEAGUE, 1, c, []) for c in ["Byron Mason"] * 5 + ["Harry Taylor"] * 2])
    assert ("**Captain:** Harry Taylor has worn the armband in the last 2 team sheets. "
            "Byron Mason wore it in 5 of 7.") in sb.compose_brief(_fx(), "", ev, TODAY)
    ev = _sholing([(LEAGUE, 1, c, []) for c in ["Harry Taylor", "Byron Mason", "Byron Mason"]])
    assert "**Captain:** Byron Mason wore the armband in 2 of 3 team sheets, including the last 2." in sb.compose_brief(_fx(), "", ev, TODAY)
    ev = _sholing([(LEAGUE, 1, "Byron Mason", [])] * 2)
    assert "**Captain:** Byron Mason wore the armband in 2 of 2 team sheets." in sb.compose_brief(_fx(), "", ev, TODAY)


def test_absentee_lever_only_when_the_preview_names_absentees():
    md = sb.compose_brief(_fx(), FIXTURE_STUB, None, TODAY)
    assert "absentee listed above" not in md and "**Absentees:** none named in the preview." in md
    notes = FIXTURE_STUB + "Their captain Harry Taylor is suspended after five bookings this season so far.\n"
    assert "absentee listed above" in sb.compose_brief(_fx(), notes, None, TODAY)


# ── Hand-edited brief protection ──────────────────────────────────────────────

def test_hand_edited_brief_is_kept_unless_overwrite(tmp_path):
    brief = tmp_path / "data/scouting/sources/dorchester_town/brief.md"
    brief.parent.mkdir(parents=True)
    brief.write_text("# Verified by staff\n", encoding="utf-8")
    res = _run(tmp_path)
    assert res["kept_manual"] and brief.read_text(encoding="utf-8") == "# Verified by staff\n"
    assert sb.AUTO_MARKER in (brief.parent / "brief_auto.md").read_text(encoding="utf-8")
    assert "Verified by staff" in res["html_path"].read_text(encoding="utf-8")

    res = _run(tmp_path, overwrite=True)
    assert not res["kept_manual"] and sb.AUTO_MARKER in brief.read_text(encoding="utf-8")
    assert not _run(tmp_path)["kept_manual"]  # an auto brief is regenerated freely
