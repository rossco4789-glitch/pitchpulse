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
                  "home_game": False, "date": "2026-09-15", "kickoff": "19:45", "venue": "The Avenue Stadium"}


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
