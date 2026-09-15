"""
tools/tests/test_league_roster.py
Southern League Division One South quick-select (cv/league_roster.py) and its Match Setup behaviour. Offline:
Wikipedia calls go through cv.club_assets._http_json / _http_bytes, which the render test replaces.

Run: python -m pytest tools/tests/test_league_roster.py -v
"""

import json
import re
import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import prewarm_league_assets as pw  # noqa: E402
from cv import club_assets as ca  # noqa: E402
from cv import league_roster as lr  # noqa: E402
from cv import vision_center as vc  # noqa: E402

BANNED = ("homography", "bounding box", "yolo", "pipeline", "acceptance rate", "silhouette", "iqr", "median",
          "settled frames", "pid", "subprocess", "kaggle", "dispatch_id", "dispatch id", "manifest", "hex")


# ── Roster ───────────────────────────────────────────────────────────────────

def test_roster_is_the_supplied_22_clubs_in_order():
    roster = lr.SOUTHERN_LEAGUE_DIV_ONE_SOUTH
    assert len(roster) == 22 and len(set(roster)) == 22
    assert roster == sorted(roster, key=str.lower)
    assert {"Dorchester Town", "Weymouth", "Tiverton Town", "Sporting Club Inkberrow"} <= set(roster)
    assert len({ca.slugify(club) for club in roster}) == 22 and ca.slugify("Bishop's Cleeve") == "bishop_s_cleeve"


def test_options_exclude_our_own_club_and_frame_the_list():
    options = lr.opponent_options()
    assert options[0] == "-- Select Opponent --" and options[-1] == "Other / Custom..."
    assert options[1:-1] == [c for c in lr.SOUTHERN_LEAGUE_DIV_ONE_SOUTH if c != "Tiverton Town"] and len(options) == 23


@pytest.mark.parametrize("pick,typed,expected", [
    ("Dorchester Town", "ignored", "Dorchester Town"),
    ("-- Select Opponent --", "Taunton Town", ""),
    ("Other / Custom...", "  Taunton Town ", "Taunton Town"),
    ("Other / Custom...", None, ""),
    ("Tiverton Town", None, ""),
    (None, None, ""),
])
def test_opponent_name(pick, typed, expected):
    assert lr.opponent_name(pick, typed) == expected


# ── Match Setup ──────────────────────────────────────────────────────────────

WIKITEXT = "{{Infobox football club\n| pattern_b1 = _whitestripes\n| body1 = 000000\n| body2 = 0000EE\n}}"
SEARCH = {"pages": [{"key": "Dorchester_Town_F.C.", "title": "Dorchester Town F.C.", "description": "Association football club in England"}]}
SUMMARY = {"title": "Dorchester Town F.C.", "thumbnail": {"source": "https://upload.wikimedia.org/crest.png"}}


def _crest() -> bytes:
    out = BytesIO()
    Image.new("RGBA", (40, 40), (90, 140, 250, 255)).save(out, "PNG")
    return out.getvalue()


def _tab_script():
    from cv.vision_center_ui import render
    render()


def _all_text(node) -> list[str]:
    out = [str(node.proto)] if getattr(node, "proto", None) is not None else []
    for child in getattr(node, "children", {}).values():
        out += _all_text(child)
    return out


# ── Batch pre-warm ───────────────────────────────────────────────────────────

def _league_wikipedia(monkeypatch):
    """Fake Wikipedia with one edge case per club: records, crest only, no page, offline, bad JSON, bad crest."""
    from urllib.error import URLError
    from urllib.parse import parse_qs, urlparse

    def fake_json(url):
        if "search/page" in url:
            club = parse_qs(urlparse(url).query)["q"][0].removesuffix(" F.C.")
            if club.startswith("Exmouth"):
                raise URLError("connection reset")
            if club.startswith("Hungerford"):
                raise ValueError("Expecting value: line 1 column 1")   # truncated JSON
            if club.startswith("Bideford"):
                return {"pages": [{"key": "Bideford", "title": "Bideford", "description": "Town in Devon, England"}]}
            key = f"{club.replace(' ', '_')}_F.C."
            return {"pages": [{"key": key, "title": key.replace("_", " "), "description": "Association football club in England"}]}
        key = url.rsplit("/", 1)[1]
        if "summary" in url:
            return {"title": key.replace("_", " ").replace("%27", "'"), "thumbnail": {"source": f"https://upload.example/{key}.png"}}
        return {"source": WIKITEXT if key.startswith("Dorchester") else ""}

    def fake_bytes(url):
        return b"<html>moved</html>" if "Paulton" in url else _crest()

    monkeypatch.setattr(ca, "_http_json", fake_json)
    monkeypatch.setattr(ca, "_http_bytes", fake_bytes)


def test_prewarm_reports_each_edge_case_and_never_crashes(tmp_path, monkeypatch):
    _league_wikipedia(monkeypatch)
    clubs = ["Dorchester Town", "Weymouth", "Bideford", "Exmouth Town", "Hungerford Town", "Paulton Rovers"]
    pauses = []
    rows = {r["club"]: r for r in pw.prewarm(clubs, sources=tmp_path, delay=0.75, sleep=pauses.append)}
    assert {c: (r["status"], r["kit_source"], r["home"], r["away"], r["crest"]) for c, r in rows.items()} == {
        "Dorchester Town": ("RESOLVED", "club records", "#000000", "#0000EE", True),
        "Weymouth": ("RESOLVED", "crest", "#5A8CFA", "#FFFFFF", True),
        "Bideford": ("NOT_FOUND", "default", "#CC2222", "#FFFFFF", False),
        "Exmouth Town": ("OFFLINE", "default", "#CC2222", "#FFFFFF", False),
        "Hungerford Town": ("OFFLINE", "default", "#CC2222", "#FFFFFF", False),
        "Paulton Rovers": ("RESOLVED", "default", "#CC2222", "#FFFFFF", False),     # page found, crest was not an image
    }
    assert rows["Dorchester Town"]["page"] == "Dorchester Town F.C." and rows["Bideford"]["page"] == ""
    assert pauses == [0.75] * 5                                   # a pause between each of the 6 lookups
    assert sorted(p.parent.name for p in tmp_path.glob("*/meta.json")) == ["bideford", "dorchester_town", "paulton_rovers", "weymouth"]

    pauses.clear()
    second = {r["club"]: r["status"] for r in pw.prewarm(clubs, sources=tmp_path, delay=0.75, sleep=pauses.append)}
    assert second == {"Dorchester Town": "CACHED", "Weymouth": "CACHED", "Bideford": "CACHED",
                      "Exmouth Town": "OFFLINE", "Hungerford Town": "OFFLINE", "Paulton Rovers": "CACHED"}
    assert pauses == [0.75]                                       # only the two retried lookups are spaced


def test_prewarm_force_refreshes_and_keeps_the_cache_when_offline(tmp_path, monkeypatch):
    _league_wikipedia(monkeypatch)
    pw.prewarm(["Dorchester Town"], sources=tmp_path, delay=0)
    assert pw.prewarm(["Dorchester Town"], sources=tmp_path, force=True, delay=0)[0]["status"] == "RESOLVED"

    def down(url):
        raise TimeoutError("timed out")

    monkeypatch.setattr(ca, "_http_json", down)
    row = pw.prewarm(["Dorchester Town"], sources=tmp_path, force=True, delay=0)[0]
    assert row["status"] == "OFFLINE"
    assert pw.prewarm(["Dorchester Town"], sources=tmp_path, delay=0)[0]["status"] == "CACHED"   # old meta.json survives


def test_prewarm_cli_covers_all_21_opponents(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ca, "_http_json", lambda url: {"pages": []})
    assert pw.opponents() == [c for c in lr.SOUTHERN_LEAGUE_DIV_ONE_SOUTH if c != "Tiverton Town"]
    assert pw.main(["--sources", str(tmp_path), "--delay", "0"]) == 0
    out = capsys.readouterr().out
    assert "Pre-warming 21" in out and "Tiverton Town" in out and out.count("NOT_FOUND ") == 21
    assert "CACHED: 0  RESOLVED: 0  NOT_FOUND: 21  OFFLINE: 0  (of 21)" in out
    assert not re.search(r"^Tiverton Town\s", out, re.M)          # named only in the excluded note
    with pytest.raises(SystemExit) as exc:
        pw.main(["--delay", "-1"])
    assert exc.value.code == 2


def test_picking_a_league_club_resolves_crest_and_kit_without_typing(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    for attr, sub in (("UI_STATE", "ui"), ("SOURCES", "sources"), ("STAGING_VIDEOS", "staging")):
        monkeypatch.setattr(vc, attr, tmp_path / sub)
    (tmp_path / "staging").mkdir()
    lookups = []

    def fake_json(url):
        lookups.append(url)
        return {"source": WIKITEXT} if "/v1/page/" in url else SUMMARY if "summary" in url else SEARCH

    monkeypatch.setattr(ca, "_http_json", fake_json)
    monkeypatch.setattr(ca, "_http_bytes", lambda url: _crest())

    at = AppTest.from_function(_tab_script, default_timeout=60)
    at.run()
    assert not at.exception
    pick = at.selectbox(key="vcc_opponent_pick")
    assert pick.value == "-- Select Opponent --" and "Tiverton Town" not in pick.options
    assert "vcc_opponent" not in [w.key for w in at.text_input]          # no free-text box until Other / Custom
    assert at.session_state["vision_opponent_name"] == "" and lookups == []

    pick.set_value("Dorchester Town").run()
    assert not at.exception
    assert at.session_state["vision_opponent_name"] == "Dorchester Town"
    assert at.session_state["vision_opponent_kit"] == "#000000"          # home shirt from club records
    text = "\n".join(_all_text(at.main))
    assert "DORCHESTER TOWN F.C." in text.upper() and "kit colours from club records" in text
    assert len(re.findall(r'url: "[^"]*\.png"', text)) == 1              # the crest
    assert [t for t in BANNED if re.search(rf"\b{t}\b", text, re.I)] == []
    meta = json.loads((tmp_path / "sources" / "dorchester_town" / "meta.json").read_text(encoding="utf-8"))
    assert (meta["home_kit"], meta["away_kit"]) == ("#000000", "#0000EE")

    at.radio(key="vcc_kit_choice").set_value("Away").run()
    assert at.session_state["vision_opponent_kit"] == "#0000EE"

    at.selectbox(key="vcc_opponent_pick").set_value("Other / Custom...").run()
    assert "vcc_opponent" in [w.key for w in at.text_input] and at.session_state["vision_opponent_name"] == ""
    at.text_input(key="vcc_opponent").set_value("Taunton Town").run()
    assert not at.exception and at.session_state["vision_opponent_name"] == "Taunton Town"
    text = "\n".join(_all_text(at.main))
    assert "Club not found online" in text                                 # mocked search only knows Dorchester
    assert [t for t in BANNED if re.search(rf"\b{t}\b", text, re.I)] == []
