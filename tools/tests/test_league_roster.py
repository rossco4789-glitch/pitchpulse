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
