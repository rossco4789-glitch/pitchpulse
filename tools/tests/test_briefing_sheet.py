"""
tools/tests/test_briefing_sheet.py
One-page printable match-day sheet (cv/briefing_sheet.py) and the print theme of the pitch graphic.

Run: python -m pytest tools/tests/test_briefing_sheet.py -v
"""

import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cv import briefing_sheet as bs  # noqa: E402
from cv import coach_brief as cb  # noqa: E402
from cv import highlight_dossier as hd  # noqa: E402
from cv import vision_center as vc  # noqa: E402

BANNED = ("homography", "bounding box", "yolo", "pipeline", "acceptance rate", "silhouette", "iqr", "median",
          "settled frames", "pid", "subprocess", "kaggle", "dispatch_id", "dispatch id", "manifest", "hex")
OOP = {
    "block_height_m": {"value": 9.88, "iqr": [5.18, 24.09], "n": 171, "reason": None},
    "compactness_depth_m": {"value": 21.28, "iqr": [14.96, 32.92], "n": 171, "reason": None},
    "compactness_width_m": {"value": 20.6, "iqr": [13.81, 29.24], "n": 171, "reason": None},
    "line_of_engagement_m": {"value": 21.02, "iqr": [18.38, 49.71], "n": 171, "reason": None},
}
SUMMARY = {"opponent": "Supporting Charities", "date": "2026-09-15", "acceptance": 0.427, "settled_frames": 171, "oop": OOP}


def _banned(text: str) -> list[str]:
    return [t for t in BANNED if re.search(rf"\b{t}\b", re.sub(r"base64,[A-Za-z0-9+/=]+", "", text), re.I)]


def test_full_sheet_holds_briefing_phases_and_print_pitch_on_one_block():
    brief = cb.build_brief(SUMMARY)
    sheet = bs.full_sheet(brief, vc.shape_geometry(OOP), generated="15 Sep 2026, 19:30")
    assert sheet.startswith('<div class="vcc-sheet">') and "\n" not in sheet          # Markdown-safe single block
    for expected in ("Supporting Charities", "DEEP COMPACT LOW BLOCK", "Team shape: Narrow · Compact · Sits deep",
                     "Defensive Line Depth", "11 yards from goal", "1. In Possession", "2. Breaking Them Down",
                     "3. Transition &amp; Rest Defence", "Data Confidence: Low", "Prepared 15 Sep 2026, 19:30",
                     "PRESS TRIGGER · 23 YDS", "SPACE ON FLANKS"):
        assert expected in sheet, expected
    assert sheet.count("<li>") == sum(len(p["points"]) for p in brief["phases"])
    assert 'id="vccp-grass"' in sheet and 'id="vcc-grass"' not in sheet           # print ids never clash with the screen pitch
    assert "vccp-turf" not in sheet.split("</defs>")[1]                            # no noise texture on paper
    assert _banned(sheet) == []


def test_unavailable_brief_prints_the_coach_action_without_a_pitch():
    oop = {k: {"value": None, "reason": "opponent_kit_not_given"} for k in cb.OOP_KEYS}
    sheet = bs.full_sheet(cb.build_brief({**SUMMARY, "oop": oop}), None)
    assert "shirt colour wasn&#x27;t set" in sheet and "<svg" not in sheet and "sh-note" in sheet


def test_dossier_sheet_prints_quick_read_and_three_sections():
    moments = [hd.new_event("goal_scored", {"channel": "Left wing", "action": "Cross", "arrival": "Far post",
                                            "counter_trigger": "Not a counter"}),
               hd.new_event("corner_against", {"marking": "Zonal", "outcome": "Second ball lost"})]
    sheet = bs.dossier_sheet(hd.build_dossier("Weymouth", moments), generated="today")
    for expected in ("HIGHLIGHT REEL", "Primary threat", "Attacks down their left wing", "1. Attacking Patterns",
                     "2. Defensive Flaws", "3. Dead-Ball Intelligence", 'class="lever"', 'class="empty"'):
        assert expected in sheet, expected
    assert "\n" not in sheet and "<svg" not in sheet and _banned(sheet) == []


def test_crest_is_embedded_for_offline_printing(tmp_path):
    crest = tmp_path / "badge.png"
    Image.new("RGBA", (24, 24), (0, 0, 0, 255)).save(crest)
    uri = bs.crest_data_uri(crest)
    assert uri.startswith("data:image/png;base64,")
    assert f'<img class="sh-crest" src="{uri}"' in bs.full_sheet(cb.build_brief(SUMMARY), None, crest_uri=uri)
    assert bs.crest_data_uri(tmp_path / "missing.png") is None and bs.crest_data_uri(None) is None
    crest.write_bytes(b"0" * (bs.CREST_MAX_B + 1))
    assert bs.crest_data_uri(crest) is None


def test_print_stylesheet_only_takes_over_when_printing_is_requested():
    css = bs.PRINT_CSS
    assert ".vcc-sheet { display:none;" in css                                       # hidden on screen
    printing = css.split("@media print", 1)[1]
    assert "html.vcc-printing body *:not(.vcc-sheet):not(.vcc-sheet *):not(:has(.vcc-sheet))" in printing
    assert "display:none !important" in printing and "background:#FFFFFF !important" in printing
    assert "break-inside:avoid" in css and _banned(css) == []


def test_dark_pitch_is_unchanged_and_print_pitch_is_low_ink():
    geo = vc.shape_geometry(OOP)
    dark, light = cb.pitch_svg(geo), cb.pitch_svg(geo, theme="print")
    assert 'fill="#1B4A21"' in dark and 'filter="url(#vcc-turf)"' in dark and 'fill="url(#vcc-net)"' in dark
    assert 'fill="#FFFFFF"/>' in light and 'stroke="#2E6B33"' in light and 'filter="url(#vccp-turf)"' not in light
    assert light.count('class="def-line" x1="9.88"') == 1 and "DEFENSIVE LINE · 11 YDS" in light
