"""
tools/tests/test_coach_brief.py
Coach-facing Opposition Analysis: coaching dictionary, phase instructions, pitch graphic, and a full offline
render of app.py tab 6 (Streamlit AppTest) that fails if any engineering term reaches the screen.

Run: python -m pytest tools/tests/test_coach_brief.py -v
"""

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cv import coach_brief as cb  # noqa: E402
from cv import vision_center as vc  # noqa: E402

BANNED = ("homography", "bounding box", "yolo", "pipeline", "acceptance rate", "silhouette", "iqr", "median",
          "settled frames", "pid", "subprocess", "kaggle", "dispatch_id", "dispatch id", "manifest", "hex",
          "exhibits a tendency", "it is important to note", "analyzing the data reveals")

# Collected Veo run d9fa85f6 (15 Sep 2026)
OOP = {
    "block_height_m": {"value": 9.88, "iqr": [5.18, 24.09], "n": 171, "reason": None},
    "compactness_depth_m": {"value": 21.28, "iqr": [14.96, 32.92], "n": 171, "reason": None},
    "compactness_width_m": {"value": 20.6, "iqr": [13.81, 29.24], "n": 171, "reason": None},
    "line_of_engagement_m": {"value": 21.02, "iqr": [18.38, 49.71], "n": 171, "reason": None},
}
SUMMARY = {"opponent": "Supporting Charities", "date": "2026-09-15", "acceptance": 0.427, "settled_frames": 171, "oop": OOP}


def _oop(block, depth, width, loe):
    return {"block_height_m": {"value": block}, "compactness_depth_m": {"value": depth},
            "compactness_width_m": {"value": width}, "line_of_engagement_m": {"value": loe}}


def banned_terms(text: str) -> list[str]:
    return [t for t in BANNED if re.search(rf"\b{re.escape(t)}\b", text, re.I)]


# ── Coaching dictionary ──────────────────────────────────────────────────────

def test_measurements_translate_into_touchline_terms():
    brief = cb.build_brief(SUMMARY)
    assert (brief["badge"], brief["shape"]) == ("DEEP COMPACT LOW BLOCK", "Narrow · Compact · Sits deep")
    assert [(m["label"], m["value"], m["detail"]) for m in brief["metrics"]] == [
        ("Defensive Line Depth", "11 yards from goal", "Inside the 18-yard box"),
        ("Press Trigger / Line of Engagement", "Engage at 23 yards", "Own defensive third"),
        ("Defensive Width", "Narrow central block", "21 m corridor"),
        ("Team Length / Spacing", "Compact", "21 m front-to-back"),
    ]
    assert brief["date"] == "15 Sep 2026" and brief["confidence"]["level"] == "Low"


def test_three_phases_give_numbered_instructions():
    p1, p2, p3 = cb.build_brief(SUMMARY)["phases"]
    assert [(p["title"], p["subtitle"]) for p in (p1, p2, p3)] == [
        ("In Possession", "How We Build"), ("Breaking Them Down", "Where the Space Is"), ("Transition & Rest Defence", "Where We Hold")]
    assert p1["points"][0] == "They start pressing 23 yards from their own goal, 92 yards from ours."
    assert "leaving 26 yards free on each flank" in p2["points"][0] and "Switch play early" in p2["points"][1]
    assert "11 yards out: cut the ball back" in p2["points"][2]
    assert p3["points"][1] == ("Rest defence: both centre-backs and the holding midfielder hold 39 yards from their goal, "
                               "18 yards inside their half.")


def test_high_press_and_wide_block_change_the_instructions():
    brief = cb.build_brief({**SUMMARY, "oop": _oop(block=34.0, depth=30.0, width=50.0, loe=60.0)})
    p1, p2, p3 = brief["phases"]
    assert brief["badge"] == "HIGH PRESSING BLOCK" and brief["shape"] == "Wide · Spaced · Steps high"
    assert p1["points"][0] == "They press into our half, 49 yards from our goal."
    assert "flanks stay closed" in p2["points"][0] and "over the top" in p2["points"][2]
    assert "on the halfway line" in p3["points"][1] and len(p3["points"]) == 3


@pytest.mark.parametrize("mapped,moments,level", [(0.8, 400, "High"), (0.6, 150, "Medium"), (0.8, 90, "Low"), (0.427, 171, "Low")])
def test_confidence_tiers(mapped, moments, level):
    assert cb.confidence({"acceptance": mapped, "settled_frames": moments})["level"] == level


@pytest.mark.parametrize("reason,phrase", [
    ("opponent_kit_not_given", "shirt colour wasn't set"),
    ("team_split_ambiguous (silhouette 0.43)", "exact shirt colour"),
    ("insufficient_coverage (n=12, windows=1)", "Wide-angle full-match video"),
    ("highlight_footage_invalid_for_shape", "Use the full-match video"),
])
def test_missing_shape_tells_the_coach_what_to_do(reason, phrase):
    oop = {k: {"value": None, "reason": reason} for k in cb.OOP_KEYS}
    brief = cb.build_brief({**SUMMARY, "oop": oop})
    assert not brief["available"] and phrase in brief["unavailable"] and brief["phases"] == []


def test_kit_verdicts():
    assert [cb.kit_verdict(s)[1].split(":")[0] for s in (0.662, 0.52, 0.43, None)] == [
        "Kits clearly different", "Kits close", "Kits too similar", "Not checked"]


# ── Pitch graphic ────────────────────────────────────────────────────────────

def test_pitch_draws_defensive_line_press_trigger_corridor_and_flank_space():
    svg = cb.pitch_svg(vc.shape_geometry(OOP))
    assert "\n" not in svg   # Markdown would turn indented lines into code blocks
    assert '<line class="def-line" x1="9.88"' in svg and 'stroke="#388BFD"' in svg
    assert re.search(r'<line class="press-line" x1="21.02"[^>]*stroke-dasharray', svg)
    assert '<rect class="corridor" x="0.00" y="23.70" width="21.02" height="20.60"' in svg
    assert svg.count('class="flank"') == 2 and svg.count("SPACE ON FLANKS") == 2 and "Switch Play Early" in svg
    assert "DEFENSIVE LINE · 11 YDS" in svg and "PRESS TRIGGER · 23 YDS" in svg
    assert svg.count('fill="url(#vcc-net)"') == 2 and "vcc-mow" in svg   # goals and mown stripes

    wide = cb.pitch_svg(vc.shape_geometry(_oop(block=25.0, depth=30.0, width=50.0, loe=45.0)))
    assert 'class="flank"' not in wide and 'class="corridor"' in wide
    assert "def-line" not in cb.pitch_svg(None)


def test_all_coach_text_is_free_of_engineering_terms():
    scenarios = [SUMMARY, {**SUMMARY, "oop": _oop(34.0, 30.0, 50.0, 60.0)}, {**SUMMARY, "oop": _oop(25.0, 40.0, 35.0, 45.0)},
                 {**SUMMARY, "oop": {k: {"value": None, "reason": "team_split_ambiguous (silhouette 0.43)"} for k in cb.OOP_KEYS}}]
    text = " ".join(cb.brief_text(cb.build_brief(s)) for s in scenarios) + cb.pitch_svg(vc.shape_geometry(OOP))
    assert banned_terms(text) == []


# ── Full tab render ──────────────────────────────────────────────────────────

def _tab_script():
    from cv.vision_center_ui import render
    render()


def _all_text(node) -> list[str]:
    out = [str(node.proto)] if getattr(node, "proto", None) is not None else []
    for child in getattr(node, "children", {}).values():
        out += _all_text(child)
    return out


@pytest.fixture
def tab_env(tmp_path, monkeypatch):
    for name, sub in (("UI_STATE", "ui"), ("SOURCES", "sources"), ("STAGING_VIDEOS", "staging")):
        monkeypatch.setattr(vc, name, tmp_path / sub)
    for fn in ("start_collect", "start_dispatch", "start_preflight"):
        monkeypatch.setattr(vc, fn, lambda *a, **k: None)   # never launch real work from a render test
    (tmp_path / "staging").mkdir()
    writer = cv2.VideoWriter(str(tmp_path / "staging" / "tiverton_v_supporting_charities.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(20):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()

    done = tmp_path / "sources" / "supporting_charities"
    done.mkdir(parents=True)
    (done / "vision_metrics.json").write_text(json.dumps({
        "footage": "full_wide", "frames": {"sampled": 13868, "accepted": 5927}, "moments": {"out_of_possession": OOP},
        "team_split": {"silhouette": 0.607, "settled_frames": 171, "players": 30872}}), encoding="utf-8")
    (done / "vision_job.json").write_text(json.dumps({"state": "COMPLETE", "collected_at": "2026-09-15T14:54:41+00:00",
                                                     "kernel_ref": "tivvy/pitchpulse-vision-x", "dispatch_id": "d9fa85f6"}), encoding="utf-8")
    running = tmp_path / "sources" / "dorchester_town"
    running.mkdir(parents=True)
    (running / "vision_job.json").write_text(json.dumps({"state": "RUNNING", "dispatched_at": "2026-09-15T14:30:28+00:00",
                                                        "kernel_ref": "tivvy/pitchpulse-vision-y"}), encoding="utf-8")
    vc._write_json(vc.preflight_result_path("supporting_charities"), {
        "status": "ok", "players": 41, "silhouette": 0.662, "kit": "#CC2222", "frames": 5,
        "video": "tiverton_v_supporting_charities.mp4",
        "opponent": {"hex": "#692C2A", "players": 11, "distance": 61.9}, "other": {"hex": "#2F3031", "players": 30, "distance": 105.5}})
    return tmp_path


def test_tab_renders_a_coach_briefing_with_no_engineering_terms(tab_env):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_tab_script, default_timeout=60)
    at.run()
    assert not at.exception
    at.text_input(key="vcc_opponent").set_value("Supporting Charities").run()
    assert not at.exception

    text = "\n".join(_all_text(at.main))
    assert banned_terms(text) == []
    for expected in ("DEEP COMPACT LOW BLOCK", "Match Setup", "Analyzing match footage", "Kits clearly different",
                     "In Possession", "Breaking Them Down", "Transition &amp; Rest Defence", "SPACE ON FLANKS",
                     "Data Confidence: Low"):
        assert expected in text, expected
