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
from cv import highlight_dossier as hd  # noqa: E402
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
    # proto text escapes quotes inside HTML bodies; unescape so 'class="..."' checks can match (and fail) for real
    out = [str(node.proto).replace('\\"', '"')] if getattr(node, "proto", None) is not None else []
    for child in getattr(node, "children", {}).values():
        out += _all_text(child)
    return out


@pytest.fixture
def tab_env(tmp_path, monkeypatch):
    for name, sub in (("UI_STATE", "ui"), ("SOURCES", "sources"), ("STAGING_VIDEOS", "staging")):
        monkeypatch.setattr(vc, name, tmp_path / sub)
    for fn in ("start_collect", "start_dispatch", "start_preflight"):
        monkeypatch.setattr(vc, fn, lambda *a, **k: None)   # never launch real work from a render test

    from urllib.error import URLError
    from cv import club_assets as ca

    def offline(url):
        raise URLError("render tests stay offline")

    monkeypatch.setattr(ca, "_http_json", offline)           # club lookups fall back to the default kit
    monkeypatch.setattr(ca, "_http_bytes", offline)
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
    at.selectbox(key="vcc_opponent_pick").set_value("Other / Custom...").run()   # not a league club
    at.text_input(key="vcc_opponent").set_value("Supporting Charities").run()
    assert not at.exception

    text = "\n".join(_all_text(at.main))
    assert banned_terms(text) == []
    for expected in ("DEEP COMPACT LOW BLOCK", "Match Setup", "Analyzing match footage", "Kits clearly different",
                     "In Possession", "Breaking Them Down", "Transition &amp; Rest Defence", "SPACE ON FLANKS",
                     "Data Confidence: Low"):
        assert expected in text, expected

    assert 'class="vcc-sheet"' in text and "sh-metrics" in text and "vccp-grass" in text   # print sheet, hidden on screen
    at.button(key="vcc_print").click().run()
    assert not at.exception
    printed = "\n".join(_all_text(at.main))
    assert "vcc-printing" in printed and "A4 landscape" in printed and banned_terms(printed) == []


# ── Highlight & Tendency Dossier ─────────────────────────────────────────────

def _moments():
    def attack(kind, channel, action, arrival, trigger="Not a counter"):
        return hd.new_event(kind, {"channel": channel, "action": action, "arrival": arrival, "counter_trigger": trigger})

    return [
        attack("goal_scored", "Left wing", "Cross", "Far post", "Regain in midfield"),
        attack("goal_scored", "Left wing", "Cross", "Far post"),
        attack("chance_created", "Left wing", "Cross", "Near post", "Regain in midfield"),
        attack("goal_scored", "Right wing", "Cut-back", "Penalty spot", "Keeper quick release"),
        attack("chance_created", "Central", "Shot from distance", "Outside the box"),
        hd.new_event("goal_conceded", {"flaw": "Isolated 1v1", "side": "Their left", "player": "3"}),
        hd.new_event("chance_conceded", {"flaw": "Isolated 1v1", "side": "Their left", "player": "3"}),
        hd.new_event("chance_conceded", {"flaw": "Space behind full-back", "side": "Their right"}),
        hd.new_event("goal_conceded", {"flaw": "Second ball after clearance", "side": "Central"}),
        hd.new_event("corner_for", {"delivery": "Inswinger", "target": "Near post", "player": "5", "outcome": "Goal"}),
        hd.new_event("corner_for", {"delivery": "Inswinger", "target": "Near post", "player": "5", "outcome": "Shot"}),
        hd.new_event("corner_for", {"delivery": "Outswinger", "target": "Far post", "outcome": "Won by defence"}),
        hd.new_event("corner_against", {"marking": "Zonal", "outcome": "Cleared"}),
        hd.new_event("corner_against", {"marking": "Zonal", "outcome": "Second ball lost"}),
        hd.new_event("corner_against", {"marking": "Man-to-man", "outcome": "Shot conceded"}),
        hd.new_event("free_kick", {"player": "10", "foot": "Right foot", "outcome": "On target"}, minute=63),
        hd.new_event("free_kick", {"player": "10", "foot": "Right foot", "outcome": "Goal"}),
    ]


def test_dossier_quick_read_and_confidence():
    d = hd.build_dossier("Weymouth", _moments())
    assert (d["footage"], d["moments"], d["confidence"]["level"]) == ("Highlight Reel", 17, "Medium")
    assert d["quick_read"] == {"threat": "Attacks down their left wing (3 of 5 goals and chances)",
                               "vulnerability": "Defenders isolated 1v1 (2 of 5 conceded moments)"}
    assert [s["title"] for s in d["sections"]] == ["Attacking Patterns", "Defensive Flaws", "Dead-Ball Intelligence"]


def test_dossier_sections_count_moments_and_end_on_an_instruction():
    attacking, defending, dead_ball = ({i["label"]: i for i in s["items"]} for s in hd.build_dossier("Weymouth", _moments())["sections"])
    assert attacking["Creation channel"] == {
        "label": "Creation channel", "read": "3 of 5 goals and chances came down their left wing, most often from crosses (3).",
        "lever": "Our right-back and right midfielder double up on their left wing; force play inside."}
    assert attacking["Box arrival runs"]["read"] == "2 of 4 finishes in the box arrived at the far post."
    assert attacking["Counter-attack triggers"]["read"] == "3 of 5 came on the counter, most often after a regain in midfield (2)."

    assert defending["Isolated 1v1 matchups"]["read"] == "Isolated 1v1 in 2 of 4 goals and chances conceded, No. 3 most often (2)."
    assert defending["Isolated 1v1 matchups"]["lever"] == "Get our right winger 1v1 against No. 3 early."
    assert defending["Space behind full-backs"]["lever"] == "Our left winger spins in behind as soon as their right-back steps up."
    assert defending["Second balls on box clearances"]["read"] == "Lost the second ball after a clearance 2 times: 1 in open play, 1 from corners."

    assert dead_ball["Attacking corners"]["read"] == "Inswingers on 2 of 3, aimed at the near post (2); No. 5 attacks it (2). 1 goal, 1 shot."
    assert dead_ball["Attacking corners"]["lever"].endswith("Our best header marks No. 5.")
    assert dead_ball["Defending corners"]["read"] == "Zonal marking on 2 of 3 corners; 0 goals and 1 shot conceded."
    assert dead_ball["Direct free kicks"]["read"] == "No. 10 took 2 of 2, right foot; 2 on target, 1 scored."
    for section in (attacking, defending, dead_ball):
        assert all(item["lever"] for item in section.values())


def test_single_moment_reads_in_the_singular():
    moment = hd.new_event("goal_scored", {"channel": "Left wing", "action": "Cross", "arrival": "Near post", "counter_trigger": "Not a counter"})
    attacking = {i["label"]: i["read"] for i in hd.build_dossier("Weymouth", [moment])["sections"][0]["items"]}
    assert hd.build_dossier("Weymouth", [moment])["quick_read"]["threat"] == "Attacks down their left wing (1 of 1 goal or chance)"
    assert attacking["Creation channel"].startswith("1 of 1 goal or chance came down their left wing")
    assert attacking["Counter-attack triggers"] == "The one goal or chance logged didn't come on the counter."
    conceded = hd.new_event("chance_conceded", {"flaw": "Beaten in the air", "side": "Central"})
    defending = {i["label"]: i["read"] for i in hd.build_dossier("Weymouth", [conceded])["sections"][1]["items"]}
    assert defending["Isolated 1v1 matchups"] == "No isolated 1v1 defending in 1 goal or chance conceded."
    assert defending["Space behind full-backs"] == "No goal or chance conceded came in behind the full-backs."


def test_empty_dossier_prompts_logging():
    d = hd.build_dossier("Weymouth", [])
    assert d["confidence"]["text"] == "Data Confidence: Low · 0 moments logged; treat as leads, not patterns"
    assert d["quick_read"]["threat"] == "Log goals and chances to find their main threat"
    items = [i for s in d["sections"] for i in s["items"]]
    assert len(items) == 9 and all("logged yet" in i["read"] and not i["lever"] for i in items)


def test_moment_validation_and_storage(tmp_path):
    with pytest.raises(ValueError, match="unknown moment type"):
        hd.new_event("throw_in", {})
    with pytest.raises(ValueError, match="Delivery: 'Banana'"):
        hd.new_event("corner_for", {"delivery": "Banana", "target": "Near post", "outcome": "Goal"})
    with pytest.raises(ValueError, match="minute"):
        hd.new_event("free_kick", {"foot": "Left foot", "outcome": "Goal"}, minute=200)
    assert len(hd.new_event("free_kick", {"player": "A" * 40, "foot": "Left foot", "outcome": "Goal"})["fields"]["player"]) == hd.TEXT_MAX

    path = hd.events_path(tmp_path, "weymouth")
    first = hd.add_event(path, "corner_against", {"marking": "Hybrid", "outcome": "Cleared"}, source="v Yeovil")
    hd.add_event(path, "free_kick", {"player": "10", "foot": "Left foot", "outcome": "Goal"}, minute=81)
    assert [e["kind"] for e in hd.load_events(path)] == ["corner_against", "free_kick"]
    assert hd.logged_opponents(tmp_path) == ["weymouth"]
    assert hd.delete_event(path, first["id"]) and not hd.delete_event(path, "missing")
    assert hd.describe_event(hd.load_events(path)[0]) == "Direct free kick · No. 10 · Left foot · Goal · 81'"
    hd.delete_event(path, hd.load_events(path)[0]["id"])
    assert hd.logged_opponents(tmp_path) == []   # an emptied log is not a dossier


def test_dossier_text_is_free_of_engineering_terms():
    moments = _moments()
    text = " ".join([hd.dossier_text(hd.build_dossier("Weymouth", moments)), hd.dossier_text(hd.build_dossier("Weymouth", [])),
                     *(hd.describe_event(e) for e in moments)])
    variants = [hd.new_event("goal_scored", {"channel": c, "action": a, "arrival": "Outside the box", "counter_trigger": t})
                for c in hd.CHANNELS for a in hd.ACTIONS for t in hd.TRIGGERS]
    variants += [hd.new_event("chance_conceded", {"flaw": f, "side": s}) for f in hd.FLAWS for s in hd.SIDES]
    variants += [hd.new_event("corner_for", {"delivery": d, "target": z, "outcome": "Shot"}) for d in hd.DELIVERIES for z in hd.CORNER_ZONE]
    variants += [hd.new_event("corner_against", {"marking": m, "outcome": o}) for m in hd.MARKING for o in hd.CORNER_AGST]
    for group in (variants[i:i + 7] for i in range(0, len(variants), 7)):
        text += " " + hd.dossier_text(hd.build_dossier("Weymouth", group))
    assert banned_terms(text) == []


def test_highlight_mode_hides_the_pitch_and_logs_moments(tab_env):
    from streamlit.testing.v1 import AppTest

    path = hd.events_path(vc.SOURCES, "supporting_charities")
    for e in _moments():
        hd.add_event(path, e["kind"], e["fields"], minute=e["minute"], source="v Weymouth")

    at = AppTest.from_function(_tab_script, default_timeout=60)
    at.run()
    at.selectbox(key="vcc_opponent_pick").set_value("Other / Custom...").run()
    at.text_input(key="vcc_opponent").set_value("Supporting Charities").run()
    at.radio(key="vcc_mode").set_value("Opposition Scouting").run()
    assert not at.exception

    text = "\n".join(_all_text(at.main))
    assert banned_terms(text) == []
    assert 'class="vcc-pitch"' not in text and "Kit contrast check" not in text   # no team shape, no video analysis
    for expected in ("HIGHLIGHT REEL", "Attacks down their left wing", "Attacking Patterns", "Defensive Flaws",
                     "Dead-Ball Intelligence", "Log a key moment", "Logged moments · 17", "Data Confidence: Medium"):
        assert expected in text, expected
    assert 'class="vcc-sheet"' in text and "sh-quick" in text and "vccp-grass" not in text    # dossier sheet has no pitch

    at.selectbox(key="vcc_hl_kind").set_value("free_kick").run()
    at.text_input(key="vcc_hl_free_kick_player").set_value("7").run()
    at.button(key="vcc_hl_add").click().run()
    assert not at.exception
    assert hd.load_events(path)[-1]["fields"] == {"player": "7", "foot": "Right foot", "outcome": "Goal"}
    assert "Logged moments · 18" in "\n".join(_all_text(at.main))


REELS = [
    {"match_date": "Sat 12 Sep", "fixture_label": "Sholing v Plymouth Parkway", "score": "1-3",
     "title": "Plymouth Parkway 3 – 1 Sholing | Match Highlights", "url": "https://www.youtube.com/watch?v=parkway0001",
     "channel": "Sholing Football Club"},
    {"match_date": "Sat 22 Aug", "fixture_label": "Sholing v Chertsey Town", "score": "2-4",
     "title": "Highlights - Sholing 2-4 Chertsey Town", "url": "https://www.youtube.com/watch?v=VqinZ1Dw3eo",
     "channel": "Sholing Football Club"},
]


def test_next_fixture_opens_opposition_scouting_with_the_match_reel_selector(tab_env, monkeypatch):
    from types import SimpleNamespace
    from streamlit.testing.v1 import AppTest
    from cv import vision_center_ui as ui

    fixture = {"opponent": "Sholing", "home_away": "Home", "venue": "The Slee Blackwell Solicitors Stadium",
               "date": "2026-09-19", "date_str": "Sat 19 Sep 2026", "kickoff": "15:00", "competition": "FA Cup",
               "slug": "sholing"}
    monkeypatch.setattr(ui, "_next_fixture", lambda: fixture)
    monkeypatch.setattr(ui, "_fixture_agent", lambda: SimpleNamespace(scaffold_preview_stub=lambda fx: "previews/sholing.md"))
    links = vc.SOURCES / "sholing" / "video_links.json"
    links.parent.mkdir(parents=True)
    links.write_text(json.dumps(REELS), encoding="utf-8")

    at = AppTest.from_function(_tab_script, default_timeout=60)
    at.run()
    assert at.radio(key="vcc_mode").value == "Post-Match Performance"
    at.button(key="vcc_next_fixture").click().run()
    assert not at.exception
    assert at.radio(key="vcc_mode").value == "Opposition Scouting"
    assert at.text_input(key="vcc_opponent").value == "Sholing"

    text = "\n".join(_all_text(at.main))
    assert banned_terms(text) == []
    assert "Kit contrast check" not in text and "Select Match Reel (Last 5 Fixtures)" in text
    assert "https://www.youtube.com/embed/parkway0001" in text                  # newest fixture plays first
    assert text.index("embed/parkway0001") < text.index("Log a key moment")     # the reel sits above the logger
    assert "2 of the last 5 fixtures have a reel." in text
    assert at.text_input(key="vcc_reel").value == "Sholing v Plymouth Parkway 1-3 · Sat 12 Sep"

    at.selectbox(key="vcc_reel_pick").set_value(1).run()
    assert not at.exception
    text = "\n".join(_all_text(at.main))
    assert "embed/VqinZ1Dw3eo" in text and "embed/parkway0001" not in text
    assert at.text_input(key="vcc_reel").value == "Sholing v Chertsey Town 2-4 · Sat 22 Aug"

    at.selectbox(key="vcc_hl_kind").set_value("goal_conceded").run()
    at.button(key="vcc_hl_add").click().run()
    assert not at.exception
    assert hd.load_events(hd.events_path(vc.SOURCES, "sholing"))[-1]["source"] == "Sholing v Chertsey Town 2-4 · Sat 22 Aug"


# ── Patterns across match reels ──────────────────────────────────────────────

def _evidence():
    def match(squad, goals):
        return {"lineup_published": True, "starters": [{"name": n} for n in squad], "bench": [],
                "events": [{"minute": m, "kind": "goal", "text": f"{who} scores"} for m, who in goals]}
    squad = ["Harry Taylor", "Jake Mccarthy"]
    return {"fixtures": [
        {"played": True, "match": match(squad, [("29", "Harry Taylor"), ("51", "Daniel Berry"), ("86", "Bradley Wilson"), ("90+1", "Nathan Minhas")])},
        {"played": True, "match": match(squad, [("45+2", "Jake Mccarthy"), ("10", "Adam Liddle"), ("49", "Ruben Bartlett-Antwi")])},
        {"played": True, "match": {"lineup_published": False}},
        {"played": False},
    ]}


def _reel_moments():
    a, b = "Sholing v Plymouth Parkway 1-3 · Sat 12 Sep", "Sholing v Chertsey Town 2-4 · Sat 22 Aug"

    def attack(channel, source):
        return hd.new_event("goal_scored", {"channel": channel, "action": "Cross", "arrival": "Far post",
                                            "counter_trigger": "Not a counter"}, source=source)

    def conceded(flaw, side, minute, source):
        return hd.new_event("goal_conceded", {"flaw": flaw, "side": side}, minute=minute, source=source)

    return [attack("Left wing", a), attack("Left half-space", b), attack("Central", b), attack("Right wing", a),
            attack("Left wing", b), conceded("Beaten in the air", "Their right", 67, a),
            conceded("Beaten in the air", "Their right", 86, b), conceded("Cut-back not tracked", "Their left", 51, b),
            conceded("Isolated 1v1", "Central", 28, a)]


def test_goal_timing_counts_only_goals_against_them():
    assert hd.goal_timing(_evidence()) == {"sheets": 2, "conceded": 5, "first_half": 1, "second_half": 4, "after_75": 2}
    assert hd.goal_timing(None) is None and hd.goal_timing({"fixtures": [{"played": True}]}) is None


def test_patterns_pool_moments_across_match_reels_and_evidence_timing():
    patterns = hd.build_dossier("Sholing", _reel_moments(), _evidence())["patterns"]
    corridors, entries, timing = patterns["items"]
    assert patterns["title"] == "Patterns Across 2 Match Reels"
    assert corridors == {"label": "Attacking corridors",
                         "read": "Left 60% · Central 20% · Right 20% of 5 goals and chances across 2 match reels.",
                         "lever": "Our right-back and right midfielder double up on their left side; force play inside."}
    assert entries["read"] == ("3 of 4 goals and chances conceded (75%) came from box entries, most often beaten in the air "
                               "from crosses (2); 2 of 3 on their right side.")
    assert entries["lever"] == "Cross early to the far post and put our best header on their weakest one."
    assert timing["read"] == ("4 of 5 goals conceded came after half-time (80%), 2 after the 75th minute, across 2 team sheets. "
                              "Logged reels: 3 of 4 conceded moments came after half-time.")
    assert timing["lever"].startswith("Save the press for the second half")


def test_patterns_without_moments_or_evidence_prompt_the_analyst():
    corridors, entries, timing = hd.build_dossier("Sholing", [])["patterns"]["items"]
    assert not (corridors["lever"] or entries["lever"] or timing["lever"])
    assert timing["read"] == "No goal times yet. Fetch their results or add minutes to conceded moments."
    balanced = [hd.new_event("goal_scored", {"channel": c, "action": "Cross", "arrival": "Far post", "counter_trigger": "Not a counter"})
                for c in ("Left wing", "Central", "Right wing")]
    assert hd.build_dossier("Sholing", balanced)["patterns"]["items"][0]["lever"].startswith("No corridor above half")


def test_patterns_reach_the_print_sheet_card_and_phone_brief():
    from cv import briefing_sheet as bs
    from cv import vision_center_ui as ui
    sys.path.insert(0, str(ROOT / "tools"))
    import scout_brief as sb

    dossier = hd.build_dossier("Sholing", _reel_moments(), _evidence())
    sheet, card = bs.dossier_sheet(dossier), ui._dossier_card(dossier)
    for html in (sheet, card):
        assert "Patterns Across 2 Match Reels" in html and "Left 60% · Central 20% · Right 20%" in html
        assert html.index("Patterns Across") < html.index("Attacking Patterns")
    assert 'class="sh-patterns"' in sheet and 'class="vcc-patterns"' in card

    fx = {"home": "Tiverton Town", "away": "Sholing", "opponent": "Sholing", "home_game": True, "date": "2026-09-19",
          "kickoff": "15:00", "venue": "The Slee Blackwell Solicitors Stadium"}
    md = sb.compose_brief(fx, "", None, patterns=dossier["patterns"])
    assert "## Highlight Reel Patterns" in md and "**Attacking corridors:** Left 60%" in md
    assert "> **Lever:** Save the press for the second half" in md
    assert md.index("## Highlight Reel Patterns") < md.index("## Key Opposition Profiles")
    assert "## Highlight Reel Patterns" not in sb.compose_brief(fx, "", None)

    assert banned_terms(" ".join([hd.dossier_text(dossier), sheet, card, md])) == []
