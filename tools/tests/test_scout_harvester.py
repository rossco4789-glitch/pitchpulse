"""
tools/tests/test_scout_harvester.py
Schema-integrity and smoke tests for the Opposition Intelligence Harvester.
Offline only — uses the built-in mock corpus and tmp_path sources.

Run: python -m pytest tools/tests/test_scout_harvester.py -v
"""

import copy
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import scout_harvester as sh  # noqa: E402


def _run_mock(opponent="Willand Rovers"):
    return sh.run_pipeline({"opponent": opponent, "slug": sh.slugify(opponent),
                            "source_dir": None, "force_mock": True})


# ── Schema integrity ──────────────────────────────────────────────────────────

def test_mock_dossier_passes_validation():
    d = _run_mock()["dossier"]
    assert d["validation"]["passed"], d["validation"]["errors"]
    assert d["data_provenance"] == "mock"
    assert d["opponent_slug"] == "willand_rovers"


def test_dossier_has_uefa_moments_and_required_fields():
    d = _run_mock()["dossier"]
    assert tuple(sh.SCHEMA) == ("in_possession", "out_of_possession", "transition_attacking",
                                "transition_defensive", "set_pieces")
    assert set(d["in_possession"]) == {"build_up_pattern", "style_preference", "key_ball_progressors"}
    assert set(d["out_of_possession"]) == {"block_height", "flank_vulnerability"}
    assert set(d["transition_attacking"]) == {"counter_attack_speed", "outlet_channels"}
    assert set(d["transition_defensive"]) == {"counter_press_intensity", "rest_defence_shape"}
    assert set(d["set_pieces"]) == {"corner_delivery_zones", "defensive_marking"}


def test_every_field_obeys_enum_and_confidence_contract():
    d = _run_mock()["dossier"]
    for moment, fields in sh.SCHEMA.items():
        for field, (kind, allowed) in fields.items():
            item = d[moment][field]
            assert 0 <= item["confidence"] <= 1
            assert item["confidence_band"] == sh._band(item["confidence"])
            if kind == "enum":
                assert item["value"] in allowed
            elif kind == "multi":
                assert set(item["value"]) <= set(allowed)
                assert sum(item["distribution"].values()) == item["evidence_count"]


def test_block_height_vocabulary_is_high_mid_low():
    kind, allowed = sh.SCHEMA["out_of_possession"]["block_height"]
    assert set(allowed) == {"high_block", "mid_block", "low_block", "unknown"}


def test_corner_zones_match_set_piece_matrix():
    text = (ROOT / "reports" / "set_piece_matrix.py").read_text(encoding="utf-8")
    zones = tuple(re.findall(r'^\s*\("([^"]+)",\s*lambda', text, re.M))
    assert zones == sh.CORNER_ZONES
    assert sh.SCHEMA["set_pieces"]["corner_delivery_zones"][1] == sh.CORNER_ZONES


# ── Agent Beta rules ──────────────────────────────────────────────────────────

def _corpus(*sentences, roster=()):
    return {"provenance": "local_sources", "roster": list(roster), "fingerprint": "x",
            "documents": [{"id": "d1", "type": "match_report", "title": "t", "sentences": list(sentences)}]}


def test_negation_guard_blocks_false_press_trigger():
    draft = sh.agent_beta({"corpus": _corpus("They did not press high at any point.", "They sat in a low block.")})
    assert draft["out_of_possession"]["block_height"]["distribution"] == {"low_block": 1}


def test_squad_number_is_not_treated_as_negation():
    roster = [{"player": "No.8", "shirt": 8, "position": "CM"}]
    draft = sh.agent_beta({"corpus": _corpus("The No.8 progressed play.", roster=roster)})
    assert draft["in_possession"]["key_ball_progressors"]["value"][0]["player"] == "No.8"


def test_marking_sentences_do_not_count_as_delivery_zones():
    draft = sh.agent_beta({"corpus": _corpus("At corners they marked zonal in the six-yard box.")})
    assert draft["set_pieces"]["corner_delivery_zones"]["value"] == []
    assert draft["set_pieces"]["defensive_marking"]["value"] == "zonal"


def test_no_evidence_yields_unknown_with_zero_confidence():
    draft = sh.agent_beta({"corpus": _corpus("Nothing tactical here.")})
    item = draft["transition_defensive"]["counter_press_intensity"]
    assert (item["value"], item["confidence"], item["confidence_band"]) == ("unknown", 0.0, "none")


# ── Agent Gamma verifier ──────────────────────────────────────────────────────

def test_gamma_rejects_invalid_zone():
    draft = copy.deepcopy(_run_mock()["draft"])
    draft["set_pieces"]["corner_delivery_zones"]["value"] = ["Near Post", "Top Bins"]
    _, errors, _ = sh.validate_dossier(draft)
    assert any("Top Bins" in e for e in errors)


def test_gamma_rejects_unknown_with_confidence_and_bad_enum():
    draft = copy.deepcopy(_run_mock()["draft"])
    draft["out_of_possession"]["block_height"].update(value="unknown", confidence=0.5)
    draft["set_pieces"]["defensive_marking"]["value"] = "a bit of both really"
    _, errors, _ = sh.validate_dossier(draft)
    assert any("block_height" in e and "unknown" in e for e in errors)
    assert any("defensive_marking" in e for e in errors)


def test_gamma_strips_fluff_and_unknown_keys():
    draft = copy.deepcopy(_run_mock()["draft"])
    draft["pundit_take"] = {"quote": "they want it more"}
    draft["in_possession"]["vibes"] = "great energy"
    draft["in_possession"]["key_ball_progressors"]["value"][0]["player"] = "  well,  No.8 "
    clean, errors, warnings = sh.validate_dossier(draft)
    assert not errors
    assert "pundit_take" not in clean and "vibes" not in clean["in_possession"]
    assert clean["in_possession"]["key_ball_progressors"]["value"][0]["player"] == "No.8"
    assert any("pundit_take" in w for w in warnings)


def test_cyclic_task_graph_is_rejected():
    graph = ({"id": "A", "agent": "alpha", "depends_on": ("B",), "writes": "corpus"},
             {"id": "B", "agent": "beta", "depends_on": ("A",), "writes": "draft"})
    with pytest.raises(RuntimeError):
        sh.run_pipeline({"opponent": "X", "slug": "x", "source_dir": None, "force_mock": True}, graph)


# ── CLI smoke ─────────────────────────────────────────────────────────────────

def test_cli_writes_valid_json_dossier(tmp_path, capsys):
    rc = sh.main(["--opponent", "Willand Rovers", "--output", str(tmp_path), "--mock"])
    assert rc == 0
    d = json.loads((tmp_path / "willand_rovers_dossier.json").read_text(encoding="utf-8"))
    assert d["validation"]["passed"]
    assert "PASS" in capsys.readouterr().out


def test_output_is_byte_deterministic(tmp_path):
    for sub in ("a", "b"):
        assert sh.main(["--opponent", "Willand Rovers", "--output", str(tmp_path / sub), "--mock"]) == 0
    name = "willand_rovers_dossier.json"
    assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_local_source_ingestion(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "report.txt").write_text(
        "They defended in a low block. Their No.10 carried the ball between the lines.", encoding="utf-8")
    (src / "xi.json").write_text(json.dumps(
        {"type": "lineup", "players": [{"shirt": 10, "name": "No.10", "position": "AM"}]}), encoding="utf-8")
    assert sh.main(["--opponent", "Test FC", "--output", str(tmp_path / "out"), "--source", str(src)]) == 0
    d = json.loads((tmp_path / "out" / "test_fc_dossier.json").read_text(encoding="utf-8"))
    assert d["data_provenance"] == "local_sources"
    assert d["out_of_possession"]["block_height"]["value"] == "low_block"
    assert d["in_possession"]["key_ball_progressors"]["value"][0]["player"] == "No.10"


def test_bad_source_json_exits_2_and_writes_nothing(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "broken.json").write_text("{not json", encoding="utf-8")
    assert sh.main(["--opponent", "Test FC", "--output", str(tmp_path / "out"), "--source", str(src)]) == 2
    assert not (tmp_path / "out").exists()


# ── Fail-loud regressions (scratchpad audit, 14 Sep 2026) ─────────────────────
# Before: realistic coach shorthand and irrelevant notes passed with 0/10 fields, a .docx beside a
# .md was skipped silently, and a missing source folder fell back to the MOCK corpus with exit 0.

COACH_SHORTHAND = (
    "Watched them v Hungerford Saturday. Keeper kicks it long nearly every time, big lad up top (9) wins most of it.\n"
    "Rarely play out short. Their 6 just sits in front of the back four. Left back bombs on, get Slough in behind him.\n"
    "When they lose it they don't really chase, drop off into shape quick. 4-4-2, fairly deep.\n"
    "Corners whipped to the back stick, big CB attacks it. They mark man for man on ours.\n"
)
IRRELEVANT_NOTES = "Travel 2h 10m. Pitch is 3G. Kick off 19:45. Weather looked poor.\n"


def _src(tmp_path, files: dict):
    src = tmp_path / "src"
    src.mkdir()
    for name, text in files.items():
        (src / name).write_text(text, encoding="utf-8")
    return src


@pytest.mark.parametrize("text", [COACH_SHORTHAND, IRRELEVANT_NOTES], ids=["coach_shorthand", "irrelevant"])
def test_zero_evidence_fails_validation_and_writes_nothing(tmp_path, capsys, text):
    src = _src(tmp_path, {"notes.md": text})
    assert sh.main(["--opponent", "Test FC", "--output", str(tmp_path / "out"), "--source", str(src)]) == 1
    assert not (tmp_path / "out").exists()
    assert "no tactical evidence" in capsys.readouterr().out


def test_unsupported_file_beside_markdown_exits_2(tmp_path, capsys):
    src = _src(tmp_path, {"one_line.md": "They press high from goal kicks.\n"})
    (src / "coach_report.docx").write_bytes(b"PK\x03\x04 placeholder")
    assert sh.main(["--opponent", "Test FC", "--output", str(tmp_path / "out"), "--source", str(src)]) == 2
    assert not (tmp_path / "out").exists()
    assert "coach_report.docx" in capsys.readouterr().out


def test_missing_default_sources_without_mock_exits_2(tmp_path, capsys):
    opponent = "No Such Opponent FC"
    assert not (ROOT / "data" / "scouting" / "sources" / sh.slugify(opponent)).exists()
    assert sh.main(["--opponent", opponent, "--output", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()
    assert "--mock" in capsys.readouterr().out


def test_hidden_files_do_not_block_valid_sources(tmp_path):
    src = _src(tmp_path, {"report.txt": "They defended in a low block.\n", ".DS_Store": "x"})
    assert sh.main(["--opponent", "Test FC", "--output", str(tmp_path / "out"), "--source", str(src)]) == 0


# ── Decoupling boundary ───────────────────────────────────────────────────────

def test_harvester_not_coupled_to_app_or_tagger():
    for rel in ("app.py", "tagger/index.html"):
        assert "scout_harvester" not in (ROOT / rel).read_text(encoding="utf-8")
