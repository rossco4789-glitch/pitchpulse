"""
tools/tests/test_vision_center.py
Tactical Vision Command Center logic (cv/vision_center.py, cv/vision_preflight.py). Offline: no Kaggle,
no YOLO; the only real subprocess is a short sleeper used to prove detached-process tracking.

Run: python -m pytest tools/tests/test_vision_center.py -v
"""

import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import psutil
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cv import vision_center as vc  # noqa: E402
from cv import vision_preflight as vp  # noqa: E402

SLUG = "supporting_charities"

# Collected Veo run d9fa85f6 (15 Sep 2026)
OOP = {
    "block_height_m": {"value": 9.88, "iqr": [5.18, 24.09], "n": 171, "reason": None},
    "compactness_depth_m": {"value": 21.28, "iqr": [14.96, 32.92], "n": 171, "reason": None},
    "compactness_width_m": {"value": 20.6, "iqr": [13.81, 29.24], "n": 171, "reason": None},
    "line_of_engagement_m": {"value": 21.02, "iqr": [18.38, 49.71], "n": 171, "reason": None},
}


@pytest.fixture
def ui_env(tmp_path, monkeypatch):
    monkeypatch.setattr(vc, "UI_STATE", tmp_path / "ui")
    monkeypatch.setattr(vc, "SOURCES", tmp_path / "sources")
    monkeypatch.setattr(vc, "STAGING_VIDEOS", tmp_path / "staging")
    (tmp_path / "staging").mkdir()
    return tmp_path


def _write_run(root: Path, slug: str = SLUG, oop=None, job=None):
    d = root / "sources" / slug
    d.mkdir(parents=True, exist_ok=True)
    doc = {"footage": "full_wide", "frames": {"sampled": 13868, "accepted": 5927},
           "moments": {"out_of_possession": oop or OOP},
           "team_split": {"silhouette": 0.607, "settled_frames": 171, "players": 30872},
           "rules": {"settled_min_players": 6, "settled_max_gap_s": 2.0}}
    (d / "vision_metrics.json").write_text(json.dumps(doc), encoding="utf-8")
    if job is not None:
        (d / "vision_job.json").write_text(json.dumps(job), encoding="utf-8")


# ── Staged footage ───────────────────────────────────────────────────────────

def test_human_size_and_staged_name():
    assert (vc.human_size(2_283_000_000), vc.human_size(19_500_000), vc.human_size(4096)) == ("2.1 GB", "19 MB", "4 KB")
    assert vc.staged_name("Tiverton v Dorchester (Veo).MP4") == "Tiverton_v_Dorchester_Veo_.MP4"
    assert vc.staged_name("../../etc/match") == "match.mp4"


def test_staged_videos_badge_size_resolution_and_duration(ui_env):
    path = vc.STAGING_VIDEOS / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(20):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    (vc.STAGING_VIDEOS / "notes.txt").write_text("ignored", encoding="utf-8")
    videos = vc.staged_videos()
    assert [v["name"] for v in videos] == ["clip.mp4"]
    assert (videos[0]["resolution"], videos[0]["duration"], videos[0]["downsampled"]) == ("48p", "0:02", False)


# ── Detached processes and pipeline state ────────────────────────────────────

def test_launch_tracks_a_detached_process_until_it_exits(ui_env):
    rec = vc.launch(SLUG, "pipeline", "collect", ["-c", "import time; time.sleep(30)"])
    try:
        assert vc.active_process(SLUG, "pipeline")["pid"] == rec["pid"]
        assert vc.start_collect(SLUG) is None                 # lane busy: no second process
        assert Path(rec["log"]).exists()
    finally:
        p = psutil.Process(rec["pid"])
        p.kill()
        p.wait(10)
    assert vc.active_process(SLUG, "pipeline") is None


def test_recycled_pid_is_not_mistaken_for_the_job(ui_env):
    me = psutil.Process()
    vc._write_json(vc._proc_path(SLUG, "preflight"), {"pid": os.getpid(), "create_time": me.create_time(), "action": "preflight"})
    assert vc.active_process(SLUG, "preflight") is not None
    vc._write_json(vc._proc_path(SLUG, "preflight"), {"pid": os.getpid(), "create_time": me.create_time() - 500, "action": "preflight"})
    assert vc.active_process(SLUG, "preflight") is None


def test_dispatch_argv_and_live_job_blocks_a_second_dispatch(ui_env, monkeypatch):
    argv = vc.dispatch_argv("data/staging/m.mp4", SLUG, "#cc2222")
    assert argv[0].endswith("cloud_vision_runner.py") and argv[-1] == "--dispatch-only"
    assert argv[argv.index("--opponent") + 1] == SLUG and argv[argv.index("--opponent-kit") + 1] == "#CC2222"
    assert vc.collect_argv(SLUG)[1:] == ["--opponent", SLUG, "--collect"]   # no --wait: one status check per launch

    launched = []
    monkeypatch.setattr(vc, "launch", lambda *a: launched.append(a) or {"action": a[2]})
    _write_run(ui_env, job={"state": "RUNNING"})
    assert vc.start_dispatch(SLUG, "m.mp4", "#CC2222") is None and launched == []
    _write_run(ui_env, job={"state": "COMPLETE"})
    assert vc.start_dispatch(SLUG, "m.mp4", "#CC2222") == {"action": "dispatch"}


@pytest.mark.parametrize("job,proc,expected", [
    (None, None, "IDLE"),
    ({"state": "COMPLETE"}, {"action": "dispatch"}, "DISPATCHING"),   # job file still holds the previous run
    ({"state": "DISPATCHED"}, None, "DISPATCHED"),
    ({"state": "RUNNING"}, {"action": "collect"}, "RUNNING"),         # a status check keeps the job state
    ({"state": "COMPLETE"}, None, "COMPLETE"),
    ({"state": "INVALID"}, None, "INVALID"),
    ({"state": "???"}, None, "IDLE"),
])
def test_pipeline_state(job, proc, expected):
    assert vc.pipeline_state(job, proc) == expected


def test_auto_collect_runs_once_a_minute_only_while_a_kernel_is_live():
    live, last = {"state": "RUNNING"}, {"action": "collect", "started_ts": 1000.0}
    assert vc.should_auto_collect(live, None, None, now=1000.0)
    assert vc.should_auto_collect(live, None, {"action": "dispatch", "started_ts": 999.0}, now=1000.0)
    assert not vc.should_auto_collect(live, None, last, now=1059.0)
    assert vc.should_auto_collect(live, None, last, now=1060.0)
    assert not vc.should_auto_collect(live, {"action": "collect"}, last, now=5000.0)
    assert not vc.should_auto_collect({"state": "COMPLETE"}, None, None, now=5000.0)
    assert vc.seconds_to_next_check(last, now=1015.0) == 45


def test_live_opponents_and_time_based_progress(ui_env):
    _write_run(ui_env, "supporting_charities", job={"state": "COMPLETE"})
    _write_run(ui_env, "dorchester_town", job={"state": "RUNNING", "dispatched_at": "2026-09-15T14:30:00+00:00"})
    assert vc.live_opponents() == ["dorchester_town"]

    start = vc.datetime.fromisoformat("2026-09-15T14:30:00+00:00").timestamp()
    running = {"state": "RUNNING", "dispatched_at": "2026-09-15T14:30:00+00:00"}
    assert vc.analysis_progress(running, None, now=start + 750) == (0.625, "Analyzing match footage…")
    assert vc.analysis_progress(running, None, now=start + 99999)[0] == 0.95          # never full until collected
    assert vc.analysis_progress({"state": "COMPLETE"}, {"action": "dispatch", "started_ts": 1000.0}, now=1300.0) == \
        (0.175, "Preparing match footage…")
    assert vc.analysis_progress({"state": "COMPLETE"}, None) == (1.0, "Report ready")
    assert vc.analysis_progress(None, None) == (0.0, "")


# ── Badges, summaries and history ────────────────────────────────────────────

def test_block_and_silhouette_badges():
    assert [vc.block_badge(m) for m in (None, 17.9, 18.0, 32.0, 32.1)] == [None, "Low Block", "Mid Block", "Mid Block", "High Block"]
    assert [vc.silhouette_badge(s)[0] for s in (None, 0.49, 0.50, 0.549, 0.55)] == ["grey", "red", "amber", "amber", "green"]


def test_reason_text_translates_worker_reasons():
    assert vc.reason_text("insufficient_coverage (n=12, windows=1)") == "12 settled frames in 1 window(s); needs 30 across 2."
    assert "0.43" in vc.reason_text("team_split_ambiguous (silhouette 0.43)")
    assert vc.reason_text("opponent_kit_not_given") == "No opponent kit was set at dispatch."


def test_match_summary_and_history_rows(ui_env):
    _write_run(ui_env, job={"state": "COMPLETE", "collected_at": "2026-09-15T14:54:41+00:00"})
    (ui_env / "sources" / "no_metrics_yet").mkdir()
    assert vc.scouted_opponents() == [SLUG]
    s = vc.match_summary(SLUG)
    assert (s["opponent"], s["date"], round(s["acceptance"], 3), s["silhouette"]) == ("Supporting Charities", "2026-09-15", 0.427, 0.607)
    row = vc.history_rows()[0]
    assert (row["Mapped %"], row["Block"], row["LoE m"], row["Settled frames"]) == (42.7, "Low Block", 21.02, 171)


def test_shape_read_leads_with_numbers_and_ends_on_a_lever(ui_env):
    _write_run(ui_env)
    read = vc.shape_read(vc.match_summary(SLUG))
    assert read.startswith("Low Block: deepest four at 9.9 m") and "Low confidence: 42.7%" in read
    assert "switch play early to the far full-back" in read and read.endswith("cut-backs.")
    banned = r"delve|testament to|tapestry|spearhead|in conclusion|worth noting|certainly!|happy to|pivotal|crucial|robust|seamless|let's dive"
    assert not re.search(banned, read, re.I)


# ── Pitch geometry and figures ───────────────────────────────────────────────

def test_shape_geometry():
    geo = vc.shape_geometry(OOP)
    assert geo["zone"] == [(0.0, 23.7), (21.02, 23.7), (21.02, 44.3), (0.0, 44.3)]   # depth 21.28 > LoE: clamped at the goal line
    assert geo["centroid"] == (10.51, 34.0) and geo["block_iqr"] == (5.18, 24.09)
    assert (geo["block"], geo["loe"], geo["depth"], geo["width"]) == (9.88, 21.02, 21.28, 20.6)

    missing = {**OOP, "compactness_width_m": {"value": None, "reason": "not_reported"}}
    assert vc.shape_geometry(missing) is None


# ── Kit separation pre-flight ────────────────────────────────────────────────

def test_kit_hex_round_trips_the_weighted_lab_feature():
    out = vc.kit_hex(vc.worker().hex_to_kit("#CC2222"))
    assert all(abs(int(out[i:i + 2], 16) - int("#CC2222"[i:i + 2], 16)) <= 3 for i in (1, 3, 5))


def _kits(n=20, seed=0):
    rng = np.random.default_rng(seed)
    red, navy = vc.worker().hex_to_kit("#CC2222"), vc.worker().hex_to_kit("#1A1A24")
    return [list(red + rng.normal(0, 3, 3)) for _ in range(n)] + [list(navy + rng.normal(0, 3, 3)) for _ in range(n)]


def test_separation_report_maps_the_opponent_kit_to_its_cluster():
    report = vc.separation_report(_kits(), "#cc2222")
    assert report["status"] == "ok" and report["silhouette"] > 0.8 and report["kit"] == "#CC2222"
    r, g, b = (int(report["opponent"]["hex"][i:i + 2], 16) for i in (1, 3, 5))
    assert r > 150 and r > 3 * g and report["opponent"]["players"] == 20
    assert vc.separation_report(_kits(n=4), "#CC2222") == {"status": "insufficient", "players": 8, "kit": "#CC2222"}


def test_preflight_main_always_writes_a_result(tmp_path):
    out = tmp_path / "preflight.json"
    assert vp.main(["--video", "match.mp4", "--kit", "#cc2222", "--out", str(out)], sampler=lambda p: (_kits(), 5)) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert (doc["status"], doc["frames"], doc["video"]) == ("ok", 5, "match.mp4")

    def broken(p):
        raise FileNotFoundError("players.pt not found")

    vp.main(["--video", "match.mp4", "--kit", "#CC2222", "--out", str(out)], sampler=broken)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["status"] == "error" and "players.pt not found" in doc["error"]
