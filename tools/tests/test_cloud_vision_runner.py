"""
tools/tests/test_cloud_vision_runner.py
Kaggle dispatcher/collector and remote worker logic. Offline only: every Kaggle CLI call goes through
cloud_vision_runner.kaggle, which these tests replace; the worker is exercised on synthetic geometry.

Run: python -m pytest tools/tests/test_cloud_vision_runner.py -v
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import cloud_vision_runner as cvr  # noqa: E402

_spec = importlib.util.spec_from_file_location("kaggle_vision_worker", ROOT / "tools" / "templates" / "kaggle_vision_worker.py")
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)

SLUG = "dorchester_town"


def cp(rc=0, out="", err=""):
    return subprocess.CompletedProcess(["kaggle"], rc, out, err)


@pytest.fixture
def env(tmp_path, monkeypatch):
    token = tmp_path / "kaggle.json"
    token.write_text(json.dumps({"username": "tivvy", "key": "secret"}), encoding="utf-8")
    monkeypatch.setattr(cvr, "KAGGLE_JSON", token)
    monkeypatch.setattr(cvr, "SOURCES", tmp_path / "sources")
    monkeypatch.setattr(cvr, "STAGING", tmp_path / "staging")
    monkeypatch.setattr(cvr.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cvr.time, "sleep", lambda s: None)
    return tmp_path


# ── Auth and setup ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("content", [None, "{not json", json.dumps({"username": "tivvy"})],
                         ids=["missing", "malformed", "no_key"])
def test_missing_or_broken_auth_exits_2(env, capsys, content):
    token = env / "kaggle.json"
    if content is None:
        token.unlink()
    else:
        token.write_text(content, encoding="utf-8")
    assert cvr.main(["--opponent", SLUG, "--collect"]) == 2
    out = capsys.readouterr().out
    assert "kaggle.json" in out and "Create New Token" in out


def test_missing_kaggle_cli_exits_2_with_install_hint(env, monkeypatch, capsys):
    (env / "match.mp4").write_bytes(b"x")
    monkeypatch.setattr(cvr.shutil, "which", lambda name: None)
    assert cvr.main(["--video", str(env / "match.mp4"), "--opponent", SLUG, "--dispatch-only"]) == 2
    assert "pip install kaggle" in capsys.readouterr().out


def test_models_dataset_missing_files_exits_2_before_upload(env, monkeypatch, capsys):
    (env / "match.mp4").write_bytes(b"x")
    calls = []

    def fake(args, timeout=600):
        calls.append(args)
        return cp(out="players.pt 1") if args[:2] == ["datasets", "files"] else cp(1)

    monkeypatch.setattr(cvr, "kaggle", fake)
    assert cvr.main(["--video", str(env / "match.mp4"), "--opponent", SLUG, "--dispatch-only"]) == 2
    assert "pitch.pt" in capsys.readouterr().out
    assert calls == [["datasets", "files", "tivvy/pitchpulse-cv-models"]]


def test_collect_without_job_exits_2(env, capsys):
    assert cvr.main(["--opponent", SLUG, "--collect"]) == 2
    assert "--dispatch-only first" in capsys.readouterr().out


# ── Payload preparation ───────────────────────────────────────────────────────

def test_ffmpeg_downsample_parameters():
    cmd = cvr.ffmpeg_downsample_cmd(Path("in.mp4"), Path("out.mp4"))
    assert cmd[0] == "ffmpeg" and cmd[cmd.index("-i") + 1] == "in.mp4" and cmd[-1] == "out.mp4"
    assert cmd[cmd.index("-vf") + 1] == "fps=2,scale=-2:720"
    assert (cmd[cmd.index("-crf") + 1], cmd[cmd.index("-preset") + 1], cmd[cmd.index("-c:v") + 1]) == ("24", "fast", "libx264")
    assert "-an" in cmd and "-y" in cmd


def test_large_video_is_downsampled_small_video_is_not(env, monkeypatch):
    video = env / "match.mp4"
    video.write_bytes(b"0123456789abcdef")
    stage = env / "stage"
    stage.mkdir()
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        Path(cmd[-1]).write_bytes(b"small")
        return cp()

    monkeypatch.setattr(cvr.subprocess, "run", fake_run)
    monkeypatch.setattr(cvr, "DOWNSAMPLE_BYTES", 10)
    payload, down = cvr.prepare_payload(video, stage)
    assert down and payload.read_bytes() == b"small" and seen == [cvr.ffmpeg_downsample_cmd(video, stage / "payload.mp4")]

    monkeypatch.setattr(cvr, "DOWNSAMPLE_BYTES", 10 ** 9)
    payload.unlink()
    payload, down = cvr.prepare_payload(video, stage)
    assert not down and payload.read_bytes() == video.read_bytes() and len(seen) == 1


def test_ffmpeg_failure_is_a_setup_error(env, monkeypatch):
    video = env / "match.mp4"
    video.write_bytes(b"0123456789abcdef")
    monkeypatch.setattr(cvr.subprocess, "run", lambda cmd, **kw: cp(1, err="Invalid data found"))
    monkeypatch.setattr(cvr, "DOWNSAMPLE_BYTES", 10)
    with pytest.raises(cvr.SetupError, match="Invalid data"):
        cvr.prepare_payload(video, env)


def test_retry_backs_off_exponentially_then_gives_up(monkeypatch):
    delays = []
    monkeypatch.setattr(cvr.time, "sleep", delays.append)
    results = iter([cp(1, err="reset")] * 4 + [cp(out="ok")])
    assert cvr.with_retry("upload", lambda: next(results)).stdout == "ok"
    assert delays == [2.0, 4.0, 8.0, 16.0]

    delays.clear()
    with pytest.raises(cvr.RemoteError, match="after 5 attempts: reset"):
        cvr.with_retry("upload", lambda: cp(1, err="reset"))
    assert len(delays) == 4


# ── Dispatch ──────────────────────────────────────────────────────────────────

def test_dispatch_only_uploads_pushes_and_records_job(env, monkeypatch):
    video = env / "match.mp4"
    video.write_bytes(b"fake video bytes")
    seen, status_calls = {}, []

    def fake(args, timeout=600):
        if args[:2] == ["datasets", "files"]:
            return cp(out="name size\nplayers.pt 1\npitch.pt 1\npitch_config.json 1")
        if args[:2] == ["datasets", "status"]:
            status_calls.append(args)
            return cp(1, err="404") if len(status_calls) == 1 else cp(out="ready")
        if args[:2] == ["datasets", "create"]:
            d = Path(args[args.index("-p") + 1])
            seen["manifest"] = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            seen["dataset"] = json.loads((d / "dataset-metadata.json").read_text(encoding="utf-8"))
            return cp()
        if args[:2] == ["kernels", "push"]:
            d = Path(args[args.index("-p") + 1])
            seen["kernel"] = json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8"))
            seen["worker"] = (d / "kaggle_vision_worker.py").read_text(encoding="utf-8")
            return cp()
        raise AssertionError(f"unexpected kaggle call {args}")

    monkeypatch.setattr(cvr, "kaggle", fake)
    assert cvr.main(["--video", str(video), "--opponent", SLUG, "--dispatch-only", "--opponent-kit", "#000000"]) == 0

    job = json.loads((env / "sources" / SLUG / "vision_job.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(b"fake video bytes").hexdigest()
    assert (job["state"], job["kernel_ref"], job["payload_sha256"]) == ("DISPATCHED", "tivvy/pitchpulse-vision-dorchester-town", digest)
    assert seen["manifest"]["payload_sha256"] == digest and seen["manifest"]["opponent_kit"] == "#000000"
    assert seen["dataset"]["id"] == "tivvy/pitchpulse-footage-dorchester-town"
    k = seen["kernel"]
    assert (k["enable_gpu"], k["is_private"], k["kernel_type"], k["code_file"]) == (True, True, "script", "kaggle_vision_worker.py")
    assert k["dataset_sources"] == ["tivvy/pitchpulse-footage-dorchester-town", "tivvy/pitchpulse-cv-models"]
    assert "def main()" in seen["worker"]
    assert not (env / "staging" / SLUG).exists()


# ── Output verification ───────────────────────────────────────────────────────

def _doc(sha="a" * 64, **over):
    moments = {m: {n: {"value": None, "n": 0, "iqr": None, "reason": "possession_not_observable_without_ball_tracking"}
                   for n in names} for m, names in cvr.MOMENTS.items()}
    moments["out_of_possession"]["block_height_m"] = {"value": 31.5, "n": 120, "iqr": [28.0, 34.0], "reason": None}
    doc = {"schema_version": 1, "status": "OK", "footage": "full_wide", "input_sha256": sha, "moments": moments,
           "frames": {"sampled": 10, "accepted": 7, "dropped": {r: (3 if r == "too_few_landmarks" else 0) for r in cvr.DROP_RULES}}}
    doc.update(over)
    return doc


def _write_out(d: Path, doc, good_sha=True) -> bytes:
    raw = json.dumps(doc).encode("utf-8")
    (d / "vision_metrics.json").write_bytes(raw)
    (d / "vision_metrics.sha256").write_text(hashlib.sha256(raw).hexdigest() if good_sha else "0" * 64, encoding="utf-8")
    return raw


def test_valid_output_passes(tmp_path):
    raw = _write_out(tmp_path, _doc())
    doc, got = cvr.verify_output(tmp_path, "a" * 64)
    assert got == raw and doc["moments"]["out_of_possession"]["block_height_m"]["value"] == 31.5


def test_output_hash_mismatch_is_invalid(tmp_path):
    _write_out(tmp_path, _doc(), good_sha=False)
    with pytest.raises(cvr.OutputError) as exc:
        cvr.verify_output(tmp_path, "a" * 64)
    assert exc.value.state == "INVALID" and "sha256" in exc.value.detail


def test_payload_hash_mismatch_is_invalid(tmp_path):
    _write_out(tmp_path, _doc(sha="b" * 64))
    with pytest.raises(cvr.OutputError, match="different payload"):
        cvr.verify_output(tmp_path, "a" * 64)


def test_worker_failed_report_is_failed_with_traceback(tmp_path):
    _write_out(tmp_path, {"status": "FAILED", "error": "Traceback ... CUDA out of memory"})
    with pytest.raises(cvr.OutputError) as exc:
        cvr.verify_output(tmp_path, "a" * 64)
    assert exc.value.state == "FAILED" and "CUDA out of memory" in exc.value.detail


def _mutate(doc, path, value):
    *keys, last = path
    target = doc
    for k in keys:
        target = target[k]
    target[last] = value
    return doc


@pytest.mark.parametrize("path,value,needle", [
    (("moments", "out_of_possession", "block_height_m", "n"), 12, "n=12"),
    (("moments", "out_of_possession", "block_height_m", "value"), 140.0, "outside"),
    (("moments", "out_of_possession", "block_height_m", "iqr"), [40.0, 50.0], "iqr"),
    (("moments", "in_possession", "settled_width_m", "reason"), None, "needs a reason"),
    (("frames", "accepted"), 9, "does not equal"),
    (("footage",), "highlight", "highlight footage"),
    (("status",), "PARTIAL", "status must be OK"),
])
def test_schema_rejections(path, value, needle):
    errors = cvr.validate_metrics(_mutate(_doc(), path, value))
    assert any(needle in e for e in errors), errors


def test_schema_rejects_missing_moment():
    doc = _doc()
    del doc["moments"]["attacking_transition"]
    assert any("moments must be exactly" in e for e in cvr.validate_metrics(doc))


# ── Collect ───────────────────────────────────────────────────────────────────

def _dispatched(env, sha="a" * 64):
    cvr.write_job(SLUG, state="DISPATCHED", kernel_ref="tivvy/pitchpulse-vision-dorchester-town", payload_sha256=sha)


def test_collect_complete_verifies_and_writes_metrics(env, monkeypatch):
    _dispatched(env)
    written = {}

    def fake(args, timeout=600):
        if args[:2] == ["kernels", "status"]:
            return cp(out='tivvy/pitchpulse-vision-dorchester-town has status "complete"')
        if args[:2] == ["kernels", "output"]:
            written["raw"] = _write_out(Path(args[args.index("-p") + 1]), _doc())
            return cp()
        raise AssertionError(args)

    monkeypatch.setattr(cvr, "kaggle", fake)
    assert cvr.main(["--opponent", SLUG, "--collect"]) == 0
    assert (env / "sources" / SLUG / "vision_metrics.json").read_bytes() == written["raw"]
    job = cvr.read_job(SLUG)
    assert job["state"] == "COMPLETE" and job["output_sha256"] == hashlib.sha256(written["raw"]).hexdigest()


def test_collect_running_exits_5_and_records_state(env, monkeypatch):
    _dispatched(env)
    monkeypatch.setattr(cvr, "kaggle", lambda args, timeout=600: cp(out='has status "KernelWorkerStatus.RUNNING"'))
    assert cvr.main(["--opponent", SLUG, "--collect"]) == 5
    assert cvr.read_job(SLUG)["state"] == "RUNNING"
    assert not (env / "sources" / SLUG / "vision_metrics.json").exists()


def test_collect_invalid_output_exits_4_and_writes_nothing(env, monkeypatch):
    _dispatched(env)

    def fake(args, timeout=600):
        if args[:2] == ["kernels", "status"]:
            return cp(out='has status "complete"')
        _write_out(Path(args[args.index("-p") + 1]), _doc(), good_sha=False)
        return cp()

    monkeypatch.setattr(cvr, "kaggle", fake)
    assert cvr.main(["--opponent", SLUG, "--collect"]) == 4
    assert cvr.read_job(SLUG)["state"] == "INVALID"
    assert not (env / "sources" / SLUG / "vision_metrics.json").exists()


def test_collect_worker_failure_exits_3(env, monkeypatch):
    _dispatched(env)

    def fake(args, timeout=600):
        if args[:2] == ["kernels", "status"]:
            return cp(out='has status "complete"')
        _write_out(Path(args[args.index("-p") + 1]), {"status": "FAILED", "error": "Traceback: boom"})
        return cp()

    monkeypatch.setattr(cvr, "kaggle", fake)
    assert cvr.main(["--opponent", SLUG, "--collect"]) == 3
    job = cvr.read_job(SLUG)
    assert job["state"] == "FAILED" and "boom" in job["error"]


# ── Worker: schema parity, geometry rejection, metrics, failure report ────────

def test_worker_schema_constants_match_runner():
    assert (worker.SCHEMA_VERSION, worker.MIN_SAMPLES, worker.DROP_RULES, worker.MOMENTS) == \
           (cvr.SCHEMA_VERSION, cvr.MIN_SAMPLES, cvr.DROP_RULES, cvr.MOMENTS)


BOX_WORLD = np.array([[88.5, 13.84], [105, 13.84], [88.5, 54.16], [105, 54.16],
                      [99.5, 24.84], [99.5, 43.16], [105, 24.84], [105, 43.16]], float)
H_TRUE = np.array([[0.05, 0.002, 20.0], [0.001, 0.06, 5.0], [0.00001, 0.00002, 1.0]])  # image → world


def _image_points(world):
    return worker.project(np.linalg.inv(H_TRUE), world)


def test_homography_accepts_clean_landmarks():
    H, reason = worker.fit_homography(_image_points(BOX_WORLD), BOX_WORLD, np.ones(8))
    assert reason is None
    assert np.abs(worker.project(H, _image_points(BOX_WORLD)) - BOX_WORLD).max() < 0.05


def test_homography_rejects_low_confidence_landmarks():
    conf = np.array([1, 1, 1, 1, 1, 0.2, 0.2, 0.2])
    assert worker.fit_homography(_image_points(BOX_WORLD), BOX_WORLD, conf) == (None, "too_few_landmarks")
    assert worker.fit_homography(_image_points(BOX_WORLD[:3]), BOX_WORLD[:3], np.ones(3)) == (None, "too_few_landmarks")


def test_homography_rejects_collinear_landmarks():
    line = np.array([[105.0, y] for y in (10, 20, 30, 40, 50, 60)])
    assert worker.fit_homography(_image_points(line), line, np.ones(6)) == (None, "degenerate_geometry")


def test_homography_rejects_inconsistent_landmarks():
    world = BOX_WORLD.copy()
    image = _image_points(world)
    world[:3] += 5.0  # three landmarks labelled 5 m away from where the image shows them
    assert worker.fit_homography(image, world, np.ones(8)) == (None, "unstable_homography")


def _rows(windows=(0, 300), noise_seed=0):
    rng = np.random.default_rng(noise_seed)
    red, blue = worker.hex_to_lab("#FF0000"), worker.hex_to_lab("#0000FF")
    rows = []
    for base in windows:
        for k in range(40):
            t = base + k * 0.5
            for i in range(10):
                rows.append({"t": t, "x": 15.0 if i < 4 else 20 + i * 1.5, "y": 6.0 + i * 6, "role": "player",
                             "kit": (red + rng.normal(0, 2, 3)).tolist()})
                rows.append({"t": t, "x": 45.0 + i * 2, "y": 6.0 + i * 6, "role": "player",
                             "kit": (blue + rng.normal(0, 2, 3)).tolist()})
    return rows


def test_metrics_from_settled_block_pass_runner_schema():
    moments, info = worker.compute_metrics(_rows(), "full_wide", "#FF0000")
    oop = moments["out_of_possession"]
    assert (oop["block_height_m"]["value"], oop["block_height_m"]["n"]) == (15.0, 80)
    assert (oop["compactness_depth_m"]["value"], oop["compactness_width_m"]["value"], oop["line_of_engagement_m"]["value"]) == (18.5, 54.0, 33.5)
    assert info["silhouette"] > 0.9
    doc = {"schema_version": 1, "status": "OK", "footage": "full_wide", "input_sha256": "c" * 64, "moments": moments,
           "frames": {"sampled": 80, "accepted": 80, "dropped": {r: 0 for r in cvr.DROP_RULES}}}
    assert cvr.validate_metrics(doc) == []


def test_metrics_mirror_when_opponent_defends_the_far_end():
    rows = _rows()
    for r in rows:
        r["x"], r["y"] = 105 - r["x"], 68 - r["y"]
    assert worker.compute_metrics(rows, "full_wide", "#FF0000")[0]["out_of_possession"]["block_height_m"]["value"] == 15.0


@pytest.mark.parametrize("rows,footage,kit,reason", [
    (_rows(), "highlight", "#FF0000", "highlight_footage_invalid_for_shape"),
    (_rows(), "full_wide", None, "opponent_kit_not_given"),
    (_rows(windows=(0,)), "full_wide", "#FF0000", "insufficient_coverage (n=40, windows=1)"),
])
def test_metrics_stay_null_with_reason(rows, footage, kit, reason):
    entry = worker.compute_metrics(rows, footage, kit)[0]["out_of_possession"]["block_height_m"]
    assert (entry["value"], entry["reason"]) == (None, reason)


def test_ambiguous_kits_null_the_team_metrics():
    rows = _rows()
    grey = worker.hex_to_lab("#808080").tolist()
    for r in rows:
        r["kit"] = grey
    reason = worker.compute_metrics(rows, "full_wide", "#FF0000")[0]["out_of_possession"]["block_height_m"]["reason"]
    assert reason.startswith("team_split_ambiguous")


def test_worker_main_writes_failed_report_instead_of_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "OUTPUT", tmp_path)
    monkeypatch.setattr(worker, "run", lambda: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")))
    assert worker.main() == 0
    with pytest.raises(cvr.OutputError) as exc:
        cvr.verify_output(tmp_path, "a" * 64)
    assert exc.value.state == "FAILED" and "RuntimeError: CUDA out of memory" in exc.value.detail
