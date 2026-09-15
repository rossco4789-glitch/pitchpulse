"""
tools/tests/test_cloud_vision_runner.py
Kaggle dispatcher/collector and remote worker logic. Offline only: every Kaggle CLI call goes through
cloud_vision_runner.kaggle, which these tests replace; the worker is exercised on synthetic geometry.

Run: python -m pytest tools/tests/test_cloud_vision_runner.py -v
"""

import hashlib
import importlib.util
import json
import re
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


# ── Subprocess encoding (dry run 14 Sep 2026: cp1252 pipe crash on '⚠️' in the kernel log) ─────

def test_kaggle_call_forces_utf8_env_and_decoding(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(cmd=cmd, **kw)
        return cp()

    monkeypatch.setenv("PITCHPULSE_KEEP_ME", "yes")
    monkeypatch.setattr(cvr.subprocess, "run", fake_run)
    cvr.kaggle(["kernels", "status", "tivvy/x"])
    assert seen["cmd"] == ["kaggle", "kernels", "status", "tivvy/x"]
    assert (seen["env"]["PYTHONUTF8"], seen["env"]["PYTHONIOENCODING"]) == ("1", "utf-8")
    assert seen["env"]["PITCHPULSE_KEEP_ME"] == "yes"  # parent environment (PATH, token vars) is preserved
    assert (seen["encoding"], seen["errors"], seen["text"], seen["capture_output"]) == ("utf-8", "replace", True, True)


def test_non_ascii_and_invalid_bytes_survive_a_real_pipe(monkeypatch):
    """Real child process: emoji printed through a pipe plus a raw 0x8f byte must not crash either side."""
    real_run = subprocess.run
    child = ("import sys; print('WARNING ⚠️ half is deprecated'); sys.stdout.flush(); "
             "sys.stdout.buffer.write(b'raw \\x8f byte\\n')")
    monkeypatch.setattr(cvr.subprocess, "run", lambda cmd, **kw: real_run([sys.executable, "-c", child], **kw))
    result = cvr.kaggle(["kernels", "output", "tivvy/x"])
    assert result.returncode == 0, result.stderr
    assert "WARNING ⚠️ half is deprecated" in result.stdout and "raw � byte" in result.stdout


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

FOOTAGE_REF = "tivvy/pitchpulse-footage-dorchester-town"
MODELS_LISTING = ("name                    size  creationDate\n-----------------  ---------  -----------\n"
                  "pitch.pt           140212306  2026-09-14 14:40:49.757000\n"
                  "pitch_config.json       1786  2026-09-14 14:40:47.580000\n"
                  "players.pt         136802409  2026-09-14 14:40:50.798000\n")


def _dispatch_fake(seen: dict, *, exists: bool, payload_appears: bool = True):
    """Kaggle CLI stand-in: `datasets files` drives existence and readiness exactly as CLI 2.2.4 behaves."""
    seen["calls"] = []

    def fake(args, timeout=600):
        seen["calls"].append(args[:2])
        if args[:2] == ["datasets", "status"]:
            raise AssertionError("datasets status is broken in CLI 2.2.4 and must not be called")
        if args[:2] == ["datasets", "files"] and args[2] == "tivvy/pitchpulse-cv-models":
            return cp(out=MODELS_LISTING)
        if args[:2] == ["datasets", "files"] and args[2] == FOOTAGE_REF:
            if "manifest" not in seen:  # before upload
                return cp(out="name size creationDate\npayload_0ld0ld0ld0ld.mp4 16 2026-09-01\n") if exists \
                    else cp(1, err="403 Client Error: Forbidden")
            rows = "manifest.json 300 2026-09-14\n"
            if payload_appears:
                rows += f"{seen['manifest']['payload']} {seen['payload_size']} 2026-09-14 15:00:00\n"
            return cp(out="name size creationDate\n" + rows)
        if args[:2] in (["datasets", "create"], ["datasets", "version"]):
            d = Path(args[args.index("-p") + 1])
            seen["upload_mode"] = args[1]
            seen["manifest"] = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            seen["payload_size"] = (d / seen["manifest"]["payload"]).stat().st_size
            seen["dataset"] = json.loads((d / "dataset-metadata.json").read_text(encoding="utf-8"))
            return cp()
        if args[:2] == ["kernels", "push"]:
            d = Path(args[args.index("-p") + 1])
            seen["kernel"] = json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8"))
            seen["worker"] = (d / "kaggle_vision_worker.py").read_text(encoding="utf-8")
            return cp()
        raise AssertionError(f"unexpected kaggle call {args}")

    return fake


def test_dataset_listing_parses_cli_table_and_treats_failure_as_absent(monkeypatch):
    calls = []
    monkeypatch.setattr(cvr, "kaggle", lambda args, timeout=600: (calls.append(args), cp(out=MODELS_LISTING))[1])
    assert cvr.dataset_listing("tivvy/pitchpulse-cv-models") == \
           {"pitch.pt": 140212306, "pitch_config.json": 1786, "players.pt": 136802409}
    assert calls == [["datasets", "files", "tivvy/pitchpulse-cv-models"]]

    monkeypatch.setattr(cvr, "kaggle", lambda args, timeout=600: cp(1, err="403 Client Error: Forbidden"))
    assert cvr.dataset_listing(FOOTAGE_REF) is None

    def timeout(args, timeout=600):
        raise subprocess.TimeoutExpired(args, timeout)

    monkeypatch.setattr(cvr, "kaggle", timeout)
    assert cvr.dataset_listing(FOOTAGE_REF) is None


def test_readiness_waits_for_payload_at_expected_size(monkeypatch):
    replies = iter([cp(1, err="403 Client Error"),                                   # not processed yet
                    cp(out="name size creationDate\npayload_abc.mp4 500 2026-09-14\n"),  # partial size
                    cp(out="name size creationDate\npayload_abc.mp4 1024 2026-09-14\n")])
    delays = []
    monkeypatch.setattr(cvr.time, "sleep", delays.append)
    monkeypatch.setattr(cvr, "kaggle", lambda args, timeout=600: next(replies))
    cvr.wait_dataset_ready(FOOTAGE_REF, "payload_abc.mp4", 1024)
    assert delays == [5, 10]


def test_new_dataset_is_created_when_files_listing_fails(env, monkeypatch):
    video = env / "match.mp4"
    video.write_bytes(b"fake video bytes")
    seen = {}
    monkeypatch.setattr(cvr, "kaggle", _dispatch_fake(seen, exists=False))
    assert cvr.main(["--video", str(video), "--opponent", SLUG, "--dispatch-only", "--opponent-kit", "#000000"]) == 0
    assert seen["upload_mode"] == "create" and ["kernels", "push"] in seen["calls"]


def test_existing_dataset_is_versioned_when_files_listing_succeeds(env, monkeypatch):
    video = env / "match.mp4"
    video.write_bytes(b"fake video bytes")
    seen = {}
    monkeypatch.setattr(cvr, "kaggle", _dispatch_fake(seen, exists=True))
    assert cvr.main(["--video", str(video), "--opponent", SLUG, "--dispatch-only"]) == 0
    assert seen["upload_mode"] == "version" and ["kernels", "push"] in seen["calls"]
    assert cvr.read_job(SLUG)["state"] == "DISPATCHED"


def test_payload_never_listed_times_out_with_exit_3_and_no_kernel(env, monkeypatch):
    video = env / "match.mp4"
    video.write_bytes(b"fake video bytes")
    seen, clock, delays = {}, [0.0], []

    def fake_sleep(s):
        delays.append(s)
        clock[0] += s

    monkeypatch.setattr(cvr.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(cvr.time, "sleep", fake_sleep)
    monkeypatch.setattr(cvr, "kaggle", _dispatch_fake(seen, exists=False, payload_appears=False))
    assert cvr.main(["--video", str(video), "--opponent", SLUG, "--dispatch-only"]) == 3

    assert delays[:5] == [5, 10, 20, 40, 60] and max(delays) == 60 and sum(delays) >= cvr.DATASET_READY_S
    job = cvr.read_job(SLUG)
    assert job["state"] == "FAILED" and "not ready after 10 min (payload not listed)" in job["error"]
    assert ["kernels", "push"] not in seen["calls"]
    assert not (env / "staging" / SLUG).exists()


def test_dispatch_only_uploads_pushes_and_records_job(env, monkeypatch):
    video = env / "match.mp4"
    video.write_bytes(b"fake video bytes")
    seen = {}
    monkeypatch.setattr(cvr, "kaggle", _dispatch_fake(seen, exists=False))
    assert cvr.main(["--video", str(video), "--opponent", SLUG, "--dispatch-only", "--opponent-kit", "#000000"]) == 0

    job = json.loads((env / "sources" / SLUG / "vision_job.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(b"fake video bytes").hexdigest()
    assert (job["state"], job["kernel_ref"], job["payload_sha256"]) == ("DISPATCHED", "tivvy/pitchpulse-vision-dorchester-town", digest)
    assert job["payload"] == seen["manifest"]["payload"] == f"payload_{digest[:12]}.mp4"
    assert seen["manifest"]["payload_sha256"] == digest and seen["manifest"]["opponent_kit"] == "#000000"
    assert seen["dataset"]["id"] == "tivvy/pitchpulse-footage-dorchester-town"
    k = seen["kernel"]
    assert (k["enable_gpu"], k["is_private"], k["kernel_type"], k["code_file"]) == (True, True, "script", "kaggle_vision_worker.py")
    assert k["dataset_sources"] == ["tivvy/pitchpulse-footage-dorchester-town", "tivvy/pitchpulse-cv-models"]
    assert "def main()" in seen["worker"] and k["machine_shape"] == "NvidiaTeslaT4"
    assert not (env / "staging" / SLUG).exists()


# ── Output verification ───────────────────────────────────────────────────────

def _doc(sha="a" * 64, **over):
    moments = {m: {n: {"value": None, "n": 0, "iqr": None, "reason": "possession_not_observable_without_ball_tracking"}
                   for n in names} for m, names in cvr.MOMENTS.items()}
    moments["out_of_possession"]["block_height_m"] = {"value": 31.5, "n": 120, "iqr": [28.0, 34.0], "reason": None}
    doc = {"schema_version": cvr.SCHEMA_VERSION, "status": "OK", "footage": "full_wide", "input_sha256": sha, "moments": moments,
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

KAGGLE_ARCH_LIST = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]  # from the dry-run kernel log


@pytest.mark.parametrize("name,capability,ok", [
    ("Tesla P100-PCIE-16GB", (6, 0), False),   # the dry-run failure
    ("Tesla T4", (7, 5), True),
    ("NVIDIA L4", (8, 9), True),               # no sm_89 build; runs on sm_86
    ("Tesla V100", (7, 0), True),
])
def test_gpu_compatibility_guard(name, capability, ok):
    error = worker.gpu_compatibility_error(name, capability, KAGGLE_ARCH_LIST)
    assert (error is None) == ok
    if not ok:
        assert name in error and "6.0" in error and "NvidiaTeslaT4" in error


def test_gpu_guard_rejects_cpu_only_build():
    assert "no CUDA architectures" in worker.gpu_compatibility_error("Tesla T4", (7, 5), [])


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


def test_homography_rejects_collinear_landmarks_as_insufficient_area():
    line = np.array([[105.0, y] for y in (10, 20, 30, 40, 50, 60)])
    assert worker.fit_homography(_image_points(line), line, np.ones(6)) == (None, "drop_insufficient_pitch_area")


def test_homography_rejects_inconsistent_landmarks_as_insufficient_inliers():
    world = BOX_WORLD.copy()
    image = _image_points(world)
    world[:3] += 5.0  # three landmarks labelled 5 m away from where the image shows them → 5/8 inliers
    assert worker.fit_homography(image, world, np.ones(8)) == (None, "drop_insufficient_inliers")


def test_homography_rejects_high_reprojection_error(monkeypatch):
    world = BOX_WORLD.copy()
    image = _image_points(world)
    world[:, 0] += np.where(np.arange(8) % 2 == 0, 0.75, -0.75)  # every label 0.75 m off: all inside 1.0 m RANSAC
    monkeypatch.setattr(worker.cv2, "findHomography", lambda *a, **k: (H_TRUE.copy(), np.ones((8, 1), np.uint8)))
    assert worker.fit_homography(image, world, np.ones(8)) == (None, "drop_reprojection_error_high")


@pytest.mark.parametrize("result", [
    (None, None),
    (np.array([[1.0, 0, 0], [0, 1e-9, 0], [0, 0, 1.0]]), np.ones((8, 1), np.uint8)),   # cond ≈ 1e9
    (np.array([[np.nan, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]), np.ones((8, 1), np.uint8)),
    "cv2_error",
], ids=["none", "ill_conditioned", "non_finite", "opencv_error"])
def test_homography_rejects_singular_matrix(monkeypatch, result):
    def fake(*a, **k):
        if result == "cv2_error":
            raise worker.cv2.error("degenerate input")
        return result

    monkeypatch.setattr(worker.cv2, "findHomography", fake)
    assert worker.fit_homography(_image_points(BOX_WORLD), BOX_WORLD, np.ones(8)) == (None, "drop_singular_matrix")


VIRTUAL_BOX_POINTS = (11, 12, 19, 20)


def test_mask_landmarks_zeroes_only_excluded_numbers():
    conf = np.linspace(0.6, 0.95, 32)
    masked = worker.mask_landmarks(conf, VIRTUAL_BOX_POINTS)
    zero_idx = [n - 1 for n in VIRTUAL_BOX_POINTS]
    assert np.all(masked[zero_idx] == 0.0)
    assert np.allclose(np.delete(masked, zero_idx), np.delete(conf, zero_idx))
    assert conf[10] > 0  # caller's array is not mutated
    assert np.array_equal(worker.mask_landmarks(conf, ()), conf)


def test_load_pitch_config_reads_and_validates_exclusions(tmp_path):
    path = tmp_path / "pitch_config.json"
    path.write_text(json.dumps({"landmarks_m": [[1.0, 1.0]] * 32, "exclude_landmarks": list(VIRTUAL_BOX_POINTS)}), encoding="utf-8")
    world, exclude = worker.load_pitch_config(path)
    assert world.shape == (32, 2) and exclude == VIRTUAL_BOX_POINTS
    path.write_text(json.dumps({"landmarks_m": [[1.0, 1.0]] * 32}), encoding="utf-8")
    assert worker.load_pitch_config(path)[1] == ()
    path.write_text(json.dumps({"landmarks_m": [[1.0, 1.0]] * 32, "exclude_landmarks": [0, 33]}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[0, 33\] outside 1..32"):
        worker.load_pitch_config(path)


def test_excluded_landmarks_are_omitted_from_the_homography_fit():
    world = np.vstack([BOX_WORLD, [[94.0, 34.0], [99.5, 34.0], [90.0, 30.0]]])
    image = _image_points(world)
    labels = world.copy()
    labels[8:] += 5.0  # landmarks 9-11 annotated 5 m from where the model places them, like #11 on the benchmark
    conf = np.ones(11)
    assert worker.fit_homography(image, labels, conf) == (None, "drop_insufficient_inliers")  # 8/11 = 73 %
    H, reason = worker.fit_homography(image, labels, worker.mask_landmarks(conf, (9, 10, 11)))
    assert reason is None
    assert np.abs(worker.project(H, image[:8]) - BOX_WORLD).max() < 0.05


def test_clamp_border_landmarks_zeroes_edge_and_off_screen_keypoints():
    xy = np.array([[960, 540], [960, 1080], [960, 1077], [960, 1075], [-3, 500], [1919, 10]], float)
    out = worker.clamp_border_landmarks(xy, np.ones(6), (1920, 1080))
    assert out.tolist() == [1, 0, 0, 1, 0, 0]  # y=1080 is the Veo #17 clamp; 1075 is 4 px inside
    assert np.array_equal(worker.clamp_border_landmarks(xy[:1], np.ones(1), (1920, 1080)), np.ones(1))


def test_pixel_tolerances_scale_with_frame_height():
    assert worker.scaled_tolerances(1080) == (4, 80)   # validated tuning unchanged
    assert worker.scaled_tolerances(720) == (3, 53)    # runner's > 1.5 GB downsampled payload
    assert worker.scaled_tolerances(100)[0] == 1       # border never collapses to 0
    xy = np.array([[640, 719], [640, 717], [640, 716]], float)  # edge distance h-1-y: 0, 2, 3 px
    assert worker.clamp_border_landmarks(xy, np.ones(3), (1280, 720)).tolist() == [0, 0, 1]  # 3 px border at 720p


# roboflow SoccerPitchConfiguration landmarks (data/staging/models/pitch_config.json)
PITCH_WORLD = np.array([[0, 0], [0, 13.84], [0, 24.84], [0, 43.16], [0, 54.16], [0, 68.0], [5.5, 24.84], [5.5, 43.16],
                        [11.0, 34.0], [16.5, 13.84], [16.5, 24.84], [16.5, 43.16], [16.5, 54.16], [52.5, 0],
                        [52.5, 24.85], [52.5, 43.15], [52.5, 68.0], [88.5, 13.84], [88.5, 24.84], [88.5, 43.16],
                        [88.5, 54.16], [94.0, 34.0], [99.5, 24.84], [99.5, 43.16], [105.0, 0], [105.0, 13.84],
                        [105.0, 24.84], [105.0, 43.16], [105.0, 54.16], [105.0, 68.0], [43.35, 34.0], [61.65, 34.0]], float)
RIGHT_BOX = (18, 21, 22, 23, 24, 26, 27, 28, 29)


def _penalty_area_view(*extra):
    conf = np.zeros(32)
    conf[[n - 1 for n in RIGHT_BOX + extra]] = 1.0
    return _image_points(PITCH_WORLD), conf


def test_isolated_centre_landmark_far_from_penalty_area_fit_is_rejected():
    image, conf = _penalty_area_view(31, 32)
    image[31] += [200.0, 0.0]   # #32 detected on plain grass, like Veo 12:00-14:00
    image[30] += [50.0, 0.0]    # #31 within tolerance: kept
    out = worker.reject_isolated_centre(image, PITCH_WORLD, conf, 1080)
    assert out[31] == 0.0 and out[30] == 1.0
    assert np.array_equal(np.delete(out, 31), np.delete(conf, 31))
    assert conf[31] == 1.0  # caller's array is not mutated


def test_isolated_centre_tolerance_tightens_on_720p_payloads():
    image, conf = _penalty_area_view(32)
    image[31] += [60.0, 0.0]    # inside 80 px at 1080p, outside 53 px at 720p
    assert worker.reject_isolated_centre(image, PITCH_WORLD, conf, 1080)[31] == 1.0
    assert worker.reject_isolated_centre(image, PITCH_WORLD, conf, 720)[31] == 0.0


def test_isolated_centre_landmark_consistent_with_penalty_area_fit_is_kept():
    image, conf = _penalty_area_view(32)
    assert np.array_equal(worker.reject_isolated_centre(image, PITCH_WORLD, conf, 1080), conf)


def test_isolated_centre_check_needs_a_same_end_cluster():
    image, conf = _penalty_area_view(32)
    conf[[n - 1 for n in RIGHT_BOX[3:]]] = 0.0  # 3 box landmarks: no reliable cluster fit
    image[31] += [200.0, 0.0]
    assert np.array_equal(worker.reject_isolated_centre(image, PITCH_WORLD, conf, 1080), conf)


def test_every_rejection_reason_is_a_schema_drop_rule():
    import inspect
    reasons = set(re.findall(r'return None, "(\w+)"', inspect.getsource(worker.fit_homography)))
    assert reasons | {"no_pitch_detection"} == set(cvr.DROP_RULES)


def test_schema_rejects_legacy_drop_buckets():
    doc = _doc()
    doc["frames"]["dropped"] = {"no_pitch_detection": 0, "too_few_landmarks": 3, "degenerate_geometry": 0, "unstable_homography": 0}
    assert any("dropped counts" in e for e in cvr.validate_metrics(doc))
    assert any("schema_version" in e for e in cvr.validate_metrics(_doc(schema_version=1)))


@pytest.mark.parametrize("fps", [25, 29.97, 30, 50, 60])
def test_timestamp_sampling_is_exactly_2fps(fps):
    n_frames = int(round(300 * fps))                      # a 300 s clip
    times = [i / fps for i in worker.sample_indices(fps, n_frames)]
    assert len(times) == 600                              # the old integer step gave 625 at 25 FPS
    assert max(abs(t - k * 0.5) for k, t in enumerate(times)) <= 0.5 / fps + 1e-9


def test_timestamp_sampling_never_repeats_slow_source_frames():
    assert list(worker.sample_indices(1, 10)) == list(range(10))
    with pytest.raises(ValueError, match="frame rate"):
        worker.TimestampSampler(0)


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
    doc = {"schema_version": cvr.SCHEMA_VERSION, "status": "OK", "footage": "full_wide", "input_sha256": "c" * 64, "moments": moments,
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
