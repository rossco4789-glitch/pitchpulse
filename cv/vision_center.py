"""
cv/vision_center.py — state, process and pitch logic for the Tactical Vision Command Center (app.py tab 6).

Streamlit-free so tools/tests/test_vision_center.py covers it offline. Long work never runs inside the
Streamlit process: dispatch, collect and the pre-flight kit check are detached subprocesses. Their PID records
live under data/scouting/cloud_staging/_ui/<slug>/ (gitignored), so page reruns and server restarts never lose
track of a running job. Job state itself stays owned by tools/cloud_vision_runner.py (vision_job.json).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import psutil

ROOT           = Path(__file__).resolve().parents[1]
STAGING_VIDEOS = ROOT / "data" / "staging"
MODELS_DIR     = STAGING_VIDEOS / "models"
SOURCES        = ROOT / "data" / "scouting" / "sources"
UI_STATE       = ROOT / "data" / "scouting" / "cloud_staging" / "_ui"
RUNNER_PATH    = ROOT / "tools" / "cloud_vision_runner.py"
WORKER_PATH    = ROOT / "tools" / "templates" / "kaggle_vision_worker.py"
PREFLIGHT_PATH = ROOT / "cv" / "vision_preflight.py"

AUTO_COLLECT_S        = 60     # one Kaggle status check per minute while a kernel runs (the runner's POLL_S)
SILHOUETTE_CLEAR      = 0.55   # pre-flight gate: comfortably above the worker's SILHOUETTE_MIN
PREFLIGHT_FRAMES      = 5
PREFLIGHT_MIN_PLAYERS = 10
LOW_BLOCK_MAX_M       = 18.0
MID_BLOCK_MAX_M       = 32.0
PITCH_L, PITCH_W      = 105.0, 68.0

STEPS       = ("IDLE", "DISPATCHED", "RUNNING", "COMPLETE")
JOB_STATES  = ("DISPATCHED", "RUNNING", "COMPLETE", "FAILED", "INVALID")
LIVE_STATES = ("DISPATCHED", "RUNNING")
OOP_KEYS    = ("block_height_m", "compactness_depth_m", "compactness_width_m", "line_of_engagement_m")

_modules: dict[str, object] = {}


# ── Shared modules and small helpers ─────────────────────────────────────────

def _load(name: str, path: Path):
    if name not in _modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _modules[name] = mod
    return _modules[name]


def runner():
    return _load("cloud_vision_runner", RUNNER_PATH)


def worker():
    return _load("kaggle_vision_worker", WORKER_PATH)


def slugify(name: str) -> str:
    return runner().slugify(name or "")


def display_name(slug: str) -> str:
    return slug.replace("_", " ").title()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_time(iso: str | None) -> str:
    """'2026-09-15T14:30:28+00:00' → '15 Sep 14:30 UTC'."""
    try:
        return datetime.fromisoformat(iso).astimezone(timezone.utc).strftime("%d %b %H:%M UTC")
    except (TypeError, ValueError):
        return "—"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ── Staged footage ───────────────────────────────────────────────────────────

def human_size(n: float) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f} MB"
    return f"{n / 1024:.0f} KB"


def _clock(seconds: float | None) -> str:
    if not seconds:
        return "—"
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()
    return {"width": w, "height": h, "duration_s": frames / fps if fps > 0 and frames > 0 else None}


def staged_videos(folder: Path | None = None) -> list[dict]:
    """Every data/staging/*.mp4, newest first, with size, resolution and duration badges."""
    folder = folder or STAGING_VIDEOS
    out = []
    for p in sorted(folder.glob("*.mp4"), key=lambda q: q.stat().st_mtime, reverse=True):
        size, info = p.stat().st_size, probe_video(p)
        out.append({"path": str(p), "name": p.name, "bytes": size, "size": human_size(size),
                    "height": info["height"], "resolution": f"{info['height']}p" if info["height"] else "unreadable",
                    "duration": _clock(info["duration_s"]), "downsampled": size > runner().DOWNSAMPLE_BYTES})
    return out


def staged_name(filename: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).strip("._") or "footage"
    return stem if stem.lower().endswith(".mp4") else f"{stem}.mp4"


# ── Detached processes ───────────────────────────────────────────────────────

def ui_dir(slug: str) -> Path:
    return UI_STATE / slug


def _proc_path(slug: str, lane: str) -> Path:
    return ui_dir(slug) / f"{lane}_process.json"


def launch(slug: str, lane: str, action: str, argv: list[str]) -> dict:
    """Start `python <argv>` detached from Streamlit; record PID and create time so reruns can find it."""
    d = ui_dir(slug)
    d.mkdir(parents=True, exist_ok=True)
    log = d / f"{action}.log"
    detach = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
              if os.name == "nt" else {"start_new_session": True})
    with open(log, "wb") as fh:
        proc = subprocess.Popen([sys.executable, *argv], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=fh,
                                stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"}, **detach)
    try:
        created = psutil.Process(proc.pid).create_time()
    except psutil.Error:
        created = None
    rec = {"lane": lane, "action": action, "pid": proc.pid, "create_time": created, "log": str(log),
           "started_at": _now_iso(), "started_ts": time.time()}
    _write_json(_proc_path(slug, lane), rec)
    return rec


def _alive(rec: dict) -> bool:
    """True only for the same process: a recycled PID has a different create time."""
    try:
        p = psutil.Process(rec["pid"])
        return (p.is_running() and p.status() != psutil.STATUS_ZOMBIE and rec.get("create_time") is not None
                and abs(p.create_time() - rec["create_time"]) < 1.0)
    except (psutil.Error, KeyError, TypeError):
        return False


def process_record(slug: str, lane: str) -> dict | None:
    return _read_json(_proc_path(slug, lane)) if slug else None


def active_process(slug: str, lane: str) -> dict | None:
    rec = process_record(slug, lane)
    return rec if rec and _alive(rec) else None


def last_log_line(rec: dict | None) -> str:
    try:
        lines = [ln.strip() for ln in Path(rec["log"]).read_text(encoding="utf-8", errors="replace").splitlines()]
    except (OSError, KeyError, TypeError):
        return ""
    lines = [ln for ln in lines if ln]
    return re.sub(r"^\[[A-Z]+\]\s*", "", lines[-1]) if lines else ""


def dispatch_argv(video: str | Path, slug: str, kit: str, footage: str = "full_wide") -> list[str]:
    return [str(RUNNER_PATH), "--video", str(video), "--opponent", slug, "--footage", footage,
            "--opponent-kit", kit.upper(), "--dispatch-only"]


def collect_argv(slug: str) -> list[str]:
    return [str(RUNNER_PATH), "--opponent", slug, "--collect"]   # no --wait: one status check, then exit


def preflight_result_path(slug: str) -> Path:
    return ui_dir(slug) / "preflight_result.json"


def preflight_argv(video: str | Path, kit: str, out: Path) -> list[str]:
    return [str(PREFLIGHT_PATH), "--video", str(video), "--kit", kit.upper(), "--out", str(out)]


def start_dispatch(slug: str, video: str | Path, kit: str, footage: str = "full_wide") -> dict | None:
    """None when the pipeline lane is busy or a kernel is already live for this opponent."""
    if not slug or active_process(slug, "pipeline") or (read_job(slug) or {}).get("state") in LIVE_STATES:
        return None
    return launch(slug, "pipeline", "dispatch", dispatch_argv(video, slug, kit, footage))


def start_collect(slug: str) -> dict | None:
    if not slug or active_process(slug, "pipeline"):
        return None
    return launch(slug, "pipeline", "collect", collect_argv(slug))


def start_preflight(slug: str, video: str | Path, kit: str) -> dict | None:
    if not slug or active_process(slug, "preflight"):
        return None
    out = preflight_result_path(slug)
    out.unlink(missing_ok=True)
    return launch(slug, "preflight", "preflight", preflight_argv(video, kit, out))


def read_preflight(slug: str) -> dict | None:
    return _read_json(preflight_result_path(slug)) if slug else None


# ── Pipeline state ───────────────────────────────────────────────────────────

def read_job(slug: str) -> dict | None:
    return _read_json(SOURCES / slug / "vision_job.json") if slug else None


def pipeline_state(job: dict | None, proc: dict | None) -> str:
    """IDLE → DISPATCHING → DISPATCHED → RUNNING → COMPLETE, or FAILED / INVALID. A collect check keeps the job state."""
    if proc and proc.get("action") == "dispatch":
        return "DISPATCHING"   # vision_job.json still holds the previous run until the upload finishes
    state = (job or {}).get("state")
    return state if state in JOB_STATES else "IDLE"


def step_index(state: str) -> int | None:
    return {"IDLE": 0, "DISPATCHING": 1, "DISPATCHED": 1, "RUNNING": 2, "COMPLETE": 3}.get(state)


def should_auto_collect(job: dict | None, proc: dict | None, last: dict | None, now: float | None = None) -> bool:
    if proc is not None or (job or {}).get("state") not in LIVE_STATES:
        return False
    started = (last or {}).get("started_ts") if (last or {}).get("action") == "collect" else None
    return started is None or (time.time() if now is None else now) - started >= AUTO_COLLECT_S


ANALYSIS_PREP_S = 600     # payload preparation and upload budget for a full match
ANALYSIS_RUN_S  = 1500    # GPU run budget (the Veo test runs took 20-24 min dispatch to collect)


def live_opponents() -> list[str]:
    """Opponents whose footage is uploading or whose analysis is still running."""
    slugs = {p.parent.name for p in SOURCES.glob("*/vision_job.json")} if SOURCES.exists() else set()
    slugs |= {p.parent.name for p in UI_STATE.glob("*/pipeline_process.json")} if UI_STATE.exists() else set()
    return [s for s in sorted(slugs)
            if pipeline_state(read_job(s), active_process(s, "pipeline")) in ("DISPATCHING", *LIVE_STATES)]


def analysis_progress(job: dict | None, proc: dict | None, now: float | None = None) -> tuple[float, str]:
    """Time-based progress for a coach-facing bar: preparation fills 5-30 %, analysis 30-95 %, never 100 % until collected."""
    now = time.time() if now is None else now
    state = pipeline_state(job, proc)
    if state == "DISPATCHING":
        elapsed = now - (proc or {}).get("started_ts", now)
        return round(0.05 + 0.25 * min(1.0, elapsed / ANALYSIS_PREP_S), 3), "Preparing match footage…"
    if state in LIVE_STATES:
        try:
            started = datetime.fromisoformat(job["dispatched_at"]).timestamp()
        except (KeyError, TypeError, ValueError):
            started = now
        return round(0.30 + 0.65 * min(1.0, max(0.0, now - started) / ANALYSIS_RUN_S), 3), "Analyzing match footage…"
    return (1.0, "Report ready") if state == "COMPLETE" else (0.0, "")


def seconds_to_next_check(last: dict | None, now: float | None = None) -> int:
    started = (last or {}).get("started_ts") if (last or {}).get("action") == "collect" else None
    if started is None:
        return 0
    return max(0, int(AUTO_COLLECT_S - ((time.time() if now is None else now) - started)))


# ── Badges and plain-language reasons ────────────────────────────────────────

def block_badge(height_m: float | None) -> str | None:
    if height_m is None:
        return None
    return "Low Block" if height_m < LOW_BLOCK_MAX_M else "Mid Block" if height_m <= MID_BLOCK_MAX_M else "High Block"


def silhouette_badge(score: float | None) -> tuple[str, str]:
    if score is None:
        return "grey", "Not measured"
    if score >= SILHOUETTE_CLEAR:
        return "green", "Clear kit separation"
    if score >= worker().SILHOUETTE_MIN:
        return "amber", "Marginal kit separation"
    return "red", "Kits overlap: team metrics stay empty"


def reason_text(reason: str | None) -> str:
    if not reason:
        return ""
    fixed = {
        "opponent_kit_not_given": "No opponent kit was set at dispatch.",
        "highlight_footage_invalid_for_shape": "Highlight clips cannot show a settled shape.",
        "possession_not_observable_without_ball_tracking": "Needs ball tracking.",
        "not_reported": "Not in this metrics file.",
    }
    if reason in fixed:
        return fixed[reason]
    if m := re.match(r"insufficient_coverage \(n=(\d+), windows=(\d+)\)", reason):
        return f"{m[1]} settled frames in {m[2]} window(s); needs {worker().MIN_SAMPLES} across 2."
    if m := re.match(r"team_split_ambiguous \(silhouette ([\d.]+)\)", reason):
        return f"Kits overlapped (silhouette {m[1]}). Re-check the opponent kit colour."
    if m := re.match(r"too_few_player_detections \((\d+)\)", reason):
        return f"Only {m[1]} player detections carried a kit colour."
    return reason.replace("_", " ")


# ── Collected metrics ────────────────────────────────────────────────────────

def scouted_opponents() -> list[str]:
    return sorted(p.parent.name for p in SOURCES.glob("*/vision_metrics.json")) if SOURCES.exists() else []


def match_summary(slug: str) -> dict | None:
    path = SOURCES / slug / "vision_metrics.json"
    doc = _read_json(path)
    if not doc:
        return None
    job = read_job(slug) or {}
    frames, split = doc.get("frames") or {}, doc.get("team_split") or {}
    sampled, accepted = frames.get("sampled") or 0, frames.get("accepted") or 0
    stamp = job.get("collected_at") or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    oop = (doc.get("moments") or {}).get("out_of_possession") or {}
    missing = {"value": None, "n": 0, "iqr": None, "reason": "not_reported"}
    return {"slug": slug, "opponent": display_name(slug), "date": stamp[:10], "footage": doc.get("footage"),
            "sampled": sampled, "accepted": accepted, "acceptance": accepted / sampled if sampled else None,
            "silhouette": split.get("silhouette"), "settled_frames": split.get("settled_frames"),
            "players": split.get("players"), "rules": doc.get("rules") or {},
            "oop": {k: oop.get(k) or dict(missing) for k in OOP_KEYS}}


def history_rows() -> list[dict]:
    rows = []
    for slug in scouted_opponents():
        s = match_summary(slug)
        if not s:
            continue
        v = {k: s["oop"][k].get("value") for k in OOP_KEYS}
        rows.append({"slug": slug, "Opponent": s["opponent"], "Date": s["date"],
                     "Mapped %": round(100 * s["acceptance"], 1) if s["acceptance"] is not None else None,
                     "Silhouette": s["silhouette"], "Block height m": v["block_height_m"],
                     "Block": block_badge(v["block_height_m"]) or "—", "Depth m": v["compactness_depth_m"],
                     "Width m": v["compactness_width_m"], "LoE m": v["line_of_engagement_m"],
                     "Settled frames": s["settled_frames"]})
    return sorted(rows, key=lambda r: r["Date"], reverse=True)


def shape_geometry(oop: dict) -> dict | None:
    """Median shape on a pitch where the opponent defends x = 0. Lateral position is not measured: the zone is centred."""
    vals = {k: (oop.get(k) or {}).get("value") for k in OOP_KEYS}
    if any(v is None for v in vals.values()):
        return None
    front = float(min(PITCH_L, max(0.0, vals["line_of_engagement_m"])))
    depth, width = float(vals["compactness_depth_m"]), float(vals["compactness_width_m"])
    back, half = max(0.0, front - depth), min(PITCH_W / 2, width / 2)
    y0, y1 = PITCH_W / 2 - half, PITCH_W / 2 + half
    iqr = (oop["block_height_m"] or {}).get("iqr")
    return {"block": float(vals["block_height_m"]), "block_iqr": tuple(iqr) if iqr else None, "loe": front,
            "depth": depth, "width": width, "zone": [(back, y0), (front, y0), (front, y1), (back, y1)],
            "centroid": ((back + front) / 2, PITCH_W / 2)}


def shape_read(summary: dict) -> str:
    """Coach-facing one-paragraph read: diagnosis with numbers, confidence, then the lever."""
    geo = shape_geometry(summary["oop"])
    if geo is None:
        return ""
    badge = block_badge(geo["block"])
    text = [f"{badge}: deepest four at {geo['block']:.1f} m, highest outfielder at {geo['loe']:.1f} m, "
            f"unit {geo['depth']:.1f} m deep and {geo['width']:.1f} m wide."]
    acc = summary.get("acceptance")
    if acc is not None and acc < 0.6:
        text.append(f"Low confidence: {100 * acc:.1f}% of frames mapped to the pitch.")
    if geo["width"] < 30:
        text.append(f"They cover {geo['width']:.0f} m of the 68 m width: switch play early to the far full-back.")
    text.append({"Low Block": "Commit a runner to the edge of the box for cut-backs.",
                 "Mid Block": "Play through the Half-Spaces behind their midfield line.",
                 "High Block": f"Run in behind: their deepest four hold {geo['block']:.0f} m from goal."}[badge])
    return " ".join(text)


# ── Kit separation (pre-flight) ──────────────────────────────────────────────

def kit_hex(feature) -> str:
    """Weighted-Lab kit feature back to a display hex."""
    lab = np.asarray(feature, float) / worker().KIT_LAB_WEIGHTS
    b, g, r = cv2.cvtColor(np.uint8([[np.clip(np.round(lab), 0, 255)]]), cv2.COLOR_LAB2BGR)[0, 0]
    return f"#{r:02X}{g:02X}{b:02X}"


def separation_report(features, kit: str) -> dict:
    W = worker()
    X = np.asarray(features, float).reshape(-1, 3)
    if len(X) < PREFLIGHT_MIN_PLAYERS:
        return {"status": "insufficient", "players": int(len(X)), "kit": kit.upper()}
    labels, centres = W.two_means(X)
    counts = np.bincount(labels, minlength=2)
    score = W.silhouette(X, labels) if counts.min() > 0 else 0.0
    dist = np.sqrt(((centres - W.hex_to_kit(kit)) ** 2).sum(1))
    opp = int(np.argmin(dist))
    side = lambda k: {"hex": kit_hex(centres[k]), "players": int(counts[k]), "distance": round(float(dist[k]), 1)}
    return {"status": "ok", "players": int(len(X)), "silhouette": round(float(score), 3), "kit": kit.upper(),
            "opponent": side(opp), "other": side(1 - opp)}
