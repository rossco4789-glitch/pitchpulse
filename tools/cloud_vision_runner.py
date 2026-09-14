#!/usr/bin/env python3
"""
cloud_vision_runner.py — PitchPulse Kaggle GPU dispatcher and collector
=======================================================================
Sends full-match footage to a free, private Kaggle GPU script kernel and brings back verified
vision metrics. Governed by the CLAUDE.md Section 5 exception: headless batch runs only, no paid
runtimes, no persistent cloud functions. Never run inside Streamlit; it is a batch CLI.

Usage:
    python tools/cloud_vision_runner.py --video match.mp4 --opponent dorchester_town \\
        --footage full_wide --opponent-kit "#000000" --dispatch-only        # upload, push, exit
    python tools/cloud_vision_runner.py --opponent dorchester_town --collect  # check once, fetch if done
    python tools/cloud_vision_runner.py --opponent dorchester_town --collect --wait 90

One-off setup:
    pip install kaggle; API token at ~/.kaggle/kaggle.json; phone-verified Kaggle account (GPU + internet);
    a private dataset <user>/pitchpulse-cv-models holding players.pt, pitch.pt and pitch_config.json.

Flow:
    dispatch  auth → models dataset check → payload (ffmpeg 2 FPS / 720p when > 1.5 GB) → SHA-256
              → private footage dataset (5 attempts, exponential backoff)
              → poll `datasets files` until payload_<sha12> is listed at its byte size (10 min cap)
              → push kernels/templates worker → vision_job.json state DISPATCHED
    collect   kernel status → download output → worker FAILED report | output hash | payload hash
              | 4-moments schema → data/scouting/sources/<slug>/vision_metrics.json

Exit codes:
    0  dispatched (--dispatch-only) or metrics verified and written
    2  setup: kaggle.json, kaggle CLI, ffmpeg, models dataset, arguments, or no dispatched job
    3  remote failure: upload/push/status exhausted retries, kernel error, or worker FAILED report
    4  output rejected by hash or schema checks; nothing written
    5  kernel still queued or running; run --collect again later
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT        = Path(__file__).resolve().parents[1]
SOURCES     = ROOT / "data" / "scouting" / "sources"
STAGING     = ROOT / "data" / "scouting" / "cloud_staging"
WORKER      = Path(__file__).resolve().parent / "templates" / "kaggle_vision_worker.py"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"

DOWNSAMPLE_BYTES = int(1.5 * 1024 ** 3)
MAX_ATTEMPTS     = 5
BACKOFF_BASE_S   = 2.0
POLL_S           = 60
DATASET_READY_S  = 600
READY_POLL_BASE_S = 5
READY_POLL_MAX_S  = 60
MODEL_FILES      = ("players.pt", "pitch.pt", "pitch_config.json")
MACHINE_SHAPE    = "NvidiaTeslaT4"  # Kaggle's default P100 (sm_60) is unsupported by the kernel's PyTorch build
FOOTAGE          = ("full_wide", "highlight")
TERMINAL         = {"complete", "error", "cancelacknowledged", "cancelrequested"}
KAGGLE_CLI_HINT  = "pip install kaggle, then reopen the terminal"
TOKEN_HINT       = "kaggle.com → Settings → API → Create New Token, then save the file as ~/.kaggle/kaggle.json"

# Shared with tools/templates/kaggle_vision_worker.py (a test keeps the two in step)
SCHEMA_VERSION = 1
MIN_SAMPLES    = 30
DROP_RULES     = ("no_pitch_detection", "too_few_landmarks", "degenerate_geometry", "unstable_homography")
MOMENTS = {
    "in_possession":        ("settled_width_m",),
    "out_of_possession":    ("block_height_m", "compactness_depth_m", "compactness_width_m", "line_of_engagement_m"),
    "defensive_transition": ("rest_defense_players",),
    "attacking_transition": ("regain_to_final_third_s",),
}
BOUNDS = {"settled_width_m": 68, "block_height_m": 105, "compactness_depth_m": 105, "compactness_width_m": 68,
          "line_of_engagement_m": 105, "rest_defense_players": 10, "regain_to_final_third_s": 60}


class SetupError(Exception):
    """Local precondition failed — exit 2."""


class RemoteError(Exception):
    """Kaggle call failed after retries — exit 3."""


class OutputError(Exception):
    def __init__(self, state: str, detail: str):
        super().__init__(detail)
        self.state, self.detail = state, detail


# ── Small helpers ─────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def kaggle_slug(name: str) -> str:
    return slugify(name).replace("_", "-")[:30].strip("-")


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_credentials(path: Path | None = None) -> dict:
    path = path or KAGGLE_JSON
    if not path.exists():
        raise SetupError(f"missing {path}. {TOKEN_HINT}")
    try:
        creds = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupError(f"cannot read {path} ({type(exc).__name__}). {TOKEN_HINT}") from exc
    if not (isinstance(creds, dict) and creds.get("username") and creds.get("key")):
        raise SetupError(f"{path} has no username and key. {TOKEN_HINT}")
    return creds


def require_tool(name: str, hint: str) -> None:
    if not shutil.which(name):
        raise SetupError(f"'{name}' not found on PATH: {hint}")


def kaggle(args: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
    """Single seam for every Kaggle CLI call (tests replace this).

    On Windows a piped CLI prints in cp1252 and crashes on kernel-log text such as '⚠️'; forcing UTF-8 in
    the child and decoding with errors="replace" in the parent keeps every call's exit code and output intact.
    """
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return subprocess.run(["kaggle", *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, env=env)


def with_retry(label: str, fn, attempts: int = MAX_ATTEMPTS, base: float = BACKOFF_BASE_S):
    last = ""
    for i in range(attempts):
        try:
            cp = fn()
            if cp.returncode == 0:
                return cp
            last = (cp.stderr or cp.stdout or "").strip()[-400:]
        except (subprocess.TimeoutExpired, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
        if i < attempts - 1:
            delay = base * 2 ** i
            print(f"  [RETRY] {label} {i + 1}/{attempts} failed ({last}); next try in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
    raise RemoteError(f"{label} failed after {attempts} attempts: {last}")


# ── Job state ─────────────────────────────────────────────────────────────────

def job_path(slug: str) -> Path:
    return SOURCES / slug / "vision_job.json"


def read_job(slug: str) -> dict | None:
    try:
        return json.loads(job_path(slug).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_job(slug: str, **fields) -> dict:
    path = job_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    job = {**(read_job(slug) or {}), **fields, "updated_at": _now()}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(job, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return job


# ── Dispatch ──────────────────────────────────────────────────────────────────

def ffmpeg_downsample_cmd(src: Path, dst: Path) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
            "-vf", "fps=2,scale=-2:720", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "24",
            "-movflags", "+faststart", str(dst)]


def prepare_payload(video: Path, stage: Path) -> tuple[Path, bool]:
    """> 1.5 GB → ffmpeg 2 FPS / 720p staging MP4; otherwise hard-link (or copy) the original."""
    if video.stat().st_size > DOWNSAMPLE_BYTES:
        require_tool("ffmpeg", "install ffmpeg and add its bin folder to PATH")
        dst = stage / "payload.mp4"
        cp = subprocess.run(ffmpeg_downsample_cmd(video, dst), capture_output=True, text=True)
        if cp.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
            raise SetupError(f"ffmpeg downsample failed: {(cp.stderr or '').strip()[-400:]}")
        return dst, True
    dst = stage / f"payload{video.suffix.lower()}"
    try:
        os.link(video, dst)
    except OSError:
        shutil.copy2(video, dst)
    return dst, False


def dataset_listing(ref: str) -> dict[str, int] | None:
    """{filename: bytes} from `kaggle datasets files`, or None if the dataset is absent or unreadable.

    `kaggle datasets status` is unusable in CLI 2.2.4 (403 for our own datasets, 404 for public ones),
    so existence and readiness both come from the file listing, which Kaggle fills only after processing.
    """
    try:
        cp = kaggle(["datasets", "files", ref])
    except (subprocess.TimeoutExpired, OSError):
        return None
    if cp.returncode != 0:
        return None
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"^\s*(\S+)\s+(\d+)(?:\s|$)", cp.stdout or "", re.M)}


def check_models(ref: str) -> None:
    listing = dataset_listing(ref)
    hint = f"upload {', '.join(MODEL_FILES)} as the private Kaggle dataset {ref}, or pass --models-dataset"
    if listing is None:
        raise SetupError(f"models dataset {ref} not reachable: {hint}")
    missing = [f for f in MODEL_FILES if f not in listing]
    if missing:
        raise SetupError(f"models dataset {ref} lacks {', '.join(missing)}: {hint}")


def wait_dataset_ready(ref: str, payload: str, size: int) -> None:
    """Poll with exponential backoff (5 s doubling to 60 s) until `payload` is listed at `size` bytes."""
    deadline = time.monotonic() + DATASET_READY_S
    delay = READY_POLL_BASE_S
    while True:
        listing = dataset_listing(ref)
        listed = None if listing is None else listing.get(payload)
        if listed == size:
            return
        if time.monotonic() >= deadline:
            state = ("dataset not listed" if listing is None else "payload not listed" if listed is None
                     else f"payload listed at {listed} bytes, expected {size}")
            raise RemoteError(f"dataset {ref} not ready after {DATASET_READY_S // 60} min ({state})")
        time.sleep(delay)
        delay = min(delay * 2, READY_POLL_MAX_S)


def dispatch(args, creds: dict) -> dict:
    video = Path(args.video)
    if not video.is_file():
        raise SetupError(f"video not found: {video}")
    require_tool("kaggle", KAGGLE_CLI_HINT)

    user, slug, ks = creds["username"], slugify(args.opponent), kaggle_slug(args.opponent)
    models_ref = args.models_dataset or f"{user}/pitchpulse-cv-models"
    dataset_ref, kernel_ref = f"{user}/pitchpulse-footage-{ks}", f"{user}/pitchpulse-vision-{ks}"
    check_models(models_ref)

    stage = STAGING / slug
    shutil.rmtree(stage, ignore_errors=True)
    data_dir, kern_dir = stage / "dataset", stage / "kernel"
    data_dir.mkdir(parents=True)
    kern_dir.mkdir(parents=True)
    try:
        payload, downsampled = prepare_payload(video, data_dir)
        digest = sha256_file(payload)
        # hash in the name: a stale file from the previous dataset version can never satisfy readiness
        payload = payload.rename(payload.with_name(f"payload_{digest[:12]}{payload.suffix}"))
        size = payload.stat().st_size
        (data_dir / "manifest.json").write_text(json.dumps({
            "opponent": slug, "footage": args.footage, "payload": payload.name, "payload_sha256": digest,
            "downsampled": downsampled, "opponent_kit": args.opponent_kit, "created_at": _now(),
        }, indent=2), encoding="utf-8")
        (data_dir / "dataset-metadata.json").write_text(json.dumps({
            "title": f"pitchpulse-footage-{ks}", "id": dataset_ref, "licenses": [{"name": "other"}],
        }, indent=2), encoding="utf-8")

        exists = dataset_listing(dataset_ref) is not None
        push = (["datasets", "version", "-p", str(data_dir), "-m", f"payload {digest[:12]}"] if exists
                else ["datasets", "create", "-p", str(data_dir)])
        print(f"  [DISPATCH] {'versioning' if exists else 'creating'} {dataset_ref} with {payload.name} "
              f"({size / 1e6:.0f} MB, {'downsampled' if downsampled else 'original'})")
        with_retry("dataset upload", lambda: kaggle(push, timeout=4 * 3600))
        wait_dataset_ready(dataset_ref, payload.name, size)

        shutil.copy2(WORKER, kern_dir / WORKER.name)
        (kern_dir / "kernel-metadata.json").write_text(json.dumps({
            "id": kernel_ref, "title": f"pitchpulse-vision-{ks}", "code_file": WORKER.name,
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "machine_shape": MACHINE_SHAPE,
            "dataset_sources": [dataset_ref, models_ref], "competition_sources": [], "kernel_sources": [],
        }, indent=2), encoding="utf-8")
        with_retry("kernel push", lambda: kaggle(["kernels", "push", "-p", str(kern_dir)]))
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    job = write_job(slug, state="DISPATCHED", kernel_ref=kernel_ref, dataset_ref=dataset_ref,
                    models_ref=models_ref, footage=args.footage, payload=payload.name, payload_sha256=digest,
                    downsampled=downsampled, dispatched_at=_now(), error=None)
    print(f"  [DISPATCH] {kernel_ref} pushed; state DISPATCHED → {job_path(slug)}")
    return job


# ── Collect ───────────────────────────────────────────────────────────────────

def kernel_status(ref: str) -> str:
    out = with_retry("kernel status", lambda: kaggle(["kernels", "status", ref])).stdout or ""
    m = re.search(r'status\s+"?(?:KernelWorkerStatus\.)?(\w+)', out, re.I)
    return m.group(1).lower() if m else "unknown"


def validate_metrics(doc) -> list[str]:
    if not isinstance(doc, dict):
        return ["top level is not an object"]
    e = []
    if doc.get("schema_version") != SCHEMA_VERSION:
        e.append(f"schema_version must be {SCHEMA_VERSION}")
    if doc.get("status") != "OK":
        e.append("status must be OK")
    if doc.get("footage") not in FOOTAGE:
        e.append(f"footage must be one of {FOOTAGE}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(doc.get("input_sha256", ""))):
        e.append("input_sha256 is not a SHA-256 hex digest")

    fr = doc.get("frames")
    counts_ok = (isinstance(fr, dict) and isinstance(fr.get("dropped"), dict) and set(fr["dropped"]) == set(DROP_RULES)
                 and all(isinstance(v, int) and v >= 0 for v in [fr.get("sampled"), fr.get("accepted"), *fr["dropped"].values()]))
    if not counts_ok:
        e.append(f"frames needs non-negative sampled, accepted and dropped counts for {DROP_RULES}")
    elif fr["accepted"] + sum(fr["dropped"].values()) != fr["sampled"]:
        e.append("frames: accepted + dropped does not equal sampled")

    moments = doc.get("moments")
    if not isinstance(moments, dict) or set(moments) != set(MOMENTS):
        return e + [f"moments must be exactly {sorted(MOMENTS)}"]
    for moment, names in MOMENTS.items():
        if not isinstance(moments[moment], dict) or set(moments[moment]) != set(names):
            e.append(f"{moment} must hold exactly {names}")
            continue
        for name in names:
            m, where = moments[moment][name], f"{moment}.{name}"
            if not isinstance(m, dict) or set(m) != {"value", "n", "iqr", "reason"}:
                e.append(f"{where} needs value, n, iqr, reason")
                continue
            v, n, iqr = m["value"], m["n"], m["iqr"]
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                e.append(f"{where}: n must be a non-negative integer")
            if v is None:
                if not m["reason"]:
                    e.append(f"{where}: a null value needs a reason")
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= BOUNDS[name]:
                e.append(f"{where}: {v!r} outside 0..{BOUNDS[name]}")
                continue
            if doc.get("footage") == "highlight":
                e.append(f"{where}: highlight footage cannot produce shape metrics")
            if isinstance(n, int) and n < MIN_SAMPLES:
                e.append(f"{where}: value from n={n} (< {MIN_SAMPLES})")
            if not (isinstance(iqr, list) and len(iqr) == 2 and all(isinstance(q, (int, float)) for q in iqr)
                    and iqr[0] <= v <= iqr[1]):
                e.append(f"{where}: iqr must be [low, high] around the value")
    return e


def verify_output(out_dir: Path, expected_sha: str) -> tuple[dict, bytes]:
    path = next(iter(sorted(out_dir.rglob("vision_metrics.json"))), None)
    if path is None:
        raise OutputError("FAILED", "worker wrote no vision_metrics.json; read the kernel log on Kaggle")
    raw = path.read_bytes()
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OutputError("INVALID", f"vision_metrics.json is not JSON ({exc})") from exc
    if isinstance(doc, dict) and doc.get("status") == "FAILED":
        raise OutputError("FAILED", str(doc.get("error", "worker reported FAILED without a traceback"))[-4000:])
    sha_file = path.with_name("vision_metrics.sha256")
    if not sha_file.exists() or sha_file.read_text(encoding="utf-8").strip() != hashlib.sha256(raw).hexdigest():
        raise OutputError("INVALID", "vision_metrics.sha256 missing or does not match the downloaded JSON")
    if doc.get("input_sha256") != expected_sha:
        raise OutputError("INVALID", "worker processed a different payload than the one dispatched (SHA-256 mismatch)")
    errors = validate_metrics(doc)
    if errors:
        raise OutputError("INVALID", "; ".join(errors[:10]))
    return doc, raw


def collect(args) -> int:
    slug = slugify(args.opponent)
    job = read_job(slug)
    if not job or not job.get("kernel_ref") or not job.get("payload_sha256"):
        raise SetupError(f"no dispatched job at {job_path(slug)}: run with --video ... --dispatch-only first")
    require_tool("kaggle", KAGGLE_CLI_HINT)

    ref = job["kernel_ref"]
    deadline = time.monotonic() + max(0, args.wait or 0) * 60
    while True:
        status = kernel_status(ref)
        if status in TERMINAL:
            break
        write_job(slug, state="RUNNING", remote_status=status)
        if time.monotonic() >= deadline:
            print(f"  [COLLECT] {ref} is {status}; run --collect again later")
            return 5
        time.sleep(POLL_S)

    with tempfile.TemporaryDirectory() as tmp:
        try:
            with_retry("kernel output download", lambda: kaggle(["kernels", "output", ref, "-p", tmp], timeout=1800))
            doc, raw = verify_output(Path(tmp), job["payload_sha256"])
        except OutputError as exc:
            write_job(slug, state=exc.state, remote_status=status, error=exc.detail)
            print(f"  [COLLECT] {exc.state}: {exc.detail[-600:]}")
            return 3 if exc.state == "FAILED" else 4
        except RemoteError as exc:
            write_job(slug, state="FAILED", remote_status=status, error=str(exc))
            print(f"  [COLLECT] FAILED: {exc}")
            return 3

    if status != "complete":
        write_job(slug, state="FAILED", remote_status=status, error=f"kernel ended with status {status}")
        print(f"  [COLLECT] FAILED: kernel ended with status {status}")
        return 3

    dest = SOURCES / slug / "vision_metrics.json"
    dest.write_bytes(raw)
    write_job(slug, state="COMPLETE", remote_status=status, output_sha256=hashlib.sha256(raw).hexdigest(),
              collected_at=_now(), error=None)
    fr = doc["frames"]
    print(f"  [COLLECT] COMPLETE: {fr['accepted']}/{fr['sampled']} frames accepted → {dest}")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except AttributeError:
            pass

    ap = argparse.ArgumentParser(description="Dispatch full-match CV to a free Kaggle GPU runner and collect verified metrics.")
    ap.add_argument("--video", type=Path, help="Local MP4 (required unless --collect)")
    ap.add_argument("--opponent", required=True, help="Opponent slug, e.g. dorchester_town")
    ap.add_argument("--footage", choices=FOOTAGE, default="full_wide")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dispatch-only", action="store_true", help="Upload and push, record DISPATCHED, exit 0")
    mode.add_argument("--collect", action="store_true", help="Check the dispatched kernel and fetch verified metrics")
    ap.add_argument("--wait", type=int, default=None,
                    help="Minutes to keep polling (default: 0 with --collect, 240 when dispatch and collect run together)")
    ap.add_argument("--opponent-kit", help="Opponent outfield shirt colour #RRGGBB; without it team metrics stay null")
    ap.add_argument("--models-dataset", help="Kaggle dataset with players.pt, pitch.pt, pitch_config.json "
                                             "(default <user>/pitchpulse-cv-models)")
    args = ap.parse_args(argv)

    if not args.collect and args.video is None:
        print("  [SETUP] --video is required unless --collect is given")
        return 2
    if args.opponent_kit and not re.fullmatch(r"#[0-9A-Fa-f]{6}", args.opponent_kit):
        print("  [SETUP] --opponent-kit must look like #RRGGBB")
        return 2

    try:
        creds = load_credentials()
        if not args.collect:
            dispatch(args, creds)
            if args.dispatch_only:
                return 0
            args.wait = 240 if args.wait is None else args.wait
        return collect(args)
    except SetupError as exc:
        print(f"  [SETUP] {exc}")
        return 2
    except RemoteError as exc:
        write_job(slugify(args.opponent), state="FAILED", error=str(exc))
        print(f"  [REMOTE] {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
