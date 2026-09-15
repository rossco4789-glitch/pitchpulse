#!/usr/bin/env python3
"""
kaggle_vision_worker.py — PitchPulse remote CV worker (Kaggle GPU script kernel)
================================================================================
Pushed verbatim by tools/cloud_vision_runner.py. The kernel holds only this file, so it imports
nothing from PitchPulse; schema constants are mirrored from the runner and a local test keeps them equal.

Inputs (/kaggle/input/**):
    manifest.json, payload.*            footage dataset written by the runner
    players.pt                          YOLO detector with classes including "player" and "goalkeeper"
    pitch.pt                            YOLO pose model, one keypoint per pitch landmark
    pitch_config.json                   {"landmarks_m": [[x, y], ...]} in the 105×68 m frame, index-aligned with pitch.pt,
                                        optional "exclude_landmarks": [1-based numbers] whose confidence is zeroed

Outputs (/kaggle/working):
    vision_metrics.json                 4-moments metrics (see tools/cloud_vision_runner.py MOMENTS)
    vision_metrics.sha256               SHA-256 of the exact JSON bytes

Rejection rules, checked in this order (a frame that fails is counted under frames.dropped, never estimated):
    no_pitch_detection            pose model found no pitch
    too_few_landmarks             < 6 landmarks at confidence ≥ 0.5 (H needs 4; 6 leaves redundancy to test the fit)
    drop_insufficient_pitch_area  landmark hull < 150 m² (near-collinear points give an ill-posed H)
    drop_singular_matrix          no H, non-finite H, OpenCV error, or cond(H) > 1e7
    drop_insufficient_inliers     RANSAC (1.0 m threshold) keeps < 80 % of landmarks or < 6
    drop_reprojection_error_high  inlier RMSE > 0.5 m
    Players projected outside the landmark hull (+5 m) are discarded.

Sampling: the frame nearest each 0.5 s target is kept (exactly 2.0 FPS for 25, 29.97, 30, 50 and 60 FPS sources).

Contract:
    Frames are read at 2 FPS and inferred in 500-frame chunks with torch.cuda.empty_cache() after each;
    a CUDA OOM retries the batch one frame at a time. Any unhandled exception is written as
    {"status": "FAILED", "error": traceback} and the script exits 0, so the collector reads the failure
    instead of waiting on a silent timeout.
"""

import hashlib
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

INPUT  = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working")
PINNED = "ultralytics==8.4.142"

PITCH_L, PITCH_W = 105.0, 68.0
SAMPLE_FPS       = 2
CHUNK            = 500
BATCH            = 16
IMGSZ            = 1280
KP_CONF          = 0.5
MIN_LANDMARKS    = 6
MIN_HULL_M2      = 150.0
MAX_RMSE_M       = 0.5
RANSAC_THRESHOLD_M = 1.0    # wider than MAX_RMSE_M so keypoint noise and systematic fit error land in separate buckets
MIN_INLIER_RATIO = 0.8
MAX_COND         = 1e7
REFERENCE_HEIGHT_PX = 1080.0  # pixel tolerances below are tuned at this frame height and scaled to the payload
BORDER_PX        = 4        # keypoints on or outside the frame edge are off-screen guesses (Veo #17 at y=1080, 64/64)
ISOLATED_CENTRE_PX = 80     # broadcast: valid #32 at 79 px from a box-cluster fit; Veo: 182/184 grass #31/#32 > 80 px
CENTRE_CIRCLE    = (31, 32)
END_ZONE_X_M     = 16.5     # penalty-area depth: landmarks within it of either goal line form a same-end cluster
HULL_MARGIN_M    = 5.0
SETTLED_MIN_PLAYERS = 6    # follow-cam rarely frames 8 opponents (Veo 15 Sep: 12 settled frames); Block height uses the deepest 4
SETTLED_MAX_DEPTH_M = 40.0
SETTLED_RUN      = 8        # consecutive sampled frames = 4 s at 2 FPS
SETTLED_MAX_GAP_S = 2.0     # a 1-2 frame homography drop (42.7 % acceptance on Veo) does not break a settled run
WINDOW_S         = 300      # attack direction and coverage are judged per 5-minute window
SILHOUETTE_MIN   = 0.5
KIT_LAB_WEIGHTS  = np.array([0.2, 1.5, 1.0])  # L*, a*, b*: kit clustering keyed to hue, not lighting

# Mirrored from tools/cloud_vision_runner.py
SCHEMA_VERSION = 2
MIN_SAMPLES    = 30
DROP_RULES     = ("no_pitch_detection", "too_few_landmarks", "drop_insufficient_pitch_area", "drop_singular_matrix",
                  "drop_insufficient_inliers", "drop_reprojection_error_high")
MOMENTS = {
    "in_possession":        ("settled_width_m",),
    "out_of_possession":    ("block_height_m", "compactness_depth_m", "compactness_width_m", "line_of_engagement_m"),
    "defensive_transition": ("rest_defense_players",),
    "attacking_transition": ("regain_to_final_third_s",),
}


# ── Geometry ──────────────────────────────────────────────────────────────────

def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


def fit_homography(image_xy, world_xy, conf) -> tuple[np.ndarray | None, str | None]:
    """Image → pitch-metre homography, or (None, drop rule). Never raises on bad geometry."""
    keep = np.asarray(conf) >= KP_CONF
    img = np.asarray(image_xy, np.float32)[keep]
    world = np.asarray(world_xy, np.float32)[keep]
    if len(img) < MIN_LANDMARKS:
        return None, "too_few_landmarks"
    if cv2.contourArea(cv2.convexHull(world)) < MIN_HULL_M2:
        return None, "drop_insufficient_pitch_area"
    try:
        H, mask = cv2.findHomography(img, world, cv2.RANSAC, RANSAC_THRESHOLD_M)
    except cv2.error:
        return None, "drop_singular_matrix"
    if H is None or mask is None or not np.all(np.isfinite(H)) or np.linalg.cond(H) > MAX_COND:
        return None, "drop_singular_matrix"
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < MIN_LANDMARKS or inliers.mean() < MIN_INLIER_RATIO:
        return None, "drop_insufficient_inliers"
    rmse = float(np.sqrt(np.mean(np.sum((project(H, img[inliers]) - world[inliers]) ** 2, axis=1))))
    if rmse > MAX_RMSE_M:
        return None, "drop_reprojection_error_high"
    return H, None


def load_pitch_config(path: Path) -> tuple[np.ndarray, tuple[int, ...]]:
    """Landmark world coordinates and the validated 1-based landmark numbers to exclude."""
    cfg = json.loads(path.read_text(encoding="utf-8"))
    world = np.asarray(cfg["landmarks_m"], float)
    if world.ndim != 2 or world.shape[1] != 2 or not ((world >= 0) & (world <= [PITCH_L, PITCH_W])).all():
        raise ValueError("pitch_config.json landmarks_m must be [[x, y], ...] inside 105×68 m")
    exclude = tuple(int(n) for n in cfg.get("exclude_landmarks", []))
    bad = [n for n in exclude if not 1 <= n <= len(world)]
    if bad:
        raise ValueError(f"pitch_config.json exclude_landmarks {bad} outside 1..{len(world)}")
    return world, exclude


def mask_landmarks(conf, exclude) -> np.ndarray:
    """Zero the confidence of excluded landmarks (1-based) so no count, hull, fit or residual uses them.

    The benchmark diagnostic (14 Sep 2026) found landmark 11 — penalty line at goal-box width, no painted
    intersection — outside RANSAC consensus in 44/44 detections; 12, 19 and 20 are its mirror points.
    """
    out = np.array(conf, dtype=float, copy=True)
    if exclude:
        out[[n - 1 for n in exclude]] = 0.0
    return out


def scaled_tolerances(image_h: float) -> tuple[int, int]:
    """(border px, isolated-centre px) for a frame `image_h` pixels tall: 1080p → (4, 80), 720p → (3, 53)."""
    scale = image_h / REFERENCE_HEIGHT_PX
    return max(1, int(round(BORDER_PX * scale))), int(round(ISOLATED_CENTRE_PX * scale))


def clamp_border_landmarks(image_xy, conf, image_wh) -> np.ndarray:
    """Zero the confidence of keypoints within the height-scaled BORDER_PX of, or outside, the image edge.

    The Veo follow-cam diagnostic (15 Sep 2026) found landmark 17 pinned to the bottom edge in 64/64 detections:
    the model clamps an off-screen point to the frame rather than dropping it.
    """
    xy = np.asarray(image_xy, float)
    w, h = image_wh
    edge = np.minimum.reduce([xy[:, 0], xy[:, 1], w - 1 - xy[:, 0], h - 1 - xy[:, 1]])
    out = np.array(conf, dtype=float, copy=True)
    out[edge < scaled_tolerances(h)[0]] = 0.0
    return out


def reject_isolated_centre(image_xy, world_xy, conf, image_h: float) -> np.ndarray:
    """Zero centre-circle landmarks more than the height-scaled ISOLATED_CENTRE_PX from where a same-end
    penalty-area fit places them.

    On Veo footage facing a penalty area the model reports #32 on plain grass at ~0.95 confidence; confidence
    thresholds cannot catch it, but a world → pixel fit from ≥ 4 box/goal-line landmarks at that end can.
    Veo 12:00-14:00: 44 → 79 of 241 frames accepted, leave-one-out median 0.47 m; broadcast 0bfacc_0 unchanged.
    """
    xy, world = np.asarray(image_xy, np.float32), np.asarray(world_xy, np.float32)
    ok = np.asarray(conf) >= KP_CONF
    out = np.array(conf, dtype=float, copy=True)
    tolerance = scaled_tolerances(image_h)[1]
    for end in (world[:, 0] <= END_ZONE_X_M, world[:, 0] >= PITCH_L - END_ZONE_X_M):
        idx = np.where(ok & end)[0]
        if len(idx) < 4 or cv2.contourArea(cv2.convexHull(world[idx])) < 50:
            continue
        try:
            G, mask = cv2.findHomography(world[idx], xy[idx], cv2.RANSAC, 10.0)
        except cv2.error:
            continue
        if G is None or mask is None or mask.sum() < 4 or not np.all(np.isfinite(G)) or np.linalg.cond(G) > MAX_COND:
            continue
        for n in CENTRE_CIRCLE:
            if ok[n - 1] and np.linalg.norm(project(G, world[n - 1:n])[0] - xy[n - 1]) > tolerance:
                out[n - 1] = 0.0
    return out


def gpu_compatibility_error(name: str, capability: tuple[int, int], arch_list: list[str]) -> str | None:
    """None if this PyTorch build can run on the GPU, else an explicit reason.

    A build compiled for sm_XY runs on any device with the same major version and minor ≥ Y
    (an L4 at 8.9 runs on sm_86); a P100 at 6.0 has no sm_6x build and fails on the first tensor op.
    """
    major, minor = capability
    built = sorted({(int(a[3:-1]), int(a[-1])) for a in arch_list if re.fullmatch(r"sm_\d{2,3}", a)})
    if any(bm == major and bn <= minor for bm, bn in built):
        return None
    return (f"GPU {name} has CUDA capability {major}.{minor}, but this PyTorch build supports "
            f"{', '.join(f'{m}.{n}' for m, n in built) or 'no CUDA architectures'}. "
            f"Pin a T4 with \"machine_shape\": \"NvidiaTeslaT4\" in kernel-metadata.json.")


def kit_feature(bgr_pixels) -> np.ndarray:
    """Median CIE-Lab of BGR pixels weighted by KIT_LAB_WEIGHTS, the space teams are clustered in.

    Plain Lab let lightness dominate: dark navy and shadowed red on 720p Veo footage clustered at silhouette
    0.494 (89 players). Down-weighting L* and up-weighting the red/green a* axis gave 0.671 on the same sample.
    """
    px = np.asarray(bgr_pixels, np.uint8).reshape(-1, 1, 3)
    return np.median(cv2.cvtColor(px, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float), axis=0) * KIT_LAB_WEIGHTS


def hex_to_kit(hex_colour: str) -> np.ndarray:
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return kit_feature([[b, g, r]])


def torso_colour(frame, x1, y1, x2, y2) -> list[float] | None:
    """kit_feature of the torso (15-50 % of box height, 25-75 % of width).

    A narrower chest crop and a turf mask were tried on the 89-player Veo sample: turf share was 0 % and the
    smaller crop lowered silhouette to 0.43 (players are 33-71 px tall at 720p), so the wider window stays.
    """
    h, w = y2 - y1, x2 - x1
    crop = frame[int(y1 + 0.15 * h):int(y1 + 0.5 * h), int(x1 + 0.25 * w):int(x2 - 0.25 * w)]
    if crop.size == 0:
        return None
    return kit_feature(crop.reshape(-1, 3)).tolist()


# ── Team split and metrics (pure numpy; testable without a GPU) ──────────────

def two_means(X: np.ndarray, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    c0 = X[np.argmax(((X - X.mean(0)) ** 2).sum(1))]
    c1 = X[np.argmax(((X - c0) ** 2).sum(1))]
    centres = np.array([c0, c1])
    labels = np.zeros(len(X), int)
    for _ in range(iters):
        labels = np.argmin(((X[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
        new = np.array([X[labels == k].mean(0) if np.any(labels == k) else centres[k] for k in (0, 1)])
        if np.allclose(new, centres):
            break
        centres = new
    return labels, centres


def silhouette(X: np.ndarray, labels: np.ndarray, cap: int = 2000) -> float:
    idx = np.linspace(0, len(X) - 1, min(cap, len(X))).astype(int)
    X, labels = X[idx], labels[idx]
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    scores = []
    for i in range(len(X)):
        same, other = labels == labels[i], labels != labels[i]
        same[i] = False
        if not same.any() or not other.any():
            continue
        a, b = D[i, same].mean(), D[i, other].mean()
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _entry(values, windows, reason=None) -> dict:
    if reason:
        return {"value": None, "n": 0, "iqr": None, "reason": reason}
    n, w = len(values), len(windows)
    if n < MIN_SAMPLES or w < 2:
        return {"value": None, "n": n, "iqr": None, "reason": f"insufficient_coverage (n={n}, windows={w})"}
    a = np.asarray(values, float)
    return {"value": round(float(np.median(a)), 2), "n": n,
            "iqr": [round(float(np.percentile(a, 25)), 2), round(float(np.percentile(a, 75)), 2)], "reason": None}


def settled_frames(series: list[tuple[float, np.ndarray]]) -> list[tuple[float, np.ndarray]]:
    """Frames where the (normalised) opponent sits in its own half, ≥ SETTLED_MIN_PLAYERS outfielders, depth ≤ 40 m,
    for ≥ SETTLED_RUN sampled frames with no gap over SETTLED_MAX_GAP_S."""
    ok = [(t, p) for t, p in series
          if len(p) >= SETTLED_MIN_PLAYERS and p[:, 0].mean() < PITCH_L / 2 and np.ptp(p[:, 0]) <= SETTLED_MAX_DEPTH_M]
    runs, cur, gap = [], [], SETTLED_MAX_GAP_S
    for t, p in ok:
        if cur and t - cur[-1][0] > gap:
            runs.append(cur)
            cur = []
        cur.append((t, p))
    runs.append(cur)
    return [fp for run in runs if len(run) >= SETTLED_RUN for fp in run]


def compute_metrics(rows: list[dict], footage: str, opponent_kit: str | None) -> tuple[dict, dict]:
    no_ball = "possession_not_observable_without_ball_tracking"
    moments = {m: {n: _entry([], set(), no_ball) for n in names} for m, names in MOMENTS.items()}
    oop = moments["out_of_possession"]
    info = {"players": 0, "silhouette": None}

    def null_oop(reason):
        for name in oop:
            oop[name] = _entry([], set(), reason)
        return moments, info

    if footage == "highlight":
        return null_oop("highlight_footage_invalid_for_shape")
    if not opponent_kit:
        return null_oop("opponent_kit_not_given")
    players = [r for r in rows if r["role"] == "player" and r.get("kit") is not None]
    info["players"] = len(players)
    if len(players) < 2 * MIN_SAMPLES:
        return null_oop(f"too_few_player_detections ({len(players)})")

    X = np.asarray([r["kit"] for r in players], float)
    labels, centres = two_means(X)
    score = silhouette(X, labels)
    info["silhouette"] = round(score, 3)
    if min(np.bincount(labels, minlength=2)) == 0 or score < SILHOUETTE_MIN:
        return null_oop(f"team_split_ambiguous (silhouette {score:.2f})")
    opp = int(np.argmin(((centres - hex_to_kit(opponent_kit)) ** 2).sum(1)))

    frames: dict[float, tuple[list, list]] = {}
    for r, lab in zip(players, labels):
        frames.setdefault(r["t"], ([], []))[0 if lab == opp else 1].append((r["x"], r["y"]))

    windows: dict[int, tuple[list, list]] = {}
    for t, (mine, theirs) in frames.items():
        if mine and theirs:
            acc = windows.setdefault(int(t // WINDOW_S), ([], []))
            acc[0].extend(x for x, _ in mine)
            acc[1].extend(x for x, _ in theirs)
    flip = {w: np.mean(a) > np.mean(b) for w, (a, b) in windows.items()}  # opponent defends x = 105 → rotate 180°

    series = []
    for t in sorted(frames):
        w, mine = int(t // WINDOW_S), frames[t][0]
        if w not in flip or not mine:
            continue
        p = np.asarray(mine, float)
        if flip[w]:
            p = np.column_stack([PITCH_L - p[:, 0], PITCH_W - p[:, 1]])
        series.append((t, p))

    settled = settled_frames(series)
    covered = {int(t // WINDOW_S) for t, _ in settled}
    values = {name: [] for name in oop}
    for _, p in settled:
        xs = np.sort(p[:, 0])
        values["block_height_m"].append(xs[:4].mean())
        values["compactness_depth_m"].append(np.ptp(xs))
        values["compactness_width_m"].append(np.ptp(p[:, 1]))
        values["line_of_engagement_m"].append(xs[-1])
    for name, v in values.items():
        oop[name] = _entry(v, covered)
    info["settled_frames"] = len(settled)
    info["direction_method"] = "per 5-minute window, team with the lower mean x defends x = 0"
    return moments, info


# ── Video and inference ───────────────────────────────────────────────────────

class TimestampSampler:
    """Keeps the frame nearest each 1/SAMPLE_FPS target time.

    An integer frame step drifts (25 FPS → round(12.5) = 12 → 2.083 FPS); targeting timestamps gives exactly
    SAMPLE_FPS samples per second of video, each within half a source frame of its target, and never takes
    a frame twice when the source is slower than SAMPLE_FPS.
    """

    def __init__(self, fps: float, rate: float = SAMPLE_FPS):
        if not fps or not 0 < fps <= 1000:
            raise ValueError(f"video reports an unusable frame rate ({fps!r})")
        self.fps, self.interval, self.next_t = float(fps), 1.0 / rate, 0.0

    def take(self, idx: int) -> bool:
        reach = idx / self.fps + 0.5 / self.fps
        if reach < self.next_t:
            return False
        while self.next_t <= reach:
            self.next_t += self.interval
        return True


def sample_indices(fps: float, n_frames: int):
    sampler = TimestampSampler(fps)
    return (i for i in range(n_frames) if sampler.take(i))


def sampled_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path.name}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        sampler = TimestampSampler(fps)
    except ValueError as exc:
        cap.release()
        raise RuntimeError(f"{path.name}: {exc}") from exc
    idx = 0
    try:
        while True:
            if sampler.take(idx):
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx / fps, frame
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()


def chunked(iterable, size: int):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


def frame_players(frame, pitch_result, player_result, world_xy, names, exclude=()) -> tuple[str | None, list[dict]]:
    kp = pitch_result.keypoints
    if kp is None or kp.conf is None or len(kp) == 0:
        return "no_pitch_detection", []
    best = int(pitch_result.boxes.conf.argmax()) if pitch_result.boxes is not None and len(pitch_result.boxes) else 0
    xy, conf = kp.xy[best].cpu().numpy(), mask_landmarks(kp.conf[best].cpu().numpy(), exclude)
    conf = reject_isolated_centre(xy, world_xy, clamp_border_landmarks(xy, conf, (frame.shape[1], frame.shape[0])),
                                  frame.shape[0])
    H, reason = fit_homography(xy, world_xy, conf)
    if reason:
        return reason, []
    hull = cv2.convexHull(np.asarray(world_xy, np.float32)[conf >= KP_CONF])
    out = []
    boxes = player_result.boxes
    for (x1, y1, x2, y2), cls in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)):
        role = names[int(cls)]
        if role not in ("player", "goalkeeper"):
            continue
        x, y = project(H, [[(x1 + x2) / 2, y2]])[0]
        if cv2.pointPolygonTest(hull, (float(x), float(y)), True) < -HULL_MARGIN_M:
            continue
        out.append({"x": float(x), "y": float(y), "role": role,
                    "kit": torso_colour(frame, x1, y1, x2, y2) if role == "player" else None})
    return None, out


def _predict(batch, pitch_model, player_model):
    import torch
    images = [f for _, f in batch]
    with torch.inference_mode():
        return (list(pitch_model.predict(images, imgsz=640, conf=0.3, half=True, verbose=False)),
                list(player_model.predict(images, imgsz=IMGSZ, conf=0.3, half=True, verbose=False)))


def process_chunk(chunk, pitch_model, player_model, world_xy, stats, rows, exclude=()) -> None:
    import torch
    try:
        for i in range(0, len(chunk), BATCH):
            batch = chunk[i:i + BATCH]
            try:
                pitch_res, player_res = _predict(batch, pitch_model, player_model)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                pitch_res, player_res = [], []
                for item in batch:  # second OOM propagates to the FAILED report
                    p, d = _predict([item], pitch_model, player_model)
                    pitch_res += p
                    player_res += d
            for (t, frame), pr, dr in zip(batch, pitch_res, player_res):
                stats["sampled"] += 1
                reason, players = frame_players(frame, pr, dr, world_xy, player_model.names, exclude)
                if reason:
                    stats["dropped"][reason] += 1
                    continue
                stats["accepted"] += 1
                rows.extend({"t": t, **p} for p in players)
    finally:
        chunk.clear()
        torch.cuda.empty_cache()


# ── Entry point ───────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _find(root: Path, name: str) -> Path:
    hit = next(iter(sorted(root.rglob(name))), None)
    if hit is None:
        raise FileNotFoundError(f"{name} not found under {root}; attach the footage and models datasets")
    return hit


def write_output(doc: dict, out_dir: Path = OUTPUT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(doc, indent=2, sort_keys=True).encode("utf-8")
    (out_dir / "vision_metrics.json").write_bytes(raw)
    (out_dir / "vision_metrics.sha256").write_text(hashlib.sha256(raw).hexdigest(), encoding="utf-8")


def run(input_dir: Path = INPUT, output_dir: Path = OUTPUT) -> None:
    manifest_path = _find(input_dir, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = manifest_path.parent / manifest["payload"]
    digest = _sha256(payload)
    if digest != manifest["payload_sha256"]:
        raise ValueError("payload SHA-256 does not match manifest; dataset upload was corrupted")

    world, exclude = load_pitch_config(_find(input_dir, "pitch_config.json"))

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU attached; kernel-metadata enable_gpu must be true")
    # before any install or tensor op, so an unsupported GPU fails in seconds with a named cause
    gpu_error = gpu_compatibility_error(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0),
                                        torch.cuda.get_arch_list())
    if gpu_error:
        raise RuntimeError(gpu_error)
    print(f"GPU {torch.cuda.get_device_name(0)} capability {torch.cuda.get_device_capability(0)}", flush=True)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", PINNED], check=True)
    from ultralytics import YOLO

    players_pt, pitch_pt = _find(input_dir, "players.pt"), _find(input_dir, "pitch.pt")
    pitch_model, player_model = YOLO(str(pitch_pt)), YOLO(str(players_pt))
    kpt_shape = (getattr(pitch_model.model, "yaml", None) or {}).get("kpt_shape")
    if kpt_shape and kpt_shape[0] != len(world):
        raise ValueError(f"pitch.pt has {kpt_shape[0]} keypoints but pitch_config.json has {len(world)} landmarks")

    stats = {"sampled": 0, "accepted": 0, "dropped": {r: 0 for r in DROP_RULES}}
    rows: list[dict] = []
    for k, chunk in enumerate(chunked(sampled_frames(payload), CHUNK)):
        process_chunk(chunk, pitch_model, player_model, world, stats, rows, exclude)
        print(f"chunk {k}: sampled {stats['sampled']} accepted {stats['accepted']} dropped {stats['dropped']}", flush=True)

    moments, team_split = compute_metrics(rows, manifest["footage"], manifest.get("opponent_kit"))
    write_output({
        "schema_version": SCHEMA_VERSION, "status": "OK", "opponent": manifest["opponent"],
        "footage": manifest["footage"], "input_sha256": digest, "frames": stats, "moments": moments,
        "team_split": team_split,
        "models": {"ultralytics": PINNED, "players_sha256": _sha256(players_pt), "pitch_sha256": _sha256(pitch_pt)},
        "rules": {"sample_fps": SAMPLE_FPS, "kp_conf": KP_CONF, "min_landmarks": MIN_LANDMARKS,
                  "min_hull_m2": MIN_HULL_M2, "max_rmse_m": MAX_RMSE_M, "ransac_threshold_m": RANSAC_THRESHOLD_M,
                  "min_inlier_ratio": MIN_INLIER_RATIO,
                  "max_cond": MAX_COND, "min_samples": MIN_SAMPLES, "silhouette_min": SILHOUETTE_MIN,
                  "exclude_landmarks": list(exclude), "border_px": BORDER_PX,
                  "isolated_centre_px": ISOLATED_CENTRE_PX, "tolerance_reference_height_px": REFERENCE_HEIGHT_PX,
                  "settled_min_players": SETTLED_MIN_PLAYERS, "settled_max_gap_s": SETTLED_MAX_GAP_S},
    }, output_dir)


def main() -> int:
    try:
        run()
    except Exception:  # contract: the collector must always receive a report
        write_output({"status": "FAILED", "error": traceback.format_exc()}, OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
