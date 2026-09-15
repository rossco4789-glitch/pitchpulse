#!/usr/bin/env python3
"""
diagnose_benchmark_homography.py — local CPU diagnostic for homography frame rejections
=====================================================================================
One-off, offline root-cause tool for the Kaggle worker's geometry drops. It imports the worker template
itself, so sampling, fit_homography and every threshold are the ones that run on the GPU.

Usage:
    python tools/diagnose_benchmark_homography.py
    python tools/diagnose_benchmark_homography.py --video data/staging/0bfacc_0.mp4 --weights data/staging/models/pitch.pt

Per sampled frame (exact 2.0 FPS via TimestampSampler, pitch.pt on CPU):
    landmarks at confidence ≥ 0.5; worker verdict now, with KP_CONF 0.7 (Fix C), and with 5 px pixel-space
    consensus (Fix B); world (1.0 m) and pixel (5 px) RANSAC inlier ratio, RMSE, and outlier landmark numbers;
    per-landmark residuals in metres and pixels.
Across frames:
    residuals and outlier rates by landmark family (halfway/touchline/centre circle vs penalty/goal boxes);
    leave-one-out displacement on the 5 accepted frames with the most confident landmarks;
    Fix A probe: box landmarks projected through a fit that uses only halfway-line/centre landmarks.

Output: printed tables and data/staging/benchmark_0bfacc/diagnostic.json (gitignored).
Security note: loading a .pt file unpickles it; run only on weights already inspected (see PROJECT_INDEX).
"""

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PIXEL_THRESHOLD_PX = 5.0
CONF_C = 0.7
# 1-based landmark numbers (roboflow SoccerPitchConfiguration order): pitch corners, halfway-line ends,
# centre-circle crossings. Everything else sits on a penalty box, goal box, or penalty spot.
LINES_CENTRE = {1, 6, 14, 15, 16, 17, 25, 30, 31, 32}
HALFWAY_CENTRE = {14, 15, 16, 17, 31, 32}


def load_worker():
    spec = importlib.util.spec_from_file_location("kaggle_vision_worker", ROOT / "tools" / "templates" / "kaggle_vision_worker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def family(n: int) -> str:
    return "lines_centre" if n in LINES_CENTRE else "boxes"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))) if len(a) else float("nan")


# Landmarks on one painted straight line (1-based). A pinhole camera keeps straight lines straight under any
# perspective, so keypoints on these lines deviating from a fitted line by more than keypoint noise indicate
# lens or stitching distortion. Box lines are omitted: with 11/12/19/20 excluded they keep only 2 points.
PAINTED_LINES = {
    "goal_line_x0 (1-6)": (1, 2, 3, 4, 5, 6),
    "goal_line_x105 (25-30)": (25, 26, 27, 28, 29, 30),
    "halfway (14-17)": (14, 15, 16, 17),
    "touchline_y0 (1,14,25)": (1, 14, 25),
    "touchline_y68 (6,17,30)": (6, 17, 30),
}


def iter_samples(W, video: Path, start_s: float = 0.0, seconds: float | None = None):
    """Stream (t, frame) at exactly 2.0 FPS; never holds more than one frame (a full match is ~13.9k samples)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = int(round(start_s * fps))
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    sampler, idx = W.TimestampSampler(fps), 0
    try:
        while seconds is None or idx / fps < seconds:
            if sampler.take(idx):
                ok, frame = cap.read()
                if not ok:
                    break
                yield (first + idx) / fps, frame
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()


def line_deviation_px(W, xy, conf, landmarks) -> tuple[float, float] | None:
    """(max orthogonal deviation, span) in pixels of confident keypoints from their total-least-squares line."""
    idx = [n - 1 for n in landmarks if conf[n - 1] >= W.KP_CONF]
    if len(idx) < 3:
        return None
    pts = np.asarray(xy, float)[idx]
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return float(np.abs(centred @ vt[-1]).max()), float(np.ptp(centred @ vt[0]))


def valid(H) -> bool:
    return H is not None and np.all(np.isfinite(H)) and np.linalg.cond(H) <= 1e7


def pixel_rule_verdict(W, img, wld, G, mask) -> str:
    """Fix B: same rules as the worker, but landmark agreement judged in pixels (world → image RANSAC)."""
    if len(img) < W.MIN_LANDMARKS:
        return "too_few_landmarks"
    if cv2.contourArea(cv2.convexHull(wld)) < W.MIN_HULL_M2:
        return "drop_insufficient_pitch_area"
    if not valid(G) or not valid(np.linalg.inv(G)):
        return "drop_singular_matrix"
    inl = mask.ravel().astype(bool)
    if inl.sum() < W.MIN_LANDMARKS or inl.mean() < W.MIN_INLIER_RATIO:
        return "drop_insufficient_inliers"
    if rmse(W.project(np.linalg.inv(G), img[inl]), wld[inl]) > W.MAX_RMSE_M:
        return "drop_reprojection_error_high"
    return "accepted"


def analyse_frame(W, xy, conf, world):
    keep = np.where(conf >= W.KP_CONF)[0]
    rec = {"n_conf": int(len(keep)), "landmarks": (keep + 1).tolist(),
           "mean_conf": float(conf[keep].mean()) if len(keep) else 0.0,
           "worker": W.fit_homography(xy, world, conf)[1] or "accepted"}
    saved = W.KP_CONF
    W.KP_CONF = CONF_C
    try:
        rec["worker_conf07"] = W.fit_homography(xy, world, conf)[1] or "accepted"
        rec["n_conf07"] = int((conf >= CONF_C).sum())
    finally:
        W.KP_CONF = saved
    residuals = []
    if len(keep) < 4:
        rec["pixel_rule"] = "too_few_landmarks"
        return rec, residuals, None

    img, wld = xy[keep].astype(np.float32), world[keep].astype(np.float32)
    Hw, mw = cv2.findHomography(img, wld, cv2.RANSAC, W.RANSAC_THRESHOLD_M)
    G, mp = cv2.findHomography(wld, img, cv2.RANSAC, PIXEL_THRESHOLD_PX)

    w_in = mw.ravel().astype(bool) if valid(Hw) else np.zeros(len(keep), bool)
    if valid(Hw):
        rec.update(w_inlier_ratio=round(float(w_in.mean()), 3), w_rmse_m=round(rmse(W.project(Hw, img[w_in]), wld[w_in]), 3),
                   w_outliers=(keep[~w_in] + 1).tolist())
    p_in = mp.ravel().astype(bool) if valid(G) else np.zeros(len(keep), bool)
    if valid(G):
        Hp = np.linalg.inv(G)
        rec.update(p_inlier_ratio=round(float(p_in.mean()), 3),
                   p_rmse_px=round(rmse(W.project(G, wld[p_in]), img[p_in]), 2),
                   p_rmse_m=round(rmse(W.project(Hp, img[p_in]), wld[p_in]), 3),
                   p_outliers=(keep[~p_in] + 1).tolist())
    rec["pixel_rule"] = pixel_rule_verdict(W, img, wld, G, mp if mp is not None else np.zeros((len(keep), 1)))

    # residuals against the pixel-consensus fit when available, else the world fit
    ref = (np.linalg.inv(G), G, "pixel") if valid(G) else ((Hw, np.linalg.inv(Hw), "world") if valid(Hw) else None)
    Gp = None
    if ref:
        Hm, Gp, ref_name = ref
        res_m = np.linalg.norm(W.project(Hm, img) - wld, axis=1)
        expected = W.project(Gp, wld)                      # where the consensus fit says each landmark should be
        res_px = np.linalg.norm(expected - img, axis=1)
        for k, i in enumerate(keep):
            residuals.append({"landmark": int(i + 1), "conf": round(float(conf[i]), 3),
                              "x": round(float(img[k][0]), 1), "y": round(float(img[k][1]), 1),
                              "exp_x": round(float(expected[k][0]), 1), "exp_y": round(float(expected[k][1]), 1),
                              "res_m": float(res_m[k]), "res_px": float(res_px[k]), "reference_fit": ref_name,
                              "world_outlier": bool(valid(Hw) and not w_in[k]), "pixel_outlier": bool(valid(G) and not p_in[k])})
    return rec, residuals, Gp


PITCH_SEGMENTS_M = [((0, 0), (105, 0)), ((0, 68), (105, 68)), ((0, 0), (0, 68)), ((105, 0), (105, 68)),
                    ((52.5, 0), (52.5, 68)),
                    ((0, 13.84), (16.5, 13.84)), ((16.5, 13.84), (16.5, 54.16)), ((16.5, 54.16), (0, 54.16)),
                    ((105, 13.84), (88.5, 13.84)), ((88.5, 13.84), (88.5, 54.16)), ((88.5, 54.16), (105, 54.16))]


def save_overlay(W, frame, t, res, Gp, world, inspect, out_dir: Path) -> Path:
    """Detected confident landmarks (red), consensus-expected positions (green), projected pitch lines (green)."""
    img = frame.copy()
    h, w = img.shape[:2]
    if Gp is not None:
        for (a, b) in PITCH_SEGMENTS_M:
            pts = W.project(Gp, np.linspace(a, b, 40).astype(np.float32))
            ok = (np.abs(pts[:, 0]) < 3 * w) & (np.abs(pts[:, 1]) < 3 * h)
            for p, q, good in zip(pts[:-1], pts[1:], ok[:-1] & ok[1:]):
                if good:
                    cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), (0, 200, 0), 2)
        for n in inspect:  # expected position of inspected landmarks even when undetected
            ex, ey = W.project(Gp, world[n - 1:n].astype(np.float32))[0]
            if -w < ex < 2 * w and -h < ey < 2 * h:
                cv2.drawMarker(img, (int(ex), int(ey)), (0, 255, 0), cv2.MARKER_CROSS, 28, 3)
                cv2.putText(img, f"exp{n}", (int(ex) + 8, int(ey) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for x in res:
        colour = (0, 0, 255) if x["world_outlier"] else (255, 255, 0)
        cv2.line(img, (int(x["x"]), int(x["y"])), (int(x["exp_x"]), int(x["exp_y"])), (0, 255, 255), 1)
        cv2.circle(img, (int(x["x"]), int(x["y"])), 9, colour, 3)
        cv2.putText(img, f"{x['landmark']} {x['conf']:.2f}", (int(x["x"]) + 10, int(x["y"]) + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.putText(img, f"t={t:.1f}s  red=world outlier  cyan=inlier  green=consensus fit", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"t{t:07.1f}.jpg"
    cv2.imwrite(str(path), cv2.resize(img, (w * 2 // 3, h * 2 // 3)))
    return path


def edge_distance(x: float, y: float, wh) -> float:
    """Pixels to the nearest image edge; negative means outside the image."""
    w, h = wh
    return float(min(x, y, w - 1 - x, h - 1 - y))


CENTRE_CIRCLE = (31, 32)
ISOLATED_CENTRE_PX = 80.0   # broadcast 0bfacc: valid #32 at 79 px (LOO 0.52 m); Veo: 182/184 hallucinations > 80 px


def isolated_centre_rejects(W, xy, conf, world) -> set[int]:
    """Change 2b: centre-circle landmarks inconsistent with a same-end box/goal-line cluster fit (world → pixel)."""
    ok = conf >= W.KP_CONF
    out = set()
    for end in (world[:, 0] <= 16.5, world[:, 0] >= 88.5):
        idx = np.where(ok & end)[0]
        if len(idx) < 4 or cv2.contourArea(cv2.convexHull(world[idx].astype(np.float32))) < 50:
            continue
        G, mask = cv2.findHomography(world[idx].astype(np.float32), xy[idx].astype(np.float32), cv2.RANSAC, 2 * PIXEL_THRESHOLD_PX)
        if not valid(G) or mask.sum() < 4:
            continue
        for n in CENTRE_CIRCLE:
            if ok[n - 1]:
                ex = W.project(G, world[n - 1:n].astype(np.float32))[0]
                if np.linalg.norm(ex - xy[n - 1]) > ISOLATED_CENTRE_PX:
                    out.add(n)
    return out


def leave_one_out(W, xy, conf, world):
    keep = np.where(conf >= W.KP_CONF)[0]
    img, wld = xy[keep].astype(np.float32), world[keep].astype(np.float32)
    out = []
    for j in range(len(keep)):
        others = np.arange(len(keep)) != j
        H, _ = cv2.findHomography(img[others], wld[others], 0)
        if not valid(H):
            continue
        disp_m = float(np.linalg.norm(W.project(H, img[j:j + 1])[0] - wld[j]))
        disp_px = float(np.linalg.norm(W.project(np.linalg.inv(H), wld[j:j + 1])[0] - img[j]))
        out.append({"landmark": int(keep[j] + 1), "family": family(int(keep[j] + 1)), "disp_m": disp_m, "disp_px": disp_px})
    return out


def box_probe(W, xy, conf, world):
    """Fix A probe: fit from halfway/centre landmarks only, report where visible box landmarks land in metres."""
    keep = [i for i in np.where(conf >= W.KP_CONF)[0]]
    anchor = [i for i in keep if i + 1 in HALFWAY_CENTRE]
    boxes = [i for i in keep if i + 1 not in LINES_CENTRE]
    if len(anchor) < 4 or not boxes or cv2.contourArea(cv2.convexHull(world[anchor].astype(np.float32))) < 20:
        return []
    H, _ = cv2.findHomography(xy[anchor].astype(np.float32), world[anchor].astype(np.float32), 0)
    if not valid(H):
        return []
    est = W.project(H, xy[boxes].astype(np.float32))
    return [{"landmark": int(i + 1), "dx_m": float(e[0] - world[i][0]), "dy_m": float(e[1] - world[i][1])}
            for i, e in zip(boxes, est)]


def pct(values, q):
    return round(float(np.percentile(values, q)), 2) if len(values) else None


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description="Local CPU diagnostic for homography rejections.")
    ap.add_argument("--video", type=Path, default=ROOT / "data" / "staging" / "0bfacc_0.mp4")
    ap.add_argument("--weights", type=Path, default=ROOT / "data" / "staging" / "models" / "pitch.pt")
    ap.add_argument("--config", type=Path, default=ROOT / "data" / "staging" / "models" / "pitch_config.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "staging" / "benchmark_0bfacc" / "diagnostic.json")
    ap.add_argument("--no-exclude", action="store_true", help="Ignore exclude_landmarks in the config (baseline)")
    ap.add_argument("--start", type=float, default=0.0, help="Start time in seconds (default 0)")
    ap.add_argument("--seconds", type=float, default=None, help="Window length in seconds (default: to the end)")
    ap.add_argument("--inspect", default="17,32", help="Comma-separated landmark numbers to inspect in detail")
    ap.add_argument("--overlay-dir", type=Path, default=None, help="Save annotated frames where inspected landmarks disagree")
    ap.add_argument("--overlays", type=int, default=6, help="Maximum overlays, at least 10 s apart (default 6)")
    ap.add_argument("--border-px", type=float, default=0.0,
                    help="Change 2a: drop keypoints within this many pixels of (or outside) the image edge (default 0 = off)")
    ap.add_argument("--filter-isolated-centre", action="store_true",
                    help="Change 2b: drop centre-circle landmarks (31, 32) inconsistent with a same-end box cluster fit")
    args = ap.parse_args(argv)
    inspect = [int(n) for n in args.inspect.split(",") if n.strip()]

    W = load_worker()
    world, exclude = W.load_pitch_config(args.config)
    if args.no_exclude:
        exclude = ()
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    window = f"{args.start:.0f}s-{'end' if args.seconds is None else f'{args.start + args.seconds:.0f}s'}"
    print(f"sampling {args.video.name} [{window}] at 2.0 FPS; running {args.weights.name} on CPU; "
          f"exclude_landmarks {list(exclude)}; border_px {args.border_px}", flush=True)

    frames, residuals, detections, t0 = [], [], {}, time.perf_counter()
    line_dev = defaultdict(list)
    overlay_paths, last_overlay_t, isolated_count = [], float("-inf"), 0
    for t, frame in iter_samples(W, args.video, args.start, args.seconds):
        if frames and len(frames) % 500 == 0:
            print(f"  progress: {len(frames)} frames, t = {t:.0f} s, {time.perf_counter() - t0:.0f} s elapsed", flush=True)
        r = model.predict(frame, imgsz=640, conf=0.3, device="cpu", verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            frames.append({"t": round(t, 2), "image_wh": [frame.shape[1], frame.shape[0]], "keypoints": [],
                           "n_conf": 0, "worker": "no_pitch_detection",
                           "worker_conf07": "no_pitch_detection", "pixel_rule": "no_pitch_detection"})
            continue
        best = int(r.boxes.conf.argmax())
        raw_conf = r.keypoints.conf[best].cpu().numpy()
        xy, conf = r.keypoints.xy[best].cpu().numpy(), W.mask_landmarks(raw_conf, exclude)
        wh = (frame.shape[1], frame.shape[0])
        clamped = {i + 1 for i in range(len(xy))
                   if args.border_px > 0 and edge_distance(xy[i][0], xy[i][1], wh) < args.border_px}
        if clamped:
            conf[[n - 1 for n in clamped]] = 0.0
        isolated = isolated_centre_rejects(W, xy, conf, world) if args.filter_isolated_centre else set()
        if isolated:
            conf[[n - 1 for n in isolated]] = 0.0
            isolated_count += len(isolated)
        keypoints = [{"landmark": i + 1, "x": round(float(xy[i][0]), 1), "y": round(float(xy[i][1]), 1),
                      "conf": round(float(raw_conf[i]), 3), "excluded": i + 1 in exclude,
                      "border_clamped": i + 1 in clamped, "isolated_centre": i + 1 in isolated} for i in range(len(xy))]
        detections[round(t, 2)] = (xy, conf)
        for name, landmarks in PAINTED_LINES.items():
            dev = line_deviation_px(W, xy, conf, landmarks)
            if dev:
                line_dev[name].append(dev)
        rec, res, Gp = analyse_frame(W, xy, conf, world)
        frames.append({"t": round(t, 2), "image_wh": [frame.shape[1], frame.shape[0]], "keypoints": keypoints, **rec})
        residuals.extend({"t": round(t, 2), **x} for x in res)
        if (args.overlay_dir and len(overlay_paths) < args.overlays and t - last_overlay_t >= 10
                and any(x["landmark"] in inspect and x["world_outlier"] for x in res)):
            overlay_paths.append(str(save_overlay(W, frame, t, res, Gp, world, inspect, args.overlay_dir)))
            last_overlay_t = t
    elapsed = time.perf_counter() - t0

    # ── Per-frame table ──
    print(f"\n{len(frames)} frames; CPU inference {elapsed:.0f} s ({elapsed / max(1, len(frames)):.2f} s/frame)\n")
    if len(frames) > 300:
        print(f"(per-frame table omitted for {len(frames)} frames; see {args.out.name})")
    else:
        print(f"{'t':>5} {'n':>3} {'worker':<24} {'conf0.7':<24} {'pixel5px':<24} {'w_in':>5} {'w_rmse':>6} {'p_in':>5} {'p_px':>5} {'p_m':>5}  world_outliers / pixel_outliers")
    for f in (frames if len(frames) <= 300 else []):
        print(f"{f['t']:5.1f} {f['n_conf']:3d} {f['worker']:<24} {f['worker_conf07']:<24} {f['pixel_rule']:<24} "
              f"{f.get('w_inlier_ratio', ''):>5} {f.get('w_rmse_m', ''):>6} {f.get('p_inlier_ratio', ''):>5} "
              f"{f.get('p_rmse_px', ''):>5} {f.get('p_rmse_m', ''):>5}  {f.get('w_outliers', [])} / {f.get('p_outliers', [])}")

    # ── Verdict counts per rule set ──
    verdicts = {name: dict(sorted(defaultdict(int, {v: sum(1 for f in frames if f[key] == v) for v in {f[key] for f in frames}}).items()))
                for name, key in (("current (1.0 m, conf 0.5)", "worker"), ("Fix C (conf 0.7)", "worker_conf07"),
                                  ("Fix B (5 px consensus)", "pixel_rule"))}
    print(f"\nChange 2b isolated centre-circle rejections: {isolated_count}" if args.filter_isolated_centre else "")
    print("\nVerdicts")
    for name, counts in verdicts.items():
        print(f"  {name:<28} accepted {counts.get('accepted', 0)}/{len(frames)}  {counts}")

    # ── Landmark families and worst landmarks ──
    fam = {}
    for name in ("lines_centre", "boxes"):
        rows = [x for x in residuals if family(x["landmark"]) == name]
        fam[name] = {"observations": len(rows),
                     "world_outlier_rate": round(sum(x["world_outlier"] for x in rows) / max(1, len(rows)), 3),
                     "pixel_outlier_rate": round(sum(x["pixel_outlier"] for x in rows) / max(1, len(rows)), 3),
                     "res_m_p50": pct([x["res_m"] for x in rows], 50), "res_m_p90": pct([x["res_m"] for x in rows], 90),
                     "res_px_p50": pct([x["res_px"] for x in rows], 50), "res_px_p90": pct([x["res_px"] for x in rows], 90)}
    print("\nLandmark families (residuals against the pixel-consensus fit)")
    for name, s in fam.items():
        print(f"  {name:<13} {s}")

    per_lm = defaultdict(list)
    for x in residuals:
        per_lm[x["landmark"]].append(x)
    lm_rows = sorted(({"landmark": n, "family": family(n), "seen": len(v),
                       "world_outliers": sum(x["world_outlier"] for x in v), "pixel_outliers": sum(x["pixel_outlier"] for x in v),
                       "res_m_p50": pct([x["res_m"] for x in v], 50), "res_px_p50": pct([x["res_px"] for x in v], 50)}
                      for n, v in per_lm.items()), key=lambda r: (-r["world_outliers"], -r["seen"]))
    print("\nLandmarks by world-RANSAC outlier count")
    for r in lm_rows:
        print(f"  #{r['landmark']:<2} {r['family']:<13} seen {r['seen']:>2}  world_out {r['world_outliers']:>2}  "
              f"pixel_out {r['pixel_outliers']:>2}  res_m p50 {r['res_m_p50']}  res_px p50 {r['res_px_p50']}")

    # ── Inspected landmarks: where do disagreeing detections sit in the image? ──
    dims = {f["t"]: f["image_wh"] for f in frames}
    inspection = {}
    print(f"\nInspected landmarks {inspect} (edge distance: px to nearest image edge, negative = outside; "
          f"expected = consensus-fit position)")
    for n in inspect:
        rows = [x for x in residuals if x["landmark"] == n]
        summary = {}
        for label, group in (("world_outlier", [x for x in rows if x["world_outlier"]]),
                             ("world_inlier", [x for x in rows if not x["world_outlier"]])):
            if not group:
                summary[label] = {"n": 0}
                continue
            det_edge = [edge_distance(x["x"], x["y"], dims[x["t"]]) for x in group]
            exp_edge = [edge_distance(x["exp_x"], x["exp_y"], dims[x["t"]]) for x in group]
            margin = 0.05 * min(dims[group[0]["t"]])
            summary[label] = {
                "n": len(group),
                "detected_outside_image": sum(d < 0 for d in det_edge),
                "detected_within_5pct_edge_margin": sum(0 <= d < margin for d in det_edge),
                "detected_edge_dist_px_p50": pct(det_edge, 50),
                "expected_outside_image": sum(d < 0 for d in exp_edge),
                "expected_edge_dist_px_p50": pct(exp_edge, 50),
                "detected_xy_p50": [pct([x["x"] for x in group], 50), pct([x["y"] for x in group], 50)],
                "expected_xy_p50": [pct([x["exp_x"] for x in group], 50), pct([x["exp_y"] for x in group], 50)],
                "detected_to_expected_px_p50": pct([x["res_px"] for x in group], 50),
                "detected_to_expected_px_p90": pct([x["res_px"] for x in group], 90),
                "conf_p50": pct([x["conf"] for x in group], 50),
                "reference_fit_pixel_share": round(sum(x["reference_fit"] == "pixel" for x in group) / len(group), 2),
            }
        examples = sorted((x for x in rows if x["world_outlier"]), key=lambda x: -x["res_px"])[:6]
        inspection[n] = {"summary": summary, "worst_outliers": examples}
        print(f"  #{n}")
        for label, s in summary.items():
            print(f"    {label:<14} {s}")
        for x in examples:
            print(f"    worst t={x['t']:.1f}s det=({x['x']:.0f},{x['y']:.0f}) exp=({x['exp_x']:.0f},{x['exp_y']:.0f}) "
                  f"conf={x['conf']:.2f} off={x['res_px']:.0f}px image={dims[x['t']]}")
    if overlay_paths:
        print("  overlays:", overlay_paths)

    # ── Leave-one-out on the 5 best accepted frames ──
    best = sorted((f for f in frames if f["worker"] == "accepted"), key=lambda f: (-f["n_conf"], -f["mean_conf"]))[:5]
    loo = [{"t": f["t"], **x} for f in best for x in leave_one_out(W, *detections[f["t"]], world)]
    loo_summary = {name: {"n": len(v), "disp_m_p50": pct([x["disp_m"] for x in v], 50), "disp_m_max": pct([x["disp_m"] for x in v], 100),
                          "disp_px_p50": pct([x["disp_px"] for x in v], 50), "disp_px_max": pct([x["disp_px"] for x in v], 100)}
                   for name in ("lines_centre", "boxes") for v in [[x for x in loo if x["family"] == name]]}
    print(f"\nLeave-one-out on frames t = {[f['t'] for f in best]}; overall disp_m p50 {pct([x['disp_m'] for x in loo], 50)}")
    for name, s in loo_summary.items():
        print(f"  {name:<13} {s}")
    worst = sorted(loo, key=lambda x: -x["disp_m"])[:6]
    print("  worst held-out:", [(x["t"], x["landmark"], round(x["disp_m"], 2), round(x["disp_px"], 1)) for x in worst])

    # ── Fix A probe ──
    probe = [{"t": t, **x} for t, (xy, conf) in detections.items() for x in box_probe(W, xy, conf, world)]
    probe_summary = {}
    for x in probe:
        probe_summary.setdefault(x["landmark"], []).append((x["dx_m"], x["dy_m"]))
    probe_rows = {n: {"n": len(v), "dx_m_median": round(float(np.median([a for a, _ in v])), 2),
                      "dy_m_median": round(float(np.median([b for _, b in v])), 2)} for n, v in sorted(probe_summary.items())}
    print("\nFix A probe (box landmarks projected through a halfway/centre-only fit; offset from pitch_config)")
    print("  " + (json.dumps(probe_rows) if probe_rows else "no frame had ≥ 4 halfway/centre landmarks alongside box landmarks"))

    # ── Painted-line straightness ──
    straightness = {name: {"frames": len(v), "max_dev_px_p50": pct([d for d, _ in v], 50),
                           "max_dev_px_p90": pct([d for d, _ in v], 90),
                           "span_px_p50": pct([s for _, s in v], 50),
                           "dev_pct_of_span_p50": pct([100 * d / s for d, s in v if s > 0], 50)}
                    for name, v in line_dev.items()}
    print("\nPainted-line straightness (pinhole cameras keep straight lines straight; deviation beyond ~2 px is distortion)")
    for name, s in straightness.items():
        print(f"  {name:<24} {s}")
    if not straightness:
        print("  no painted line had ≥ 3 confident keypoints in any frame")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"video": args.video.name, "window": window, "exclude_landmarks": list(exclude),
                                    "border_px": args.border_px, "filter_isolated_centre": args.filter_isolated_centre,
                                    "isolated_centre_rejections": isolated_count, "line_straightness": straightness, "inspection": inspection,
                                    "overlays": overlay_paths,
                                    "cpu_seconds": round(elapsed, 1), "frames": frames,
                                    "verdicts": verdicts, "families": fam, "landmarks": lm_rows,
                                    "leave_one_out": {"frames": [f["t"] for f in best], "summary": loo_summary, "rows": loo},
                                    "fix_a_probe": probe_rows}, indent=2), encoding="utf-8")
    print(f"\nDiagnostic → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
