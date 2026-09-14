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


def read_samples(W, video: Path) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sampler, idx, out = W.TimestampSampler(fps), 0, []
    while True:
        if sampler.take(idx):
            ok, frame = cap.read()
            if not ok:
                break
            out.append((idx / fps, frame))
        elif not cap.grab():
            break
        idx += 1
    cap.release()
    return out


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
        return rec, residuals

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
    ref = (np.linalg.inv(G), G) if valid(G) else ((Hw, np.linalg.inv(Hw)) if valid(Hw) else None)
    if ref:
        Hm, Gp = ref
        res_m = np.linalg.norm(W.project(Hm, img) - wld, axis=1)
        res_px = np.linalg.norm(W.project(Gp, wld) - img, axis=1)
        for k, i in enumerate(keep):
            residuals.append({"landmark": int(i + 1), "res_m": float(res_m[k]), "res_px": float(res_px[k]),
                              "world_outlier": bool(valid(Hw) and not w_in[k]), "pixel_outlier": bool(valid(G) and not p_in[k])})
    return rec, residuals


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
    args = ap.parse_args(argv)

    W = load_worker()
    world, exclude = W.load_pitch_config(args.config)
    if args.no_exclude:
        exclude = ()
    samples = read_samples(W, args.video)
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    print(f"{len(samples)} frames sampled from {args.video.name}; running {args.weights.name} on CPU; "
          f"exclude_landmarks {list(exclude)}")

    frames, residuals, detections, t0 = [], [], {}, time.perf_counter()
    for t, frame in samples:
        r = model.predict(frame, imgsz=640, conf=0.3, device="cpu", verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            frames.append({"t": round(t, 2), "n_conf": 0, "worker": "no_pitch_detection",
                           "worker_conf07": "no_pitch_detection", "pixel_rule": "no_pitch_detection"})
            continue
        best = int(r.boxes.conf.argmax())
        xy, conf = r.keypoints.xy[best].cpu().numpy(), W.mask_landmarks(r.keypoints.conf[best].cpu().numpy(), exclude)
        detections[round(t, 2)] = (xy, conf)
        rec, res = analyse_frame(W, xy, conf, world)
        frames.append({"t": round(t, 2), **rec})
        residuals.extend({"t": round(t, 2), **x} for x in res)
    elapsed = time.perf_counter() - t0

    # ── Per-frame table ──
    print(f"\nCPU inference {elapsed:.0f} s ({elapsed / max(1, len(samples)):.2f} s/frame)\n")
    print(f"{'t':>5} {'n':>3} {'worker':<24} {'conf0.7':<24} {'pixel5px':<24} {'w_in':>5} {'w_rmse':>6} {'p_in':>5} {'p_px':>5} {'p_m':>5}  world_outliers / pixel_outliers")
    for f in frames:
        print(f"{f['t']:5.1f} {f['n_conf']:3d} {f['worker']:<24} {f['worker_conf07']:<24} {f['pixel_rule']:<24} "
              f"{f.get('w_inlier_ratio', ''):>5} {f.get('w_rmse_m', ''):>6} {f.get('p_inlier_ratio', ''):>5} "
              f"{f.get('p_rmse_px', ''):>5} {f.get('p_rmse_m', ''):>5}  {f.get('w_outliers', [])} / {f.get('p_outliers', [])}")

    # ── Verdict counts per rule set ──
    verdicts = {name: dict(sorted(defaultdict(int, {v: sum(1 for f in frames if f[key] == v) for v in {f[key] for f in frames}}).items()))
                for name, key in (("current (1.0 m, conf 0.5)", "worker"), ("Fix C (conf 0.7)", "worker_conf07"),
                                  ("Fix B (5 px consensus)", "pixel_rule"))}
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

    # ── Leave-one-out on the 5 best accepted frames ──
    best = sorted((f for f in frames if f["worker"] == "accepted"), key=lambda f: (-f["n_conf"], -f["mean_conf"]))[:5]
    loo = [{"t": f["t"], **x} for f in best for x in leave_one_out(W, *detections[f["t"]], world)]
    loo_summary = {name: {"n": len(v), "disp_m_p50": pct([x["disp_m"] for x in v], 50), "disp_m_max": pct([x["disp_m"] for x in v], 100),
                          "disp_px_p50": pct([x["disp_px"] for x in v], 50), "disp_px_max": pct([x["disp_px"] for x in v], 100)}
                   for name in ("lines_centre", "boxes") for v in [[x for x in loo if x["family"] == name]]}
    print(f"\nLeave-one-out on frames t = {[f['t'] for f in best]}")
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"video": args.video.name, "exclude_landmarks": list(exclude),
                                    "cpu_seconds": round(elapsed, 1), "frames": frames,
                                    "verdicts": verdicts, "families": fam, "landmarks": lm_rows,
                                    "leave_one_out": {"frames": [f["t"] for f in best], "summary": loo_summary, "rows": loo},
                                    "fix_a_probe": probe_rows}, indent=2), encoding="utf-8")
    print(f"\nDiagnostic → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
