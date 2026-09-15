"""
cv/vision_preflight.py — background pre-flight kit separation check for the Vision Command Center.

Launched detached by cv/vision_center.start_preflight; never run by hand. Samples PREFLIGHT_FRAMES frames at
the resolution the runner will upload (720p when the file is over the 1.5 GB downsample limit), runs players.pt
on CPU and scores the kit split with the worker's own torso_colour / two_means / silhouette. Always writes a
JSON result, including on error, so the dashboard never waits on a crashed check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv import vision_center as vc  # noqa: E402


def sample_features(video: Path, frames: int = vc.PREFLIGHT_FRAMES) -> tuple[list[list[float]], int]:
    W = vc.worker()
    weights = vc.MODELS_DIR / "players.pt"
    if not weights.exists():
        raise FileNotFoundError(f"players.pt not found in {vc.MODELS_DIR}")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    downsample = video.stat().st_size > vc.runner().DOWNSAMPLE_BYTES
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video.name}")
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    features, used = [], 0
    try:
        for pos in (np.linspace(total * 0.05, total * 0.95, frames) if total > 0 else []):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            if downsample and h > 720:
                frame = cv2.resize(frame, (round(w * 720 / h), 720))
            used += 1
            r = model.predict(frame, imgsz=W.IMGSZ, conf=0.3, device="cpu", verbose=False)[0]
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                if model.names[int(c)] == "player" and (k := W.torso_colour(frame, x1, y1, x2, y2)) is not None:
                    features.append(k)
    finally:
        cap.release()
    return features, used


def main(argv=None, sampler=sample_features) -> int:
    ap = argparse.ArgumentParser(description="Pre-flight kit separation check (launched by the dashboard).")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    started = vc._now_iso()
    try:
        features, used = sampler(args.video)
        doc = {**vc.separation_report(features, args.kit), "frames": used}
    except Exception as exc:  # the dashboard shows this; a missing result would leave it waiting
        doc = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "kit": args.kit.upper()}
    doc.update(video=args.video.name, started_at=started, finished_at=vc._now_iso())
    vc._write_json(args.out, doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
