"""
cv/calibration.py
Tier 1 Computer-Assisted Calibration — Planar Homography Matrix Computation.

Computes the 3×3 homography matrix H that maps camera pixel coordinates (u, v)
to real-world pitch coordinates (x_m, y_m) on a 105×68 m FIFA-standard pitch.

Mathematics
-----------
  [x']   [h00 h01 h02]   [u]
  [y'] = [h10 h11 h12] × [v]
  [w']   [h20 h21 h22]   [1]

  x_m = x'/w',  y_m = y'/w'

Usage
-----
  # Interactive (click reference anchors on a video frame):
  python cv/calibration.py --image data/raw/frame.jpg --out data/raw/camera_calibration.json

  # Non-interactive (supply pre-mapped pixel/world pairs as JSON):
  python cv/calibration.py --points-json data/raw/my_points.json --out data/raw/camera_calibration.json

  # Load and verify an existing calibration:
  python cv/calibration.py --verify data/raw/camera_calibration.json

This module does NOT touch run_matchday.py — it is a standalone midweek utility.
Requires: opencv-python, numpy, matplotlib (all already in the environment).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Pitch dimensions ──────────────────────────────────────────────────────────
PITCH_LENGTH: float = 105.0
PITCH_WIDTH:  float = 68.0

# ── Default calibration output path ──────────────────────────────────────────
DEFAULT_CALIB_PATH = ROOT / "data" / "raw" / "camera_calibration.json"

# ── Standard reference anchors (world coordinates on the FIFA pitch) ──────────
#   x_m : 0 = defensive goal line → 105 = attacking goal line
#   y_m : 0 = left touchline      → 68  = right touchline
#
# These are the points the analyst clicks in sequence during interactive mode.
ANCHORS: list[dict] = [
    {"label": "Centre spot",             "world": [52.5,  34.0]},
    {"label": "Halfway × Left touch",    "world": [52.5,   0.0]},
    {"label": "Halfway × Right touch",   "world": [52.5,  68.0]},
    {"label": "Left penalty spot",       "world": [11.0,  34.0]},
    {"label": "Right penalty spot",      "world": [94.0,  34.0]},
    {"label": "L pen area — near post",  "world": [ 0.0,  13.84]},
    {"label": "L pen area — far post",   "world": [ 0.0,  54.16]},
    {"label": "L pen area — 16.5m NP",   "world": [16.5,  13.84]},
    {"label": "L pen area — 16.5m FP",   "world": [16.5,  54.16]},
    {"label": "R pen area — near post",  "world": [105.0, 13.84]},
    {"label": "R pen area — far post",   "world": [105.0, 54.16]},
    {"label": "R pen area — 88.5m NP",   "world": [88.5,  13.84]},
    {"label": "R pen area — 88.5m FP",   "world": [88.5,  54.16]},
]


# ══════════════════════════════════════════════════════════════════════════════
# Core computation
# ══════════════════════════════════════════════════════════════════════════════

class ReferencePoint(NamedTuple):
    pixel: tuple[float, float]   # (u, v)
    world: tuple[float, float]   # (x_m, y_m)
    label: str = ""


def calibrate_pitch(
    reference_points: list[ReferencePoint],
    pitch_dims: tuple[float, float] = (PITCH_LENGTH, PITCH_WIDTH),
) -> np.ndarray:
    """
    Compute the 3×3 planar homography matrix H from ≥4 reference points.

    Parameters
    ----------
    reference_points : list of ReferencePoint
        Each point pairs a pixel coordinate (u, v) with a world coordinate
        (x_m, y_m) on the FIFA pitch.  Minimum 4 points; more improves accuracy.
    pitch_dims : (length, width) in metres — for documentation only; not used
        in the computation itself.

    Returns
    -------
    H : np.ndarray, shape (3, 3), dtype float64
        The homography matrix.  Apply with pixel_to_pitch() in cv/picker.py.

    Raises
    ------
    ValueError  : fewer than 4 reference points supplied.
    RuntimeError: cv2.findHomography could not compute a valid matrix.
    """
    if len(reference_points) < 4:
        raise ValueError(
            f"At least 4 reference points are required; got {len(reference_points)}."
        )

    src = np.array([p.pixel for p in reference_points], dtype=np.float64)
    dst = np.array([p.world  for p in reference_points], dtype=np.float64)

    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)

    if H is None or H.shape != (3, 3):
        raise RuntimeError(
            "cv2.findHomography failed to compute a valid 3×3 matrix.  "
            "Check that your reference points are not collinear."
        )

    return H.astype(np.float64)


def reprojection_error(
    reference_points: list[ReferencePoint],
    H: np.ndarray,
) -> float:
    """
    Compute mean reprojection error in pixels.

    Projects each world point back through H⁻¹ to pixel space and measures
    the Euclidean distance from the original clicked pixel.

    Returns
    -------
    float : mean error in pixels across all reference points.
    """
    H_inv = np.linalg.inv(H)
    errors: list[float] = []

    for rp in reference_points:
        wx, wy = rp.world
        w_h = H_inv @ np.array([wx, wy, 1.0])
        u_proj = w_h[0] / w_h[2]
        v_proj = w_h[1] / w_h[2]
        err = float(np.sqrt((u_proj - rp.pixel[0])**2 + (v_proj - rp.pixel[1])**2))
        errors.append(err)

    return float(np.mean(errors))


# ══════════════════════════════════════════════════════════════════════════════
# Save / load
# ══════════════════════════════════════════════════════════════════════════════

def save_calibration(
    H: np.ndarray,
    reference_points: list[ReferencePoint],
    out_path: Path = DEFAULT_CALIB_PATH,
    reprojection_error_px: float | None = None,
) -> Path:
    """
    Save calibration matrix and metadata to JSON.

    Schema
    ------
    {
      "calibrated_at": "ISO-8601 UTC",
      "pitch_dims": [105.0, 68.0],
      "n_reference_points": 4,
      "reprojection_error_px": 1.23,
      "H": [[h00, h01, h02], [h10, h11, h12], [h20, h21, h22]],
      "reference_points": [
        {"pixel": [u, v], "world": [x_m, y_m], "label": "..."},
        ...
      ]
    }
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "calibrated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pitch_dims": [PITCH_LENGTH, PITCH_WIDTH],
        "n_reference_points": len(reference_points),
        "reprojection_error_px": round(reprojection_error_px, 4) if reprojection_error_px is not None else None,
        "H": H.tolist(),
        "reference_points": [
            {"pixel": list(rp.pixel), "world": list(rp.world), "label": rp.label}
            for rp in reference_points
        ],
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_calibration(calib_path: Path = DEFAULT_CALIB_PATH) -> np.ndarray:
    """
    Load and return the 3×3 homography matrix H from a saved calibration JSON.

    Raises
    ------
    FileNotFoundError : calibration file does not exist.
    KeyError          : JSON is missing the "H" key (corrupt or wrong file).
    """
    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {calib_path}\n"
            "Run: python cv/calibration.py --image <frame.jpg> "
            "to create one."
        )
    payload = json.loads(calib_path.read_text(encoding="utf-8"))
    return np.array(payload["H"], dtype=np.float64)


def load_calibration_meta(calib_path: Path = DEFAULT_CALIB_PATH) -> dict:
    """Return the full calibration JSON dict (metadata + H matrix)."""
    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    return json.loads(calib_path.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════════════
# Interactive calibration (matplotlib click interface)
# ══════════════════════════════════════════════════════════════════════════════

def _run_interactive_calibration(
    image_path: Path,
    anchor_indices: list[int],
) -> list[ReferencePoint]:
    """
    Display the image in a matplotlib window and ask the analyst to click each
    selected anchor in sequence.  Returns a list of ReferencePoint objects.

    Parameters
    ----------
    image_path     : path to a PNG or JPEG still frame from the match footage.
    anchor_indices : indices into ANCHORS[] for the reference points to collect.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    selected_anchors = [ANCHORS[i] for i in anchor_indices]
    collected: list[ReferencePoint] = []
    click_queue: list[int] = list(range(len(selected_anchors)))

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img_rgb)
    ax.set_title(
        f"PITCHPULSE CALIBRATION — Click: {selected_anchors[0]['label']}",
        color="#f59e0b", fontsize=12, pad=10,
    )
    fig.patch.set_facecolor("#09090b")
    ax.set_facecolor("#09090b")

    dots: list = []

    def _on_click(event) -> None:
        if event.inaxes is not ax:
            return
        if not click_queue:
            return

        idx = click_queue.pop(0)
        anchor = selected_anchors[idx]
        u, v = float(event.xdata), float(event.ydata)
        collected.append(ReferencePoint(pixel=(u, v), world=tuple(anchor["world"]), label=anchor["label"]))

        # Draw confirmation dot
        dot = ax.plot(u, v, "o", color="#f59e0b", markersize=8, zorder=5)[0]
        ax.annotate(
            anchor["label"], (u, v),
            textcoords="offset points", xytext=(6, 6),
            fontsize=7, color="#f8fafc",
        )
        dots.append(dot)

        if click_queue:
            next_label = selected_anchors[click_queue[0]]["label"]
            ax.set_title(
                f"PITCHPULSE CALIBRATION — Click: {next_label}  "
                f"({idx + 1}/{len(selected_anchors)} done)",
                color="#f59e0b", fontsize=12,
            )
        else:
            ax.set_title(
                f"✓ All {len(selected_anchors)} anchors captured — close this window to proceed.",
                color="#22c55e", fontsize=12,
            )
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_click)
    plt.tight_layout()
    plt.show()

    if len(collected) < 4:
        raise RuntimeError(
            f"Only {len(collected)} anchor(s) clicked — minimum 4 required.  "
            "Re-run and click all anchors before closing the window."
        )

    return collected


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _print_anchor_menu() -> None:
    print("\n  Available reference anchors:")
    print("  ─────────────────────────────────────────────────────────")
    for i, a in enumerate(ANCHORS):
        x, y = a["world"]
        print(f"  {i + 1:2}.  {a['label']:<35} ({x:5.1f}, {y:5.2f}) m")
    print()


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="PitchPulse Tier 1 — Pitch Homography Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",       metavar="FRAME",  help="Path to a still-frame image (PNG/JPG)")
    group.add_argument("--points-json", metavar="FILE",   help="Path to pre-mapped pixel/world pairs JSON")
    group.add_argument("--verify",      metavar="CALIB",  help="Verify and display an existing calibration JSON")

    parser.add_argument(
        "--out", metavar="PATH",
        default=str(DEFAULT_CALIB_PATH),
        help=f"Output calibration JSON path (default: {DEFAULT_CALIB_PATH.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--anchors", metavar="N", nargs="+", type=int,
        help="Anchor numbers to use from the menu (e.g. --anchors 1 2 3 4 5). "
             "If omitted, an interactive selection prompt is shown.",
    )

    args = parser.parse_args(argv)
    out_path = Path(args.out)

    # ── Verify mode ──────────────────────────────────────────────────────────
    if args.verify:
        calib_path = Path(args.verify)
        meta = load_calibration_meta(calib_path)
        print(f"\n  Calibration: {calib_path}")
        print(f"  Calibrated : {meta.get('calibrated_at', 'unknown')}")
        print(f"  Points     : {meta.get('n_reference_points', '?')}")
        err = meta.get("reprojection_error_px")
        if err is not None:
            quality = "GOOD" if err < 3.0 else "WARN" if err < 8.0 else "POOR"
            print(f"  Re-proj err: {err:.2f} px  [{quality}]")
        print(f"  H matrix:")
        H = np.array(meta["H"])
        for row in H:
            print(f"    [{row[0]:12.6f}  {row[1]:12.6f}  {row[2]:12.6f}]")
        print()
        return

    # ── Points-JSON mode ─────────────────────────────────────────────────────
    if args.points_json:
        pts_path = Path(args.points_json)
        raw = json.loads(pts_path.read_text(encoding="utf-8"))
        ref_pts = [
            ReferencePoint(
                pixel=tuple(p["pixel"]),
                world=tuple(p["world"]),
                label=p.get("label", ""),
            )
            for p in raw
        ]
        print(f"  Loaded {len(ref_pts)} reference points from {pts_path.name}")

    # ── Interactive image mode ────────────────────────────────────────────────
    else:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"  ✗  Image not found: {image_path}", file=sys.stderr)
            sys.exit(1)

        _print_anchor_menu()

        if args.anchors:
            anchor_indices = [n - 1 for n in args.anchors]  # 1-based → 0-based
        else:
            raw = input(
                "  Enter anchor numbers (space-separated, ≥4 required): "
            ).strip()
            anchor_indices = [int(n) - 1 for n in raw.split()]

        if len(anchor_indices) < 4:
            print("  ✗  Fewer than 4 anchors selected.", file=sys.stderr)
            sys.exit(1)

        print(f"\n  Opening image — click each anchor in sequence:")
        for i, idx in enumerate(anchor_indices):
            print(f"    {i + 1}. {ANCHORS[idx]['label']}")
        print()

        ref_pts = _run_interactive_calibration(image_path, anchor_indices)

    # ── Compute H ─────────────────────────────────────────────────────────────
    H = calibrate_pitch(ref_pts)
    err = reprojection_error(ref_pts, H)

    quality = "GOOD" if err < 3.0 else "WARN — consider re-clicking" if err < 8.0 else "POOR — redo calibration"
    print(f"  ✓  Homography computed  |  reprojection error: {err:.2f} px  [{quality}]")

    saved = save_calibration(H, ref_pts, out_path=out_path, reprojection_error_px=err)
    print(f"  ✓  Calibration saved → {saved.relative_to(ROOT)}")
    print()


if __name__ == "__main__":
    main()
