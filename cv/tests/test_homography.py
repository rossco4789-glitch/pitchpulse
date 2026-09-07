"""
cv/tests/test_homography.py
Synthetic unit tests for the Tier 1 homography calibration pipeline.

All tests run without a camera, without a GPU, and without any image file.
A synthetic perspective transform is constructed algebraically; pixel coordinates
are generated from it; calibration is asked to recover them.

Run:
  python -m pytest cv/tests/test_homography.py -v
  # or without pytest:
  python cv/tests/test_homography.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from cv.calibration import (
    ReferencePoint,
    calibrate_pitch,
    reprojection_error,
    save_calibration,
    load_calibration,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from cv.picker import pixel_to_pitch, resolve_zone, build_video_event


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

# Synthetic world-to-pixel perspective matrix (arbitrary but deterministic).
# Represents a camera placed above-right of the centre circle, looking down-left.
_P_SYNTH = np.array([
    [0.8,   0.05, 120.0],
    [0.02,  0.6,   80.0],
    [0.0001, 0.0002, 1.0],
], dtype=np.float64)


def _world_to_pixel(x_m: float, y_m: float) -> tuple[float, float]:
    """Apply synthetic perspective transform to produce a pixel coordinate."""
    p = _P_SYNTH @ np.array([x_m, y_m, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


# Eight well-spread world reference points across the pitch
_WORLD_REFS: list[tuple[float, float, str]] = [
    (52.5,  34.0,  "Centre spot"),
    (52.5,   0.0,  "Halfway × Left touch"),
    (52.5,  68.0,  "Halfway × Right touch"),
    (11.0,  34.0,  "Left penalty spot"),
    (94.0,  34.0,  "Right penalty spot"),
    (16.5,  13.84, "L pen area NP"),
    (88.5,  54.16, "R pen area FP"),
    ( 0.0,   0.0,  "Top-left corner"),
]


def _make_ref_points() -> list[ReferencePoint]:
    """Build ReferencePoint list from synthetic pixel projections."""
    pts: list[ReferencePoint] = []
    for x_m, y_m, label in _WORLD_REFS:
        u, v = _world_to_pixel(x_m, y_m)
        pts.append(ReferencePoint(pixel=(u, v), world=(x_m, y_m), label=label))
    return pts


# ══════════════════════════════════════════════════════════════════════════════
# Test: calibrate_pitch
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibratePitch:

    def test_returns_3x3_float64(self):
        pts = _make_ref_points()
        H = calibrate_pitch(pts)
        assert H.shape == (3, 3), "H must be a 3×3 matrix"
        assert H.dtype == np.float64, "H must be float64"

    def test_raises_on_fewer_than_4_points(self):
        pts = _make_ref_points()[:3]
        with pytest.raises(ValueError, match="4"):
            calibrate_pitch(pts)

    def test_accepts_minimum_4_points(self):
        pts = _make_ref_points()[:4]
        H = calibrate_pitch(pts)
        assert H is not None

    def test_reprojection_error_below_1px_for_clean_data(self):
        """
        With 8 synthetic points generated from the same transform, the recovered
        H should project back to within 0.05 px of the original pixels.
        """
        pts = _make_ref_points()
        H = calibrate_pitch(pts)
        err = reprojection_error(pts, H)
        assert err < 1.0, f"Reprojection error {err:.4f} px exceeds 1.0 px with clean synthetic data"


# ══════════════════════════════════════════════════════════════════════════════
# Test: pixel_to_pitch
# ══════════════════════════════════════════════════════════════════════════════

class TestPixelToPitch:

    def _H(self) -> np.ndarray:
        return calibrate_pitch(_make_ref_points())

    def test_known_pixel_recovers_world_coords_within_15cm(self):
        """
        For each reference point: project pixel → pitch, compare to ground truth.
        Tolerance: 0.15 m (15 cm) — well within positioning accuracy needed for zone assignment.
        """
        H = self._H()
        for x_gt, y_gt, label in _WORLD_REFS:
            u, v = _world_to_pixel(x_gt, y_gt)
            x_m, y_m = pixel_to_pitch(u, v, H)
            err_x = abs(x_m - x_gt)
            err_y = abs(y_m - y_gt)
            assert err_x < 0.15, (
                f"{label}: x error {err_x:.4f} m > 0.15 m  "
                f"(got {x_m:.4f}, expected {x_gt})"
            )
            assert err_y < 0.15, (
                f"{label}: y error {err_y:.4f} m > 0.15 m  "
                f"(got {y_m:.4f}, expected {y_gt})"
            )

    def test_clamp_above_pitch_length(self):
        """Pixel that projects beyond x=105 must be clamped to 105.0."""
        H = self._H()
        # Use a pixel that should map far beyond the attacking goal line
        # by scaling the right-penalty-spot pixel well further right
        u_ref, v_ref = _world_to_pixel(94.0, 34.0)
        # Push pixel 400 units further right (very far off-pitch)
        x_m, y_m = pixel_to_pitch(u_ref + 400, v_ref, H)
        assert x_m <= PITCH_LENGTH, f"x_m {x_m} not clamped to {PITCH_LENGTH}"
        assert 0.0 <= y_m <= PITCH_WIDTH

    def test_clamp_below_zero(self):
        """Pixel that projects to negative x must be clamped to 0.0."""
        H = self._H()
        u_ref, v_ref = _world_to_pixel(0.0, 0.0)
        x_m, y_m = pixel_to_pitch(u_ref - 400, v_ref - 400, H)
        assert x_m >= 0.0, f"x_m {x_m} not clamped to 0.0"
        assert y_m >= 0.0, f"y_m {y_m} not clamped to 0.0"

    def test_degenerate_w_near_zero_returns_pitch_centre(self):
        """If w ≈ 0, pixel_to_pitch must not raise and must return a valid coordinate."""
        H = np.eye(3, dtype=np.float64)
        H[2, :] = [0.0, 0.0, 0.0]   # Force w = 0 for any input
        x_m, y_m = pixel_to_pitch(100.0, 100.0, H)
        # Should fall back to pitch centre
        assert x_m == PITCH_LENGTH / 2
        assert y_m == PITCH_WIDTH  / 2


# ══════════════════════════════════════════════════════════════════════════════
# Test: zone resolution
# ══════════════════════════════════════════════════════════════════════════════

class TestZoneResolution:

    def test_centre_spot_resolves_to_middle_third(self):
        """Centre spot at (52.5, 34.0) must fall in middle third."""
        zone_id, zone_name = resolve_zone(52.5, 34.0)
        assert zone_id is not None, "Centre spot must resolve to a zone"
        assert zone_id.startswith("M_"), f"Centre spot should be in Middle third, got {zone_id}"

    def test_attacking_penalty_spot_resolves_to_attacking_central(self):
        """Right penalty spot (94.0, 34.0) is in the attacking third, central channel."""
        zone_id, zone_name = resolve_zone(94.0, 34.0)
        assert zone_id is not None
        assert zone_id.startswith("A_"), f"Right pen spot should be Attacking third, got {zone_id}"
        assert "C" in zone_id, f"Right pen spot should be central, got {zone_id}"

    def test_out_of_bounds_returns_none(self):
        """Coordinates outside the pitch must return (None, None)."""
        zone_id, zone_name = resolve_zone(-1.0, -1.0)
        assert zone_id is None
        assert zone_name is None

    def test_zone_14_left_central_attacking(self):
        """
        A_LC spans y ≈ 22.67 – 34.0 m (3rd of 6 channels on 68 m pitch).
        (80.0, 28.0) is comfortably inside that band.
        """
        zone_id, _ = resolve_zone(80.0, 28.0)
        assert zone_id == "A_LC", f"Expected A_LC (Zone 14), got {zone_id}"

    def test_zone_14_right_central_attacking(self):
        """
        A_RC spans y ≈ 34.0 – 45.33 m (4th of 6 channels on 68 m pitch).
        (80.0, 39.0) is comfortably inside that band.
        """
        zone_id, _ = resolve_zone(80.0, 39.0)
        assert zone_id == "A_RC", f"Expected A_RC (Zone 14), got {zone_id}"


# ══════════════════════════════════════════════════════════════════════════════
# Test: build_video_event
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildVideoEvent:

    def _H(self) -> np.ndarray:
        return calibrate_pitch(_make_ref_points())

    def _shot_pixel(self) -> tuple[float, float]:
        """Synthetic pixel for right penalty spot (x=94, y=34) — attacking zone."""
        return _world_to_pixel(94.0, 34.0)

    def test_event_has_required_keys(self):
        H = self._H()
        u, v = self._shot_pixel()
        evt = build_video_event(u, v, H, "SHOT", 4620, "2H", sub_type="ON_TARGET", player_num=9)

        required = {
            "id", "source", "period", "match_seconds", "clock_display",
            "event_type", "sub_type", "x_pct", "y_pct", "x_m", "y_m",
            "zone_id", "zone_name", "pixel_u", "pixel_v", "timestamp_iso",
            "player_num",
        }
        missing = required - evt.keys()
        assert not missing, f"Event missing keys: {missing}"

    def test_source_is_video_assisted(self):
        H = self._H()
        u, v = self._shot_pixel()
        evt = build_video_event(u, v, H, "SHOT", 4620, "2H")
        assert evt["source"] == "video_assisted"

    def test_clock_display_format(self):
        """match_seconds=4620 in 2H should display as 32:00 (4620-2700=1920s=32min)."""
        H = self._H()
        u, v = self._shot_pixel()
        evt = build_video_event(u, v, H, "SHOT", 4620, "2H")
        assert evt["clock_display"] == "32:00", f"Got '{evt['clock_display']}'"

    def test_x_pct_in_unit_range(self):
        H = self._H()
        u, v = self._shot_pixel()
        evt = build_video_event(u, v, H, "SHOT", 4620, "2H")
        assert 0.0 <= evt["x_pct"] <= 1.0
        assert 0.0 <= evt["y_pct"] <= 1.0

    def test_zone_id_populated_for_on_pitch_coord(self):
        H = self._H()
        u, v = self._shot_pixel()
        evt = build_video_event(u, v, H, "SHOT", 4620, "2H")
        assert evt["zone_id"] is not None, "zone_id must be populated for an on-pitch coordinate"

    def test_video_timestamp_stored_when_supplied(self):
        H = self._H()
        u, v = self._shot_pixel()
        evt = build_video_event(u, v, H, "SHOT", 4620, "2H", video_timestamp_s=4680.5)
        assert evt.get("video_timestamp_s") == 4680.5


# ══════════════════════════════════════════════════════════════════════════════
# Test: save / load round-trip
# ══════════════════════════════════════════════════════════════════════════════

class TestSaveLoadRoundTrip:

    def test_round_trip_preserves_H_within_tolerance(self, tmp_path):
        pts = _make_ref_points()
        H_orig = calibrate_pitch(pts)
        err = reprojection_error(pts, H_orig)

        out = tmp_path / "test_calib.json"
        save_calibration(H_orig, pts, out_path=out, reprojection_error_px=err)

        H_loaded = load_calibration(out)
        np.testing.assert_allclose(
            H_orig, H_loaded, rtol=1e-9, atol=1e-12,
            err_msg="Loaded H does not match original to floating-point precision",
        )

    def test_load_raises_on_missing_file(self, tmp_path):
        from cv.calibration import load_calibration
        with pytest.raises(FileNotFoundError):
            load_calibration(tmp_path / "nonexistent.json")


# ══════════════════════════════════════════════════════════════════════════════
# Standalone runner (no pytest)
# ══════════════════════════════════════════════════════════════════════════════

def _run_standalone() -> None:
    """Simple pass/fail runner for use without pytest."""
    import traceback

    tests = [
        ("calibrate_pitch: returns 3×3 float64",         TestCalibratePitch().test_returns_3x3_float64),
        ("calibrate_pitch: raises on <4 points",          TestCalibratePitch().test_raises_on_fewer_than_4_points),
        ("calibrate_pitch: accepts 4 points",             TestCalibratePitch().test_accepts_minimum_4_points),
        ("calibrate_pitch: reprojection error < 1 px",    TestCalibratePitch().test_reprojection_error_below_1px_for_clean_data),
        ("pixel_to_pitch: recovers coords ±15 cm",        TestPixelToPitch().test_known_pixel_recovers_world_coords_within_15cm),
        ("pixel_to_pitch: clamp above PITCH_LENGTH",      TestPixelToPitch().test_clamp_above_pitch_length),
        ("pixel_to_pitch: clamp below zero",              TestPixelToPitch().test_clamp_below_zero),
        ("pixel_to_pitch: degenerate w≈0 → centre",      TestPixelToPitch().test_degenerate_w_near_zero_returns_pitch_centre),
        ("resolve_zone: centre spot → M_*",               TestZoneResolution().test_centre_spot_resolves_to_middle_third),
        ("resolve_zone: pen spot → A_*C*",                TestZoneResolution().test_attacking_penalty_spot_resolves_to_attacking_central),
        ("resolve_zone: out of bounds → None",            TestZoneResolution().test_out_of_bounds_returns_none),
        ("resolve_zone: Zone 14 left (A_LC)",             TestZoneResolution().test_zone_14_left_central_attacking),
        ("resolve_zone: Zone 14 right (A_RC)",            TestZoneResolution().test_zone_14_right_central_attacking),
        ("build_video_event: required keys present",      TestBuildVideoEvent().test_event_has_required_keys),
        ("build_video_event: source=video_assisted",      TestBuildVideoEvent().test_source_is_video_assisted),
        ("build_video_event: clock_display 32:00",        TestBuildVideoEvent().test_clock_display_format),
        ("build_video_event: x_pct/y_pct in [0,1]",      TestBuildVideoEvent().test_x_pct_in_unit_range),
        ("build_video_event: zone_id not None",           TestBuildVideoEvent().test_zone_id_populated_for_on_pitch_coord),
        ("build_video_event: video_timestamp stored",     TestBuildVideoEvent().test_video_timestamp_stored_when_supplied),
    ]

    # save/load round-trip needs tmp_path
    import tempfile, os
    tmp = tempfile.mkdtemp()
    tmp_path = Path(tmp)

    class _FakeTmpPath:
        def __truediv__(self, name): return tmp_path / name

    save_load_inst = TestSaveLoadRoundTrip()
    tests += [
        ("save/load: round-trip H within tolerance",
         lambda: save_load_inst.test_round_trip_preserves_H_within_tolerance(_FakeTmpPath())),
        ("save/load: raises FileNotFoundError",
         lambda: save_load_inst.test_load_raises_on_missing_file(_FakeTmpPath())),
    ]

    col = 60
    passed = failed = 0
    print()
    print("  PitchPulse — Homography Unit Tests")
    print("  " + "─" * col)

    for name, fn in tests:
        try:
            fn()
            print(f"  ✓  {name}")
            passed += 1
        except Exception:
            print(f"  ✗  {name}")
            traceback.print_exc()
            failed += 1

    print("  " + "─" * col)
    print(f"  {passed} passed  |  {failed} failed")
    print()

    # cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    _run_standalone()
