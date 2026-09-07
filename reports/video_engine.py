"""
reports/video_engine.py
FFmpeg video slicing engine for Tiverton Town FC match clip export.

Provides:
    check_ffmpeg()               — verify FFmpeg is on PATH
    calculate_clip_bounds(...)   — map match_seconds → Veo file timestamps
    slice_clip(...)              — stream-copy a clip from the source MP4
    export_playlist_m3u(...)     — write an M3U playlist for VLC / mpv

Clip naming convention:
    data/clips/{DD-MM-YYYY}_{opponent-slug}/{minute}m_{zone}_{event}_{sub}.mp4

Sync model (dual-offset):
    First half:   veo_t = offset_1h + match_seconds
    Second half:  veo_t = offset_2h + (match_seconds - 2700)
    ET1:          veo_t = offset_2h + (match_seconds - 2700)   [best effort]
    ET2:          veo_t = offset_2h + (match_seconds - 2700)   [best effort]

The caller supplies offset_1h as "seconds into the Veo file at which the
first-half kick-off whistle sounded", and offset_2h as the equivalent for
the second-half restart whistle.  This eliminates half-time stoppage drift.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


# ══════════════════════════════════════════════════════════════════════════════
# FFmpeg availability
# ══════════════════════════════════════════════════════════════════════════════

def check_ffmpeg() -> bool:
    """Return True if FFmpeg is found on PATH, False otherwise."""
    return shutil.which("ffmpeg") is not None


# ══════════════════════════════════════════════════════════════════════════════
# Timestamp arithmetic
# ══════════════════════════════════════════════════════════════════════════════

def calculate_clip_bounds(
    match_seconds: int,
    period: str,
    offset_1h: float,
    offset_2h: float,
    lead_in: float = 5.0,
    follow_through: float = 3.0,
) -> tuple[float, float]:
    """
    Convert a tagged event's match_seconds to Veo file timestamps.

    Parameters
    ----------
    match_seconds   : elapsed seconds since the first-half kick-off whistle
    period          : "1H", "2H", "ET1", or "ET2"
    offset_1h       : seconds into the Veo file at the 1H kick-off whistle
    offset_2h       : seconds into the Veo file at the 2H restart whistle
    lead_in         : seconds of footage before the event (default 5s)
    follow_through  : seconds of footage after the event (default 3s)

    Returns
    -------
    (start_s, end_s) — absolute seconds from the start of the Veo file,
    clamped so start_s >= 0.
    """
    if period in ("1H",):
        veo_event = offset_1h + match_seconds
    else:
        # 2H / ET1 / ET2: measure from the second-half offset.
        # match_seconds for a 2H event is cumulative from kick-off (e.g. 70' = 4200s);
        # subtract the nominal 45-minute boundary (2700s) to get time into the half.
        elapsed_in_half = max(0, match_seconds - 2700)
        veo_event = offset_2h + elapsed_in_half

    start_s = max(0.0, veo_event - lead_in)
    end_s   = veo_event + follow_through
    return start_s, end_s


# ══════════════════════════════════════════════════════════════════════════════
# Clip output path helpers
# ══════════════════════════════════════════════════════════════════════════════

def _slugify(text: str) -> str:
    """Lower-case, replace spaces and special chars with hyphens."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text


def build_clip_path(
    clips_root: Path,
    match_date: str,
    opponent: str,
    match_seconds: int,
    event_type: str,
    sub_type: str = "",
    zone_id: str = "",
) -> Path:
    """
    Construct a structured output path for a single clip.

    Example:
        data/clips/15-08-2026_st-blazey/72m_a-lc_shot_on-target.mp4
    """
    clips_root    = Path(clips_root)
    minute        = match_seconds // 60
    folder_name   = f"{match_date}_{_slugify(opponent)}"
    zone_part     = _slugify(zone_id)  if zone_id  else "unknown-zone"
    event_part    = _slugify(event_type)
    sub_part      = f"_{_slugify(sub_type)}" if sub_type else ""
    filename      = f"{minute}m_{zone_part}_{event_part}{sub_part}.mp4"
    out_dir       = clips_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


# ══════════════════════════════════════════════════════════════════════════════
# FFmpeg slice
# ══════════════════════════════════════════════════════════════════════════════

def slice_clip(
    video_path: str | Path,
    start_s: float,
    end_s: float,
    output_path: str | Path,
) -> tuple[bool, str]:
    """
    Extract a clip from video_path using fast input seek + stream copy.

    Uses `ffmpeg -ss <start> -to <end> -i <input> -c copy -y <output>`.
    No re-encoding; near-instant on any CPU.

    Returns
    -------
    (success: bool, message: str)
    """
    if not check_ffmpeg():
        return False, "FFmpeg not found on PATH. Install with: winget install Gyan.FFmpeg"

    video_path  = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        return False, f"Video file not found: {video_path}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-ss", f"{start_s:.3f}",
        "-to", f"{end_s:.3f}",
        "-i",  str(video_path),
        "-c",  "copy",
        "-y",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            # FFmpeg often writes useful info to stderr
            err_tail = result.stderr.strip().splitlines()
            err_msg  = err_tail[-1] if err_tail else "unknown FFmpeg error"
            return False, f"FFmpeg error: {err_msg}"
        return True, str(output_path)
    except subprocess.TimeoutExpired:
        return False, "FFmpeg timed out (>60s) — check source file."
    except Exception as exc:
        return False, f"Subprocess error: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# M3U playlist export
# ══════════════════════════════════════════════════════════════════════════════

def export_playlist_m3u(
    clip_paths: Sequence[Path | str],
    m3u_path: Path | str,
) -> None:
    """
    Write an extended M3U playlist file for the given clip paths.
    Opens in VLC, mpv, Windows Media Player, and most media players.
    """
    m3u_path = Path(m3u_path)
    m3u_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["#EXTM3U"]
    for p in clip_paths:
        p = Path(p)
        # Duration -1 = unknown (fine for VLC / mpv)
        lines.append(f"#EXTINF:-1,{p.stem}")
        lines.append(str(p.resolve()))

    m3u_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
