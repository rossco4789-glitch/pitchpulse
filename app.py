"""
app.py — PitchPulse Desktop Dashboard
Tiverton Town FC | Tactical Analysis Command Centre

Tabs
----
  1. Match Ingestion & Context
  2. Veo Video Lab  (homography calibration + event picker)
  3. Agent Cockpit  (run pipeline + direct approval gate)
  4. Deliverables Hub

Launch
------
  streamlit run app.py

Architecture
------------
  - Calls existing functions from reconcile/, agents/, cv/, reports/.
  - run_matchday.py is NOT imported or modified.
  - Uploaded files are staged to data/raw/staged/ before backend reading.
  - Video frames are cached in st.session_state to avoid repeated cv2 reads.
  - The approval gate calls _build_dossier() + _write_dossier() directly;
    terminal input() is never invoked.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── Project root on sys.path ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Lazy project imports (avoid crashing if a dep is missing at startup) ───────
def _import_or_none(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception:
        return None

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR     = ROOT / "data" / "raw"
STAGED_DIR  = ROOT / "data" / "raw" / "staged"
PROC_DIR    = ROOT / "data" / "processed"
PLOTS_DIR   = PROC_DIR / "plots"
REPORTS_DIR = ROOT / "reports"
LEDGER_PATH = PROC_DIR / "match_ledger.json"
CONTEXT_PATH = RAW_DIR / "match_context.json"
DOF_CARD_PATH = PLOTS_DIR / "dof_match_card.png"
HTML_REPORT_PATH = PROC_DIR / "tivvy_tactical_dossier.html"
CALIB_PATH  = RAW_DIR / "camera_calibration.json"

for _d in (RAW_DIR, STAGED_DIR, PROC_DIR, PLOTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PitchPulse · Tiverton Town FC",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* OLED surface override for cards / expanders */
.stExpander > details            { background: #18181b; border: 1px solid #27272a; border-radius:8px; }
div[data-testid="stMetricValue"] { color: #f59e0b; font-family: monospace; }
div[data-testid="metric-container"] { background:#18181b; border-radius:8px; padding:12px; }
/* Amber headings */
h1, h2, h3 { color: #f59e0b !important; }
/* Sidebar polish */
section[data-testid="stSidebar"] { background: #0f0f11; }
/* Button styles */
.stButton > button[kind="primary"]   { background:#f59e0b; color:#09090b; font-weight:700; border:none; }
.stButton > button[kind="primary"]:hover { background:#d97706; }
.stButton > button[kind="secondary"] { border:1px solid #f59e0b; color:#f59e0b; background:transparent; }
/* Status badges */
.badge-green { background:#16a34a; color:#fff; padding:2px 8px; border-radius:4px; font-size:.8rem; }
.badge-amber { background:#d97706; color:#fff; padding:2px 8px; border-radius:4px; font-size:.8rem; }
.badge-red   { background:#dc2626; color:#fff; padding:2px 8px; border-radius:4px; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session-state initialisation (runs once per session)
# ══════════════════════════════════════════════════════════════════════════════

_SS_DEFAULTS: dict = {
    "ledger":           None,   # dict — reconciled match ledger
    "ctx":              None,   # dict — match_context.json
    "agent_outputs":    {},     # {agent_name: markdown_str}
    "dossier_content":  None,   # str — approved markdown dossier
    "dossier_path":     None,   # Path — written dossier file
    "rejection_count":  0,
    "rejection_feedback": "",
    "pipeline_log":     "",     # captured stdout from reconcile/visuals steps
    "approval_status":  None,   # None | "approved" | "rejected"
    # Video Lab
    "H":                None,   # np.ndarray — calibration matrix
    "calib_points":     [],     # list of {"pixel":[u,v], "world":[x,y], "label":str}
    "current_frame":    None,   # np.ndarray RGB — cached video frame
    "video_path":       "",
    "frame_index":      0,
    "last_click":       None,   # (u, v) from plotly selection
    "last_pitch_coord": None,   # (x_m, y_m) from picker
    "video_events_logged": 0,
}

for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _capture(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), capture stdout, return (result, captured_text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


def _stage_upload(uploaded_file, filename: str) -> Path:
    """Write a Streamlit UploadedFile buffer to STAGED_DIR and return the Path."""
    dest = STAGED_DIR / filename
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


def _load_parse_report():
    """importlib-load data/parse_report.py → return module."""
    spec = importlib.util.spec_from_file_location(
        "parse_report", ROOT / "data" / "parse_report.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _badge(text: str, colour: str = "green") -> str:
    return f'<span class="badge-{colour}">{text}</span>'


def _extract_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    """Extract one frame (RGB) from a local video file. Returns None on failure."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚽ PitchPulse")
    st.markdown("**Tiverton Town FC**  \nTactical Analysis Command Centre")
    st.divider()

    # Pipeline status
    ledger_ok  = LEDGER_PATH.exists()
    ctx_ok     = CONTEXT_PATH.exists()
    dof_ok     = DOF_CARD_PATH.exists()
    html_ok    = HTML_REPORT_PATH.exists()
    calib_ok   = CALIB_PATH.exists()

    st.markdown("### System Status")
    def _status_row(label: str, ok: bool):
        icon  = "🟢" if ok else "🔴"
        state = "Ready" if ok else "Missing"
        st.markdown(f"{icon} **{label}** — {state}")

    _status_row("Ledger",       ledger_ok)
    _status_row("Match Context",ctx_ok)
    _status_row("Calibration",  calib_ok)
    _status_row("DoF Card",     dof_ok)
    _status_row("HTML Report",  html_ok)

    st.divider()

    if st.session_state["ledger"]:
        summary = st.session_state["ledger"].get("summary", {})
        st.markdown("### Active Ledger")
        st.metric("Events tagged",  summary.get("total_events", "—"))
        st.metric("Matched",        summary.get("matched",      "—"))
        st.metric("Unmatched tags", summary.get("unmatched_tags","—"))
        st.metric("Substitutions",  summary.get("substitutions", "—"))

    st.divider()
    st.caption(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    st.caption("Local-only · No cloud · No paid APIs")


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📥 Match Ingestion",
    "🎥 Veo Video Lab",
    "🧠 Agent Cockpit",
    "📦 Deliverables Hub",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Match Ingestion & Context
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Match Ingestion & Context")
    st.caption("Stage tagger JSON exports and the Word match report. "
               "Files are written to `data/raw/` for backend processing.")

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── Left: uploaders ───────────────────────────────────────────────────────
    with col_left:
        st.subheader("1. Tag Exports (JSON)")
        json_files = st.file_uploader(
            "Upload tagger JSON(s) — one per period",
            type="json",
            accept_multiple_files=True,
            key="json_uploader",
        )
        if json_files:
            for jf in json_files:
                dest = _stage_upload(jf, jf.name)
                # Copy to raw/ so sync.py glob picks it up
                (RAW_DIR / jf.name).write_bytes(dest.read_bytes())
            st.success(f"✓ {len(json_files)} JSON file(s) staged → `data/raw/`")

            # Preview event count
            total_ev = 0
            for jf in json_files:
                try:
                    data = json.loads(jf.getvalue())
                    if isinstance(data, list):
                        total_ev += len(data)
                except Exception:
                    pass
            st.metric("Total events across files", total_ev)

        st.divider()

        st.subheader("2. Match Report (.docx)")
        docx_file = st.file_uploader(
            "Upload the Word match report",
            type=["docx"],
            key="docx_uploader",
        )
        if docx_file:
            staged_docx = _stage_upload(docx_file, docx_file.name)
            st.info(f"Staged: `{staged_docx.relative_to(ROOT)}`")

            if st.button("⚙ Parse Report", type="primary", key="btn_parse"):
                with st.spinner("Parsing…"):
                    try:
                        mod = _load_parse_report()
                        (ctx, stdout_txt) = _capture(mod.parse_report, staged_docx)
                        # write outputs to data/raw/
                        x_feed = mod.build_x_feed(ctx)
                        (RAW_DIR / "match_context.json").write_text(
                            json.dumps(ctx, indent=2, ensure_ascii=False),
                            encoding="utf-8"
                        )
                        (RAW_DIR / "tivvy_x_feed.json").write_text(
                            json.dumps(x_feed, indent=2, ensure_ascii=False),
                            encoding="utf-8"
                        )
                        st.session_state["ctx"] = ctx
                        st.success("✓ match_context.json + tivvy_x_feed.json written")
                        if stdout_txt.strip():
                            with st.expander("Parser log", expanded=False):
                                st.code(stdout_txt, language="text")
                    except Exception as exc:
                        st.error(f"Parse failed: {exc}")

    # ── Right: parsed context display ─────────────────────────────────────────
    with col_right:
        st.subheader("Parsed Match Context")

        # Load from disk if not in session state
        if st.session_state["ctx"] is None and CONTEXT_PATH.exists():
            st.session_state["ctx"] = json.loads(CONTEXT_PATH.read_text(encoding="utf-8"))

        ctx = st.session_state["ctx"]
        if ctx is None:
            st.markdown(
                "_No match context loaded. Upload and parse a .docx report._"
            )
        else:
            score  = ctx.get("score", {})
            tiv_g  = score.get("tiverton", "?")
            opp_g  = score.get("opponent", "?")
            opp    = ctx.get("opponent", "Unknown")
            result = ctx.get("result", "—")
            comp   = ctx.get("competition", "—")
            venue  = ctx.get("venue", "—")
            home   = "Home" if ctx.get("home_game", True) else "Away"
            date   = ctx.get("date", "—")

            st.markdown(f"### Tiverton {tiv_g}–{opp_g} {opp}")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Result",      result)
            col_b.metric("Competition", comp)
            col_c.metric("Venue",       f"{venue} ({home})")

            st.caption(f"Date: {date}")
            st.divider()

            # Starting XI
            lineup = ctx.get("lineup_tiverton", [])
            if lineup:
                st.markdown("**Starting XI**")
                for p in lineup:
                    flag = " *(sub)*" if p.get("is_sub") else ""
                    st.markdown(f"- #{p.get('number','?')} {p.get('name','?')}{flag}")

            # Scorers
            scorers = ctx.get("scorers", {})
            tiv_sc = scorers.get("tiverton", [])
            opp_sc = scorers.get("opponent", [])
            if tiv_sc or opp_sc:
                st.divider()
                st.markdown("**Scorers**")
                for s in tiv_sc:
                    st.markdown(f"⚽ **{s['player']}** {s.get('minute_raw', s['minute'])}′ *(Tiverton)*")
                for s in opp_sc:
                    st.markdown(f"⚽ {s['player']} {s.get('minute_raw', s['minute'])}′ *({opp})*")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Veo Video Lab
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Veo Video Lab")
    st.caption(
        "Calibrate camera homography from a local video frame, then pick pixel "
        "coordinates for video-assisted events. All computation is CPU-local."
    )

    # ── Video file picker ─────────────────────────────────────────────────────
    st.subheader("1. Video Frame")
    video_path_input = st.text_input(
        "Local .mp4 path",
        value=st.session_state["video_path"],
        placeholder="C:/Users/... /match.mp4",
        help="Enter the full path to the local Veo video file. "
             "Files are too large to upload — point directly to the file.",
        key="video_path_input",
    )
    st.session_state["video_path"] = video_path_input

    col_frame_left, col_frame_right = st.columns([3, 1])
    with col_frame_right:
        frame_index = st.number_input(
            "Frame #", min_value=0, value=st.session_state["frame_index"],
            step=50, key="frame_idx_input"
        )
        extract_btn = st.button("Extract Frame", type="primary", key="btn_extract")

    if extract_btn and video_path_input:
        with st.spinner("Reading frame…"):
            frame = _extract_frame(video_path_input, int(frame_index))
        if frame is None:
            st.error("Could not read frame. Check the file path and frame number.")
        else:
            st.session_state["current_frame"] = frame
            st.session_state["frame_index"]   = int(frame_index)
            st.session_state["last_click"]    = None  # reset click on new frame

    # ── Frame display via Plotly (click-to-pick pixel coordinates) ────────────
    frame = st.session_state.get("current_frame")
    if frame is not None:
        try:
            import plotly.express as px  # plotly ships with streamlit
            fig = px.imshow(
                frame,
                title=(
                    f"Frame {st.session_state['frame_index']}  "
                    "— click to record pixel coordinate"
                ),
            )
            fig.update_layout(
                paper_bgcolor="#09090b",
                plot_bgcolor="#09090b",
                font_color="#f8fafc",
                title_font_color="#f59e0b",
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=36, b=0),
                dragmode="select",
            )
            fig.update_xaxes(showticklabels=True, title="U (pixels)")
            fig.update_yaxes(showticklabels=True, title="V (pixels)")

            event_data = st.plotly_chart(
                fig,
                on_select="rerun",
                use_container_width=True,
                key="frame_chart",
            )

            # Capture click
            if (
                event_data is not None
                and hasattr(event_data, "selection")
                and event_data.selection.points
            ):
                pt = event_data.selection.points[0]
                st.session_state["last_click"] = (int(pt["x"]), int(pt["y"]))

            if st.session_state["last_click"]:
                u, v = st.session_state["last_click"]
                st.markdown(
                    f"📍 **Last click:** U={u}, V={v}  "
                    f"| Frame {st.session_state['frame_index']}"
                )

        except ImportError:
            st.image(frame, use_container_width=True, caption="Frame (install plotly for click picking)")

        with col_frame_right:
            st.caption(f"Frame: {st.session_state['frame_index']}")
            h_saved, w_saved = frame.shape[:2]
            st.caption(f"Size: {w_saved}×{h_saved} px")

    st.divider()

    # ── Calibration ───────────────────────────────────────────────────────────
    st.subheader("2. Homography Calibration")
    st.caption(
        "Click an anchor on the frame above, then add it below. "
        "4+ points required. World coordinates are the FIFA 105×68 m pitch."
    )

    try:
        from cv.calibration import (
            ANCHORS, ReferencePoint, calibrate_pitch,
            save_calibration, load_calibration,
        )
        calib_available = True
    except ImportError as e:
        st.warning(f"cv.calibration unavailable: {e}")
        calib_available = False

    if calib_available:
        col_cal_a, col_cal_b = st.columns([2, 1])

        with col_cal_a:
            anchor_labels = [a["label"] for a in ANCHORS]
            selected_anchor_label = st.selectbox(
                "Which anchor is the clicked point?",
                anchor_labels,
                key="anchor_select",
            )

        with col_cal_b:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Add Point", key="btn_add_calib"):
                click = st.session_state.get("last_click")
                if click is None:
                    st.warning("Click on the frame first.")
                else:
                    anchor = next(
                        (a for a in ANCHORS if a["label"] == selected_anchor_label), None
                    )
                    if anchor:
                        u, v = click
                        # Check for duplicate label
                        existing_labels = [p["label"] for p in st.session_state["calib_points"]]
                        if selected_anchor_label in existing_labels:
                            # Update existing
                            for p in st.session_state["calib_points"]:
                                if p["label"] == selected_anchor_label:
                                    p["pixel"] = [u, v]
                            st.success(f"Updated: {selected_anchor_label}")
                        else:
                            st.session_state["calib_points"].append({
                                "pixel": [u, v],
                                "world": anchor["world"],
                                "label": anchor["label"],
                            })
                            st.success(f"Added: {selected_anchor_label} → ({u}, {v})")

        # Show collected points
        n_pts = len(st.session_state["calib_points"])
        st.markdown(f"**Calibration points collected: {n_pts} / 4+ required**")
        if n_pts > 0:
            for i, pt in enumerate(st.session_state["calib_points"]):
                u, v = pt["pixel"]
                xw, yw = pt["world"]
                st.markdown(
                    f"  {i+1}. `{pt['label']}` "
                    f"— pixel ({u}, {v}) → world ({xw}m, {yw}m)"
                )
            if st.button("🗑 Clear All Points", key="btn_clear_calib"):
                st.session_state["calib_points"] = []
                st.session_state["H"] = None
                st.rerun()

        col_compute, col_load = st.columns(2)
        with col_compute:
            compute_disabled = n_pts < 4
            if st.button(
                "⚙ Compute & Save H",
                type="primary",
                disabled=compute_disabled,
                key="btn_compute_h",
            ):
                with st.spinner("Computing homography (RANSAC)…"):
                    try:
                        ref_pts = [
                            ReferencePoint(
                                pixel=tuple(p["pixel"]),
                                world=tuple(p["world"]),
                                label=p["label"],
                            )
                            for p in st.session_state["calib_points"]
                        ]
                        H = calibrate_pitch(ref_pts)
                        save_calibration(H, ref_pts)
                        st.session_state["H"] = H
                        st.success(
                            f"✓ H computed from {n_pts} points. "
                            f"Saved → `data/raw/camera_calibration.json`"
                        )
                    except Exception as exc:
                        st.error(f"Calibration failed: {exc}")

        with col_load:
            if st.button("📂 Load Saved Calibration", key="btn_load_calib"):
                if CALIB_PATH.exists():
                    try:
                        H = load_calibration(CALIB_PATH)
                        st.session_state["H"] = H
                        st.success("✓ Calibration loaded from disk.")
                    except Exception as exc:
                        st.error(f"Load failed: {exc}")
                else:
                    st.warning("No calibration file found.")

        if st.session_state["H"] is not None:
            st.markdown(
                '<span class="badge-green">✓ Calibration active</span>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Event Picker ──────────────────────────────────────────────────────────
    st.subheader("3. Video-Assisted Event Picker")
    st.caption(
        "Click a location on the frame, select the event type, "
        "and log it to the ledger. Requires an active calibration."
    )

    try:
        from cv.picker import pixel_to_pitch, build_video_event, append_video_event
        from cv.zones  import get_zone_by_coords
        picker_available = True
    except ImportError as e:
        st.warning(f"cv.picker unavailable: {e}")
        picker_available = False

    if picker_available:
        col_ev_a, col_ev_b, col_ev_c = st.columns([2, 2, 2])

        EVENT_TYPES = [
            "SHOT", "BOX_ENTRY", "HIGH_REGAIN", "DEF_TURNOVER",
            "SET_PIECE", "AERIAL_DUEL", "SECOND_BALL",
        ]
        SUB_TYPES: dict[str, list[str]] = {
            "SHOT":        ["ON_TARGET", "OFF_TARGET", "BLOCKED"],
            "BOX_ENTRY":   ["CENTRAL", "LEFT_FLANK", "RIGHT_FLANK"],
            "HIGH_REGAIN": ["WON"],
            "DEF_TURNOVER":["LOST"],
            "SET_PIECE":   ["ATT_CORNER", "DEF_CORNER", "FREE_KICK"],
            "AERIAL_DUEL": ["WON", "LOST"],
            "SECOND_BALL": ["WON", "LOST"],
        }

        with col_ev_a:
            ev_type = st.selectbox("Event type", EVENT_TYPES, key="ev_type")
            sub_type = st.selectbox(
                "Sub-type", SUB_TYPES.get(ev_type, [""]), key="ev_sub_type"
            )
            player_num = st.number_input("Player #", min_value=1, max_value=18, value=9, key="ev_player")

        with col_ev_b:
            period = st.selectbox("Period", ["1H", "2H", "ET1", "ET2"], key="ev_period")
            match_seconds = st.number_input(
                "Match seconds", min_value=0, max_value=7200, value=0,
                step=1, key="ev_seconds",
                help="0 = half time 45:00 → 2700, full time 90:00 → 5400"
            )
            # Derive pitch coord from last click + H
            H = st.session_state.get("H")
            click = st.session_state.get("last_click")

            if H is not None and click:
                u, v = click
                x_m, y_m = pixel_to_pitch(u, v, H)
                zone_id   = get_zone_by_coords(x_m, y_m)
                st.session_state["last_pitch_coord"] = (x_m, y_m)
                st.markdown(
                    f"**Pitch coord:** ({x_m:.1f}m, {y_m:.1f}m)  \n"
                    f"**Zone:** `{zone_id}`"
                )
            elif H is None:
                st.warning("No calibration active.")
            else:
                st.info("Click a frame location to get pitch coordinates.")

        with col_ev_c:
            st.markdown("<br>", unsafe_allow_html=True)
            log_disabled = H is None or click is None
            if st.button(
                "📝 Log to Ledger",
                type="primary",
                disabled=log_disabled,
                key="btn_log_event",
            ):
                try:
                    u, v = st.session_state["last_click"]
                    x_m, y_m = pixel_to_pitch(u, v, H)
                    zone_id   = get_zone_by_coords(x_m, y_m)
                    event = build_video_event(
                        u=u, v=v, H=H,
                        event_type=ev_type,
                        sub_type=sub_type,
                        player_num=int(player_num),
                        period=period,
                        match_seconds=int(match_seconds),
                    )
                    append_video_event(LEDGER_PATH, event)
                    st.session_state["video_events_logged"] += 1
                    st.success(
                        f"✓ Event logged: {ev_type}/{sub_type} "
                        f"at ({x_m:.1f}m, {y_m:.1f}m) zone={zone_id}"
                    )
                except Exception as exc:
                    st.error(f"Failed to log event: {exc}")

            if st.session_state["video_events_logged"] > 0:
                st.metric(
                    "Events logged this session",
                    st.session_state["video_events_logged"]
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Agent Cockpit & Approval Gate
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Agent Cockpit & Approval Gate")
    st.caption(
        "Run the full reconcile → visuals → agents pipeline, review each agent's "
        "output, then approve or reject the dossier — no terminal required."
    )

    # ── Pipeline runner ───────────────────────────────────────────────────────
    st.subheader("Pipeline Control")
    col_run, col_status = st.columns([2, 3])

    with col_run:
        run_pipeline = st.button(
            "▶ Run Full Pipeline",
            type="primary",
            key="btn_run_pipeline",
            use_container_width=True,
        )

    with col_status:
        if st.session_state["ledger"] is not None:
            st.markdown(
                '<span class="badge-green">✓ Ledger built</span>  '
                + (
                    '<span class="badge-green">✓ Agents run</span>'
                    if st.session_state["agent_outputs"]
                    else '<span class="badge-amber">⏸ Agents pending</span>'
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="badge-red">○ No ledger — run pipeline</span>',
                unsafe_allow_html=True,
            )

    if run_pipeline:
        log_lines: list[str] = []

        # Step 1: Reconcile
        with st.spinner("Step 1/3 — Reconciling events…"):
            try:
                from reconcile.sync import (
                    ensure_dirs, load_tag_events, load_club_feed,
                    extract_club_events, reconcile_events, build_ledger,
                    write_ledger, RECON_WINDOW_S,
                )
                ensure_dirs()
                (tag_events, t_log)     = _capture(load_tag_events, RAW_DIR)
                (feed_data, t2_log)     = _capture(load_club_feed, RAW_DIR)
                feed_data, feed_type    = feed_data  # load_club_feed returns tuple
                (club_events, t3_log)   = _capture(extract_club_events, feed_data, feed_type)
                (recon_out, t4_log)     = _capture(
                    reconcile_events, tag_events, club_events, window=RECON_WINDOW_S
                )
                matched, unmatched_tags, unmatched_club = recon_out
                (ledger, t5_log)        = _capture(build_ledger, matched, unmatched_tags, unmatched_club)
                write_ledger(ledger, LEDGER_PATH)
                st.session_state["ledger"] = ledger
                log_lines.append("✓ Step 1: Reconcile complete")
                log_lines.append(t_log + t2_log + t3_log + t4_log + t5_log)
            except Exception as exc:
                st.error(f"Reconcile failed: {exc}")
                st.stop()

        # Step 2: Visuals
        with st.spinner("Step 2/3 — Generating visuals…"):
            try:
                from reports.visualizer import (
                    plot_shot_map, plot_transition_map, plot_zonal_heatmap
                )
                PLOTS_DIR.mkdir(parents=True, exist_ok=True)
                ledger = st.session_state["ledger"]
                _, v1 = _capture(plot_shot_map,       ledger, PLOTS_DIR)
                _, v2 = _capture(plot_transition_map, ledger, PLOTS_DIR)
                _, v3 = _capture(plot_zonal_heatmap,  ledger, PLOTS_DIR)
                log_lines.append("✓ Step 2: Visuals generated")
            except Exception as exc:
                log_lines.append(f"⚠ Visuals skipped: {exc}")

        # Step 3: Agents
        with st.spinner("Step 3/3 — Running agents…"):
            try:
                from agents.synthesis import (
                    run_in_possession_agent, run_press_agent,
                    run_set_piece_agent, run_nonleague_agent,
                    _load_match_context,
                )
                ledger = st.session_state["ledger"]
                ctx, _ = _capture(_load_match_context)
                st.session_state["ctx"] = ctx

                feedback = st.session_state.get("rejection_feedback", "")
                outputs = {}
                for name, fn in [
                    ("In Possession",     run_in_possession_agent),
                    ("Press / LoE",       run_press_agent),
                    ("Set Pieces",        run_set_piece_agent),
                    ("Non-League Physics",run_nonleague_agent),
                ]:
                    result, _ = _capture(fn, ledger, feedback, ctx=ctx)
                    outputs[name] = result

                st.session_state["agent_outputs"]   = outputs
                st.session_state["approval_status"] = None
                st.session_state["dossier_content"] = None
                log_lines.append("✓ Step 3: All 4 agents complete")
            except Exception as exc:
                st.error(f"Agents failed: {exc}")
                st.stop()

        st.session_state["pipeline_log"] = "\n".join(log_lines)
        st.success("✓ Pipeline complete — review agent outputs below.")
        st.rerun()

    # Pipeline log
    if st.session_state["pipeline_log"]:
        with st.expander("Pipeline log", expanded=False):
            st.code(st.session_state["pipeline_log"], language="text")

    st.divider()

    # ── Agent output cards ────────────────────────────────────────────────────
    outputs = st.session_state.get("agent_outputs", {})
    if outputs:
        st.subheader("Agent Outputs")
        AGENT_ICONS = {
            "In Possession":      "⚡",
            "Press / LoE":        "🔵",
            "Set Pieces":         "🎯",
            "Non-League Physics": "💪",
        }
        for agent_name, markdown_text in outputs.items():
            icon = AGENT_ICONS.get(agent_name, "📊")
            with st.expander(f"{icon} Agent — {agent_name}", expanded=False):
                st.markdown(markdown_text)

    st.divider()

    # ── Approval Gate ─────────────────────────────────────────────────────────
    st.subheader("Approval Gate")

    approval_status = st.session_state.get("approval_status")
    rejection_count = st.session_state.get("rejection_count", 0)

    if not outputs:
        st.info("Run the pipeline first to unlock the approval gate.")
    else:
        if approval_status == "approved":
            dossier_path = st.session_state.get("dossier_path")
            st.markdown(
                '<span class="badge-green">✓ DOSSIER APPROVED</span>',
                unsafe_allow_html=True,
            )
            if dossier_path:
                st.caption(f"Written → `{Path(dossier_path).relative_to(ROOT)}`")
        else:
            col_approve, col_reject = st.columns([1, 2], gap="large")

            with col_approve:
                if st.button(
                    "✅ Approve Dossier",
                    type="primary",
                    key="btn_approve",
                    use_container_width=True,
                ):
                    with st.spinner("Writing dossier…"):
                        try:
                            from agents.synthesis import (
                                _build_dossier, _write_dossier,
                                _match_id_from_ledger,
                            )
                            ledger  = st.session_state["ledger"]
                            ctx     = st.session_state.get("ctx") or {}
                            content = _build_dossier(ledger, feedback="", ctx=ctx)
                            match_id= _match_id_from_ledger(ledger)
                            path    = _write_dossier(content, match_id)

                            # Build HTML report
                            try:
                                from reports.packager import build_html
                                build_html(
                                    content,
                                    ledger=ledger,
                                    plots_dir=PLOTS_DIR,
                                    out_dir=PROC_DIR,
                                )
                            except Exception:
                                pass  # HTML packager optional

                            # Build DoF card
                            try:
                                from reports.dof_card import build_dof_card
                                build_dof_card(ledger=ledger)
                            except Exception:
                                pass

                            st.session_state["dossier_content"] = content
                            st.session_state["dossier_path"]    = path
                            st.session_state["approval_status"] = "approved"
                            st.session_state["rejection_feedback"] = ""
                            st.success(f"✓ Dossier approved and written to disk.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Approval failed: {exc}")

            with col_reject:
                st.markdown("**Reject & Re-run with feedback**")
                feedback_text = st.text_area(
                    "Manager feedback for agents",
                    value=st.session_state.get("rejection_feedback", ""),
                    height=80,
                    key="feedback_input",
                    placeholder="e.g. Focus on right-flank press triggers; disregard the Zone 14 data.",
                )

                max_rejections = 3
                cycle_label = (
                    f"Cycle {rejection_count + 1} / {max_rejections}"
                    if rejection_count > 0
                    else ""
                )

                reject_disabled = rejection_count >= max_rejections
                if st.button(
                    f"🔁 Reject & Re-run  {cycle_label}",
                    type="secondary",
                    disabled=reject_disabled,
                    key="btn_reject",
                    use_container_width=True,
                ):
                    st.session_state["rejection_feedback"] = feedback_text
                    st.session_state["rejection_count"]    = rejection_count + 1

                    # Re-run agents with feedback
                    with st.spinner("Re-running agents with feedback…"):
                        try:
                            from agents.synthesis import (
                                run_in_possession_agent, run_press_agent,
                                run_set_piece_agent, run_nonleague_agent,
                            )
                            ledger  = st.session_state["ledger"]
                            ctx     = st.session_state.get("ctx") or {}
                            new_outputs = {}
                            for name, fn in [
                                ("In Possession",     run_in_possession_agent),
                                ("Press / LoE",       run_press_agent),
                                ("Set Pieces",        run_set_piece_agent),
                                ("Non-League Physics",run_nonleague_agent),
                            ]:
                                result, _ = _capture(fn, ledger, feedback_text, ctx=ctx)
                                new_outputs[name] = result
                            st.session_state["agent_outputs"] = new_outputs
                            st.success("Agents re-run. Review outputs above, then approve.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Re-run failed: {exc}")

                if reject_disabled:
                    st.warning(
                        f"Maximum {max_rejections} rejection cycles reached. "
                        "Approve to finalise."
                    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Deliverables Hub
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Deliverables Hub")
    st.caption("Preview and download the DoF match card and the full tactical dossier.")

    col_dof, col_doss = st.columns([1, 2], gap="large")

    # ── DoF Match Card ────────────────────────────────────────────────────────
    with col_dof:
        st.subheader("Director of Football Card")
        if DOF_CARD_PATH.exists():
            st.image(str(DOF_CARD_PATH), use_container_width=True)
            with open(DOF_CARD_PATH, "rb") as f:
                st.download_button(
                    "⬇ Download DoF Card (PNG)",
                    data=f,
                    file_name=DOF_CARD_PATH.name,
                    mime="image/png",
                    key="dl_dof",
                )
        else:
            st.info(
                "DoF card not generated yet.  \n"
                "Approve the dossier in the Agent Cockpit tab to produce it."
            )

        # Shot / heatmap previews
        for plot_name, label in [
            ("shot_map.png",        "Shot Map"),
            ("zonal_heatmap.png",   "Zonal Heatmap"),
            ("transition_map.png",  "Transition Map"),
        ]:
            plot_path = PLOTS_DIR / plot_name
            if plot_path.exists():
                st.divider()
                st.markdown(f"**{label}**")
                st.image(str(plot_path), use_container_width=True)

    # ── HTML Dossier ──────────────────────────────────────────────────────────
    with col_doss:
        st.subheader("Tactical Dossier")

        # Prefer the session-state content (freshest), fall back to disk
        dossier_content = st.session_state.get("dossier_content")
        if dossier_content is None and HTML_REPORT_PATH.exists():
            html_content = HTML_REPORT_PATH.read_text(encoding="utf-8")
        elif dossier_content is not None and not HTML_REPORT_PATH.exists():
            # Render raw markdown if HTML packager hasn't run yet
            html_content = None
        elif HTML_REPORT_PATH.exists():
            html_content = HTML_REPORT_PATH.read_text(encoding="utf-8")
        else:
            html_content = None

        if html_content:
            st.components.v1.html(html_content, height=720, scrolling=True)
            st.download_button(
                "⬇ Download HTML Dossier",
                data=html_content.encode("utf-8"),
                file_name=HTML_REPORT_PATH.name,
                mime="text/html",
                key="dl_html",
            )
        elif dossier_content:
            # Fallback: render markdown directly in Streamlit
            st.markdown(dossier_content)
            st.download_button(
                "⬇ Download Markdown Dossier",
                data=dossier_content.encode("utf-8"),
                file_name="tivvy_tactical_dossier.md",
                mime="text/markdown",
                key="dl_md",
            )

            # Find the written .md file
            dossier_path = st.session_state.get("dossier_path")
            if dossier_path and Path(dossier_path).exists():
                with open(dossier_path, "rb") as f:
                    st.download_button(
                        f"⬇ Download {Path(dossier_path).name}",
                        data=f,
                        file_name=Path(dossier_path).name,
                        mime="text/markdown",
                        key="dl_md2",
                    )
        else:
            st.info(
                "No dossier available yet.  \n"
                "Run the pipeline and approve the dossier in the Agent Cockpit tab."
            )
