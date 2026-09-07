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

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR          = ROOT / "data" / "raw"
STAGED_DIR       = ROOT / "data" / "raw" / "staged"
PROC_DIR         = ROOT / "data" / "processed"
PLOTS_DIR        = PROC_DIR / "plots"
REPORTS_DIR      = ROOT / "reports"
LEDGER_PATH      = PROC_DIR / "match_ledger.json"
CONTEXT_PATH     = RAW_DIR / "match_context.json"
DOF_CARD_PATH    = PLOTS_DIR / "dof_match_card.png"
HTML_REPORT_PATH = PROC_DIR / "tivvy_tactical_dossier.html"
CALIB_PATH       = RAW_DIR / "camera_calibration.json"

for _d in (RAW_DIR, STAGED_DIR, PROC_DIR, PLOTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PitchPulse · Tiverton Town FC",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — injected into the Streamlit shell
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;900&family=Inter:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── ERASE STREAMLIT CHROME ─────────────────────────────────────────────── */
#MainMenu, footer, header, [data-testid="stToolbar"],
.stDeployButton, [data-testid="stDecoration"] { visibility: hidden !important; }
[data-testid="stHeader"] { display: none !important; }

/* ── TOKENS ────────────────────────────────────────────────────────────────── */
:root {
  --bg:         #08080a;
  --surface:    #111116;
  --surface-hi: #1a1a22;
  --border:     rgba(255,255,255,0.055);
  --amber:      #f59e0b;
  --amber-dim:  rgba(245,158,11,0.11);
  --amber-glow: rgba(245,158,11,0.22);
  --green:      #22c55e;
  --green-dim:  rgba(34,197,94,0.12);
  --red:        #ef4444;
  --red-dim:    rgba(239,68,68,0.12);
  --blue:       #3b82f6;
  --purple:     #a855f7;
  --text:       #ededf0;
  --text-2:     #71717a;
  --text-3:     #3f3f46;
  --mono:       'JetBrains Mono', 'Fira Code', monospace;
  --sans:       'Inter', system-ui, sans-serif;
  --display:    'Barlow Condensed', Impact, sans-serif;
}

/* ── GROUND ────────────────────────────────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] { background: var(--bg) !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px !important; }

/* ── SIDEBAR ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: #0b0b0e !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] label { color: var(--text-2) !important; font-size: .75rem !important; }

/* ── TABS ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  gap: 2px !important;
  border: 1px solid var(--border) !important;
  margin-bottom: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-2) !important;
  border-radius: 7px !important;
  padding: 8px 22px !important;
  font-family: var(--sans) !important;
  font-weight: 600 !important;
  font-size: .82rem !important;
  letter-spacing: .025em !important;
  border: none !important;
  outline: none !important;
  transition: color .2s !important;
}
.stTabs [aria-selected="true"] {
  background: var(--amber-dim) !important;
  color: var(--amber) !important;
}
.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── BUTTONS ─────────────────────────────────────────────────────────────── */
.stButton > button {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: var(--sans) !important;
  font-weight: 600 !important;
  font-size: .82rem !important;
  border-radius: 8px !important;
  letter-spacing: .02em !important;
  transition: border-color .15s, color .15s, box-shadow .15s !important;
}
.stButton > button:hover {
  border-color: var(--amber) !important;
  color: var(--amber) !important;
  background: var(--amber-dim) !important;
}
.stButton > button[kind="primary"] {
  background: var(--amber) !important;
  border-color: var(--amber) !important;
  color: #09090b !important;
  font-weight: 700 !important;
  letter-spacing: .04em !important;
}
.stButton > button[kind="primary"]:hover {
  background: #d97706 !important;
  border-color: #d97706 !important;
  color: #09090b !important;
  box-shadow: 0 0 28px rgba(245,158,11,.28) !important;
}
.stButton > button:disabled { opacity: .35 !important; cursor: not-allowed !important; }

/* ── INPUTS ──────────────────────────────────────────────────────────────── */
.stTextInput > label, .stNumberInput > label,
.stSelectbox > label, .stTextArea > label,
.stFileUploader > label { color: var(--text-2) !important; font-family: var(--sans) !important;
  font-size: .75rem !important; font-weight: 600 !important; letter-spacing: .08em !important;
  text-transform: uppercase !important; }
.stTextInput input, .stNumberInput input, .stTextArea textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: .85rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--amber) !important;
  box-shadow: 0 0 0 3px var(--amber-dim) !important;
  outline: none !important;
}
[data-baseweb="select"] > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: .85rem !important;
}
[data-baseweb="select"] > div:focus-within {
  border-color: var(--amber) !important;
  box-shadow: 0 0 0 3px var(--amber-dim) !important;
}
[data-baseweb="popover"] { background: var(--surface-hi) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
[role="option"] { background: transparent !important; color: var(--text) !important; font-family: var(--mono) !important; font-size: .85rem !important; }
[role="option"]:hover, [aria-selected="true"] { background: var(--amber-dim) !important; color: var(--amber) !important; }

/* ── FILE UPLOADER ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] section {
  background: var(--surface) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 12px !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"] section:hover {
  border-color: var(--amber) !important;
  background: var(--amber-dim) !important;
}
[data-testid="stFileUploader"] section p { color: var(--text-2) !important; font-family: var(--sans) !important; }
[data-testid="stFileUploader"] section svg { opacity: .4 !important; }

/* ── ALERTS ──────────────────────────────────────────────────────────────── */
.stSuccess { background: var(--green-dim) !important; border: 1px solid rgba(34,197,94,.3) !important;
  border-radius: 8px !important; color: #86efac !important; }
.stError   { background: var(--red-dim)   !important; border: 1px solid rgba(239,68,68,.3)  !important;
  border-radius: 8px !important; color: #fca5a5 !important; }
.stWarning { background: var(--amber-dim) !important; border: 1px solid rgba(245,158,11,.3) !important;
  border-radius: 8px !important; color: #fcd34d !important; }
.stInfo    { background: rgba(59,130,246,.1) !important; border: 1px solid rgba(59,130,246,.3) !important;
  border-radius: 8px !important; color: #93c5fd !important; }

/* ── EXPANDERS ───────────────────────────────────────────────────────────── */
.stExpander { border: 1px solid var(--border) !important; border-radius: 10px !important;
  background: var(--surface) !important; }
.stExpander > details > summary { color: var(--text) !important; font-family: var(--sans) !important;
  font-weight: 600 !important; font-size: .85rem !important; }
.stExpander > details > summary:hover { color: var(--amber) !important; }

/* ── METRICS (native fallback) ───────────────────────────────────────────── */
[data-testid="stMetricLabel"] { color: var(--text-2) !important; font-family: var(--sans) !important;
  font-size: .72rem !important; text-transform: uppercase !important; letter-spacing: .08em !important; }
[data-testid="stMetricValue"] { color: var(--amber) !important; font-family: var(--mono) !important; }

/* ── SPINNER ─────────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--amber) !important; }

/* ── SCROLLBAR ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber); }

/* ── UTILITY CLASSES ─────────────────────────────────────────────────────── */
.pp-mono { font-family: var(--mono) !important; }
.pp-display { font-family: var(--display) !important; }

/* ── PULSE ANIMATION ─────────────────────────────────────────────────────── */
@keyframes ppulse {
  0%, 100% { opacity: 1; }
  50% { opacity: .4; }
}
@keyframes amberglow {
  0%, 100% { box-shadow: 0 0 20px rgba(245,158,11,.18); }
  50% { box-shadow: 0 0 36px rgba(245,158,11,.36); }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HTML COMPONENT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _tile(label: str, value: str, sub: str = "", color: str = "#f59e0b") -> str:
    sub_html = f'<div style="color:#71717a;font-size:.72rem;font-family:var(--sans);margin-top:6px">{sub}</div>' if sub else ""
    return f"""
    <div style="background:#111116;border:1px solid rgba(255,255,255,0.055);
                border-radius:10px;padding:20px 20px;text-align:center;">
      <div style="color:#71717a;font-size:.65rem;text-transform:uppercase;
                  letter-spacing:.14em;font-family:'Inter',sans-serif;margin-bottom:10px">{label}</div>
      <div style="color:{color};font-size:1.9rem;font-weight:500;
                  font-family:'JetBrains Mono',monospace;line-height:1">{value}</div>
      {sub_html}
    </div>"""


def _status_row(label: str, ok: bool) -> str:
    col = "#22c55e" if ok else "#3f3f46"
    state = "READY" if ok else "——"
    pulse = "animation:ppulse 2.5s ease-in-out infinite;" if ok else ""
    return f"""
    <div style="display:flex;align-items:center;gap:10px;padding:5px 0;
                border-bottom:1px solid rgba(255,255,255,0.04)">
      <span style="width:7px;height:7px;border-radius:50%;background:{col};
                   flex-shrink:0;{pulse}"></span>
      <span style="font-family:'Inter',sans-serif;font-size:.8rem;color:#ededf0;flex:1">{label}</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:{col}">{state}</span>
    </div>"""


def _section_label(text: str, color: str = "#f59e0b") -> str:
    return f"""
    <div style="display:flex;align-items:center;gap:12px;margin:28px 0 16px">
      <span style="width:3px;height:18px;background:{color};border-radius:2px;flex-shrink:0"></span>
      <span style="font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:1.05rem;
                   color:#ededf0;letter-spacing:.06em;text-transform:uppercase">{text}</span>
    </div>"""


def _agent_card_header(title: str, subtitle: str, color: str) -> str:
    return f"""
    <div style="display:flex;align-items:center;gap:14px;padding:20px 24px 16px;
                border-bottom:1px solid rgba(255,255,255,0.04)">
      <span style="width:4px;align-self:stretch;background:{color};border-radius:4px;flex-shrink:0"></span>
      <div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:1.1rem;
                    letter-spacing:.05em;text-transform:uppercase;color:#ededf0">{title}</div>
        <div style="font-family:'Inter',sans-serif;font-size:.73rem;color:#71717a;
                    margin-top:2px;letter-spacing:.02em">{subtitle}</div>
      </div>
    </div>"""


def _badge(text: str, bg: str, fg: str = "#fff") -> str:
    return f'<span style="background:{bg};color:{fg};padding:3px 9px;border-radius:5px;font-size:.7rem;font-family:\'Inter\',sans-serif;font-weight:700;letter-spacing:.04em">{text}</span>'


def _divider() -> str:
    return '<hr style="border:none;border-top:1px solid rgba(255,255,255,0.055);margin:24px 0">'


# ══════════════════════════════════════════════════════════════════════════════
# Session-state initialisation
# ══════════════════════════════════════════════════════════════════════════════

_SS: dict = {
    "ledger": None, "ctx": None, "agent_outputs": {}, "dossier_content": None,
    "dossier_path": None, "rejection_count": 0, "rejection_feedback": "",
    "pipeline_log": "", "approval_status": None,
    "H": None, "calib_points": [], "current_frame": None,
    "video_path": "", "frame_index": 0, "last_click": None,
    "last_pitch_coord": None, "video_events_logged": 0,
}
for _k, _v in _SS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# Backend helpers
# ══════════════════════════════════════════════════════════════════════════════

def _capture(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result, buf.getvalue()


def _stage_upload(uploaded_file, filename: str) -> Path:
    dest = STAGED_DIR / filename
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


def _load_parse_report():
    spec = importlib.util.spec_from_file_location(
        "parse_report", ROOT / "data" / "parse_report.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extract_frame(video_path: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # Wordmark
    st.markdown("""
    <div style="padding:8px 0 20px">
      <div style="font-family:'Barlow Condensed',sans-serif;font-weight:900;
                  font-size:2.1rem;color:#f59e0b;letter-spacing:-.01em;line-height:1">
        PITCHPULSE
      </div>
      <div style="font-family:'Inter',sans-serif;font-size:.7rem;color:#71717a;
                  letter-spacing:.12em;text-transform:uppercase;margin-top:3px">
        Tiverton Town FC
      </div>
      <div style="width:100%;height:1px;background:linear-gradient(90deg,#f59e0b,transparent);
                  margin-top:14px"></div>
    </div>
    """, unsafe_allow_html=True)

    # System status
    st.markdown(_section_label("System Status", "#71717a"), unsafe_allow_html=True)
    ledger_ok = LEDGER_PATH.exists()
    ctx_ok    = CONTEXT_PATH.exists()
    calib_ok  = CALIB_PATH.exists()
    dof_ok    = DOF_CARD_PATH.exists()
    html_ok   = HTML_REPORT_PATH.exists()

    st.markdown(
        _status_row("Match Ledger",    ledger_ok) +
        _status_row("Match Context",   ctx_ok) +
        _status_row("Calibration (H)", calib_ok) +
        _status_row("DoF Card",        dof_ok) +
        _status_row("HTML Report",     html_ok),
        unsafe_allow_html=True,
    )

    # Active ledger stats
    if st.session_state["ledger"]:
        summary = st.session_state["ledger"].get("summary", {})
        st.markdown(_section_label("Active Ledger", "#71717a"), unsafe_allow_html=True)
        cols = st.columns(2)
        cols[0].markdown(
            _tile("Tagged", str(summary.get("total_events", "—"))),
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            _tile("Matched", str(summary.get("matched", "—"))),
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        cols2 = st.columns(2)
        cols2[0].markdown(
            _tile("Unmatched", str(summary.get("unmatched_tags", "—")), color="#71717a"),
            unsafe_allow_html=True,
        )
        cols2[1].markdown(
            _tile("Subs", str(summary.get("substitutions", "—")), color="#71717a"),
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div style="position:fixed;bottom:16px;left:0;width:260px;padding:0 16px;
                box-sizing:border-box">
      <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#3f3f46;
                  text-align:center">LOCAL · OFFLINE · NO CLOUD</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="margin-bottom:28px">
  <div style="display:flex;align-items:baseline;gap:16px;margin-bottom:10px">
    <span style="font-family:'Barlow Condensed',sans-serif;font-weight:900;font-size:2.8rem;
                 color:#f59e0b;letter-spacing:-.02em;line-height:1">COMMAND CENTRE</span>
    <span style="font-family:'Inter',sans-serif;font-size:.75rem;color:#71717a;
                 letter-spacing:.1em;text-transform:uppercase">UEFA 4 Moments Framework</span>
  </div>
  <div style="width:100%;height:1px;background:linear-gradient(90deg,#f59e0b 120px,rgba(245,158,11,.15) 400px,transparent 700px)"></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📥  Match Ingestion",
    "🎥  Veo Video Lab",
    "🧠  Agent Cockpit",
    "📦  Deliverables",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Match Ingestion & Context
# ══════════════════════════════════════════════════════════════════════════════

with tab1:

    # ── How it works ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#111116;border:1px solid rgba(255,255,255,0.055);
                border-radius:12px;padding:24px 28px;margin-bottom:28px">
      <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:.85rem;
                  letter-spacing:.14em;text-transform:uppercase;color:#71717a;margin-bottom:16px">
        HOW IT WORKS
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0">

        <div style="display:flex;flex-direction:column;align-items:center;text-align:center;padding:0 12px">
          <div style="width:44px;height:44px;border-radius:50%;background:rgba(245,158,11,.12);
                      border:1px solid rgba(245,158,11,.3);display:flex;align-items:center;
                      justify-content:center;font-size:1.2rem;margin-bottom:12px">📱</div>
          <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:.95rem;
                      letter-spacing:.04em;color:#ededf0;margin-bottom:6px">TAG LIVE</div>
          <div style="font-family:'Inter',sans-serif;font-size:.75rem;color:#71717a;line-height:1.5">
            Tap events on your phone during the match — every shot, box entry, press, and sub
          </div>
        </div>

        <div style="display:flex;flex-direction:column;align-items:center;text-align:center;
                    padding:0 12px;border-left:1px solid rgba(255,255,255,0.055)">
          <div style="width:44px;height:44px;border-radius:50%;background:rgba(245,158,11,.12);
                      border:1px solid rgba(245,158,11,.3);display:flex;align-items:center;
                      justify-content:center;font-size:1.2rem;margin-bottom:12px">📤</div>
          <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:.95rem;
                      letter-spacing:.04em;color:#ededf0;margin-bottom:6px">UPLOAD HERE</div>
          <div style="font-family:'Inter',sans-serif;font-size:.75rem;color:#71717a;line-height:1.5">
            Drop the tagger JSON + the club Word report + your Veo video on this page
          </div>
        </div>

        <div style="display:flex;flex-direction:column;align-items:center;text-align:center;
                    padding:0 12px;border-left:1px solid rgba(255,255,255,0.055)">
          <div style="width:44px;height:44px;border-radius:50%;background:rgba(245,158,11,.12);
                      border:1px solid rgba(245,158,11,.3);display:flex;align-items:center;
                      justify-content:center;font-size:1.2rem;margin-bottom:12px">▶</div>
          <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:.95rem;
                      letter-spacing:.04em;color:#ededf0;margin-bottom:6px">RUN PIPELINE</div>
          <div style="font-family:'Inter',sans-serif;font-size:.75rem;color:#71717a;line-height:1.5">
            One click — reconcile events, generate visuals, run 4 tactical agents automatically
          </div>
        </div>

        <div style="display:flex;flex-direction:column;align-items:center;text-align:center;
                    padding:0 12px;border-left:1px solid rgba(255,255,255,0.055)">
          <div style="width:44px;height:44px;border-radius:50%;background:rgba(34,197,94,.12);
                      border:1px solid rgba(34,197,94,.3);display:flex;align-items:center;
                      justify-content:center;font-size:1.2rem;margin-bottom:12px">📊</div>
          <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:.95rem;
                      letter-spacing:.04em;color:#ededf0;margin-bottom:6px">DOSSIER OUT</div>
          <div style="font-family:'Inter',sans-serif;font-size:.75rem;color:#71717a;line-height:1.5">
            UEFA-standard tactical report + DoF WhatsApp card, approved and downloaded in seconds
          </div>
        </div>

      </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown(_section_label("Tagger Exports"), unsafe_allow_html=True)
        json_files = st.file_uploader(
            "JSON exports — one per period (1H, 2H)",
            type="json", accept_multiple_files=True, key="json_uploader",
        )
        if json_files:
            for jf in json_files:
                dest = _stage_upload(jf, jf.name)
                (RAW_DIR / jf.name).write_bytes(dest.read_bytes())

            total_ev = 0
            for jf in json_files:
                try:
                    data = json.loads(jf.getvalue())
                    if isinstance(data, list):
                        total_ev += len(data)
                except Exception:
                    pass

            st.success(f"✓  {len(json_files)} file(s) staged — {total_ev} events total")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown(_section_label("Match Report"), unsafe_allow_html=True)
        docx_file = st.file_uploader(
            "Club Word report (.docx) — extracts scorers, XI, competition",
            type=["docx"], key="docx_uploader",
        )
        if docx_file:
            staged_docx = _stage_upload(docx_file, docx_file.name)
            st.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.75rem;'
                f'color:#71717a;margin-bottom:8px">staged → {staged_docx.relative_to(ROOT)}</div>',
                unsafe_allow_html=True,
            )
            if st.button("⚙  Parse Report", type="primary", key="btn_parse"):
                with st.spinner("Parsing report…"):
                    try:
                        mod = _load_parse_report()
                        (ctx, _stdout) = _capture(mod.parse_report, staged_docx)
                        x_feed = mod.build_x_feed(ctx)
                        (RAW_DIR / "match_context.json").write_text(
                            json.dumps(ctx, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                        (RAW_DIR / "tivvy_x_feed.json").write_text(
                            json.dumps(x_feed, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                        st.session_state["ctx"] = ctx
                        st.success("✓  match_context.json + tivvy_x_feed.json written")
                    except Exception as exc:
                        st.error(f"Parse failed: {exc}")

    with col_right:
        st.markdown(_section_label("Parsed Context"), unsafe_allow_html=True)

        if st.session_state["ctx"] is None and CONTEXT_PATH.exists():
            st.session_state["ctx"] = json.loads(CONTEXT_PATH.read_text(encoding="utf-8"))

        ctx = st.session_state["ctx"]
        if ctx is None:
            st.markdown("""
            <div style="background:#111116;border:1px solid rgba(255,255,255,0.055);
                        border-radius:12px;padding:40px;text-align:center">
              <div style="font-size:2rem;margin-bottom:12px">📋</div>
              <div style="font-family:'Inter',sans-serif;font-size:.82rem;color:#71717a">
                No match context loaded yet.<br>Upload and parse a .docx report.
              </div>
            </div>
            """, unsafe_allow_html=True)
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

            result_color = {"W": "#22c55e", "D": "#f59e0b", "L": "#ef4444"}.get(result, "#f59e0b")

            st.markdown(f"""
            <div style="background:#111116;border:1px solid rgba(255,255,255,0.055);
                        border-radius:12px;padding:24px 28px;margin-bottom:16px">
              <div style="font-family:'Barlow Condensed',sans-serif;font-weight:900;font-size:2rem;
                          color:#ededf0;letter-spacing:-.01em;margin-bottom:4px">
                Tiverton Town
                <span style="color:#f59e0b">{tiv_g}–{opp_g}</span>
                {opp}
              </div>
              <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:12px">
                <span style="background:{result_color};color:#08080a;font-family:'Barlow Condensed',sans-serif;
                             font-weight:700;font-size:.8rem;letter-spacing:.08em;padding:3px 10px;border-radius:5px">
                  {result}
                </span>
                <span style="background:#1a1a22;color:#ededf0;font-family:'Inter',sans-serif;
                             font-size:.75rem;padding:3px 10px;border-radius:5px">{comp}</span>
                <span style="background:#1a1a22;color:#ededf0;font-family:'Inter',sans-serif;
                             font-size:.75rem;padding:3px 10px;border-radius:5px">{venue} ({home})</span>
                <span style="background:#1a1a22;color:#71717a;font-family:'JetBrains Mono',monospace;
                             font-size:.72rem;padding:3px 10px;border-radius:5px">{date}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # XI + scorers in two mini-columns
            c1, c2 = st.columns(2)
            with c1:
                lineup = ctx.get("lineup_tiverton", [])
                if lineup:
                    rows = ""
                    for p in lineup:
                        flag = " <span style='color:#71717a'>(sub)</span>" if p.get("is_sub") else ""
                        rows += f"""
                        <div style="display:flex;gap:10px;align-items:baseline;padding:4px 0;
                                    border-bottom:1px solid rgba(255,255,255,0.04)">
                          <span style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
                                       color:#f59e0b;width:20px;text-align:right">
                            {p.get('number', '?')}
                          </span>
                          <span style="font-family:'Inter',sans-serif;font-size:.8rem;
                                       color:#ededf0">{p.get('name', '?')}{flag}</span>
                        </div>"""
                    st.markdown(f"""
                    <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                                font-size:.8rem;letter-spacing:.1em;text-transform:uppercase;
                                color:#71717a;margin-bottom:8px">Starting XI</div>
                    <div style="background:#111116;border:1px solid rgba(255,255,255,0.055);
                                border-radius:10px;padding:12px 14px">{rows}</div>
                    """, unsafe_allow_html=True)

            with c2:
                scorers = ctx.get("scorers", {})
                tiv_sc = scorers.get("tiverton", [])
                opp_sc = scorers.get("opponent", [])
                if tiv_sc or opp_sc:
                    rows = ""
                    for s in tiv_sc:
                        rows += f"""
                        <div style="display:flex;gap:8px;align-items:baseline;padding:4px 0;
                                    border-bottom:1px solid rgba(255,255,255,0.04)">
                          <span style="color:#f59e0b">⚽</span>
                          <span style="font-family:'Inter',sans-serif;font-size:.8rem;
                                       color:#ededf0">{s['player']}</span>
                          <span style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
                                       color:#71717a;margin-left:auto">{s.get('minute_raw', s['minute'])}′</span>
                        </div>"""
                    for s in opp_sc:
                        rows += f"""
                        <div style="display:flex;gap:8px;align-items:baseline;padding:4px 0;
                                    border-bottom:1px solid rgba(255,255,255,0.04)">
                          <span style="color:#71717a">⚽</span>
                          <span style="font-family:'Inter',sans-serif;font-size:.8rem;
                                       color:#71717a">{s['player']} ({opp})</span>
                          <span style="font-family:'JetBrains Mono',monospace;font-size:.72rem;
                                       color:#3f3f46;margin-left:auto">{s.get('minute_raw', s['minute'])}′</span>
                        </div>"""
                    st.markdown(f"""
                    <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                                font-size:.8rem;letter-spacing:.1em;text-transform:uppercase;
                                color:#71717a;margin-bottom:8px">Scorers</div>
                    <div style="background:#111116;border:1px solid rgba(255,255,255,0.055);
                                border-radius:10px;padding:12px 14px">{rows}</div>
                    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Veo Video Lab
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(_section_label("Video Frame Extraction"), unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:\'Inter\',sans-serif;font-size:.8rem;color:#71717a;'
        'margin-bottom:16px">Point to your local Veo .mp4. '
        'Click the frame in the viewer to record pixel coordinates for calibration or event picking.</div>',
        unsafe_allow_html=True,
    )

    col_path, col_frame_ctrl = st.columns([3, 1])
    with col_path:
        video_path_input = st.text_input(
            "Local video path",
            value=st.session_state["video_path"],
            placeholder="C:/Users/.../match.mp4",
            key="video_path_input",
        )
        st.session_state["video_path"] = video_path_input
    with col_frame_ctrl:
        frame_index = st.number_input(
            "Frame #", min_value=0, value=st.session_state["frame_index"],
            step=50, key="frame_idx_input",
        )
        extract_btn = st.button("Extract Frame", type="primary", key="btn_extract")

    if extract_btn and video_path_input:
        with st.spinner("Reading frame…"):
            frame = _extract_frame(video_path_input, int(frame_index))
        if frame is None:
            st.error("Could not read frame — check the path and frame number.")
        else:
            st.session_state["current_frame"] = frame
            st.session_state["frame_index"]   = int(frame_index)
            st.session_state["last_click"]    = None

    frame = st.session_state.get("current_frame")
    if frame is not None:
        try:
            import plotly.express as px
            fig = px.imshow(frame, title=f"Frame {st.session_state['frame_index']} — click to pick pixel")
            fig.update_layout(
                paper_bgcolor="#08080a", plot_bgcolor="#08080a", font_color="#ededf0",
                title_font=dict(family="Barlow Condensed", size=14, color="#71717a"),
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=32, b=0),
            )
            fig.update_xaxes(title="U (px)", color="#71717a", gridcolor="#1a1a22", tickfont=dict(family="JetBrains Mono", size=10))
            fig.update_yaxes(title="V (px)", color="#71717a", gridcolor="#1a1a22", tickfont=dict(family="JetBrains Mono", size=10))
            event_data = st.plotly_chart(fig, on_select="rerun", use_container_width=True, key="frame_chart")
            if event_data and hasattr(event_data, "selection") and event_data.selection.points:
                pt = event_data.selection.points[0]
                st.session_state["last_click"] = (int(pt["x"]), int(pt["y"]))
            if st.session_state["last_click"]:
                u, v = st.session_state["last_click"]
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.8rem;'
                    f'color:#f59e0b;margin:8px 0">📍 U={u} V={v} &nbsp;·&nbsp; '
                    f'Frame {st.session_state["frame_index"]}</div>',
                    unsafe_allow_html=True,
                )
        except ImportError:
            st.image(frame, use_container_width=True)

    st.markdown(_divider(), unsafe_allow_html=True)

    # ── Calibration ───────────────────────────────────────────────────────────
    st.markdown(_section_label("Homography Calibration"), unsafe_allow_html=True)
    try:
        from cv.calibration import ANCHORS, ReferencePoint, calibrate_pitch, save_calibration, load_calibration
        calib_ok_import = True
    except ImportError as e:
        st.warning(f"cv.calibration unavailable: {e}")
        calib_ok_import = False

    if calib_ok_import:
        col_ca, col_cb, col_cc = st.columns([2, 1, 1])
        with col_ca:
            anchor_labels = [a["label"] for a in ANCHORS]
            selected_anchor_label = st.selectbox("Anchor point (match to clicked location)", anchor_labels, key="anchor_select")
        with col_cb:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("➕  Add Point", key="btn_add_calib"):
                click = st.session_state.get("last_click")
                if click is None:
                    st.warning("Click on the frame first.")
                else:
                    anchor = next((a for a in ANCHORS if a["label"] == selected_anchor_label), None)
                    if anchor:
                        u, v = click
                        existing = [p["label"] for p in st.session_state["calib_points"]]
                        if selected_anchor_label in existing:
                            for p in st.session_state["calib_points"]:
                                if p["label"] == selected_anchor_label:
                                    p["pixel"] = [u, v]
                        else:
                            st.session_state["calib_points"].append(
                                {"pixel": [u, v], "world": anchor["world"], "label": anchor["label"]}
                            )
        with col_cc:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🗑  Clear", key="btn_clear_calib"):
                st.session_state["calib_points"] = []
                st.session_state["H"] = None
                st.rerun()

        n_pts = len(st.session_state["calib_points"])
        if n_pts > 0:
            rows = ""
            for i, pt in enumerate(st.session_state["calib_points"]):
                u, v = pt["pixel"]; xw, yw = pt["world"]
                rows += f"""
                <div style="display:flex;gap:16px;padding:5px 0;
                            border-bottom:1px solid rgba(255,255,255,0.04);
                            font-family:'JetBrains Mono',monospace;font-size:.75rem">
                  <span style="color:#71717a;width:20px">{i+1}</span>
                  <span style="color:#ededf0;flex:1">{pt['label']}</span>
                  <span style="color:#f59e0b">U={u} V={v}</span>
                  <span style="color:#71717a">→ ({xw}m, {yw}m)</span>
                </div>"""
            st.markdown(f"""
            <div style="background:#111116;border:1px solid rgba(255,255,255,0.055);
                        border-radius:10px;padding:14px 18px;margin:12px 0">
              <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:.8rem;
                          letter-spacing:.1em;color:#71717a;margin-bottom:8px">
                CALIBRATION POINTS &nbsp;
                <span style="color:{'#f59e0b' if n_pts >= 4 else '#ef4444'}">{n_pts}/4+</span>
              </div>
              {rows}
            </div>
            """, unsafe_allow_html=True)

        col_comp, col_load = st.columns(2)
        with col_comp:
            if st.button("⚙  Compute & Save H", type="primary", disabled=(n_pts < 4), key="btn_compute_h"):
                with st.spinner("RANSAC homography…"):
                    try:
                        ref_pts = [ReferencePoint(tuple(p["pixel"]), tuple(p["world"]), p["label"])
                                   for p in st.session_state["calib_points"]]
                        H = calibrate_pitch(ref_pts)
                        save_calibration(H, ref_pts)
                        st.session_state["H"] = H
                        st.success(f"✓  H computed from {n_pts} points — saved to data/raw/")
                    except Exception as exc:
                        st.error(f"Failed: {exc}")
        with col_load:
            if st.button("📂  Load Saved Calibration", key="btn_load_calib"):
                if CALIB_PATH.exists():
                    try:
                        H = load_calibration(CALIB_PATH)
                        st.session_state["H"] = H
                        st.success("✓  Calibration loaded.")
                    except Exception as exc:
                        st.error(f"Load failed: {exc}")
                else:
                    st.warning("No calibration file found.")

        if st.session_state["H"] is not None:
            st.markdown(
                f'<div style="display:inline-flex;align-items:center;gap:8px;'
                f'background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);'
                f'border-radius:8px;padding:6px 14px;margin-top:8px">'
                f'<span style="color:#22c55e;font-size:.8rem">●</span>'
                f'<span style="font-family:\'Barlow Condensed\',sans-serif;font-weight:700;'
                f'font-size:.85rem;letter-spacing:.05em;color:#86efac">CALIBRATION ACTIVE</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(_divider(), unsafe_allow_html=True)

    # ── Event Picker ──────────────────────────────────────────────────────────
    st.markdown(_section_label("Event Picker"), unsafe_allow_html=True)
    try:
        from cv.picker import pixel_to_pitch, build_video_event, append_video_event
        from cv.zones import get_zone_by_coords
        picker_ok = True
    except ImportError as e:
        st.warning(f"cv.picker unavailable: {e}")
        picker_ok = False

    if picker_ok:
        EVENT_TYPES = ["SHOT","BOX_ENTRY","HIGH_REGAIN","DEF_TURNOVER","SET_PIECE","AERIAL_DUEL","SECOND_BALL"]
        SUB_TYPES = {
            "SHOT": ["ON_TARGET","OFF_TARGET","BLOCKED"],
            "BOX_ENTRY": ["CENTRAL","LEFT_FLANK","RIGHT_FLANK"],
            "HIGH_REGAIN": ["WON"], "DEF_TURNOVER": ["LOST"],
            "SET_PIECE": ["ATT_CORNER","DEF_CORNER","FREE_KICK"],
            "AERIAL_DUEL": ["WON","LOST"], "SECOND_BALL": ["WON","LOST"],
        }
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        with col_e1:
            ev_type  = st.selectbox("Event", EVENT_TYPES, key="ev_type")
        with col_e2:
            sub_type = st.selectbox("Sub-type", SUB_TYPES.get(ev_type, [""]), key="ev_sub_type")
        with col_e3:
            period       = st.selectbox("Period", ["1H","2H","ET1","ET2"], key="ev_period")
            match_seconds = st.number_input("Match seconds", 0, 7200, 0, key="ev_seconds")
        with col_e4:
            player_num = st.number_input("Player #", 1, 18, 9, key="ev_player")

        H     = st.session_state.get("H")
        click = st.session_state.get("last_click")
        if H is not None and click:
            u, v = click
            x_m, y_m = pixel_to_pitch(u, v, H)
            zone_id   = get_zone_by_coords(x_m, y_m)
            st.session_state["last_pitch_coord"] = (x_m, y_m)
            st.markdown(
                f'<div style="background:#111116;border:1px solid rgba(245,158,11,.25);'
                f'border-radius:10px;padding:14px 18px;display:inline-flex;gap:24px;'
                f'font-family:\'JetBrains Mono\',monospace;font-size:.8rem;margin:10px 0">'
                f'<span style="color:#71717a">Pitch coord</span>'
                f'<span style="color:#f59e0b">{x_m:.1f}m, {y_m:.1f}m</span>'
                f'<span style="color:#71717a">Zone</span>'
                f'<span style="color:#f59e0b">{zone_id}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if st.button("📝  Log to Ledger", type="primary", disabled=(H is None or click is None), key="btn_log_event"):
            try:
                u, v = st.session_state["last_click"]
                event = build_video_event(u=u, v=v, H=H, event_type=ev_type,
                                          sub_type=sub_type, player_num=int(player_num),
                                          period=period, match_seconds=int(match_seconds))
                append_video_event(LEDGER_PATH, event)
                st.session_state["video_events_logged"] += 1
                x_m, y_m = pixel_to_pitch(u, v, H)
                st.success(f"✓  {ev_type}/{sub_type} logged at ({x_m:.1f}m, {y_m:.1f}m)")
            except Exception as exc:
                st.error(f"Failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Agent Cockpit & Approval Gate
# ══════════════════════════════════════════════════════════════════════════════

with tab3:

    # ── Big pipeline CTA ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#111116 0%,#1a1610 100%);
                border:1px solid rgba(245,158,11,.2);border-radius:14px;
                padding:32px 36px;margin-bottom:24px;
                animation: amberglow 4s ease-in-out infinite">
      <div style="font-family:'Barlow Condensed',sans-serif;font-weight:900;font-size:1.5rem;
                  color:#f59e0b;letter-spacing:.04em;text-transform:uppercase;margin-bottom:6px">
        Run Full Pipeline
      </div>
      <div style="font-family:'Inter',sans-serif;font-size:.8rem;color:#71717a;max-width:520px">
        Reconciles tagger events with the club feed, generates shot map &amp; heatmaps,
        runs all four tactical agents (In Possession · Press · Set Pieces · Non-League Physics),
        and builds the draft dossier for your approval.
      </div>
    </div>
    """, unsafe_allow_html=True)

    run_col, status_col = st.columns([1, 2], gap="large")
    with run_col:
        run_pipeline = st.button("▶  RUN PIPELINE", type="primary", key="btn_run_pipeline", use_container_width=True)
    with status_col:
        ledger_loaded = st.session_state["ledger"] is not None
        agents_done   = bool(st.session_state["agent_outputs"])
        approved      = st.session_state["approval_status"] == "approved"

        def _step_dot(label, done, active=False):
            if done:
                bg, txt = "rgba(34,197,94,.15)", "#22c55e"
            elif active:
                bg, txt = "rgba(245,158,11,.15)", "#f59e0b"
            else:
                bg, txt = "rgba(255,255,255,0.04)", "#3f3f46"
            return (f'<div style="display:flex;align-items:center;gap:8px">'
                    f'<span style="width:8px;height:8px;border-radius:50%;background:{txt}"></span>'
                    f'<span style="font-family:\'Inter\',sans-serif;font-size:.78rem;color:{txt}">{label}</span>'
                    f'</div>')

        st.markdown(
            f'<div style="display:flex;gap:20px;align-items:center;height:100%">'
            + _step_dot("Reconcile",  ledger_loaded, not ledger_loaded)
            + _step_dot("Visuals",    ledger_loaded, ledger_loaded and not agents_done)
            + _step_dot("Agents",     agents_done,   ledger_loaded and not agents_done)
            + _step_dot("Approved",   approved,      agents_done and not approved)
            + f'</div>',
            unsafe_allow_html=True,
        )

    if run_pipeline:
        log_lines: list[str] = []

        with st.spinner("Step 1 — Reconciling events…"):
            try:
                from reconcile.sync import (
                    ensure_dirs, load_tag_events, load_club_feed,
                    extract_club_events, reconcile_events, build_ledger,
                    write_ledger, RECON_WINDOW_S,
                )
                ensure_dirs()
                (tag_events, _)       = _capture(load_tag_events, RAW_DIR)
                (feed_tuple, _)       = _capture(load_club_feed, RAW_DIR)
                feed_data, feed_type  = feed_tuple
                (club_events, _)      = _capture(extract_club_events, feed_data, feed_type)
                (recon_out, _)        = _capture(reconcile_events, tag_events, club_events, window=RECON_WINDOW_S)
                matched, unmatched_tags, unmatched_club = recon_out
                (ledger, _)           = _capture(build_ledger, matched, unmatched_tags, unmatched_club)
                write_ledger(ledger, LEDGER_PATH)
                st.session_state["ledger"] = ledger
                log_lines.append(f"✓ Reconcile — {ledger.get('summary',{}).get('total_events','?')} events")
            except Exception as exc:
                st.error(f"Reconcile failed: {exc}")
                st.stop()

        with st.spinner("Step 2 — Generating visuals…"):
            try:
                from reports.visualizer import plot_shot_map, plot_transition_map, plot_zonal_heatmap
                PLOTS_DIR.mkdir(parents=True, exist_ok=True)
                for fn in (plot_shot_map, plot_transition_map, plot_zonal_heatmap):
                    _capture(fn, st.session_state["ledger"], PLOTS_DIR)
                log_lines.append("✓ Visuals — shot map, transition map, zonal heatmap")
            except Exception as exc:
                log_lines.append(f"⚠ Visuals skipped: {exc}")

        with st.spinner("Step 3 — Running agents…"):
            try:
                from agents.synthesis import (
                    run_in_possession_agent, run_press_agent,
                    run_set_piece_agent, run_nonleague_agent, _load_match_context,
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
                log_lines.append("✓ Agents — 4 sections drafted")
            except Exception as exc:
                st.error(f"Agents failed: {exc}")
                st.stop()

        st.session_state["pipeline_log"] = "\n".join(log_lines)
        st.rerun()

    if st.session_state["pipeline_log"]:
        with st.expander("Pipeline log", expanded=False):
            st.code(st.session_state["pipeline_log"], language="text")

    st.markdown(_divider(), unsafe_allow_html=True)

    # ── Agent output cards ────────────────────────────────────────────────────
    outputs = st.session_state.get("agent_outputs", {})
    if outputs:
        st.markdown(_section_label("Agent Outputs"), unsafe_allow_html=True)

        AGENT_META = {
            "In Possession":      ("#f59e0b", "Moment 1 — In Possession",       "SHOT · BOX_ENTRY · half-space exploitation · Zone 14"),
            "Press / LoE":        ("#3b82f6", "Moments 2 & 3 — Press / LoE",    "HIGH_REGAIN · DEF_TURNOVER · Counter-Pressing Phase · Compactness"),
            "Set Pieces":         ("#a855f7", "Moment 1 & 3 — Dead Ball",        "SET_PIECE · Rest Defense · ATT_CORNER · DEF_CORNER · FREE_KICK"),
            "Non-League Physics": ("#22c55e", "All 4 Moments — Physical Battle", "AERIAL_DUEL · SECOND_BALL · Quantitative Superiority · Direct Play"),
        }
        for agent_name, markdown_text in outputs.items():
            color, moment, tags = AGENT_META.get(agent_name, ("#71717a", "", ""))
            st.markdown(f"""
            <div style="background:#111116;border:1px solid rgba(255,255,255,0.055);
                        border-radius:12px;margin-bottom:14px;overflow:hidden">
              {_agent_card_header(agent_name, moment, color)}
              <div style="padding:6px 24px 4px">
                <div style="font-family:'JetBrains Mono',monospace;font-size:.68rem;
                            color:#3f3f46;letter-spacing:.06em">{tags}</div>
              </div>
            """, unsafe_allow_html=True)
            with st.expander("View analysis", expanded=False):
                st.markdown(markdown_text)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(_divider(), unsafe_allow_html=True)

    # ── Approval Gate ─────────────────────────────────────────────────────────
    st.markdown(_section_label("Approval Gate"), unsafe_allow_html=True)

    if not outputs:
        st.markdown("""
        <div style="background:#111116;border:1px solid rgba(255,255,255,0.055);
                    border-radius:12px;padding:40px;text-align:center">
          <div style="font-family:'Inter',sans-serif;font-size:.82rem;color:#71717a">
            Run the pipeline first to unlock the approval gate.
          </div>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state["approval_status"] == "approved":
        dossier_path = st.session_state.get("dossier_path")
        path_str = str(Path(dossier_path).relative_to(ROOT)) if dossier_path else ""
        st.markdown(f"""
        <div style="background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.25);
                    border-radius:12px;padding:28px 32px;display:flex;align-items:center;gap:20px">
          <span style="font-size:2rem">✅</span>
          <div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-weight:900;font-size:1.3rem;
                        letter-spacing:.04em;color:#22c55e">DOSSIER APPROVED</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:.75rem;color:#71717a;
                        margin-top:4px">{path_str}</div>
          </div>
          <div style="margin-left:auto;font-family:'Inter',sans-serif;font-size:.8rem;color:#71717a">
            Switch to the Deliverables tab to preview and download.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_app, col_rej = st.columns([1, 1], gap="large")

        with col_app:
            st.markdown("""
            <div style="font-family:'Inter',sans-serif;font-size:.8rem;color:#71717a;
                        margin-bottom:12px">
              Approve the draft to write the dossier to disk, generate the HTML report,
              and produce the DoF WhatsApp card.
            </div>
            """, unsafe_allow_html=True)
            if st.button("✅  Approve Dossier", type="primary", key="btn_approve", use_container_width=True):
                with st.spinner("Writing dossier…"):
                    try:
                        from agents.synthesis import _build_dossier, _write_dossier, _match_id_from_ledger
                        ledger  = st.session_state["ledger"]
                        ctx     = st.session_state.get("ctx") or {}
                        content = _build_dossier(ledger, feedback="", ctx=ctx)
                        match_id = _match_id_from_ledger(ledger)
                        path    = _write_dossier(content, match_id)
                        try:
                            from reports.packager import build_html
                            build_html(content, ledger=ledger, plots_dir=PLOTS_DIR, out_dir=PROC_DIR)
                        except Exception:
                            pass
                        try:
                            from reports.dof_card import build_dof_card
                            build_dof_card(ledger=ledger)
                        except Exception:
                            pass
                        st.session_state["dossier_content"] = content
                        st.session_state["dossier_path"]    = path
                        st.session_state["approval_status"] = "approved"
                        st.session_state["rejection_feedback"] = ""
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Approval failed: {exc}")

        with col_rej:
            rejection_count = st.session_state.get("rejection_count", 0)
            st.markdown(f"""
            <div style="font-family:'Inter',sans-serif;font-size:.8rem;color:#71717a;
                        margin-bottom:12px">
              Reject with feedback and all four agents re-run with your notes injected.
              Cycle {rejection_count} / 3 maximum.
            </div>
            """, unsafe_allow_html=True)
            feedback_text = st.text_area(
                "Manager feedback", value=st.session_state.get("rejection_feedback", ""),
                height=80, key="feedback_input",
                placeholder="e.g. Focus on right-flank press triggers; disregard Zone 14 data.",
            )
            reject_disabled = rejection_count >= 3
            if st.button("🔁  Reject & Re-run", type="secondary", disabled=reject_disabled,
                          key="btn_reject", use_container_width=True):
                st.session_state["rejection_feedback"] = feedback_text
                st.session_state["rejection_count"]    = rejection_count + 1
                with st.spinner("Re-running agents with feedback…"):
                    try:
                        from agents.synthesis import (
                            run_in_possession_agent, run_press_agent,
                            run_set_piece_agent, run_nonleague_agent,
                        )
                        ledger = st.session_state["ledger"]
                        ctx    = st.session_state.get("ctx") or {}
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
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Re-run failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Deliverables Hub
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown(_section_label("Deliverables Hub"), unsafe_allow_html=True)

    col_dof, col_doss = st.columns([1, 2], gap="large")

    with col_dof:
        st.markdown("""
        <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:.85rem;
                    letter-spacing:.1em;text-transform:uppercase;color:#71717a;margin-bottom:12px">
          Director of Football Card
        </div>
        """, unsafe_allow_html=True)

        if DOF_CARD_PATH.exists():
            st.image(str(DOF_CARD_PATH), use_container_width=True)
            with open(DOF_CARD_PATH, "rb") as f:
                st.download_button(
                    "⬇  Download DoF Card (PNG)",
                    data=f, file_name=DOF_CARD_PATH.name,
                    mime="image/png", key="dl_dof",
                    use_container_width=True,
                )
        else:
            st.markdown("""
            <div style="background:#111116;border:1px solid rgba(255,255,255,0.055);
                        border-radius:12px;padding:40px;text-align:center">
              <div style="font-size:2rem;margin-bottom:10px">🖼</div>
              <div style="font-family:'Inter',sans-serif;font-size:.78rem;color:#71717a">
                Approve the dossier to generate the DoF card.
              </div>
            </div>
            """, unsafe_allow_html=True)

        for plot_name, label in [
            ("shot_map.png", "Shot Map"),
            ("zonal_heatmap.png", "Zonal Heatmap"),
        ]:
            p = PLOTS_DIR / plot_name
            if p.exists():
                st.markdown(f"""
                <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;
                            font-size:.8rem;letter-spacing:.1em;text-transform:uppercase;
                            color:#71717a;margin:16px 0 8px">{label}</div>
                """, unsafe_allow_html=True)
                st.image(str(p), use_container_width=True)

    with col_doss:
        st.markdown("""
        <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:.85rem;
                    letter-spacing:.1em;text-transform:uppercase;color:#71717a;margin-bottom:12px">
          Tactical Dossier
        </div>
        """, unsafe_allow_html=True)

        dossier_content = st.session_state.get("dossier_content")
        html_content = None
        if HTML_REPORT_PATH.exists():
            html_content = HTML_REPORT_PATH.read_text(encoding="utf-8")

        if html_content:
            st.components.v1.html(html_content, height=700, scrolling=True)
            st.download_button(
                "⬇  Download HTML Dossier",
                data=html_content.encode("utf-8"),
                file_name=HTML_REPORT_PATH.name,
                mime="text/html", key="dl_html",
                use_container_width=True,
            )
        elif dossier_content:
            st.markdown(dossier_content)
            st.download_button(
                "⬇  Download Markdown Dossier",
                data=dossier_content.encode("utf-8"),
                file_name="tivvy_tactical_dossier.md",
                mime="text/markdown", key="dl_md",
                use_container_width=True,
            )
        else:
            st.markdown("""
            <div style="background:#111116;border:1px solid rgba(255,255,255,0.055);
                        border-radius:12px;padding:60px;text-align:center">
              <div style="font-family:'Barlow Condensed',sans-serif;font-weight:900;font-size:1.4rem;
                          color:#3f3f46;letter-spacing:.04em;margin-bottom:10px">
                NO DOSSIER YET
              </div>
              <div style="font-family:'Inter',sans-serif;font-size:.8rem;color:#71717a;
                          max-width:320px;margin:0 auto;line-height:1.6">
                Run the pipeline in the Agent Cockpit tab, review the four agent outputs,
                and approve to generate all deliverables.
              </div>
            </div>
            """, unsafe_allow_html=True)
