"""
cv/vision_center_ui.py — Opposition Analysis (app.py tab 6), written for managers and coaching staff.

Rendering only. Match setup lives in a collapsible drawer; the main screen is the match-day briefing card,
the tactical pitch and the scouting library. Wording comes from cv/coach_brief.py; background work
(upload, analysis, kit check) comes from cv/vision_center.py and runs silently behind progress bars.
"""

from __future__ import annotations

import shutil
import time
from html import escape
from pathlib import Path

import streamlit as st

from cv import coach_brief as cb
from cv import vision_center as vc

_CSS = """
<style>
.st-key-vcc { background:radial-gradient(1200px 400px at 10% -10%, rgba(35,134,54,.10), transparent 60%), #0D1117;
  border:1px solid #30363D; border-radius:16px; padding:24px 26px 28px; }
.st-key-vcc [data-testid="stVerticalBlockBorderWrapper"] { background:#161B22; border-color:#30363D !important; border-radius:14px; }
.st-key-vcc [data-testid="stExpander"] details { background:#161B22; border:1px solid #30363D; border-radius:14px; }
.st-key-vcc [data-testid="stExpander"] summary { font-family:Bahnschrift,'Arial Narrow',Arial,sans-serif; letter-spacing:.08em;
  text-transform:uppercase; font-size:.9rem; color:#E6EDF3; }
.st-key-vcc .stButton > button { background:#21262D !important; border-color:#30363D !important; color:#E6EDF3 !important; }
.st-key-vcc .stButton > button:hover { background:rgba(35,134,54,.12) !important; border-color:#238636 !important; color:#3FB950 !important; }
.st-key-vcc .stButton > button[kind="primary"] { background:#238636 !important; border-color:#238636 !important; color:#FFFFFF !important;
  font-family:Bahnschrift,'Arial Narrow',Arial,sans-serif !important; letter-spacing:.08em !important; text-transform:uppercase; }
.st-key-vcc .stButton > button[kind="primary"]:hover { background:#2EA043 !important; border-color:#2EA043 !important;
  color:#FFFFFF !important; box-shadow:0 0 24px rgba(35,134,54,.35) !important; }
.st-key-vcc .stButton > button:disabled { opacity:.45 !important; }
.st-key-vcc [data-testid="stProgress"] > div > div > div > div { background:linear-gradient(90deg,#238636,#3FB950) !important; }
.vcc-head { display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap; margin-bottom:14px; }
.vcc-eyebrow { font:600 .7rem Bahnschrift,'Arial Narrow',Arial,sans-serif; letter-spacing:.22em; color:#3FB950; text-transform:uppercase; }
.vcc-head h2 { margin:2px 0 0; font:700 1.9rem Bahnschrift,'Arial Narrow',Arial,sans-serif; color:#F0F6FC; letter-spacing:.01em; }
.vcc-head p { margin:4px 0 0; font:.86rem 'Segoe UI',Inter,system-ui,sans-serif; color:#8B949E; }
.vcc-label { font:600 .7rem Bahnschrift,'Arial Narrow',Arial,sans-serif; letter-spacing:.18em; text-transform:uppercase; color:#8B949E; margin:0 0 8px; }
.vcc-note { font:.82rem/1.5 'Segoe UI',Inter,system-ui,sans-serif; color:#8B949E; margin:4px 0 8px; }
.vcc-pill { display:inline-flex; align-items:center; padding:4px 11px; border-radius:999px; margin:0 6px 6px 0; border:1px solid #30363D;
  font:600 .74rem 'Segoe UI',Inter,system-ui,sans-serif; color:#C9D1D9; background:#0D1117; }
.vcc-pill.green { color:#3FB950; border-color:rgba(35,134,54,.6); background:rgba(35,134,54,.12); }
.vcc-pill.amber { color:#E3B341; border-color:rgba(210,153,34,.6); background:rgba(210,153,34,.12); }
.vcc-pill.red { color:#FF7B72; border-color:rgba(218,54,51,.6); background:rgba(218,54,51,.12); }
.vcc-alert { border-left:3px solid #D29922; background:rgba(210,153,34,.08); padding:9px 13px; border-radius:6px; color:#E3B341;
  font:.84rem/1.5 'Segoe UI',Inter,system-ui,sans-serif; margin:6px 0 10px; }
.vcc-alert.red { border-color:#DA3633; background:rgba(218,54,51,.08); color:#FF7B72; }
.vcc-swatches { display:flex; gap:10px; flex-wrap:wrap; margin:8px 0; }
.vcc-swatch { display:flex; align-items:center; gap:10px; background:#0D1117; border:1px solid #30363D; border-radius:10px; padding:8px 12px 8px 8px; }
.vcc-swatch i { width:30px; height:30px; border-radius:8px; border:1px solid rgba(240,246,252,.2); display:block; }
.vcc-swatch span { font:.78rem 'Segoe UI',Inter,system-ui,sans-serif; color:#C9D1D9; }
.vcc-status { display:flex; justify-content:space-between; align-items:baseline; gap:12px; margin-bottom:6px; }
.vcc-status b { font:700 1.05rem Bahnschrift,'Arial Narrow',Arial,sans-serif; letter-spacing:.04em; color:#F0F6FC; text-transform:uppercase; }
.vcc-status span { font:.84rem 'Segoe UI',Inter,system-ui,sans-serif; color:#3FB950; }
.vcc-brief { border:1px solid #30363D; border-radius:16px; overflow:hidden; background:#0D1117; margin:10px 0 16px; }
.vcc-brief-head { display:flex; justify-content:space-between; align-items:flex-start; gap:18px; flex-wrap:wrap; padding:22px 24px;
  background:linear-gradient(115deg,#0B2A13 0%,#10361A 38%,#161B22 38.2%,#161B22 100%); border-bottom:1px solid #30363D; }
.vcc-brief-head .opp { font:700 2.3rem/1 Bahnschrift,'Arial Narrow',Arial,sans-serif; color:#FFFFFF; text-transform:uppercase; letter-spacing:.02em; margin-top:6px; }
.vcc-brief-head .shape { font:600 .9rem 'Segoe UI',Inter,system-ui,sans-serif; color:#C9D1D9; margin-top:8px; }
.vcc-brief-head .right { display:flex; flex-direction:column; align-items:flex-end; gap:8px; }
.vcc-struct { font:700 1rem Bahnschrift,'Arial Narrow',Arial,sans-serif; letter-spacing:.14em; color:#0D1117; background:#F2CC60;
  padding:8px 14px; border-radius:6px; box-shadow:0 0 0 1px rgba(242,204,96,.4), 0 6px 20px rgba(242,204,96,.15); }
.vcc-metrics { display:grid; grid-template-columns:repeat(4,1fr); gap:1px; background:#30363D; }
.vcc-metrics .m { background:#161B22; padding:16px 20px; }
.vcc-metrics .k { font:600 .68rem Bahnschrift,'Arial Narrow',Arial,sans-serif; letter-spacing:.16em; color:#8B949E; text-transform:uppercase; }
.vcc-metrics .v { font:700 1.35rem Bahnschrift,'Arial Narrow',Arial,sans-serif; color:#F0F6FC; margin:6px 0 3px; }
.vcc-metrics .d { font:.8rem 'Segoe UI',Inter,system-ui,sans-serif; color:#3FB950; }
.vcc-phases { display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:#30363D; }
.vcc-phases .p { background:#0D1117; padding:18px 20px 20px; }
.vcc-phases .top { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
.vcc-phases .num { width:32px; height:32px; border-radius:8px; display:grid; place-items:center; font:700 1.05rem Bahnschrift,Arial,sans-serif; color:#0D1117; }
.vcc-phases .p1 .num { background:#58A6FF; } .vcc-phases .p2 .num { background:#3FB950; } .vcc-phases .p3 .num { background:#F2CC60; }
.vcc-phases .t { font:700 1rem Bahnschrift,'Arial Narrow',Arial,sans-serif; color:#F0F6FC; text-transform:uppercase; letter-spacing:.05em; }
.vcc-phases .st { font:.76rem 'Segoe UI',Inter,system-ui,sans-serif; color:#8B949E; }
.vcc-phases ul { margin:0; padding-left:18px; }
.vcc-phases li { font:.88rem/1.5 'Segoe UI',Inter,system-ui,sans-serif; color:#E6EDF3; margin:0 0 7px; }
.vcc-phases li::marker { color:#3FB950; }
.vcc-unavailable { padding:20px 24px; font:.92rem/1.55 'Segoe UI',Inter,system-ui,sans-serif; color:#E3B341; background:#161B22; }
.vcc-pitch { width:100%; height:auto; display:block; border-radius:10px; }
.vcc-legend { display:flex; gap:18px; flex-wrap:wrap; margin:12px 2px 2px; font:.8rem 'Segoe UI',Inter,system-ui,sans-serif; color:#C9D1D9; }
.vcc-legend i { display:inline-block; vertical-align:middle; margin-right:7px; }
.vcc-legend .def { width:22px; height:3px; background:#388BFD; }
.vcc-legend .press { width:22px; height:0; border-top:3px dashed #F85149; }
.vcc-legend .corr { width:16px; height:12px; background:rgba(56,139,253,.35); border:1px solid #58A6FF; }
.vcc-legend .flank { width:16px; height:12px; background:rgba(63,185,80,.3); border:1px dashed #E3B341; }
.vcc-lib { width:100%; border-collapse:collapse; font:.84rem 'Segoe UI',Inter,system-ui,sans-serif; }
.vcc-lib th { text-align:left; font:600 .66rem Bahnschrift,'Arial Narrow',Arial,sans-serif; letter-spacing:.16em; text-transform:uppercase;
  color:#8B949E; padding:8px 10px; border-bottom:1px solid #30363D; }
.vcc-lib td { padding:10px; border-bottom:1px solid #21262D; color:#E6EDF3; }
.vcc-lib td.opp { font:700 .95rem Bahnschrift,'Arial Narrow',Arial,sans-serif; text-transform:uppercase; letter-spacing:.04em; }
.vcc-empty { text-align:center; padding:48px 20px; border:1px dashed #30363D; border-radius:14px; background:#161B22; }
.vcc-empty b { display:block; font:700 1.3rem Bahnschrift,'Arial Narrow',Arial,sans-serif; color:#F0F6FC; letter-spacing:.04em; text-transform:uppercase; }
@media (max-width: 1000px) { .vcc-metrics, .vcc-phases { grid-template-columns:1fr 1fr; } .vcc-phases .p3 { grid-column:span 2; } }
@media (max-width: 640px) { .vcc-metrics, .vcc-phases { grid-template-columns:1fr; } .vcc-phases .p3 { grid-column:auto; }
  .vcc-brief-head .right { align-items:flex-start; } }
</style>
"""


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _pill(text: str, tone: str = "") -> str:
    return f'<span class="vcc-pill {tone}">{escape(str(text))}</span>'


def _label(text: str) -> str:
    return f'<div class="vcc-label">{escape(text)}</div>'


def _note(text: str) -> str:
    return f'<div class="vcc-note">{escape(text)}</div>'


def _alert(text: str, tone: str = "") -> str:
    return f'<div class="vcc-alert {tone}">{escape(text)}</div>'


def _setup_slug() -> str:
    return vc.slugify(st.session_state.get("vcc_opponent", ""))


# ── Entry point ──────────────────────────────────────────────────────────────

def render() -> None:
    _html(_CSS)
    with st.container(key="vcc"):
        scouted = vc.scouted_opponents()
        _html('<div class="vcc-head"><div><div class="vcc-eyebrow">Opposition analysis</div>'
              '<h2>Match Intelligence Report</h2>'
              '<p>How they defend, where the space is, and where we hold when we lose it.</p></div></div>')
        with st.expander("Match Setup", expanded=not scouted, icon=":material/tune:"):
            _match_setup()
        _analysis_status()
        _report(scouted)


# ── Match Setup drawer ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _staged_videos(signature: tuple) -> list[dict]:
    return vc.staged_videos()


def _video_signature() -> tuple:
    if not vc.STAGING_VIDEOS.exists():
        return ()
    return tuple((p.name, p.stat().st_size, p.stat().st_mtime) for p in sorted(vc.STAGING_VIDEOS.glob("*.mp4")))


def _match_setup() -> None:
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        _html(_label("Match video"))
        video = _video_picker()
    with right:
        _html(_label("Opponent"))
        c1, c2 = st.columns([2, 1])
        with c1:
            st.text_input("Opponent name", key="vcc_opponent", placeholder="e.g. Dorchester Town")
        with c2:
            kit = st.color_picker("Their shirt colour", value="#CC2222", key="vcc_kit")
        footage = st.radio("Video type", ["full_wide", "highlight"], key="vcc_footage", horizontal=True,
                           format_func=lambda f: {"full_wide": "Full match", "highlight": "Highlights only"}[f])
        if footage == "highlight":
            _html(_note("Highlights can't show a defensive shape. Use the full match for a complete report."))
    slug = _setup_slug()
    k1, k2 = st.columns([1.1, 1], gap="large")
    with k1:
        _kit_check(slug, video, kit.upper())
    with k2:
        _start_analysis(slug, video, kit.upper(), footage)


def _video_picker() -> str | None:
    videos = _staged_videos(_video_signature())
    by_name = {v["name"]: v for v in videos}
    pending = st.session_state.pop("vcc_video_pending", None)
    if pending in by_name:
        st.session_state["vcc_video"] = pending
    elif st.session_state.get("vcc_video") not in by_name:
        st.session_state.pop("vcc_video", None)

    chosen = None
    if videos:
        name = st.selectbox("Match video", list(by_name), key="vcc_video", label_visibility="collapsed",
                            format_func=lambda n: f"{n}  ·  {by_name[n]['duration']}  ·  {by_name[n]['size']}")
        chosen = by_name[name]
        quality = "green" if (chosen["height"] or 0) >= 1080 else "amber"
        _html(_pill(chosen["resolution"], quality) + _pill(chosen["duration"]) + _pill(chosen["size"]))
    else:
        _html(_note("No match videos yet. Add one below."))

    up = st.file_uploader("Add a match video (.mp4)", type=["mp4"], key="vcc_upload")
    if up is not None:
        target = vc.STAGING_VIDEOS / vc.staged_name(up.name)
        if target.exists():
            _html(_alert(f"{target.name} is already in the list above."))
        elif st.button("Add to match videos", key="vcc_save_upload"):
            vc.STAGING_VIDEOS.mkdir(parents=True, exist_ok=True)
            part = target.with_suffix(".part")
            with open(part, "wb") as fh:
                shutil.copyfileobj(up, fh, 16 * 1024 * 1024)
            part.replace(target)
            _staged_videos.clear()
            st.session_state["vcc_video_pending"] = target.name
            st.rerun()
    return chosen["path"] if chosen else None


@st.fragment(run_every=3)
def _kit_check(slug: str, video: str | None, kit: str) -> None:
    _html(_label("Kit contrast check"))
    if not (slug and video):
        _html(_note("Pick the video and name the opponent, then check both kits stand out on camera."))
        return
    proc = vc.active_process(slug, "preflight")
    if st.button("Check Kit Contrast", key="vcc_kit_check", disabled=proc is not None, width="stretch"):
        vc.start_preflight(slug, video, kit)
        st.rerun(scope="fragment")
    if proc:
        elapsed = time.time() - proc.get("started_ts", time.time())
        st.progress(min(0.9, 0.1 + elapsed / 50), text="Checking both kits across five moments of the match…")
        return

    result = vc.read_preflight(slug)
    if not result:
        _html(_note("Takes under a minute. Run it before the analysis so the report can tell the teams apart."))
        return
    if result.get("status") == "error":
        _html(_alert("The kit check couldn't read this video. Try another file.", "red"))
        return
    if result.get("status") == "insufficient":
        _html(_alert("Too few players in view to compare kits. Try another video."))
        return
    tone, verdict = cb.kit_verdict(result.get("silhouette"))
    _html(_pill(verdict, tone))
    _html('<div class="vcc-swatches">'
          + _swatch(result["kit"], "Their shirt (picked)")
          + _swatch(result["opponent"]["hex"], "Their team on camera")
          + _swatch(result["other"]["hex"], "Other team on camera")
          + "</div>")
    if result.get("video") != Path(video).name or result.get("kit") != kit:
        _html(_alert("Checked with a different video or shirt colour. Run the check again."))


def _swatch(colour: str, caption: str) -> str:
    return f'<div class="vcc-swatch"><i style="background:{escape(colour)}"></i><span>{escape(caption)}</span></div>'


@st.fragment(run_every=10)
def _start_analysis(slug: str, video: str | None, kit: str, footage: str) -> None:
    _html(_label("Run the analysis"))
    job = vc.read_job(slug) if slug else None
    proc = vc.active_process(slug, "pipeline") if slug else None
    live = vc.pipeline_state(job, proc) in ("DISPATCHING", *vc.LIVE_STATES)
    if live:
        _html(_note("Analysis in progress for this opponent. Progress shows above the report."))
    elif not (slug and video):
        _html(_note("Pick a video and name the opponent to start."))
    else:
        _html(_note("A full match takes about 30 minutes. Keep working; the report appears here when it's ready."))
    if st.button("Analyze Match", type="primary", key="vcc_analyze", width="stretch",
                 disabled=not (slug and video) or live or proc is not None):
        if vc.start_dispatch(slug, video, kit, footage):
            st.rerun()


# ── Silent progress ──────────────────────────────────────────────────────────

@st.fragment(run_every=5)
def _analysis_status() -> None:
    live = vc.live_opponents()
    for slug in live:
        job, proc, last = vc.read_job(slug), vc.active_process(slug, "pipeline"), vc.process_record(slug, "pipeline")
        if vc.should_auto_collect(job, proc, last):
            vc.start_collect(slug)
        fraction, text = vc.analysis_progress(job, proc)
        with st.container(border=True):
            _html(f'<div class="vcc-status"><b>{escape(vc.display_name(slug))}</b><span>{escape(text)}</span></div>')
            st.progress(fraction)
            _html(_note("Usually 20–30 minutes. Safe to leave this page; the report completes the next time it's open."))

    finished = set(st.session_state.get("vcc_live", [])) - set(live)
    st.session_state["vcc_live"] = live
    if finished:
        st.rerun()   # an analysis just finished: redraw the report below

    slug = _setup_slug()
    if slug and slug not in live:
        job, last = vc.read_job(slug) or {}, vc.process_record(slug, "pipeline")
        if job.get("state") in ("FAILED", "INVALID"):
            _html(_alert(f"The analysis for {vc.display_name(slug)} didn't finish. Check the video plays end to end, then run it again.", "red"))
        elif last and last.get("action") == "dispatch" and (job.get("dispatched_at") or "") < last.get("started_at", ""):
            _html(_alert(f"The analysis for {vc.display_name(slug)} couldn't start. Check the video file and try again.", "red"))


# ── Report ───────────────────────────────────────────────────────────────────

def _report(scouted: list[str]) -> None:
    if not scouted:
        _html('<div class="vcc-empty"><b>No opposition reports yet</b>'
              + _note("Open Match Setup, add the match video, name the opponent and press Analyze Match.") + "</div>")
        return
    order = [r["slug"] for r in vc.history_rows()]
    setup = _setup_slug()
    if st.session_state.get("vcc_report_opp") not in order:
        st.session_state["vcc_report_opp"] = setup if setup in order else order[0]
    if len(order) > 1:
        st.selectbox("Opponent report", order, key="vcc_report_opp", format_func=vc.display_name)
    summary = vc.match_summary(st.session_state["vcc_report_opp"])
    if not summary:
        _html(_alert("This report couldn't be opened. Run the analysis again.", "red"))
        return

    brief = cb.build_brief(summary)
    _html(_brief_card(brief))
    with st.container(border=True):
        _html(_label("Their defensive shape")
              + cb.pitch_svg(vc.shape_geometry(summary["oop"]) if brief["available"] else None)
              + '<div class="vcc-legend"><span><i class="def"></i>Defensive line</span>'
                '<span><i class="press"></i>Press trigger</span><span><i class="corr"></i>Compact corridor</span>'
                '<span><i class="flank"></i>Space on flanks</span></div>')
    if len(order) > 1:
        _library(order)


def _brief_card(brief: dict) -> str:
    conf = brief["confidence"]
    head = ('<div class="vcc-brief-head"><div>'
            f'<div class="vcc-eyebrow">Match-day briefing · {escape(brief["date"])}</div>'
            f'<div class="opp">{escape(brief["opponent"])}</div>'
            + (f'<div class="shape">Team shape: {escape(brief["shape"])}</div>' if brief["shape"] else "")
            + '</div><div class="right">'
            + (f'<span class="vcc-struct">{escape(brief["badge"])}</span>' if brief["badge"] else "")
            + _pill(conf["text"], conf["tone"]) + "</div></div>")
    if not brief["available"]:
        return f'<div class="vcc-brief">{head}<div class="vcc-unavailable">{escape(brief["unavailable"])}</div></div>'
    metrics = "".join(f'<div class="m"><div class="k">{escape(m["label"])}</div><div class="v">{escape(m["value"])}</div>'
                      f'<div class="d">{escape(m["detail"])}</div></div>' for m in brief["metrics"])
    phases = "".join(
        f'<div class="p p{i}"><div class="top"><div class="num">{i}</div><div><div class="t">{escape(p["title"])}</div>'
        f'<div class="st">{escape(p["subtitle"])}</div></div></div>'
        f'<ul>{"".join(f"<li>{escape(pt)}</li>" for pt in p["points"])}</ul></div>'
        for i, p in enumerate(brief["phases"], start=1))
    return f'<div class="vcc-brief">{head}<div class="vcc-metrics">{metrics}</div><div class="vcc-phases">{phases}</div></div>'


def _library(order: list[str]) -> None:
    rows = [cb.library_row(s) for s in (vc.match_summary(slug) for slug in order) if s]
    body = "".join(f'<tr><td class="opp">{r["opponent"]}</td><td>{escape(r["date"])}</td><td>{escape(r["badge"])}</td>'
                   f'<td>{escape(r["line"])}</td><td>{escape(r["press"])}</td>'
                   f'<td>{_pill(r["confidence"]["level"], r["confidence"]["tone"])}</td></tr>' for r in rows)
    with st.container(border=True):
        _html(_label("Scouting library")
              + '<div style="overflow-x:auto"><table class="vcc-lib"><thead><tr><th>Opponent</th><th>Report date</th>'
                '<th>Structure</th><th>Defensive line</th><th>Press trigger</th><th>Confidence</th></tr></thead>'
              + f"<tbody>{body}</tbody></table></div>")
