"""
cv/vision_center_ui.py — Opposition Analysis (app.py tab 6), written for managers and coaching staff.

Rendering only. Match setup lives in a collapsible drawer; the main screen is the match-day briefing card,
the tactical pitch and the scouting library. Wording comes from cv/coach_brief.py; background work
(upload, analysis, kit check) comes from cv/vision_center.py and runs silently behind progress bars.
"""

from __future__ import annotations

import importlib.util
import shutil
import time
from datetime import datetime
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from cv import briefing_sheet as bs
from cv import club_assets as ca
from cv import coach_brief as cb
from cv import highlight_dossier as hd
from cv import league_roster as lr
from cv import vision_center as vc

MODE_FULL      = "Post-Match Performance"
MODE_HIGHLIGHT = "Opposition Scouting"
MODE_CAPTIONS  = ["Our full match video: pitch calibration, defensive line depth and settled shape.",
                  "Their highlight reels: tag key moments, tendency ratios and a print-ready briefing."]

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
.vcc-quick { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:#30363D; }
.vcc-quick .q { background:#161B22; padding:16px 20px; }
.vcc-quick .threat { box-shadow:inset 4px 0 0 #F85149; } .vcc-quick .weak { box-shadow:inset 4px 0 0 #3FB950; }
.vcc-quick .k { font:600 .68rem Bahnschrift,'Arial Narrow',Arial,sans-serif; letter-spacing:.16em; color:#8B949E; text-transform:uppercase; }
.vcc-quick .v { font:700 1.2rem/1.3 Bahnschrift,'Arial Narrow',Arial,sans-serif; color:#F0F6FC; margin-top:6px; }
.vcc-phases .item { margin:0 0 16px; }
.vcc-phases .il { font:600 .68rem Bahnschrift,'Arial Narrow',Arial,sans-serif; letter-spacing:.14em; color:#8B949E; text-transform:uppercase; }
.vcc-phases .ir { font:.88rem/1.5 'Segoe UI',Inter,system-ui,sans-serif; color:#E6EDF3; margin-top:3px; }
.vcc-phases .ir.empty { color:#6E7681; }
.vcc-phases .iv { font:600 .84rem/1.5 'Segoe UI',Inter,system-ui,sans-serif; color:#3FB950; margin-top:4px; }
.vcc-moment { font:.85rem/1.4 'Segoe UI',Inter,system-ui,sans-serif; color:#E6EDF3; padding:7px 0; border-bottom:1px solid #21262D; }
.vcc-moment span { color:#8B949E; }
.vcc-club { font:700 1.1rem Bahnschrift,'Arial Narrow',Arial,sans-serif; color:#F0F6FC; letter-spacing:.03em; text-transform:uppercase; margin-bottom:4px; }
.vcc-crest-empty { width:52px; height:52px; border-radius:12px; border:1px dashed #30363D; display:grid; place-items:center;
  font:700 1.3rem Bahnschrift,Arial,sans-serif; color:#6E7681; }
@media (max-width: 1000px) { .vcc-metrics, .vcc-phases { grid-template-columns:1fr 1fr; } .vcc-phases .p3 { grid-column:span 2; } }
@media (max-width: 640px) { .vcc-metrics, .vcc-phases, .vcc-quick { grid-template-columns:1fr; } .vcc-phases .p3 { grid-column:auto; }
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


def _opponent_name() -> str:
    """League club from the quick-select, or the typed name when Other / Custom is chosen."""
    return lr.opponent_name(st.session_state.get("vcc_opponent_pick"), st.session_state.get("vcc_opponent"))


def _setup_slug() -> str:
    return vc.slugify(_opponent_name())


# ── Entry point ──────────────────────────────────────────────────────────────

def render() -> None:
    _html(_CSS)
    _html(bs.PRINT_CSS)
    with st.container(key="vcc"):
        highlight = st.session_state.get("vcc_mode", MODE_FULL) == MODE_HIGHLIGHT
        scouted = hd.logged_opponents(vc.SOURCES) if highlight else vc.scouted_opponents()
        subtitle = ("Their threats, weak spots and set pieces, logged from highlight reels." if highlight
                    else "How they defend, where the space is, and where we hold when we lose it.")
        head, action = st.columns([5, 2], vertical_alignment="bottom")
        with head:
            _html('<div class="vcc-head"><div><div class="vcc-eyebrow">Opposition analysis</div>'
                  f'<h2>Match Intelligence Report</h2><p>{escape(subtitle)}</p></div></div>')
        with action:
            st.button("Export Briefing (Print / PDF)", key="vcc_print", icon=":material/print:", width="stretch",
                      disabled=not scouted, on_click=_request_print,
                      help="Opens the print dialog with a one-page A4 sheet. Choose Save as PDF for a file.")
        logging_moments = highlight and bool(_setup_slug())    # keep the drawer open while the analyst logs a reel
        with st.expander("Match Setup", expanded=not scouted or logging_moments, icon=":material/tune:"):
            _match_setup()
        _analysis_status()
        if highlight:
            _dossier_report(scouted)
        else:
            _report(scouted)
        nonce = st.session_state.pop("vcc_print_nonce", None)
        if nonce:
            _print_now(nonce)


def _request_print() -> None:
    st.session_state["vcc_print_nonce"] = time.time()


def _print_now(nonce: float) -> None:
    """Mark the page for the print stylesheet, open the browser print dialog, then restore the page."""
    components.html(f"""<script>
// print request {nonce}
const w = window.parent, d = w.document;
d.documentElement.classList.add('vcc-printing');
const page = d.createElement('style');
page.textContent = '@page {{ size: A4 landscape; margin: 8mm; }}';
d.head.appendChild(page);
w.addEventListener('afterprint', () => {{ d.documentElement.classList.remove('vcc-printing'); page.remove(); }}, {{ once: true }});
setTimeout(() => w.print(), 350);
</script>""", height=0)


def _crest_uri(slug: str) -> str | None:
    club = ca.cached_assets(vc.display_name(slug), sources=vc.SOURCES)
    return bs.crest_data_uri(club["badge_path"]) if club and club.get("badge_path") else None


def _prepared() -> str:
    return datetime.now().strftime("%d %b %Y, %H:%M")


# ── Match Setup drawer ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _staged_videos(signature: tuple) -> list[dict]:
    return vc.staged_videos()


def _video_signature() -> tuple:
    if not vc.STAGING_VIDEOS.exists():
        return ()
    return tuple((p.name, p.stat().st_size, p.stat().st_mtime) for p in sorted(vc.STAGING_VIDEOS.glob("*.mp4")))


@st.cache_data(show_spinner=False, ttl=900)
def _club_assets(name: str, sources: str) -> dict:
    return ca.resolve_club_assets(name, sources=sources)


def _fixture_agent():
    spec = importlib.util.spec_from_file_location("fixture_agent", vc.ROOT / "tools" / "fixture_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_data(show_spinner=False, ttl=3600)
def _next_fixture() -> dict | None:
    try:
        return _fixture_agent().get_next_fixture()
    except Exception:  # the quick-load button is a convenience; never break Match Setup
        return None


def _load_next_fixture(fx: dict) -> None:
    """Fill the opponent pickers from the fixture (league club if it matches, else Other / Custom)."""
    league = {vc.slugify(club): club for club in lr.SOUTHERN_LEAGUE_DIV_ONE_SOUTH}
    club = league.get(vc.slugify(fx["opponent"]))
    st.session_state["vcc_opponent_pick"] = club or lr.CUSTOM
    st.session_state["vcc_opponent"] = "" if club else fx["opponent"]
    st.session_state["vision_fixture"] = {**fx, "preview": _fixture_agent().scaffold_preview_stub(fx)}
    st.session_state["vcc_mode"] = MODE_HIGHLIGHT           # an upcoming opponent is scouted from highlights


def _match_setup() -> None:
    nfx = _next_fixture()
    if nfx:
        st.button(f"Next Fixture: {nfx['opponent']} ({nfx['home_away']})", key="vcc_next_fixture",
                  icon=":material/event:", on_click=_load_next_fixture, args=(nfx,))
    loaded = st.session_state.get("vision_fixture")
    if loaded:
        _html(_note(f"{loaded['home_away']} · {loaded['competition']} · {loaded['date_str']}, {loaded['kickoff']} · "
                    f"{loaded['venue']}. Preview draft: {loaded['preview']}"))
    st.radio("Analysis track", [MODE_FULL, MODE_HIGHLIGHT], key="vcc_mode", horizontal=True, captions=MODE_CAPTIONS)
    highlight = st.session_state.get("vcc_mode") == MODE_HIGHLIGHT
    name = _opponent_name()
    st.session_state["vision_opponent_name"] = name
    club = None
    if name:
        with st.spinner("Looking up the club…"):
            club = _club_assets(name, str(vc.SOURCES))
    c1, c2 = st.columns([2, 1])
    with c1:
        pick = st.selectbox("Opponent · Southern League Division One South", lr.opponent_options(), key="vcc_opponent_pick")
        if pick == lr.CUSTOM:
            st.text_input("Opponent name", key="vcc_opponent", placeholder="e.g. a cup or friendly opponent")
        if club:
            _club_banner(club)
    with c2:
        if highlight:
            reel = st.text_input("Highlight reel", key="vcc_reel", placeholder="e.g. v Weymouth, 12 Aug")
        else:
            kit = _kit_selector(club)
    slug = _setup_slug()

    if highlight:
        _html(_note("Highlight clips can't show team shape. Log each goal, chance, corner and direct free kick; "
                    "the dossier updates as you go."))
        _highlight_video(name)
        _moment_entry(slug, reel)
        return
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        _html(_label("Match video"))
        video = _video_picker()
    with right:
        _kit_check(slug, video, kit.upper())
    _start_analysis(slug, video, kit.upper(), "full_wide")


def _club_banner(club: dict) -> None:
    crest, text = st.columns([1, 6], vertical_alignment="center")
    with crest:
        if club["badge_path"]:
            st.image(club["badge_path"], width=52)
        else:
            _html('<div class="vcc-crest-empty">?</div>')
    with text:
        if not club["verified"]:
            status = _pill("Club not found online · set their kit colour", "amber")
        elif club.get("kit_source") == "club records":
            status = _pill("Club found · kit colours from club records", "green")
        else:
            status = _pill("Club found · kit colours guessed from the crest; check them against the video", "amber")
        _html(f'<div class="vcc-club">{escape(club["name"])}</div>{status}')


def _kit_selector(club: dict | None) -> str:
    home = club["home_kit"] if club else ca.DEFAULT_HOME
    away = club["away_kit"] if club else ca.DEFAULT_AWAY
    choice = st.radio("Their kit", ["Home", "Away"], key="vcc_kit_choice", horizontal=True)
    with st.popover("Custom kit colour", icon=":material/palette:"):
        custom_on = st.checkbox("Use a custom colour (third kit or clash)", key="vcc_kit_custom_on")
        custom = st.color_picker("Custom colour", value=ca.DEFAULT_HOME, key="vcc_kit_custom")
    kit = (custom if custom_on else home if choice == "Home" else away).upper()
    st.session_state["vision_opponent_kit"] = kit
    _html('<div class="vcc-swatches">' + _swatch(kit, "Custom colour" if custom_on else f"{choice} shirt") + "</div>")
    return kit


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
    geo = vc.shape_geometry(summary["oop"]) if brief["available"] else None
    _html(bs.full_sheet(brief, geo, _crest_uri(summary["slug"]), _prepared()))      # hidden until printed
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


# ── Highlight & Tendency Dossier ─────────────────────────────────────────────

MOMENTS_SHOWN = 15


def _highlight_finder():
    spec = importlib.util.spec_from_file_location("highlight_finder", vc.ROOT / "tools" / "highlight_finder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _highlight_video(name: str) -> None:
    """Top-ranked public highlight package for the opponent, shown above the moment logger."""
    if not name:
        return
    hf = _highlight_finder()
    links = hf.cached_highlights(name, vc.SOURCES)
    search = st.button("Search again" if links else "Find highlight videos", key="vcc_hl_search",
                       icon=":material/travel_explore:")
    if search:
        with st.spinner(f"Searching for {name} highlights…"):
            links = hf.find_highlights(name, vc.SOURCES, refresh=True)
        for warning in links.get("warnings", []):
            _html(_alert(warning))
    if not links:
        return
    if not links["videos"]:
        _html(_note(f"No public highlight videos found for {name}."))
        return
    top = links["videos"][0]
    _html(_label("Highlight reel"))
    st.video(top["url"])
    match = f" · {top['fixture']}" if top.get("fixture") else ""
    _html(_note(f"{top['title']} · {top['channel']}{match}. {len(links['videos'])} video{'s' if len(links['videos']) != 1 else ''} found."))


def _moment_entry(slug: str, reel: str) -> None:
    _html(_label("Log a key moment"))
    if not slug:
        _html(_note("Name the opponent to start logging moments."))
        return
    path = hd.events_path(vc.SOURCES, slug)
    t1, t2 = st.columns([2, 1])
    with t1:
        kind = st.selectbox("Moment", list(hd.KINDS), key="vcc_hl_kind", format_func=lambda k: hd.KINDS[k]["label"])
    with t2:
        minute = st.number_input("Minute", min_value=0, max_value=130, value=None, step=1, placeholder="e.g. 63", key="vcc_hl_minute")
    group = hd.KINDS[kind]["group"]
    values = {}
    for col, (name, field) in zip(st.columns(len(hd.FIELDS[group])), hd.FIELDS[group].items()):
        with col:
            key = f"vcc_hl_{group}_{name}"
            values[name] = (st.selectbox(field["label"], field["options"], key=key) if field["options"]
                            else st.text_input(field["label"], key=key, placeholder=field["placeholder"]))
    if st.button("Add Moment", type="primary", key="vcc_hl_add", width="stretch"):
        hd.add_event(path, kind, values, minute=minute, source=reel)
        st.toast(f"{hd.KINDS[kind]['label']} added")
        st.rerun()

    events = hd.load_events(path)
    if not events:
        return
    _html(_label(f"Logged moments · {len(events)}"))
    for event in reversed(events[-MOMENTS_SHOWN:]):
        line, remove = st.columns([12, 1], vertical_alignment="center")
        source = f' <span>· {escape(event["source"])}</span>' if event.get("source") else ""
        line.markdown(f'<div class="vcc-moment">{escape(hd.describe_event(event))}{source}</div>', unsafe_allow_html=True)
        if remove.button(":material/close:", key=f"vcc_hl_del_{event['id']}", help="Remove this moment"):
            hd.delete_event(path, event["id"])
            st.rerun()
    if len(events) > MOMENTS_SHOWN:
        _html(_note(f"Showing the latest {MOMENTS_SHOWN} of {len(events)} moments."))


def _dossier_report(logged: list[str]) -> None:
    if not logged:
        _html('<div class="vcc-empty"><b>No highlight dossiers yet</b>'
              + _note("Open Match Setup, choose Opposition Scouting, name the opponent and log key moments.") + "</div>")
        return
    setup = _setup_slug()
    if setup in logged and st.session_state.get("vcc_dossier_setup") != setup:
        st.session_state["vcc_dossier_opp"] = setup      # follow the opponent being logged
    st.session_state["vcc_dossier_setup"] = setup
    if st.session_state.get("vcc_dossier_opp") not in logged:
        st.session_state["vcc_dossier_opp"] = logged[0]
    if len(logged) > 1:
        st.selectbox("Opponent dossier", logged, key="vcc_dossier_opp", format_func=vc.display_name)
    slug = st.session_state["vcc_dossier_opp"]
    dossier = hd.build_dossier(vc.display_name(slug), hd.load_events(hd.events_path(vc.SOURCES, slug)))
    _html(_dossier_card(dossier))
    _html(bs.dossier_sheet(dossier, _crest_uri(slug), _prepared()))                   # hidden until printed


def _dossier_card(d: dict) -> str:
    conf, quick = d["confidence"], d["quick_read"]
    head = ('<div class="vcc-brief-head"><div><div class="vcc-eyebrow">Opposition scouting dossier</div>'
            f'<div class="opp">{escape(d["opponent"])}</div><div class="shape">{escape(d["note"])}</div></div>'
            f'<div class="right"><span class="vcc-struct">{escape(d["footage"].upper())}</span>'
            f'{_pill(conf["text"], conf["tone"])}</div></div>')
    quick_read = ('<div class="vcc-quick">'
                  f'<div class="q threat"><div class="k">Primary threat</div><div class="v">{escape(quick["threat"])}</div></div>'
                  f'<div class="q weak"><div class="k">Primary vulnerability</div><div class="v">{escape(quick["vulnerability"])}</div></div>'
                  "</div>")
    sections = []
    for i, section in enumerate(d["sections"], start=1):
        items = []
        for item in section["items"]:
            empty = "" if item["lever"] else " empty"
            lever = f'<div class="iv">→ {escape(item["lever"])}</div>' if item["lever"] else ""
            items.append(f'<div class="item"><div class="il">{escape(item["label"])}</div>'
                         f'<div class="ir{empty}">{escape(item["read"])}</div>{lever}</div>')
        sections.append(f'<div class="p p{i}"><div class="top"><div class="num">{i}</div>'
                        f'<div class="t">{escape(section["title"])}</div></div>{"".join(items)}</div>')
    return f'<div class="vcc-brief">{head}{quick_read}<div class="vcc-phases">{"".join(sections)}</div></div>'
