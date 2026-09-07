# PitchPulse — Project Index
## Tiverton Town FC | Tactical Performance Platform

---

## Architecture Overview

```
PitchPulse/
├── CLAUDE.md                  # System constitution & persona constraints
├── PROJECT_INDEX.md           # This file — master map of the stack
├── requirements.txt           # Python dependencies (pandas, mplsoccer, opencv-python, streamlit)
├── app.py                     # ✅ COMPLETE — Streamlit desktop dashboard; 5 tabs; OLED dark; local-only; Tab 2 Clip Workspace; Tab 4 Archive & Clear
├── .streamlit/config.toml     # Streamlit theme: OLED #09090b bg, Tivvy Amber #f59e0b primary
├── run_matchday.py            # ✅ COMPLETE — Master pipeline runner (reconcile → visuals → agents)
├── tagger/
│   └── index.html             # ✅ COMPLETE (v2.2) — Broadcast-grade OLED tactical pad; SVG letterbox fix; UNDO toast; ↔ SUB modal with player-off/on selects
├── reconcile/
│   └── sync.py                # ✅ COMPLETE — Post-match reconciliation engine; ±90s temporal match, roster resolution, JSON/TXT feed, ledger output
├── cv/
│   ├── zones.py               # ✅ COMPLETE — 18-Zone tactical matrix (105×68m); bbox, centroid, get_zone_by_coords(), get_zone_centroid()
│   ├── calibration.py         # ✅ COMPLETE — Tier 1: planar homography computation; interactive anchor picker; save/load camera_calibration.json
│   ├── picker.py              # ✅ COMPLETE — Tier 1: pixel_to_pitch(); build_video_event(); append_video_event() → ledger; CLI + frame picker
│   └── tests/
│       └── test_homography.py # ✅ COMPLETE — 21-test synthetic suite; no image/GPU required; 21/21 passing
├── reports/
│   ├── visualizer.py          # ✅ COMPLETE — OLED-dark mplsoccer engine; shot map, transition map, zonal heatmap; multi_match_shot_map/heatmap added
│   ├── packager.py            # ✅ COMPLETE — Self-contained HTML dossier packager; base64 PNGs, OLED dark, mobile responsive, iOS Safari A4 print
│   ├── dof_card.py            # ✅ COMPLETE — 1080×1920 DoF match card PNG; OLED dark, Tivvy amber, KPI tiles, pitch miniatures, exec bullets
│   ├── progress_review.py     # ✅ COMPLETE — Longitudinal tactical review engine; rolling 4–8 game window; CLI + Streamlit Tab 5
│   └── video_engine.py        # ✅ COMPLETE — FFmpeg clip engine; dual kick-off offset sync; stream-copy slicing; M3U playlist; tactical section export (4 windows)
├── agents/
│   ├── __init__.py            # Package marker
│   └── synthesis.py           # ✅ COMPLETE — 3-agent UEFA tactical analysis engine + CLI approval gate
├── data/
│   ├── raw/                   # Drop zone: tagger JSON exports + club feed files
│   ├── parse_report.py        # ✅ COMPLETE — Parses .docx match report → match_context.json + tivvy_x_feed.json
│   └── processed/             # Output: match_ledger.json + plots/ + dossier_*.md + dof_match_card.png
└── analysis/
    ├── engine.py              # Local Python engine — shot maps, turnover maps, box entries
    └── report_prompt.md       # Tactical briefing template — 3-bullet halftime diagnosis (4 Moments framework)
```

---

## File Responsibilities

### `CLAUDE.md`
System constitution and persona constraints. Governs all agent behaviour:
canary drift detection, tactical analysis framework, ELI5 requirement, token conservation, and local-first architecture rules.

### `app.py` — Streamlit Desktop Dashboard
GUI alternative to the terminal pipeline. **Does not modify `run_matchday.py`.**

**Launch:**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Local only — no cloud, no external traffic.

**Tabs:**
| Tab | Purpose |
|-----|---------|
| 📥 Match Ingestion | Upload tagger JSONs + .docx; auto-parse to match_context.json + tivvy_x_feed.json |
| 🎥 Veo Video Lab | Local video frame extraction; Plotly click-picker; homography calibration; video event logging; **Clip Workspace** — dual kick-off sync offsets, per-event ✂ Clip buttons, inline `st.video()` preview, M3U playlist export; **Tactical Section Export** — 4 fixed windows (Opening 15, 1H Final 15, 2H Opening 15, Final 20) sliced to `sections/` with M3U for manager review |
| 🧠 Agent Cockpit | Run reconcile→visuals→agents pipeline; review 4 agent outputs; direct approve/reject gate (bypasses terminal `input()`) |
| 📦 Deliverables Hub | Preview DoF card + HTML dossier; download buttons; **Archive & Clear** — auto-stamps `ledger_DD-MM-YYYY.json` for Progress Review, moves working files to `data/archive/{date}_{opponent}/`, resets session state, leaves clips untouched |

State persists across tab switches via `st.session_state`. Uploaded files are staged to `data/raw/staged/` before backend processing so existing function signatures (which expect `Path`) receive valid paths.

### `run_matchday.py`
Master matchday orchestration script. Runs the full pipeline in four sequential steps:
1. `reconcile/sync.py` → `data/processed/match_ledger.json`
2. `reports/visualizer.py` → `data/processed/plots/*.png`
3. `agents/synthesis.py` → CLI approval gate → `reports/dossier_<date>.md`
4. `reports/packager.py` → `data/processed/tivvy_tactical_dossier.html`  *(runs only on approve)*

Flags: `--skip-reconcile`, `--skip-visuals`, `--latest`

### `tagger/index.html`
Offline-first mobile tap-pad for live match tagging. **v2.2: SUB event added.**

**Step 1:** Select player chip + tap action button.  
**Step 2:** Full-screen SVG pitch overlay appears — optional sub-type chip, then **tap pitch** to record real spatial coordinates. Auto-dismisses on pitch tap.

Attacking direction auto-sets: `1H → attacking_right=true`, `2H → attacking_right=false`. Manual override via the `→ ATT` toggle button at all times.

Coordinate calculation uses `getBoundingClientRect()` with `touchstart` + `preventDefault()` to block scroll/zoom distortion. Ghost-click suppressed via `lastTouchTime` guard. Coordinates clamped to `[0.0, 1.0]`.

**v2.1 fixes:** `aspect-ratio: 360 / 232` on `#pitch-svg` eliminates portrait-mode SVG letterbox coordinate drift (previously caused all flank taps to mis-map as half-space). 1-touch UNDO button (`#btn-undo`) removes last event from localStorage with amber toast confirmation; disabled when no events are logged.

| Event type     | Sub-types                               | Notes |
|----------------|-----------------------------------------|-------|
| `SHOT`         | `ON_TARGET`, `OFF_TARGET`, `BLOCKED`    | |
| `BOX_ENTRY`    | `PASS`, `CROSS`, `CARRY`                | |
| `DEF_TURNOVER` | *(none)*                                | |
| `HIGH_REGAIN`  | *(none)*                                | |
| `SET_PIECE`    | `ATT_CORNER`, `DEF_CORNER`, `FREE_KICK` | |
| `AERIAL_DUEL`  | `WON`, `LOST`                           | **Non-league physics** |
| `SECOND_BALL`  | `WON`, `LOST`                           | **Non-league physics** |
| `SUB`          | *(none — uses `player_off` + `player_on`)* | No pitch tap; inline modal |

**SUB event** is logged via `logSubEvent(playerOff, playerOn)` — no pitch overlay, no spatial data. Taps `↔ SUB` button → inline centred modal with two native `<select>` dropdowns (#1–#18) → `LOG SUB` confirms. Guard: same shirt both ends → red border flash, no log.

**v2 event object schema** (additions in bold):

```json
{
  "id": "uuid", "period": "1H", "match_seconds": 1234,
  "clock_display": "20:34", "player_num": 9,
  "event_type": "SHOT", "sub_type": "ON_TARGET",
  "x_pct": 0.8720, "y_pct": 0.5140,
  "x_m": 91.56, "y_m": 34.95,
  "attacking_right": true,
  "timestamp_iso": "2026-09-06T15:20:34.000Z"
}
```

`zone_id` is NOT computed in the tagger — derived by `reconcile/sync.py` via `cv.zones.get_zone_by_coords(x_m, y_m)`.

Outputs tagged events as local JSON. No server required.

### `reconcile/sync.py`
Post-match reconciliation engine. Merges tagger JSON exports with the club's X feed (JSON or TXT)
using a ±90 s temporal window. Resolves shirt numbers to player names. Writes `match_ledger.json`.

**v2 additions:**
- `RECONCILABLE_TAGS` expanded to include `AERIAL_DUEL`, `SECOND_BALL`
- `_stamp_zone(tag)`: for every tag event with `x_m`, `y_m`, calls `cv.zones.get_zone_by_coords()`
  and stamps `zone_id` + `zone_name` in-place. Legacy events without spatial data get `zone_id: null`.
- Ledger carries `schema_version: 2` and `summary.events_with_zone` count.

**v2.2 additions (substitution support):**
- `reconcile_events()`: non-`RECONCILABLE_TAGS` events (e.g. `SUB`) now pass through to
  `unmatched_tags` instead of being silently dropped.
- `build_ledger()`: extracts `SUB` events into a dedicated top-level `"substitutions"` list;
  stamps `player_off_name` / `player_on_name` via `resolve_player()`; keeps `unmatched_tags`
  spatial-events-only. `summary.substitutions` count added.
- `print_audit()`: shows substitution count in the terminal audit table.

### `cv/zones.py`
18-Zone tactical matrix on a 105 × 68 m FIFA pitch.
Zone ID convention: `{THIRD}_{CHANNEL}` — e.g. `A_LH` (attacking left half-space), `A_LC` / `A_RC` (Zone 14).

### `cv/calibration.py`
Tier 1 homography calibration — standalone midweek utility, not part of `run_matchday.py`.

Computes the 3×3 planar homography matrix **H** that maps camera pixel coordinates `(u, v)` to
real-world pitch coordinates `(x_m, y_m)` on the 105 × 68 m pitch.

**Core functions:**
- `calibrate_pitch(reference_points, pitch_dims) → np.ndarray` — calls `cv2.findHomography` (RANSAC) on ≥4 anchor pairs
- `reprojection_error(reference_points, H) → float` — mean pixel error across all reference points
- `save_calibration(H, reference_points, out_path, reprojection_error_px) → Path` — writes `camera_calibration.json`
- `load_calibration(calib_path) → np.ndarray` — returns H from JSON
- `load_calibration_meta(calib_path) → dict` — full JSON payload

**13 standard anchors** defined in `ANCHORS[]` (penalty spots, penalty area corners, halfway-line intersections).
Reprojection quality bands: GOOD < 3 px · WARN < 8 px · POOR ≥ 8 px.

**CLI modes:**
```bash
# Interactive — click anchors on a still frame in matplotlib:
python cv/calibration.py --image data/raw/frame.jpg --anchors 1 2 3 4 5

# Non-interactive — supply pre-mapped pairs as JSON:
python cv/calibration.py --points-json data/raw/my_points.json --out data/raw/camera_calibration.json

# Verify an existing calibration:
python cv/calibration.py --verify data/raw/camera_calibration.json
```

ELI5: It teaches the computer where the pitch lines are in the camera view so it can convert pixel positions to real metres.

### `cv/picker.py`
Tier 1 pixel-to-pitch converter and ledger amendment tool — standalone midweek utility.

**Core functions:**
- `pixel_to_pitch(u, v, H) → (x_m, y_m)` — homogeneous matrix multiply; clamped to `[0, 105] × [0, 68]`
- `resolve_zone(x_m, y_m) → (zone_id, zone_name)` — delegates to `cv.zones.get_zone_by_coords()` (spatial integrity contract upheld)
- `build_video_event(u, v, H, event_type, ...) → dict` — v2-compatible event with `source="video_assisted"`
- `append_video_event(ledger_path, event) → Path` — appends to `unmatched_tags` in `match_ledger.json`; increments `summary.video_assisted_events`

**Spatial integrity contract:** zone_id is derived exclusively via `cv.zones.get_zone_by_coords()` — no sub-type inference from pixel position.

**v2 event additions:** `source`, `zone_id`, `zone_name`, `pixel_u`, `pixel_v`, `video_timestamp_s`.

**CLI modes:**
```bash
# Supply pixel coordinate directly:
python cv/picker.py --pixel 540 320 --event-type SHOT --sub-type ON_TARGET \
  --match-seconds 4620 --period 2H --ledger data/processed/match_ledger.json

# Click a video frame interactively:
python cv/picker.py --frame data/raw/frame.jpg --event-type BOX_ENTRY --sub-type CARRY \
  --match-seconds 2830 --period 1H --ledger data/processed/match_ledger.json
```

ELI5: It takes a pixel you click on a video still, works out where that is on the real pitch in metres, and saves it to the match data file.

### `cv/tests/test_homography.py`
21-test synthetic unit suite. No image file, no camera, no GPU required.

A fixed synthetic perspective matrix generates pixel coordinates from known world points.
`calibrate_pitch()` is asked to recover the mapping; `pixel_to_pitch()` is verified against
ground truth at ±0.15 m tolerance.

| Test group | Tests | Key assertions |
|---|---|---|
| `calibrate_pitch` | 4 | 3×3 float64, raises on <4 pts, reprojection error < 1 px |
| `pixel_to_pitch` | 4 | ±15 cm accuracy, clamp high/low, degenerate w≈0 fallback |
| `resolve_zone` | 5 | Centre=M_*, pen spot=A_*C*, out of bounds=None, Zone 14 LC+RC |
| `build_video_event` | 6 | Required keys, source, clock_display, x_pct range, zone_id, video_ts |
| `save/load` | 2 | H round-trip within fp tolerance, FileNotFoundError on missing |

Run: `python cv/tests/test_homography.py`  or  `python -m pytest cv/tests/test_homography.py -v`

### `reports/video_engine.py`
FFmpeg-based match clip slicing engine. Local-only; no cloud, no re-encoding.

**Sync model — dual kick-off offsets:**
| Period | Formula |
|--------|---------|
| 1H | `veo_t = offset_1h + match_seconds` |
| 2H / ET | `veo_t = offset_2h + (match_seconds − 2700)` |

`offset_1h` = seconds into the Veo file at the 1H whistle; `offset_2h` = seconds into the Veo file at the 2H restart whistle. Eliminates half-time stoppage drift without per-event manual correction.

**Key functions:**
- `check_ffmpeg() → bool` — `shutil.which("ffmpeg")` guard; UI surfaces install instructions if absent
- `calculate_clip_bounds(match_seconds, period, offset_1h, offset_2h, lead_in=5, follow_through=3) → (start_s, end_s)` — clamped to `≥ 0`
- `slice_clip(video_path, start_s, end_s, output_path) → (bool, str)` — `ffmpeg -ss {start} -to {end} -i {input} -c copy -y {output}`; near-instant stream copy
- `build_clip_path(clips_root, match_date, opponent, match_seconds, event_type, sub_type, zone_id) → Path` — e.g. `data/clips/15-08-2026_st-blazey/72m_a-lc_shot_on-target.mp4`
- `export_playlist_m3u(clip_paths, m3u_path)` — extended M3U for VLC / mpv coach presentation
- `_section_windows(offset_1h, offset_2h) → list[(label, start_s, end_s)]` — 4 fixed tactical windows in Veo file time
- `export_tactical_sections(video_path, output_dir, offset_1h, offset_2h) → (succeeded, errors)` — slices full match into 4 analytical windows; writes `tactical_sections.m3u`

**Tactical sections:**
| # | Label | Match time |
|---|---|---|
| 01 | Opening 15 min | Kick-off → 15′ |
| 02 | 1H Final 15 min | 30′ → 45′ |
| 03 | 2H Opening 15 min | 2H restart → 60′ |
| 04 | Final 20 min | 70′ → 90′ |

**Output directories:**
- Event clips: `data/clips/{DD-MM-YYYY}_{opponent-slug}/`
- Tactical sections: `data/clips/{DD-MM-YYYY}_{opponent-slug}/sections/`

### `reports/visualizer.py`
OLED-dark mplsoccer pitch rendering engine. Accepts a `list[dict]` of events with optional
`x, y` or `zone_id` keys. Missing spatial data falls back to zone centroids (from `cv/zones.py`).
Produces `shot_map.png`, `transition_map.png`, `zonal_heatmap.png`.

### `agents/synthesis.py`
Multi-agent tactical analysis engine. **v3: context-aware — loads `data/raw/match_context.json`
produced by `data/parse_report.py` to enrich every agent with match facts.**
Four specialist agents produce a structured markdown dossier, then a human-in-the-loop CLI
approval gate (`approve` / `reject <feedback>` / `quit`) validates it before writing to disk.
Max 3 rejection cycles, then forced approval. `run_approval_gate()` returns `Path | None`
(the written dossier Path on approve, `None` on quit/interrupt).

**v2.2 additions (substitution support):**
- `_build_sub_timeline(ledger)`: reads `ledger["substitutions"]`, returns list sorted by `match_seconds`.
- `_resolve_player(player_num, match_seconds, sub_timeline)`: resolves shirt number to name; emits
  `⚠subbed off HH:MM′ — #N Name entered` annotation on any event tagged after that player's exit —
  data quality flag, not a correction.
- `_format_subs_line(sub_timeline)`: formats `#9 D. Waters → #14 N. Aves (67:00)` strings for the
  dossier header.
- `_build_dossier()`: adds `**Tagger-confirmed Subs:**` line when substitutions present; shows sub
  count in DATA QUALITY footer.

| Agent | Events | UEFA Formal Language | v3 Context Layer |
|---|---|---|---|
| In-Possession | `SHOT`, `BOX_ENTRY` | Half-Spaces, Qualitative Superiority, Zone 14 | — |
| Out-of-Possession / Press | `HIGH_REGAIN`, `DEF_TURNOVER` | Counter-Pressing Phase, Line of Engagement (spatial), Compactness (channel spread) | **Score-state split**: leading vs level/trailing phase efficiency |
| Set-Piece | `SET_PIECE` | Rest Defense, Unit Cohesion, Attacking Transition | Opposition name, competition, tactical keyword context |
| Non-League Physics | `AERIAL_DUEL`, `SECOND_BALL` | Second Ball, Direct Play, Quantitative Superiority | — |

**Dossier header (v3):** Full match facts from `match_context.json` — opponent, competition,
venue (home/away), scoreline, scorer names with minute_raw display, full XI with sub windows,
subs used. Gracefully falls back to ledger-only mode if `match_context.json` is absent.

**Score timeline:** Built from `ctx['scorers']` → `_build_score_timeline()`. Each tagged event
is annotated with the live score at its `match_seconds` via `_score_at_second()`. Agent 2 splits
possession-change events into leading vs level/trailing phases for state-conditional analysis.

**Spatial integrity contract:** Zone classification is derived exclusively from `zone_id` stamped
by `reconcile/sync.py` → `cv.zones.get_zone_by_coords(x_m, y_m)`. No sub-type inference tables.
Events without `zone_id` (legacy v1 exports) are flagged explicitly in the dossier output.

### `reports/packager.py`
Self-contained HTML dossier packager. Converts the approved `dossier_*.md` into a fully
offline HTML file — no CDN, no external fonts, no JavaScript.

- **Inputs:** dossier markdown string, `plots_dir/` containing the three PNGs
- **Output:** `data/processed/tivvy_tactical_dossier.html`
- Base64-embeds `shot_map.png`, `transition_map.png`, `zonal_heatmap.png`; renders a
  styled placeholder `<div>` if any PNG is missing or zero bytes
- Section mapping: MOMENT 1 → shot map, MOMENT 2 & 3 → transition map, SET PIECE → zonal heatmap, NON-LEAGUE PHYSICS → full-width (no plot)
- OLED dark theme (`#09090b` / `#f59e0b` gold), 2-column CSS Grid (55 %/45 %)
- `@media print`: white A4, `break-before: page` per section, `break-inside: avoid` on grids/alerts/blockquotes
- Regex-only Markdown→HTML conversion: bold, italic, H1/H2, `>` manager notes, emoji alert boxes (🚨 ⚡ ⚠), bar-chart code blocks

ELI5: It turns the text report into a one-file webpage the manager can open on any device — pictures and all — with no internet needed.

### `reports/dof_card.py`
Director of Football matchday summary card. Renders a 1080×1920 px (9:16 portrait) OLED-dark
PNG at `data/processed/plots/dof_match_card.png` — sized for immediate WhatsApp delivery.

**Layout sections (top to bottom):**
1. **Top Banner** — club crest, scoreline (large), result badge, teams, competition, venue/date
2. **KPI Tiles (2×2)** — Box Entries (dominant corridor), Press Efficiency %, Aerial Win %, Second Ball Recovery %
3. **Pitch Miniatures** — inset shot map (attacking actions) + transition map (press & turnovers)
4. **Tactical Takeaways** — 3 rule-based tactical bullets (In Possession / Pressing / Priority Lever)
5. **Data Quality** — schema version, matched/zoned event counts

**RAG system:** Green ≥ threshold, Amber = contested, Red = alert. Thresholds: Press ≥60%/40%,
Aerial ≥55%/45%, Second Ball ≥55%/40%. Half-space box entry dominance → green.

**Fallbacks:** Missing crest → amber "TT" circle. Missing plots → labelled placeholder. Missing
`match_context.json` → banner shows ledger date only; all KPI tiles still render.

**Step 5 in `run_matchday.py`** — runs unconditionally after HTML packaging (non-blocking).
Also callable standalone: `python reports/dof_card.py`

ELI5: It makes one picture with all the important numbers so the DoF can see everything on their phone without opening any files.

### `analysis/engine.py`
Legacy local Python analytical engine (pre-reconcile era). Retained for reference.

### `analysis/report_prompt.md`
Structured template for a 3-bullet halftime tactical diagnosis (4 Moments framework).

---

## Data Flow

```
[Mobile Tagger (tagger/index.html)]
    │  tivvy_events_1H.json / tivvy_events_2H.json
    ▼
[reconcile/sync.py]  +  tivvy_x_feed.json / .txt (club feed)
    │  data/processed/match_ledger.json
    ▼
[reports/visualizer.py]
    │  data/processed/plots/{shot_map,transition_map,zonal_heatmap}.png
    ▼
[agents/synthesis.py]
    │  CLI approval gate  (approve / reject / quit)
    ▼
[reports/dossier_YYYY-MM-DD.md]
    │
    ▼
[reports/packager.py]   ← base64-embeds PNGs; OLED dark + A4 print CSS
    │
    ▼
[data/processed/tivvy_tactical_dossier.html]  →  Manager Briefing (offline, self-contained)
```

Single command: `python run_matchday.py`

---

## Dependencies

| Package     | Purpose                          | Cost |
|-------------|----------------------------------|------|
| pandas      | Event data wrangling             | Free |
| mplsoccer   | Pitch drawing & visualisation    | Free |
| matplotlib  | Rendering backend (Agg)          | Free |
| numpy       | Heatmap computation              | Free |

---

*All components run locally. Zero paid services. Zero cloud dependency.*
