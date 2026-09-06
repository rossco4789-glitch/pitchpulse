# PitchPulse — Project Index
## Tiverton Town FC | Zero-Budget Performance Stack

---

## Architecture Overview

```
PitchPulse/
├── CLAUDE.md                  # System constitution & persona constraints
├── PROJECT_INDEX.md           # This file — master map of the stack
├── requirements.txt           # Python dependencies (pandas, mplsoccer)
├── run_matchday.py            # ✅ COMPLETE — Master pipeline runner (reconcile → visuals → agents)
├── tagger/
│   └── index.html             # ✅ COMPLETE (v2) — Broadcast-grade OLED tactical pad; glassmorphic HUD, SVG icons, semantic gradients, drift-free clock
├── reconcile/
│   └── sync.py                # ✅ COMPLETE — Post-match reconciliation engine; ±90s temporal match, roster resolution, JSON/TXT feed, ledger output
├── cv/
│   └── zones.py               # ✅ COMPLETE — 18-Zone tactical matrix (105×68m); bbox, centroid, get_zone_by_coords(), get_zone_centroid()
├── reports/
│   └── visualizer.py          # ✅ COMPLETE — OLED-dark mplsoccer engine; shot map, transition map, zonal heatmap → data/processed/plots/
├── agents/
│   ├── __init__.py            # Package marker
│   └── synthesis.py           # ✅ COMPLETE — 3-agent UEFA tactical analysis engine + CLI approval gate
├── data/
│   ├── raw/                   # Drop zone: tagger JSON exports + club feed files
│   └── processed/             # Output: match_ledger.json + plots/ + dossier_*.md
└── analysis/
    ├── engine.py              # Local Python engine — shot maps, turnover maps, box entries
    └── report_prompt.md       # Tactical briefing template — 3-bullet UEFA Pro halftime diagnosis
```

---

## File Responsibilities

### `CLAUDE.md`
System constitution and persona constraints. Governs all agent behaviour:
canary drift detection, UEFA Pro tactical standard, ELI5 requirement, token conservation, and zero-budget architecture rules.

### `run_matchday.py`
Master matchday orchestration script. Runs the full pipeline in three sequential steps:
1. `reconcile/sync.py` → `data/processed/match_ledger.json`
2. `reports/visualizer.py` → `data/processed/plots/*.png`
3. `agents/synthesis.py` → CLI approval gate → `reports/dossier_<date>.md`

Flags: `--skip-reconcile`, `--skip-visuals`, `--latest`

### `tagger/index.html`
Offline-first mobile tap-pad for live match tagging. **v2: 2-step capture flow.**

**Step 1:** Select player chip + tap action button.  
**Step 2:** Full-screen SVG pitch overlay appears — optional sub-type chip, then **tap pitch** to record real spatial coordinates. Auto-dismisses on pitch tap.

Attacking direction auto-sets: `1H → attacking_right=true`, `2H → attacking_right=false`. Manual override via the `→ ATT` toggle button at all times.

Coordinate calculation uses `getBoundingClientRect()` with `touchstart` + `preventDefault()` to block scroll/zoom distortion. Ghost-click suppressed via `lastTouchTime` guard. Coordinates clamped to `[0.0, 1.0]`.

| Event type     | Sub-types                               | Notes |
|----------------|-----------------------------------------|-------|
| `SHOT`         | `ON_TARGET`, `OFF_TARGET`, `BLOCKED`    | |
| `BOX_ENTRY`    | `PASS`, `CROSS`, `CARRY`                | |
| `DEF_TURNOVER` | *(none)*                                | |
| `HIGH_REGAIN`  | *(none)*                                | |
| `SET_PIECE`    | `ATT_CORNER`, `DEF_CORNER`, `FREE_KICK` | |
| `AERIAL_DUEL`  | `WON`, `LOST`                           | **Non-league physics** |
| `SECOND_BALL`  | `WON`, `LOST`                           | **Non-league physics** |

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

### `cv/zones.py`
18-Zone tactical matrix on a 105 × 68 m FIFA pitch.
Zone ID convention: `{THIRD}_{CHANNEL}` — e.g. `A_LH` (attacking left half-space), `A_LC` / `A_RC` (Zone 14).

### `reports/visualizer.py`
OLED-dark mplsoccer pitch rendering engine. Accepts a `list[dict]` of events with optional
`x, y` or `zone_id` keys. Missing spatial data falls back to zone centroids (from `cv/zones.py`).
Produces `shot_map.png`, `transition_map.png`, `zonal_heatmap.png`.

### `agents/synthesis.py`
Multi-agent tactical analysis engine. **v2: all spatial analysis uses real `zone_id` from ledger.**
Four specialist agents produce a structured markdown dossier, then a human-in-the-loop CLI
approval gate (`approve` / `reject <feedback>` / `quit`) validates it before writing to disk.
Max 3 rejection cycles, then forced approval.

| Agent | Events | UEFA Formal Language |
|---|---|---|
| In-Possession | `SHOT`, `BOX_ENTRY` | Half-Spaces, Qualitative Superiority, Zone 14 |
| Out-of-Possession / Press | `HIGH_REGAIN`, `DEF_TURNOVER` | Counter-Pressing Phase, Line of Engagement (spatial), Compactness (channel spread) |
| Set-Piece | `SET_PIECE` | Rest Defense, Unit Cohesion, Attacking Transition |
| Non-League Physics | `AERIAL_DUEL`, `SECOND_BALL` | Second Ball, Direct Play, Quantitative Superiority |

**Spatial integrity contract:** Zone classification is derived exclusively from `zone_id` stamped
by `reconcile/sync.py` → `cv.zones.get_zone_by_coords(x_m, y_m)`. No sub-type inference tables.
Events without `zone_id` (legacy v1 exports) are flagged explicitly in the dossier output.

### `analysis/engine.py`
Legacy local Python analytical engine (pre-reconcile era). Retained for reference.

### `analysis/report_prompt.md`
Structured template for a 3-bullet UEFA Pro Licence halftime tactical diagnosis.

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
    │  CLI approval gate
    ▼
[reports/dossier_YYYY-MM-DD.md]  →  Manager Briefing
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
