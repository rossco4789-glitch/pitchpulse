# PitchPulse — Project Index
## Tiverton Town FC | Zero-Budget Performance Stack

---

## Architecture Overview

```
PitchPulse/
├── CLAUDE.md                  # System constitution & persona constraints
├── PROJECT_INDEX.md           # This file — master map of the stack
├── requirements.txt           # Python dependencies (pandas, mplsoccer)
├── tagger/
│   └── index.html             # ✅ COMPLETE (v2) — Broadcast-grade OLED tactical pad; glassmorphic HUD, SVG icons, semantic gradients, drift-free clock
└── analysis/
    ├── engine.py              # Local Python engine — shot maps, turnover maps, box entries
    └── report_prompt.md       # Tactical briefing template — 3-bullet UEFA Pro halftime diagnosis
```

---

## File Responsibilities

### `CLAUDE.md`
System constitution and persona constraints. Governs all agent behaviour:
canary drift detection, UEFA Pro tactical standard, ELI5 requirement, token conservation, and zero-budget architecture rules.

### `tagger/index.html`
Offline-first mobile tap-pad for live match tagging. Captures 5 core UEFA match actions:
- Shot
- Pass (Key Pass)
- Turnover Won
- Turnover Lost
- Box Entry

Outputs tagged events as local JSON for downstream analysis. No server required — runs entirely in the browser.

### `analysis/engine.py`
Local Python analytical engine. Uses **pandas** for event data wrangling and **mplsoccer** for pitch visualisation. Produces:
- Shot maps (xG-weighted where applicable)
- Turnover maps (won/lost by zone)
- Box entry maps (approach corridor analysis)

Zero external API calls. All inputs are local JSON/CSV files from the tagger.

### `analysis/report_prompt.md`
Structured template for generating a 3-bullet UEFA Pro Licence halftime tactical diagnosis. Covers:
1. In Possession finding
2. Out of Possession finding
3. Transition priority recommendation

---

## Data Flow

```
[Mobile Tagger] → local JSON → [engine.py] → pitch maps + report_prompt.md → [Manager Briefing]
```

---

## Dependencies

| Package     | Purpose                          | Cost |
|-------------|----------------------------------|------|
| pandas      | Event data wrangling             | Free |
| mplsoccer   | Pitch drawing & visualisation    | Free |

---

*All components run locally. Zero paid services. Zero cloud dependency.*
