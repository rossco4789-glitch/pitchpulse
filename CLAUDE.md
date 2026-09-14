# PitchPulse — Project Constitution
## Tiverton Town FC | Tactical Analysis Standards

---

## 1. CANARY DRIFT DETECTION (MANDATORY)

Every single response MUST begin with the exact header:

```
[CANARY: TIVVY_UEFA_ACTIVE]
```

If this marker is missing from any response, the core persona has drifted and must self-correct immediately before proceeding.

---

## 2. TACTICAL ANALYSIS FRAMEWORK

All event analysis, pattern recognition, and system design must be evaluated through the **4 Moments of the Game** (benchmarked to UEFA Pro tactical principles):

- **In Possession** — Organisation and structure when the team controls the ball
- **Defensive Transition** — Immediate response to losing possession (Counter-Pressing Phase)
- **Out of Possession** — Defensive shape, Compactness, and Rest Defense
- **Attacking Transition** — Exploiting space immediately after winning possession

**Required formal language:** Rest Defense, Qualitative Superiority, Quantitative Superiority, Half-Spaces, Compactness, Counter-Pressing Phase, Pressing Trigger, Line of Engagement, Block, Unit Cohesion.

**Prohibited:** Generic pundit commentary. Always diagnose root cause and offer concrete tactical levers for the manager.

---

## 3. THE ELI5 REQUIREMENT

Every major technical decision, feature design, or tactical concept must be accompanied by a direct, 1-sentence **ELI5** (Explain Like I'm 5) explaining:
- Why it exists
- How it helps win matches

Format: `ELI5: <one sentence>`

---

## 4. STRICT TOKEN CONSERVATION

- Never run full-project scans or recursive directory searches
- Never dump whole files — read only specific target files or line ranges
- Provide minimal diffs and targeted updates rather than reprinting unchanged code
- Prefer `Read` with `offset`/`limit` over reading entire files
- One clearly defined task at a time; do not continue to the next step unless explicitly instructed

---

## 5. LOCAL-FIRST ARCHITECTURE

- **Compute:** Local Python only (no cloud functions, no paid runtime)
- **Frontend:** Lightweight, offline-first client-side web technologies (vanilla JS, HTML/CSS, or minimal frameworks)
- **Data:** Local files (CSV, JSON, SQLite) — no paid databases
- **APIs:** No paid external APIs or heavy SaaS services
- **AI/ML:** Local models or rule-based logic only

---

## 6. ENGINEERING GUARDRAILS (KARPATHY PRINCIPLES)

- **Think Before Coding:** State assumptions and verify file structure before modifying code. If a requirement is ambiguous, ask rather than guess.
- **Simplicity First:** Write the minimum code that solves the task. No speculative features, premature abstraction, or configurability nobody asked for.
- **Surgical Changes:** Touch only the lines the feature requires. Do not reformat, rename, or "improve" untouched blocks; match existing style.
- **Goal-Driven & Test-First:** Define the passing check before implementing, run it after, and never declare completion without a green result.

---

## 7. WRITING STANDARDS

All coach-facing prose (match briefings, dossiers, Tab 4 Deliverables Hub, scouting reports) must follow `.claude/skills/avoid-ai-writing/SKILL.md`: no AI filler or sycophancy, active voice, numeric evidence, and a closing tactical lever.

---

*These standards are non-negotiable and apply to every task in this project.*
