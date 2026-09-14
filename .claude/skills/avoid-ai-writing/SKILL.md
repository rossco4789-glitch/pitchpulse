---
name: avoid-ai-writing
description: Prose auditor for PitchPulse deliverables. Use when writing or editing any text a coach, manager or Director of Football will read — match briefings and dossiers (agents/synthesis.py), Tab 4 Deliverables Hub copy (app.py), HTML dossiers (reports/packager.py), DoF match cards (reports/dof_card.py), halftime prompts (analysis/report_prompt.md) and scouting output (tools/scout_harvester.py). Strips AI filler, sycophancy and passive hedging so reports read like a UEFA Pro Licence technical director.
---

# Avoid AI Writing — PitchPulse Prose Auditor

Write the way a technical director talks in a staff meeting: diagnosis, evidence, lever. No warm-up, no sign-off.

## Scope

Apply to every string a coach reads:

| Surface | File |
|---|---|
| Moment agents + approval-gate dossier | `agents/synthesis.py` (`run_*_agent`, `_build_dossier`) |
| Tab 4 Deliverables Hub copy | `app.py` (`# TAB 4 — Deliverables Hub`) |
| HTML dossier | `reports/packager.py` |
| DoF match card bullets | `reports/dof_card.py` |
| Halftime briefing template | `analysis/report_prompt.md` |
| Scouting dossier + terminal summary | `tools/scout_harvester.py` |

Out of scope: code identifiers, log/debug output, test names.

## Hard bans

Never emit these. If one appears, rewrite the sentence — do not swap in a synonym.

**Filler vocabulary:** delve, testament to, tapestry, spearhead, in conclusion, it's worth noting, it is important to note, navigate (figurative), landscape (figurative), realm, pivotal, crucial, robust, seamless, leverage (verb), unlock, elevate, showcase, underscore, multifaceted, holistic, game-changer, a myriad of, plethora, in today's, at the end of the day, when it comes to.

**Sycophancy and chat filler:** Certainly!, Absolutely!, Great question, I'd be happy to, I hope this helps, Let me know if, Here's a breakdown, Let's dive in, As an AI.

**Structural tics:**
- "Not just X, but Y" / "It's not X — it's Y" contrasts.
- Reflexive triplets ("fast, fluid and fearless") where one precise claim would do.
- Closing summaries that repeat the bullets ("Overall…", "In summary…").
- Hedge stacks ("may potentially", "could arguably", "somewhat").
- Em-dash chains — more than one per sentence.

## Required style

1. **Lead with the diagnosis.** First clause states what happened; no scene-setting.
2. **Active voice, named actor.** "Their No.6 screened the Half-Space", not "the Half-Space was screened".
3. **Evidence in numbers.** Minute, zone, count or percentage beside every claim ("3 of 4 corners hit the Near Post").
4. **End on a lever.** Each diagnosis closes with a concrete instruction the manager can give.
5. **Short sentences.** One idea each; aim under 20 words.
6. **Confidence stated plainly.** "Low confidence — 2 sources" beats "it may perhaps be the case".

**Keep** the formal tactical vocabulary CLAUDE.md requires (Rest Defense, Qualitative/Quantitative Superiority, Half-Spaces, Compactness, Counter-Pressing Phase, Pressing Trigger, Line of Engagement, Block, Unit Cohesion). Precise jargon is not filler.

## Rewrites

| AI draft | Technical director |
|---|---|
| It's worth noting that the press was a crucial factor in our struggles. | Their press won the ball 6 times in our defensive third, all from goal kicks. Build up through the 6 instead. |
| This performance is a testament to the team's robust Compactness. | Mid-block held at 28m between lines for 70 minutes; they created 1 shot through the middle. |
| Certainly! Here's a breakdown of the set pieces. | Set pieces: 5 corners, 4 to the Near Post, 0 shots. |
| Not only did we dominate possession, but we also controlled the tempo. | 61% possession; 14 progressive passes through the left Half-Space. |
| In conclusion, the Rest Defense needs to be more solid going forward. | Rest Defense was a 2+1 against their 3-man counter — add the far full-back to make it 3+1. |

## Self-audit before delivering

1. Scan the draft for every hard-ban term (case-insensitive).
2. Delete any opening sentence that does not carry a fact.
3. Delete any closing sentence that only restates.
4. Check each diagnosis has evidence and a lever.
5. For generator code, grep the changed file:

```bash
rg -n -i "delve|testament to|tapestry|spearhead|in conclusion|worth noting|certainly!|happy to|pivotal|crucial|robust|seamless|let's dive" <file>
```

Zero matches is the pass condition.
