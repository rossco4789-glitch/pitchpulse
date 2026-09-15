"""
cv/coach_brief.py — coach-facing language and pitch graphic for the Opposition Analysis tab (app.py tab 6).

Turns the four out-of-possession measurements (metres from the goal the opponent defends) into match-day
instructions: touchline distances in yards, a structure badge, three phases of play and an SVG pitch.
Pure functions, no Streamlit. Sentences follow .claude/skills/avoid-ai-writing/SKILL.md: numbers, then a lever.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

M_TO_YD            = 1.09361
PITCH_L, PITCH_W   = 105.0, 68.0
BOX_DEPTH_M        = 16.5
THIRD_M            = 35.0
HALFWAY_M          = 52.5
LOW_BLOCK_MAX_M    = 18.0       # same Low / Mid / High cut-offs as cv/vision_center.block_badge
MID_BLOCK_MAX_M    = 32.0
COMPACT_MAX_M      = 25.0
NARROW_MAX_M       = 25.0
OPEN_FLANK_M       = 12.0       # free width per flank that makes a switch of play the instruction
REST_DEFENCE_GAP_M = 15.0       # rest-defence line sits this far beyond their furthest-forward outfielder
OOP_KEYS = ("block_height_m", "compactness_depth_m", "compactness_width_m", "line_of_engagement_m")


def yards(metres: float) -> int:
    return int(round(metres * M_TO_YD))


# ── Coaching dictionary ──────────────────────────────────────────────────────

def line_depth(block_m: float) -> tuple[str, str]:
    detail = ("Inside the 18-yard box" if block_m <= BOX_DEPTH_M else "In their defensive third" if block_m <= THIRD_M
              else "In their own half" if block_m <= HALFWAY_M else "Beyond halfway")
    return f"{yards(block_m)} yards from goal", detail


def press_trigger(loe_m: float) -> tuple[str, str]:
    detail = ("Inside their own box" if loe_m <= BOX_DEPTH_M else "Own defensive third" if loe_m <= THIRD_M
              else "Middle third, short of halfway" if loe_m <= 48.0 else "Halfway line" if loe_m <= 57.0 else "In our half")
    return f"Engage at {yards(loe_m)} yards", detail


def defensive_width(width_m: float) -> tuple[str, str]:
    label = "Narrow central block" if width_m <= NARROW_MAX_M else "Medium-width block" if width_m <= 40.0 else "Wide block"
    return label, f"{width_m:.0f} m corridor"


def team_length(depth_m: float) -> tuple[str, str]:
    label = "Compact" if depth_m <= COMPACT_MAX_M else "Moderate spacing" if depth_m <= 35.0 else "Stretched"
    return label, f"{depth_m:.0f} m front-to-back"


def structure_badge(block_m: float, depth_m: float, loe_m: float) -> str:
    compact = depth_m <= COMPACT_MAX_M
    if block_m < LOW_BLOCK_MAX_M:
        return "DEEP COMPACT LOW BLOCK" if compact else "DEEP LOW BLOCK"
    if block_m <= MID_BLOCK_MAX_M:
        return "COMPACT MID BLOCK" if compact else "MID BLOCK"
    return "HIGH PRESSING BLOCK" if loe_m > HALFWAY_M else "HIGH BLOCK"


def team_shape(block_m: float, depth_m: float, width_m: float) -> str:
    width = "Narrow" if width_m <= NARROW_MAX_M else "Medium width" if width_m <= 40.0 else "Wide"
    length = "Compact" if depth_m <= COMPACT_MAX_M else "Spaced" if depth_m <= 35.0 else "Stretched"
    height = "Sits deep" if block_m < LOW_BLOCK_MAX_M else "Holds mid-pitch" if block_m <= MID_BLOCK_MAX_M else "Steps high"
    return f"{width} · {length} · {height}"


def confidence(summary: dict) -> dict:
    """Share of the match the camera mapped to the pitch, and moments with their defence in view together."""
    mapped, moments = summary.get("acceptance") or 0.0, summary.get("settled_frames") or 0
    if mapped >= 0.7 and moments >= 300:
        return {"level": "High", "tone": "green", "text": "Data Confidence: High · full-match sample"}
    if mapped >= 0.5 and moments >= 100:
        return {"level": "Medium", "tone": "amber", "text": "Data Confidence: Medium · check key moments on video"}
    return {"level": "Low", "tone": "red", "text": "Data Confidence: Low · camera caught part of their shape; confirm on video"}


def kit_verdict(score: float | None) -> tuple[str, str]:
    if score is None:
        return "grey", "Not checked"
    if score >= 0.55:
        return "green", "Kits clearly different"
    if score >= 0.50:
        return "amber", "Kits close: pick their exact shirt colour"
    return "red", "Kits too similar: pick their exact shirt colour"


def unavailable_reason(oop: dict) -> str:
    reason = next((r for r in ((oop.get(k) or {}).get("reason") for k in OOP_KEYS) if r), "")
    if reason == "opponent_kit_not_given":
        return "Their shirt colour wasn't set. Add it in Match Setup and run the analysis again."
    if reason.startswith("team_split_ambiguous"):
        return "Their kit looked too close to the other team on camera. Pick their exact shirt colour and run it again."
    if reason.startswith("insufficient_coverage"):
        return "Their defence was in view together too rarely. Wide-angle full-match video gives a complete report."
    if reason == "highlight_footage_invalid_for_shape":
        return "Highlight clips can't show a defensive shape. Use the full-match video."
    if reason.startswith("too_few_player_detections"):
        return "Too few players were visible. Use a wider camera angle."
    return "Defensive shape isn't available for this match."


def phases(block_m: float, loe_m: float, depth_m: float, width_m: float) -> list[dict]:
    loe_yd, from_ours_yd = yards(loe_m), yards(PITCH_L - loe_m)
    if loe_m <= THIRD_M:
        build = [f"They start pressing {loe_yd} yards from their own goal, {from_ours_yd} yards from ours.",
                 "Centre-backs step in with the ball and carry it to halfway.",
                 "Holding midfielder takes it in the centre circle facing forward; no need to go long."]
    elif loe_m <= HALFWAY_M:
        build = [f"They start pressing {loe_yd} yards from their own goal, short of halfway.",
                 "Centre-backs split wide and the holding midfielder drops between them to make three.",
                 "First pass goes past their front line into a midfielder between the lines."]
    else:
        build = [f"They press into our half, {from_ours_yd} yards from our goal.",
                 "Keeper joins the build-up and the centre-backs split to the edge of the box.",
                 "Play past their front line or go direct to the striker; no square passes across our box."]

    flank_m = (PITCH_W - width_m) / 2
    space = ([f"Their block covers a {width_m:.0f} m central corridor, leaving {yards(flank_m)} yards free on each flank.",
              "Switch play early: the full-back overlaps the wide forward on the far side."]
             if flank_m >= OPEN_FLANK_M else
             [f"Their block covers {width_m:.0f} m of the 68 m width, so the flanks stay closed.",
              "Overload one Half-Space with the 8, the 10 and the wide forward, then switch."])
    if block_m <= BOX_DEPTH_M:
        space.append(f"Their back line sits {yards(block_m)} yards out: cut the ball back to the penalty spot and the edge of the box.")
    elif block_m <= MID_BLOCK_MAX_M:
        space.append(f"Their back line holds {yards(block_m)} yards from goal: the striker runs in behind as soon as the ball goes wide.")
    else:
        space.append(f"Their back line holds {yards(block_m)} yards from goal: play over the top for the striker and wide forwards.")
    space.append(f"Only {depth_m:.0f} m between their front and back lines: go around them, not through the middle."
                 if depth_m <= COMPACT_MAX_M else
                 f"{depth_m:.0f} m between their front and back lines: find the 10 between the lines and turn.")

    rest_m = min(HALFWAY_M, loe_m + REST_DEFENCE_GAP_M)
    rest = [f"Their furthest-forward player holds {loe_yd} yards from their goal when they defend.",
            (f"Rest defence: both centre-backs and the holding midfielder hold {yards(rest_m)} yards from their goal, "
             f"{yards(HALFWAY_M - rest_m)} yards inside their half.") if rest_m < HALFWAY_M else
            "Rest defence: both centre-backs and the holding midfielder hold on the halfway line.",
            "Win their clearances there before their front player can turn and run."]
    if flank_m >= OPEN_FLANK_M:
        rest.append("When the ball is on one flank, the far-side full-back tucks in level with the rest-defence line.")

    return [{"title": "In Possession", "subtitle": "How We Build", "points": build},
            {"title": "Breaking Them Down", "subtitle": "Where the Space Is", "points": space},
            {"title": "Transition & Rest Defence", "subtitle": "Where We Hold", "points": rest}]


def _report_date(iso_date: str) -> str:
    try:
        return datetime.strptime(iso_date[:10], "%Y-%m-%d").strftime("%d %b %Y")
    except (TypeError, ValueError):
        return ""


def build_brief(summary: dict) -> dict:
    oop = summary.get("oop") or {}
    vals = {k: (oop.get(k) or {}).get("value") for k in OOP_KEYS}
    brief = {"opponent": summary.get("opponent", ""), "date": _report_date(summary.get("date", "")),
             "confidence": confidence(summary)}
    if any(v is None for v in vals.values()):
        return {**brief, "available": False, "badge": None, "shape": None, "metrics": [], "phases": [],
                "unavailable": unavailable_reason(oop)}
    block, depth = vals["block_height_m"], vals["compactness_depth_m"]
    width, loe = vals["compactness_width_m"], vals["line_of_engagement_m"]
    metrics = [("Defensive Line Depth", line_depth(block)), ("Press Trigger / Line of Engagement", press_trigger(loe)),
               ("Defensive Width", defensive_width(width)), ("Team Length / Spacing", team_length(depth))]
    return {**brief, "available": True, "unavailable": None, "badge": structure_badge(block, depth, loe),
            "shape": team_shape(block, depth, width),
            "metrics": [{"label": label, "value": v, "detail": d} for label, (v, d) in metrics],
            "phases": phases(block, loe, depth, width)}


def brief_text(brief: dict) -> str:
    """Every visible string in a brief, for jargon checks."""
    parts = [brief["opponent"], brief["date"], brief["confidence"]["text"], brief.get("badge") or "",
             brief.get("shape") or "", brief.get("unavailable") or ""]
    parts += [f"{m['label']} {m['value']} {m['detail']}" for m in brief["metrics"]]
    parts += [f"{p['title']} {p['subtitle']} {' '.join(p['points'])}" for p in brief["phases"]]
    return " ".join(parts)


# ── Pitch graphic ────────────────────────────────────────────────────────────

def _n(v: float) -> str:
    return f"{v:.2f}"


PITCH_THEMES = {
    # "dark" matches the dashboard; "print" is the low-ink match-day sheet with its own element ids
    "dark":  {"id": "vcc", "surround": "#1B4A21", "grass": ("#2F7F36", "#23652B"), "mow": 0.065, "turf": True,
              "line": "#F4F7F1", "net": "#FFFFFF", "flank_text": "#F2CC60", "flank_sub": "#FFFFFF", "halo": "#0D1117",
              "def_label": "#79C0FF", "press_label": "#FF7B72", "caption": "#C9D1D9"},
    "print": {"id": "vccp", "surround": "#FFFFFF", "grass": ("#EEF6EC", "#E2EEDF"), "mow": 0.35, "turf": False,
              "line": "#2E6B33", "net": "#2E6B33", "flank_text": "#7A5A00", "flank_sub": "#1E3A1E", "halo": "#FFFFFF",
              "def_label": "#1F5FBF", "press_label": "#B3261E", "caption": "#3D4A3D"},
}


def pitch_svg(geo: dict | None, theme: str = "dark") -> str:
    """Single-line SVG (Markdown-safe) of a mown pitch; the opponent defends the left goal.

    `geo` is cv/vision_center.shape_geometry output: defensive line, press line, compact corridor and centroid.
    theme "print" draws the same pitch in low-ink colours for the match-day sheet.
    """
    t = PITCH_THEMES[theme]
    p = t["id"]
    L, W, arc = PITCH_L, PITCH_W, 7.31   # arc: sqrt(9.15² − 5.5²), where the penalty arc meets the box edge
    s = ['<svg class="vcc-pitch" viewBox="-6 -8 117 86" xmlns="http://www.w3.org/2000/svg" role="img" '
         'aria-label="Opposition defensive shape" font-family="Bahnschrift,Arial Narrow,Arial,sans-serif">',
         '<defs>',
         f'<linearGradient id="{p}-grass" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{t["grass"][0]}"/>'
         f'<stop offset="1" stop-color="{t["grass"][1]}"/></linearGradient>',
         f'<pattern id="{p}-mow" width="21" height="68" patternUnits="userSpaceOnUse">'
         f'<rect width="10.5" height="68" fill="#FFFFFF" fill-opacity="{t["mow"]}"/></pattern>',
         f'<filter id="{p}-turf"><feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="3" seed="4"/>'
         '<feColorMatrix type="saturate" values="0"/></filter>',
         f'<pattern id="{p}-mesh" width="2" height="2" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         '<line x1="0" y1="0" x2="0" y2="2" stroke="#79C0FF" stroke-opacity="0.55" stroke-width="0.22"/></pattern>',
         f'<pattern id="{p}-net" width="0.7" height="0.7" patternUnits="userSpaceOnUse">'
         f'<path d="M0 0H0.7M0 0V0.7" stroke="{t["net"]}" stroke-opacity="0.55" stroke-width="0.09"/></pattern>',
         '</defs>',
         f'<rect x="-6" y="-8" width="117" height="86" rx="2" fill="{t["surround"]}"/>',
         f'<rect x="0" y="0" width="{L:g}" height="{W:g}" fill="url(#{p}-grass)"/>',
         f'<rect x="0" y="0" width="{L:g}" height="{W:g}" fill="url(#{p}-mow)"/>']
    if t["turf"]:
        s.append(f'<rect x="-6" y="-8" width="117" height="86" filter="url(#{p}-turf)" opacity="0.07"/>')
    s += [f'<g fill="none" stroke="{t["line"]}" stroke-width="0.3" stroke-linecap="round" stroke-linejoin="round">',
          f'<rect x="0" y="0" width="{L:g}" height="{W:g}"/>',
          f'<line x1="52.5" y1="0" x2="52.5" y2="{W:g}"/>',
          '<circle cx="52.5" cy="34" r="9.15"/>',
          '<rect x="0" y="13.84" width="16.5" height="40.32"/><rect x="88.5" y="13.84" width="16.5" height="40.32"/>',
          '<rect x="0" y="24.84" width="5.5" height="18.32"/><rect x="99.5" y="24.84" width="5.5" height="18.32"/>',
          f'<path d="M16.5 {_n(34 - arc)} A9.15 9.15 0 0 1 16.5 {_n(34 + arc)}"/>',
          f'<path d="M88.5 {_n(34 - arc)} A9.15 9.15 0 0 0 88.5 {_n(34 + arc)}"/>',
          '<path d="M0 1A1 1 0 0 0 1 0M104 0A1 1 0 0 0 105 1M105 67A1 1 0 0 0 104 68M1 68A1 1 0 0 0 0 67"/>',
          '</g>',
          f'<g fill="{t["line"]}"><circle cx="11" cy="34" r="0.35"/><circle cx="94" cy="34" r="0.35"/>'
          '<circle cx="52.5" cy="34" r="0.4"/></g>']
    for x in (-2.0, L):
        s.append(f'<rect x="{x:g}" y="30.34" width="2" height="7.32" fill="url(#{p}-net)" stroke="{t["line"]}" stroke-width="0.3"/>')

    if geo:
        (back, y0), (front, _), (_, y1), _ = geo["zone"]
        if y0 >= OPEN_FLANK_M:   # free width on each side of the corridor
            reach = min(HALFWAY_M, front + 12.0)
            for top, height in ((0.0, y0), (y1, W - y1)):
                cy = top + height / 2
                s.append(f'<rect class="flank" x="0" y="{_n(top)}" width="{_n(reach)}" height="{_n(height)}" fill="#3FB950" '
                         'fill-opacity="0.22" stroke="#E3B341" stroke-width="0.3" stroke-dasharray="1 0.7"/>')
                s.append(f'<text x="{_n(reach / 2)}" y="{_n(cy - 0.4)}" text-anchor="middle" font-size="2.3" font-weight="700" '
                         f'fill="{t["flank_text"]}" letter-spacing="0.15" paint-order="stroke" stroke="{t["halo"]}" '
                         'stroke-width="0.5">SPACE ON FLANKS</text>')
                s.append(f'<text x="{_n(reach / 2)}" y="{_n(cy + 2.6)}" text-anchor="middle" font-size="1.9" font-weight="700" '
                         f'fill="{t["flank_sub"]}" paint-order="stroke" stroke="{t["halo"]}" stroke-width="0.45">Switch Play Early</text>')
        s.append(f'<rect class="corridor" x="{_n(back)}" y="{_n(y0)}" width="{_n(front - back)}" height="{_n(y1 - y0)}" '
                 'fill="#388BFD" fill-opacity="0.22" stroke="#58A6FF" stroke-width="0.3"/>')
        s.append(f'<rect x="{_n(back)}" y="{_n(y0)}" width="{_n(front - back)}" height="{_n(y1 - y0)}" fill="url(#{p}-mesh)"/>')
        s.append(f'<line class="def-line" x1="{_n(geo["block"])}" y1="0" x2="{_n(geo["block"])}" y2="{W:g}" '
                 'stroke="#388BFD" stroke-width="0.75"/>')
        s.append(f'<line class="press-line" x1="{_n(geo["loe"])}" y1="0" x2="{_n(geo["loe"])}" y2="{W:g}" '
                 'stroke="#F85149" stroke-width="0.6" stroke-dasharray="1.6 1.1"/>')
        s.append(f'<text x="{_n(max(0.4, geo["block"]))}" y="71.9" font-size="2" font-weight="700" fill="{t["def_label"]}" '
                 f'letter-spacing="0.2">DEFENSIVE LINE · {yards(geo["block"])} YDS</text>')
        s.append(f'<text x="{_n(max(0.4, geo["loe"]))}" y="-2.3" font-size="2" font-weight="700" fill="{t["press_label"]}" '
                 f'letter-spacing="0.2">PRESS TRIGGER · {yards(geo["loe"])} YDS</text>')

    s.append(f'<text x="-3.7" y="34" transform="rotate(-90 -3.7 34)" text-anchor="middle" font-size="1.7" fill="{t["caption"]}" '
             'letter-spacing="0.3">THEIR GOAL</text>')
    s.append(f'<text x="{L:g}" y="71.9" text-anchor="end" font-size="1.8" font-weight="700" fill="{t["caption"]}" '
             'letter-spacing="0.2">WE ATTACK ←</text>')
    s.append("</svg>")
    return "".join(s)

def library_row(summary: dict) -> dict:
    brief = build_brief(summary)
    by_label = {m["label"]: m["value"] for m in brief["metrics"]}
    return {"opponent": escape(brief["opponent"]), "date": brief["date"], "badge": brief["badge"] or "Shape not available",
            "line": by_label.get("Defensive Line Depth", "—"),
            "press": by_label.get("Press Trigger / Line of Engagement", "—"), "confidence": brief["confidence"]}
