"""
cv/highlight_dossier.py — Highlight & Tendency Dossier for the Opposition Analysis tab (app.py tab 6).

Highlight clips cannot show team shape, so this mode never touches the video analysis. The analyst logs goals,
chances, corners and direct free kicks from YouTube / Wyscout reels in Match Setup; the dossier counts them into
a coach-facing report. Moments live in data/scouting/sources/<slug>/highlight_events.json.

Every option is a fixed football term so counts stay reliable. Wording follows
.claude/skills/avoid-ai-writing/SKILL.md: the count first, then the instruction.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path

SCHEMA_VERSION = 1
EVENTS_FILE    = "highlight_events.json"
TEXT_MAX       = 24     # shirt number or surname
SOURCE_MAX     = 60

# ── Moment vocabulary (sides are always the opponent's own left / right) ─────

CHANNELS    = ("Left wing", "Left half-space", "Central", "Right half-space", "Right wing")
ACTIONS     = ("Cross", "Cut-back", "Through ball", "Shot from distance", "Dribble", "Rebound")
ARRIVALS    = ("Near post", "Far post", "Penalty spot", "Six-yard box", "Edge of the box", "Outside the box")
TRIGGERS    = ("Not a counter", "Regain in midfield", "Interception in own half", "After defending a corner", "Keeper quick release")
FLAWS       = ("Isolated 1v1", "Space behind full-back", "Second ball after clearance", "Cut-back not tracked",
               "Beaten in the air", "Set piece")
SIDES       = ("Their left", "Central", "Their right")
DELIVERIES  = ("Inswinger", "Outswinger", "Driven", "Short")
CORNER_ZONE = ("Near post", "Central six-yard box", "Far post", "Penalty spot", "Edge of the box")
CORNER_FOR  = ("Goal", "Shot", "Won by defence")
MARKING     = ("Zonal", "Man-to-man", "Hybrid")
CORNER_AGST = ("Cleared", "Second ball lost", "Shot conceded", "Goal conceded")
FEET        = ("Right foot", "Left foot")
FK_RESULTS  = ("Goal", "On target", "Off target", "Blocked by wall")

KINDS = {
    "goal_scored":     {"label": "Goal scored", "group": "attack"},
    "chance_created":  {"label": "Chance created", "group": "attack"},
    "goal_conceded":   {"label": "Goal conceded", "group": "defence"},
    "chance_conceded": {"label": "Chance conceded", "group": "defence"},
    "corner_for":      {"label": "Corner taken", "group": "corner_for"},
    "corner_against":  {"label": "Corner defended", "group": "corner_against"},
    "free_kick":       {"label": "Direct free kick", "group": "free_kick"},
}


def _opt(label, options):
    return {"label": label, "options": options, "placeholder": ""}


def _txt(label, placeholder="Shirt no. or name"):
    return {"label": label, "options": None, "placeholder": placeholder}


FIELDS = {
    "attack": {"channel": _opt("Channel (their side)", CHANNELS), "action": _opt("Created by", ACTIONS),
               "arrival": _opt("Finished from", ARRIVALS), "counter_trigger": _opt("Counter-attack", TRIGGERS)},
    "defence": {"flaw": _opt("What went wrong", FLAWS), "side": _opt("Where (their side)", SIDES),
                "player": _txt("Defender at fault")},
    "corner_for": {"delivery": _opt("Delivery", DELIVERIES), "target": _opt("Aimed at", CORNER_ZONE),
                   "player": _txt("Target player"), "outcome": _opt("Result", CORNER_FOR)},
    "corner_against": {"marking": _opt("Their marking", MARKING), "outcome": _opt("Result", CORNER_AGST)},
    "free_kick": {"player": _txt("Taker"), "foot": _opt("Foot", FEET), "outcome": _opt("Result", FK_RESULTS)},
}


# ── Moments and storage ──────────────────────────────────────────────────────

def new_event(kind: str, fields: dict, minute: int | None = None, source: str = "") -> dict:
    """Validated moment. Raises ValueError for an unknown type, an option outside the list or a bad minute."""
    if kind not in KINDS:
        raise ValueError(f"unknown moment type {kind!r}")
    clean = {}
    for name, spec in FIELDS[KINDS[kind]["group"]].items():
        value = str(fields.get(name) or "").strip()
        if spec["options"] and value not in spec["options"]:
            raise ValueError(f"{spec['label']}: {value!r} is not one of the options")
        clean[name] = value if spec["options"] else value[:TEXT_MAX]
    if minute is not None and not 0 <= int(minute) <= 130:
        raise ValueError("minute must be between 0 and 130")
    return {"id": uuid.uuid4().hex[:8], "kind": kind, "fields": clean, "minute": None if minute is None else int(minute),
            "source": str(source or "").strip()[:SOURCE_MAX], "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def events_path(sources: Path, slug: str) -> Path:
    return Path(sources) / slug / EVENTS_FILE


def load_events(path: Path) -> list[dict]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [e for e in doc.get("events", []) if isinstance(e, dict) and e.get("kind") in KINDS] if isinstance(doc, dict) else []


def _save(path: Path, events: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "events": events}, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def add_event(path: Path, kind: str, fields: dict, minute: int | None = None, source: str = "") -> dict:
    event = new_event(kind, fields, minute, source)
    _save(path, load_events(path) + [event])
    return event


def delete_event(path: Path, event_id: str) -> bool:
    events = load_events(path)
    kept = [e for e in events if e.get("id") != event_id]
    if len(kept) == len(events):
        return False
    _save(path, kept)
    return True


def logged_opponents(sources: Path) -> list[str]:
    sources = Path(sources)
    return sorted(p.parent.name for p in sources.glob(f"*/{EVENTS_FILE}") if load_events(p)) if sources.exists() else []


def player_label(value: str) -> str:
    return f"No. {value}" if value.isdigit() else value


def describe_event(event: dict) -> str:
    parts = [KINDS[event["kind"]]["label"]]
    for name, value in event.get("fields", {}).items():
        if value and value != "Not a counter":
            parts.append(player_label(value) if name == "player" else value)
    if event.get("minute") is not None:
        parts.append(f"{event['minute']}'")
    return " · ".join(parts)


# ── Dossier ──────────────────────────────────────────────────────────────────

ACTION_PLURAL   = {"Cross": "crosses", "Cut-back": "cut-backs", "Through ball": "through balls",
                   "Shot from distance": "shots from distance", "Dribble": "dribbles", "Rebound": "rebounds"}
ARRIVAL_LEVER   = {"Near post": "Near-post defender attacks the first ball; nobody runs across in front of them.",
                   "Far post": "Far-side full-back tucks in to mark the back post.",
                   "Penalty spot": "Holding midfielder tracks the late runner to the penalty spot.",
                   "Six-yard box": "Keeper commands the six-yard box; centre-backs hold the posts.",
                   "Edge of the box": "A midfielder guards the edge of the box on every cross."}
TRIGGER_PHRASE  = {"Regain in midfield": "a regain in midfield", "Interception in own half": "an interception in their own half",
                   "After defending a corner": "defending a corner", "Keeper quick release": "a quick keeper release"}
TRIGGER_LEVER   = {"Regain in midfield": "Keep two holding players behind the ball when our midfield commits forward.",
                   "Interception in own half": "No square passes across midfield in their half; play forward or long.",
                   "After defending a corner": "Leave two defenders and the keeper back on our corners.",
                   "Keeper quick release": "Recover shape the moment their keeper holds the ball; nobody jogs back."}
DELIVERY_PLURAL = {"Inswinger": "Inswingers", "Outswinger": "Outswingers", "Driven": "Driven deliveries", "Short": "Short corners"}
CORNER_LEVER    = {"Near post": "Put a zonal marker on the near post to win the first ball.",
                   "Central six-yard box": "Keeper and a zonal marker own the six-yard box.",
                   "Far post": "Far-post marker starts goal-side; the keeper stays on the line unless certain.",
                   "Penalty spot": "Man-mark the penalty-spot runner; no free header.",
                   "Edge of the box": "Send a midfielder to the edge of the box for the pull-back."}
MARKING_PHRASE  = {"Zonal": "Zonal marking", "Man-to-man": "Man-to-man marking", "Hybrid": "Hybrid marking (zones plus markers)"}
MARKING_LEVER   = {"Zonal": "Attack the gaps between their zonal markers with runs from deep.",
                   "Man-to-man": "Use blocks to free our main target from the marker.",
                   "Hybrid": "Crowd the keeper to pin their zonal players, then aim at the far post."}
FLAW_PHRASE     = {"Isolated 1v1": "Defenders isolated 1v1", "Second ball after clearance": "Second balls after clearances",
                   "Cut-back not tracked": "Cut-backs not tracked", "Beaten in the air": "Beaten in the air", "Set piece": "Set pieces"}
OUR_SIDE        = {"Their left": "right", "Their right": "left"}


def _count(events: list[dict], field: str, skip: tuple = ()) -> Counter:
    return Counter(v for v in (e["fields"].get(field) for e in events) if v and v not in skip)


def _top(counter: Counter) -> tuple[str | None, int]:
    return counter.most_common(1)[0] if counter else (None, 0)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _item(label: str, read: str, lever: str = "") -> dict:
    return {"label": label, "read": read, "lever": lever}


def _goals_chances(total: int, conceded: bool = False) -> str:
    words = "goal or chance" if total == 1 else "goals and chances"
    return f"{words} conceded" if conceded else words


def _channel_phrase(channel: str) -> str:
    if channel == "Central":
        return "through the middle"
    return f"down their {channel.lower()}" if "wing" in channel else f"through their {channel.split()[0].lower()} Half-Space"


def _channel_lever(channel: str) -> str:
    if channel == "Central":
        return "Holding midfielder stays in front of the centre-backs; block shooting lanes at the edge of the box."
    side = channel.split()[0].lower()
    ours = "right" if side == "left" else "left"
    if "wing" in channel:
        return f"Our {ours}-back and {ours} midfielder double up on their {channel.lower()}; force play inside."
    return f"Holding midfielder screens their {side} Half-Space; our {ours} centre-back steps out on runners."


def _confidence(n: int) -> dict:
    if n >= 20:
        return {"level": "High", "tone": "green", "text": f"Data Confidence: High · {n} moments logged"}
    if n >= 8:
        return {"level": "Medium", "tone": "amber", "text": f"Data Confidence: Medium · {n} moments logged; add reels to confirm"}
    return {"level": "Low", "tone": "red", "text": f"Data Confidence: Low · {_plural(n, 'moment')} logged; treat as leads, not patterns"}


def _threat(attack: list[dict], corners_for: list[dict]) -> str:
    if attack:
        channel, n = _top(_count(attack, "channel"))
        return f"Attacks {_channel_phrase(channel)} ({n} of {len(attack)} {_goals_chances(len(attack))})"
    goals = sum(e["fields"]["outcome"] == "Goal" for e in corners_for)
    return f"Attacking corners ({_plural(goals, 'goal')} from {len(corners_for)})" if goals else "Log goals and chances to find their main threat"


def _vulnerability(defence: list[dict], corners_against: list[dict]) -> str:
    flaws = _count(defence, "flaw")
    second = sum(e["fields"]["outcome"] == "Second ball lost" for e in corners_against)
    if second:
        flaws["Second ball after clearance"] += second
    if not flaws:
        return "Log goals and chances conceded to find their weak spot"
    flaw, n = _top(flaws)
    if flaw == "Space behind full-back":
        side, _ = _top(_count([e for e in defence if e["fields"]["flaw"] == flaw], "side"))
        phrase = "Space behind their centre-backs" if side == "Central" else f"Space behind their {side.split()[1]}-back"
    else:
        phrase = FLAW_PHRASE[flaw]
    return f"{phrase} ({n} of {len(defence) + second} conceded moments)"


def _creation(attack: list[dict]) -> dict:
    label = "Creation channel"
    if not attack:
        return _item(label, "No goals or chances logged yet.")
    channel, n = _top(_count(attack, "channel"))
    action, m = _top(_count(attack, "action"))
    return _item(label, f"{n} of {len(attack)} {_goals_chances(len(attack))} came {_channel_phrase(channel)}, "
                        f"most often from {ACTION_PLURAL[action]} ({m}).", _channel_lever(channel))


def _arrivals(attack: list[dict]) -> dict:
    label = "Box arrival runs"
    if not attack:
        return _item(label, "No goals or chances logged yet.")
    inside = [e for e in attack if e["fields"]["arrival"] != "Outside the box"]
    if not inside:
        return _item(label, f"All {len(attack)} came from outside the box.", "Close shooters down within 25 yards; no free strikes.")
    zone, n = _top(_count(inside, "arrival"))
    return _item(label, f"{n} of {len(inside)} finishes in the box arrived at the {zone.lower()}.", ARRIVAL_LEVER[zone])


def _counters(attack: list[dict]) -> dict:
    label = "Counter-attack triggers"
    if not attack:
        return _item(label, "No goals or chances logged yet.")
    counters = [e for e in attack if e["fields"]["counter_trigger"] != "Not a counter"]
    if not counters:
        read = ("The one goal or chance logged didn't come on the counter." if len(attack) == 1
                else f"None of {len(attack)} goals and chances came on the counter.")
        return _item(label, read,
                     "Their goals come from set attacks: hold a set shape before pushing numbers forward.")
    trigger, n = _top(_count(counters, "counter_trigger"))
    return _item(label, f"{len(counters)} of {len(attack)} came on the counter, most often after {TRIGGER_PHRASE[trigger]} ({n}).",
                 TRIGGER_LEVER[trigger])


def _isolation(defence: list[dict]) -> dict:
    label = "Isolated 1v1 matchups"
    if not defence:
        return _item(label, "No goals or chances conceded logged yet.")
    iso = [e for e in defence if e["fields"]["flaw"] == "Isolated 1v1"]
    if not iso:
        return _item(label, f"No isolated 1v1 defending in {len(defence)} {_goals_chances(len(defence), True)}.")
    player, p = _top(_count(iso, "player"))
    side, _ = _top(_count(iso, "side"))
    read = f"Isolated 1v1 in {len(iso)} of {len(defence)} {_goals_chances(len(defence), True)}" + (
        f", {player_label(player)} most often ({p})." if player else ".")
    if side in OUR_SIDE:
        target = player_label(player) if player else f"their {side.split()[1]}-sided defender"
        return _item(label, read, f"Get our {OUR_SIDE[side]} winger 1v1 against {target} early.")
    return _item(label, read, "Run the striker at their centre-backs 1v1 early.")


def _behind(defence: list[dict]) -> dict:
    label = "Space behind full-backs"
    if not defence:
        return _item(label, "No goals or chances conceded logged yet.")
    behind = [e for e in defence if e["fields"]["flaw"] == "Space behind full-back"]
    conceded = _goals_chances(len(defence), True)
    if not behind:
        return _item(label, f"No {conceded} came in behind the full-backs." if len(defence) == 1
                     else f"None of {len(defence)} {conceded} came in behind the full-backs.")
    side, n = _top(_count(behind, "side"))
    if side == "Central":
        return _item(label, f"Space behind their centre-backs in {n} of {len(defence)} {conceded}.",
                     "Striker runs in behind between the centre-backs when their full-backs push on.")
    back = side.split()[1]
    return _item(label, f"Space behind their {back}-back in {n} of {len(defence)} {conceded}.",
                 f"Our {OUR_SIDE[side]} winger spins in behind as soon as their {back}-back steps up.")


def _second_balls(defence: list[dict], corners_against: list[dict]) -> dict:
    label = "Second balls on box clearances"
    if not (defence or corners_against):
        return _item(label, "No goals, chances or corners conceded logged yet.")
    open_play = sum(e["fields"]["flaw"] == "Second ball after clearance" for e in defence)
    from_corners = sum(e["fields"]["outcome"] == "Second ball lost" for e in corners_against)
    if not open_play + from_corners:
        return _item(label, "No second balls lost after clearances in the moments logged.")
    return _item(label, f"Lost the second ball after a clearance {_plural(open_play + from_corners, 'time')}: "
                        f"{open_play} in open play, {from_corners} from corners.",
                 "Two midfielders hold the edge of the box on our crosses and corners; shoot first time.")


def _corners_for(corners: list[dict]) -> dict:
    label = "Attacking corners"
    if not corners:
        return _item(label, "No attacking corners logged yet.")
    delivery, n = _top(_count(corners, "delivery"))
    zone, z = _top(_count(corners, "target"))
    player, p = _top(_count(corners, "player"))
    goals = sum(e["fields"]["outcome"] == "Goal" for e in corners)
    shots = sum(e["fields"]["outcome"] == "Shot" for e in corners)
    read = (f"{DELIVERY_PLURAL[delivery]} on {n} of {len(corners)}, aimed at the {zone.lower()} ({z})"
            + (f"; {player_label(player)} attacks it ({p})" if player else "") + f". {_plural(goals, 'goal')}, {_plural(shots, 'shot')}.")
    if delivery == "Short":
        return _item(label, read, "Send a second player out to the short option straight away.")
    return _item(label, read, CORNER_LEVER[zone] + (f" Our best header marks {player_label(player)}." if player else ""))


def _corners_against(corners: list[dict]) -> dict:
    label = "Defending corners"
    if not corners:
        return _item(label, "No corners against them logged yet.")
    marking, n = _top(_count(corners, "marking"))
    goals = sum(e["fields"]["outcome"] == "Goal conceded" for e in corners)
    shots = sum(e["fields"]["outcome"] == "Shot conceded" for e in corners)
    return _item(label, f"{MARKING_PHRASE[marking]} on {n} of {len(corners)} corners; "
                        f"{_plural(goals, 'goal')} and {_plural(shots, 'shot')} conceded.", MARKING_LEVER[marking])


def _free_kicks(kicks: list[dict]) -> dict:
    label = "Direct free kicks"
    if not kicks:
        return _item(label, "No direct free kicks logged yet.")
    taker, n = _top(_count(kicks, "player"))
    foot, _ = _top(_count(kicks, "foot"))
    on_target = sum(e["fields"]["outcome"] in ("Goal", "On target") for e in kicks)
    goals = sum(e["fields"]["outcome"] == "Goal" for e in kicks)
    who = f"{player_label(taker)} took {n}" if taker else f"Unnamed takers took {len(kicks)}"
    return _item(label, f"{who} of {len(kicks)}, {foot.lower()}; {on_target} on target, {goals} scored.",
                 f"Give away no fouls within 30 yards of goal; set the wall for a {foot.split()[0].lower()}-footed taker.")


def build_dossier(opponent: str, events: list[dict]) -> dict:
    of = lambda *kinds: [e for e in events if e.get("kind") in kinds]
    attack, defence = of("goal_scored", "chance_created"), of("goal_conceded", "chance_conceded")
    corners_for, corners_against, kicks = of("corner_for"), of("corner_against"), of("free_kick")
    return {
        "opponent": opponent, "footage": "Highlight Reel", "moments": len(events), "confidence": _confidence(len(events)),
        "note": "Highlight reels show goals and big chances, not the full match.",
        "quick_read": {"threat": _threat(attack, corners_for), "vulnerability": _vulnerability(defence, corners_against)},
        "sections": [
            {"title": "Attacking Patterns", "items": [_creation(attack), _arrivals(attack), _counters(attack)]},
            {"title": "Defensive Flaws", "items": [_isolation(defence), _behind(defence), _second_balls(defence, corners_against)]},
            {"title": "Dead-Ball Intelligence", "items": [_corners_for(corners_for), _corners_against(corners_against), _free_kicks(kicks)]},
        ],
    }


def dossier_text(dossier: dict) -> str:
    """Every visible string in a dossier, for jargon checks."""
    parts = [dossier["opponent"], dossier["footage"], dossier["note"], dossier["confidence"]["text"],
             dossier["quick_read"]["threat"], dossier["quick_read"]["vulnerability"]]
    for section in dossier["sections"]:
        parts.append(section["title"])
        parts += [f"{i['label']} {i['read']} {i['lever']}" for i in section["items"]]
    return " ".join(parts)
