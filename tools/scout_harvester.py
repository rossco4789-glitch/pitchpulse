#!/usr/bin/env python3
"""
scout_harvester.py — PitchPulse Opposition Intelligence Harvester
==================================================================
Standalone, off-pitch pre-match scouting tool. Turns local opponent material
(match reports, lineups, event summaries) into a UEFA 4-Moments dossier.

Usage:
    python tools/scout_harvester.py --opponent "Dorchester Town" --output data/scouting/
    python tools/scout_harvester.py --opponent "Dorchester Town" --source path/to/notes/
    python tools/scout_harvester.py --opponent "Dorchester Town" --mock

Sources (--source DIR, else data/scouting/sources/<slug>/ if it exists):
    *.txt / *.md   narrative match reports
    *.json         {"type": "lineup", "players": [{"shirt": 8, "name": "...", "position": "CM"}]}
                   {"type": "event_summary", "events": [{"minute": 12, "description": "..."}]}
No sources found → built-in MOCK corpus (flagged data_provenance="mock").

Orchestration follows Ruflo concepts (role-scoped agents, declarative task graph,
shared memory namespace) implemented in stdlib Python — no Ruflo runtime, network
or API keys. Deliberately NOT imported by app.py or tagger/index.html.

    T1  Agent Alpha  (Harvester)            request → corpus
    T2  Agent Beta   (Tactical Categoriser) corpus  → draft (rule-based, with confidence)
    T3  Agent Gamma  (Reviewer / Verifier)  draft   → dossier (schema-clean + verdict)

Exit codes:
    0  — dossier written (validation PASS)
    1  — validation FAIL (nothing written)
    2  — input error (bad --source, unreadable JSON)
"""

import argparse
import hashlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Force UTF-8 output so Unicode symbols render on Windows terminals
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

# ── ANSI colours ──────────────────────────────────────────────────────────────
_USE_COLOUR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"{code}{text}\033[0m" if _USE_COLOUR else text

GRN  = "\033[92m"
YLW  = "\033[93m"
RED  = "\033[91m"
CYN  = "\033[96m"
DIM  = "\033[2m"
BOLD = "\033[1m"

# ── Schema vocabulary ─────────────────────────────────────────────────────────
SCHEMA_VERSION = "1.0"
MOMENTS = ("in_possession", "out_of_possession", "transition_attacking",
           "transition_defensive", "set_pieces")

# Mirrors _ZONES in reports/set_piece_matrix.py (duplicated, not imported, so this
# tool stays free of matplotlib/mplsoccer). Parity enforced by the test suite.
CORNER_ZONES = ("Near Post", "Central / Six-Yard", "Penalty Spot / 12-Yd",
                "Back Post", "Edge / Cutback", "Second Ball")

FULL_EVIDENCE = 3  # supporting sentences needed before volume stops discounting confidence

# Negation within ~3 words before a match ("did not press high"). "No.8" is not a negation.
_NEG = re.compile(r"\b(?:not|never|rarely|seldom|no\b(?!\.?\s?\d)|\w+n['’]t)\b(?:\W+\w+){0,2}\W*$", re.I)


def _r(*patterns: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.I)


def _field(moment, field, kind, rules, tie=None, gate=None, exclude=None) -> dict:
    values = tuple(v for v, _ in rules)
    allowed = values + ((tie,) if tie else ()) + ("unknown",) if kind == "enum" else values
    return {"moment": moment, "field": field, "kind": kind, "rules": rules, "tie": tie,
            "gate": gate, "exclude": exclude, "allowed": allowed}


# kind "enum"  → one vote per sentence; first non-negated rule wins (rule order = priority)
# kind "multi" → every matching rule in a sentence scores
FIELDS = (
    # ── In Possession ──
    _field("in_possession", "build_up_pattern", "enum", tie="mixed", rules=[
        ("short_from_gk", _r(r"play(?:s|ed|ing)? out from the back", r"short goal[- ]?kicks?",
                             r"split(?:s|ting)? (?:the )?cent(?:re|er)[- ]backs")),
        ("long_from_gk",  _r(r"long goal[- ]?kicks?", r"\b(?:went|goes|go|going) long\b",
                             r"bypass\w* (?:the )?midfield", r"launch\w* (?:it|the ball)")),
    ]),
    _field("in_possession", "style_preference", "enum", tie="mixed", rules=[
        ("direct",     _r(r"\bdirect\b", r"long balls?", r"target (?:man|striker)",
                          r"into the channels", r"second balls?")),
        ("positional", _r(r"patient (?:build[- ]?up|possession)", r"possession[- ]based",
                          r"circulat\w+ (?:the ball|possession)", r"positional rotations?",
                          r"overload\w* (?:in |the )?(?:midfield|half[- ]spaces?)")),
    ]),
    # ── Out of Possession ──
    _field("out_of_possession", "block_height", "enum", rules=[
        ("high_block", _r(r"high press", r"press(?:ed|es|ing)? high", r"high (?:line|block)",
                          r"press(?:ed|es|ing)? from the front", r"engag\w+ (?:in|inside) our half")),
        ("mid_block",  _r(r"mid[- ]block", r"medium block", r"engag\w+ (?:at|around) (?:the )?halfway")),
        ("low_block",  _r(r"low block", r"deep (?:block|line)", r"sat deep",
                          r"defend\w* (?:the|their) (?:own )?box")),
    ]),
    # Flanks are the OPPONENT'S own left/right.
    _field("out_of_possession", "flank_vulnerability", "enum", tie="both", rules=[
        ("left",  _r(r"(?:space|gaps?) (?:behind|down|on) (?:their|the) left",
                     r"left[- ]back (?:was |were )?(?:exposed|caught|pulled|isolated)",
                     r"left (?:side|flank) (?:was|looked) (?:vulnerable|exposed|open)")),
        ("right", _r(r"(?:space|gaps?) (?:behind|down|on) (?:their|the) right",
                     r"right[- ]back (?:was |were )?(?:exposed|caught|pulled|isolated)",
                     r"right (?:side|flank) (?:was|looked) (?:vulnerable|exposed|open)")),
    ]),
    # ── Attacking Transition ──
    _field("transition_attacking", "counter_attack_speed", "enum", rules=[
        ("slow",     _r(r"slow (?:to break|counter\w*|transitions?)", r"(?:rarely|seldom) (?:broke|countered)")),
        ("measured", _r(r"measured (?:counter\w*|transitions?)", r"(?:reset|recycl\w+) (?:possession )?after winning",
                        r"slow\w* (?:the game|it|play) down")),
        ("fast",     _r(r"(?:rapid|quick|fast|lightning) (?:counter\w*|breaks?|transitions?)",
                        r"broke (?:quickly|at pace|with pace)", r"counter\w* at pace")),
    ]),
    _field("transition_attacking", "outlet_channels", "multi",
           gate=_r(r"counter\w*", r"\bbr(?:eak|oke)\b", r"transitions?", r"turnover",
                   r"won (?:the ball|possession)", r"outlet", r"releas\w+"),
           rules=[
        ("left_flank",       _r(r"left (?:flank|wing|channel)")),
        ("left_half_space",  _r(r"left half[- ]spaces?")),
        ("central",          _r(r"through the middle", r"down the middle", r"\bcentral(?:ly)?\b")),
        ("right_half_space", _r(r"right half[- ]spaces?")),
        ("right_flank",      _r(r"right (?:flank|wing|channel)")),
    ]),
    # ── Defensive Transition ──
    _field("transition_defensive", "counter_press_intensity", "enum", rules=[
        ("low",    _r(r"(?:did not|didn't|never|no) counter[- ]?press\w*",
                      r"(?:dropped|retreated) (?:back )?(?:into shape|behind the ball) (?:immediately|quickly|after losing)")),
        ("medium", _r(r"(?:selective|occasional)(?:ly)? counter[- ]?press\w*")),
        ("high",   _r(r"counter[- ]?press\w*", r"gegenpress\w*", r"swarm\w* (?:around )?the ball",
                      r"win (?:it|the ball) back (?:quickly|immediately)")),
    ]),
    _field("transition_defensive", "rest_defence_shape", "enum",
           gate=_r(r"rest[- ]defen[cs]e", r"stay\w* back", r"held back", r"left behind"),
           rules=[
        ("3+2", _r(r"\b3\s*\+\s*2\b")),
        ("2+2", _r(r"\b2\s*\+\s*2\b")),
        ("3+1", _r(r"\b3\s*\+\s*1\b", r"back three (?:and|plus) (?:a|one) (?:holder|pivot|six)")),
        ("2+1", _r(r"\b2\s*\+\s*1\b", r"two cent(?:re|er)[- ]backs (?:and|plus) (?:a|one) (?:holder|pivot|six|holding midfielder)")),
    ]),
    # ── Set Pieces ──
    _field("set_pieces", "corner_delivery_zones", "multi",
           gate=_r(r"corners?", r"set[- ]pieces?", r"in-?swing\w*", r"out-?swing\w*", r"deliver(?:y|ies)"),
           exclude=_r(r"\bmark(?:ed|ing|ers?)\b", r"\bzonal\b", r"man[- ]to[- ]man"),
           rules=[
        ("Near Post",            _r(r"near post")),
        ("Central / Six-Yard",   _r(r"six[- ]yard", r"6[- ]yard", r"goalkeeper'?s? area")),
        ("Penalty Spot / 12-Yd", _r(r"penalty spot", r"12[- ]yard", r"twelve yards?")),
        ("Back Post",            _r(r"back post", r"far post")),
        ("Edge / Cutback",       _r(r"edge of the (?:box|area)", r"cut[- ]?backs?", r"short corners?")),
        ("Second Ball",          _r(r"second balls?")),
    ]),
    _field("set_pieces", "defensive_marking", "enum",
           gate=_r(r"corners?", r"set[- ]pieces?", r"crosses"),
           rules=[
        ("hybrid",     _r(r"hybrid", r"mixed marking", r"zonal (?:with|plus|and) (?:a few |some |two |three )?man[- ]?markers",
                          r"man[- ]to[- ]man\b.*\bzonal", r"zonal\b.*\bman[- ]to[- ]man")),
        ("man_to_man", _r(r"man[- ]to[- ]man", r"man[- ]mark\w*")),
        ("zonal",      _r(r"\bzonal\b")),
    ]),
)

SCHEMA: dict[str, dict[str, tuple]] = {m: {} for m in MOMENTS}
for _spec in FIELDS:
    SCHEMA[_spec["moment"]][_spec["field"]] = (_spec["kind"], _spec["allowed"])
SCHEMA["in_possession"]["key_ball_progressors"] = ("players", ())

_PROGRESSION = _r(r"progress\w*", r"carr(?:y|ies|ied|ying)", r"drove forward", r"line[- ]breaking",
                  r"switch\w* (?:the )?play", r"dictat\w+ (?:the )?tempo", r"between the lines")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _band(conf: float) -> str:
    return "high" if conf >= 0.7 else "medium" if conf >= 0.4 else "low" if conf > 0 else "none"


def _confidence(total: int, share: float) -> float:
    return round(share * min(1.0, total / FULL_EVIDENCE), 2)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _match(rx: re.Pattern, sentence: str) -> bool:
    return any(not _NEG.search(sentence[max(0, m.start() - 40):m.start()]) for m in rx.finditer(sentence))


# ══════════════════════════════════════════════════════════════════════════════
# Agent Alpha — Harvester
# ══════════════════════════════════════════════════════════════════════════════

def _mock_documents(opponent: str) -> list[dict]:
    """Offline fallback corpus. Fictional, squad-number labels only — never real intel."""
    o = opponent
    return [
        {"id": "mock_report_01", "type": "match_report", "title": f"{o} — recent league match (MOCK)", "text": (
            f"{o} tried to play out from the back with short goal kicks. "
            "When pressed they went long towards a target man. "
            "Their No.8 carried the ball through midfield and was the main line-breaking passer. "
            "Out of possession they pressed high and engaged inside our half. "
            "Their left-back was caught high, leaving space behind the left side. "
            "After winning the ball they broke quickly down the right flank. "
            "Their rest defence was a 2+1 with the No.6 holding. "
            "They counter-pressed immediately after losing the ball. "
            "Corners were in-swinging to the near post. "
            "They defended corners zonal with two man markers.")},
        {"id": "mock_report_02", "type": "match_report", "title": f"{o} — recent cup match (MOCK)", "text": (
            f"{o} were direct, playing long balls into the channels and chasing second balls. "
            "The No.8 progressed play again with line-breaking passes. "
            "They sat in a mid-block for long spells in the second half. "
            "Space behind their left-back was exposed repeatedly. "
            "On the counter they went through the left half-space with rapid transitions. "
            "Their rest defence stayed as a 2+1 when the full-backs advanced. "
            "After losing the ball they counter-pressed aggressively. "
            "Corners were aimed at the six-yard box and the near post. "
            "At defensive corners they marked man-to-man on the edge but zonal in the six-yard box.")},
        {"id": "mock_events_03", "type": "event_summary", "title": f"{o} — public event summary (MOCK)", "text": "\n".join([
            "12' Quick counter down the right flank after a turnover, No.11 cross cleared.",
            "34' Pressed high from a goal kick, forced turnover.",
            "58' Corner to the near post, headed wide.",
            "71' No.4 carried the ball out of defence and played a line-breaking pass.",
            "80' They did not press high after going ahead, dropping into a low block.",
        ])},
        {"id": "mock_lineup_04", "type": "lineup", "title": f"{o} — recent Southern League XI (MOCK)", "players": [
            {"shirt": n, "name": f"No.{n}", "position": pos} for n, pos in
            [(1, "GK"), (2, "RB"), (3, "LB"), (4, "CB"), (5, "CB"), (6, "DM"),
             (7, "RW"), (8, "CM"), (9, "ST"), (10, "AM"), (11, "LW")]
        ]},
    ]


def load_source_dir(src: Path) -> list[dict]:
    docs = []
    for p in sorted(src.iterdir(), key=lambda p: p.name):
        suffix = p.suffix.lower()
        if suffix in (".txt", ".md"):
            docs.append({"id": p.stem, "type": "match_report", "title": p.name,
                         "text": p.read_text(encoding="utf-8")})
        elif suffix == ".json":
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p.name}: invalid JSON ({exc})") from exc
            kind = raw.get("type") if isinstance(raw, dict) else None
            if kind == "lineup":
                docs.append({"id": p.stem, "type": "lineup", "title": raw.get("title", p.name),
                             "players": raw.get("players", [])})
            elif kind == "event_summary":
                text = "\n".join(f"{e.get('minute', '?')}' {e.get('description', '')}" for e in raw.get("events", []))
                docs.append({"id": p.stem, "type": "event_summary", "title": raw.get("title", p.name), "text": text})
            else:
                raise ValueError(f"{p.name}: unknown source type {kind!r} (expected 'lineup' or 'event_summary')")
    if not docs:
        raise ValueError(f"no .txt/.md/.json sources found in {src}")
    return docs


def agent_alpha(memory: dict) -> dict:
    req = memory["request"]
    if req.get("force_mock") or req.get("source_dir") is None:
        raw, provenance = _mock_documents(req["opponent"]), "mock"
    else:
        raw, provenance = load_source_dir(Path(req["source_dir"])), "local_sources"

    documents, roster = [], {}
    for d in raw:
        documents.append({"id": d["id"], "type": d["type"], "title": d["title"],
                          "sentences": _sentences(d.get("text", ""))})
        for pl in d.get("players", []):
            name = " ".join(str(pl.get("name", "")).split())
            if not name:
                continue
            shirt = pl.get("shirt")
            shirt = int(shirt) if str(shirt).isdigit() else None
            roster[name] = {"player": name, "shirt": shirt, "position": (str(pl.get("position") or "").strip() or None)}

    corpus = {"provenance": provenance, "documents": documents, "roster": [roster[k] for k in sorted(roster)]}
    corpus["fingerprint"] = hashlib.sha256(json.dumps(corpus, sort_keys=True).encode("utf-8")).hexdigest()
    return corpus


# ══════════════════════════════════════════════════════════════════════════════
# Agent Beta — Tactical Categoriser
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate(spec: dict, votes: Counter, sources: set) -> dict:
    order = [v for v, _ in spec["rules"]]
    total = sum(votes.values())
    distribution = {v: votes[v] for v in order if votes[v]}
    ranked = sorted(distribution, key=lambda v: (-votes[v], order.index(v)))

    if spec["kind"] == "multi":
        value, conf = ranked, (_confidence(total, 1.0) if total else 0.0)
    elif not total:
        value, conf = "unknown", 0.0
    else:
        tied = len(ranked) > 1 and votes[ranked[1]] == votes[ranked[0]]
        value = (spec["tie"] or "unknown") if tied else ranked[0]
        conf = 0.0 if value == "unknown" else _confidence(total, votes[ranked[0]] / total)

    return {"value": value, "confidence": conf, "confidence_band": _band(conf),
            "evidence_count": total, "distribution": distribution, "sources": sorted(sources)}


def _progressors(corpus: dict) -> dict:
    roster = corpus["roster"]
    patterns = {}
    for p in roster:
        alts = [re.escape(p["player"])]
        if p["shirt"] is not None:
            alts.append(rf"(?:no\.?\s?|number\s|#){p['shirt']}")
        patterns[p["player"]] = re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)", re.I)

    mentions, sources = Counter(), set()
    for doc in corpus["documents"]:
        for s in doc["sentences"]:
            if not _match(_PROGRESSION, s):
                continue
            for name, rx in patterns.items():
                if rx.search(s):
                    mentions[name] += 1
                    sources.add(doc["id"])

    by_name = {p["player"]: p for p in roster}
    value = [{**by_name[n], "mentions": mentions[n]}
             for n in sorted(mentions, key=lambda n: (-mentions[n], n))[:3]]
    total = sum(mentions.values())
    conf = _confidence(total, 1.0) if total else 0.0
    return {"value": value, "confidence": conf, "confidence_band": _band(conf),
            "evidence_count": total, "sources": sorted(sources)}


def agent_beta(memory: dict) -> dict:
    corpus = memory["corpus"]
    draft = {m: {} for m in MOMENTS}
    for spec in FIELDS:
        votes, sources = Counter(), set()
        for doc in corpus["documents"]:
            for s in doc["sentences"]:
                if spec["gate"] and not spec["gate"].search(s):
                    continue
                if spec["exclude"] and spec["exclude"].search(s):
                    continue
                for value, rx in spec["rules"]:
                    if _match(rx, s):
                        votes[value] += 1
                        sources.add(doc["id"])
                        if spec["kind"] == "enum":
                            break
        draft[spec["moment"]][spec["field"]] = _aggregate(spec, votes, sources)
    draft["in_possession"]["key_ball_progressors"] = _progressors(corpus)
    return draft


# ══════════════════════════════════════════════════════════════════════════════
# Agent Gamma — Reviewer / Verifier
# ══════════════════════════════════════════════════════════════════════════════

_ITEM_KEYS = {"value", "confidence", "confidence_band", "evidence_count", "distribution", "sources"}
_FLUFF = re.compile(r"^(?:um+|uh+|well|basically|so|like|i think|honestly)[\s,:-]+", re.I)


def _clean_text(s, limit: int = 40) -> str:
    if not isinstance(s, str):
        return ""
    return _FLUFF.sub("", " ".join(s.split()))[:limit].strip()


def _is_int(n) -> bool:
    return isinstance(n, int) and not isinstance(n, bool)


def _check_field(path: str, item, kind: str, allowed: tuple, known_sources) -> tuple:
    if not isinstance(item, dict):
        return None, [f"{path}: missing or not an object"], []
    errors, warnings = [], []
    for k in sorted(set(item) - _ITEM_KEYS):
        warnings.append(f"stripped unknown key '{path}.{k}'")

    conf = item.get("confidence")
    if not (_is_int(conf) or isinstance(conf, float)) or not 0 <= conf <= 1:
        errors.append(f"{path}: confidence {conf!r} outside 0–1")
        conf = 0.0
    count = item.get("evidence_count")
    if not _is_int(count) or count < 0:
        errors.append(f"{path}: evidence_count {count!r} must be a non-negative int")
        count = 0

    value = item.get("value")
    if kind == "enum":
        if not isinstance(value, str) or value not in allowed:
            errors.append(f"{path}: invalid value {value!r}; allowed {list(allowed)}")
        elif value == "unknown" and conf:
            errors.append(f"{path}: 'unknown' must carry 0.00 confidence")
    elif kind == "multi":
        if not isinstance(value, list):
            errors.append(f"{path}: value must be a list")
        else:
            bad = [v for v in value if v not in allowed]
            if bad:
                errors.append(f"{path}: invalid zone/channel {bad}; allowed {list(allowed)}")
            if len(set(map(str, value))) != len(value):
                errors.append(f"{path}: duplicate entries")
    else:  # players
        if not isinstance(value, list):
            errors.append(f"{path}: value must be a list")
        else:
            cleaned = []
            for p in value:
                if not isinstance(p, dict):
                    errors.append(f"{path}: player entry {p!r} is not an object")
                    continue
                name, shirt, mentions = _clean_text(p.get("player")), p.get("shirt"), p.get("mentions")
                if not name:
                    errors.append(f"{path}: player entry has no name")
                if shirt is not None and (not _is_int(shirt) or not 1 <= shirt <= 99):
                    errors.append(f"{path}: {name or '?'} shirt {shirt!r} outside 1–99")
                if not _is_int(mentions) or mentions < 1:
                    errors.append(f"{path}: {name or '?'} mentions {mentions!r} must be ≥1")
                cleaned.append({"player": name, "shirt": shirt,
                                "position": _clean_text(p.get("position")) or None, "mentions": mentions})
            value = cleaned

    out = {"value": value, "confidence": round(conf, 2), "confidence_band": _band(conf), "evidence_count": count}
    if kind != "players":
        dist = item.get("distribution", {})
        if (not isinstance(dist, dict) or any(k not in allowed for k in dist)
                or any(not _is_int(n) or n < 1 for n in dist.values())):
            errors.append(f"{path}: distribution {dist!r} invalid")
        elif sum(dist.values()) != count:
            errors.append(f"{path}: distribution total {sum(dist.values())} ≠ evidence_count {count}")
        out["distribution"] = dist
    sources = item.get("sources", [])
    if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
        errors.append(f"{path}: sources must be a list of ids")
    elif known_sources is not None and set(sources) - known_sources:
        errors.append(f"{path}: unknown source ids {sorted(set(sources) - known_sources)}")
    out["sources"] = sources

    if not errors and 0 < conf < 0.4:
        warnings.append(f"{path}: low confidence ({conf:.2f}) — corroborate before briefing")
    return (None if errors else out), errors, warnings


def validate_dossier(draft: dict, known_sources: set | None = None) -> tuple[dict, list[str], list[str]]:
    """Audit a draft against SCHEMA. Returns (clean body, errors, warnings); unknown keys are stripped."""
    body, errors, warnings = {}, [], []
    for key in draft:
        if key not in SCHEMA:
            warnings.append(f"stripped unknown section '{key}'")
    for moment, fields in SCHEMA.items():
        section = draft.get(moment)
        if not isinstance(section, dict):
            errors.append(f"{moment}: section missing")
            continue
        for key in section:
            if key not in fields:
                warnings.append(f"stripped unknown field '{moment}.{key}'")
        body[moment] = {}
        for field, (kind, allowed) in fields.items():
            item, errs, warns = _check_field(f"{moment}.{field}", section.get(field), kind, allowed, known_sources)
            errors += errs
            warnings += warns
            if item is not None:
                body[moment][field] = item
    return body, errors, warnings


def agent_gamma(memory: dict) -> dict:
    req, corpus = memory["request"], memory["corpus"]
    body, errors, warnings = validate_dossier(memory["draft"], {d["id"] for d in corpus["documents"]})
    if not req["opponent"].strip() or req["slug"] != slugify(req["opponent"]):
        errors.insert(0, f"opponent/slug mismatch: {req['opponent']!r} vs {req['slug']!r}")
    if corpus["provenance"] == "mock":
        warnings.insert(0, "MOCK corpus — illustrative only, not real opposition intel")
    return {
        "schema_version": SCHEMA_VERSION,
        "opponent": req["opponent"],
        "opponent_slug": req["slug"],
        "data_provenance": corpus["provenance"],
        "source_fingerprint": corpus["fingerprint"],
        "conventions": {
            "flanks": "opponent's own left/right",
            "corner_zones": "reports/set_piece_matrix.py delivery zones",
            "confidence": f"winning share × min(1, evidence/{FULL_EVIDENCE}); bands high≥0.7, medium≥0.4",
        },
        "sources": [{"id": d["id"], "type": d["type"], "title": d["title"]} for d in corpus["documents"]],
        **body,
        "validation": {"passed": not errors, "errors": errors, "warnings": warnings},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator — Ruflo-style declarative task graph over shared memory
# ══════════════════════════════════════════════════════════════════════════════

TASK_GRAPH = (
    {"id": "T1_harvest",    "agent": "alpha", "role": "Harvester",            "depends_on": (),                 "writes": "corpus"},
    {"id": "T2_categorise", "agent": "beta",  "role": "Tactical Categoriser", "depends_on": ("T1_harvest",),    "writes": "draft"},
    {"id": "T3_verify",     "agent": "gamma", "role": "Reviewer / Verifier",  "depends_on": ("T2_categorise",), "writes": "dossier"},
)
AGENTS = {"alpha": agent_alpha, "beta": agent_beta, "gamma": agent_gamma}


def run_pipeline(request: dict, graph=TASK_GRAPH) -> dict:
    memory, done, pending = {"request": request}, [], list(graph)
    while pending:
        ready = [t for t in pending if all(d in done for d in t["depends_on"])]
        if not ready:
            raise RuntimeError(f"task graph has unmet or cyclic dependencies: {[t['id'] for t in pending]}")
        for task in ready:
            memory[task["writes"]] = AGENTS[task["agent"]](memory)
            done.append(task["id"])
            pending.remove(task)
    memory["trace"] = done
    return memory


def write_dossier(dossier: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dossier['opponent_slug']}_dossier.json"
    path.write_text(json.dumps(dossier, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path


# ── Terminal summary ──────────────────────────────────────────────────────────
_LABEL = {"in_possession": "IN POSSESSION", "out_of_possession": "OUT OF POSSESSION",
          "transition_attacking": "ATTACKING TRANSITION", "transition_defensive": "DEFENSIVE TRANSITION",
          "set_pieces": "SET PIECES"}


def _fmt_value(kind: str, value) -> str:
    if kind == "players":
        return ", ".join(f"{p['player']} ×{p['mentions']}" for p in value) or "—"
    if kind == "multi":
        return ", ".join(value) or "—"
    return str(value)


def print_summary(memory: dict, path: Path | None) -> None:
    d, corpus = memory["dossier"], memory["corpus"]
    rule = "─" * 72
    print(_c(BOLD, "PitchPulse · Opposition Intelligence Harvester"))
    print(rule)
    prov = _c(YLW, "MOCK (offline fallback)") if d["data_provenance"] == "mock" else _c(GRN, "local sources")
    n_sent = sum(len(doc["sentences"]) for doc in corpus["documents"])
    print(f"Opponent    {d['opponent']}  [{d['opponent_slug']}]")
    print(f"Provenance  {prov}")
    print(f"Corpus      {len(corpus['documents'])} docs · {n_sent} sentences · {len(corpus['roster'])} players")
    print("Tasks       " + "  ".join(_c(GRN, "✓ ") + t for t in memory["trace"]))

    for moment, fields in SCHEMA.items():
        print()
        print(_c(CYN, _LABEL[moment]))
        for field, (kind, _) in fields.items():
            item = d.get(moment, {}).get(field)
            if item is None:
                print(f"  {field:<26}{_c(RED, 'invalid — see errors')}")
                continue
            band = item["confidence_band"]
            col = {"high": GRN, "medium": YLW, "low": RED}.get(band, DIM)
            conf = f"{band:<6} {item['confidence']:.2f}"
            print(f"  {field:<26}{_fmt_value(kind, item['value']):<34}{_c(col, conf)}  n={item['evidence_count']}")

    v = d["validation"]
    print()
    print(rule)
    verdict = _c(GRN, "PASS") if v["passed"] else _c(RED, "FAIL")
    print(f"VALIDATION  {verdict}  ({len(v['errors'])} errors, {len(v['warnings'])} warnings)")
    for e in v["errors"]:
        print("  " + _c(RED, "✗ ") + e)
    for w in v["warnings"]:
        print("  " + _c(YLW, "! ") + w)
    print(f"Dossier     {path}" if path else _c(RED, "Dossier     not written — fix errors above"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a UEFA 4-Moments opposition dossier from local sources.")
    ap.add_argument("--opponent", required=True, help='Opponent name, e.g. "Dorchester Town"')
    ap.add_argument("--output", type=Path, default=Path("data/scouting"), help="Output directory")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--source", type=Path, help="Directory of .txt/.md reports and .json lineups/event summaries")
    src.add_argument("--mock", action="store_true", help="Force the built-in offline mock corpus")
    args = ap.parse_args(argv)

    opponent = " ".join(args.opponent.split())
    slug = slugify(opponent)
    if not slug:
        print(_c(RED, "✗ --opponent must contain letters or digits"))
        return 2

    source_dir = None
    if args.source is not None:
        if not args.source.is_dir():
            print(_c(RED, f"✗ --source {args.source} is not a directory"))
            return 2
        source_dir = args.source
    elif not args.mock:
        default = ROOT / "data" / "scouting" / "sources" / slug
        source_dir = default if default.is_dir() else None

    try:
        memory = run_pipeline({"opponent": opponent, "slug": slug,
                               "source_dir": source_dir, "force_mock": args.mock})
    except (ValueError, OSError) as exc:
        print(_c(RED, f"✗ {exc}"))
        return 2

    dossier = memory["dossier"]
    path = write_dossier(dossier, args.output) if dossier["validation"]["passed"] else None
    print_summary(memory, path)
    return 0 if path else 1


if __name__ == "__main__":
    sys.exit(main())
