"""
cv/club_assets.py — opponent club crest and kit colours for the Opposition Analysis tab (app.py tab 6).

resolve_club_assets() reads data/scouting/sources/<slug>/meta.json first and makes no network call when it exists.
On a cache miss it looks the club up on Wikipedia (free Wikimedia REST API) and saves the crest thumbnail to
badge.png (gitignored: club crests are usually non-free logos). Kit colours come from the football infobox
(body1 / body2, "club records") when present. Otherwise they come from the crest's dominant colours via Pillow,
which is only a guess: Dorchester Town's crest is mostly sky blue, while they play in black and white stripes.

Offline or unknown clubs fall back to DEFAULT_HOME / DEFAULT_AWAY with no badge and never raise. A club Wikipedia
does not know is cached; a network failure is not, so the next session retries.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT         = Path(__file__).resolve().parents[1]
SOURCES      = ROOT / "data" / "scouting" / "sources"
META_FILE    = "meta.json"
BADGE_FILE   = "badge.png"
DEFAULT_HOME = "#CC2222"
DEFAULT_AWAY = "#FFFFFF"

SEARCH_URL   = "https://en.wikipedia.org/w/rest.php/v1/search/page?"
SUMMARY_URL  = "https://en.wikipedia.org/api/rest_v1/page/summary/"
PAGE_URL     = "https://en.wikipedia.org/w/rest.php/v1/page/"          # wikitext, for the infobox kit colours
USER_AGENT   = "PitchPulse/1.0 (Tiverton Town FC opposition scouting; local desktop tool)"
TIMEOUT_S    = 4
MAX_IMAGE_B  = 2_000_000

COLOUR_BIN      = 32     # 8 levels per channel
NEAR_WHITE      = 235
NEAR_BLACK      = 20
MIN_ALPHA       = 128
AWAY_MIN_SHARE  = 0.05   # a second colour must cover 5 % of the crest's coloured pixels
AWAY_MIN_DIST   = 80.0   # and sit this far (RGB) from the home colour
HEX_RE          = re.compile(r"^#[0-9A-F]{6}$")
NETWORK_ERRORS  = (URLError, TimeoutError, OSError, ValueError)   # ValueError covers bad JSON


def slugify(name: str) -> str:
    """Same folder name as tools/cloud_vision_runner.slugify."""
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")


# ── HTTP (replaced in tests) ─────────────────────────────────────────────────

def _http_json(url: str) -> dict:
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}), timeout=TIMEOUT_S) as r:
        return json.loads(r.read().decode("utf-8"))


def _http_bytes(url: str) -> bytes:
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=TIMEOUT_S) as r:
        data = r.read(MAX_IMAGE_B + 1)
    if len(data) > MAX_IMAGE_B:
        raise ValueError("crest image larger than 2 MB")
    return data


# ── Wikipedia lookup ─────────────────────────────────────────────────────────

def _is_club_page(page: dict, name: str) -> bool:
    title, description = (page.get("title") or ""), (page.get("description") or "").lower()
    club = "football club" in description or title.endswith(("F.C.", "A.F.C.", " FC"))
    words = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) >= 3 and w not in ("afc", "football", "club")]
    named = all(w in title.lower() for w in words) if words else False
    women = "women" in title.lower() and "women" not in name.lower()
    return club and named and not women


def parse_infobox_kits(wikitext: str) -> tuple[str | None, str | None]:
    """Home and away shirt colours (body1 / body2) from Wikipedia's football club infobox."""
    found = [re.search(rf"\|\s*body{n}\s*=\s*#?([0-9A-Fa-f]{{6}})\b", wikitext or "") for n in (1, 2)]
    return tuple(f"#{m.group(1).upper()}" if m else None for m in found)


def find_club_page(name: str) -> dict | None:
    """{'title', 'thumbnail', 'kits'} for the club's Wikipedia page, or None. Network errors propagate."""
    for query in (f"{name} F.C.", name):
        for page in _http_json(SEARCH_URL + urlencode({"q": query, "limit": 5})).get("pages", []):
            if _is_club_page(page, name):
                key = quote(page.get("key") or page["title"].replace(" ", "_"), safe="")
                summary = _http_json(SUMMARY_URL + key)
                return {"title": summary.get("title") or page["title"],
                        "thumbnail": (summary.get("thumbnail") or {}).get("source"),
                        "kits": parse_infobox_kits(_http_json(PAGE_URL + key).get("source", ""))}
    return None


# ── Colours ──────────────────────────────────────────────────────────────────

def _hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(round(c)) for c in rgb))


def kit_colours(image: Image.Image) -> tuple[str, str]:
    """(home, away) from the crest's dominant colours, ignoring transparency, near-white and near-black."""
    rgba = image.convert("RGBA")
    rgba.thumbnail((96, 96))
    px = np.asarray(rgba, dtype=np.int32).reshape(-1, 4)
    rgb = px[px[:, 3] >= MIN_ALPHA, :3]
    rgb = rgb[~((rgb >= NEAR_WHITE).all(axis=1) | (rgb <= NEAR_BLACK).all(axis=1))]
    if not len(rgb):
        return DEFAULT_HOME, DEFAULT_AWAY
    bins = defaultdict(list)
    for key, colour in zip(map(tuple, rgb // COLOUR_BIN), rgb):
        bins[key].append(colour)
    ranked = sorted(bins.values(), key=len, reverse=True)
    home = np.mean(ranked[0], axis=0)
    away = next((np.mean(group, axis=0) for group in ranked[1:]
                 if len(group) >= AWAY_MIN_SHARE * len(rgb) and np.linalg.norm(np.mean(group, axis=0) - home) >= AWAY_MIN_DIST), None)
    return _hex(home), _hex(away) if away is not None else DEFAULT_AWAY


# ── Resolver ─────────────────────────────────────────────────────────────────

def fallback(name: str, slug: str) -> dict:
    return {"name": name, "slug": slug, "badge_path": None, "home_kit": DEFAULT_HOME, "away_kit": DEFAULT_AWAY,
            "verified": False, "kit_source": "default"}


def _stored_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _read_meta(path: Path) -> dict | None:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or not all(HEX_RE.match(str(meta.get(k, ""))) for k in ("home_kit", "away_kit")):
        return None
    badge = meta.get("badge_path")
    if badge:
        badge_file = Path(badge) if Path(badge).is_absolute() else ROOT / badge
        meta["badge_path"] = str(badge_file) if badge_file.exists() else None
    return {**fallback(str(meta.get("name", "")), str(meta.get("slug", ""))), **meta}


def _write_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = {**meta, "badge_path": _stored_path(Path(meta["badge_path"])) if meta["badge_path"] else None}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def resolve_club_assets(opponent_name: str, sources: str | Path | None = None, refresh: bool = False) -> dict:
    """Crest and kit colours for an opponent; cached per club in meta.json. Never raises."""
    name = (opponent_name or "").strip()
    slug = slugify(name)
    if not slug:
        return fallback(name, slug)
    folder = Path(sources or SOURCES) / slug
    meta_path = folder / META_FILE
    if not refresh and (cached := _read_meta(meta_path)):
        return cached

    try:
        page = find_club_page(name)
    except NETWORK_ERRORS:
        return fallback(name, slug)           # offline: nothing cached, retry next time
    if page is None:
        meta = fallback(name, slug)
        _write_meta(meta_path, meta)          # Wikipedia has no such club: don't ask again
        return meta

    home, away = page.get("kits") or (None, None)
    meta = {**fallback(page["title"], slug), "verified": True, "home_kit": home or DEFAULT_HOME,
            "away_kit": away or DEFAULT_AWAY, "kit_source": "club records" if home else "default"}
    if page.get("thumbnail"):
        try:
            data = _http_bytes(page["thumbnail"])
        except NETWORK_ERRORS:
            return meta                       # crest download failed: show the club, retry the crest next time
        try:
            crest = Image.open(BytesIO(data))
            crest.load()
        except (UnidentifiedImageError, OSError, ValueError):   # not an image: cache the club without a crest
            crest = None
        if crest is not None:
            badge = folder / BADGE_FILE
            badge.parent.mkdir(parents=True, exist_ok=True)
            crest.convert("RGBA").save(badge, "PNG")
            meta["badge_path"] = str(badge)
            if not home:                      # no recorded shirt colours: the crest is the best guess left
                crest_home, crest_away = kit_colours(crest)
                meta.update(home_kit=crest_home, away_kit=away or crest_away, kit_source="crest")
    _write_meta(meta_path, meta)
    return meta
