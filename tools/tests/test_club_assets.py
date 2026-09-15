"""
tools/tests/test_club_assets.py
Opponent crest and kit colour resolver (cv/club_assets.py) plus its Match Setup controls. Offline: every
Wikipedia call goes through club_assets._http_json / _http_bytes, which these tests replace.

Run: python -m pytest tools/tests/test_club_assets.py -v
"""

import json
import re
import sys
from io import BytesIO
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cv import club_assets as ca  # noqa: E402
from cv import vision_center as vc  # noqa: E402

CLUB = "Dorchester Town"
SEARCH = {"pages": [
    {"key": "Dorchester_Town_F.C._Women", "title": "Dorchester Town F.C. Women", "description": "Women's football club"},
    {"key": "Dorchester_Town_railway_station", "title": "Dorchester Town railway station", "description": "Railway station in Dorset"},
    {"key": "Dorchester_Town_F.C.", "title": "Dorchester Town F.C.", "description": "Association football club in England"},
]}
SUMMARY = {"title": "Dorchester Town F.C.", "thumbnail": {"source": "https://upload.wikimedia.org/crest.png"}}
# Live infobox excerpt, 15 Sep 2026: black shirts with white stripes at home, blue away
WIKITEXT = ("{{Infobox football club\n| clubname = Dorchester Town\n| pattern_b1 = _whitestripes\n| body1 = 000000\n"
            "| shorts1 = 000000\n| pattern_b2 = _bluequarters23\n| body2 = 0000EE\n| shorts2 = 0000EE\n}}")


def _crest(main=(204, 34, 34), second=(26, 26, 36), size=60) -> bytes:
    """Transparent corners, a white and a black border, then 60 % main colour and 25 % second colour."""
    px = np.zeros((size, size, 4), np.uint8)
    px[..., 3] = 255
    px[:, :] = (*main, 255)
    px[:, : int(size * 0.25)] = (*second, 255)
    px[:4], px[-4:] = (255, 255, 255, 255), (0, 0, 0, 255)
    px[:8, :8, 3] = 0
    out = BytesIO()
    Image.fromarray(px, "RGBA").save(out, "PNG")
    return out.getvalue()


@pytest.fixture
def online(monkeypatch):
    calls = []

    def fake_json(url):
        calls.append(url)
        return SUMMARY if "page/summary" in url else SEARCH

    def fake_bytes(url):
        calls.append(url)
        return _crest()

    monkeypatch.setattr(ca, "_http_json", fake_json)
    monkeypatch.setattr(ca, "_http_bytes", fake_bytes)
    return calls


def _no_network(monkeypatch):
    def refuse(url):
        raise AssertionError(f"unexpected network call {url}")

    monkeypatch.setattr(ca, "_http_json", refuse)
    monkeypatch.setattr(ca, "_http_bytes", refuse)


def _offline(monkeypatch):
    def down(url):
        raise URLError("no route to host")

    monkeypatch.setattr(ca, "_http_json", down)
    monkeypatch.setattr(ca, "_http_bytes", down)


# ── Colours ──────────────────────────────────────────────────────────────────

def test_dominant_colours_ignore_transparency_white_and_black():
    assert ca.kit_colours(Image.open(BytesIO(_crest()))) == ("#CC2222", "#1A1A24")


def test_single_colour_crest_keeps_the_default_away_kit_and_blank_crest_falls_back():
    assert ca.kit_colours(Image.open(BytesIO(_crest(second=(204, 34, 34))))) == ("#CC2222", ca.DEFAULT_AWAY)
    white = Image.new("RGBA", (40, 40), (255, 255, 255, 255))
    assert ca.kit_colours(white) == (ca.DEFAULT_HOME, ca.DEFAULT_AWAY)
    mostly_white = np.full((50, 50, 3), 255, np.uint8)
    mostly_white[20:30, 20:30] = (0, 102, 51)     # green stripe on a white crest
    assert ca.kit_colours(Image.fromarray(mostly_white, "RGB"))[0] == "#006633"


# ── Resolver ─────────────────────────────────────────────────────────────────

def test_infobox_kit_colours_parse_and_beat_the_crest(tmp_path, monkeypatch):
    assert ca.parse_infobox_kits(WIKITEXT) == ("#000000", "#0000EE")
    assert ca.parse_infobox_kits("| body1 = #ffff00\n| shorts1 = 000000") == ("#FFFF00", None)
    assert ca.parse_infobox_kits("no infobox here") == (None, None)

    monkeypatch.setattr(ca, "_http_json", lambda url: {"source": WIKITEXT} if "/v1/page/" in url
                        else SUMMARY if "summary" in url else SEARCH)
    monkeypatch.setattr(ca, "_http_bytes", lambda url: _crest())        # a red crest must not override club records
    meta = ca.resolve_club_assets(CLUB, sources=tmp_path)
    assert (meta["home_kit"], meta["away_kit"], meta["kit_source"]) == ("#000000", "#0000EE", "club records")
    assert meta["badge_path"] is not None


def test_cache_miss_fetches_once_then_reads_meta_without_network(tmp_path, online, monkeypatch):
    meta = ca.resolve_club_assets(CLUB, sources=tmp_path)
    assert (meta["name"], meta["slug"], meta["home_kit"], meta["away_kit"], meta["verified"], meta["kit_source"]) == (
        "Dorchester Town F.C.", "dorchester_town", "#CC2222", "#1A1A24", True, "crest")   # no kit fields: crest guess
    badge = tmp_path / "dorchester_town" / "badge.png"
    assert meta["badge_path"] == str(badge) and Image.open(badge).format == "PNG"
    assert any("q=Dorchester+Town+F.C." in c for c in online) and any("Dorchester_Town_F.C." in c for c in online)
    assert "Women" not in "".join(c for c in online if "summary" in c)     # women's team and station skipped

    stored = json.loads((tmp_path / "dorchester_town" / "meta.json").read_text(encoding="utf-8"))
    assert set(stored) == {"name", "slug", "badge_path", "home_kit", "away_kit", "verified", "kit_source"}

    _no_network(monkeypatch)
    assert ca.resolve_club_assets("dorchester town", sources=tmp_path) == meta


def test_unknown_club_is_cached_as_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "_http_json", lambda url: {"pages": [{"title": "Supporting Charities", "description": "Charity"}]})
    meta = ca.resolve_club_assets("Supporting Charities", sources=tmp_path)
    assert meta == ca.fallback("Supporting Charities", "supporting_charities")
    assert (tmp_path / "supporting_charities" / "meta.json").exists()
    _no_network(monkeypatch)
    assert ca.resolve_club_assets("Supporting Charities", sources=tmp_path)["verified"] is False


def test_offline_falls_back_without_caching_then_recovers_online(tmp_path, monkeypatch):
    _offline(monkeypatch)
    meta = ca.resolve_club_assets(CLUB, sources=tmp_path)
    assert meta == {"name": CLUB, "slug": "dorchester_town", "badge_path": None, "home_kit": "#CC2222",
                    "away_kit": "#FFFFFF", "verified": False, "kit_source": "default"}
    assert not (tmp_path / "dorchester_town" / "meta.json").exists()

    monkeypatch.setattr(ca, "_http_json", lambda url: SUMMARY if "summary" in url else SEARCH)
    monkeypatch.setattr(ca, "_http_bytes", lambda url: _crest())
    assert ca.resolve_club_assets(CLUB, sources=tmp_path)["verified"] is True


def test_crest_download_failure_and_bad_image_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "_http_json", lambda url: SUMMARY if "summary" in url else SEARCH)

    def timeout(url):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(ca, "_http_bytes", timeout)
    meta = ca.resolve_club_assets(CLUB, sources=tmp_path)
    assert meta["verified"] and meta["badge_path"] is None and not (tmp_path / "dorchester_town" / "meta.json").exists()

    monkeypatch.setattr(ca, "_http_bytes", lambda url: b"<html>not an image</html>")
    meta = ca.resolve_club_assets(CLUB, sources=tmp_path)
    assert meta["verified"] and meta["badge_path"] is None and meta["home_kit"] == ca.DEFAULT_HOME
    assert (tmp_path / "dorchester_town" / "meta.json").exists()


def test_broken_cache_and_blank_names_never_raise(tmp_path, online):
    folder = tmp_path / "dorchester_town"
    folder.mkdir()
    (folder / "meta.json").write_text("{not json", encoding="utf-8")
    assert ca.resolve_club_assets(CLUB, sources=tmp_path)["verified"] is True      # unreadable cache → refetch
    (folder / "badge.png").unlink()
    assert ca.resolve_club_assets(CLUB, sources=tmp_path)["badge_path"] is None    # missing crest file → no crest
    assert ca.resolve_club_assets("   ", sources=tmp_path) == ca.fallback("", "")


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        return self.body if n is None or n < 0 else self.body[:n]


def _http_error(code: int, retry_after: str | None = None):
    from urllib.error import HTTPError

    return HTTPError("https://en.wikipedia.org/w/rest.php/v1/search/page", code, "error",
                     {"Retry-After": retry_after} if retry_after is not None else {}, None)


def test_rate_limited_request_waits_for_retry_after_then_succeeds(monkeypatch):
    replies, waits, agents = [_http_error(429, "4"), _http_error(429, "soon"), _Response(b'{"pages": []}')], [], []

    def fake_urlopen(request, timeout):
        agents.append(request.get_header("User-agent"))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(ca, "urlopen", fake_urlopen)
    monkeypatch.setattr(ca, "_sleep", waits.append)
    assert ca._http_json("https://en.wikipedia.org/w/rest.php/v1/search/page?q=x") == {"pages": []}
    assert waits == [4.0, 4.0]                              # Retry-After, then exponential backoff for a bad header
    assert all("github.com/rossco4789-glitch/pitchpulse" in a for a in agents)   # Wikimedia asks for contact details


def test_persistent_rate_limit_falls_back_without_caching_and_404_does_not_wait(tmp_path, monkeypatch):
    waits = []

    def always_429(request, timeout):
        raise _http_error(429, "60")

    monkeypatch.setattr(ca, "urlopen", always_429)
    monkeypatch.setattr(ca, "_sleep", waits.append)
    report = {}
    meta = ca.resolve_club_assets(CLUB, sources=tmp_path, report=report)
    assert meta["verified"] is False and report == {"saved": False}
    assert not (tmp_path / "dorchester_town" / "meta.json").exists()
    assert waits == [ca.MAX_RETRY_WAIT_S] * ca.RETRIES      # Retry-After 60 s capped at 10 s, then give up

    waits.clear()

    def missing(request, timeout):
        raise _http_error(404)

    monkeypatch.setattr(ca, "urlopen", missing)
    assert ca.resolve_club_assets(CLUB, sources=tmp_path)["verified"] is False and waits == []


def test_rugby_clubs_are_not_football_clubs():
    rugby = {"key": "Hartpury_University_R.F.C.", "title": "Hartpury University R.F.C.", "description": "Rugby union football club"}
    football = {"key": "Hartpury_University_F.C.", "title": "Hartpury University F.C.", "description": "Association football club in England"}
    assert not ca._is_club_page(rugby, "Hartpury University")
    assert not ca._is_club_page({**rugby, "description": ""}, "Hartpury University")          # R.F.C. title alone
    assert ca._is_club_page(football, "Hartpury University")


def test_slug_matches_the_scouting_folder_name():
    for name in ("Dorchester Town", "AFC Totton", "Hayes & Yeading United", "  Weymouth FC "):
        assert ca.slugify(name) == vc.slugify(name)


# ── Match Setup controls ─────────────────────────────────────────────────────

BANNED = ("homography", "bounding box", "yolo", "pipeline", "acceptance rate", "silhouette", "iqr", "median",
          "settled frames", "pid", "subprocess", "kaggle", "dispatch_id", "dispatch id", "manifest", "hex")


def _tab_script():
    from cv.vision_center_ui import render
    render()


def _all_text(node) -> list[str]:
    out = [str(node.proto)] if getattr(node, "proto", None) is not None else []
    for child in getattr(node, "children", {}).values():
        out += _all_text(child)
    return out


def test_setup_shows_crest_and_home_away_kit_toggle(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    for attr, sub in (("UI_STATE", "ui"), ("SOURCES", "sources"), ("STAGING_VIDEOS", "staging")):
        monkeypatch.setattr(vc, attr, tmp_path / sub)
    (tmp_path / "staging").mkdir()
    folder = tmp_path / "sources" / "weymouth"
    folder.mkdir(parents=True)
    Image.open(BytesIO(_crest(main=(0, 51, 153), second=(255, 204, 0)))).save(folder / "badge.png")
    (folder / "meta.json").write_text(json.dumps({
        "name": "Weymouth F.C.", "slug": "weymouth", "badge_path": str(folder / "badge.png"),
        "home_kit": "#003399", "away_kit": "#FFCC00", "verified": True, "kit_source": "club records"}), encoding="utf-8")
    _no_network(monkeypatch)

    at = AppTest.from_function(_tab_script, default_timeout=60)
    at.run()
    at.selectbox(key="vcc_opponent_pick").set_value("Weymouth").run()
    assert not at.exception
    assert at.session_state["vision_opponent_kit"] == "#003399"
    text = "\n".join(_all_text(at.main))
    assert "WEYMOUTH F.C." in text.upper() and "kit colours from club records" in text and "Custom kit colour" in text
    assert [t for t in BANNED if re.search(rf"\b{t}\b", text, re.I)] == []
    assert len(re.findall(r'url: "[^"]*\.png"', text)) == 1              # the crest, served as a PNG

    at.radio(key="vcc_kit_choice").set_value("Away").run()
    assert at.session_state["vision_opponent_kit"] == "#FFCC00"
    at.checkbox(key="vcc_kit_custom_on").check().run()
    at.color_picker(key="vcc_kit_custom").set_value("#00AA55").run()
    assert not at.exception and at.session_state["vision_opponent_kit"] == "#00AA55"
