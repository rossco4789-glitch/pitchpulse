"""
cv/league_roster.py — league opponent list for the Match Setup quick-select in the Opposition Analysis tab.

SOUTHERN_LEAGUE_DIV_ONE_SOUTH is the 2026/27 Southern League Division One South roster as supplied by the club.
Tiverton Town is in the league but is never an opponent, so opponent_options() leaves it out.
"""

from __future__ import annotations

OWN_CLUB    = "Tiverton Town"
PLACEHOLDER = "-- Select Opponent --"
CUSTOM      = "Other / Custom..."

SOUTHERN_LEAGUE_DIV_ONE_SOUTH = [
    "Barnstaple Town",
    "Bideford",
    "Bishop's Cleeve",
    "Bristol Manor Farm",
    "Dorchester Town",
    "Exmouth Town",
    "Falmouth Town",
    "Hartpury University",
    "Hungerford Town",
    "Larkhall Athletic",
    "Melksham Town",
    "Paulton Rovers",
    "Portland United",
    "Shaftesbury",
    "Slimbridge",
    "Sporting Club Inkberrow",
    "Swindon Supermarine",
    "Tiverton Town",
    "Westbury United",
    "Weymouth",
    "Willand Rovers",
    "Worcester Raiders",
]


def opponent_options() -> list[str]:
    return [PLACEHOLDER, *(club for club in SOUTHERN_LEAGUE_DIV_ONE_SOUTH if club != OWN_CLUB), CUSTOM]


def opponent_name(pick: str | None, custom_text: str | None) -> str:
    """Opponent chosen in Match Setup: a league club, the typed name for Other / Custom, or '' for no choice."""
    if pick == CUSTOM:
        return (custom_text or "").strip()
    return pick if pick in SOUTHERN_LEAGUE_DIV_ONE_SOUTH and pick != OWN_CLUB else ""
