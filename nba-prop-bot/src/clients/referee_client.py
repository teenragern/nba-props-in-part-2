"""
Fetch today's NBA referee assignments from official.nba.com.

Returns a list of assignment dicts:
    [{'home': str, 'away': str, 'refs': [str, str, str]}, ...]

Falls back to [] on any network or parse error — all downstream logic
treats missing data as a no-op (foul_rate_multiplier = 1.0).
"""

import requests
from datetime import date
from typing import List, Dict, Any
from bs4 import BeautifulSoup

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_OFFICIALS_URL = "https://official.nba.com/referee-assignments/"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_today_assignments() -> List[Dict[str, Any]]:
    """
    Scrape official.nba.com/referee-assignments/ for today's crew assignments.

    Returns [{'home': str, 'away': str, 'refs': [str, ...]}, ...].
    Falls back to [] on any error.
    """
    try:
        resp = requests.get(_OFFICIALS_URL, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.debug(f"Referee assignments fetch failed: {e}")
        return []

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        assignments = _parse_assignments(soup)
        logger.info(
            f"Referee assignments: {len(assignments)} game(s) for "
            f"{date.today().isoformat()}."
        )
        return assignments
    except Exception as e:
        logger.debug(f"Referee assignments parse error: {e}")
        return []


def match_event_refs(
    home_team: str, away_team: str, assignments: List[Dict[str, Any]]
) -> List[str]:
    """
    Find the referee crew for a game from today's pre-fetched assignments.

    Matches by comparing team nicknames (last word of team name) against
    each assignment's home/away strings — e.g. 'Celtics' in 'Boston Celtics'.

    Returns a list of ref name strings (may be empty if no match).
    """
    home_nick = home_team.lower().split()[-1]  # 'celtics'
    away_nick  = away_team.lower().split()[-1]  # 'hawks'

    for a in assignments:
        a_home = a["home"].lower()
        a_away = a["away"].lower()
        if home_nick in a_home and away_nick in a_away:
            return a["refs"]
    return []


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _today_formats() -> List[str]:
    """Return multiple date-format strings for today (for flexible matching)."""
    d = date.today()
    m, day, yr = d.month, d.day, d.year
    return [
        f"{m}/{day}/{yr}",           # 3/20/2026
        f"{m:02d}/{day:02d}/{yr}",   # 03/20/2026
        d.strftime("%B %d, %Y"),     # March 20, 2026
        d.strftime("%b %d, %Y"),     # Mar 20, 2026
    ]


def _parse_assignments(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """
    Walk every <table> in the page and collect rows where the date cell
    matches today. Assumes column layout: [Date, Game, Ref1, Ref2, Ref3].
    """
    today_fmts = _today_formats()
    results: List[Dict[str, Any]] = []

    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in row.find_all("td")]
            if len(cells) < 4:
                continue

            # Column 0: date — only keep today's rows
            if not any(fmt in cells[0] for fmt in today_fmts):
                continue

            # Column 1: "Away Team at Home Team" or "Away Team vs. Home Team"
            home, away = _parse_game_cell(cells[1])
            if not home or not away:
                continue

            refs = [c for c in cells[2:] if c.strip()]
            results.append({"home": home, "away": away, "refs": refs[:3]})

    return results


def _parse_game_cell(text: str):
    """
    Parse 'Atlanta Hawks at Boston Celtics' → ('Boston Celtics', 'Atlanta Hawks').
    The 'at' convention means first team is away, second is home.
    For 'vs.' the first team is listed as away on the officials site.
    Returns ('', '') on failure.
    """
    for sep in (" at ", " @ ", " vs. ", " vs "):
        if sep in text:
            parts = text.split(sep, 1)
            away, home = parts[0].strip(), parts[1].strip()
            return home, away
    return "", ""
