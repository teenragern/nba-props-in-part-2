"""
Kalshi team-name matching utilities.

Shared by game_arb.py and the live trading engine.  Provides the team-name
lookup table and the fuzzy-matching function that maps an NBA matchup to the
correct Kalshi KXNBAGAME market ticker(s).
"""

from typing import Dict, List, Optional, Tuple

from src.clients.exchange_client import ExchangeClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Maps last word of team name (lower-case) → (city_fragment, abbreviation).
# Kalshi titles often say "Minnesota" not "Timberwolves", so we check all variants.
TEAM_LOOKUP: Dict[str, Tuple[str, str]] = {
    "timberwolves": ("minnesota",     "min"),
    "spurs":        ("san antonio",   "sa"),
    "celtics":      ("boston",        "bos"),
    "lakers":       ("los angeles",   "lal"),
    "warriors":     ("golden state",  "gsw"),
    "nets":         ("brooklyn",      "bkn"),
    "knicks":       ("new york",      "nyk"),
    "bulls":        ("chicago",       "chi"),
    "heat":         ("miami",         "mia"),
    "76ers":        ("philadelphia",  "phi"),
    "raptors":      ("toronto",       "tor"),
    "bucks":        ("milwaukee",     "mil"),
    "pacers":       ("indiana",       "ind"),
    "cavaliers":    ("cleveland",     "cle"),
    "pistons":      ("detroit",       "det"),
    "hawks":        ("atlanta",       "atl"),
    "hornets":      ("charlotte",     "cha"),
    "wizards":      ("washington",    "was"),
    "magic":        ("orlando",       "orl"),
    "nuggets":      ("denver",        "den"),
    "thunder":      ("oklahoma city", "okc"),
    "jazz":         ("utah",          "uta"),
    "suns":         ("phoenix",       "phx"),
    "clippers":     ("los angeles",   "lac"),
    "kings":        ("sacramento",    "sac"),
    "blazers":      ("portland",      "por"),
    "trail":        ("portland",      "por"),
    "grizzlies":    ("memphis",       "mem"),
    "pelicans":     ("new orleans",   "nop"),
    "mavericks":    ("dallas",        "dal"),
    "rockets":      ("houston",       "hou"),
}


def team_variants(team_name: str) -> List[str]:
    """
    Return all name variants for a team name string.

    E.g. "Minnesota Timberwolves" → ["timberwolves", "minnesota", "min"]
    """
    last = team_name.split()[-1].lower()
    variants = [last]
    info = TEAM_LOOKUP.get(last)
    if info:
        variants.append(info[0])   # city name
        variants.append(info[1])   # abbreviation
    return variants


def find_kalshi_game_markets(
    client: ExchangeClient,
    home_team: str,
    away_team: str,
) -> List[Dict]:
    """
    Fetch all open KXNBAGAME markets and return those matching the given matchup.

    Matches against team nickname, city name, and abbreviation in both ticker
    and title.  Falls back to text search when the series fetch returns nothing
    (e.g. Kalshi API change).

    Args:
        client:    Authenticated ExchangeClient.
        home_team: Full team name, e.g. "Boston Celtics".
        away_team: Full team name, e.g. "New York Knicks".

    Returns:
        List of Kalshi market dicts for this matchup (may be empty).
    """
    all_markets = client.get_nba_game_markets()
    logger.debug(f"kalshi_matching: {len(all_markets)} KXNBAGAME markets fetched")

    home_variants = team_variants(home_team)
    away_variants = team_variants(away_team)

    matches = []
    for m in all_markets:
        text = (m.get("ticker", "") + " " + m.get("title", "")).lower()
        if any(v in text for v in home_variants) and any(v in text for v in away_variants):
            matches.append(m)

    # Fallback: text search if series fetch returned nothing
    if not all_markets:
        home_last = home_team.split()[-1].lower()
        city = TEAM_LOOKUP.get(home_last, (home_last,))[0]
        for m in client.search_markets(city, limit=30):
            text = (m.get("ticker", "") + " " + m.get("title", "")).lower()
            if any(v in text for v in home_variants) and any(v in text for v in away_variants):
                matches.append(m)

    if matches:
        logger.debug(
            f"kalshi_matching: {len(matches)} market(s) for "
            f"{home_team} vs {away_team}"
        )
    return matches


def resolve_team_for_market(
    market: Dict,
    home_team: str,
    away_team: str,
) -> Optional[str]:
    """
    Given a Kalshi market dict, return the team name whose WIN the YES contract
    represents, or None if it cannot be determined.

    Args:
        market:    Kalshi market dict with 'ticker' and 'title' keys.
        home_team: Full home team name.
        away_team: Full away team name.

    Returns:
        home_team, away_team, or None.
    """
    text = (market.get("ticker", "") + " " + market.get("title", "")).lower()
    home_last = home_team.split()[-1].lower()
    away_last = away_team.split()[-1].lower()

    if home_last in text:
        return home_team
    if away_last in text:
        return away_team
    return None
