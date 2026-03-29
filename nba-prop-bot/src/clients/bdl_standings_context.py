"""
BDL standings context — blowout-risk minutes adjustment based on team records.

Fetches season standings once per season and derives per-game minutes
adjustment factors. Strong home favorites may rest starters in garbage time.
"""

from typing import Dict
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Apply adjustment only when win-pct gap is this large
_BLOWOUT_THRESHOLD = 0.20   # 20 pp win-pct difference
_BLOWOUT_ADJ = 0.95         # ~5% minutes reduction for potential garbage time


class BDLStandingsContext:
    """
    Fetches BDL standings and provides per-game context for the scan pipeline.

    Main output: minutes_adj_home / minutes_adj_away (1.0 = no adj;
    <1.0 = possible garbage-time rest for strong favorite's stars).
    """

    def __init__(self, bdl_client):
        self.bdl = bdl_client
        self._cache: Dict[int, Dict[str, dict]] = {}  # season → team_index

    def get_game_context(
        self, home_team: str, away_team: str, season: int
    ) -> Dict[str, float]:
        """
        Return blowout-risk adjustment and win percentages for a matchup.

        Returns:
            minutes_adj_home  float  1.0 or <1.0 (home team star minutes adj)
            minutes_adj_away  float  1.0 or <1.0 (away team star minutes adj)
            home_win_pct      float  home team season win %
            away_win_pct      float  away team season win %
        """
        idx = self._get_indexed_standings(season)
        home_info = self._find_team(idx, home_team)
        away_info = self._find_team(idx, away_team)

        home_win_pct = home_info.get('win_pct', 0.5)
        away_win_pct = away_info.get('win_pct', 0.5)

        diff = home_win_pct - away_win_pct
        home_adj = _BLOWOUT_ADJ if diff >= _BLOWOUT_THRESHOLD else 1.0
        away_adj = _BLOWOUT_ADJ if diff <= -_BLOWOUT_THRESHOLD else 1.0

        logger.debug(
            f"Standings ctx: {home_team}({home_win_pct:.2f}) vs "
            f"{away_team}({away_win_pct:.2f}) → "
            f"home_adj={home_adj} away_adj={away_adj}"
        )
        return {
            'minutes_adj_home': home_adj,
            'minutes_adj_away': away_adj,
            'home_win_pct':     home_win_pct,
            'away_win_pct':     away_win_pct,
        }

    def _get_indexed_standings(self, season: int) -> Dict[str, dict]:
        if season not in self._cache:
            try:
                standings = self.bdl.get_standings(season)
                self._cache[season] = self._index_standings(standings)
                logger.debug(
                    f"BDL standings: indexed {len(self._cache[season])} entries "
                    f"for season {season}"
                )
            except Exception as e:
                logger.warning(f"BDL standings fetch failed for {season}: {e}")
                self._cache[season] = {}
        return self._cache[season]

    @staticmethod
    def _index_standings(standings) -> Dict[str, dict]:
        idx: Dict[str, dict] = {}
        for s in standings:
            team = s.get('team') or {}
            # BDL may use 'wins'/'losses' or 'won'/'lost' depending on version
            wins   = int(s.get('wins', s.get('won', 0)) or 0)
            losses = int(s.get('losses', s.get('lost', 0)) or 0)
            total  = wins + losses
            win_pct = wins / total if total > 0 else 0.5
            info = {'wins': wins, 'losses': losses, 'win_pct': win_pct}
            abbr = (team.get('abbreviation') or '').lower()
            full = (team.get('full_name') or '').lower()
            if abbr:
                idx[abbr] = info
            if full:
                idx[full] = info
        return idx

    @staticmethod
    def _find_team(idx: dict, team_name: str) -> dict:
        low = team_name.lower()
        if low in idx:
            return idx[low]
        for k, v in idx.items():
            if k and (k in low or low in k):
                return v
        return {'win_pct': 0.5}
