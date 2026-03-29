"""
BDL game logs adapter.

Wraps bdl.get_game_stats() and returns a DataFrame with nba_api-compatible
column names so scan_props.py can use BDL as a drop-in replacement for
nba_api game log fetches (no rate limit vs. nba_api's 0.6s/call).

Columns returned: MIN, PTS, REB, AST, FG3M, BLK, STL, FGA, FTA, TOV,
                  TEAM_ABBREVIATION, MATCHUP, GAME_DATE, WL
Sorted: newest game first.
"""

import pandas as pd
from typing import Dict, List
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _parse_minutes(min_val) -> float:
    """Parse BDL 'min' field ('MM:SS' or numeric) → float minutes."""
    if min_val is None:
        return 0.0
    s = str(min_val).strip()
    if ':' in s:
        try:
            parts = s.split(':')
            return float(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, IndexError):
            return 0.0
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


class BDLGameLogs:
    """
    Fetches per-game player box score stats from BDL and returns
    a DataFrame compatible with the nba_api game logs shape that
    scan_props.py and ml_model.py already expect.
    """

    def __init__(self, bdl_client):
        self.bdl = bdl_client
        self._log_cache: Dict[tuple, pd.DataFrame] = {}
        self._game_id_cache: Dict[tuple, List[int]] = {}  # (pid, season) → BDL game IDs

    def get_player_game_logs(self, bdl_player_id: int, season: int) -> pd.DataFrame:
        """
        Return game logs for a player/season as a nba_api-compatible DataFrame.
        Fetches from BDL (no sleep needed) and caches the result.
        Returns empty DataFrame on failure or if player has no recorded stats.
        """
        cache_key = (bdl_player_id, season)
        if cache_key in self._log_cache:
            return self._log_cache[cache_key]

        try:
            stats = self.bdl.get_game_stats(
                player_ids=[bdl_player_id], seasons=[season]
            )
        except Exception as e:
            logger.warning(f"BDLGameLogs fetch error for player {bdl_player_id}: {e}")
            empty = pd.DataFrame()
            self._log_cache[cache_key] = empty
            return empty

        if not stats:
            empty = pd.DataFrame()
            self._log_cache[cache_key] = empty
            return empty

        rows: list = []
        game_ids: List[int] = []

        for s in stats:
            minutes = _parse_minutes(s.get('min'))
            if minutes <= 0:
                continue

            game = s.get('game') or {}
            team = s.get('team') or {}
            team_abbr = (team.get('abbreviation') or '').upper()

            home_team = game.get('home_team') or {}
            away_team = game.get('visitor_team') or {}
            home_abbr = (home_team.get('abbreviation') or '').upper()
            away_abbr = (away_team.get('abbreviation') or '').upper()

            is_home = bool(home_abbr and home_abbr == team_abbr)
            opp_abbr = away_abbr if is_home else home_abbr

            if is_home:
                matchup = f"{team_abbr} vs. {opp_abbr}"
            else:
                matchup = f"{team_abbr} @ {opp_abbr}"

            home_score = int(game.get('home_team_score') or 0)
            away_score = int(game.get('visitor_team_score') or 0)
            if home_score > 0 and away_score > 0:
                own_score = home_score if is_home else away_score
                opp_score = away_score if is_home else home_score
                wl = 'W' if own_score > opp_score else 'L'
            else:
                wl = ''

            raw_date = game.get('date') or game.get('datetime') or ''
            game_date = str(raw_date)[:10] if raw_date else ''

            rows.append({
                'MIN':               minutes,
                'PTS':               float(s.get('pts') or 0),
                'REB':               float(s.get('reb') or 0),
                'AST':               float(s.get('ast') or 0),
                'FG3M':              float(s.get('fg3m') or 0),
                'BLK':               float(s.get('blk') or 0),
                'STL':               float(s.get('stl') or 0),
                'FGA':               float(s.get('fga') or 0),
                'FTA':               float(s.get('fta') or 0),
                'TOV':               float(s.get('turnover') or 0),
                'TEAM_ABBREVIATION': team_abbr,
                'MATCHUP':           matchup,
                'GAME_DATE':         game_date,
                'WL':                wl,
            })

            gid = game.get('id')
            if gid:
                game_ids.append(int(gid))

        if not rows:
            empty = pd.DataFrame()
            self._log_cache[cache_key] = empty
            return empty

        df = pd.DataFrame(rows)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
        df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)

        self._log_cache[cache_key] = df
        self._game_id_cache[cache_key] = game_ids
        logger.debug(
            f"BDL game logs: {len(df)} games for player {bdl_player_id} (season {season})"
        )
        return df

    def get_recent_game_ids(
        self, bdl_player_id: int, season: int, n: int = 5
    ) -> List[int]:
        """
        Return the n most recent BDL game IDs for a player/season.
        Ensures game logs are loaded first (populates the game_id cache).
        """
        self.get_player_game_logs(bdl_player_id, season)
        ids = self._game_id_cache.get((bdl_player_id, season), [])
        return ids[:n]
