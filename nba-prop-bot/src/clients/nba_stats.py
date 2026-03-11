import time
import os
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from nba_api.stats.endpoints import (
    playergamelogs, leaguedashteamstats,
    commonplayerinfo, boxscoretraditionalv2
)
from src.utils.retry import retry_with_backoff
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Build static team lookup tables once at import time
try:
    from nba_api.stats.static import teams as _nba_teams
    _ALL_TEAMS = _nba_teams.get_teams()
    _ABBR_TO_FULL = {t['abbreviation'].lower(): t['full_name'].lower() for t in _ALL_TEAMS}
    _FULL_TO_ID   = {t['full_name'].lower(): t['id'] for t in _ALL_TEAMS}
    _ABBR_TO_ID   = {t['abbreviation'].lower(): t['id'] for t in _ALL_TEAMS}
except Exception:
    _ABBR_TO_FULL = {}
    _FULL_TO_ID   = {}
    _ABBR_TO_ID   = {}

# Market → stat column mapping
_MARKET_OPP_COL = {
    'player_points':                   'OPP_PTS',
    'player_rebounds':                 'OPP_REB',
    'player_assists':                  'OPP_AST',
    'player_threes':                   'OPP_FG3M',
    'player_points_rebounds_assists':  None,
}


class NbaStatsClient:
    def __init__(self, season: str = "2024-25"):
        self.season = season
        self.cache_db = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'stats_cache.db')
        self._init_cache()
        self._opp_stats_cache: Optional[pd.DataFrame] = None  # in-memory for current run

    def _init_cache(self):
        os.makedirs(os.path.dirname(self.cache_db), exist_ok=True)
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS player_logs_cache (
                    player_id INTEGER PRIMARY KEY,
                    date_fetched TEXT,
                    data_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS opp_stats_cache (
                    season TEXT PRIMARY KEY,
                    date_fetched TEXT,
                    data_json TEXT
                )
            """)

    # ------------------------------------------------------------------ #
    #  Player game logs                                                    #
    # ------------------------------------------------------------------ #

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_player_game_logs(self, player_id: int) -> pd.DataFrame:
        today = datetime.now().strftime('%Y-%m-%d')

        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT date_fetched, data_json FROM player_logs_cache WHERE player_id = ?", (player_id,))
            row = cursor.fetchone()
            if row and row[0] == today:
                import io
                return pd.read_json(io.StringIO(row[1]))

        logger.info(f"Fetching game logs for player {player_id}")
        logs = playergamelogs.PlayerGameLogs(
            player_id_nullable=player_id,
            season_nullable=self.season
        )
        time.sleep(0.6)
        df = logs.get_data_frames()[0]

        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO player_logs_cache (player_id, date_fetched, data_json) VALUES (?, ?, ?)",
                (player_id, today, df.to_json())
            )

        return df

    # ------------------------------------------------------------------ #
    #  Team stats (pace / ratings)                                         #
    # ------------------------------------------------------------------ #

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_team_stats(self) -> pd.DataFrame:
        logger.info("Fetching team advanced stats (pace, ratings)")
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=self.season,
            measure_type_detailed_defense='Advanced'
        )
        time.sleep(0.6)
        return stats.get_data_frames()[0]

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_opponent_stats(self) -> pd.DataFrame:
        """
        Priority 2: Fetch per-team opponent-allowed stats (what each team gives up).
        Cached in-memory for the duration of a scan run.
        """
        if self._opp_stats_cache is not None:
            return self._opp_stats_cache

        today = datetime.now().strftime('%Y-%m-%d')
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT date_fetched, data_json FROM opp_stats_cache WHERE season = ?", (self.season,))
            row = cursor.fetchone()
            if row and row[0] == today:
                import io
                df = pd.read_json(io.StringIO(row[1]))
                self._opp_stats_cache = df
                return df

        logger.info("Fetching opponent-allowed stats from nba_api")
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=self.season,
            measure_type_detailed_defense='Opponent',
            per_mode_detailed='PerGame'
        )
        time.sleep(0.6)
        df = stats.get_data_frames()[0]

        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO opp_stats_cache (season, date_fetched, data_json) VALUES (?, ?, ?)",
                (self.season, today, df.to_json())
            )

        self._opp_stats_cache = df
        return df

    # ------------------------------------------------------------------ #
    #  Priority 2: Opponent defensive multiplier per market                #
    # ------------------------------------------------------------------ #

    def get_opponent_def_multiplier(self, opp_team_name: str, market: str) -> float:
        """
        Return a defensive multiplier for `market` when playing against `opp_team_name`.
        > 1.0 → opponent allows more than average (weak defense → inflate projection).
        < 1.0 → opponent allows less than average (strong defense → deflate).
        """
        opp_col = _MARKET_OPP_COL.get(market)
        if not opp_col:
            return 1.0  # PRA handled below

        try:
            df = self.get_opponent_stats()
            if df.empty or opp_col not in df.columns:
                return 1.0

            league_avg = df[opp_col].mean()
            if league_avg <= 0:
                return 1.0

            # Match team by full name (case-insensitive)
            opp_lower = opp_team_name.lower()
            match = df[df['TEAM_NAME'].str.lower() == opp_lower]
            if match.empty:
                # Try partial match
                match = df[df['TEAM_NAME'].str.lower().str.contains(opp_lower.split()[-1], na=False)]

            if match.empty:
                return 1.0

            team_val = float(match.iloc[0][opp_col])
            return round(team_val / league_avg, 4)

        except Exception as e:
            logger.warning(f"Could not compute opp multiplier for {opp_team_name}/{market}: {e}")
            return 1.0

    def get_opponent_def_multiplier_pra(self, opp_team_name: str) -> float:
        """Defensive multiplier for PRA (composite of PTS + REB + AST allowed)."""
        try:
            df = self.get_opponent_stats()
            needed = ['OPP_PTS', 'OPP_REB', 'OPP_AST']
            if df.empty or not all(c in df.columns for c in needed):
                return 1.0

            df['_pra'] = df['OPP_PTS'] + df['OPP_REB'] + df['OPP_AST']
            league_avg = df['_pra'].mean()
            if league_avg <= 0:
                return 1.0

            opp_lower = opp_team_name.lower()
            match = df[df['TEAM_NAME'].str.lower() == opp_lower]
            if match.empty:
                match = df[df['TEAM_NAME'].str.lower().str.contains(opp_lower.split()[-1], na=False)]
            if match.empty:
                return 1.0

            return round(float(match.iloc[0]['_pra']) / league_avg, 4)
        except Exception:
            return 1.0

    def get_team_pace(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Return pace for both teams. Used to replace hardcoded 99.0."""
        try:
            df = self.get_team_stats()
            if df.empty or 'PACE' not in df.columns:
                return {'home_pace': 99.0, 'away_pace': 99.0}

            def _pace(name: str) -> float:
                low = name.lower()
                row = df[df['TEAM_NAME'].str.lower() == low]
                if row.empty:
                    row = df[df['TEAM_NAME'].str.lower().str.contains(low.split()[-1], na=False)]
                return float(row.iloc[0]['PACE']) if not row.empty else 99.0

            return {'home_pace': _pace(home_team), 'away_pace': _pace(away_team)}
        except Exception:
            return {'home_pace': 99.0, 'away_pace': 99.0}

    # ------------------------------------------------------------------ #
    #  Priority 4: Home/away + rest day helpers                            #
    # ------------------------------------------------------------------ #

    def is_home_team(self, player_team_abbr: str, home_team_full: str) -> bool:
        """Return True if the player's team is the home team in this event."""
        if not isinstance(player_team_abbr, str) or not player_team_abbr:
            return False
        player_full = _ABBR_TO_FULL.get(player_team_abbr.lower(), '')
        return player_full == home_team_full.lower()

    @staticmethod
    def calculate_rest_days(logs: pd.DataFrame) -> int:
        """
        Priority 4: Days of rest the player has before the upcoming game.
        0 = back-to-back, 1 = one day rest, 3+ = extended rest.
        """
        if logs.empty or 'GAME_DATE' not in logs.columns:
            return 2  # default: assume normal rest

        try:
            last_game_date = pd.to_datetime(logs.iloc[0]['GAME_DATE'])
            today = pd.Timestamp.today().normalize()
            return max(0, int((today - last_game_date).days) - 1)
        except Exception:
            return 2

    # ------------------------------------------------------------------ #
    #  Priority 5: Starter/lineup detection                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def infer_starter_flag(logs: pd.DataFrame, minutes_threshold: float = 25.0) -> bool:
        """
        Priority 5: Infer whether a player is a starter based on recent minutes.
        Players averaging >= 25 min over their last 10 games are treated as starters.
        """
        if logs.empty or 'MIN' not in logs.columns:
            return False
        avg = logs.head(10)['MIN'].mean()
        return bool(avg >= minutes_threshold) if not pd.isna(avg) else False

    # ------------------------------------------------------------------ #
    #  Existing helpers                                                    #
    # ------------------------------------------------------------------ #

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_player_info(self, player_id: int) -> Dict[str, Any]:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        time.sleep(0.6)
        return info.get_dict()

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_box_score(self, game_id: str) -> pd.DataFrame:
        logger.info(f"Fetching box score for game {game_id}")
        if not str(game_id).startswith('00'):
            logger.warning(f"Game ID {game_id} may not be NBA API format.")
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        time.sleep(0.6)
        return box.get_data_frames()[0]
