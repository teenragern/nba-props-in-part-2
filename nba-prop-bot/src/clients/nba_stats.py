import time
import os
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from nba_api.stats.endpoints import (
    playergamelogs, leaguedashteamstats, leaguedashplayerstats,
    commonplayerinfo, boxscoretraditionalv2
)
from src.utils.retry import retry_with_backoff
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_current_nba_season() -> str:
    """
    Return the current NBA season string (e.g. '2025-26').
    October or later → current year is the start year.
    January–September → previous year is the start year.
    """
    now = datetime.now()
    start_year = now.year if now.month >= 10 else now.year - 1
    end_suffix = str(start_year + 1)[-2:]
    return f"{start_year}-{end_suffix}"


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

# ---------------------------------------------------------------------------
# Positional defensive efficiency
# ---------------------------------------------------------------------------
# For each position group, encodes how sensitive each prop market is to the
# OPPONENT team's defensive strength relative to the average player.
# Factor > 1.0  → this position benefits MORE from a weak opponent defense.
# Factor < 1.0  → this position is less impacted (e.g. guards vs REB defense).
_POSITION_FACTORS: dict = {
    'Guard': {
        'player_points':                  1.05,
        'player_rebounds':                0.50,
        'player_assists':                 1.30,
        'player_threes':                  1.25,
        'player_blocks':                  0.25,
        'player_steals':                  1.20,
        'player_points_rebounds_assists': 0.90,
    },
    'Forward': {
        'player_points':                  1.00,
        'player_rebounds':                0.85,
        'player_assists':                 0.70,
        'player_threes':                  0.95,
        'player_blocks':                  0.65,
        'player_steals':                  0.85,
        'player_points_rebounds_assists': 0.90,
    },
    'Center': {
        'player_points':                  0.95,
        'player_rebounds':                1.50,
        'player_assists':                 0.40,
        'player_threes':                  0.20,
        'player_blocks':                  1.80,
        'player_steals':                  0.55,
        'player_points_rebounds_assists': 1.05,
    },
}
# How much the positional factor modulates the team-level multiplier.
# 0.25 → 25% of deviation from 1.0 comes from positional adjustment.
_POSITION_BLEND = 0.25


def _playoff_blend_weight(po_gp: float) -> float:
    """
    Blend weight placed on playoff stats vs. regular season, based on the number
    of playoff games the team has played.
      < 3 games → 0.0  (too small a sample, pure RS)
      3-4 games → 0.40
      5+ games  → 0.60
    """
    if po_gp < 3:
        return 0.0
    if po_gp < 5:
        return 0.40
    return 0.60


def _blend_playoff_stats(rs_df: pd.DataFrame, po_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-team weighted blend of RS and Playoff team-stats frames.
    Teams below the playoff-games threshold keep pure RS values; others blend
    per-column based on their playoff GP.
    Non-numeric and identifier columns (TEAM_ID/GP/W/L/MIN) are never blended.
    """
    if rs_df is None or rs_df.empty:
        return rs_df
    if po_df is None or po_df.empty:
        return rs_df.copy()
    if 'TEAM_NAME' not in rs_df.columns or 'TEAM_NAME' not in po_df.columns:
        return rs_df.copy()

    skip = {'TEAM_ID', 'TEAM_NAME', 'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'MIN'}
    numeric_cols = [
        c for c in rs_df.columns
        if c not in skip
        and c in po_df.columns
        and pd.api.types.is_numeric_dtype(rs_df[c])
        and pd.api.types.is_numeric_dtype(po_df[c])
    ]

    blended = rs_df.copy()
    po_indexed = po_df.copy()
    po_indexed['_tn_lower'] = po_indexed['TEAM_NAME'].astype(str).str.lower()
    po_indexed = po_indexed.drop_duplicates(subset=['_tn_lower']).set_index('_tn_lower')

    # Ensure numeric columns are float to avoid FutureWarnings when blending
    for col in numeric_cols:
        if col in blended.columns:
            blended[col] = blended[col].astype(float)

    for idx in blended.index:
        team_name_lower = str(blended.at[idx, 'TEAM_NAME']).lower()
        if team_name_lower not in po_indexed.index:
            continue
        po_row = po_indexed.loc[team_name_lower]
        po_gp = float(po_row.get('GP', 0) or 0)
        w_po = _playoff_blend_weight(po_gp)
        if w_po <= 0.0:
            continue
        for col in numeric_cols:
            rs_val = float(blended.at[idx, col])
            po_val = float(po_row.get(col, rs_val))
            blended.at[idx, col] = (1.0 - w_po) * rs_val + w_po * po_val
    return blended


class NbaStatsClient:
    def __init__(self, season: str = None):
        self.season = season or get_current_nba_season()
        self.cache_db = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'stats_cache.db')
        self._init_cache()
        # Caches keyed by season_type ('Regular Season' | 'Playoffs')
        self._opp_stats_cache: Dict[str, pd.DataFrame] = {}
        self._adv_stats_cache: Dict[str, pd.DataFrame] = {}
        self._def_stats_cache: Dict[str, pd.DataFrame] = {}
        # Blended (RS + Playoffs per-team) caches
        self._blended_adv_cache: Optional[pd.DataFrame] = None
        self._blended_def_cache: Optional[pd.DataFrame] = None
        self._blended_opp_cache: Optional[pd.DataFrame] = None
        self._player_season_cache: Optional[pd.DataFrame] = None  # PerGame player stats (PG/big detection)

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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS def_stats_cache (
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
        logs_rs = playergamelogs.PlayerGameLogs(
            player_id_nullable=player_id,
            season_nullable=self.season,
            season_type_nullable='Regular Season'
        )
        time.sleep(0.6)
        df_rs = logs_rs.get_data_frames()[0]

        logs_po = playergamelogs.PlayerGameLogs(
            player_id_nullable=player_id,
            season_nullable=self.season,
            season_type_nullable='Playoffs'
        )
        time.sleep(0.6)
        df_po = logs_po.get_data_frames()[0]

        df = pd.concat([df_po, df_rs], ignore_index=True)
        if not df.empty and 'GAME_DATE' in df.columns:
            df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)

        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO player_logs_cache (player_id, date_fetched, data_json) VALUES (?, ?, ?)",
                (player_id, today, df.to_json())
            )

        return df

    def get_player_game_logs_season(self, player_id: int, season: str) -> pd.DataFrame:
        """
        Fetch game logs for a specific season string (e.g. '2022-23').
        Persisted in a separate backtest cache table — never expires so historical
        data is only downloaded once.
        """
        cache_key = f"{player_id}_{season}"
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_logs_cache (
                    cache_key TEXT PRIMARY KEY,
                    data_json TEXT
                )
            """)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data_json FROM backtest_logs_cache WHERE cache_key = ?",
                (cache_key,)
            )
            row = cursor.fetchone()
            if row:
                import io
                return pd.read_json(io.StringIO(row[0]))

        logger.info(f"Fetching historical logs: player={player_id} season={season}")
        try:
            logs_rs = playergamelogs.PlayerGameLogs(
                player_id_nullable=player_id,
                season_nullable=season,
                season_type_nullable='Regular Season'
            )
            time.sleep(0.7)
            df_rs = logs_rs.get_data_frames()[0]

            logs_po = playergamelogs.PlayerGameLogs(
                player_id_nullable=player_id,
                season_nullable=season,
                season_type_nullable='Playoffs'
            )
            time.sleep(0.7)
            df_po = logs_po.get_data_frames()[0]

            df = pd.concat([df_po, df_rs], ignore_index=True)
            if not df.empty and 'GAME_DATE' in df.columns:
                df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Historical log fetch failed ({player_id}/{season}): {e}")
            return pd.DataFrame()

        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO backtest_logs_cache (cache_key, data_json) VALUES (?, ?)",
                (cache_key, df.to_json()),
            )
        return df

    def get_all_active_player_ids(self, min_gp: int = 20) -> list:
        """
        Return list of player IDs for all active players with >= min_gp games
        this season. Used by backtest and train_ml pipelines.
        """
        try:
            from nba_api.stats.endpoints import leaguedashplayerstats
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=self.season,
                per_mode_detailed='Totals',
            )
            time.sleep(0.6)
            df = stats.get_data_frames()[0]
            if df.empty or 'GP' not in df.columns:
                return []
            qualified = df[df['GP'] >= min_gp]
            return qualified['PLAYER_ID'].tolist()
        except Exception as e:
            logger.warning(f"Could not fetch active player list: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Team stats (pace / ratings)                                         #
    # ------------------------------------------------------------------ #

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_team_stats(self, season_type: str = 'Regular Season') -> pd.DataFrame:
        """Advanced stats: PACE, DEF_RATING, DREB_PCT, NET_RATING, etc."""
        if season_type in self._adv_stats_cache:
            return self._adv_stats_cache[season_type]
        logger.info(f"Fetching team advanced stats (pace, ratings) [{season_type}]")
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=self.season,
            season_type_all_star=season_type,
            measure_type_detailed_defense='Advanced'
        )
        time.sleep(0.6)
        df = stats.get_data_frames()[0]
        self._adv_stats_cache[season_type] = df
        return df

    def get_blended_team_stats(self) -> pd.DataFrame:
        """Advanced stats blended per-team with playoff data (see _blend_playoff_stats)."""
        if self._blended_adv_cache is not None:
            return self._blended_adv_cache
        rs = self.get_team_stats('Regular Season')
        try:
            po = self.get_team_stats('Playoffs')
        except Exception as e:
            logger.warning(f"Playoff advanced stats fetch failed, falling back to RS: {e}")
            po = pd.DataFrame()
        self._blended_adv_cache = _blend_playoff_stats(rs, po)
        return self._blended_adv_cache

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_team_defense_stats(self, season_type: str = 'Regular Season') -> pd.DataFrame:
        """
        Defense dashboard stats per game: OPP_PTS_PAINT, OPP_PTS_OFF_TOV, etc.
        Regular-season data is cached in SQLite (refreshed daily); playoff data
        is in-memory only.
        """
        if season_type in self._def_stats_cache:
            return self._def_stats_cache[season_type]

        if season_type == 'Regular Season':
            today = datetime.now().strftime('%Y-%m-%d')
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT date_fetched, data_json FROM def_stats_cache WHERE season = ?", (self.season,))
                row = cursor.fetchone()
                if row and row[0] == today:
                    import io
                    df = pd.read_json(io.StringIO(row[1]))
                    self._def_stats_cache[season_type] = df
                    return df

        logger.info(f"Fetching team defense dashboard stats from nba_api [{season_type}]")
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=self.season,
            season_type_all_star=season_type,
            measure_type_detailed_defense='Defense',
            per_mode_detailed='PerGame',
        )
        time.sleep(0.6)
        df = stats.get_data_frames()[0]

        if season_type == 'Regular Season':
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO def_stats_cache (season, date_fetched, data_json) VALUES (?, ?, ?)",
                    (self.season, today, df.to_json())
                )

        self._def_stats_cache[season_type] = df
        return df

    def get_blended_team_defense_stats(self) -> pd.DataFrame:
        """Defense dashboard blended per-team with playoff data."""
        if self._blended_def_cache is not None:
            return self._blended_def_cache
        rs = self.get_team_defense_stats('Regular Season')
        try:
            po = self.get_team_defense_stats('Playoffs')
        except Exception as e:
            logger.warning(f"Playoff defense stats fetch failed, falling back to RS: {e}")
            po = pd.DataFrame()
        self._blended_def_cache = _blend_playoff_stats(rs, po)
        return self._blended_def_cache

    def get_opponent_matchup_context(
        self, opp_team_name: str, playoff_blend: bool = False
    ) -> Dict[str, float]:
        """
        Return normalized matchup context features for a given opponent team.
        All values normalized so league average = 1.0.

        When playoff_blend=True, per-team stats are blended with playoff data
        (see _blend_playoff_stats). League averages are also drawn from the
        blended frames so normalization stays consistent.

        Returns:
            opp_pace:        opponent pace / league avg pace
            opp_rebound_pct: opponent DREB_PCT / league avg DREB_PCT
            opp_pts_paint:   opponent OPP_PTS_PAINT / league avg OPP_PTS_PAINT
        """
        result: Dict[str, float] = {
            'opp_pace': 1.0,
            'opp_rebound_pct': 1.0,
            'opp_pts_paint': 1.0,
            'opp_fta_rate': 1.0,   # normalized opponent FTA drawn (foul aggressiveness proxy)
        }
        try:
            if playoff_blend:
                adv_df = self.get_blended_team_stats()
                def_df = self.get_blended_team_defense_stats()
                opp_df = self.get_blended_opponent_stats()
            else:
                adv_df = self.get_team_stats()
                def_df = self.get_team_defense_stats()
                opp_df = self.get_opponent_stats()  # OPP_FTA lives here

            opp_lower = opp_team_name.lower()

            def _match(df: pd.DataFrame):
                row = df[df['TEAM_NAME'].str.lower() == opp_lower]
                if row.empty:
                    row = df[df['TEAM_NAME'].str.lower().str.contains(
                        opp_lower.split()[-1], na=False)]
                return row.iloc[0] if not row.empty else None

            adv_row = _match(adv_df)
            def_row = _match(def_df)
            opp_row = _match(opp_df)

            if adv_row is not None:
                if 'PACE' in adv_df.columns:
                    league_avg = adv_df['PACE'].mean()
                    if league_avg > 0:
                        result['opp_pace'] = float(adv_row['PACE']) / league_avg
                if 'DREB_PCT' in adv_df.columns:
                    league_avg = adv_df['DREB_PCT'].mean()
                    if league_avg > 0:
                        result['opp_rebound_pct'] = float(adv_row['DREB_PCT']) / league_avg

            if def_row is not None and 'OPP_PTS_PAINT' in def_df.columns:
                league_avg = def_df['OPP_PTS_PAINT'].mean()
                if league_avg > 0:
                    result['opp_pts_paint'] = float(def_row['OPP_PTS_PAINT']) / league_avg

            # OPP_FTA: free throw attempts drawn by the opponent per game.
            # Higher → opponent's offense draws more fouls → elevated foul-trouble risk.
            if opp_row is not None and 'OPP_FTA' in opp_df.columns:
                league_avg = opp_df['OPP_FTA'].mean()
                if league_avg > 0:
                    result['opp_fta_rate'] = float(opp_row['OPP_FTA']) / league_avg

        except Exception as e:
            logger.warning(f"Could not build matchup context for {opp_team_name}: {e}")
        return result

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_opponent_stats(self, season_type: str = 'Regular Season') -> pd.DataFrame:
        """
        Priority 2: Fetch per-team opponent-allowed stats (what each team gives up).
        Regular-season data is cached in SQLite daily; playoff data is in-memory only.
        """
        if season_type in self._opp_stats_cache:
            return self._opp_stats_cache[season_type]

        if season_type == 'Regular Season':
            today = datetime.now().strftime('%Y-%m-%d')
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT date_fetched, data_json FROM opp_stats_cache WHERE season = ?", (self.season,))
                row = cursor.fetchone()
                if row and row[0] == today:
                    import io
                    df = pd.read_json(io.StringIO(row[1]))
                    self._opp_stats_cache[season_type] = df
                    return df

        logger.info(f"Fetching opponent-allowed stats from nba_api [{season_type}]")
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=self.season,
            season_type_all_star=season_type,
            measure_type_detailed_defense='Opponent',
            per_mode_detailed='PerGame'
        )
        time.sleep(0.6)
        df = stats.get_data_frames()[0]

        if season_type == 'Regular Season':
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO opp_stats_cache (season, date_fetched, data_json) VALUES (?, ?, ?)",
                    (self.season, today, df.to_json())
                )

        self._opp_stats_cache[season_type] = df
        return df

    def get_blended_opponent_stats(self) -> pd.DataFrame:
        """Opponent-allowed stats blended per-team with playoff data."""
        if self._blended_opp_cache is not None:
            return self._blended_opp_cache
        rs = self.get_opponent_stats('Regular Season')
        try:
            po = self.get_opponent_stats('Playoffs')
        except Exception as e:
            logger.warning(f"Playoff opponent stats fetch failed, falling back to RS: {e}")
            po = pd.DataFrame()
        self._blended_opp_cache = _blend_playoff_stats(rs, po)
        return self._blended_opp_cache

    # ------------------------------------------------------------------ #
    #  SGP cross-player: identify starting PG and C/PF by team            #
    # ------------------------------------------------------------------ #

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def _get_player_season_stats(self) -> pd.DataFrame:
        """League-wide per-game player stats for the current season. Cached in memory."""
        if self._player_season_cache is not None:
            return self._player_season_cache
        logger.info("Fetching league-wide player season stats (PerGame) for PG/big detection")
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=self.season,
            per_mode_detailed='PerGame',
        )
        time.sleep(0.6)
        df = stats.get_data_frames()[0]
        self._player_season_cache = df
        return df

    def get_team_pg_and_big(self, team_name: str) -> Dict[str, Any]:
        """
        Identify the starting PG (highest AST/game) and C/PF (highest REB/game)
        for the given team among qualified players (>= 15 GP, >= 20 MIN/game).

        Returns:
            {'pg':  {'id': int, 'name': str} or None,
             'big': {'id': int, 'name': str} or None}
        """
        result: Dict[str, Any] = {'pg': None, 'big': None}
        try:
            df = self._get_player_season_stats()
            if df.empty:
                return result

            # Match team by abbreviation or by full-name suffix
            team_lower = team_name.lower()
            team_df = df[df['TEAM_ABBREVIATION'].str.lower() == team_lower]
            if team_df.empty:
                # team_name might be a full name — resolve to abbreviation
                for abbr, full in _ABBR_TO_FULL.items():
                    if full == team_lower or team_lower.split()[-1] in full:
                        team_df = df[df['TEAM_ABBREVIATION'].str.lower() == abbr]
                        break
            if team_df.empty:
                return result

            # Qualified starters: >= 15 GP and >= 20 MIN per game
            qualified = team_df[(team_df['GP'] >= 15) & (team_df['MIN'] >= 20.0)]
            if qualified.empty:
                qualified = team_df[team_df['GP'] >= 10]
            if qualified.empty:
                return result

            # PG = highest AST/game
            pg_row = qualified.nlargest(1, 'AST').iloc[0]
            result['pg'] = {'id': int(pg_row['PLAYER_ID']), 'name': str(pg_row['PLAYER_NAME'])}

            # C/PF = highest REB/game among remaining players
            non_pg = qualified[qualified['PLAYER_ID'] != pg_row['PLAYER_ID']]
            if non_pg.empty:
                return result
            big_row = non_pg.nlargest(1, 'REB').iloc[0]
            result['big'] = {'id': int(big_row['PLAYER_ID']), 'name': str(big_row['PLAYER_NAME'])}

        except Exception as e:
            logger.warning(f"Could not identify PG/big for {team_name}: {e}")
        return result

    # ------------------------------------------------------------------ #
    #  Priority 2: Opponent defensive multiplier per market                #
    # ------------------------------------------------------------------ #

    def get_opponent_def_multiplier(
        self, opp_team_name: str, market: str, playoff_blend: bool = False
    ) -> float:
        """
        Return a defensive multiplier for `market` when playing against `opp_team_name`.
        > 1.0 → opponent allows more than average (weak defense → inflate projection).
        < 1.0 → opponent allows less than average (strong defense → deflate).
        When playoff_blend=True, uses the playoff-blended opponent frame.
        """
        opp_col = _MARKET_OPP_COL.get(market)
        if not opp_col:
            return 1.0  # PRA handled below

        try:
            df = self.get_blended_opponent_stats() if playoff_blend else self.get_opponent_stats()
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

    def get_opponent_def_multiplier_pra(
        self, opp_team_name: str, playoff_blend: bool = False
    ) -> float:
        """Defensive multiplier for PRA (composite of PTS + REB + AST allowed)."""
        try:
            df = self.get_blended_opponent_stats() if playoff_blend else self.get_opponent_stats()
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

    @staticmethod
    def infer_position_group(logs: pd.DataFrame) -> str:
        """
        Infer position group (Guard / Forward / Center) from a player's
        recent game-log stat profile.

        Heuristic — in priority order:
          1. AST ≥ 4.5 /game  → Guard  (primary ball-handlers)
          2. REB ≥ 7.5 /game  → Center (rim anchors)
          3. REB ≥ 4.5 /game  → Forward
          4. AST ≥ 2.5 /game  → Guard  (wing-guard types)
          5. default          → Forward
        """
        if logs is None or logs.empty or len(logs) < 3:
            return 'Forward'
        recent   = logs.head(15)
        avg_ast  = float(recent['AST'].mean()) if 'AST' in recent.columns else 0.0
        avg_reb  = float(recent['REB'].mean()) if 'REB' in recent.columns else 0.0
        if avg_ast >= 4.5:
            return 'Guard'
        if avg_reb >= 7.5:
            return 'Center'
        if avg_reb >= 4.5:
            return 'Forward'
        if avg_ast >= 2.5:
            return 'Guard'
        return 'Forward'

    def get_positional_def_multiplier(
        self, opp_team: str, market: str, position_group: str,
        playoff_blend: bool = False,
    ) -> float:
        """
        Positional defensive efficiency multiplier.

        Combines the team's overall defensive strength for this market (existing
        logic) with a static position factor that encodes how much each position
        group is impacted by that defensive dimension.

        Formula:
            blend   = (1 - BLEND) + BLEND × position_factor
            result  = team_overall_mult × blend
            clamped to [0.70, 1.30]

        Example — Center vs Lakers (strong rebounding defense, OPP_REB mult = 0.82):
            blend  = 0.75 + 0.25 × 1.50 = 1.125
            result = 0.82 × 1.125 = 0.923   ← bigger penalty for the Center

        Example — Guard in same game (REB mult = 0.82):
            blend  = 0.75 + 0.25 × 0.50 = 0.875
            result = 0.82 × 0.875 = 0.718   ← guard barely affected
        """
        # Base team-level multiplier (existing market-level logic)
        if market == 'player_points_rebounds_assists':
            base = self.get_opponent_def_multiplier_pra(opp_team, playoff_blend=playoff_blend)
        else:
            base = self.get_opponent_def_multiplier(opp_team, market, playoff_blend=playoff_blend)

        pf    = _POSITION_FACTORS.get(position_group, {}).get(market, 1.0)
        blend = (1.0 - _POSITION_BLEND) + _POSITION_BLEND * pf
        return round(max(0.70, min(1.30, base * blend)), 4)

    def get_team_win_pct_map(self) -> Dict[str, float]:
        """
        Return {team_name_lower: win_pct} for all teams this season.
        Reuses the in-memory advanced stats cache — no extra API call after the
        first fetch.
        """
        try:
            df = self.get_team_stats()
            if df.empty or 'W_PCT' not in df.columns:
                return {}
            return {
                str(row['TEAM_NAME']).lower(): float(row['W_PCT'])
                for _, row in df.iterrows()
            }
        except Exception as e:
            logger.warning(f"get_team_win_pct_map failed: {e}")
            return {}

    def get_team_pace(
        self, home_team: str, away_team: str, playoff_blend: bool = False
    ) -> Dict[str, float]:
        """Return pace for both teams + data-driven league average."""
        try:
            df = self.get_blended_team_stats() if playoff_blend else self.get_team_stats()
            if df.empty or 'PACE' not in df.columns:
                return {'home_pace': 99.0, 'away_pace': 99.0, 'league_avg': 99.0}

            def _pace(name: str) -> float:
                low = name.lower()
                row = df[df['TEAM_NAME'].str.lower() == low]
                if row.empty:
                    row = df[df['TEAM_NAME'].str.lower().str.contains(low.split()[-1], na=False)]
                return float(row.iloc[0]['PACE']) if not row.empty else 99.0

            league_avg = float(df['PACE'].mean()) if not df['PACE'].isna().all() else 99.0
            return {
                'home_pace': _pace(home_team),
                'away_pace': _pace(away_team),
                'league_avg': league_avg,
            }
        except Exception:
            return {'home_pace': 99.0, 'away_pace': 99.0, 'league_avg': 99.0}

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

    @staticmethod
    def resolve_player_id(player_name: str) -> Optional[int]:
        """
        Return the NBA Stats player ID for a player name using the static
        players lookup (no API call, no rate limiting needed).
        Returns None if the name cannot be resolved.
        """
        try:
            from nba_api.stats.static import players as _static_players
            matches = _static_players.find_players_by_full_name(player_name)
            if matches:
                return int(matches[0]['id'])
        except Exception:
            pass
        return None

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_player_info(self, player_id: int) -> Dict[str, Any]:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        time.sleep(0.6)
        return info.get_dict()

    def get_player_previous_teams(self, player_id: int) -> set:
        """
        Return the set of full team names (lowercased) this player has
        previously played for.

        Sources (zero extra API calls after first run per player):
          1. Previous-season game logs via get_player_game_logs_season —
             cached permanently in backtest_logs_cache.
          2. Current-season game logs (already cached today) — catches
             mid-season trades where some logs show the old team abbreviation.

        Matching against opp_team (full name) works because _ABBR_TO_FULL
        maps every NBA team abbreviation to its lowercased full name.
        """
        if not hasattr(self, '_prev_teams_cache'):
            self._prev_teams_cache: Dict[int, set] = {}
        if player_id in self._prev_teams_cache:
            return self._prev_teams_cache[player_id]

        prev_teams: set = set()

        # ── Previous season logs ────────────────────────────────────────────
        try:
            parts      = self.season.split('-')
            prev_start = int(parts[0]) - 1
            prev_end   = str(int(parts[0]))[-2:]   # "24" from 2024
            prev_season = f"{prev_start}-{prev_end}"
            prev_logs   = self.get_player_game_logs_season(player_id, prev_season)
            if not prev_logs.empty and 'TEAM_ABBREVIATION' in prev_logs.columns:
                for abbr in prev_logs['TEAM_ABBREVIATION'].unique():
                    full = _ABBR_TO_FULL.get(str(abbr).lower(), '')
                    if full:
                        prev_teams.add(full)
        except Exception:
            pass

        # ── Current season: catch mid-season trades ─────────────────────────
        try:
            curr_logs = self.get_player_game_logs(player_id)
            if not curr_logs.empty and 'TEAM_ABBREVIATION' in curr_logs.columns:
                curr_abbr = str(curr_logs.iloc[0]['TEAM_ABBREVIATION']).lower()
                for abbr in curr_logs['TEAM_ABBREVIATION'].unique():
                    abbr_lower = str(abbr).lower()
                    if abbr_lower != curr_abbr:
                        full = _ABBR_TO_FULL.get(abbr_lower, '')
                        if full:
                            prev_teams.add(full)
        except Exception:
            pass

        self._prev_teams_cache[player_id] = prev_teams
        return prev_teams

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_box_score(self, game_id: str) -> pd.DataFrame:
        logger.info(f"Fetching box score for game {game_id}")
        if not str(game_id).startswith('00'):
            logger.warning(f"Game ID {game_id} may not be NBA API format.")
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        time.sleep(0.6)
        return box.get_data_frames()[0]
