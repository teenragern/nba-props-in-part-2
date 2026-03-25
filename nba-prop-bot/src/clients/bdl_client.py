"""
BallDontLie GOAT-tier API client.

Replaces multiple fragile data sources with a single reliable API:
  - Player props (live)     → replaces Odds API prop scanning (saves credits)
  - Player injuries         → replaces Rotowire/CBS scraping
  - Lineups (confirmed)     → replaces infer_starter_flag guessing
  - Advanced stats V2       → feeds XGBoost with usage%, pace, tracking
  - Game player stats       → supplements nba_api game logs
  - Betting odds (spreads/totals) → cross-references Odds API
  - Play-by-play            → replaces nba_api PBP (faster, no rate limit)

Rate limit: 600 req/min on GOAT tier.
Auth: API key in Authorization header.
Pagination: cursor-based (next_cursor in meta).

Usage:
    from src.clients.bdl_client import BDLClient
    bdl = BDLClient()
    props = bdl.get_player_props(game_id=18447073)
    injuries = bdl.get_injuries()
"""

import time
import requests
from typing import Any, Dict, List, Optional, Tuple
from src.config import BDL_API_KEY
from src.utils.retry import retry_with_backoff
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_BASE_V1 = "https://api.balldontlie.io/v1"
_BASE_V2 = "https://api.balldontlie.io/v2"
_BASE_NBA_V1 = "https://api.balldontlie.io/nba/v1"
_BASE_NBA_V2 = "https://api.balldontlie.io/nba/v2"


class BDLClient:
    """BallDontLie GOAT-tier API client."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or BDL_API_KEY
        if not self.api_key:
            logger.warning("BDL_API_KEY not set — BDL client will fail on all calls.")
        self._session = requests.Session()
        self._session.headers.update({"Authorization": self.api_key or ""})
        self._request_count = 0

    # ── HTTP helpers ─────────────────────────────────────────────────────

    @retry_with_backoff(retries=3, backoff_in_seconds=1)
    def _get(self, url: str, params: dict = None, timeout: int = 15) -> dict:
        """Execute GET request with retry and rate-limit awareness."""
        resp = self._session.get(url, params=params, timeout=timeout)
        self._request_count += 1

        if resp.status_code == 429:
            logger.warning("BDL rate limit hit — backing off 5s")
            time.sleep(5)
            resp = self._session.get(url, params=params, timeout=timeout)

        resp.raise_for_status()
        return resp.json()

    def _get_all_pages(self, url: str, params: dict = None,
                       max_pages: int = 20) -> List[dict]:
        """Fetch all pages of a paginated endpoint."""
        params = dict(params or {})
        params.setdefault("per_page", 100)
        all_data = []

        for _ in range(max_pages):
            result = self._get(url, params)
            data = result.get("data", [])
            all_data.extend(data)

            meta = result.get("meta", {})
            next_cursor = meta.get("next_cursor")
            if not next_cursor or not data:
                break
            params["cursor"] = next_cursor

        return all_data

    # ── Player Props (LIVE) ──────────────────────────────────────────────

    def get_player_props(
        self,
        game_id: int,
        player_id: int = None,
        prop_type: str = None,
        vendors: List[str] = None,
    ) -> List[dict]:
        """
        Fetch live player props for a game.

        Returns all props in a single response (no pagination).
        Each prop has: player_id, vendor, prop_type, line_value, market
        (over_under with over_odds/under_odds, or milestone with odds).
        """
        params: dict = {"game_id": game_id}
        if player_id:
            params["player_id"] = player_id
        if prop_type:
            params["prop_type"] = prop_type
        if vendors:
            for i, v in enumerate(vendors):
                params[f"vendors[{i}]"] = v

        try:
            result = self._get(f"{_BASE_V2}/odds/player_props", params)
            props = result.get("data", [])
            logger.info(f"BDL: {len(props)} player props for game {game_id}")
            return props
        except Exception as e:
            logger.warning(f"BDL player props failed for game {game_id}: {e}")
            return []

    # ── Player Injuries ──────────────────────────────────────────────────

    def get_injuries(
        self,
        team_ids: List[int] = None,
        player_ids: List[int] = None,
    ) -> List[dict]:
        """
        Fetch all current player injuries.

        Returns: [{player: {id, first_name, last_name, ...},
                   status, return_date, description}, ...]
        """
        params: dict = {"per_page": 100}
        if team_ids:
            for i, tid in enumerate(team_ids):
                params[f"team_ids[{i}]"] = tid
        if player_ids:
            for i, pid in enumerate(player_ids):
                params[f"player_ids[{i}]"] = pid

        try:
            injuries = self._get_all_pages(f"{_BASE_V1}/player_injuries", params)
            logger.info(f"BDL: {len(injuries)} injury records fetched")
            return injuries
        except Exception as e:
            logger.warning(f"BDL injuries fetch failed: {e}")
            return []

    # ── Lineups (confirmed starters) ─────────────────────────────────────

    def get_lineups(self, game_ids: List[int]) -> List[dict]:
        """
        Fetch confirmed lineups for games (available once game begins).

        Returns: [{id, game_id, starter: bool, position, player: {...},
                   team: {...}}, ...]
        """
        if not game_ids:
            return []

        params: dict = {"per_page": 100}
        for i, gid in enumerate(game_ids):
            params[f"game_ids[{i}]"] = gid

        try:
            lineups = self._get_all_pages(f"{_BASE_V1}/lineups", params)
            logger.info(f"BDL: {len(lineups)} lineup entries for {len(game_ids)} games")
            return lineups
        except Exception as e:
            logger.warning(f"BDL lineups fetch failed: {e}")
            return []

    def get_starters_for_game(self, game_id: int) -> Dict[str, List[dict]]:
        """
        Return confirmed starters grouped by team abbreviation.

        Returns: {team_abbr: [{player_id, name, position}, ...]}
        """
        lineups = self.get_lineups([game_id])
        result: Dict[str, List[dict]] = {}
        for entry in lineups:
            if not entry.get("starter"):
                continue
            team = entry.get("team", {})
            abbr = team.get("abbreviation", "UNK")
            player = entry.get("player", {})
            if abbr not in result:
                result[abbr] = []
            result[abbr].append({
                "player_id": player.get("id"),
                "name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                "position": entry.get("position", ""),
                "bdl_player_id": player.get("id"),
            })
        return result

    # ── Games ────────────────────────────────────────────────────────────

    def get_games_by_date(self, date: str) -> List[dict]:
        """
        Fetch all games for a date (YYYY-MM-DD).
        Games include real-time scores, quarter scores, and status.
        """
        params = {"dates[]": date, "per_page": 100}
        try:
            result = self._get(f"{_BASE_V1}/games", params)
            games = result.get("data", [])
            logger.info(f"BDL: {len(games)} games on {date}")
            return games
        except Exception as e:
            logger.warning(f"BDL games fetch failed for {date}: {e}")
            return []

    def get_game(self, game_id: int) -> Optional[dict]:
        """Fetch a single game by ID."""
        try:
            result = self._get(f"{_BASE_V1}/games/{game_id}")
            return result.get("data")
        except Exception:
            return None

    # ── Game Player Stats ────────────────────────────────────────────────

    def get_game_stats(
        self,
        game_ids: List[int] = None,
        player_ids: List[int] = None,
        dates: List[str] = None,
        seasons: List[int] = None,
        start_date: str = None,
        end_date: str = None,
    ) -> List[dict]:
        """
        Fetch per-game player box score stats.

        Each record has: min, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct,
        ftm, fta, ft_pct, oreb, dreb, reb, ast, stl, blk, turnover,
        pf, pts, plus_minus, player, team, game.
        """
        params: dict = {"per_page": 100}
        if game_ids:
            for i, gid in enumerate(game_ids):
                params[f"game_ids[{i}]"] = gid
        if player_ids:
            for i, pid in enumerate(player_ids):
                params[f"player_ids[{i}]"] = pid
        if dates:
            for i, d in enumerate(dates):
                params[f"dates[{i}]"] = d
        if seasons:
            for i, s in enumerate(seasons):
                params[f"seasons[{i}]"] = s
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        try:
            return self._get_all_pages(f"{_BASE_V1}/stats", params)
        except Exception as e:
            logger.warning(f"BDL game stats fetch failed: {e}")
            return []

    # ── Season Averages ──────────────────────────────────────────────────

    def get_season_averages(
        self,
        season: int,
        player_ids: List[int] = None,
        category: str = "general",
        stat_type: str = "base",
        season_type: str = "regular",
    ) -> List[dict]:
        """
        Fetch season averages with rich category/type combos.

        Key categories for prop modeling:
          - general/usage    → usage_percentage, pct_fga, pct_pts
          - general/advanced → off_rating, def_rating, pace, ts%
          - tracking/drives  → drives per game
          - tracking/passing → potential assists, secondary assists
          - playtype/prballhandler → PnR ball handler frequency
          - playtype/prrollman     → PnR roll man frequency
        """
        params: dict = {
            "season": season,
            "season_type": season_type,
            "per_page": 100,
        }
        if stat_type and category != "hustle":
            params["type"] = stat_type
        if player_ids:
            for i, pid in enumerate(player_ids):
                params[f"player_ids[{i}]"] = pid

        try:
            return self._get_all_pages(
                f"{_BASE_NBA_V1}/season_averages/{category}", params
            )
        except Exception as e:
            logger.warning(f"BDL season averages failed ({category}/{stat_type}): {e}")
            return []

    # ── Team Season Averages ─────────────────────────────────────────────

    def get_team_season_averages(
        self,
        season: int,
        category: str = "general",
        stat_type: str = "opponent",
        season_type: str = "regular",
        team_ids: List[int] = None,
    ) -> List[dict]:
        """
        Fetch team-level season averages.

        Key combos for prop modeling:
          - general/opponent → OPP stats (what they give up)
          - general/base     → team pace, pts, reb, ast
          - general/advanced → off_rating, def_rating, pace, net_rating
          - general/defense  → blocks, steals, contested shots
        """
        params: dict = {
            "season": season,
            "season_type": season_type,
            "per_page": 100,
        }
        if stat_type and category != "hustle":
            params["type"] = stat_type
        if team_ids:
            for i, tid in enumerate(team_ids):
                params[f"team_ids[{i}]"] = tid

        try:
            return self._get_all_pages(
                f"{_BASE_NBA_V1}/team_season_averages/{category}", params
            )
        except Exception as e:
            logger.warning(f"BDL team season averages failed ({category}/{stat_type}): {e}")
            return []

    # ── Advanced Stats V2 (per-game) ─────────────────────────────────────

    def get_advanced_stats(
        self,
        game_ids: List[int] = None,
        player_ids: List[int] = None,
        seasons: List[int] = None,
        dates: List[str] = None,
        period: int = 0,
    ) -> List[dict]:
        """
        Fetch V2 advanced stats: hustle, tracking, defensive matchups, scoring.

        Key fields for prop modeling:
          - usage_percentage  → true usage rate (better than our proxy)
          - touches           → ball-handling volume
          - contested_fg_pct  → shot quality signal
          - speed, distance   → fatigue / effort proxy
          - deflections       → steals/blocks predictor
          - points_paint      → inside scoring tendency
          - matchup_fg_pct    → defensive matchup quality
        """
        params: dict = {"per_page": 100, "period": period}
        if game_ids:
            for i, gid in enumerate(game_ids):
                params[f"game_ids[{i}]"] = gid
        if player_ids:
            for i, pid in enumerate(player_ids):
                params[f"player_ids[{i}]"] = pid
        if seasons:
            for i, s in enumerate(seasons):
                params[f"seasons[{i}]"] = s
        if dates:
            for i, d in enumerate(dates):
                params[f"dates[{i}]"] = d

        try:
            return self._get_all_pages(f"{_BASE_NBA_V2}/stats/advanced", params)
        except Exception as e:
            logger.warning(f"BDL advanced stats V2 failed: {e}")
            return []

    # ── Betting Odds (spreads/totals) ────────────────────────────────────

    def get_betting_odds(
        self,
        dates: List[str] = None,
        game_ids: List[int] = None,
    ) -> List[dict]:
        """
        Fetch game-level betting odds (spreads, totals, moneylines).

        Multiple vendors per game: DraftKings, FanDuel, Caesars,
        BetMGM, Bet365, etc.
        """
        params: dict = {"per_page": 100}
        if dates:
            for i, d in enumerate(dates):
                params[f"dates[{i}]"] = d
        if game_ids:
            for i, gid in enumerate(game_ids):
                params[f"game_ids[{i}]"] = gid

        try:
            return self._get_all_pages(f"{_BASE_V2}/odds", params)
        except Exception as e:
            logger.warning(f"BDL betting odds failed: {e}")
            return []

    # ── Play-by-Play ─────────────────────────────────────────────────────

    def get_plays(self, game_id: int) -> List[dict]:
        """
        Fetch play-by-play data for a game.

        Each play: game_id, order, type, text, home_score, away_score,
        period, clock, scoring_play, shooting_play, score_value, team,
        coordinate_x, coordinate_y, participants.
        """
        try:
            result = self._get(
                f"{_BASE_V1}/plays",
                params={"game_id": game_id},
            )
            plays = result.get("data", [])
            logger.debug(f"BDL: {len(plays)} plays for game {game_id}")
            return plays
        except Exception as e:
            logger.warning(f"BDL plays fetch failed for game {game_id}: {e}")
            return []

    # ── Players ──────────────────────────────────────────────────────────

    def search_player(self, name: str) -> Optional[dict]:
        """Search for a player by name. Returns first match or None."""
        try:
            result = self._get(
                f"{_BASE_V1}/players",
                params={"search": name, "per_page": 5},
            )
            data = result.get("data", [])
            return data[0] if data else None
        except Exception:
            return None

    def get_active_players(self) -> List[dict]:
        """Fetch all active NBA players."""
        try:
            return self._get_all_pages(f"{_BASE_V1}/players/active")
        except Exception as e:
            logger.warning(f"BDL active players failed: {e}")
            return []

    # ── Teams ────────────────────────────────────────────────────────────

    def get_teams(self) -> List[dict]:
        """Fetch all NBA teams."""
        try:
            result = self._get(f"{_BASE_V1}/teams")
            return result.get("data", [])
        except Exception:
            return []

    # ── Standings ─────────────────────────────────────────────────────────

    def get_standings(self, season: int) -> List[dict]:
        """Fetch team standings for a season."""
        try:
            result = self._get(
                f"{_BASE_V1}/standings",
                params={"season": season},
            )
            return result.get("data", [])
        except Exception:
            return []

    # ── Convenience: BDL prop format → our internal format ───────────────

    @staticmethod
    def normalize_prop_type(bdl_prop_type: str) -> Optional[str]:
        """Map BDL prop_type strings to our PROP_MARKETS keys."""
        _MAP = {
            "points":                   "player_points",
            "rebounds":                 "player_rebounds",
            "assists":                  "player_assists",
            "threes":                   "player_threes",
            "points_rebounds_assists":  "player_points_rebounds_assists",
            "blocks":                   "player_blocks",
            "steals":                   "player_steals",
            "points_rebounds":          None,  # not in our markets
            "points_assists":           None,
            "rebounds_assists":         None,
            "double_double":            None,
            "triple_double":            None,
        }
        return _MAP.get(bdl_prop_type)

    @staticmethod
    def normalize_injury_status(bdl_status: str) -> str:
        """Map BDL injury status to our standard status strings."""
        if not bdl_status:
            return "Unknown"
        s = bdl_status.lower()
        if "out" in s:
            return "Out"
        if "doubtful" in s:
            return "Doubtful"
        if "questionable" in s or "game time" in s or "gtd" in s:
            return "Questionable"
        if "probable" in s:
            return "Probable"
        if "day-to-day" in s or "dtd" in s:
            return "Questionable"
        return "Unknown"

    @staticmethod
    def american_to_decimal(american: int) -> float:
        """Convert American odds to decimal odds."""
        if american > 0:
            return 1.0 + (american / 100.0)
        elif american < 0:
            return 1.0 + (100.0 / abs(american))
        return 2.0  # even money fallback

    def extract_props_for_scan(
        self, game_id: int, vendors: List[str] = None
    ) -> List[dict]:
        """
        Fetch and normalize BDL player props into scan-ready format.

        Returns list of dicts, each representing one over/under line:
        {
            player_name, bdl_player_id, market, line, side,
            book, decimal_odds, vendor
        }

        Filters to over_under markets only (skips milestones).
        """
        raw = self.get_player_props(game_id, vendors=vendors)
        normalized = []

        for prop in raw:
            market_data = prop.get("market", {})
            if market_data.get("type") != "over_under":
                continue

            our_market = self.normalize_prop_type(prop.get("prop_type", ""))
            if not our_market:
                continue

            player_id = prop.get("player_id")
            vendor = prop.get("vendor", "")
            line = float(prop.get("line_value", 0))
            over_odds = market_data.get("over_odds")
            under_odds = market_data.get("under_odds")

            if over_odds is not None:
                normalized.append({
                    "bdl_player_id": player_id,
                    "market": our_market,
                    "line": line,
                    "side": "OVER",
                    "book": vendor,
                    "decimal_odds": self.american_to_decimal(over_odds),
                    "american_odds": over_odds,
                    "vendor": vendor,
                })
            if under_odds is not None:
                normalized.append({
                    "bdl_player_id": player_id,
                    "market": our_market,
                    "line": line,
                    "side": "UNDER",
                    "book": vendor,
                    "decimal_odds": self.american_to_decimal(under_odds),
                    "american_odds": under_odds,
                    "vendor": vendor,
                })

        logger.info(
            f"BDL: {len(normalized)} normalized prop lines "
            f"from {len(raw)} raw props (game {game_id})"
        )
        return normalized

    @property
    def requests_made(self) -> int:
        return self._request_count
