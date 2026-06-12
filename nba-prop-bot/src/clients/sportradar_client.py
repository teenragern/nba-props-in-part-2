import os
import time
import json
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class SportradarClient:
    """
    Client for Sportradar REST APIs.
    Designed for trial-key efficiency (1,000 requests/month).
    Supports multiple API families (NBA, Odds Comparison, Synergy, etc.)
    """

    API_VERSIONS = {
        "nba": "v8",
        "oddscomparison-prematch": "v2",
        "oddscomparison-player-props": "v2",
        "oddscomparison-futures": "v2",
        "basketball-nba-synergy": "v3", # typical version for synergy
        "basketball": "v2", # global basketball
    }

    def __init__(self, api_key: Optional[str] = None, access_level: str = "trial"):
        # Map api_family to specific environment variable keys
        self.api_keys = {
            "nba": os.getenv("SPORTRADAR_NBA_API_KEY") or os.getenv("PBP_API_KEY"),
            "oddscomparison-prematch": os.getenv("SPORTRADAR_PREMATCH_API_KEY") or os.getenv("PBP_API_KEY"),
            "oddscomparison-player-props": os.getenv("SPORTRADAR_PROPS_API_KEY") or os.getenv("PBP_API_KEY"),
            "oddscomparison-futures": os.getenv("SPORTRADAR_FUTURES_API_KEY") or os.getenv("PBP_API_KEY"),
            "basketball-nba-synergy": os.getenv("SPORTRADAR_SYNERGY_API_KEY") or os.getenv("PBP_API_KEY"),
            "basketball": os.getenv("SPORTRADAR_GLOBAL_API_KEY") or os.getenv("PBP_API_KEY"),
        }
        
        # Fallback for backwards compatibility if needed
        self.api_key = os.getenv("PBP_API_KEY")
        self.access_level = access_level
        self.base_domain = "https://api.sportradar.com"

    def _get_base_url(self, api_family: str = "nba") -> str:
        version = self.API_VERSIONS.get(api_family, "v2")
        return f"{self.base_domain}/{api_family}/{self.access_level}/{version}/en"

    def _get(self, endpoint: str, params: Optional[Dict] = None, api_family: str = "nba", retries: int = 3) -> Optional[Dict]:
        key_to_use = self.api_keys.get(api_family) or self.api_key
        if not key_to_use:
            logger.warning(f"SportradarClient: No API key found for family {api_family}.")
            return None
            
        url = f"{self._get_base_url(api_family)}/{endpoint}"
        query_params = {"api_key": key_to_use}
        if params:
            query_params.update(params)
            
        try:
            # Respect 1 QPS trial limit
            time.sleep(1.1)
            response = requests.get(url, params=query_params, timeout=10)
            
            if response.status_code == 403:
                logger.error(f"Sportradar: 403 Forbidden. Check API key/level for {api_family}. URL: {url}")
                return None
            if response.status_code == 429:
                if retries <= 0:
                    logger.error("Sportradar: 429 Limit Exceeded. Monthly quota likely exhausted.")
                    return None
                logger.warning(f"Sportradar: 429 Rate Limited. Cooling down... ({retries} retries left)")
                time.sleep(5)
                return self._get(endpoint, params, api_family, retries=retries-1)
                
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Sportradar API error ({api_family}): {e}")
            return None

    # --- NBA API (v8) ---
    def get_daily_schedule(self, date: Optional[str] = None) -> Optional[Dict]:
        """Fetch schedule for a specific date (YYYY/MM/DD)."""
        date_path = date or datetime.now(timezone.utc).strftime("%Y/%m/%d")
        return self._get(f"games/{date_path}/schedule.json", api_family="nba")

    def get_game_summary(self, game_id: str) -> Optional[Dict]:
        """Fetch full game summary (including lineups and rosters)."""
        return self._get(f"games/{game_id}/summary.json", api_family="nba")

    def get_game_boxscore(self, game_id: str) -> Optional[Dict]:
        """Fetch team/player boxscore stats."""
        return self._get(f"games/{game_id}/boxscore.json", api_family="nba")

    def get_game_pbp(self, game_id: str) -> Optional[Dict]:
        """Fetch full play-by-play for a game."""
        return self._get(f"games/{game_id}/pbp.json", api_family="nba")

    def get_season_schedule(self, year: int, season_type: str = "REG") -> Optional[Dict]:
        """Fetch full season schedule (REG, PRE, PST)."""
        return self._get(f"games/{year}/{season_type}/schedule.json", api_family="nba")

    # --- Odds Comparison Prematch API ---
    def get_prematch_odds(self, match_id: str) -> Optional[Dict]:
        """Fetch prematch odds for a specific game."""
        # Clean ID: if it doesn't have the prefix, add it.
        if not match_id.startswith("sr:sport_event:"):
            match_id = f"sr:sport_event:{match_id}"
        return self._get(f"sport_events/{match_id}/sport_event_markets.json", api_family="oddscomparison-prematch")
        
    def get_prematch_schedule(self, date: str) -> Optional[Dict]:
        """Fetch prematch odds schedule for a date (YYYY-MM-DD)."""
        return self._get(f"sports/sr:sport:2/schedules/{date}/schedules.json", api_family="oddscomparison-prematch")

    # --- Odds Comparison Player Props API ---
    def get_player_props_odds(self, match_id: str) -> Optional[Dict]:
        """Fetch player prop odds for a specific game."""
        if not match_id.startswith("sr:sport_event:"):
            match_id = f"sr:sport_event:{match_id}"
        return self._get(f"sport_events/{match_id}/players_props.json", api_family="oddscomparison-player-props")
        
    def get_event_odds_oddsapi_format(self, match_id: str) -> Dict:
        """
        Fetch Odds Comparison data and adapt it to the The-Odds-API format 
        so it can be used as a drop-in fallback.
        """
        # Fetch both prematch and player props if possible
        props_data = self.get_player_props_odds(match_id) or {}
        prematch_data = self.get_prematch_odds(match_id) or {}
        
        props_root = props_data.get('sport_event_players_props', {})
        props_markets = props_root.get('players_props') or props_root.get('players_markets') or []
        
        all_markets = props_markets + prematch_data.get('markets', [])
        
        if not all_markets:
            return {"id": match_id, "bookmakers": []}
            
        # Basic translation logic (best effort mapping from Sportradar -> OddsAPI)
        bookmakers_map = {}
        
        for market in all_markets:
            mkt_name = market.get('name', '').lower()
            # map sportradar market names to odds api keys
            mkt_key = mkt_name
            if 'point' in mkt_name and 'player' in mkt_name: mkt_key = 'player_points'
            elif 'rebound' in mkt_name: mkt_key = 'player_rebounds'
            elif 'assist' in mkt_name: mkt_key = 'player_assists'
            elif 'spread' in mkt_name or 'handicap' in mkt_name: mkt_key = 'spreads'
            elif 'total' in mkt_name: mkt_key = 'totals'
            elif 'moneyline' in mkt_name or '2-way' in mkt_name or 'winner' in mkt_name: mkt_key = 'h2h'
            
            for book in market.get('books', []):
                b_name = book.get('name')
                b_id = b_name.lower().replace(' ', '')
                if b_id not in bookmakers_map:
                    bookmakers_map[b_id] = {
                        "key": b_id,
                        "title": b_name,
                        "markets": []
                    }
                
                outcomes = []
                for outcome in book.get('outcomes', []):
                    # sportradar outcome structure varies, we try to extract point/price
                    desc = outcome.get('description', '')
                    point = outcome.get('total') or outcome.get('handicap') or outcome.get('line')
                    name = outcome.get('name') or outcome.get('type') or ''
                    name = str(name).title()
                    if 'Over' in name: name = 'Over'
                    elif 'Under' in name: name = 'Under'
                    
                    price = outcome.get('odds_decimal') or outcome.get('odds') or 1.0
                    o_dict = {"name": name, "price": float(price)}
                    if point is not None: o_dict["point"] = float(point)
                    if desc: o_dict["description"] = desc
                    outcomes.append(o_dict)
                    
                bookmakers_map[b_id]["markets"].append({
                    "key": mkt_key,
                    "outcomes": outcomes
                })
                
        return {"id": match_id, "bookmakers": list(bookmakers_map.values())}

    def get_player_logo_url(self, player_id: str) -> str:
        """
        Construct a logo URL for a player.
        Often these don't require an API key to view if they are public assets,
        but typically we fetch from the player props logo endpoint or NBA API.
        """
        # Telegram just needs a URL string
        return f"{self.base_domain}/oddscomparison-player-props/{self.access_level}/v2/en/players/{player_id}/logo"

    # --- Synergy Basketball NBA API ---
    def get_synergy_player_stats(self, player_id: str) -> Optional[Dict]:
        """Fetch advanced Synergy play-type stats for a player."""
        return self._get(f"players/{player_id}/statistics.json", api_family="basketball-nba-synergy")

    def get_synergy_team_stats(self, team_id: str) -> Optional[Dict]:
        """Fetch advanced Synergy play-type stats for a team."""
        return self._get(f"teams/{team_id}/statistics.json", api_family="basketball-nba-synergy")

    # --- Global Basketball API ---
    def get_global_basketball_schedule(self, date: str) -> Optional[Dict]:
        """Fetch global basketball schedule."""
        return self._get(f"schedules/{date}/schedule.json", api_family="basketball")
        
    # --- Odds Comparison Futures API ---
    def get_futures_markets(self, season_id: str) -> Optional[Dict]:
        """Fetch futures markets for a season (e.g. NBA Championship, MVP)."""
        return self._get(f"tournaments/{season_id}/markets.json", api_family="oddscomparison-futures")
