import requests
from typing import List, Dict, Any
from src.config import ODDS_API_KEY, ODDS_REGION, BOOKMAKERS, SHARP_BOOKS
from src.utils.retry import retry_with_backoff
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class OddsApiClient:
    BASE_URL = "https://api.the-odds-api.com/v4/sports"
    SPORT = "basketball_nba"

    def __init__(self, api_key: str = ODDS_API_KEY):
        self.api_key = api_key
        self.requests_used = 0
        self.requests_remaining = 0

    def _update_quota(self, headers: Any):
        used = headers.get('x-requests-used')
        remaining = headers.get('x-requests-remaining')
        if used is not None:
            self.requests_used = int(used)
        if remaining is not None:
            self.requests_remaining = int(remaining)
        logger.debug(f"Odds API Quota - Used: {self.requests_used}, Remaining: {self.requests_remaining}")

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_events(self) -> List[Dict[str, Any]]:
        url = f"{self.BASE_URL}/{self.SPORT}/events"
        params = {
            "apiKey": self.api_key
        }
        logger.info("Fetching NBA events from Odds API")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        self._update_quota(response.headers)
        return response.json()

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def get_event_odds(self, event_id: str, markets: List[str]) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/{self.SPORT}/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": ODDS_REGION,
            "markets": ",".join(markets),
            "bookmakers": ",".join(list(dict.fromkeys(BOOKMAKERS + SHARP_BOOKS))) or None,
            "oddsFormat": "decimal"
        }
        params = {k: v for k, v in params.items() if v is not None}
        
        logger.info(f"Fetching odds for event {event_id} markets: {markets}")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        self._update_quota(response.headers)
        return response.json()
