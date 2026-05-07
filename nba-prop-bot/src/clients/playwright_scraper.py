"""
Playwright Scraper HTTP client — Phase 3B.

Thin client that calls the external `scraper-worker` Railway service, which
runs Playwright inside its own container (browser binaries are too large for
the main bot image).

Falls back gracefully when SCRAPER_SERVICE_URL is unset — callers receive
None / empty results, identical to a no-data API response.

Environment:
    SCRAPER_SERVICE_URL   https://scraper-worker.railway.internal  (required)
    SCRAPER_TIMEOUT_SEC   default 30

Usage:
    from src.clients.playwright_scraper import PlaywrightScraperClient
    client = PlaywrightScraperClient()
    lines = client.scrape_book_props("draftkings", "LeBron James")
    # Returns list of {"market": str, "line": float, "over_odds": float, ...}
    # Returns [] when scraper service is unreachable or disabled.
"""

import os
from typing import Any, Dict, List, Optional

import requests

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_DEFAULT_TIMEOUT = int(os.getenv("SCRAPER_TIMEOUT_SEC", "30"))


class PlaywrightScraperClient:
    """
    HTTP wrapper for the scraper-worker microservice.

    All methods return empty lists / None on failure so callers
    can treat the scraper as an optional data enrichment layer.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = _DEFAULT_TIMEOUT):
        self._base_url = (base_url or os.getenv("SCRAPER_SERVICE_URL", "")).rstrip("/")
        self._timeout  = timeout
        self._enabled  = bool(self._base_url)

        if not self._enabled:
            logger.info("PlaywrightScraper: SCRAPER_SERVICE_URL not set — disabled.")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Public API ────────────────────────────────────────────────────────────

    def scrape_book_props(
        self,
        book: str,
        player_name: str,
        markets: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Scrape prop lines for a player from a specific book.

        Args:
            book:        e.g. "draftkings", "fanduel", "betmgm"
            player_name: Full player name, e.g. "LeBron James"
            markets:     List of markets, e.g. ["player_points", "player_rebounds"]
                         If None, the scraper returns all available markets.

        Returns:
            List of dicts:
                {"market": str, "line": float, "over_odds": float, "under_odds": float}
            Empty list on error or when disabled.
        """
        if not self._enabled:
            return []

        params: Dict[str, Any] = {"player": player_name, "book": book}
        if markets:
            params["markets"] = ",".join(markets)

        return self._get("/scrape/props", params, default=[])

    def scrape_game_markets(
        self,
        book: str,
        home_team: str,
        away_team: str,
    ) -> List[Dict[str, Any]]:
        """
        Scrape game-level markets (spread, total, ML) for a matchup.

        Returns:
            List of dicts:
                {"market": str, "line": float, "over_odds": float, "under_odds": float}
        """
        if not self._enabled:
            return []

        params = {"book": book, "home": home_team, "away": away_team}
        return self._get("/scrape/game", params, default=[])

    def health(self) -> bool:
        """Return True if the scraper service is reachable."""
        if not self._enabled:
            return False
        try:
            r = requests.get(f"{self._base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get(self, path: str, params: Dict[str, Any], default: Any) -> Any:
        url = f"{self._base_url}{path}"
        try:
            r = requests.get(url, params=params, timeout=self._timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            logger.warning(f"PlaywrightScraper: service unreachable at {url}")
            return default
        except requests.exceptions.Timeout:
            logger.warning(f"PlaywrightScraper: timeout on {url}")
            return default
        except Exception as e:
            logger.warning(f"PlaywrightScraper: request failed {url}: {e}")
            return default
