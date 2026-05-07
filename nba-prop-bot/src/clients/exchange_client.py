"""
Exchange API client — Phase 3C.

Thin wrapper around the Kalshi REST API (v2) for fetching NBA prop market
prices. Paper-mode only initially — no order placement.

Kalshi markets relevant to NBA props follow the slug pattern:
    NBAPROP-{PLAYER_ID}-{STAT}-{DATE}

Environment:
    KALSHI_API_KEY      Kalshi API key (enables this client)
    KALSHI_API_SECRET   Kalshi API secret (used for HMAC request signing)
    KALSHI_BASE_URL     default: https://trading-api.kalshi.com/trade-api/v2

Falls back gracefully when KALSHI_API_KEY is unset — all methods return
empty results so callers treat exchange data as optional enrichment.

Usage:
    from src.clients.exchange_client import ExchangeClient
    client = ExchangeClient()
    markets = client.get_nba_prop_markets()          # all active NBA props
    price   = client.get_market_price("NBAPROP-...")  # yes/no price for one market
"""

import hashlib
import hmac
import os
import time
from typing import Any, Dict, List, Optional

import requests

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_DEFAULT_BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"
_REQUEST_TIMEOUT  = 15


class ExchangeClient:
    """
    Read-only Kalshi REST client. All writes (order placement) are intentionally
    absent — this client is for price discovery and paper-mode arbitrage only.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._api_key    = api_key    or os.getenv("KALSHI_API_KEY",    "")
        self._api_secret = api_secret or os.getenv("KALSHI_API_SECRET", "")
        self._base_url   = (base_url  or os.getenv("KALSHI_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")
        self._enabled    = bool(self._api_key)
        self._session    = requests.Session()

        if not self._enabled:
            logger.info("ExchangeClient: KALSHI_API_KEY not set — disabled.")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Public API ────────────────────────────────────────────────────────────

    def get_nba_prop_markets(
        self,
        limit: int = 200,
        status: str = "open",
    ) -> List[Dict[str, Any]]:
        """
        Return all open Kalshi NBA prop markets.

        Each market dict includes:
            ticker, title, yes_bid, yes_ask, no_bid, no_ask,
            volume, open_interest, close_time, result
        """
        if not self._enabled:
            return []

        # Kalshi uses series_ticker prefix filter for NBA props
        params = {
            "limit":         limit,
            "status":        status,
            "series_ticker": "NBAPROP",
        }
        data = self._get("/markets", params)
        return data.get("markets", []) if data else []

    def get_market_price(self, ticker: str) -> Optional[Dict[str, float]]:
        """
        Return the current bid/ask for a single market.

        Returns:
            {"yes_bid": float, "yes_ask": float, "no_bid": float, "no_ask": float,
             "implied_yes": float}   (mid-price implied probability 0–1)
        or None on error / when disabled.
        """
        if not self._enabled:
            return None

        data = self._get(f"/markets/{ticker}", {})
        if not data:
            return None

        market = data.get("market", {})
        yes_bid = market.get("yes_bid", 0) / 100.0   # Kalshi prices are cents (0–100)
        yes_ask = market.get("yes_ask", 0) / 100.0
        no_bid  = market.get("no_bid",  0) / 100.0
        no_ask  = market.get("no_ask",  0) / 100.0

        implied_yes = (yes_bid + yes_ask) / 2.0 if (yes_bid + yes_ask) > 0 else 0.0

        return {
            "yes_bid":     yes_bid,
            "yes_ask":     yes_ask,
            "no_bid":      no_bid,
            "no_ask":      no_ask,
            "implied_yes": implied_yes,
        }

    def search_markets(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Full-text search across Kalshi market titles.
        Useful for fuzzy-matching player names to market tickers.
        """
        if not self._enabled:
            return []

        data = self._get("/markets", {"limit": limit, "search": query, "status": "open"})
        return data.get("markets", []) if data else []

    # ── Internal ──────────────────────────────────────────────────────────────

    def _headers(self, method: str, path: str) -> Dict[str, str]:
        """
        Build Kalshi HMAC-signed request headers.
        If no API secret is configured, returns key-only headers (read-only endpoints
        on some environments do not require signing).
        """
        ts = str(int(time.time() * 1000))
        headers = {
            "Content-Type":  "application/json",
            "Kalshi-API-Key": self._api_key,
            "Kalshi-Timestamp": ts,
        }
        if self._api_secret:
            msg       = ts + method.upper() + path
            signature = hmac.new(
                self._api_secret.encode(),
                msg.encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["Kalshi-Signature"] = signature
        return headers

    def _get(self, path: str, params: Dict[str, Any]) -> Optional[Dict]:
        url = f"{self._base_url}{path}"
        try:
            r = self._session.get(
                url,
                params=params,
                headers=self._headers("GET", path),
                timeout=_REQUEST_TIMEOUT,
            )
            if r.status_code == 401:
                logger.warning("ExchangeClient: authentication failed — check KALSHI_API_KEY.")
                return None
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            logger.warning(f"ExchangeClient: connection error reaching {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"ExchangeClient: timeout on {url}")
            return None
        except Exception as e:
            logger.warning(f"ExchangeClient: request failed {url}: {e}")
            return None
