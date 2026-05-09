"""
Exchange API client — Phase 3C.

Kalshi REST API (v2) wrapper for reading NBA prop market prices and
placing live limit orders.

Environment:
    KALSHI_API_KEY      Kalshi API key (enables this client)
    KALSHI_API_SECRET   RSA private key (PEM or base64)
    KALSHI_BASE_URL     default: https://api.elections.kalshi.com/trade-api/v2
    KALSHI_MAX_STAKE    Max dollars per order (default 5.00)
    KALSHI_MAX_ORDERS_PER_DAY  Daily order cap (default 10)

Falls back gracefully when KALSHI_API_KEY is unset.
"""

import json
import os
import time
from datetime import date
from typing import Any, Dict, List, Optional

import requests

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
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
        # Individual player prop series on Kalshi:
        #   KXNBAPTS  — points    KXNBAREB — rebounds
        #   KXNBAAST  — assists   KXNBA3PT — 3-pointers
        all_markets = []
        for series in ("KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBA3PT"):
            params = {"limit": limit, "status": status, "series_ticker": series}
            data = self._get("/markets", params)
            all_markets.extend(data.get("markets", []) if data else [])
        return all_markets

    def get_nba_game_markets(
        self,
        limit: int = 200,
        status: str = "open",
    ) -> List[Dict[str, Any]]:
        """Return all open Kalshi NBA game-winner markets (series KXNBAGAME)."""
        if not self._enabled:
            return []
        data = self._get("/markets", {"limit": limit, "status": status, "series_ticker": "KXNBAGAME"})
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

    def place_order(
        self,
        ticker: str,
        side: str,
        limit_price_cents: int,
        max_stake_dollars: float = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Place a live limit order on Kalshi.

        Args:
            ticker:              Kalshi market ticker, e.g. "KXNBAPTS-..."
            side:                "yes" (OVER) or "no" (UNDER)
            limit_price_cents:   Price in cents (1–99), e.g. 55 = 55¢
            max_stake_dollars:   Max dollars to spend. Defaults to KALSHI_MAX_STAKE env var.

        Returns:
            Order response dict on success, None on failure.
        """
        if not self._enabled:
            return None

        max_stake = max_stake_dollars or float(os.getenv("KALSHI_MAX_STAKE", "5.00"))
        price_frac = limit_price_cents / 100.0
        if price_frac <= 0:
            return None

        # Number of $1-face-value contracts we can afford
        count = max(1, int(max_stake / price_frac))

        body = {
            "ticker":     ticker,
            "action":     "buy",
            "side":       side,
            "count":      count,
            "type":       "limit",
            f"{side}_price": limit_price_cents,
        }

        logger.info(
            f"ExchangeClient: placing order — {ticker} {side.upper()} "
            f"{limit_price_cents}¢ x {count} contracts (${count * price_frac:.2f})"
        )
        return self._post("/portfolio/orders", body)

    def sell_position(
        self,
        ticker: str,
        side: str,
        contracts: int,
        min_price_cents: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Sell an existing position on Kalshi.

        Args:
            ticker:           Kalshi market ticker, e.g. "KXNBAGAME-..."
            side:             "yes" or "no" — same side as the original buy
            contracts:        Exact number of contracts to sell (from live_positions)
            min_price_cents:  Floor limit in cents (1–99). Use max(1, bid_cents - 1)
                              for a quick fill while guarding against bad executions.

        Returns:
            Order response dict on success, None on failure.
        """
        if not self._enabled:
            return None

        if contracts <= 0 or not (1 <= min_price_cents <= 99):
            logger.warning(
                f"ExchangeClient.sell_position: invalid args "
                f"contracts={contracts} min_price_cents={min_price_cents}"
            )
            return None

        body = {
            "ticker":           ticker,
            "action":           "sell",
            "side":             side,
            "count":            contracts,
            "type":             "limit",
            f"{side}_price":    min_price_cents,
        }

        logger.info(
            f"ExchangeClient: selling position — {ticker} {side.upper()} "
            f"{min_price_cents}¢ x {contracts} contracts"
        )
        return self._post("/portfolio/orders", body)

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
        Build Kalshi RSA-signed request headers (API v2).
        Signature = base64(RSA-SHA256(timestamp + method.upper() + path))
        """
        import base64
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        ts = str(int(time.time() * 1000))
        headers = {
            "Content-Type":       "application/json",
            "KALSHI-ACCESS-KEY":  self._api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }
        if self._api_secret:
            msg = (ts + method.upper() + path).encode()
            # Load the PEM private key — handle both full PEM and raw base64 blobs
            secret = self._api_secret.strip()
            if not secret.startswith("-----"):
                secret = f"-----BEGIN RSA PRIVATE KEY-----\n{secret}\n-----END RSA PRIVATE KEY-----"
            private_key = serialization.load_pem_private_key(secret.encode(), password=None)
            sig = private_key.sign(msg, padding.PKCS1v15(), hashes.SHA256())
            headers["KALSHI-ACCESS-SIGNATURE"] = base64.b64encode(sig).decode()
        return headers

    def _post(self, path: str, body: Dict[str, Any]) -> Optional[Dict]:
        url = f"{self._base_url}{path}"
        try:
            r = self._session.post(
                url,
                data=json.dumps(body),
                headers=self._headers("POST", path),
                timeout=_REQUEST_TIMEOUT,
            )
            if r.status_code == 401:
                logger.warning("ExchangeClient: authentication failed on POST.")
                return None
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            logger.warning(f"ExchangeClient: connection error on POST {url}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"ExchangeClient: timeout on POST {url}")
            return None
        except Exception as e:
            logger.warning(f"ExchangeClient: POST failed {url}: {e}")
            return None

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
