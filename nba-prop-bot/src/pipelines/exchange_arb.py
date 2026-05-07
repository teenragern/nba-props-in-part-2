"""
Exchange arbitrage scanner — Phase 3C.

Compares the bot's model-implied probabilities (stored in line_history / alerts_sent)
against Kalshi NBA prop market prices. When the divergence exceeds
EXCHANGE_ARB_MIN_EDGE, an alert is queued to the pending_alerts digest.

Paper mode only — no orders are placed.

Schedule: every 30 minutes on game days via run_scheduler.py.
Gated by KALSHI_API_KEY env var — no-ops silently when unset.

Environment:
    KALSHI_API_KEY          Required to enable this pipeline
    EXCHANGE_ARB_MIN_EDGE   Minimum edge to alert on (default 0.03 = 3%)
    EXCHANGE_ARB_MAX_ALERTS Max alerts per run (default 5)
"""

import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.clients.exchange_client import ExchangeClient
from src.clients.telegram_bot import TelegramBotClient
from src.data.db import get_db_client
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_MIN_EDGE   = float(os.getenv("EXCHANGE_ARB_MIN_EDGE",   "0.03"))
_MAX_ALERTS = int(os.getenv("EXCHANGE_ARB_MAX_ALERTS",   "5"))

# Markets the bot tracks → human-readable short labels for the alert message
_MARKET_LABELS = {
    "player_points":   "PTS",
    "player_rebounds": "REB",
    "player_assists":  "AST",
    "player_threes":   "3PM",
}

# Kalshi stat label patterns in market tickers / titles
_KALSHI_STAT_MAP = {
    "point": "player_points",
    "rebound": "player_rebounds",
    "assist": "player_assists",
    "three": "player_threes",
    "3-point": "player_threes",
}


def _match_kalshi_market(
    title: str,
    player_name: str,
    market: str,
) -> bool:
    """
    Fuzzy-match a Kalshi market title to a (player, market) pair.
    Kalshi titles look like: "LeBron James Over 27.5 Points 2026-05-07"
    """
    title_lower = title.lower()
    # Player name: match last name at minimum
    last_name = player_name.split()[-1].lower()
    if last_name not in title_lower:
        return False

    # Stat keyword
    for kw, mkt in _KALSHI_STAT_MAP.items():
        if kw in title_lower and mkt == market:
            return True
    return False


def _extract_line_from_title(title: str) -> Optional[float]:
    """
    Pull the numeric line from a Kalshi title like
    "LeBron James Over 27.5 Points".
    """
    m = re.search(r"(\d+\.?\d*)\s+(?:point|rebound|assist|three|3-point)", title, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Fallback: first decimal in title
    m = re.search(r"\b(\d+\.\d+)\b", title)
    return float(m.group(1)) if m else None


def _load_today_model_probs(db) -> List[Dict]:
    """
    Pull today's alerts (sent or pending) along with the model probability
    that drove them so we can compare against exchange prices.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with db.get_conn() as conn:
        rows = conn.execute(
            """
            SELECT a.player_name, a.market, a.edge, a.side
            FROM alerts_sent a
            WHERE date(a.timestamp) = ?
              AND a.market IN ('player_points','player_rebounds',
                               'player_assists','player_threes')
            ORDER BY a.edge DESC
            """,
            (today,),
        ).fetchall()
    return [dict(r) for r in rows] if rows else []


def run_exchange_arb() -> Dict:
    """
    Main entry point called by the scheduler.

    Returns summary dict: {"checked": int, "alerts_queued": int, "skipped": int}
    """
    client = ExchangeClient()
    if not client.enabled:
        logger.debug("exchange_arb: ExchangeClient disabled (no KALSHI_API_KEY).")
        return {"checked": 0, "alerts_queued": 0, "skipped": 0}

    db  = get_db_client()
    bot = TelegramBotClient()

    # 1. Fetch all open Kalshi NBA prop markets
    logger.info("exchange_arb: fetching Kalshi NBA prop markets...")
    kalshi_markets = client.get_nba_prop_markets(limit=500)
    if not kalshi_markets:
        logger.info("exchange_arb: no Kalshi markets returned.")
        return {"checked": 0, "alerts_queued": 0, "skipped": 0}
    logger.info(f"exchange_arb: {len(kalshi_markets)} Kalshi markets loaded.")

    # 2. Load today's model-implied probabilities from alerts
    model_rows = _load_today_model_probs(db)
    if not model_rows:
        logger.info("exchange_arb: no today alerts to compare against.")
        return {"checked": 0, "alerts_queued": 0, "skipped": 0}

    # Build Kalshi ticker→price cache (avoid repeat API calls per player)
    price_cache: Dict[str, Optional[Dict]] = {}

    checked = 0
    alerts_queued = 0
    skipped = 0
    candidates: List[Tuple[float, str]] = []  # (edge, message) for ranking

    for row in model_rows:
        player = row["player_name"]
        market = row["market"]
        # edge here is the raw edge from the alert (model_prob - 0.5),
        # so model_prob ≈ edge + 0.50
        model_prob = max(0.01, min(0.99, (row.get("edge") or 0.0) + 0.50))

        # Find matching Kalshi market(s)
        for km in kalshi_markets:
            title  = km.get("title", "")
            ticker = km.get("ticker", "")
            if not _match_kalshi_market(title, player, market):
                continue

            # Fetch price (cached)
            if ticker not in price_cache:
                price_cache[ticker] = client.get_market_price(ticker)
            price = price_cache[ticker]
            if price is None:
                skipped += 1
                continue

            kalshi_implied = price["implied_yes"]   # Kalshi YES = OVER side
            if kalshi_implied <= 0:
                skipped += 1
                continue

            checked += 1
            divergence = model_prob - kalshi_implied

            if abs(divergence) < _MIN_EDGE:
                continue

            # Determine which side the bot favours
            side = "OVER" if divergence > 0 else "UNDER"
            # Kalshi YES = OVER, NO = UNDER
            kalshi_price = price["yes_bid"] if side == "OVER" else price["no_bid"]
            kalshi_pct   = kalshi_price * 100  # display as cents → percent

            stat_label = _MARKET_LABELS.get(market, market)
            kalshi_line = _extract_line_from_title(title)
            line_str    = f" {kalshi_line}" if kalshi_line else ""

            msg = (
                f"⚡ <b>Exchange Edge: {player}</b> {stat_label}{line_str} {side}\n"
                f"   Model: <b>{model_prob:.1%}</b>  ←→  Kalshi: <b>{kalshi_implied:.1%}</b>\n"
                f"   Divergence: <b>{divergence:+.1%}</b>  |  Kalshi price: {kalshi_pct:.0f}¢\n"
                f"   Ticker: <code>{ticker}</code>\n"
                f"   <i>Paper mode — informational only</i>"
            )
            candidates.append((abs(divergence), msg, player, market, ticker))

    # Sort by divergence descending, cap at _MAX_ALERTS
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:_MAX_ALERTS]

    for edge_val, msg, player, market, ticker in top:
        try:
            db.queue_pending_alert(
                alert_type="prop",
                title=f"Exchange Edge: {player} {_MARKET_LABELS.get(market, market)} | {edge_val:+.1%} vs Kalshi",
                body=msg,
            )
            alerts_queued += 1
            logger.info(f"exchange_arb: queued alert — {player} {market} edge={edge_val:+.1%}")
        except Exception as e:
            logger.warning(f"exchange_arb: failed to queue alert: {e}")

    if top:
        bot.send_message(
            f"📊 <b>Exchange Arb Scan</b> — {alerts_queued} edge(s) found "
            f"(checked {checked}, min edge {_MIN_EDGE:.0%})\n"
            f"<i>Digests include details.</i>"
        )

    logger.info(
        f"exchange_arb complete: checked={checked} alerts_queued={alerts_queued} "
        f"skipped={skipped} kalshi_markets={len(kalshi_markets)}"
    )
    return {"checked": checked, "alerts_queued": alerts_queued, "skipped": skipped}


if __name__ == "__main__":
    result = run_exchange_arb()
    print(result)
