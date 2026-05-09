"""
Exchange arbitrage scanner — Phase 3C.

Compares the bot's model-implied probabilities (stored in alerts_sent)
against Kalshi NBA prop market prices. When the divergence exceeds
EXCHANGE_ARB_MIN_EDGE, places a live limit order and sends a Telegram alert.

Schedule: every 30 minutes on game days via run_scheduler.py.
Gated by KALSHI_API_KEY env var — no-ops silently when unset.

Environment:
    KALSHI_API_KEY              Required to enable this pipeline
    EXCHANGE_ARB_MIN_EDGE       Minimum edge to act on (default 0.03 = 3%)
    EXCHANGE_ARB_MAX_ALERTS     Max orders per run (default 5)
    KALSHI_MAX_STAKE            Max dollars per order (default 5.00)
    KALSHI_MAX_ORDERS_PER_DAY   Daily order cap safety limit (default 10)
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

_MIN_EDGE          = float(os.getenv("EXCHANGE_ARB_MIN_EDGE",       "0.03"))
_MAX_ALERTS        = int(os.getenv("EXCHANGE_ARB_MAX_ALERTS",       "5"))
_MAX_ORDERS_PER_DAY = int(os.getenv("KALSHI_MAX_ORDERS_PER_DAY",   "10"))

# In-process daily order counter — resets when the process restarts (once/day)
_orders_today: Dict[str, int] = {}   # date_str → count

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
    with db.get_conn() as conn:
        # Use a rolling 12-hour window instead of a strict UTC date
        # to handle the evening/night rollover correctly.
        rows = conn.execute(
            """
            SELECT a.player_name, a.market, a.edge, a.side
            FROM alerts_sent a
            WHERE a.timestamp >= NOW() - INTERVAL '12 hours'
              AND a.market IN ('player_points','player_rebounds',
                               'player_assists','player_threes')
            ORDER BY a.edge DESC
            """
        ).fetchall()
    return [dict(r) for r in rows] if rows else []


def _orders_placed_today() -> int:
    # Use local date for the order cap
    today = datetime.now().strftime("%Y-%m-%d")
    return _orders_today.get(today, 0)


def _increment_order_count():
    # Use local date for the order cap
    today = datetime.now().strftime("%Y-%m-%d")
    _orders_today[today] = _orders_today.get(today, 0) + 1


def run_exchange_arb() -> Dict:
    """
    Main entry point called by the scheduler.

    Returns summary dict: {"checked": int, "orders_placed": int, "skipped": int}
    """
    client = ExchangeClient()
    if not client.enabled:
        logger.debug("exchange_arb: ExchangeClient disabled (no KALSHI_API_KEY).")
        return {"checked": 0, "orders_placed": 0, "skipped": 0}

    db  = get_db_client()
    bot = TelegramBotClient()

    # 1. Fetch all open Kalshi NBA prop markets
    logger.info("exchange_arb: fetching Kalshi NBA prop markets...")
    kalshi_markets = client.get_nba_prop_markets(limit=500)
    if not kalshi_markets:
        logger.info("exchange_arb: no Kalshi markets returned.")
        return {"checked": 0, "orders_placed": 0, "skipped": 0}
    logger.info(f"exchange_arb: {len(kalshi_markets)} Kalshi markets loaded.")

    # 2. Load today's model-implied probabilities from alerts
    model_rows = _load_today_model_probs(db)
    if not model_rows:
        logger.info("exchange_arb: no today alerts to compare against.")
        return {"checked": 0, "orders_placed": 0, "skipped": 0}

    # Build Kalshi ticker→price cache (avoid repeat API calls per player)
    price_cache: Dict[str, Optional[Dict]] = {}

    checked = 0
    orders_placed = 0
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
            # Use ask price so the buy limit order crosses the spread and fills immediately
            kalshi_price = price["yes_ask"] if side == "OVER" else price["no_ask"]
            kalshi_pct   = kalshi_price * 100  # display as cents → percent

            stat_label  = _MARKET_LABELS.get(market, market)
            kalshi_line = _extract_line_from_title(title)
            line_str    = f" {kalshi_line}" if kalshi_line else ""
            kalshi_side = "yes" if side == "OVER" else "no"
            limit_cents = int(kalshi_price * 100)

            candidates.append((
                abs(divergence), player, market, ticker,
                stat_label, line_str, side, kalshi_side,
                limit_cents, model_prob, kalshi_implied, divergence, kalshi_pct,
            ))

    # Sort by divergence descending, cap at _MAX_ALERTS
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:_MAX_ALERTS]

    for (edge_val, player, market, ticker, stat_label, line_str, side,
         kalshi_side, limit_cents, model_prob, kalshi_implied, divergence,
         kalshi_pct) in top:

        # Daily order cap check
        if _orders_placed_today() >= _MAX_ORDERS_PER_DAY:
            logger.warning(f"exchange_arb: daily order cap ({_MAX_ORDERS_PER_DAY}) reached — skipping.")
            break

        # Place live limit order on Kalshi
        order_result = client.place_order(
            ticker=ticker,
            side=kalshi_side,
            limit_price_cents=limit_cents,
        )

        if order_result is None:
            logger.warning(f"exchange_arb: order failed — {ticker} {kalshi_side} {limit_cents}¢")
            continue

        _increment_order_count()
        orders_placed += 1

        order_id    = order_result.get("order", {}).get("order_id", "n/a")
        count_filled = order_result.get("order", {}).get("count", "?")
        cost        = (limit_cents / 100) * (count_filled if isinstance(count_filled, int) else 0)

        msg = (
            f"⚡ <b>Kalshi Order Placed: {player}</b> {stat_label}{line_str} {side}\n"
            f"   Model: <b>{model_prob:.1%}</b>  vs  Kalshi: <b>{kalshi_implied:.1%}</b>\n"
            f"   Edge: <b>{divergence:+.1%}</b>  |  Price: {limit_cents}¢  |  Cost: ${cost:.2f}\n"
            f"   Order ID: <code>{order_id}</code>"
        )
        bot.send_message(msg)
        logger.info(f"exchange_arb: order placed — {ticker} {kalshi_side} {limit_cents}¢ id={order_id}")

        try:
            db.queue_pending_alert(
                alert_type="prop",
                title=f"Kalshi Order: {player} {stat_label} {side} | {divergence:+.1%} edge",
                body=msg,
            )
        except Exception as e:
            logger.warning(f"exchange_arb: failed to log alert: {e}")

    logger.info(
        f"exchange_arb complete: checked={checked} orders_placed={orders_placed} "
        f"skipped={skipped} daily_total={_orders_placed_today()}"
    )
    return {"checked": checked, "orders_placed": orders_placed, "skipped": skipped}


if __name__ == "__main__":
    result = run_exchange_arb()
    print(result)
