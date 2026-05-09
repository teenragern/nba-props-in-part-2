"""
Live game-winner arbitrage scanner.

Every 5 minutes on game days, this pipeline:
  1. Loads today's games from the DB
  2. Fetches Pinnacle (or best available sharp book) moneylines via Odds API
  3. Devigs them into true win probabilities using the Shin method
  4. Adjusts for today's injury news (newly OUT players shift the line)
  5. Compares against live Kalshi KXNBAGAME market prices
  6. Places a limit order instantly when divergence > GAME_ARB_MIN_EDGE

Example:
    Model says Knicks WIN = 71%
    Kalshi prices Knicks WIN at 60¢ (implied 60%)
    Divergence = +11% → buy YES at 60¢

Environment:
    GAME_ARB_MIN_EDGE           Min divergence to trade (default 0.05 = 5%)
    GAME_ARB_MAX_ORDERS_PER_DAY Daily order cap (default 6)
    KALSHI_MAX_STAKE            Max $ per order (shared with prop arb, default 5.00)
    SHARP_BOOKS                 Priority order for sharp books (default: pinnacle)
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.clients.exchange_client import ExchangeClient
from src.clients.odds_api import OddsApiClient
from src.clients.telegram_bot import TelegramBotClient
from src.config import SHARP_BOOKS
from src.data.db import get_db_client
from src.models.devig import devig_two_way
from src.utils.kalshi_matching import TEAM_LOOKUP, find_kalshi_game_markets, resolve_team_for_market
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_MIN_EDGE        = float(os.getenv("GAME_ARB_MIN_EDGE",           "0.05"))
_MAX_ORDERS_DAY  = int(os.getenv("GAME_ARB_MAX_ORDERS_PER_DAY",  "6"))

# Per-star-player-out win probability adjustment
_INJURY_ADJ_PER_STAR = 0.05

# Team lookup is shared via src/utils/kalshi_matching.TEAM_LOOKUP.
_TEAM_LOOKUP = TEAM_LOOKUP

# In-process daily order counter
_orders_today: Dict[str, int] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _orders_placed_today() -> int:
    return _orders_today.get(datetime.now(timezone.utc).strftime("%Y-%m-%d"), 0)


def _inc_orders():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _orders_today[today] = _orders_today.get(today, 0) + 1


# ── Step 1: Sharp moneyline → win probabilities ───────────────────────────────

def _get_sharp_probs(
    odds_client: OddsApiClient,
    game_id: str,
    home_team: str,
    away_team: str,
) -> Optional[Tuple[float, float]]:
    """
    Fetch h2h moneyline from the sharpest available book and return
    devigged (home_prob, away_prob).

    Uses Shin devig (most accurate for 2-way markets).
    Returns None when no sharp line is available.
    """
    try:
        data = odds_client.get_event_odds(game_id, markets=["h2h"])
    except Exception as e:
        logger.warning(f"game_arb: odds fetch failed for {game_id}: {e}")
        return None

    bookmakers = data.get("bookmakers", [])
    priority   = [b.lower() for b in SHARP_BOOKS] if SHARP_BOOKS else ["pinnacle"]

    def _book_priority(bm):
        key = bm.get("key", "").lower()
        try:
            return priority.index(key)
        except ValueError:
            return 999

    bookmakers_sorted = sorted(bookmakers, key=_book_priority)

    home_last = home_team.split()[-1].lower()
    away_last = away_team.split()[-1].lower()

    for bm in bookmakers_sorted:
        for mkt in bm.get("markets", []):
            if mkt.get("key") != "h2h":
                continue
            outcomes = mkt.get("outcomes", [])
            if len(outcomes) < 2:
                continue

            home_price = away_price = None
            for o in outcomes:
                name = o.get("name", "").lower()
                if home_last in name:
                    home_price = o.get("price")
                elif away_last in name:
                    away_price = o.get("price")

            if home_price and away_price and home_price > 1 and away_price > 1:
                home_raw = 1.0 / home_price
                away_raw = 1.0 / away_price
                home_prob, away_prob = devig_two_way(home_raw, away_raw)
                logger.debug(
                    f"game_arb: {home_team} vs {away_team} — "
                    f"{bm.get('key')} ML {home_price}/{away_price} → "
                    f"devigged {home_prob:.1%}/{away_prob:.1%}"
                )
                return home_prob, away_prob

    logger.info(f"game_arb: no sharp h2h line found for {game_id}")
    return None


# ── Step 2: Injury adjustment ─────────────────────────────────────────────────

def _injury_adjustment(home_team: str, away_team: str, db) -> Tuple[float, float]:
    """
    Check today's injury report for OUT players.
    Returns (home_adj, away_adj) — a positive value boosts that team's prob.

    Logic: if a home team player is OUT, the away team gets a boost (home_adj < 0).
    Severity is used as a multiplier when available (0-10 scale).
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        with db.get_conn() as conn:
            rows = conn.execute(
                """
                SELECT player_name, team, severity
                FROM injury_reports
                WHERE game_date = ? AND LOWER(status) IN ('out', 'doubtful')
                """,
                (today,),
            ).fetchall()
    except Exception as e:
        logger.debug(f"game_arb: injury query failed: {e}")
        return 0.0, 0.0

    home_last = home_team.split()[-1].lower()
    away_last = away_team.split()[-1].lower()

    home_adj = away_adj = 0.0
    for row in rows:
        team     = (row["team"] or "").lower()
        severity = int(row["severity"] or 5)
        scale    = max(0.5, min(1.5, severity / 5.0))   # severity 5 = normal, 10 = superstar

        if home_last in team:
            # Home player out → boost away team, penalise home
            home_adj -= _INJURY_ADJ_PER_STAR * scale
        elif away_last in team:
            # Away player out → boost home team
            home_adj += _INJURY_ADJ_PER_STAR * scale

    if home_adj != 0:
        logger.info(f"game_arb: injury adj {home_team}={home_adj:+.2f} {away_team}={-home_adj:+.2f}")

    away_adj = -home_adj
    return home_adj, away_adj


# ── Step 3: Find Kalshi game markets ─────────────────────────────────────────
# Delegates to the shared kalshi_matching utility.

def _find_kalshi_game_markets(
    client: ExchangeClient,
    home_team: str,
    away_team: str,
) -> List[Dict]:
    return find_kalshi_game_markets(client, home_team, away_team)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_game_arb() -> Dict:
    """
    Called by the scheduler every 5 minutes on game days.
    Returns {"games_checked": int, "orders_placed": int}
    """
    client = ExchangeClient()
    if not client.enabled:
        logger.debug("game_arb: ExchangeClient disabled (no KALSHI_API_KEY).")
        return {"games_checked": 0, "orders_placed": 0}

    db          = get_db_client()
    odds_client = OddsApiClient()
    bot         = TelegramBotClient()
    today       = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Load today's games
    try:
        with db.get_conn() as conn:
            games = conn.execute(
                "SELECT game_id, home_team, away_team FROM games WHERE date(commence_time) = ?",
                (today,),
            ).fetchall()
    except Exception as e:
        logger.warning(f"game_arb: failed to load games: {e}")
        return {"games_checked": 0, "orders_placed": 0}

    if not games:
        logger.debug("game_arb: no games today.")
        return {"games_checked": 0, "orders_placed": 0}

    games_checked = 0
    orders_placed = 0

    for game in games:
        if _orders_placed_today() >= _MAX_ORDERS_DAY:
            logger.warning(f"game_arb: daily cap ({_MAX_ORDERS_DAY}) reached — stopping.")
            break

        game_id   = game["game_id"]
        home_team = game["home_team"]
        away_team = game["away_team"]

        # 1. Sharp win probabilities
        probs = _get_sharp_probs(odds_client, game_id, home_team, away_team)
        if probs is None:
            continue
        home_prob, away_prob = probs

        # 2. Injury adjustment
        home_adj, away_adj = _injury_adjustment(home_team, away_team, db)
        home_prob = min(0.97, max(0.03, home_prob + home_adj))
        away_prob = min(0.97, max(0.03, away_prob + away_adj))
        # Re-normalise so they sum to 1
        total     = home_prob + away_prob
        home_prob /= total
        away_prob /= total

        games_checked += 1
        logger.info(
            f"game_arb: {home_team} {home_prob:.1%} vs {away_team} {away_prob:.1%}"
        )

        # 3. Find Kalshi markets for this game
        kalshi_markets = _find_kalshi_game_markets(client, home_team, away_team)
        if not kalshi_markets:
            logger.info(f"game_arb: no Kalshi market for {home_team} vs {away_team}")
            continue

        for km in kalshi_markets:
            ticker = km.get("ticker", "")

            # Which team does this YES contract represent winning?
            team_label = resolve_team_for_market(km, home_team, away_team)
            if team_label is None:
                continue
            model_prob = home_prob if team_label == home_team else away_prob

            # 4. Get live Kalshi price
            price = client.get_market_price(ticker)
            if not price:
                continue
            kalshi_implied = price["implied_yes"]
            if kalshi_implied <= 0:
                continue

            divergence = model_prob - kalshi_implied

            logger.debug(
                f"game_arb: {team_label} model={model_prob:.1%} "
                f"kalshi={kalshi_implied:.1%} div={divergence:+.1%}"
            )

            if abs(divergence) < _MIN_EDGE:
                continue

            # 5. Determine side and price
            if divergence > 0:
                # YES (team wins) is underpriced on Kalshi — buy YES
                side        = "yes"
                limit_cents = int(price["yes_ask"] * 100)
            else:
                # NO (team loses) is underpriced — buy NO
                side        = "no"
                limit_cents = int(price["no_ask"] * 100)

            if limit_cents <= 0 or limit_cents >= 100:
                continue

            # 6. Place live limit order
            logger.info(
                f"game_arb: EDGE {team_label} model={model_prob:.1%} "
                f"kalshi={kalshi_implied:.1%} div={divergence:+.1%} "
                f"→ {side.upper()} {limit_cents}¢"
            )

            order = client.place_order(
                ticker=ticker,
                side=side,
                limit_price_cents=limit_cents,
            )
            if order is None:
                logger.warning(f"game_arb: order failed — {ticker}")
                continue

            _inc_orders()
            orders_placed += 1

            order_id = order.get("order", {}).get("order_id", "n/a")
            count    = order.get("order", {}).get("count", 0)
            cost     = (limit_cents / 100) * (count if isinstance(count, int) else 0)

            msg = (
                f"🏀 <b>Game Arb: {team_label} WIN</b>\n"
                f"   {home_team} vs {away_team}\n"
                f"   Model: <b>{model_prob:.1%}</b>  ←→  Kalshi: <b>{kalshi_implied:.1%}</b>\n"
                f"   Edge: <b>{divergence:+.1%}</b>  |  {side.upper()} @ {limit_cents}¢"
                f"  |  ${cost:.2f}\n"
                f"   Order ID: <code>{order_id}</code>"
            )
            bot.send_message(msg)
            logger.info(f"game_arb: placed {ticker} {side} {limit_cents}¢ id={order_id}")

    logger.info(
        f"game_arb complete: games_checked={games_checked} "
        f"orders_placed={orders_placed} daily_total={_orders_placed_today()}"
    )
    return {"games_checked": games_checked, "orders_placed": orders_placed}


if __name__ == "__main__":
    print(run_game_arb())
