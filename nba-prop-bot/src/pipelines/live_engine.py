"""
Live trading engine — Phase 3 live trading.

The "brain" of the live trading pipeline.

Subscribes to nba:live:state_update (from live_state_tracker), fetches the
matching Kalshi KXNBAGAME market price, and computes a live in-game win
probability.  When the model disagrees with the market beyond configured
thresholds it publishes BUY or SELL signals to nba:live:execution_queue for
the kalshi_trader to execute.

Win-probability model
---------------------
Uses a Normal-approximation anchored to the pre-game devigged sharp line:

    sigma = 11.0 * sqrt(minutes_remaining / 48.0)
    prior_adj = norm.ppf(pre_game_prob) * sigma
    win_prob = norm.cdf((score_diff + prior_adj) / sigma)

This gives:
  • win_prob == pre_game_prob at tip-off (score_diff=0, full time remaining).
  • win_prob → 1.0 / 0.0 as sigma → 0 when the clock expires.
  • sigma = 11.0 at tip-off, matching empirical NBA scoring standard deviation.

Entry signal:  (win_prob - kalshi_price) > LIVE_BUY_THRESHOLD (default 0.07)
Exit signal:   holding position AND (kalshi_price - win_prob) > LIVE_SELL_THRESHOLD (0.03)

Kelly stake: bankroll * kelly_fraction * (edge / (1 - kalshi_price)), clamped to
             [0.50, KALSHI_MAX_STAKE].

Environment:
    REDIS_URL               Redis connection URL (required)
    KALSHI_API_KEY          Kalshi key (required for price fetching)
    KALSHI_API_SECRET       Kalshi RSA secret
    BDL_API_KEY             BDL key (for pre-game prior fallback)
    ODDS_API_KEY            Odds API key (for pre-game prior fallback)
    BANKROLL                Total bankroll (default: 1000.0)
    KELLY_FRACTION          Kelly fraction (default: 0.25)
    KALSHI_MAX_STAKE        Max USD per order (default: 5.00)
    LIVE_BUY_THRESHOLD      Entry edge threshold (default: 0.07)
    LIVE_SELL_THRESHOLD     Exit edge threshold (default: 0.03)

Run:
    python -m src.pipelines.live_engine
"""

import asyncio
import dataclasses
import math
import os
import signal
import time
from typing import Dict, Optional

import scipy.stats

from src.clients.exchange_client import ExchangeClient
from src.clients.odds_api import OddsApiClient
from src.config import BANKROLL, KELLY_FRACTION, SHARP_BOOKS
from src.data.db import get_db_client
from src.events.bus import EventBus, get_bus
from src.models.devig import devig_shin
from src.utils.async_bridge import make_event_bridge
from src.utils.kalshi_matching import find_kalshi_game_markets, resolve_team_for_market
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_BUY_THRESHOLD  = float(os.getenv("LIVE_BUY_THRESHOLD",  "0.07"))
_SELL_THRESHOLD = float(os.getenv("LIVE_SELL_THRESHOLD", "0.03"))
_KALSHI_MAX_STAKE = float(os.getenv("KALSHI_MAX_STAKE",   "5.00"))

# Empirical NBA scoring standard deviation for a full 48-min game (in points).
_NBA_SIGMA_FULL = 11.0


# ── Win-probability model ─────────────────────────────────────────────────────

def compute_live_win_prob(
    score_diff: int,
    minutes_elapsed: float,
    pre_game_prob: float,
) -> float:
    """
    Normal-approximation live win probability.

    Args:
        score_diff:     home_score - away_score (positive = home leading).
        minutes_elapsed: Total game minutes elapsed (0–48+).
        pre_game_prob:  Pre-game devigged P(home wins) — the model's prior.

    Returns:
        Float in (0.02, 0.98) — probability the home team wins.
    """
    minutes_remaining = max(0.1, 48.0 - minutes_elapsed)
    sigma = _NBA_SIGMA_FULL * math.sqrt(minutes_remaining / 48.0)

    # prior_adj shifts the distribution centre so that at score_diff=0 and
    # full time remaining win_prob == pre_game_prob exactly.
    prior_adj = scipy.stats.norm.ppf(pre_game_prob) * sigma

    win_prob = float(scipy.stats.norm.cdf((score_diff + prior_adj) / sigma))
    # Hard floors to avoid degenerate probabilities near the buzzer.
    return max(0.02, min(0.98, win_prob))


# ── Kelly stake sizing ────────────────────────────────────────────────────────

def _kelly_stake(edge: float, kalshi_price: float) -> float:
    """
    Fractional Kelly stake in dollars.

    f* = edge / (1 - kalshi_price)   (binary Kelly for a $1 payout contract)
    stake = bankroll * kelly_fraction * f*

    Clamped to [0.50, KALSHI_MAX_STAKE].
    """
    denominator = 1.0 - kalshi_price
    if denominator <= 0:
        return 0.0
    f_star = edge / denominator
    raw = BANKROLL * KELLY_FRACTION * f_star
    return max(0.50, min(_KALSHI_MAX_STAKE, raw))


# ── In-process state ──────────────────────────────────────────────────────────

@dataclasses.dataclass
class EngineState:
    """Cached pre-game data for one game — loaded once, reused every tick."""
    game_id:       str
    home_team:     str
    away_team:     str
    pre_game_prob: float   # devigged P(home wins) from sharp pre-game line


# Keyed by ticker — tracks which markets we currently hold a position in.
# Value: "yes" or "no" (the side we bought).
_holdings: Dict[str, str] = {}

# Keyed by game_id — cached EngineState to avoid repeated DB / API calls.
_engine_states: Dict[str, EngineState] = {}


def _load_holdings_from_db(db) -> int:
    """
    Restore _holdings from the live_positions table so positions
    opened before a restart can still be sold.  Returns count loaded.
    """
    try:
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT ticker, side FROM live_positions"
            ).fetchall()
        for row in rows:
            _holdings[row[0]] = row[1]
        if rows:
            logger.info(
                f"live_engine: restored {len(rows)} open position(s) from DB: "
                + ", ".join(r[0] for r in rows)
            )
        return len(rows)
    except Exception as e:
        logger.warning(f"live_engine: failed to load holdings from DB: {e}")
        return 0


# ── Pre-game probability loading ──────────────────────────────────────────────

def _load_pre_game_prob_sync(
    game_id: str,
    home_team: str,
    away_team: str,
    odds_client: OddsApiClient,
) -> Optional[float]:
    """
    Synchronous.  Fetch and devig the pre-game sharp moneyline.

    1. Try the Odds API for a live h2h line from the sharpest available book.
    2. Devig using the Shin method (best for binary markets).
    3. Return None if no sharp line is found — caller must skip the game.
    """
    try:
        data = odds_client.get_event_odds(game_id, markets=["h2h"])
        bookmakers = data.get("bookmakers", [])
        priority = [b.lower() for b in SHARP_BOOKS] if SHARP_BOOKS else ["pinnacle"]

        def _rank(bm):
            key = bm.get("key", "").lower()
            return priority.index(key) if key in priority else 999

        bookmakers = sorted(bookmakers, key=_rank)
        home_last  = home_team.split()[-1].lower()
        away_last  = away_team.split()[-1].lower()

        for bm in bookmakers:
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                outcomes = mkt.get("outcomes", [])
                home_price = away_price = None
                for o in outcomes:
                    name = o.get("name", "").lower()
                    if home_last in name:
                        home_price = o.get("price")
                    elif away_last in name:
                        away_price = o.get("price")
                if home_price and away_price and home_price > 1 and away_price > 1:
                    home_raw   = 1.0 / home_price
                    away_raw   = 1.0 / away_price
                    home_prob, _ = devig_shin(home_raw, away_raw)
                    logger.info(
                        f"live_engine: pre-game prior loaded — "
                        f"{home_team} {home_prob:.1%} (book: {bm.get('key')})"
                    )
                    return home_prob
    except Exception as e:
        logger.warning(f"live_engine: pre-game prob fetch failed for {game_id}: {e}")

    logger.warning(
        f"live_engine: no sharp h2h line for {game_id} — "
        f"skipping game (refusing to trade without a valid prior)"
    )
    return None


async def _get_or_load_engine_state(
    game_id: str,
    home_team: str,
    away_team: str,
    odds_client: OddsApiClient,
) -> Optional[EngineState]:
    """Load from cache or fetch pre-game prob for this game.
    Returns None if no sharp line is available — caller must skip."""
    if game_id in _engine_states:
        return _engine_states[game_id]

    pre_game_prob = await asyncio.to_thread(
        _load_pre_game_prob_sync, game_id, home_team, away_team, odds_client
    )
    if pre_game_prob is None:
        return None

    state = EngineState(
        game_id       = game_id,
        home_team     = home_team,
        away_team     = away_team,
        pre_game_prob = pre_game_prob,
    )
    _engine_states[game_id] = state
    return state


# ── Kalshi price fetch ────────────────────────────────────────────────────────

async def _fetch_kalshi_price(
    exchange: ExchangeClient,
    ticker: str,
) -> Optional[float]:
    """Async wrapper around ExchangeClient.get_market_price()."""
    try:
        result = await asyncio.to_thread(exchange.get_market_price, ticker)
        if result:
            return result.get("implied_yes")
    except Exception as e:
        logger.warning(f"live_engine: price fetch failed for {ticker}: {e}")
    return None


# ── Signal builder ────────────────────────────────────────────────────────────

def _make_signal(
    action: str,
    ticker: str,
    game_id: str,
    team: str,
    side: str,
    model_prob: float,
    kalshi_price: float,
    edge: float,
    stake_usd: float,
) -> dict:
    return {
        "action":       action,       # "buy" or "sell"
        "ticker":       ticker,
        "game_id":      game_id,
        "team":         team,
        "side":         side,         # "yes" or "no"
        "model_prob":   model_prob,
        "kalshi_price": kalshi_price,
        "edge":         edge,
        "stake_usd":    stake_usd,
        "ts":           time.time(),  # used by latency circuit-breaker in trader
    }


# ── Core state-update processor ───────────────────────────────────────────────

async def process_state_update(
    payload: dict,
    exchange: ExchangeClient,
    odds_client: OddsApiClient,
    bus: EventBus,
) -> None:
    """
    Called once per LIVE_STATE_UPDATE message.

    1. Derive score diff and game clock from payload.
    2. Load or cache EngineState (pre-game probability).
    3. Find matching Kalshi KXNBAGAME market(s) for this game.
    4. Fetch current Kalshi price.
    5. Compute live win_prob.
    6. Apply BUY / SELL logic and publish signals.
    """
    game_id         = str(payload.get("game_id", ""))
    home_team       = payload.get("home_team", "")
    away_team       = payload.get("away_team", "")
    home_score      = int(payload.get("home_score", 0))
    away_score      = int(payload.get("away_score", 0))
    minutes_elapsed = float(payload.get("minutes_elapsed", 0.0))

    if not game_id or not home_team:
        return

    eng = await _get_or_load_engine_state(game_id, home_team, away_team, odds_client)
    if eng is None:
        return  # no sharp prior — skip this game entirely
    score_diff = home_score - away_score

    # Find matching Kalshi markets (blocking — runs in thread pool)
    markets = await asyncio.to_thread(
        find_kalshi_game_markets, exchange, home_team, away_team
    )
    if not markets:
        logger.debug(f"live_engine: no Kalshi markets for game {game_id}")
        return

    for market in markets:
        ticker = market.get("ticker", "")
        if not ticker:
            continue

        # Which team does YES represent?
        team_for_yes = resolve_team_for_market(market, home_team, away_team)
        if team_for_yes is None:
            continue

        # Align score_diff to the perspective of the YES team
        if team_for_yes == home_team:
            model_prob = compute_live_win_prob(score_diff, minutes_elapsed, eng.pre_game_prob)
        else:
            # Away team: flip score_diff and use (1 - pre_game_prob) as prior
            model_prob = compute_live_win_prob(-score_diff, minutes_elapsed, 1.0 - eng.pre_game_prob)

        # Fetch live Kalshi price
        kalshi_price = await _fetch_kalshi_price(exchange, ticker)
        if kalshi_price is None or kalshi_price <= 0:
            continue

        buy_edge  = model_prob - kalshi_price
        sell_edge = kalshi_price - model_prob

        logger.debug(
            f"live_engine: {team_for_yes} model={model_prob:.1%} "
            f"kalshi={kalshi_price:.1%} buy_edge={buy_edge:+.1%} "
            f"sell_edge={sell_edge:+.1%} holding={ticker in _holdings}"
        )

        # ── BUY logic ──────────────────────────────────────────────────
        if buy_edge > _BUY_THRESHOLD and ticker not in _holdings:
            stake = _kelly_stake(buy_edge, kalshi_price)
            signal = _make_signal(
                action      = "buy",
                ticker      = ticker,
                game_id     = game_id,
                team        = team_for_yes,
                side        = "yes",
                model_prob  = model_prob,
                kalshi_price= kalshi_price,
                edge        = buy_edge,
                stake_usd   = stake,
            )
            published = await asyncio.to_thread(
                bus.publish, EventBus.LIVE_EXECUTION_QUEUE, signal
            )
            if published:
                _holdings[ticker] = "yes"
                logger.info(
                    f"live_engine: BUY signal — {team_for_yes} "
                    f"model={model_prob:.1%} kalshi={kalshi_price:.1%} "
                    f"edge={buy_edge:+.1%} stake=${stake:.2f}"
                )

        # ── SELL logic (profit-take) ───────────────────────────────────
        elif sell_edge > _SELL_THRESHOLD and ticker in _holdings:
            signal = _make_signal(
                action      = "sell",
                ticker      = ticker,
                game_id     = game_id,
                team        = team_for_yes,
                side        = _holdings[ticker],
                model_prob  = model_prob,
                kalshi_price= kalshi_price,
                edge        = sell_edge,
                stake_usd   = 0.0,   # determined by open contracts in trader
            )
            published = await asyncio.to_thread(
                bus.publish, EventBus.LIVE_EXECUTION_QUEUE, signal
            )
            if published:
                del _holdings[ticker]
                logger.info(
                    f"live_engine: SELL signal — {team_for_yes} "
                    f"market overvalued model={model_prob:.1%} "
                    f"kalshi={kalshi_price:.1%} sell_edge={sell_edge:+.1%}"
                )


# ── Main event loop ───────────────────────────────────────────────────────────

async def main() -> None:
    """
    1. Set up the LIVE_STATE_UPDATE → asyncio.Queue bridge.
    2. Initialise ExchangeClient, OddsApiClient.
    3. Install SIGINT/SIGTERM shutdown handler.
    4. Loop: consume state-update payloads and call process_state_update().
    """
    bus = get_bus()
    if not bus.is_available():
        logger.warning(
            "live_engine: Redis unavailable — no events will be received. "
            "Set REDIS_URL to enable."
        )

    loop = asyncio.get_event_loop()
    queue, _sub_thread = make_event_bridge(EventBus.LIVE_STATE_UPDATE, loop)

    exchange    = ExchangeClient()
    odds_client = OddsApiClient()
    db          = get_db_client()

    # Restore open positions from DB so we can sell positions opened before restart
    _load_holdings_from_db(db)

    if not exchange.enabled:
        logger.warning(
            "live_engine: ExchangeClient disabled (no KALSHI_API_KEY) — "
            "prices cannot be fetched and no signals will be generated."
        )

    shutdown = asyncio.Event()

    def _handle_signal(sig, _frame):
        logger.info(f"live_engine: received {signal.Signals(sig).name} — shutting down.")
        shutdown.set()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        f"live_engine: started (buy_threshold={_BUY_THRESHOLD:.0%}, "
        f"sell_threshold={_SELL_THRESHOLD:.0%}, max_stake=${_KALSHI_MAX_STAKE:.2f})"
    )

    while not shutdown.is_set():
        try:
            payload = await asyncio.wait_for(queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            # Heartbeat — no live games or Redis is quiet. Loop to check shutdown.
            continue
        except asyncio.CancelledError:
            break

        try:
            await process_state_update(payload, exchange, odds_client, bus)
        except Exception as e:
            logger.error(f"live_engine: unhandled error processing state update: {e}")

    logger.info("live_engine: stopped.")


if __name__ == "__main__":
    asyncio.run(main())
