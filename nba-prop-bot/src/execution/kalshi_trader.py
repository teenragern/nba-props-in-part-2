"""
Kalshi live trader — Phase 3 live trading.

The "hands" of the live trading pipeline.

Subscribes to nba:live:execution_queue (signals from live_engine) and executes
live BUY and SELL orders on Kalshi after running three circuit breakers:

  1. Latency shield  — rejects signals older than LIVE_SIGNAL_LATENCY_LIMIT s.
  2. Wash-trade guard — prevents acting on the same ticker twice within a
                        LIVE_WASH_COOLDOWN window.
  3. Exposure cap    — no more than LIVE_MAX_EXPOSURE_PER_TEAM USD at risk per
                        team per game (queried from live_positions table).

Every execution attempt — buy, sell, or rejection — is written to the
live_trades Postgres table for full auditability and P&L tracking.
Open positions are tracked in the live_positions table (upserted on BUY,
deleted on SELL).

Environment:
    REDIS_URL                    Redis URL (required)
    PG_DSN / DB_PATH             Postgres or SQLite connection
    KALSHI_API_KEY               Kalshi key (required for live orders)
    KALSHI_API_SECRET            Kalshi RSA secret
    LIVE_MAX_EXPOSURE_PER_TEAM   Max USD per team per game (default: 100.0)
    LIVE_SIGNAL_LATENCY_LIMIT    Max signal age in seconds (default: 1.5)
    LIVE_WASH_COOLDOWN           Min seconds between actions on same ticker (default: 2.0)
    EXECUTION_MODE               "paper" (default) or "live"
    EXECUTION_STYLE              "maker" (default, resting limit) or "taker" (cross spread)
    ORDER_POLL_INTERVAL          Seconds between pending order status checks (default: 5.0)
    ORDER_POLL_TIMEOUT           Seconds before a resting order is auto-cancelled (default: 120.0)

Run:
    python -m src.execution.kalshi_trader
"""

import asyncio
import dataclasses
import os
import signal
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from src.clients.exchange_client import ExchangeClient
from src.data.db import get_db_client
from src.events.bus import EventBus, get_bus
from src.models.portfolio_risk import PortfolioRiskManager, ProposedTrade
from src.utils.async_bridge import make_event_bridge
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_MAX_EXPOSURE     = float(os.getenv("LIVE_MAX_EXPOSURE_PER_TEAM", "100.0"))
_LATENCY_LIMIT    = float(os.getenv("LIVE_SIGNAL_LATENCY_LIMIT",  "1.5"))
_WASH_COOLDOWN    = float(os.getenv("LIVE_WASH_COOLDOWN",          "2.0"))
_EXECUTION_MODE   = os.getenv("EXECUTION_MODE", "paper").lower()
_EXECUTION_STYLE  = os.getenv("EXECUTION_STYLE", "maker").lower()  # "maker", "taker", or "l2"
_ORDER_POLL_INTERVAL = float(os.getenv("ORDER_POLL_INTERVAL", "5.0"))
_ORDER_POLL_TIMEOUT  = float(os.getenv("ORDER_POLL_TIMEOUT", "120.0"))

# L2 Execution Engine (lazy import to avoid circular deps)
_l2_engine = None

def _get_l2_engine(exchange: 'ExchangeClient'):
    """Lazily initialize the L2 execution engine."""
    global _l2_engine
    if _l2_engine is None:
        from src.execution.l2_execution import L2ExecutionEngine
        _l2_engine = L2ExecutionEngine(exchange)
    return _l2_engine


# ── In-process session state ──────────────────────────────────────────────────

@dataclasses.dataclass
class PendingOrder:
    """Tracks an order submitted to the exchange awaiting fill confirmation."""
    order_id: str
    ticker: str
    game_id: str
    team: str
    side: str
    action: str           # "buy" or "sell"
    total_contracts: int
    price_cents: int
    model_prob: float
    kalshi_price: float
    edge: float
    stake_usd: float
    avg_entry_cents: Optional[int]  # only for sell orders — entry price
    submitted_at: float
    filled_so_far: int = 0


@dataclasses.dataclass
class TraderState:
    """Session-scoped, in-process state for circuit-breaker fast-paths."""
    # ticker → timestamp of last action on that ticker
    last_action_ts: Dict[str, float] = dataclasses.field(default_factory=dict)
    # session date string YYYY-MM-DD
    session_id: str = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    # order_id → PendingOrder (live mode only)
    pending_orders: Dict[str, PendingOrder] = dataclasses.field(default_factory=dict)


# ── Limit price helpers ──────────────────────────────────────────────────────

def _compute_limit_price_cents(
    model_prob: float, kalshi_price: float, style: str,
) -> int:
    """Compute buy limit price. Maker posts at fair value; taker crosses spread."""
    if style == "maker":
        return max(1, min(99, int(model_prob * 100)))
    return max(1, min(99, int(kalshi_price * 100) + 1))


def _compute_sell_price_cents(
    model_prob: float, kalshi_price: float, style: str,
) -> int:
    """Compute sell limit price. Maker posts at fair value; taker floors 2% below bid."""
    if style == "maker":
        return max(1, min(99, int(model_prob * 100)))
    return max(1, int(kalshi_price * 100 * 0.98))


# ── Circuit breakers ──────────────────────────────────────────────────────────

def _check_latency(signal_ts: float) -> tuple:
    """(ok: bool, reason: str). Fails if signal age > _LATENCY_LIMIT seconds."""
    age = time.time() - signal_ts
    if age > _LATENCY_LIMIT:
        return False, f"stale signal: {age:.2f}s > {_LATENCY_LIMIT}s limit"
    return True, ""


def _check_wash_trade(ticker: str, action: str, state: TraderState) -> tuple:
    """
    (ok: bool, reason: str).

    Rejects if:
    - We're trying to BUY a ticker we just acted on within the cooldown window.
    - We're trying to SELL a ticker we just acted on within the cooldown window.

    A BUY after a recent SELL (or vice versa) on the same ticker is the classic
    wash-trade pattern we want to prevent.
    """
    last_ts = state.last_action_ts.get(ticker)
    if last_ts is not None:
        elapsed = time.time() - last_ts
        if elapsed < _WASH_COOLDOWN:
            return False, (
                f"wash-trade guard: {ticker} acted {elapsed:.2f}s ago "
                f"(cooldown={_WASH_COOLDOWN}s)"
            )
    return True, ""


async def _check_exposure(
    game_id: str,
    team: str,
    new_stake: float,
    db,
) -> tuple:
    """
    (ok: bool, reason: str).

    Queries live_positions to sum existing exposure for this game+team.
    Rejects if adding new_stake would exceed _MAX_EXPOSURE.
    """
    def _query():
        try:
            with db.get_conn() as conn:
                row = conn.execute(
                    "SELECT COALESCE(SUM(total_stake_usd), 0) FROM live_positions "
                    "WHERE game_id = %s AND team = %s",
                    (game_id, team),
                ).fetchone()
                return float(row[0]) if row else 0.0
        except Exception as e:
            logger.warning(f"kalshi_trader: exposure query failed: {e}")
            return 0.0

    existing = await asyncio.to_thread(_query)
    if existing + new_stake > _MAX_EXPOSURE:
        return False, (
            f"exposure cap: {team} in game {game_id} — "
            f"existing=${existing:.2f} + new=${new_stake:.2f} "
            f"> limit=${_MAX_EXPOSURE:.2f}"
        )
    return True, ""


# ── DB helpers ────────────────────────────────────────────────────────────────

def _upsert_position_sync(
    ticker: str,
    game_id: str,
    team: str,
    side: str,
    contracts: int,
    price_cents: int,
    stake_usd: float,
    order_id: str,
    db,
) -> None:
    """
    INSERT ... ON CONFLICT DO UPDATE for live_positions.

    Weighted average price is recalculated on additional buys into the same
    ticker (e.g. scaling in).
    """
    try:
        with db.get_conn() as conn:
            conn.execute(
                """
                INSERT INTO live_positions
                    (ticker, game_id, team, side, contracts, avg_price_cents,
                     total_stake_usd, order_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker) DO UPDATE SET
                    contracts = live_positions.contracts + EXCLUDED.contracts,
                    avg_price_cents = (
                        live_positions.avg_price_cents * live_positions.contracts
                        + EXCLUDED.avg_price_cents * EXCLUDED.contracts
                    ) / (live_positions.contracts + EXCLUDED.contracts),
                    total_stake_usd = live_positions.total_stake_usd + EXCLUDED.total_stake_usd,
                    order_id = EXCLUDED.order_id
                """,
                (ticker, game_id, team, side, contracts, price_cents, stake_usd, order_id),
            )
    except Exception as e:
        logger.error(f"kalshi_trader: failed to upsert live_positions for {ticker}: {e}")


def _delete_position_sync(ticker: str, db) -> None:
    """Remove a closed position from live_positions."""
    try:
        with db.get_conn() as conn:
            conn.execute("DELETE FROM live_positions WHERE ticker = %s", (ticker,))
    except Exception as e:
        logger.error(f"kalshi_trader: failed to delete live_positions for {ticker}: {e}")


def _reduce_position_sync(ticker: str, filled_contracts: int, db) -> None:
    """Reduce contracts in live_positions by filled_contracts. Delete if zero."""
    try:
        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT contracts FROM live_positions WHERE ticker = %s",
                (ticker,),
            ).fetchone()
            if not row:
                return
            remaining = row[0] - filled_contracts
            if remaining <= 0:
                conn.execute("DELETE FROM live_positions WHERE ticker = %s", (ticker,))
            else:
                conn.execute(
                    """UPDATE live_positions
                       SET contracts = %s,
                           total_stake_usd = total_stake_usd * (%s::real / (%s + %s)::real)
                       WHERE ticker = %s""",
                    (remaining, remaining, remaining, filled_contracts, ticker),
                )
    except Exception as e:
        logger.error(f"kalshi_trader: failed to reduce position for {ticker}: {e}")


def _get_position_sync(ticker: str, db) -> Optional[dict]:
    """Fetch the open position row for ticker, or None."""
    try:
        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT contracts, avg_price_cents, side FROM live_positions WHERE ticker = %s",
                (ticker,),
            ).fetchone()
            if row:
                return {
                    "contracts":       row[0],
                    "avg_price_cents": row[1],
                    "side":            row[2],
                }
    except Exception as e:
        logger.warning(f"kalshi_trader: position lookup failed for {ticker}: {e}")
    return None


def _insert_trade_sync(
    ticker: str,
    game_id: str,
    team: str,
    side: str,
    action: str,
    contracts: int,
    price_cents: int,
    stake_usd: float,
    model_prob: float,
    kalshi_price: float,
    edge: float,
    order_id: Optional[str],
    session_id: str,
    rejection_reason: Optional[str],
    realized_pnl: Optional[float],
    db,
) -> None:
    try:
        with db.get_conn() as conn:
            conn.execute(
                """
                INSERT INTO live_trades
                    (ticker, game_id, team, side, action, contracts, price_cents,
                     stake_usd, model_prob, kalshi_price, edge, order_id,
                     session_id, rejection_reason, realized_pnl)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    ticker, game_id, team, side, action, contracts, price_cents,
                    stake_usd, model_prob, kalshi_price, edge, order_id,
                    session_id, rejection_reason, realized_pnl,
                ),
            )
    except Exception as e:
        logger.error(f"kalshi_trader: failed to insert live_trade for {ticker}: {e}")


# ── Rejection logger ──────────────────────────────────────────────────────────

async def _log_rejection(sig: dict, reason: str, db, state: TraderState) -> None:
    logger.warning(
        f"kalshi_trader: REJECTED {sig.get('action','?').upper()} "
        f"{sig.get('ticker','?')} — {reason}"
    )
    await asyncio.to_thread(
        _insert_trade_sync,
        sig.get("ticker", ""),
        sig.get("game_id", ""),
        sig.get("team", ""),
        sig.get("side", "yes"),
        "rejected",
        0, 0, 0.0,
        sig.get("model_prob",   0.0),
        sig.get("kalshi_price", 0.0),
        sig.get("edge",         0.0),
        None,
        state.session_id,
        reason,
        None,
        db,
    )


# ── BUY execution ─────────────────────────────────────────────────────────────

async def execute_buy(
    sig: dict,
    state: TraderState,
    exchange: ExchangeClient,
    db,
) -> bool:
    """
    Run circuit breakers then place a live BUY order on Kalshi.
    Returns True on successful execution.
    """
    ticker     = sig["ticker"]
    game_id    = sig["game_id"]
    team       = sig["team"]
    side       = sig["side"]
    stake_usd  = sig["stake_usd"]
    model_prob = sig["model_prob"]
    kalshi_price = sig["kalshi_price"]
    edge       = sig["edge"]

    # ── Circuit breakers ──────────────────────────────────────────────
    ok, reason = _check_latency(sig["ts"])
    if not ok:
        await _log_rejection(sig, reason, db, state)
        return False

    ok, reason = _check_wash_trade(ticker, "buy", state)
    if not ok:
        await _log_rejection(sig, reason, db, state)
        return False

    # ── Portfolio Risk Manager (Covariance Sizing) ────────────────────
    risk_mgr = PortfolioRiskManager(db, bankroll=float(os.getenv("BANKROLL", "1000.0")))
    proposed = ProposedTrade(
        ticker=ticker,
        game_id=game_id,
        team=team,
        side=side,
        market_type=sig.get("market_type", "unknown"),
        player_name=sig.get("player_name"),
        model_prob=model_prob,
        kalshi_price=kalshi_price,
        edge=edge,
        raw_kelly_stake=stake_usd,
        score_diff=sig.get("score_diff", 0),
        period=sig.get("period", 1),
        minutes_elapsed=sig.get("minutes_elapsed", 0.0),
    )
    sizing = await risk_mgr.size_position(proposed)
    
    if sizing.adjusted_stake <= 0:
        await _log_rejection(sig, f"portfolio risk: {sizing.reason}", db, state)
        return False
        
    stake_usd = sizing.adjusted_stake

    ok, reason = await _check_exposure(game_id, team, stake_usd, db)
    if not ok:
        await _log_rejection(sig, reason, db, state)
        return False

    # ── Order placement ───────────────────────────────────────────────
    limit_cents = _compute_limit_price_cents(model_prob, kalshi_price, _EXECUTION_STYLE)

    if _EXECUTION_MODE == "live" and _EXECUTION_STYLE == "l2":
        # ── L2 Order Book Execution (TWAP/Passive) ────────────────────
        l2_engine = _get_l2_engine(exchange)
        price_frac = limit_cents / 100.0
        total_contracts = max(1, int(stake_usd / price_frac))

        result = await l2_engine.execute_order(
            ticker=ticker, side=side, action="buy",
            total_contracts=total_contracts,
            model_prob=model_prob, kalshi_price=kalshi_price, edge=edge,
        )

        if result.total_contracts_filled == 0:
            reason = f"L2 execution: 0 fills ({result.strategy.value} strategy)"
            await _log_rejection(sig, reason, db, state)
            return False

        contracts = result.total_contracts_filled
        fill_price_cents = int(result.avg_fill_price_cents)
        actual_cost = contracts * (fill_price_cents / 100.0)
        order_id = result.tranches[0].order_id if result.tranches else f"l2-{int(time.time())}"

        # L2 fills are confirmed immediately — persist position
        await asyncio.to_thread(
            _upsert_position_sync,
            ticker, game_id, team, side, contracts, fill_price_cents, actual_cost, order_id, db,
        )
        await asyncio.to_thread(
            _insert_trade_sync,
            ticker, game_id, team, side, "buy", contracts, fill_price_cents, actual_cost,
            model_prob, kalshi_price, edge, order_id, state.session_id, None, None, db,
        )
        logger.info(
            f"kalshi_trader [l2]: BUY executed — {ticker} {side.upper()} "
            f"{fill_price_cents}¢ x {contracts} contracts (${actual_cost:.2f}) "
            f"strategy={result.strategy.value} fill_rate={result.fill_rate:.0%} "
            f"duration={result.total_duration_seconds:.1f}s"
        )
        state.last_action_ts[ticker] = time.time()
        return True

    elif _EXECUTION_MODE == "live":
        order = await asyncio.to_thread(
            exchange.place_order, ticker, side, limit_cents, stake_usd
        )
    else:
        # Paper mode — simulate a fill at the signal price.
        logger.info(f"kalshi_trader [paper]: BUY {ticker} {side} {limit_cents}¢ ${stake_usd:.2f}")
        order = {"order": {"order_id": f"paper-{int(time.time())}", "count": max(1, int(stake_usd / (limit_cents / 100)))}}

    if order is None:
        reason = "exchange rejected order (returned None)"
        await _log_rejection(sig, reason, db, state)
        return False

    order_data = order.get("order", {})
    order_id   = str(order_data.get("order_id", ""))
    contracts  = int(order_data.get("count", 0)) or max(1, int(stake_usd / (limit_cents / 100)))
    actual_cost = contracts * (limit_cents / 100.0)

    # ── Persist ───────────────────────────────────────────────────────
    if _EXECUTION_MODE == "live":
        # Defer position update — fill polling will confirm actual fills
        pending = PendingOrder(
            order_id=order_id, ticker=ticker, game_id=game_id, team=team,
            side=side, action="buy", total_contracts=contracts,
            price_cents=limit_cents, model_prob=model_prob,
            kalshi_price=kalshi_price, edge=edge, stake_usd=stake_usd,
            avg_entry_cents=None, submitted_at=time.time(),
        )
        state.pending_orders[order_id] = pending
        logger.info(
            f"kalshi_trader [live]: BUY submitted — {ticker} {side.upper()} "
            f"{limit_cents}¢ x {contracts} contracts (${actual_cost:.2f}) "
            f"order_id={order_id} — tracking for fill"
        )
    else:
        # Paper mode — assume immediate full fill
        await asyncio.to_thread(
            _upsert_position_sync,
            ticker, game_id, team, side, contracts, limit_cents, actual_cost, order_id, db,
        )
        await asyncio.to_thread(
            _insert_trade_sync,
            ticker, game_id, team, side, "buy", contracts, limit_cents, actual_cost,
            model_prob, kalshi_price, edge, order_id, state.session_id, None, None, db,
        )
        logger.info(
            f"kalshi_trader [paper]: BUY executed — {ticker} {side.upper()} "
            f"{limit_cents}¢ x {contracts} contracts (${actual_cost:.2f}) "
            f"order_id={order_id}"
        )

    state.last_action_ts[ticker] = time.time()
    return True


# ── SELL execution ────────────────────────────────────────────────────────────

async def execute_sell(
    sig: dict,
    state: TraderState,
    exchange: ExchangeClient,
    db,
) -> bool:
    """
    Run latency + wash-trade checks, then place a live SELL order on Kalshi.
    Computes realized P&L against the average entry price in live_positions.
    Returns True on successful execution.
    """
    ticker      = sig["ticker"]
    game_id     = sig["game_id"]
    team        = sig["team"]
    side        = sig["side"]
    model_prob  = sig["model_prob"]
    kalshi_price = sig["kalshi_price"]
    edge        = sig["edge"]

    # ── Circuit breakers (no exposure check — we're reducing risk) ────
    ok, reason = _check_latency(sig["ts"])
    if not ok:
        await _log_rejection(sig, reason, db, state)
        return False

    ok, reason = _check_wash_trade(ticker, "sell", state)
    if not ok:
        await _log_rejection(sig, reason, db, state)
        return False

    # ── Look up open position ──────────────────────────────────────────
    position = await asyncio.to_thread(_get_position_sync, ticker, db)
    if position is None:
        reason = f"no open position found for {ticker}"
        await _log_rejection(sig, reason, db, state)
        return False

    contracts       = position["contracts"]
    avg_price_cents = position["avg_price_cents"]
    open_side       = position["side"]

    # Protect against crossing the market — sell the side we own.
    if open_side != side:
        logger.warning(
            f"kalshi_trader: SELL signal side={side} but position side={open_side} "
            f"for {ticker} — adjusting to {open_side}"
        )
        side = open_side

    min_price_cents = _compute_sell_price_cents(model_prob, kalshi_price, _EXECUTION_STYLE)

    if _EXECUTION_MODE == "live" and _EXECUTION_STYLE == "l2":
        # ── L2 Order Book Execution for SELL ──────────────────────────
        l2_engine = _get_l2_engine(exchange)
        result = await l2_engine.execute_order(
            ticker=ticker, side=side, action="sell",
            total_contracts=contracts,
            model_prob=model_prob, kalshi_price=kalshi_price, edge=edge,
        )

        if result.total_contracts_filled == 0:
            reason = f"L2 sell execution: 0 fills ({result.strategy.value} strategy)"
            await _log_rejection(sig, reason, db, state)
            return False

        filled = result.total_contracts_filled
        fill_cents = int(result.avg_fill_price_cents)
        realized_pnl = (fill_cents - avg_price_cents) / 100.0 * filled

        if filled >= contracts:
            await asyncio.to_thread(_delete_position_sync, ticker, db)
        else:
            await asyncio.to_thread(_reduce_position_sync, ticker, filled, db)

        order_id = result.tranches[0].order_id if result.tranches else f"l2-sell-{int(time.time())}"
        await asyncio.to_thread(
            _insert_trade_sync,
            ticker, game_id, team, side, "sell", filled, fill_cents,
            0.0, model_prob, kalshi_price, edge, order_id,
            state.session_id, None, realized_pnl, db,
        )
        logger.info(
            f"kalshi_trader [l2]: SELL executed — {ticker} {side.upper()} "
            f"{fill_cents}¢ x {filled} contracts | "
            f"avg_entry={avg_price_cents}¢ | realized_pnl=${realized_pnl:+.2f} "
            f"strategy={result.strategy.value} duration={result.total_duration_seconds:.1f}s"
        )
        state.last_action_ts[ticker] = time.time()
        return True

    elif _EXECUTION_MODE == "live":
        order = await asyncio.to_thread(
            exchange.sell_position, ticker, side, contracts, min_price_cents
        )
    else:
        logger.info(
            f"kalshi_trader [paper]: SELL {ticker} {side} {min_price_cents}¢ "
            f"x {contracts} contracts"
        )
        order = {"order": {"order_id": f"paper-sell-{int(time.time())}", "count": contracts}}

    if order is None:
        reason = "exchange rejected sell order (returned None)"
        await _log_rejection(sig, reason, db, state)
        return False

    order_data = order.get("order", {})
    order_id   = str(order_data.get("order_id", ""))

    # ── Persist ───────────────────────────────────────────────────────
    if _EXECUTION_MODE == "live":
        # Defer position update — fill polling will confirm actual fills
        pending = PendingOrder(
            order_id=order_id, ticker=ticker, game_id=game_id, team=team,
            side=side, action="sell", total_contracts=contracts,
            price_cents=min_price_cents, model_prob=model_prob,
            kalshi_price=kalshi_price, edge=edge, stake_usd=0.0,
            avg_entry_cents=avg_price_cents, submitted_at=time.time(),
        )
        state.pending_orders[order_id] = pending
        logger.info(
            f"kalshi_trader [live]: SELL submitted — {ticker} {side.upper()} "
            f"{min_price_cents}¢ x {contracts} contracts | "
            f"avg_entry={avg_price_cents}¢ | order_id={order_id} — tracking for fill"
        )
    else:
        # Paper mode — assume immediate full fill
        fill_cents = min_price_cents
        realized_pnl = (fill_cents - avg_price_cents) / 100.0 * contracts
        await asyncio.to_thread(_delete_position_sync, ticker, db)
        await asyncio.to_thread(
            _insert_trade_sync,
            ticker, game_id, team, side, "sell", contracts, fill_cents,
            0.0, model_prob, kalshi_price, edge, order_id,
            state.session_id, None, realized_pnl, db,
        )
        logger.info(
            f"kalshi_trader [paper]: SELL executed — {ticker} {side.upper()} "
            f"{fill_cents}¢ x {contracts} contracts | "
            f"avg_entry={avg_price_cents}¢ | realized_pnl=${realized_pnl:+.2f} "
            f"order_id={order_id}"
        )

    state.last_action_ts[ticker] = time.time()
    return True


# ── Partial fill processing ───────────────────────────────────────────────────

async def _process_fill(
    pending: PendingOrder,
    filled_count: int,
    status: str,
    state: TraderState,
    db,
) -> None:
    """Process a (partial or full) fill for a pending order."""
    if filled_count <= 0:
        return

    if pending.action == "buy":
        actual_cost = filled_count * (pending.price_cents / 100.0)
        await asyncio.to_thread(
            _upsert_position_sync,
            pending.ticker, pending.game_id, pending.team, pending.side,
            filled_count, pending.price_cents, actual_cost, pending.order_id, db,
        )
        await asyncio.to_thread(
            _insert_trade_sync,
            pending.ticker, pending.game_id, pending.team, pending.side,
            "buy", filled_count, pending.price_cents, actual_cost,
            pending.model_prob, pending.kalshi_price, pending.edge,
            pending.order_id, state.session_id, None, None, db,
        )
        logger.info(
            f"kalshi_trader: BUY fill — {pending.ticker} {filled_count} contracts "
            f"at {pending.price_cents}¢ (status={status})"
        )

    elif pending.action == "sell":
        avg_entry = pending.avg_entry_cents or 0
        realized_pnl = (pending.price_cents - avg_entry) / 100.0 * filled_count
        await asyncio.to_thread(_reduce_position_sync, pending.ticker, filled_count, db)
        await asyncio.to_thread(
            _insert_trade_sync,
            pending.ticker, pending.game_id, pending.team, pending.side,
            "sell", filled_count, pending.price_cents, 0.0,
            pending.model_prob, pending.kalshi_price, pending.edge,
            pending.order_id, state.session_id, None, realized_pnl, db,
        )
        logger.info(
            f"kalshi_trader: SELL fill — {pending.ticker} {filled_count} contracts "
            f"realized_pnl=${realized_pnl:+.2f} (status={status})"
        )


async def _poll_pending_orders(
    state: TraderState,
    exchange: ExchangeClient,
    db,
    shutdown: asyncio.Event,
) -> None:
    """Background task that polls pending order status and processes fills."""
    while not shutdown.is_set():
        try:
            await asyncio.sleep(_ORDER_POLL_INTERVAL)
        except asyncio.CancelledError:
            break

        if not state.pending_orders:
            continue

        completed = []
        for order_id, pending in list(state.pending_orders.items()):
            try:
                status_data = await asyncio.to_thread(
                    exchange.get_order_status, order_id
                )
                if status_data is None:
                    continue

                status = status_data.get("status", "")
                total = status_data.get("count", pending.total_contracts)
                remaining = status_data.get("remaining_count", total)
                filled = total - remaining

                if status == "executed":
                    new_fills = filled - pending.filled_so_far
                    await _process_fill(pending, new_fills, status, state, db)
                    completed.append(order_id)

                elif status == "canceled":
                    new_fills = filled - pending.filled_so_far
                    if new_fills > 0:
                        await _process_fill(pending, new_fills, status, state, db)
                    else:
                        logger.info(f"kalshi_trader: order {order_id} cancelled with 0 fills")
                    completed.append(order_id)

                elif status == "resting":
                    new_fills = filled - pending.filled_so_far
                    if new_fills > 0:
                        await _process_fill(pending, new_fills, "partial", state, db)
                        pending.filled_so_far = filled

                    # Cancel stale orders
                    age = time.time() - pending.submitted_at
                    if age > _ORDER_POLL_TIMEOUT:
                        logger.info(
                            f"kalshi_trader: order {order_id} stale ({age:.0f}s) — cancelling"
                        )
                        await asyncio.to_thread(exchange.cancel_order, order_id)

            except Exception as e:
                logger.error(f"kalshi_trader: error polling order {order_id}: {e}")

        for oid in completed:
            del state.pending_orders[oid]


# ── Main event loop ───────────────────────────────────────────────────────────

async def main() -> None:
    """
    1. Set up the LIVE_EXECUTION_QUEUE → asyncio.Queue bridge.
    2. Initialise ExchangeClient, DB, TraderState.
    3. Install SIGINT/SIGTERM shutdown handler.
    4. Loop: consume signals and dispatch to execute_buy / execute_sell.
    """
    bus = get_bus()
    if not bus.is_available():
        logger.warning(
            "kalshi_trader: Redis unavailable — no signals will be received. "
            "Set REDIS_URL to enable."
        )

    loop  = asyncio.get_event_loop()
    queue, _sub_thread = make_event_bridge(EventBus.LIVE_EXECUTION_QUEUE, loop)

    exchange = ExchangeClient()
    db       = get_db_client()
    state    = TraderState()

    if not exchange.enabled:
        logger.warning(
            "kalshi_trader: ExchangeClient disabled (no KALSHI_API_KEY). "
            "Orders will be paper-traded regardless of EXECUTION_MODE."
        )

    shutdown = asyncio.Event()

    def _handle_signal(sig, _frame):
        logger.info(f"kalshi_trader: received {signal.Signals(sig).name} — shutting down.")
        shutdown.set()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        f"kalshi_trader: started in {_EXECUTION_MODE.upper()} mode "
        f"(style={_EXECUTION_STYLE}, max_exposure=${_MAX_EXPOSURE:.0f}, "
        f"latency_limit={_LATENCY_LIMIT}s, "
        f"wash_cooldown={_WASH_COOLDOWN}s)"
    )

    # Background fill-polling task (only meaningful in live mode)
    poll_task = asyncio.create_task(
        _poll_pending_orders(state, exchange, db, shutdown)
    )

    while not shutdown.is_set():
        try:
            sig = await asyncio.wait_for(queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break

        action = sig.get("action", "")
        try:
            if action == "buy":
                await execute_buy(sig, state, exchange, db)
            elif action == "sell":
                await execute_sell(sig, state, exchange, db)
            else:
                logger.warning(f"kalshi_trader: unknown signal action '{action}' — ignoring")
        except Exception as e:
            logger.error(f"kalshi_trader: unhandled error on {action} {sig.get('ticker','?')}: {e}")

    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass

    logger.info("kalshi_trader: stopped.")


if __name__ == "__main__":
    asyncio.run(main())
