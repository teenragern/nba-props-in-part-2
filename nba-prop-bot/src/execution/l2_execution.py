"""
L2 Order Book Execution Engine (TWAP / Passive Routing).

Institutional-grade execution that replaces naive spread-crossing with
intelligent order placement strategies:

  1. PASSIVE: Post at best bid (buy) / best ask (sell), capture maker rebates.
  2. TWAP: Split large orders into time-weighted tranches to minimize impact.
  3. AGGRESSIVE: Cross spread immediately when edge decay is critical.

Strategy selection is automatic based on order size and book state:
  - <= 5 contracts: PASSIVE (post and wait)
  - 6-20 contracts: TWAP (3 tranches over 30-60s)
  - > 20 contracts: TWAP (5 tranches over 2-5 min)

Edge Decay Monitor:
  While waiting for passive/TWAP fills, continuously monitors the model's
  edge vs. market price. If edge drops below EDGE_DECAY_CROSS_THRESHOLD,
  immediately crosses the spread to capture remaining alpha before it vanishes.

Usage:
    from src.execution.l2_execution import L2ExecutionEngine

    engine = L2ExecutionEngine(exchange_client)
    results = await engine.execute_order(
        ticker="KXNBAGAME-...", side="yes", action="buy",
        total_contracts=15, model_prob=0.62, kalshi_price=0.55, edge=0.07,
    )
"""

import asyncio
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

_EDGE_DECAY_CROSS_THRESHOLD = float(os.getenv("EDGE_DECAY_THRESHOLD", "0.03"))
_TWAP_MAX_DURATION = float(os.getenv("TWAP_MAX_DURATION_SECONDS", "120"))
_MAX_TRANCHE_WAIT = 30.0           # Max seconds to wait for a single tranche fill
_FILL_POLL_INTERVAL = 2.0          # Seconds between fill checks
_BOOK_REFRESH_INTERVAL = 5.0       # Seconds between L2 book refreshes
_PASSIVE_REPOST_INTERVAL = 10.0    # Re-post passive order at new best price


# ── Data Structures ──────────────────────────────────────────────────────────

class ExecutionStrategy(str, Enum):
    PASSIVE = "passive"
    TWAP = "twap"
    AGGRESSIVE = "aggressive"


@dataclass
class OrderBookLevel:
    """Single price level in the order book."""
    price_cents: int
    quantity: int


@dataclass
class L2Snapshot:
    """Parsed L2 order book state for a single market."""
    ticker: str
    yes_bids: List[OrderBookLevel]  # Descending price (best bid first)
    yes_asks: List[OrderBookLevel]  # Ascending price (best ask first)
    timestamp: float = field(default_factory=time.time)

    @property
    def best_yes_bid(self) -> int:
        """Best (highest) bid price for YES in cents."""
        return self.yes_bids[0].price_cents if self.yes_bids else 0

    @property
    def best_yes_ask(self) -> int:
        """Best (lowest) ask price for YES in cents."""
        return self.yes_asks[0].price_cents if self.yes_asks else 99

    @property
    def spread_cents(self) -> int:
        """Bid-ask spread in cents."""
        return max(0, self.best_yes_ask - self.best_yes_bid)

    @property
    def mid_price_cents(self) -> int:
        """Midpoint price in cents."""
        return (self.best_yes_bid + self.best_yes_ask) // 2

    @property
    def bid_depth(self) -> int:
        """Total quantity available at all bid levels."""
        return sum(level.quantity for level in self.yes_bids)

    @property
    def ask_depth(self) -> int:
        """Total quantity available at all ask levels."""
        return sum(level.quantity for level in self.yes_asks)


@dataclass
class TWAPPlan:
    """Execution plan for a TWAP order."""
    total_contracts: int
    n_tranches: int
    tranche_size: int
    remainder: int                    # extra contracts in last tranche
    interval_seconds: float
    max_duration_seconds: float
    price_limit_cents: int            # never pay more (buy) / accept less (sell)
    edge_decay_threshold: float


@dataclass
class TrancheResult:
    """Result of executing a single tranche."""
    tranche_idx: int
    contracts_requested: int
    contracts_filled: int
    avg_fill_price_cents: float
    order_id: str
    strategy_used: ExecutionStrategy
    was_aggressive: bool              # True if we crossed the spread
    duration_seconds: float


@dataclass
class ExecutionResult:
    """Aggregate result of a full order execution."""
    ticker: str
    side: str
    action: str
    total_contracts_requested: int
    total_contracts_filled: int
    avg_fill_price_cents: float
    strategy: ExecutionStrategy
    tranches: List[TrancheResult]
    total_duration_seconds: float

    @property
    def fill_rate(self) -> float:
        return self.total_contracts_filled / max(1, self.total_contracts_requested)


# ── L2 Execution Engine ──────────────────────────────────────────────────────

class L2ExecutionEngine:
    """
    Institutional execution engine with L2 order book awareness.

    Selects optimal strategy (passive/TWAP/aggressive) based on:
      - Order size relative to book depth
      - Current spread width
      - Available edge magnitude
    """

    def __init__(self, exchange):
        """
        Args:
            exchange: ExchangeClient instance with get_order_book(),
                     place_order(), get_order_status(), cancel_order() methods.
        """
        self.exchange = exchange

    async def execute_order(
        self,
        ticker: str,
        side: str,
        action: str,
        total_contracts: int,
        model_prob: float,
        kalshi_price: float,
        edge: float,
    ) -> ExecutionResult:
        """
        Execute an order using the optimal strategy for size and book state.

        Args:
            ticker: Kalshi market ticker.
            side: "yes" or "no".
            action: "buy" or "sell".
            total_contracts: Desired number of contracts.
            model_prob: Model's fair probability (0-1).
            kalshi_price: Current market price (0-1).
            edge: model_prob - kalshi_price (for buys).

        Returns:
            ExecutionResult with fill details.
        """
        start_time = time.time()

        # Fetch L2 order book
        l2 = await self._fetch_l2(ticker)
        if l2 is None:
            # Fallback to aggressive if can't read book
            logger.warning(f"l2_exec: no order book for {ticker}, falling back to aggressive")
            result = await self._execute_aggressive(
                ticker, side, action, total_contracts, model_prob
            )
            return ExecutionResult(
                ticker=ticker, side=side, action=action,
                total_contracts_requested=total_contracts,
                total_contracts_filled=result.contracts_filled,
                avg_fill_price_cents=result.avg_fill_price_cents,
                strategy=ExecutionStrategy.AGGRESSIVE,
                tranches=[result],
                total_duration_seconds=time.time() - start_time,
            )

        # Select strategy
        strategy = self._select_strategy(total_contracts, l2, edge)
        logger.info(
            f"l2_exec: {action} {total_contracts}x {ticker} @ strategy={strategy.value} "
            f"spread={l2.spread_cents}c bid_depth={l2.bid_depth} ask_depth={l2.ask_depth}"
        )

        if strategy == ExecutionStrategy.PASSIVE:
            tranche = await self._execute_passive(
                ticker, side, action, total_contracts, model_prob, l2
            )
            tranches = [tranche]

        elif strategy == ExecutionStrategy.TWAP:
            plan = self._build_twap_plan(total_contracts, l2, edge, model_prob)
            tranches = await self._execute_twap(
                ticker, side, action, plan, model_prob
            )

        else:  # AGGRESSIVE
            tranche = await self._execute_aggressive(
                ticker, side, action, total_contracts, model_prob
            )
            tranches = [tranche]

        # Aggregate results
        total_filled = sum(t.contracts_filled for t in tranches)
        if total_filled > 0:
            weighted_price = sum(
                t.avg_fill_price_cents * t.contracts_filled for t in tranches
            )
            avg_price = weighted_price / total_filled
        else:
            avg_price = 0.0

        return ExecutionResult(
            ticker=ticker, side=side, action=action,
            total_contracts_requested=total_contracts,
            total_contracts_filled=total_filled,
            avg_fill_price_cents=avg_price,
            strategy=strategy,
            tranches=tranches,
            total_duration_seconds=time.time() - start_time,
        )

    # ── Strategy Selection ────────────────────────────────────────────

    def _select_strategy(self, total_contracts: int, l2: L2Snapshot,
                         edge: float) -> ExecutionStrategy:
        """
        Select optimal execution strategy.

        Rules:
          - Locked market (spread=0): AGGRESSIVE (no maker advantage)
          - Edge > 15% (very strong): AGGRESSIVE (speed > cost)
          - Small order (<=5) AND reasonable spread: PASSIVE
          - Otherwise: TWAP
        """
        spread = l2.spread_cents

        # Locked market: no benefit to passive posting
        if spread <= 1:
            return ExecutionStrategy.AGGRESSIVE

        # Very strong edge: prioritize speed over cost
        if edge > 0.15:
            return ExecutionStrategy.AGGRESSIVE

        # Small orders: passive is efficient
        if total_contracts <= 5:
            return ExecutionStrategy.PASSIVE

        # Medium/large orders: TWAP to reduce impact
        return ExecutionStrategy.TWAP

    # ── TWAP Plan Builder ─────────────────────────────────────────────

    def _build_twap_plan(self, total_contracts: int, l2: L2Snapshot,
                         edge: float, model_prob: float) -> TWAPPlan:
        """Build TWAP execution plan based on order size and book state."""
        # Number of tranches: scale with order size
        if total_contracts <= 10:
            n_tranches = 2
        elif total_contracts <= 20:
            n_tranches = 3
        elif total_contracts <= 50:
            n_tranches = 4
        else:
            n_tranches = 5

        tranche_size = total_contracts // n_tranches
        remainder = total_contracts - (tranche_size * n_tranches)

        # Time distribution: spread over duration proportional to edge decay risk
        if edge > 0.10:
            # Strong edge: execute faster (30-60s)
            max_duration = min(60.0, _TWAP_MAX_DURATION)
        elif edge > 0.05:
            # Moderate edge: standard pace (60-120s)
            max_duration = min(120.0, _TWAP_MAX_DURATION)
        else:
            # Thin edge: even slower, more passive (full duration)
            max_duration = _TWAP_MAX_DURATION

        interval = max_duration / n_tranches

        # Price limit: never pay more than model fair value (in cents)
        price_limit = max(1, min(99, int(model_prob * 100)))

        return TWAPPlan(
            total_contracts=total_contracts,
            n_tranches=n_tranches,
            tranche_size=tranche_size,
            remainder=remainder,
            interval_seconds=interval,
            max_duration_seconds=max_duration,
            price_limit_cents=price_limit,
            edge_decay_threshold=_EDGE_DECAY_CROSS_THRESHOLD,
        )

    # ── Passive Execution ─────────────────────────────────────────────

    async def _execute_passive(
        self,
        ticker: str,
        side: str,
        action: str,
        contracts: int,
        model_prob: float,
        l2: L2Snapshot,
    ) -> TrancheResult:
        """
        Post limit order at the best bid (for buys) to capture maker rebate.
        Monitor for fill. If not filled within MAX_TRANCHE_WAIT, check edge
        and decide: re-post at new best, or cross the spread.
        """
        start = time.time()

        if action == "buy":
            # Post at best bid + 1 (just inside the spread to get priority)
            post_price = min(l2.best_yes_bid + 1, int(model_prob * 100))
        else:
            # Selling: post at best ask - 1
            post_price = max(l2.best_yes_ask - 1, int(model_prob * 100))

        post_price = max(1, min(99, post_price))

        # Place the passive order
        order = await asyncio.to_thread(
            self.exchange.place_order, ticker, side, post_price, None
        )
        if order is None:
            return TrancheResult(
                tranche_idx=0, contracts_requested=contracts,
                contracts_filled=0, avg_fill_price_cents=0.0,
                order_id="", strategy_used=ExecutionStrategy.PASSIVE,
                was_aggressive=False, duration_seconds=time.time() - start,
            )

        order_id = order.get("order_id", "")
        total_count = order.get("count", contracts)

        # Wait for fill with edge decay monitoring
        filled, was_aggressive = await self._wait_for_fill_or_cross(
            ticker, side, action, order_id, total_count, model_prob,
            timeout=_MAX_TRANCHE_WAIT,
        )

        return TrancheResult(
            tranche_idx=0, contracts_requested=contracts,
            contracts_filled=filled,
            avg_fill_price_cents=float(post_price),
            order_id=order_id,
            strategy_used=ExecutionStrategy.PASSIVE,
            was_aggressive=was_aggressive,
            duration_seconds=time.time() - start,
        )

    # ── TWAP Execution ────────────────────────────────────────────────

    async def _execute_twap(
        self,
        ticker: str,
        side: str,
        action: str,
        plan: TWAPPlan,
        model_prob: float,
    ) -> List[TrancheResult]:
        """
        Execute TWAP plan tranche by tranche.

        For each tranche:
          1. Fetch fresh L2 snapshot
          2. Post at best_bid + 1 (passive-aggressive)
          3. Wait interval for fill
          4. If not filled: check edge. If decayed → cross. If healthy → cancel, re-post
        """
        results: List[TrancheResult] = []

        for i in range(plan.n_tranches):
            tranche_start = time.time()

            # Last tranche gets the remainder
            contracts = plan.tranche_size + (plan.remainder if i == plan.n_tranches - 1 else 0)
            if contracts <= 0:
                continue

            # Refresh L2 book
            l2 = await self._fetch_l2(ticker)

            if l2 is None or l2.spread_cents <= 1:
                # Can't get book or locked: go aggressive for this tranche
                result = await self._execute_aggressive(
                    ticker, side, action, contracts, model_prob
                )
                result.tranche_idx = i
                results.append(result)
            else:
                # Post passively for this tranche
                if action == "buy":
                    post_price = min(l2.best_yes_bid + 1, plan.price_limit_cents)
                else:
                    post_price = max(l2.best_yes_ask - 1, plan.price_limit_cents)
                post_price = max(1, min(99, post_price))

                order = await asyncio.to_thread(
                    self.exchange.place_order, ticker, side, post_price, None
                )

                if order is None:
                    results.append(TrancheResult(
                        tranche_idx=i, contracts_requested=contracts,
                        contracts_filled=0, avg_fill_price_cents=0.0,
                        order_id="", strategy_used=ExecutionStrategy.TWAP,
                        was_aggressive=False,
                        duration_seconds=time.time() - tranche_start,
                    ))
                    continue

                order_id = order.get("order_id", "")
                total_count = order.get("count", contracts)

                # Wait for this tranche with edge monitoring
                wait_time = min(plan.interval_seconds, _MAX_TRANCHE_WAIT)
                filled, was_aggressive = await self._wait_for_fill_or_cross(
                    ticker, side, action, order_id, total_count, model_prob,
                    timeout=wait_time,
                )

                results.append(TrancheResult(
                    tranche_idx=i, contracts_requested=contracts,
                    contracts_filled=filled,
                    avg_fill_price_cents=float(post_price),
                    order_id=order_id,
                    strategy_used=ExecutionStrategy.TWAP,
                    was_aggressive=was_aggressive,
                    duration_seconds=time.time() - tranche_start,
                ))

            # Inter-tranche delay (skip for last tranche)
            if i < plan.n_tranches - 1:
                remaining_interval = plan.interval_seconds - (time.time() - tranche_start)
                if remaining_interval > 0:
                    await asyncio.sleep(remaining_interval)

        return results

    # ── Aggressive Execution ──────────────────────────────────────────

    async def _execute_aggressive(
        self,
        ticker: str,
        side: str,
        action: str,
        contracts: int,
        model_prob: float,
    ) -> TrancheResult:
        """Cross the spread immediately at the best available price."""
        start = time.time()

        # Fetch current best ask (for buys) or best bid (for sells)
        l2 = await self._fetch_l2(ticker)

        if l2 is not None:
            if action == "buy":
                cross_price = min(l2.best_yes_ask, int(model_prob * 100) + 2)
            else:
                cross_price = max(l2.best_yes_bid, int(model_prob * 100) - 2)
        else:
            # No book data: use model price + spread tolerance
            cross_price = int(model_prob * 100) + (1 if action == "buy" else -1)

        cross_price = max(1, min(99, cross_price))

        order = await asyncio.to_thread(
            self.exchange.place_order, ticker, side, cross_price, None
        )

        if order is None:
            return TrancheResult(
                tranche_idx=0, contracts_requested=contracts,
                contracts_filled=0, avg_fill_price_cents=0.0,
                order_id="", strategy_used=ExecutionStrategy.AGGRESSIVE,
                was_aggressive=True, duration_seconds=time.time() - start,
            )

        order_id = order.get("order_id", "")
        filled = order.get("count", 0)

        return TrancheResult(
            tranche_idx=0, contracts_requested=contracts,
            contracts_filled=filled,
            avg_fill_price_cents=float(cross_price),
            order_id=order_id,
            strategy_used=ExecutionStrategy.AGGRESSIVE,
            was_aggressive=True,
            duration_seconds=time.time() - start,
        )

    # ── Edge Decay Monitor + Fill Waiting ─────────────────────────────

    async def _wait_for_fill_or_cross(
        self,
        ticker: str,
        side: str,
        action: str,
        order_id: str,
        total_count: int,
        model_prob: float,
        timeout: float,
    ) -> Tuple[int, bool]:
        """
        Poll for fill while monitoring edge decay.

        Returns:
            (contracts_filled, was_aggressive): was_aggressive=True if we
            had to cancel and cross the spread due to edge decay.
        """
        start = time.time()
        was_aggressive = False
        filled_so_far = 0

        while (time.time() - start) < timeout:
            await asyncio.sleep(_FILL_POLL_INTERVAL)

            # Check fill status
            status = await asyncio.to_thread(
                self.exchange.get_order_status, order_id
            )
            if status is None:
                break

            order_status = status.get("status", "")
            remaining = status.get("remaining_count", total_count)
            filled_so_far = total_count - remaining

            if order_status == "executed":
                # Fully filled
                return filled_so_far, False

            if order_status == "canceled":
                return filled_so_far, was_aggressive

            # Check edge decay: has the market moved against us?
            edge_ok = await self._check_edge_health(ticker, model_prob, action)
            if not edge_ok:
                # Edge decayed below threshold — cancel passive, cross spread
                logger.info(
                    f"l2_exec: edge decay detected for {ticker}, "
                    f"crossing spread for remaining {remaining} contracts"
                )
                await asyncio.to_thread(self.exchange.cancel_order, order_id)
                was_aggressive = True

                if remaining > 0:
                    # Cross the spread for unfilled portion
                    cross_result = await self._execute_aggressive(
                        ticker, side, action, remaining, model_prob
                    )
                    filled_so_far += cross_result.contracts_filled

                return filled_so_far, True

        # Timeout: cancel remaining order
        if filled_so_far < total_count:
            await asyncio.to_thread(self.exchange.cancel_order, order_id)

        return filled_so_far, was_aggressive

    async def _check_edge_health(self, ticker: str, model_prob: float,
                                 action: str) -> bool:
        """
        Check if the model's edge is still above the decay threshold.

        Returns True if edge is healthy, False if it has decayed.
        """
        price_data = await asyncio.to_thread(
            self.exchange.get_market_price, ticker
        )
        if price_data is None:
            return True  # Can't check, assume healthy

        current_price = price_data.get("implied_yes", 0.0)
        if current_price <= 0:
            return True

        if action == "buy":
            current_edge = model_prob - current_price
        else:
            current_edge = current_price - model_prob

        return current_edge >= _EDGE_DECAY_CROSS_THRESHOLD

    # ── L2 Book Fetching ──────────────────────────────────────────────

    async def _fetch_l2(self, ticker: str) -> Optional[L2Snapshot]:
        """Fetch and parse L2 order book from exchange."""
        raw = await asyncio.to_thread(self.exchange.get_order_book, ticker)
        if raw is None:
            return None

        try:
            yes_bids = [
                OrderBookLevel(price_cents=int(lvl["price"]), quantity=int(lvl["quantity"]))
                for lvl in raw.get("yes", [])
                if lvl.get("price") and lvl.get("quantity")
            ]
            # Bids: descending price order
            yes_bids.sort(key=lambda x: x.price_cents, reverse=True)

            no_bids = [
                OrderBookLevel(price_cents=int(lvl["price"]), quantity=int(lvl["quantity"]))
                for lvl in raw.get("no", [])
                if lvl.get("price") and lvl.get("quantity")
            ]
            no_bids.sort(key=lambda x: x.price_cents, reverse=True)

            # YES asks = implied from NO bids (Kalshi binary: yes_ask = 100 - no_bid)
            yes_asks = [
                OrderBookLevel(price_cents=100 - lvl.price_cents, quantity=lvl.quantity)
                for lvl in no_bids
            ]
            yes_asks.sort(key=lambda x: x.price_cents)  # Ascending for asks

            return L2Snapshot(
                ticker=ticker,
                yes_bids=yes_bids,
                yes_asks=yes_asks,
                timestamp=time.time(),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"l2_exec: failed to parse order book for {ticker}: {e}")
            return None
