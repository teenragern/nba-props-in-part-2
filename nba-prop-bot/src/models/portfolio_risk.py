"""
Portfolio-Level Covariance Sizing — Markowitz-Kelly Risk Manager.

Upgrades scalar Kelly sizing to a portfolio-aware risk manager that:
  1. Reads currently open positions from the live_positions table
  2. Builds a covariance matrix using correlation data from:
     - sgp_correlations (same-player, same-game)
     - cross_player_correlations (same-team, different players)
     - cross_team_correlations (opposing teams)
  3. Solves a constrained Markowitz-Kelly optimization to find the
     optimal position size that maximizes edge while controlling
     portfolio-level risk
  4. Applies hard concentration limits (per-game, per-team, total)
  5. Implements blowout detection dampening

Mathematical Framework:
    maximize:  f^T * mu - (lambda/2) * f^T * Sigma * f
    subject to:
        sum(f) <= MAX_TOTAL_EXPOSURE
        per-game exposure <= MAX_PER_GAME
        per-team exposure <= MAX_PER_TEAM
        f_new >= 0 (can only go long)
        f_existing = fixed (existing positions are sunk costs)

Usage:
    from src.models.portfolio_risk import PortfolioRiskManager, ProposedTrade

    risk_mgr = PortfolioRiskManager(db, bankroll=10000.0)
    sizing = await risk_mgr.size_position(proposed_trade)
    final_stake = sizing.adjusted_stake
"""

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import minimize

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

_MAX_TOTAL_EXPOSURE_PCT = float(os.getenv("PORTFOLIO_MAX_EXPOSURE_PCT", "0.20"))
_MAX_PER_GAME_PCT = 0.10
_MAX_PER_TEAM_PCT = 0.08
_RISK_AVERSION_LAMBDA = float(os.getenv("PORTFOLIO_RISK_LAMBDA", "2.0"))
_BLOWOUT_SCORE_DIFF = 15     # Points differential to trigger blowout dampening
_BLOWOUT_PERIOD_THRESHOLD = 3  # Only dampen in Q3+ (period >= 3)
_BLOWOUT_DAMPENING = 0.50    # Reduce position by 50% during blowouts
_MIN_STAKE = 0.50            # Minimum viable position in USD


# ── Default Correlation Assumptions ──────────────────────────────────────────

# When no empirical data exists, use these structural defaults
_DEFAULT_CORRELATIONS = {
    "same_game_same_team": 0.35,   # Same game, same team props
    "same_game_diff_team": -0.15,  # Same game, opposing team props
    "same_team_diff_game": 0.05,   # Same team across different games
    "game_winner_vs_prop": 0.30,   # Game winner correlated with props
    "independent": 0.0,            # Different teams, different games
}


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    """Represents a single open position from live_positions."""
    ticker: str
    game_id: str
    team: str
    side: str                   # "yes" or "no"
    contracts: int
    avg_price_cents: int
    total_stake_usd: float
    market_type: str = "game_winner"  # "game_winner", "player_points", etc.
    player_name: Optional[str] = None
    current_price: float = 0.0  # Current market price (0-1)


@dataclass
class ProposedTrade:
    """A trade being considered by the engine."""
    ticker: str
    game_id: str
    team: str
    side: str
    market_type: str
    player_name: Optional[str]
    model_prob: float
    kalshi_price: float
    edge: float
    raw_kelly_stake: float      # Unconstrained Kelly stake from _kelly_stake()
    # Optional context for blowout detection
    score_diff: int = 0         # home - away
    period: int = 1
    minutes_elapsed: float = 0.0


@dataclass
class PortfolioSizingResult:
    """Output from the portfolio risk manager."""
    adjusted_stake: float
    raw_kelly_stake: float
    dampening_factor: float     # adjusted / raw (0-1)
    concentration_penalty: float
    covariance_penalty: float
    blowout_penalty: float
    reason: str

    @property
    def total_penalty(self) -> float:
        return 1.0 - self.dampening_factor


# ── Portfolio Risk Manager ───────────────────────────────────────────────────

class PortfolioRiskManager:
    """
    Portfolio-aware position sizing using covariance-adjusted Kelly.

    Prevents over-concentration:
      - Multiple "Over Points" bets on same team in a blowout
      - Correlated game-winner positions
      - Excessive total exposure to a single game/team
    """

    def __init__(self, db, bankroll: float):
        self.db = db
        self.bankroll = bankroll
        self._correlation_cache: Dict[Tuple[str, str], float] = {}

    async def size_position(self, proposed: ProposedTrade) -> PortfolioSizingResult:
        """
        Main entry point: compute covariance-adjusted position size.

        Steps:
          1. Load all open positions
          2. Check hard concentration limits
          3. Build covariance matrix (existing + proposed)
          4. Solve constrained optimization
          5. Apply blowout dampening
          6. Return final sized position

        Returns:
            PortfolioSizingResult with adjusted stake and explanations.
        """
        raw_stake = proposed.raw_kelly_stake

        # 1. Load open positions
        positions = self._load_open_positions()

        # 2. Check hard concentration limits first (fast rejection)
        conc_stake, conc_reason = self._check_concentration_limits(
            positions, proposed, raw_stake
        )
        concentration_penalty = 1.0 - (conc_stake / max(0.01, raw_stake))

        # 3. Build covariance matrix and solve optimization
        if positions and conc_stake > 0:
            cov_stake = self._covariance_adjusted_size(
                positions, proposed, conc_stake
            )
            covariance_penalty = 1.0 - (cov_stake / max(0.01, conc_stake))
        else:
            cov_stake = conc_stake
            covariance_penalty = 0.0

        # 4. Apply blowout dampening
        blowout_stake, blowout_penalty = self._apply_blowout_dampening(
            proposed, cov_stake
        )

        # 5. Final stake
        final_stake = max(0.0, blowout_stake)
        if final_stake < _MIN_STAKE and final_stake > 0:
            final_stake = 0.0  # Too small to execute

        dampening = final_stake / max(0.01, raw_stake)

        # Build reason
        reasons = []
        if concentration_penalty > 0.01:
            reasons.append(f"concentration:{conc_reason}")
        if covariance_penalty > 0.01:
            reasons.append(f"covariance:-{covariance_penalty:.0%}")
        if blowout_penalty > 0.01:
            reasons.append(f"blowout:-{blowout_penalty:.0%}")
        if not reasons:
            reasons.append("no dampening")

        return PortfolioSizingResult(
            adjusted_stake=round(final_stake, 2),
            raw_kelly_stake=raw_stake,
            dampening_factor=dampening,
            concentration_penalty=concentration_penalty,
            covariance_penalty=covariance_penalty,
            blowout_penalty=blowout_penalty,
            reason="; ".join(reasons),
        )

    # ── Position Loading ──────────────────────────────────────────────

    def _load_open_positions(self) -> List[OpenPosition]:
        """Load all open positions from live_positions table."""
        try:
            with self.db.get_conn() as conn:
                rows = conn.execute(
                    """SELECT ticker, game_id, team, side, contracts,
                              avg_price_cents, total_stake_usd
                       FROM live_positions
                       WHERE contracts > 0"""
                ).fetchall()

            positions = []
            for row in rows:
                ticker, game_id, team, side, contracts, avg_price, stake = row
                # Infer market type from ticker prefix
                market_type = self._infer_market_type(ticker)
                positions.append(OpenPosition(
                    ticker=ticker,
                    game_id=game_id,
                    team=team,
                    side=side,
                    contracts=contracts,
                    avg_price_cents=avg_price,
                    total_stake_usd=stake,
                    market_type=market_type,
                    current_price=avg_price / 100.0,
                ))
            return positions

        except Exception as e:
            logger.warning(f"portfolio_risk: failed to load positions: {e}")
            return []

    @staticmethod
    def _infer_market_type(ticker: str) -> str:
        """Infer market type from Kalshi ticker prefix."""
        ticker_upper = ticker.upper()
        if "KXNBAGAME" in ticker_upper:
            return "game_winner"
        elif "KXNBAPTS" in ticker_upper:
            return "player_points"
        elif "KXNBAREB" in ticker_upper:
            return "player_rebounds"
        elif "KXNBAAST" in ticker_upper:
            return "player_assists"
        elif "KXNBA3PT" in ticker_upper:
            return "player_threes"
        return "unknown"

    # ── Concentration Limits ──────────────────────────────────────────

    def _check_concentration_limits(
        self,
        positions: List[OpenPosition],
        proposed: ProposedTrade,
        raw_stake: float,
    ) -> Tuple[float, str]:
        """
        Apply hard concentration limits.

        Returns:
            (capped_stake, reason): stake after applying caps.
        """
        max_total = self.bankroll * _MAX_TOTAL_EXPOSURE_PCT
        max_per_game = self.bankroll * _MAX_PER_GAME_PCT
        max_per_team = self.bankroll * _MAX_PER_TEAM_PCT

        # Calculate current exposure
        total_exposure = sum(p.total_stake_usd for p in positions)
        game_exposure = sum(
            p.total_stake_usd for p in positions
            if p.game_id == proposed.game_id
        )
        team_exposure = sum(
            p.total_stake_usd for p in positions
            if p.team == proposed.team
        )

        # Apply caps
        capped = raw_stake
        reason = ""

        # Total exposure cap
        total_headroom = max(0, max_total - total_exposure)
        if capped > total_headroom:
            capped = total_headroom
            reason = f"total_cap(${max_total:.0f})"

        # Per-game cap
        game_headroom = max(0, max_per_game - game_exposure)
        if capped > game_headroom:
            capped = game_headroom
            reason = f"game_cap(${max_per_game:.0f})"

        # Per-team cap
        team_headroom = max(0, max_per_team - team_exposure)
        if capped > team_headroom:
            capped = team_headroom
            reason = f"team_cap(${max_per_team:.0f})"

        return max(0.0, capped), reason

    # ── Covariance-Adjusted Sizing ────────────────────────────────────

    def _covariance_adjusted_size(
        self,
        positions: List[OpenPosition],
        proposed: ProposedTrade,
        budget: float,
    ) -> float:
        """
        Apply Markowitz-Kelly covariance adjustment.

        For the proposed trade, compute the penalty from correlation
        with existing positions and reduce the stake accordingly.

        Uses the analytical solution for a single new position:
            f_adjusted = f_raw - lambda * sum(rho_ij * sigma_i * sigma_j * f_i)

        Where:
            rho_ij = correlation between proposed and position i
            sigma = sqrt(p * (1-p)) for binary contracts
            f_i = stake in position i / bankroll
        """
        if not positions:
            return budget

        # Compute variance of the proposed trade
        p_new = proposed.model_prob
        sigma_new = math.sqrt(p_new * (1.0 - p_new))

        # Compute correlation-weighted penalty
        correlation_cost = 0.0

        for pos in positions:
            rho = self._get_pairwise_correlation(pos, proposed)
            if abs(rho) < 0.01:
                continue  # negligible correlation

            p_i = pos.current_price if pos.current_price > 0 else pos.avg_price_cents / 100.0
            sigma_i = math.sqrt(max(0.01, p_i * (1.0 - p_i)))
            weight_i = pos.total_stake_usd / self.bankroll

            # Covariance contribution
            correlation_cost += rho * sigma_i * sigma_new * weight_i

        # Apply risk aversion penalty
        penalty_factor = _RISK_AVERSION_LAMBDA * correlation_cost
        dampening = max(0.0, 1.0 - penalty_factor)

        return budget * dampening

    def _get_pairwise_correlation(
        self,
        position: OpenPosition,
        proposed: ProposedTrade,
    ) -> float:
        """
        Determine correlation between an existing position and proposed trade.

        Correlation hierarchy:
          1. Same ticker → 1.0 (identity)
          2. Same game, same team → lookup sgp_correlations / default 0.35
          3. Same game, different teams → lookup cross_team / default -0.15
          4. Game winner + same-team prop → 0.30 (game script correlation)
          5. Same team, different game → 0.05 (weak schedule correlation)
          6. Different team, different game → 0.0 (independent)
        """
        # Cache check
        cache_key = (position.ticker, proposed.ticker)
        if cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]

        rho = self._compute_correlation(position, proposed)
        self._correlation_cache[cache_key] = rho
        return rho

    def _compute_correlation(self, pos: OpenPosition, proposed: ProposedTrade) -> float:
        """Compute structural correlation between two positions."""
        # Same ticker
        if pos.ticker == proposed.ticker:
            return 1.0

        same_game = pos.game_id == proposed.game_id
        same_team = pos.team == proposed.team

        if same_game and same_team:
            # Try DB lookup first
            db_corr = self._lookup_sgp_correlation(pos, proposed)
            if db_corr is not None:
                return db_corr
            return _DEFAULT_CORRELATIONS["same_game_same_team"]

        if same_game and not same_team:
            # Game winner vs prop on opposing team: negative correlation
            if pos.market_type == "game_winner" or proposed.market_type == "game_winner":
                return -0.25  # Opposing game outcomes
            db_corr = self._lookup_cross_team_correlation(pos, proposed)
            if db_corr is not None:
                return db_corr
            return _DEFAULT_CORRELATIONS["same_game_diff_team"]

        if same_team and not same_game:
            return _DEFAULT_CORRELATIONS["same_team_diff_game"]

        # Different teams, different games: independent
        return _DEFAULT_CORRELATIONS["independent"]

    def _lookup_sgp_correlation(self, pos: OpenPosition, proposed: ProposedTrade) -> Optional[float]:
        """Look up same-game player correlation from DB."""
        try:
            # Map market types to correlation table format
            market_a = pos.market_type.replace("player_", "")
            market_b = proposed.market_type.replace("player_", "")

            with self.db.get_conn() as conn:
                row = conn.execute(
                    """SELECT correlation FROM sgp_correlations
                       WHERE (market_a = %s AND market_b = %s)
                          OR (market_a = %s AND market_b = %s)
                       LIMIT 1""",
                    (market_a, market_b, market_b, market_a),
                ).fetchone()

            if row:
                return float(row[0])
        except Exception:
            pass
        return None

    def _lookup_cross_team_correlation(self, pos: OpenPosition, proposed: ProposedTrade) -> Optional[float]:
        """Look up cross-team correlation from DB."""
        try:
            market_a = pos.market_type.replace("player_", "")
            market_b = proposed.market_type.replace("player_", "")
            # Matchup key: sorted team pair
            teams = sorted([pos.team, proposed.team])
            matchup = f"{teams[0]}_vs_{teams[1]}"

            with self.db.get_conn() as conn:
                row = conn.execute(
                    """SELECT correlation FROM cross_team_correlations
                       WHERE matchup = %s
                         AND ((market_a = %s AND market_b = %s)
                           OR (market_a = %s AND market_b = %s))
                       LIMIT 1""",
                    (matchup, market_a, market_b, market_b, market_a),
                ).fetchone()

            if row:
                return float(row[0])
        except Exception:
            pass
        return None

    # ── Blowout Dampening ─────────────────────────────────────────────

    def _apply_blowout_dampening(
        self, proposed: ProposedTrade, stake: float
    ) -> Tuple[float, float]:
        """
        Detect blowout conditions and dampen "Over" prop positions.

        Rationale: In blowouts (score_diff > 15 in Q3+), the leading team's
        starters get pulled early, crushing "Over" prop expectations. The
        correlation between all "Over" props on the winning team spikes.

        Returns:
            (dampened_stake, blowout_penalty_factor)
        """
        if stake <= 0:
            return 0.0, 0.0

        # Only apply in Q3+ with significant lead
        if proposed.period < _BLOWOUT_PERIOD_THRESHOLD:
            return stake, 0.0

        abs_diff = abs(proposed.score_diff)
        if abs_diff < _BLOWOUT_SCORE_DIFF:
            return stake, 0.0

        # Determine if the proposed bet is an "Over" on the leading team
        is_prop = proposed.market_type.startswith("player_")
        if not is_prop:
            return stake, 0.0

        # Check if proposed team is leading (their starters getting pulled)
        # score_diff > 0 means home leading; check if proposed.team is home
        # This is a simplification — in production you'd check explicitly
        is_over = proposed.side == "yes"

        # Blowout dampening: reduce position for "Over" on any team in a blowout
        # (both teams affected: leading team starters pulled, trailing team in desperation mode)
        if is_over:
            penalty = _BLOWOUT_DAMPENING
            dampened = stake * (1.0 - penalty)
            return dampened, penalty

        return stake, 0.0

    # ── Full Portfolio Optimization (for large portfolios) ────────────

    def solve_portfolio_optimization(
        self,
        positions: List[OpenPosition],
        proposed: ProposedTrade,
        budget: float,
    ) -> float:
        """
        Full Markowitz-Kelly portfolio optimization using scipy.

        Solves:
            maximize:  f^T * mu - (lambda/2) * f^T * Sigma * f
            subject to: sum(f) <= budget, f[-1] >= 0

        This is the "heavy" path, used when the portfolio has > 5 positions
        and the analytical approximation in _covariance_adjusted_size may
        not be sufficient.

        Returns optimal stake for the proposed trade.
        """
        n = len(positions) + 1  # existing + proposed
        if n <= 2:
            # For small portfolios, the analytical method is sufficient
            return self._covariance_adjusted_size(positions, proposed, budget)

        # Build edge vector (mu)
        mu = np.zeros(n)
        for i, pos in enumerate(positions):
            # Existing positions: their edge is their cost basis vs current price
            p_i = pos.current_price if pos.current_price > 0 else pos.avg_price_cents / 100.0
            mu[i] = max(0, p_i - pos.avg_price_cents / 100.0)  # unrealized edge
        mu[-1] = proposed.edge  # new position's edge

        # Build covariance matrix
        sigma_mat = np.zeros((n, n))
        all_items = list(positions) + [proposed]

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    # Variance of binary contract
                    if i < len(positions):
                        p = positions[i].current_price or positions[i].avg_price_cents / 100.0
                    else:
                        p = proposed.model_prob
                    sigma_mat[i, j] = p * (1.0 - p)
                else:
                    # Covariance = rho * sigma_i * sigma_j
                    if i < len(positions) and j < len(positions):
                        # Both existing — use a simplified correlation
                        pi = positions[i]
                        pj = positions[j]
                        rho = self._compute_correlation_between_positions(pi, pj)
                    elif j == n - 1:
                        # Existing vs proposed
                        rho = self._get_pairwise_correlation(positions[i], proposed)
                    else:
                        rho = 0.0

                    p_i = (positions[i].current_price or positions[i].avg_price_cents / 100.0) if i < len(positions) else proposed.model_prob
                    p_j = (positions[j].current_price or positions[j].avg_price_cents / 100.0) if j < len(positions) else proposed.model_prob
                    sigma_i = math.sqrt(max(0.01, p_i * (1.0 - p_i)))
                    sigma_j = math.sqrt(max(0.01, p_j * (1.0 - p_j)))
                    sigma_mat[i, j] = rho * sigma_i * sigma_j
                    sigma_mat[j, i] = sigma_mat[i, j]

        # Objective: maximize f^T * mu - (lambda/2) * f^T * Sigma * f
        # scipy minimizes, so negate
        def objective(f):
            return -(f @ mu - (_RISK_AVERSION_LAMBDA / 2.0) * f @ sigma_mat @ f)

        # Constraints
        existing_stakes = np.array([p.total_stake_usd for p in positions])
        max_total = self.bankroll * _MAX_TOTAL_EXPOSURE_PCT

        # Bounds: existing positions fixed, new position bounded [0, budget]
        bounds = [(s, s) for s in existing_stakes] + [(0, budget)]

        constraints = [
            {"type": "ineq", "fun": lambda f: max_total - np.sum(f)},
        ]

        # Initial guess
        x0 = np.append(existing_stakes, budget * 0.5)

        try:
            result = minimize(
                objective, x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 100, "ftol": 1e-8},
            )
            if result.success:
                optimal_stake = max(0.0, result.x[-1])
                return min(optimal_stake, budget)
        except Exception as e:
            logger.warning(f"portfolio_risk: optimization failed: {e}")

        # Fallback to analytical method
        return self._covariance_adjusted_size(positions, proposed, budget)

    def _compute_correlation_between_positions(
        self, pos_a: OpenPosition, pos_b: OpenPosition
    ) -> float:
        """Compute correlation between two existing positions."""
        same_game = pos_a.game_id == pos_b.game_id
        same_team = pos_a.team == pos_b.team

        if pos_a.ticker == pos_b.ticker:
            return 1.0
        if same_game and same_team:
            return _DEFAULT_CORRELATIONS["same_game_same_team"]
        if same_game:
            return _DEFAULT_CORRELATIONS["same_game_diff_team"]
        if same_team:
            return _DEFAULT_CORRELATIONS["same_team_diff_game"]
        return 0.0
