"""
State-Space Possession Win Probability Model.

Replaces the simple Normal-approximation time-decay model with a hybrid
approach:

  1. Enhanced Normal (early/mid-game, >6 min remaining):
     - Possession value adjustment (+PPP for team with ball)
     - Bonus/foul-state adjustment (higher PPP when opponent in penalty)
     - Pace-calibrated sigma (possessions-based, not time-based)

  2. Monte Carlo Endgame Simulation (<=6 min remaining):
     - Forward-simulates discrete remaining possessions
     - Samples scoring outcomes from a calibrated transition matrix
     - Accounts for bonus FT advantage, possession alternation, pace

This model produces significantly more accurate win probabilities in:
  - Close endgame scenarios (where Normal approximation breaks down)
  - Foul-heavy games (bonus state creates asymmetric scoring expectations)
  - Possession-critical moments (which team has the ball matters enormously)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm


# ── Configuration ────────────────────────────────────────────────────────────

# Empirical NBA constants
_LEAGUE_AVG_PPP = 1.10          # Points per possession (2023-24 season)
_LEAGUE_AVG_PACE = 100.0        # Possessions per 48 minutes
_PER_POSS_STD = 2.20            # Empirical std of scoring margin per possession
_NBA_SIGMA_FULL = 11.0          # Full-game scoring std (for fallback)

# Bonus state adjustments (extra expected points per possession when in bonus)
_BONUS_PPP_BOOST = 0.12         # ~0.12 extra PPP from bonus FT trips
_DOUBLE_BONUS_PPP_BOOST = 0.18  # ~0.18 extra PPP from double bonus

# Monte Carlo parameters
_SIMULATION_THRESHOLD_MIN = 6.0  # Switch to MC below this many minutes
_N_SIMULATIONS = 5000
_PROB_FLOOR = 0.02
_PROB_CEILING = 0.98


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class PossessionModelParams:
    """Pre-game calibrated parameters for the Markov model."""
    ppp_home: float = _LEAGUE_AVG_PPP
    ppp_away: float = _LEAGUE_AVG_PPP
    pace: float = _LEAGUE_AVG_PACE
    pre_game_prob: float = 0.50
    home_bonus_boost: float = _BONUS_PPP_BOOST
    away_bonus_boost: float = _BONUS_PPP_BOOST


@dataclass
class GameMicroState:
    """Enriched game state consumed by the possession model."""
    score_diff: int                 # home_score - away_score
    minutes_remaining: float        # 0.1 to 48.0
    possession: str = "unknown"     # "home", "away", "unknown"
    home_in_bonus: bool = False
    away_in_bonus: bool = False
    home_in_double_bonus: bool = False
    away_in_double_bonus: bool = False


# ── Transition Matrix Builder ────────────────────────────────────────────────

def _build_scoring_distribution(ppp: float, opponent_in_bonus: bool,
                                opponent_in_double_bonus: bool) -> np.ndarray:
    """
    Build a probability distribution over per-possession scoring outcomes.

    Returns array of shape (4,) representing P(score 0, 1, 2, 3 points).
    Calibrated so E[points] = ppp (with bonus adjustments).

    NBA empirical baselines (2023-24):
      P(3pt made on possession) ~ 0.12
      P(2pt made on possession) ~ 0.35
      P(FT-only points)         ~ 0.08
      P(0 pts: miss/TO)         ~ 0.45
    """
    # Base probabilities (calibrated to ~1.10 PPP)
    p3 = 0.120  # probability of scoring exactly 3
    p2 = 0.350  # probability of scoring exactly 2
    p1 = 0.065  # probability of scoring exactly 1 (single FT make, and-1 scenarios)
    p0 = 1.0 - p3 - p2 - p1  # ~0.465

    # Adjust for opponent's bonus state (team gets more FT opportunities
    # when they're in the bonus — this boosts P(1) and P(2))
    if opponent_in_double_bonus:
        # In double bonus: all fouls = 2 FTs, significant boost
        ft_boost = _DOUBLE_BONUS_PPP_BOOST
    elif opponent_in_bonus:
        # In bonus: non-shooting fouls still give 1-and-1, moderate boost
        ft_boost = _BONUS_PPP_BOOST
    else:
        ft_boost = 0.0

    if ft_boost > 0:
        # Redistribute: reduce P(0), increase P(1) and P(2)
        p1 += ft_boost * 0.4
        p2 += ft_boost * 0.3
        p0 -= ft_boost * 0.7
        p0 = max(0.20, p0)  # safety floor

    # Scale to match target PPP
    current_ev = 3 * p3 + 2 * p2 + 1 * p1
    target_ppp = ppp + ft_boost
    if current_ev > 0:
        scale = target_ppp / current_ev
        p3 *= scale
        p2 *= scale
        p1 *= scale
        p0 = max(0.10, 1.0 - p3 - p2 - p1)
        # Renormalize
        total = p0 + p1 + p2 + p3
        p0 /= total
        p1 /= total
        p2 /= total
        p3 /= total

    return np.array([p0, p1, p2, p3], dtype=np.float64)


# ── Possession Win Probability Model ────────────────────────────────────────

class PossessionWinModel:
    """
    Hybrid win probability model using Markov possession dynamics.

    For early/mid-game (>6 min remaining):
        Enhanced Normal approximation with possession value and bonus adjustments.

    For endgame (<=6 min remaining):
        Monte Carlo forward simulation over discrete remaining possessions.
    """

    def __init__(self, params: PossessionModelParams):
        self.params = params
        self._rng = np.random.default_rng(seed=42)
        # Pre-compute base transition distributions
        self._home_dist_base = _build_scoring_distribution(params.ppp_home, False, False)
        self._away_dist_base = _build_scoring_distribution(params.ppp_away, False, False)

    def compute_win_prob(self, state: GameMicroState) -> float:
        """
        Compute P(home wins) given current game micro-state.

        Dispatches to enhanced Normal approximation or Monte Carlo simulation
        based on remaining time.

        Returns: float in (0.02, 0.98)
        """
        minutes_remaining = max(0.1, state.minutes_remaining)

        if minutes_remaining > _SIMULATION_THRESHOLD_MIN:
            prob = self._enhanced_normal(state, minutes_remaining)
        else:
            prob = self._simulate_endgame(state, minutes_remaining)

        return max(_PROB_FLOOR, min(_PROB_CEILING, prob))

    def _enhanced_normal(self, state: GameMicroState,
                         minutes_remaining: float) -> float:
        """
        Enhanced Normal approximation incorporating:
          1. Score differential
          2. Possession value (team with ball gets +PPP adjustment)
          3. Bonus state differential (asymmetric scoring expectations)
          4. Pace-calibrated sigma (based on remaining possessions)
          5. Pre-game prior anchor

        Formula:
            remaining_poss = pace * (minutes_remaining / 48.0)
            sigma = sqrt(remaining_poss) * per_poss_std
            effective_diff = score_diff + possession_value + bonus_differential
            prior_adj = norm.ppf(pre_game_prob) * sigma
            win_prob = norm.cdf((effective_diff + prior_adj) / sigma)
        """
        pace = self.params.pace
        remaining_poss = pace * (minutes_remaining / 48.0)

        # Sigma: standard deviation of final margin based on remaining possessions
        # Each possession has ~2.2 point std, possessions are roughly independent
        sigma = math.sqrt(remaining_poss) * _PER_POSS_STD
        if sigma < 0.5:
            sigma = 0.5  # prevent degenerate near-zero sigma

        # Possession value: team with ball gets expected points advantage
        possession_value = 0.0
        if state.possession == "home":
            possession_value = self.params.ppp_home
        elif state.possession == "away":
            possession_value = -self.params.ppp_away

        # Bonus differential: projects additional scoring advantage over
        # remaining possessions from being in the bonus
        bonus_diff = 0.0
        remaining_fraction = minutes_remaining / 48.0
        # Home team benefit: when AWAY team is in bonus, home draws more FTs
        if state.away_in_double_bonus:
            bonus_diff += _DOUBLE_BONUS_PPP_BOOST * remaining_poss * 0.1
        elif state.away_in_bonus:
            bonus_diff += _BONUS_PPP_BOOST * remaining_poss * 0.1
        # Away team benefit: when HOME team is in bonus, away draws more FTs
        if state.home_in_double_bonus:
            bonus_diff -= _DOUBLE_BONUS_PPP_BOOST * remaining_poss * 0.1
        elif state.home_in_bonus:
            bonus_diff -= _BONUS_PPP_BOOST * remaining_poss * 0.1

        # Effective differential
        effective_diff = state.score_diff + possession_value + bonus_diff

        # Prior adjustment: anchors to pre-game probability
        pre_game_prob = max(0.01, min(0.99, self.params.pre_game_prob))
        prior_adj = norm.ppf(pre_game_prob) * sigma

        win_prob = float(norm.cdf((effective_diff + prior_adj) / sigma))
        return win_prob

    def _simulate_endgame(self, state: GameMicroState,
                          minutes_remaining: float) -> float:
        """
        Monte Carlo forward simulation over discrete remaining possessions.

        Algorithm:
          1. Estimate remaining possessions from pace and time
          2. For N simulations:
             a. Start with current score diff and possession
             b. Alternate possessions, sampling scoring from transition matrix
             c. Apply bonus-state adjusted distributions
             d. Record final outcome (home win/loss/tie)
          3. Return fraction of home wins (ties broken by pre-game prior)
        """
        pace = self.params.pace
        remaining_poss = max(1, int(round(pace * (minutes_remaining / 48.0))))

        # Build scoring distributions for current bonus state
        home_dist = _build_scoring_distribution(
            self.params.ppp_home,
            opponent_in_bonus=state.away_in_bonus,
            opponent_in_double_bonus=state.away_in_double_bonus,
        )
        away_dist = _build_scoring_distribution(
            self.params.ppp_away,
            opponent_in_bonus=state.home_in_bonus,
            opponent_in_double_bonus=state.home_in_double_bonus,
        )

        # Determine starting possession for simulation
        if state.possession == "home":
            home_starts = True
        elif state.possession == "away":
            home_starts = False
        else:
            # Unknown: randomly assign based on 50/50
            home_starts = self._rng.random() < 0.5

        # Vectorized Monte Carlo simulation
        n_sims = _N_SIMULATIONS
        outcomes = np.array([0, 1, 2, 3], dtype=np.int32)

        # Pre-generate all random possession outcomes
        # Shape: (n_sims, remaining_poss)
        home_scores = self._rng.choice(
            outcomes, size=(n_sims, remaining_poss), p=home_dist
        )
        away_scores = self._rng.choice(
            outcomes, size=(n_sims, remaining_poss), p=away_dist
        )

        # Build possession mask: True = home possessing on this possession
        # Alternating possessions starting from whoever has the ball
        poss_mask = np.zeros(remaining_poss, dtype=bool)
        for i in range(remaining_poss):
            if home_starts:
                poss_mask[i] = (i % 2 == 0)
            else:
                poss_mask[i] = (i % 2 == 1)

        # Calculate final margins for each simulation
        # Home points from home possessions + Away points from away possessions
        home_pts = np.sum(home_scores[:, poss_mask], axis=1) if poss_mask.any() else np.zeros(n_sims)
        away_pts = np.sum(away_scores[:, ~poss_mask], axis=1) if (~poss_mask).any() else np.zeros(n_sims)

        final_margins = state.score_diff + home_pts.astype(np.int32) - away_pts.astype(np.int32)

        # Count outcomes
        home_wins = np.sum(final_margins > 0)
        ties = np.sum(final_margins == 0)
        # Ties broken by pre-game prior (OT probability approximation)
        tie_home_wins = ties * self.params.pre_game_prob

        win_prob = (home_wins + tie_home_wins) / n_sims
        return float(win_prob)

    @staticmethod
    def from_pre_game(
        pre_game_prob: float,
        home_pace: float = _LEAGUE_AVG_PACE,
        away_pace: float = _LEAGUE_AVG_PACE,
        home_off_rating: float = 110.0,
        away_off_rating: float = 110.0,
    ) -> 'PossessionWinModel':
        """
        Factory: construct model from pre-game contextual data.

        Derives PPP from offensive rating: ppp = off_rating / 100.0
        Derives pace: average of both teams' pace ratings.

        Args:
            pre_game_prob: Devigged P(home wins) from sharp books.
            home_pace: Home team's pace rating (possessions per 48 min).
            away_pace: Away team's pace rating.
            home_off_rating: Home offensive rating (points per 100 possessions).
            away_off_rating: Away offensive rating.

        Returns:
            Configured PossessionWinModel instance.
        """
        params = PossessionModelParams(
            ppp_home=home_off_rating / 100.0,
            ppp_away=away_off_rating / 100.0,
            pace=(home_pace + away_pace) / 2.0,
            pre_game_prob=max(0.02, min(0.98, pre_game_prob)),
        )
        return PossessionWinModel(params)


# ── Backward-Compatible Wrapper ──────────────────────────────────────────────

# Module-level cache of models per game
_model_cache: dict = {}


def compute_live_win_prob(
    score_diff: int,
    minutes_elapsed: float,
    pre_game_prob: float,
    possession_adj: float = 0.0,
    *,
    # New enriched parameters from micro_state
    possession: str = "unknown",
    home_in_bonus: bool = False,
    away_in_bonus: bool = False,
    home_in_double_bonus: bool = False,
    away_in_double_bonus: bool = False,
    ppp_home: float = _LEAGUE_AVG_PPP,
    ppp_away: float = _LEAGUE_AVG_PPP,
    pace: float = _LEAGUE_AVG_PACE,
    game_id: str = "",
) -> float:
    """
    Drop-in replacement for the original compute_live_win_prob.

    When enriched micro-state parameters are provided (possession != "unknown"
    or bonus state is set), delegates to the full PossessionWinModel.

    When only basic parameters are available, falls back to the classic
    Normal approximation for backward compatibility.

    Args:
        score_diff:     home_score - away_score (positive = home leading).
        minutes_elapsed: Total game minutes elapsed (0-48+).
        pre_game_prob:  Pre-game devigged P(home wins).
        possession_adj: Legacy possession adjustment (unused when micro_state available).
        possession:     "home", "away", or "unknown" (from micro_state).
        home_in_bonus:  Whether home team is in the bonus.
        away_in_bonus:  Whether away team is in the bonus.
        home_in_double_bonus: Whether home team is in double bonus.
        away_in_double_bonus: Whether away team is in double bonus.
        ppp_home:       Home team points per possession.
        ppp_away:       Away team points per possession.
        pace:           Game pace (possessions per 48 min).
        game_id:        Optional game ID for model caching.

    Returns:
        Float in (0.02, 0.98) — probability the home team wins.
    """
    minutes_remaining = max(0.1, 48.0 - minutes_elapsed)
    has_enriched_state = (
        possession != "unknown"
        or home_in_bonus
        or away_in_bonus
        or home_in_double_bonus
        or away_in_double_bonus
    )

    if has_enriched_state:
        # Use full possession model
        cache_key = game_id or f"{ppp_home:.2f}_{ppp_away:.2f}_{pace:.0f}"
        if cache_key not in _model_cache:
            params = PossessionModelParams(
                ppp_home=ppp_home,
                ppp_away=ppp_away,
                pace=pace,
                pre_game_prob=pre_game_prob,
            )
            _model_cache[cache_key] = PossessionWinModel(params)

        model = _model_cache[cache_key]
        state = GameMicroState(
            score_diff=score_diff,
            minutes_remaining=minutes_remaining,
            possession=possession,
            home_in_bonus=home_in_bonus,
            away_in_bonus=away_in_bonus,
            home_in_double_bonus=home_in_double_bonus,
            away_in_double_bonus=away_in_double_bonus,
        )
        return model.compute_win_prob(state)

    # ── Fallback: Classic Normal Approximation ────────────────────────
    sigma = _NBA_SIGMA_FULL * math.sqrt(minutes_remaining / 48.0)
    prior_adj = norm.ppf(max(0.01, min(0.99, pre_game_prob))) * sigma
    win_prob = float(norm.cdf((score_diff + possession_adj + prior_adj) / sigma))
    return max(_PROB_FLOOR, min(_PROB_CEILING, win_prob))
