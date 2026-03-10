"""
Priority 10: Same-Game Parlay (SGP) correlation modeling.

Books price SGPs assuming prop legs are independent — they aren't.
This module computes Pearson correlations between a player's stat lines
from historical game logs and uses them to estimate true joint probability.

Adjustment formula (bivariate normal approximation):
  P(A AND B) ≈ P(A)*P(B) + corr(A,B) * σ(A) * σ(B)
  where σ(X) = sqrt(P(X) * (1 - P(X)))

Usage:
  from src.models.sgp_correlations import get_sgp_edge
  result = get_sgp_edge(legs, player_logs)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# League-wide default correlations between stat pairs (empirically calibrated)
# Key = tuple of sorted market names; value = Pearson correlation
LEAGUE_AVG_CORRELATIONS: Dict[Tuple[str, str], float] = {
    ('player_assists',  'player_points'):   0.25,   # usage-correlated
    ('player_assists',  'player_rebounds'): 0.05,
    ('player_assists',  'player_threes'):   0.10,
    ('player_points',   'player_rebounds'): 0.15,
    ('player_points',   'player_threes'):   0.55,   # threes are a subset of points
    ('player_rebounds', 'player_threes'):  -0.10,
}

_MARKET_COL = {
    'player_points':   'PTS',
    'player_rebounds': 'REB',
    'player_assists':  'AST',
    'player_threes':   'FG3M',
}


def compute_player_correlations(logs: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise Pearson correlations from a player's game log DataFrame.
    Returns a dict keyed by sorted (market_a, market_b) tuples.
    Requires >= 15 games to compute meaningful correlations.
    """
    if logs.empty or len(logs) < 15:
        return {}

    cols = [c for c in ['PTS', 'REB', 'AST', 'FG3M'] if c in logs.columns]
    if len(cols) < 2:
        return {}

    corr_matrix = logs[cols].corr()
    col_to_market = {v: k for k, v in _MARKET_COL.items()}

    result = {}
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1:]:
            mkt_a = col_to_market.get(col_a)
            mkt_b = col_to_market.get(col_b)
            if not mkt_a or not mkt_b:
                continue
            val = corr_matrix.loc[col_a, col_b]
            if pd.isna(val):
                continue
            key = tuple(sorted([mkt_a, mkt_b]))
            result[key] = float(val)

    return result


def get_pairwise_correlation(market_a: str, market_b: str,
                              player_logs: Optional[pd.DataFrame] = None) -> float:
    """
    Return the correlation between two markets for a player.
    Uses player-specific historical data if available (>= 15 games),
    otherwise falls back to league-wide defaults.
    """
    key = tuple(sorted([market_a, market_b]))
    if key[0] == key[1]:
        return 1.0  # same market

    if player_logs is not None and not player_logs.empty and len(player_logs) >= 15:
        player_corrs = compute_player_correlations(player_logs)
        if key in player_corrs:
            return player_corrs[key]

    return LEAGUE_AVG_CORRELATIONS.get(key, 0.0)


def adjust_joint_probability(prob_a: float, prob_b: float, correlation: float) -> float:
    """
    Bivariate normal adjustment for correlated binary outcomes.
    P(A AND B) ≈ P(A)*P(B) + corr * σ(A) * σ(B)
    """
    prob_a = float(np.clip(prob_a, 0.001, 0.999))
    prob_b = float(np.clip(prob_b, 0.001, 0.999))
    sigma_a = np.sqrt(prob_a * (1 - prob_a))
    sigma_b = np.sqrt(prob_b * (1 - prob_b))
    joint = prob_a * prob_b + correlation * sigma_a * sigma_b
    return float(np.clip(joint, 0.001, 0.999))


def get_sgp_edge(legs: List[Dict], player_logs: Optional[pd.DataFrame] = None) -> Dict:
    """
    Evaluate a same-game parlay's true edge accounting for inter-stat correlations.

    Args:
        legs: list of dicts, each with keys:
              {market, side, prob (model prob), implied_prob}
        player_logs: player game log DataFrame for player-specific correlations

    Returns:
        {joint_true_prob, joint_book_prob, sgp_edge, correlation_applied}

    The book's SGP implied probability = product of individual implied probs
    (books assume independence, which is wrong when correlations exist).
    """
    if len(legs) < 2:
        return {}

    true_probs    = [leg['prob']         for leg in legs]
    implied_probs = [leg['implied_prob'] for leg in legs]

    if len(legs) == 2:
        corr = get_pairwise_correlation(legs[0]['market'], legs[1]['market'], player_logs)
        joint_true = adjust_joint_probability(true_probs[0], true_probs[1], corr)
        corr_applied = corr
    else:
        # Multi-leg: sequentially apply pairwise corrections
        joint_true = true_probs[0]
        corr_applied = 0.0
        for i in range(1, len(legs)):
            corr = get_pairwise_correlation(legs[i - 1]['market'], legs[i]['market'], player_logs)
            joint_true = adjust_joint_probability(joint_true, true_probs[i], corr)
            corr_applied += corr
        corr_applied /= len(legs) - 1

    # Books assume independence → joint implied = product of individual probs
    joint_book_prob = float(np.prod(implied_probs))
    sgp_edge = joint_true - joint_book_prob

    return {
        'joint_true_prob':  joint_true,
        'joint_book_prob':  joint_book_prob,
        'sgp_edge':         float(sgp_edge),
        'correlation_applied': float(corr_applied),
    }
