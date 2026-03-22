"""
Priority 10: Same-Game Parlay (SGP) correlation modeling.

Books price SGPs assuming prop legs are independent — they aren't.
This module computes Pearson correlations between a player's stat lines
from historical game logs and uses them to estimate true joint probability.

Two joint-probability methods are available:

1. Gaussian Copula + Poisson marginals (low-count props, line ≤ 3.5)
   Used for threes, steals, blocks — anything that is bounded at zero,
   heavily right-skewed, and hits single-digit values most of the time.
   The bivariate normal formula wildly overestimates P(both rare events)
   in this regime because it ignores the probability mass at zero.

   Steps:
     u_i = poisson.cdf(floor(line_i), mu=mean_i)   # marginal CDF
     z_i = Φ⁻¹(u_i)                                # standard-normal quantile
     P(A>line, B>line) = 1 − u_a − u_b + Φ₂(z_a, z_b, ρ)

2. Bivariate normal approximation (all other props):
   P(A AND B) ≈ P(A)·P(B) + ρ·σ(A)·σ(B)
   where σ(X) = sqrt(P(X)·(1−P(X)))

Usage:
  from src.models.sgp_correlations import get_sgp_edge
  result = get_sgp_edge(legs, player_logs)
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson, norm, multivariate_normal
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# League-wide default correlations between stat pairs (empirically calibrated)
# Key = tuple of sorted market names; value = Pearson correlation
LEAGUE_AVG_CORRELATIONS: Dict[Tuple[str, str], float] = {
    ('player_assists',  'player_points'):   0.25,   # usage-correlated
    ('player_assists',  'player_rebounds'): 0.05,
    ('player_assists',  'player_threes'):   0.10,
    ('player_blocks',   'player_points'):  -0.05,   # rim protectors rarely score much
    ('player_blocks',   'player_rebounds'): 0.30,   # both tied to interior presence
    ('player_blocks',   'player_steals'):   0.10,   # weak positive — defensive versatility
    ('player_points',   'player_rebounds'): 0.15,
    ('player_points',   'player_steals'):   0.10,   # guards who score also pressure ball
    ('player_points',   'player_threes'):   0.55,   # threes are a subset of points
    ('player_rebounds', 'player_steals'):  -0.05,
    ('player_rebounds', 'player_threes'):  -0.10,
    ('player_steals',   'player_threes'):   0.05,
}

_MARKET_COL = {
    'player_points':   'PTS',
    'player_rebounds': 'REB',
    'player_assists':  'AST',
    'player_threes':   'FG3M',
    'player_blocks':   'BLK',
    'player_steals':   'STL',
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


# Props with lines at or below this threshold use the Poisson–Gaussian copula
# instead of the bivariate normal approximation.
_LOW_COUNT_LINE = 3.5


def _gaussian_copula_joint(
    mean_a: float, mean_b: float,
    line_a: float, line_b: float,
    correlation: float,
    side_a: str = 'OVER',
    side_b: str = 'OVER',
) -> float:
    """
    Joint probability of two bets both hitting via Gaussian copula with Poisson marginals.

    Let u_i = P(X_i ≤ floor(line_i))  [Poisson CDF = "under" probability]
        z_i = Φ⁻¹(u_i)
        Φ₂  = P(X_a ≤ line_a AND X_b ≤ line_b)  [joint under CDF]

    Exact formulas for all four side combinations derived from inclusion-exclusion:
        OVER  / OVER  :  1 − u_a − u_b + Φ₂
        OVER  / UNDER :  u_b − Φ₂
        UNDER / OVER  :  u_a − Φ₂
        UNDER / UNDER :  Φ₂

    These four expressions partition the probability space and sum to 1.
    """
    if mean_a <= 0 or mean_b <= 0:
        return 0.0

    _EPS = 1e-9
    u_a = float(np.clip(poisson.cdf(int(np.floor(line_a)), mu=mean_a), _EPS, 1 - _EPS))
    u_b = float(np.clip(poisson.cdf(int(np.floor(line_b)), mu=mean_b), _EPS, 1 - _EPS))

    z_a = float(norm.ppf(u_a))
    z_b = float(norm.ppf(u_b))

    rho = float(np.clip(correlation, -0.999, 0.999))
    cov = [[1.0, rho], [rho, 1.0]]
    # joint_cdf = P(X_a ≤ line_a AND X_b ≤ line_b)
    joint_cdf = float(multivariate_normal.cdf([z_a, z_b], mean=[0.0, 0.0], cov=cov))

    a_over = side_a.upper() == 'OVER'
    b_over = side_b.upper() == 'OVER'

    if a_over and b_over:
        result = 1.0 - u_a - u_b + joint_cdf
    elif a_over and not b_over:
        result = u_b - joint_cdf
    elif not a_over and b_over:
        result = u_a - joint_cdf
    else:
        result = joint_cdf

    return float(np.clip(result, 0.001, 0.999))


def adjust_joint_probability(
    prob_a: float,
    prob_b: float,
    correlation: float,
    mean_a: Optional[float] = None,
    mean_b: Optional[float] = None,
    line_a: Optional[float] = None,
    line_b: Optional[float] = None,
    side_a: str = 'OVER',
    side_b: str = 'OVER',
) -> float:
    """
    Return P(leg_a hits AND leg_b hits) adjusted for correlation.

    Correlation sign:
      Same side (both OVER or both UNDER): positive stat correlation → bets
        move together → use raw correlation.
      Mixed sides (one OVER, one UNDER): positive stat correlation means if
        stat_a is high, stat_b is also high → OVER_a is more likely AND
        UNDER_b is less likely → outcomes are negatively correlated.
        The bivariate normal path negates the correlation automatically.
        The copula path derives the exact formula from inclusion-exclusion.
    """
    use_copula = (
        mean_a is not None and mean_b is not None
        and line_a is not None and line_b is not None
        and (line_a <= _LOW_COUNT_LINE or line_b <= _LOW_COUNT_LINE)
    )
    if use_copula:
        return _gaussian_copula_joint(
            mean_a, mean_b, line_a, line_b, correlation, side_a, side_b)

    # Bivariate normal approximation.
    # Negate the stat correlation when sides differ: a positive Pearson
    # correlation between statistics becomes a negative correlation between
    # bet outcomes when one is OVER and the other is UNDER.
    effective_corr = (
        correlation if side_a.upper() == side_b.upper() else -correlation
    )

    prob_a = float(np.clip(prob_a, 0.001, 0.999))
    prob_b = float(np.clip(prob_b, 0.001, 0.999))
    sigma_a = np.sqrt(prob_a * (1 - prob_a))
    sigma_b = np.sqrt(prob_b * (1 - prob_b))
    joint = prob_a * prob_b + effective_corr * sigma_a * sigma_b
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
        joint_true = adjust_joint_probability(
            true_probs[0], true_probs[1], corr,
            mean_a=legs[0].get('mean'), mean_b=legs[1].get('mean'),
            line_a=legs[0].get('line'), line_b=legs[1].get('line'),
            side_a=legs[0].get('side', 'OVER'), side_b=legs[1].get('side', 'OVER'),
        )
        corr_applied = corr
    else:
        # Multi-leg: sequentially apply pairwise corrections.
        # Copula is applied for the first pair when means/lines are present;
        # subsequent steps use bivariate normal (joint_true has no Poisson mean).
        joint_true = true_probs[0]
        corr_applied = 0.0
        for i in range(1, len(legs)):
            corr = get_pairwise_correlation(legs[i - 1]['market'], legs[i]['market'], player_logs)
            mean_a = legs[i - 1].get('mean') if i == 1 else None
            line_a = legs[i - 1].get('line') if i == 1 else None
            joint_true = adjust_joint_probability(
                joint_true, true_probs[i], corr,
                mean_a=mean_a, mean_b=legs[i].get('mean'),
                line_a=line_a, line_b=legs[i].get('line'),
                side_a=legs[i - 1].get('side', 'OVER'), side_b=legs[i].get('side', 'OVER'),
            )
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


# ── Cross-player (SGP teammate) correlations ──────────────────────────────────

# Market pairs we compute for the PG→Big relationship.
# Tuple layout: (market_a, market_b, pg_col, big_col)
_CROSS_PAIRS = [
    ('player_assists',  'player_points',   'PG_AST',  'BIG_PTS'),
    ('player_assists',  'player_rebounds', 'PG_AST',  'BIG_REB'),
    ('player_points',   'player_points',   'PG_PTS',  'BIG_PTS'),
]

# League-wide fallback when we don't have enough shared games
CROSS_PLAYER_DEFAULTS: Dict[Tuple[str, str], float] = {
    ('player_assists', 'player_points'):   0.22,   # PnR: PG dime → C bucket
    ('player_assists', 'player_rebounds'): 0.05,
    ('player_points',  'player_points'):   0.10,   # co-movement from pace/game script
}


def compute_cross_player_correlations(
    pg_logs: pd.DataFrame,
    big_logs: pd.DataFrame,
    min_games: int = 20,
) -> Dict[Tuple[str, str], Tuple[float, int]]:
    """
    Align two players' game logs on GAME_ID and compute cross-player Pearson
    correlations.  Primary signal: PG assists ↔ C/PF points (pick-and-roll).

    Returns
    -------
    {(market_a, market_b): (pearson_correlation, n_aligned_games)}
    Empty dict when fewer than `min_games` shared games exist.
    """
    if pg_logs.empty or big_logs.empty:
        return {}
    if 'GAME_ID' not in pg_logs.columns or 'GAME_ID' not in big_logs.columns:
        return {}

    needed_pg  = {'GAME_ID', 'AST', 'PTS', 'REB'}
    needed_big = {'GAME_ID', 'PTS', 'REB', 'AST'}
    if not needed_pg.issubset(pg_logs.columns) or not needed_big.issubset(big_logs.columns):
        return {}

    pg_slim = pg_logs[['GAME_ID', 'AST', 'PTS', 'REB']].rename(
        columns={'AST': 'PG_AST', 'PTS': 'PG_PTS', 'REB': 'PG_REB'})
    big_slim = big_logs[['GAME_ID', 'PTS', 'REB', 'AST']].rename(
        columns={'PTS': 'BIG_PTS', 'REB': 'BIG_REB', 'AST': 'BIG_AST'})

    merged = pd.merge(pg_slim, big_slim, on='GAME_ID')
    if len(merged) < min_games:
        return {}

    result: Dict[Tuple[str, str], Tuple[float, int]] = {}
    for mkt_a, mkt_b, col_a, col_b in _CROSS_PAIRS:
        if col_a not in merged.columns or col_b not in merged.columns:
            continue
        corr = merged[col_a].corr(merged[col_b])
        if pd.notna(corr):
            result[(mkt_a, mkt_b)] = (float(corr), len(merged))

    return result


def build_team_correlation_matrix(
    team_name: str,
    stats_client: Any,
    db: Any,
) -> None:
    """
    Identify the team's starting PG and C/PF, compute cross-player correlations
    from their shared game logs, and persist to DB (7-day TTL).

    Skips silently when:
      • The PG or big cannot be identified (roster data unavailable).
      • A fresh DB record already exists (avoids redundant API calls).
    """
    roster = stats_client.get_team_pg_and_big(team_name)
    pg  = roster.get('pg')
    big = roster.get('big')
    if not pg or not big:
        logger.debug(f"build_team_correlation_matrix: no PG/big found for '{team_name}'")
        return

    # Skip recomputation if DB already has a fresh record for the primary pair
    if db.get_cross_player_correlation(
        team_name, pg['name'], big['name'], 'player_assists', 'player_points'
    ) is not None:
        return

    try:
        pg_logs  = stats_client.get_player_game_logs(pg['id'])
        big_logs = stats_client.get_player_game_logs(big['id'])
    except Exception as e:
        logger.warning(f"build_team_correlation_matrix: log fetch failed for {team_name}: {e}")
        return

    corr_pairs = compute_cross_player_correlations(pg_logs, big_logs)

    if not corr_pairs:
        # Not enough shared games — store league defaults so we don't retry every scan
        for (mkt_a, mkt_b), default in CROSS_PLAYER_DEFAULTS.items():
            db.upsert_cross_player_correlation(
                team_name, pg['name'], big['name'], mkt_a, mkt_b, default, 0)
        logger.debug(f"{team_name}: insufficient shared games; stored league-avg cross-player defaults")
        return

    for (mkt_a, mkt_b), (corr, n) in corr_pairs.items():
        db.upsert_cross_player_correlation(
            team_name, pg['name'], big['name'], mkt_a, mkt_b, corr, n)
    logger.info(
        f"Cross-player corr computed for {team_name}: "
        f"{pg['name']} ↔ {big['name']} — "
        + ", ".join(
            f"{m_a.split('_')[1]}/{m_b.split('_')[1]}={c:.3f}(n={n})"
            for (m_a, m_b), (c, n) in corr_pairs.items()
        )
    )
