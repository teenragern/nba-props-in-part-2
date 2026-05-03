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

# Playoff defaults: rotations compress, minutes concentrate on stars, and role
# players' opportunities cluster — so within-player cross-stat correlations
# rise ~20-35% on the load-bearing pairs. Weak RS pairs stay approximately flat.
LEAGUE_AVG_CORRELATIONS_PLAYOFF: Dict[Tuple[str, str], float] = {
    ('player_assists',  'player_points'):   0.32,   # 0.25 → +28%
    ('player_assists',  'player_rebounds'): 0.06,
    ('player_assists',  'player_threes'):   0.14,   # 0.10 → +40%
    ('player_blocks',   'player_points'):  -0.05,
    ('player_blocks',   'player_rebounds'): 0.36,   # 0.30 → +20%
    ('player_blocks',   'player_steals'):   0.12,
    ('player_points',   'player_rebounds'): 0.20,   # 0.15 → +33%
    ('player_points',   'player_steals'):   0.12,
    ('player_points',   'player_threes'):   0.62,   # 0.55 → +13%
    ('player_rebounds', 'player_steals'):  -0.05,
    ('player_rebounds', 'player_threes'):  -0.13,
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
                              player_logs: Optional[pd.DataFrame] = None,
                              playoff_mode: bool = False) -> float:
    """
    Return the correlation between two markets for a player.
    Uses player-specific historical data if available (>= 15 games),
    otherwise falls back to league-wide defaults. When playoff_mode is True
    and no player-specific correlation is available, uses the playoff default
    dict (higher within-player correlations under compressed rotations).
    """
    key = tuple(sorted([market_a, market_b]))
    if key[0] == key[1]:
        return 1.0  # same market

    if player_logs is not None and not player_logs.empty and len(player_logs) >= 15:
        player_corrs = compute_player_correlations(player_logs)
        if key in player_corrs:
            return player_corrs[key]

    defaults = LEAGUE_AVG_CORRELATIONS_PLAYOFF if playoff_mode else LEAGUE_AVG_CORRELATIONS
    return defaults.get(key, 0.0)


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
    # Fréchet bounds: P(A∩B) cannot exceed min(P(A), P(B)) and cannot fall
    # below max(0, P(A)+P(B)-1). The bivariate-normal approximation does
    # not enforce these — without the clamp, a 90%×40% leg pair with high
    # correlation can produce a joint > 40%, which is mathematically illegal.
    lower = max(0.001, prob_a + prob_b - 1.0)
    upper = min(0.999, prob_a, prob_b)
    return float(np.clip(joint, lower, upper))


def get_sgp_edge(legs: List[Dict], player_logs: Optional[pd.DataFrame] = None,
                 playoff_mode: bool = False) -> Dict:
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
        corr = get_pairwise_correlation(legs[0]['market'], legs[1]['market'],
                                        player_logs, playoff_mode=playoff_mode)
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
            corr = get_pairwise_correlation(legs[i - 1]['market'], legs[i]['market'],
                                            player_logs, playoff_mode=playoff_mode)
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


def compute_synthetic_pra_prob(
    logs: pd.DataFrame,
    mean_pts: float,
    mean_reb: float,
    mean_ast: float,
    pra_line: float,
    n_sims: int = 10_000,
    playoff_mode: bool = False,
) -> Dict[str, float]:
    """
    Compute P(PTS + REB + AST > pra_line) via a trivariate Gaussian copula
    with Poisson marginals.

    Why this matters: sportsbooks price PRA using their own derivative model
    that often drifts out of sync with the individual-market prices. This
    function derives PRA probability directly from the component distributions
    and their historical covariance, exposing mis-pricings the book can't see.

    Method:
      1. Build the 3×3 Pearson correlation matrix for (PTS, REB, AST) from
         the player's game logs (falls back to league averages if < 15 games).
      2. Sample z ~ N(0, Σ)  where Σ = correlation matrix.
      3. Transform each z through Φ → uniform, then through Poisson PPF
         to get simulated integer counts correlated as in real life.
      4. Return the fraction of sims where PTS + REB + AST > pra_line.

    Returns:
        {'prob_over': float, 'prob_under': float}
        Empty dict when projected means are missing or matrix is degenerate.
    """
    if mean_pts <= 0 or mean_reb <= 0 or mean_ast <= 0 or pra_line <= 0:
        return {}

    rho_pr = get_pairwise_correlation('player_points',   'player_rebounds', logs, playoff_mode=playoff_mode)
    rho_pa = get_pairwise_correlation('player_points',   'player_assists',  logs, playoff_mode=playoff_mode)
    rho_ra = get_pairwise_correlation('player_rebounds', 'player_assists',  logs, playoff_mode=playoff_mode)

    corr_matrix = np.array([
        [1.0,    rho_pr, rho_pa],
        [rho_pr, 1.0,    rho_ra],
        [rho_pa, rho_ra, 1.0],
    ])

    rng = np.random.default_rng()
    try:
        z_samples = rng.multivariate_normal(
            mean=[0.0, 0.0, 0.0], cov=corr_matrix, size=n_sims
        )
    except (np.linalg.LinAlgError, ValueError):
        # Correlation matrix not positive semi-definite — fall back gracefully
        logger.debug("compute_synthetic_pra_prob: degenerate correlation matrix, skipping")
        return {}

    # Map standard-normal quantiles → uniform → Poisson integer counts
    # poisson.ppf(u, mu) is the smallest k s.t. Poisson.CDF(k, mu) >= u
    u = norm.cdf(z_samples)  # shape (n_sims, 3)
    sim_pts = poisson.ppf(u[:, 0], mu=mean_pts)
    sim_reb = poisson.ppf(u[:, 1], mu=mean_reb)
    sim_ast = poisson.ppf(u[:, 2], mu=mean_ast)

    prob_over = float(np.mean(sim_pts + sim_reb + sim_ast > pra_line))
    return {'prob_over': prob_over, 'prob_under': 1.0 - prob_over}


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


def build_full_team_correlation_matrix(
    team_name: str,
    player_logs_dict: Dict[str, pd.DataFrame],
    db: Any,
) -> None:
    """
    Compute and persist the full N x N cross-player correlation matrix
    for all combinations of active players with game logs on a team.
    """
    players = list(player_logs_dict.keys())
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            p_a, p_b = players[i], players[j]
            logs_a, logs_b = player_logs_dict[p_a], player_logs_dict[p_b]
            
            corr_pairs = compute_cross_player_correlations(logs_a, logs_b)
            for (mkt_a, mkt_b), (corr, n) in corr_pairs.items():
                # Insert forward pair
                db.upsert_cross_player_correlation(
                    team_name, p_a, p_b, mkt_a, mkt_b, corr, n)
                # Insert reverse pair
                db.upsert_cross_player_correlation(
                    team_name, p_b, p_a, mkt_b, mkt_a, corr, n)


# ── Cross-team (opposing-player) correlations ─────────────────────────────────
#
# Players on opposing teams compete for the same finite resources in a game.
# The strongest signal is rebounding: roughly 110 total boards per contest.
# When one big dominates the glass, the opponent's big collects fewer — a
# meaningful negative Pearson correlation.
#
# Sportsbook SGP builders universally assume independence across teams (ρ = 0).
# A negative stat correlation becomes a POSITIVE bet-outcome correlation when
# one leg is OVER and the opposing leg is UNDER, yielding a structural edge
# the book has not priced.  The same finite-resource logic applies to any
# metric with a conserved game total (possession counts, shots, etc.).

# League-wide defaults for opposing-player finite-resource correlations.
# Key = sorted (market_a, market_b) tuple; value = Pearson stat correlation.
CROSS_TEAM_DEFAULTS: Dict[Tuple[str, str], float] = {
    ('player_rebounds', 'player_rebounds'): -0.20,  # ~110 total REB conserved
    ('player_points',   'player_points'):   -0.08,  # game-script / pace symmetry
    ('player_assists',  'player_assists'):  -0.05,  # playmaking flow
    ('player_blocks',   'player_blocks'):   -0.05,  # interior paint dominance
}

# Market pairs stored in the DB for each cross-team matchup.
_CROSS_TEAM_PAIRS: List[Tuple[str, str]] = [
    ('player_rebounds', 'player_rebounds'),
    ('player_points',   'player_points'),
]


def get_tiered_correlation(
    market_a: str,
    market_b: str,
    player_name: Optional[str] = None,
    db=None,
    outcome_min_samples: int = 30,
    gamelog_min_samples: int = 15,
    playoff_mode: bool = False,
) -> Tuple[float, str]:
    """
    3-tier correlation lookup for same-player market pairs, returning
    (correlation, source_label).

    Tier 1 — Empirical outcome phi (≥ outcome_min_samples settled bet pairs):
      Computed from our actual bet history.  The gold standard: the outcomes
      we observe are the exact random variables the copula is modelling.

    Tier 2 — Game-log Pearson from sgp_correlations DB (≥ gamelog_min_samples):
      Pre-computed from the player's historical box scores.  Player-specific
      and more stable than league averages, but measures stat correlation
      rather than outcome correlation directly.

    Tier 3 — League-wide defaults (LEAGUE_AVG_CORRELATIONS / playoff variant):
      Always available.  Used when the player has little history.

    Source labels: 'outcome_phi' | 'gamelog' | 'league_default'
    """
    key = tuple(sorted([market_a, market_b]))
    if key[0] == key[1]:
        return (1.0, 'identity')

    # Tier 1: empirical outcome phi from settled bets
    if player_name and db is not None:
        phi, n = db.get_outcome_correlation(
            player_name, market_a, market_b, min_samples=outcome_min_samples
        )
        if phi is not None:
            logger.debug(
                f"Corr tier-1 ({player_name} {market_a}/{market_b}): "
                f"phi={phi:.3f} n={n}"
            )
            return (float(np.clip(phi, -0.999, 0.999)), 'outcome_phi')

    # Tier 2: game-log Pearson from sgp_correlations table
    if player_name and db is not None:
        corr, n = db.get_player_sgp_correlation(
            player_name, market_a, market_b, min_samples=gamelog_min_samples
        )
        if corr is not None:
            logger.debug(
                f"Corr tier-2 ({player_name} {market_a}/{market_b}): "
                f"Pearson={corr:.3f} n={n}"
            )
            return (float(np.clip(corr, -0.999, 0.999)), 'gamelog')

    # Tier 3: league defaults
    defaults = LEAGUE_AVG_CORRELATIONS_PLAYOFF if playoff_mode else LEAGUE_AVG_CORRELATIONS
    corr = defaults.get(key, 0.0)
    return (corr, 'league_default')


def get_cross_team_default_corr(market_a: str, market_b: str) -> float:
    """
    Return the league-wide default cross-team correlation for an opposing-player
    market pair.  Returns 0.0 for pairs with no known finite-resource signal.
    """
    key = tuple(sorted([market_a, market_b]))
    return CROSS_TEAM_DEFAULTS.get(key, 0.0)


def compute_empirical_series_correlation(
    logs_a: pd.DataFrame,
    logs_b: pd.DataFrame,
    col_a: str,
    col_b: str,
    opp_abbr_for_a: str,
    min_games: int = 3,
) -> Optional[float]:
    """
    Pearson correlation between stat col_a (player A) and col_b (player B) across
    the current playoff series, determined by filtering both logs to playoff
    games (SEASON_ID prefix '4') and aligning rows by GAME_ID.

    Returns None when fewer than min_games aligned series games exist or when
    required columns are missing. The returned correlation is in stat-space, so
    the caller should treat it the same way as league CROSS_TEAM_DEFAULTS values.
    """
    required = {'GAME_ID', 'MATCHUP', 'SEASON_ID'}
    if (logs_a is None or logs_b is None or logs_a.empty or logs_b.empty
            or not required.issubset(logs_a.columns)
            or not required.issubset(logs_b.columns)
            or col_a not in logs_a.columns or col_b not in logs_b.columns):
        return None

    opp_upper = str(opp_abbr_for_a).upper()
    series_a = logs_a[
        logs_a['SEASON_ID'].astype(str).str.startswith('4')
        & logs_a['MATCHUP'].astype(str).str.upper().str.contains(
            opp_upper, na=False, regex=False
        )
    ]
    if len(series_a) < min_games:
        return None

    series_b = logs_b[logs_b['SEASON_ID'].astype(str).str.startswith('4')]

    a_side = series_a[['GAME_ID', col_a]].rename(columns={col_a: '__stat_a'})
    b_side = series_b[['GAME_ID', col_b]].rename(columns={col_b: '__stat_b'})
    merged = a_side.merge(b_side, on='GAME_ID', how='inner')
    if len(merged) < min_games:
        return None

    val = merged['__stat_a'].corr(merged['__stat_b'])
    if pd.isna(val):
        return None
    return float(val)


def build_cross_team_correlation_matrix(
    home_team: str,
    away_team: str,
    stats_client: Any,
    db: Any,
    home_player_logs: Optional[Dict[str, pd.DataFrame]] = None,
    away_player_logs: Optional[Dict[str, pd.DataFrame]] = None,
    away_abbr: str = "",
    playoff_mode: bool = False,
) -> None:
    """
    Identify the opposing bigs for the home/away matchup and persist
    cross-team correlation records for finite-resource market pairs.

    In playoff_mode, when ≥3 series games exist and aligned player logs are
    supplied, compute Pearson correlations empirically from the head-to-head
    games and overwrite the league defaults. Otherwise use the defaults.

    Caching behavior:
      - Regular season: 7-day TTL (matches the cross_player_correlations cache).
      - Playoff mode: the TTL check is bypassed so the record tracks the
        evolving series score each scan.
    """
    home_roster = stats_client.get_team_pg_and_big(home_team)
    away_roster = stats_client.get_team_pg_and_big(away_team)
    home_big = home_roster.get('big')
    away_big = away_roster.get('big')

    if not home_big or not away_big:
        logger.debug(
            f"build_cross_team_correlation_matrix: no big identified for "
            f"'{home_team}' or '{away_team}'"
        )
        return

    matchup = "|".join(sorted([home_team.lower(), away_team.lower()]))

    # Regular-season fast path: skip recomputation if a fresh DB record exists.
    # Playoff scans always recompute — the series state changes game-to-game.
    if not playoff_mode and db.get_cross_team_correlation(
        matchup, home_big['name'], away_big['name'],
        'player_rebounds', 'player_rebounds',
    ) is not None:
        return

    home_logs = home_player_logs.get(home_big['name']) if home_player_logs else None
    away_logs = away_player_logs.get(away_big['name']) if away_player_logs else None

    used_empirical_pairs: List[str] = []
    for mkt_a, mkt_b in _CROSS_TEAM_PAIRS:
        default_corr = CROSS_TEAM_DEFAULTS.get(tuple(sorted([mkt_a, mkt_b])), 0.0)
        corr = default_corr
        n_games = 0

        if playoff_mode and home_logs is not None and away_logs is not None and away_abbr:
            empirical = compute_empirical_series_correlation(
                home_logs, away_logs,
                _MARKET_COL.get(mkt_a, ''), _MARKET_COL.get(mkt_b, ''),
                opp_abbr_for_a=away_abbr,
                min_games=3,
            )
            if empirical is not None:
                corr = empirical
                # Recount aligned series games for DB bookkeeping
                series_a = home_logs[
                    home_logs['SEASON_ID'].astype(str).str.startswith('4')
                    & home_logs['MATCHUP'].astype(str).str.upper().str.contains(
                        away_abbr.upper(), na=False, regex=False
                    )
                ]
                n_games = int(len(series_a))
                used_empirical_pairs.append(f"{mkt_a.split('_')[1]}/{mkt_b.split('_')[1]}")

        # Store both orderings so lookups succeed regardless of leg order
        db.upsert_cross_team_correlation(
            matchup, home_big['name'], away_big['name'], mkt_a, mkt_b, corr, n_games)
        db.upsert_cross_team_correlation(
            matchup, away_big['name'], home_big['name'], mkt_b, mkt_a, corr, n_games)

    tag = "EMPIRICAL" if used_empirical_pairs else "DEFAULT"
    logger.debug(
        f"Cross-team corr [{tag}]: {home_team} {home_big['name']} ↔ "
        f"{away_team} {away_big['name']}"
        + (f" | empirical: {','.join(used_empirical_pairs)}" if used_empirical_pairs else "")
    )
