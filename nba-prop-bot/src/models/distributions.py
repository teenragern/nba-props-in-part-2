from scipy.stats import poisson, norm, nbinom
import numpy as np
import pandas as pd
from typing import Dict, Optional


POISSON_MARKETS = {'player_blocks', 'player_steals'}


def get_market_col(market: str) -> str:
    market_map = {
        "player_points": "PTS",
        "player_rebounds": "REB",
        "player_assists": "AST",
        "player_threes": "FG3M",
        "player_points_rebounds_assists": "PRA",
        "player_blocks": "BLK",
        "player_steals": "STL",
    }
    return market_map.get(market, "")


def poisson_over_under(mean: float, line: float) -> Dict[str, float]:
    if mean <= 0: return {"prob_over": 0.0, "prob_under": 1.0}
    prob_under = poisson.cdf(np.floor(line), mu=mean)
    return {"prob_over": 1.0 - prob_under, "prob_under": prob_under}


def negative_binomial_over_under(mean: float, variance: float, line: float) -> Dict[str, float]:
    if mean <= 0: return {"prob_over": 0.0, "prob_under": 1.0}
    if variance <= mean:
        return poisson_over_under(mean, line)

    p = mean / variance
    n = (mean ** 2) / (variance - mean)

    prob_under = nbinom.cdf(np.floor(line), n, p)
    return {"prob_over": 1.0 - prob_under, "prob_under": prob_under}


def estimate_zero_inflate(logs: pd.DataFrame, col: str) -> float:
    """
    Estimate the structural zero-inflation parameter π for a discrete stat.

    π represents the probability of a structural zero beyond what a pure
    Poisson distribution would predict (e.g. player never gets a chance to
    block/steal in some games regardless of ability).

    Method: method-of-moments — match empirical zero fraction to ZIP formula.
      π = (empirical_zeros - e^(-λ)) / (1 - e^(-λ))  where λ = sample mean.
    """
    if logs.empty or col not in logs.columns or len(logs) < 5:
        return 0.0
    data = logs[col].dropna().values
    if len(data) < 5:
        return 0.0
    sample_mean = float(np.mean(data))
    if sample_mean <= 0:
        return 0.0
    empirical_zeros = float(np.mean(data == 0))
    expected_zeros  = float(np.exp(-sample_mean))
    pi = (empirical_zeros - expected_zeros) / max(1.0 - expected_zeros, 1e-6)
    return float(np.clip(pi, 0.0, 0.50))


def zip_over_under(mean: float, line: float, zero_inflate: float = 0.0) -> Dict[str, float]:
    """
    Zero-Inflated Poisson P(X > line).

    Parameterisation:
      π  = structural zero probability (estimated from logs)
      λ  = Poisson rate such that E[X] = (1-π)·λ = mean  →  λ = mean/(1-π)
      P(X ≤ k) = π + (1-π)·Poisson.cdf(k, λ)

    Degenerates to standard Poisson when zero_inflate == 0.
    """
    if mean <= 0:
        return {"prob_over": 0.0, "prob_under": 1.0}
    pi  = float(np.clip(zero_inflate, 0.0, 0.50))
    lam = mean / max(1.0 - pi, 0.50)
    k   = int(np.floor(line))
    prob_under = pi + (1.0 - pi) * float(poisson.cdf(k, mu=lam))
    prob_under = float(np.clip(prob_under, 0.0, 1.0))
    return {"prob_over": 1.0 - prob_under, "prob_under": prob_under}


def normal_over_under(mean: float, variance: float, line: float) -> Dict[str, float]:
    if mean <= 0: return {"prob_over": 0.0, "prob_under": 1.0}
    prob_under = norm.cdf(line, loc=mean, scale=np.sqrt(variance))
    return {"prob_over": 1.0 - prob_under, "prob_under": prob_under}


def bootstrap_over_under(logs: pd.DataFrame, col: str, line: float, num_draws: int = 10000) -> Dict[str, float]:
    if logs.empty or len(logs) < 10:
        return {}

    if col == "PRA":
        if not all(c in logs.columns for c in ['PTS', 'REB', 'AST']): return {}
        data = (logs['PTS'] + logs['REB'] + logs['AST']).values
    else:
        if col not in logs.columns:
            return {}
        data = logs[col].values

    if len(data) == 0:
        return {}

    draws = np.random.choice(data, size=num_draws, replace=True)
    prob_over = np.mean(draws > line)
    return {"prob_over": float(prob_over), "prob_under": float(1.0 - prob_over)}


DISPERSION_ALPHAS = {
    'player_rebounds': 0.15,
    'player_assists':  0.12,
    'player_threes':   0.20,
    'player_blocks':   0.45,   # highly bursty — rim protectors vs. perimeter
    'player_steals':   0.35,   # opportunity-driven, game-script dependent
}


# ---------------------------------------------------------------------------
# Blowout & foul-trouble helpers
# ---------------------------------------------------------------------------

def spread_to_blowout_prob(point_spread: float, margin_sigma: float = 11.0) -> float:
    """
    P(|final_margin| > 15) given the game's point spread.

    Models final margin as Normal(|spread|, margin_sigma).
    Historical NBA margin std dev ≈ 11 points.

    Examples:
      spread=0  (pick-em)   → ~17% blowout probability
      spread=7  (one score) → ~25%
      spread=15 (heavy fav) → ~50%
    """
    abs_spread = abs(point_spread)
    # P(margin > 15) + P(margin < -15) where margin ~ N(abs_spread, sigma)
    p_blowout = (1.0 - norm.cdf(15.0, loc=abs_spread, scale=margin_sigma)
                 + norm.cdf(-15.0, loc=abs_spread, scale=margin_sigma))
    return float(np.clip(p_blowout, 0.0, 1.0))


def compute_player_foul_rate(logs: pd.DataFrame) -> float:
    """
    Personal fouls per minute from last 15 games.
    Falls back to ~NBA average (2 PF / 24 min ≈ 0.083) when data is missing.
    """
    if logs.empty or 'PF' not in logs.columns or 'MIN' not in logs.columns:
        return 0.083
    recent = logs.head(15)
    total_min = float(recent['MIN'].sum())
    if total_min <= 0:
        return 0.083
    return float(max(0.0, recent['PF'].sum() / total_min))


_NBA_AVG_TOTAL    = 220.0
_Q3_TIME_FRACTION = 0.75
_Q3_MARGIN_SIGMA  = 11.0 * np.sqrt(_Q3_TIME_FRACTION)   # ≈ 9.53 pts
_BLOWOUT_MARGIN   = 15.0


def classify_bench_tier(proj_minutes: float) -> int:
    """
    Classify a player into a bench tier based on projected minutes.

    Returns:
        0 – Star         (≥ 32 min)
        1 – Starter      (≥ 24 min)
        2 – Rotation     (≥ 15 min)
        3 – Bench        (<  15 min)  ← garbage-time beneficiary
    """
    if proj_minutes >= 32:
        return 0
    if proj_minutes >= 24:
        return 1
    if proj_minutes >= 15:
        return 2
    return 3


def monte_carlo_over_under(
    mean_proj: float,
    proj_minutes: float,
    line: float,
    spread: float = 0.0,
    total: float = 0.0,
    blowout_prob: float = 0.0,
    blowout_minute_cap: float = 28.0,
    player_foul_rate: float = 0.0,
    opp_foul_rate: float = 1.0,
    n_sims: int = 1000,
    market: str = 'player_points',
    bench_tier: int = 1,
) -> Dict[str, float]:
    """
    Vectorized Monte Carlo simulation over n_sims game scenarios.

    Blowout model — two modes
    ─────────────────────────
    Mode A (preferred): spread + total provided.
      Each sim independently draws a Q3-end margin:
          q3_margin ~ N(|spread| × 0.75,  σ_q3)
      where σ_q3 = 9.53 × √(total / 220) scales with game pace.
      If |q3_margin| > 15, starter sits Q4:
          sim_min = min(proj_minutes, blowout_minute_cap=28)

    Mode B (fallback): blowout_prob scalar provided.
      P(blowout) gate → same minute cap (no random 65-85% factor).

    Foul-trouble model (two-phase Poisson)
    ──────────────────────────────────────
    Phase 1 – first 24 minutes of sim time:
        fouls ~ Poisson(adj_foul_rate × first_half_minutes)
        If ≥ 2 fouls → benched 5–8 minutes (early foul trouble).

    Phase 2 – remaining minutes:
        If total fouls reach 5 → benched 4–7 additional minutes.

    adj_foul_rate = player_foul_rate × max(0.5, opp_foul_rate)

    Stat distribution (Gamma-Poisson = Negative Binomial)
    ──────────────────────────────────────────────────────
    λ  ~ Gamma(n_nb, scale)  where n_nb = mean²/(var-mean), scale = (var-mean)/mean
    stat ~ Poisson(λ)

    For non-overdispersed cases (variance ≤ mean), λ = expected directly.
    """
    if proj_minutes <= 0 or mean_proj <= 0:
        return {"prob_over": 0.0, "prob_under": 1.0}

    rng = np.random.default_rng()
    rate = mean_proj / proj_minutes

    # ── 1. Simulate effective minutes ──────────────────────────────────────
    sim_min = np.full(n_sims, proj_minutes, dtype=np.float64)

    if spread != 0.0 and total > 0.0:
        # Mode A: per-sim Q3 game state derived from spread + total.
        # Higher totals (faster pace) → more quarter-level variance.
        pace_scale = np.sqrt(max(total, 180.0) / _NBA_AVG_TOTAL)
        q3_sigma   = _Q3_MARGIN_SIGMA * pace_scale
        q3_margin  = rng.normal(abs(spread) * _Q3_TIME_FRACTION, q3_sigma, n_sims)
        is_blowout = np.abs(q3_margin) > _BLOWOUT_MARGIN
        if bench_tier == 3:
            # Garbage-time beneficiary: minutes rise to 12–15 in blowouts
            garbage_min = rng.uniform(12.0, 15.0, n_sims)
            sim_min = np.where(is_blowout, garbage_min, sim_min)
        else:
            # Stars / starters / rotation players sit in blowouts
            sim_min = np.where(is_blowout,
                               np.minimum(sim_min, blowout_minute_cap),
                               sim_min)
    elif blowout_prob > 0.0:
        # Mode B: precomputed scalar probability (legacy / backtest path).
        is_blowout = rng.random(n_sims) < blowout_prob
        if bench_tier == 3:
            garbage_min = rng.uniform(12.0, 15.0, n_sims)
            sim_min = np.where(is_blowout, garbage_min, sim_min)
        else:
            sim_min = np.where(is_blowout,
                               np.minimum(sim_min, blowout_minute_cap),
                               sim_min)

    adj_foul_rate = player_foul_rate * max(0.5, float(opp_foul_rate))
    if adj_foul_rate > 0.0:
        # Phase 1: first-half foul accumulation
        first_half = np.minimum(sim_min, 24.0)
        fouls_1h = rng.poisson(np.maximum(adj_foul_rate * first_half, 1e-9))
        bench_1h = rng.uniform(5.0, 8.0, n_sims)
        sim_min = np.where(fouls_1h >= 2, np.maximum(0.0, sim_min - bench_1h), sim_min)

        # Phase 2: toward 5th foul
        second_half = np.maximum(0.0, sim_min - 24.0)
        fouls_2h = rng.poisson(np.maximum(adj_foul_rate * second_half, 1e-9))
        bench_5th = rng.uniform(4.0, 7.0, n_sims)
        sim_min = np.where(fouls_1h + fouls_2h >= 5,
                           np.maximum(0.0, sim_min - bench_5th), sim_min)

    # ── 1b. Re-centre rate so E[sim_min * rate] = mean_proj ─────────────────
    # mean_proj already incorporates average blowout/DNP risk. The MC
    # stochastically removes minutes on top of that, double-counting downside
    # and biasing the distribution toward Under. Rescaling the per-minute rate
    # so that mean(sim_min) × rate == mean_proj preserves the projected mean as
    # the distribution centre while retaining the variance shape from blowout /
    # foul-trouble spread.
    sim_min_mean = float(np.mean(sim_min))
    if sim_min_mean > 1e-6:
        rate = mean_proj / sim_min_mean

    # ── 2. Sample stat outcome (Gamma-Poisson mixture) ─────────────────────
    expected = np.maximum(rate * sim_min, 0.0)
    alpha = DISPERSION_ALPHAS.get(market, 0.10)
    variance = expected + alpha * (expected ** 2)

    overdispersed = (variance > expected * 1.001) & (expected > 0.0)

    # NB shape/scale: n = mean²/(var-mean), scale = (var-mean)/mean
    n_nb = np.where(
        overdispersed,
        np.maximum((expected ** 2) / np.maximum(variance - expected, 1e-9), 0.01),
        1.0,   # placeholder — not used in np.where result
    )
    scale_nb = np.where(
        overdispersed,
        np.maximum((variance - expected) / np.maximum(expected, 1e-9), 1e-9),
        1.0,
    )

    # Draw λ from Gamma (NB path); non-overdispersed uses expected directly
    lam_gamma = rng.gamma(np.clip(n_nb, 1e-3, 1e6), np.clip(scale_nb, 1e-9, 1e6))
    lam = np.where(overdispersed, lam_gamma, expected)

    sampled = np.where(expected > 0.0, rng.poisson(np.maximum(lam, 1e-10)), 0)

    prob_over = float(np.mean(sampled > line))
    return {"prob_over": prob_over, "prob_under": 1.0 - prob_over}


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------

def get_probability_distribution(
    market: str,
    mean: float,
    line: float,
    logs: Optional[pd.DataFrame] = None,
    variance_scale: float = 1.0,
    proj_minutes: float = 0.0,
    spread: float = 0.0,
    total: float = 0.0,
    blowout_prob: float = 0.0,
    player_foul_rate: float = 0.0,
    opp_foul_rate: float = 1.0,
    bench_tier: int = 1,
) -> Dict[str, float]:
    """
    Return {prob_over, prob_under} for a player prop.

    Monte Carlo path engages when proj_minutes > 0 AND any of:
      - spread + total provided  (per-sim Q3 game simulation)
      - blowout_prob > 0         (scalar fallback)
      - player_foul_rate > 0     (foul-trouble model)

    Falls back to parametric / bootstrap when none of the above apply.
    """
    # Hard cap: no single leg can ever exceed 85% to account for unknown unknowns
    # (unexpected hot games, stale lines, last-minute lineup changes, etc.)
    _MAX_SINGLE_PROB = 0.85

    if mean <= 0:
        return {"prob_over": 0.0, "prob_under": min(1.0, _MAX_SINGLE_PROB)}

    def _cap(result: Dict[str, float]) -> Dict[str, float]:
        return {
            "prob_over":  min(result["prob_over"],  _MAX_SINGLE_PROB),
            "prob_under": min(result["prob_under"], _MAX_SINGLE_PROB),
        }

    # Monte Carlo path — engaged when meaningful contextual signals exist
    _use_mc = proj_minutes > 0.0 and (
        (spread != 0.0 and total > 0.0)
        or blowout_prob > 0.0
        or player_foul_rate > 0.0
    )
    if _use_mc:
        return _cap(monte_carlo_over_under(
            mean, proj_minutes, line,
            spread=spread,
            total=total,
            blowout_prob=blowout_prob,
            player_foul_rate=player_foul_rate,
            opp_foul_rate=opp_foul_rate,
            n_sims=1000,
            market=market,
            bench_tier=bench_tier,
        ))

    # ── Parametric fallback ─────────────────────────────────────────────────
    if market in ['player_points', 'player_points_rebounds_assists']:
        if logs is not None and not logs.empty:
            col = get_market_col(market)
            bootstrapped = bootstrap_over_under(logs.head(20), col, line)
            if bootstrapped:
                return _cap(bootstrapped)
        variance = max(mean * 1.25, 4.0) * variance_scale
        return _cap(normal_over_under(mean, variance, line))

    elif market in ['player_rebounds', 'player_assists', 'player_threes']:
        alpha = DISPERSION_ALPHAS.get(market, 0.1)
        variance = (mean + (alpha * (mean ** 2))) * variance_scale
        return _cap(negative_binomial_over_under(mean, variance, line))

    elif market in POISSON_MARKETS:
        col      = get_market_col(market)
        zero_inf = 0.0
        if logs is not None and not logs.empty and col:
            zero_inf = estimate_zero_inflate(logs.head(20), col)
        return _cap(zip_over_under(mean, line, zero_inflate=zero_inf))

    else:
        variance = max(mean * 1.25, 4.0) * variance_scale
        return _cap(normal_over_under(mean, variance, line))
