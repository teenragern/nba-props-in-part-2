from scipy.stats import poisson, norm, nbinom
import numpy as np
import pandas as pd
from typing import Dict, Optional


def get_market_col(market: str) -> str:
    market_map = {
        "player_points": "PTS",
        "player_rebounds": "REB",
        "player_assists": "AST",
        "player_threes": "FG3M",
        "player_points_rebounds_assists": "PRA"
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
    'player_assists': 0.12,
    'player_threes': 0.20,
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
        sim_min    = np.where(is_blowout,
                              np.minimum(sim_min, blowout_minute_cap),
                              sim_min)
    elif blowout_prob > 0.0:
        # Mode B: precomputed scalar probability (legacy / backtest path).
        is_blowout = rng.random(n_sims) < blowout_prob
        sim_min    = np.where(is_blowout,
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
) -> Dict[str, float]:
    """
    Return {prob_over, prob_under} for a player prop.

    Monte Carlo path engages when proj_minutes > 0 AND any of:
      - spread + total provided  (per-sim Q3 game simulation)
      - blowout_prob > 0         (scalar fallback)
      - player_foul_rate > 0     (foul-trouble model)

    Falls back to parametric / bootstrap when none of the above apply.
    """
    if mean <= 0:
        return {"prob_over": 0.0, "prob_under": 1.0}

    # Monte Carlo path — engaged when meaningful contextual signals exist
    _use_mc = proj_minutes > 0.0 and (
        (spread != 0.0 and total > 0.0)
        or blowout_prob > 0.0
        or player_foul_rate > 0.0
    )
    if _use_mc:
        return monte_carlo_over_under(
            mean, proj_minutes, line,
            spread=spread,
            total=total,
            blowout_prob=blowout_prob,
            player_foul_rate=player_foul_rate,
            opp_foul_rate=opp_foul_rate,
            n_sims=1000,
            market=market,
        )

    # ── Parametric fallback ─────────────────────────────────────────────────
    if market in ['player_points', 'player_points_rebounds_assists']:
        if logs is not None and not logs.empty:
            col = get_market_col(market)
            bootstrapped = bootstrap_over_under(logs.head(20), col, line)
            if bootstrapped:
                return bootstrapped
        variance = max(mean * 1.25, 4.0) * variance_scale
        return normal_over_under(mean, variance, line)

    elif market in ['player_rebounds', 'player_assists', 'player_threes']:
        alpha = DISPERSION_ALPHAS.get(market, 0.1)
        variance = (mean + (alpha * (mean ** 2))) * variance_scale
        return negative_binomial_over_under(mean, variance, line)

    else:
        variance = max(mean * 1.25, 4.0) * variance_scale
        return normal_over_under(mean, variance, line)
