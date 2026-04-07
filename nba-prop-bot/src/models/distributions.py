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
    n_sims: int = 5000,
    market: str = 'player_points',
    bench_tier: int = 1,
    next_opp_win_pct: float = 0.0,
    revenge_game: bool = False,
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

    # ── 0. Pre-game blowout cap ───────────────────────────────────────────
    # When the book spread signals a likely blowout (|spread| >= 11),
    # slash projected minutes for starters/stars BEFORE the MC loop.
    # The Q3 simulation (Mode A) already captures in-game blowout
    # variance, but it under-weights guaranteed garbage-time benching
    # because it only activates per-sim when the random Q3 margin
    # exceeds 15. A -14 spread means the market is TELLING us the
    # starters will sit early — force that into the baseline minutes.
    #
    # Scale: -11 → -1.5 min, -12 → -2.0, -13 → -2.5, -14 → -3.0,
    #        -15+ → -3.5 min (capped). Only applies to starters/stars
    #        (bench_tier 0-1). Bench players (tier 3) get a boost.
    _pregame_blowout_cut = 0.0
    abs_spread = abs(spread)
    if abs_spread >= 11.0 and bench_tier <= 1:
        _pregame_blowout_cut = min(1.5 + (abs_spread - 11.0) * 0.5, 3.5)
        proj_minutes = max(proj_minutes - _pregame_blowout_cut, 12.0)
        # Also tighten the blowout cap — in sims where the blowout
        # actually materialises, starters sit even earlier.
        blowout_minute_cap = min(blowout_minute_cap, 26.0)
    elif abs_spread >= 11.0 and bench_tier == 3:
        # Garbage-time beneficiary: projected minutes tick UP slightly
        proj_minutes += min((abs_spread - 11.0) * 0.5, 2.0)

    # Shift mean_proj down proportionally so the re-centering step (1b)
    # doesn't undo the pre-game cut.  The cut IS the edge — the market
    # prices the full-game minutes; we know they'll sit.
    if _pregame_blowout_cut > 0.0:
        mean_proj = mean_proj * (proj_minutes / (proj_minutes + _pregame_blowout_cut))
    rate = mean_proj / proj_minutes

    # Look-ahead spot: player's team is a heavy favourite tonight AND faces a
    # tough opponent next game (win_pct > 0.65).  Coaches rest stars earlier
    # in garbage time to protect them — tighten the blowout minute cap.
    _effective_blowout_cap = blowout_minute_cap
    if next_opp_win_pct > 0.65 and bench_tier <= 1:
        _effective_blowout_cap = min(blowout_minute_cap, 24.0)

    # ── 1. Simulate effective minutes ──────────────────────────────────────
    sim_min = np.full(n_sims, proj_minutes, dtype=np.float64)
    is_blowout = np.zeros(n_sims, dtype=bool)

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
                               np.minimum(sim_min, _effective_blowout_cap),
                               sim_min)
    elif blowout_prob > 0.0:
        # Mode B: precomputed scalar probability (legacy / backtest path).
        is_blowout = rng.random(n_sims) < blowout_prob
        if bench_tier == 3:
            garbage_min = rng.uniform(12.0, 15.0, n_sims)
            sim_min = np.where(is_blowout, garbage_min, sim_min)
        else:
            sim_min = np.where(is_blowout,
                               np.minimum(sim_min, _effective_blowout_cap),
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
    # Revenge game: player faces a team they previously played for.
    # Motivation spikes usage and shot attempts → widen the distribution for
    # scoring/playmaking markets so the model captures tail upside the book misses.
    _REVENGE_MARKETS = {
        'player_points', 'player_assists',
        'player_points_rebounds_assists', 'player_threes',
    }
    if revenge_game and market in _REVENGE_MARKETS:
        alpha *= 1.10
    # Garbage-time variance: bench players in blowouts face other bench
    # players → stat production is noisier. Inflate alpha per-sim where
    # is_blowout triggered for bench-tier players.
    if bench_tier == 3 and np.any(is_blowout):
        alpha_arr = np.full(n_sims, alpha)
        alpha_arr[is_blowout] *= 1.40
        variance = expected + alpha_arr * (expected ** 2)
    else:
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
    next_opp_win_pct: float = 0.0,
    revenge_game: bool = False,
) -> Dict[str, float]:
    """
    Return {prob_over, prob_under} for a player prop.

    Monte Carlo path engages when proj_minutes > 0 AND any of:
      - spread + total provided  (per-sim Q3 game simulation)
      - blowout_prob > 0         (scalar fallback)
      - player_foul_rate > 0     (foul-trouble model)

    Falls back to parametric / bootstrap when none of the above apply.
    """
    # Dynamic cap: scale by how far the line is from the mean.
    # Extreme mismatches (e.g. 28-pt scorer on a 15.5 line) can express
    # higher confidence, but we still cap to guard against unknown unknowns.
    _z = abs(mean - line) / max(mean * 0.3, 1.0) if mean > 0 else 0.0
    _MAX_SINGLE_PROB = min(0.75 + _z * 0.05, 0.93)

    if mean <= 0:
        return {"prob_over": 0.0, "prob_under": min(1.0, 0.75)}

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
            n_sims=5000,
            market=market,
            bench_tier=bench_tier,
            next_opp_win_pct=next_opp_win_pct,
            revenge_game=revenge_game,
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


# ── Game market (team-level) projections ─────────────────────────────────────

_MARGIN_STD = 11.5   # Final-score margin std dev (historical NBA ≈ 11–12 pts)
_TOTAL_STD  = 14.0   # Combined-total std dev     (historical NBA ≈ 13–15 pts)


def project_game_markets(
    home_expected_score: float,
    away_expected_score: float,
    book_spread: float,
    book_total: float,
) -> Dict[str, float]:
    """
    Project true probabilities for Moneyline, Spread, and Total markets
    using a Normal distribution on final-score margin / combined total.

    Parameters
    ----------
    home_expected_score : Pace-and-opponent-adjusted projected score, home team.
    away_expected_score : Same for the away team.
    book_spread         : Home team spread (OddsApiClient convention — negative
                          means home is favored, e.g. -6.5).
    book_total          : Combined over/under total from the book.

    Returns
    -------
    dict with keys: home_win, away_win, home_cover, away_cover, over, under.
    Cover and total probabilities are conditional on no push (push returns stake).

    Convention
    ----------
    expected_margin = away_expected - home_expected.
    Negative  → home projected to win.
    Home covers when (away_score - home_score) < book_spread.
    """
    expected_margin = away_expected_score - home_expected_score
    expected_total  = home_expected_score + away_expected_score

    # ── Moneyline ─────────────────────────────────────────────────────────────
    # P(home wins) = P(away - home < 0) = CDF(0; μ=expected_margin, σ=_MARGIN_STD)
    home_win = float(norm.cdf(0.0, loc=expected_margin, scale=_MARGIN_STD))
    away_win = 1.0 - home_win

    # ── Spread ────────────────────────────────────────────────────────────────
    # Whole-number spreads allow a push; condition on no-push for bet probability.
    is_whole_spread = book_spread != 0.0 and abs(book_spread - round(book_spread)) < 0.01
    if is_whole_spread:
        push_prob = float(
            norm.cdf(book_spread + 0.5, loc=expected_margin, scale=_MARGIN_STD)
            - norm.cdf(book_spread - 0.5, loc=expected_margin, scale=_MARGIN_STD)
        )
        raw_home = float(norm.cdf(book_spread - 0.5, loc=expected_margin, scale=_MARGIN_STD))
        raw_away = 1.0 - float(norm.cdf(book_spread + 0.5, loc=expected_margin, scale=_MARGIN_STD))
        denom    = max(1.0 - push_prob, 1e-6)
        home_cover = raw_home / denom
        away_cover = raw_away / denom
    else:
        # Half-point spread: no push possible
        home_cover = float(norm.cdf(book_spread, loc=expected_margin, scale=_MARGIN_STD))
        away_cover = 1.0 - home_cover

    # ── Total ─────────────────────────────────────────────────────────────────
    is_whole_total = book_total > 0.0 and abs(book_total - round(book_total)) < 0.01
    if is_whole_total:
        push_prob = float(
            norm.cdf(book_total + 0.5, loc=expected_total, scale=_TOTAL_STD)
            - norm.cdf(book_total - 0.5, loc=expected_total, scale=_TOTAL_STD)
        )
        raw_over  = 1.0 - float(norm.cdf(book_total + 0.5, loc=expected_total, scale=_TOTAL_STD))
        raw_under = float(norm.cdf(book_total - 0.5, loc=expected_total, scale=_TOTAL_STD))
        denom     = max(1.0 - push_prob, 1e-6)
        prob_over  = raw_over  / denom
        prob_under = raw_under / denom
    else:
        prob_over  = 1.0 - float(norm.cdf(book_total, loc=expected_total, scale=_TOTAL_STD))
        prob_under = float(norm.cdf(book_total, loc=expected_total, scale=_TOTAL_STD))

    return {
        'home_win':   home_win,
        'away_win':   away_win,
        'home_cover': float(np.clip(home_cover, 0.0, 1.0)),
        'away_cover': float(np.clip(away_cover, 0.0, 1.0)),
        'over':       float(np.clip(prob_over,  0.0, 1.0)),
        'under':      float(np.clip(prob_under, 0.0, 1.0)),
    }


# ── Q1 / 1H game-market projections ──────────────────────────────────────────
# Q1 has no blowout risk and negligible foul-trouble variance, so std devs are
# much tighter than full-game.  Empirical NBA Q1 margins: σ ≈ 5–6 pts.

_Q1_MARGIN_STD = 5.5   # Q1 score-margin std dev
_Q1_TOTAL_STD  = 3.5   # Q1 combined-total std dev


def project_q1_markets(
    home_q1_expected: float,
    away_q1_expected: float,
    book_q1_spread: float,
    book_q1_total: float,
) -> Dict[str, float]:
    """
    Project true probabilities for Q1 Moneyline, Spread, and Total markets.

    Parameters
    ----------
    home_q1_expected : Pace-adjusted Q1 projected score, home team (≈ full_game * 0.25).
    away_q1_expected : Same for away team.
    book_q1_spread   : Q1 home-team spread (negative = home favored).
    book_q1_total    : Q1 combined over/under total.

    Returns the same keys as project_game_markets: home_win, away_win,
    home_cover, away_cover, over, under.
    """
    expected_margin = away_q1_expected - home_q1_expected
    expected_total  = home_q1_expected + away_q1_expected

    # ── Moneyline ─────────────────────────────────────────────────────────────
    home_win = float(norm.cdf(0.0, loc=expected_margin, scale=_Q1_MARGIN_STD))
    away_win = 1.0 - home_win

    # ── Spread ────────────────────────────────────────────────────────────────
    is_whole_spread = book_q1_spread != 0.0 and abs(book_q1_spread - round(book_q1_spread)) < 0.01
    if is_whole_spread:
        push_prob  = float(
            norm.cdf(book_q1_spread + 0.5, loc=expected_margin, scale=_Q1_MARGIN_STD)
            - norm.cdf(book_q1_spread - 0.5, loc=expected_margin, scale=_Q1_MARGIN_STD)
        )
        raw_home   = float(norm.cdf(book_q1_spread - 0.5, loc=expected_margin, scale=_Q1_MARGIN_STD))
        raw_away   = 1.0 - float(norm.cdf(book_q1_spread + 0.5, loc=expected_margin, scale=_Q1_MARGIN_STD))
        denom      = max(1.0 - push_prob, 1e-6)
        home_cover = raw_home / denom
        away_cover = raw_away / denom
    else:
        home_cover = float(norm.cdf(book_q1_spread, loc=expected_margin, scale=_Q1_MARGIN_STD))
        away_cover = 1.0 - home_cover

    # ── Total ─────────────────────────────────────────────────────────────────
    is_whole_total = book_q1_total > 0.0 and abs(book_q1_total - round(book_q1_total)) < 0.01
    if is_whole_total:
        push_prob  = float(
            norm.cdf(book_q1_total + 0.5, loc=expected_total, scale=_Q1_TOTAL_STD)
            - norm.cdf(book_q1_total - 0.5, loc=expected_total, scale=_Q1_TOTAL_STD)
        )
        raw_over   = 1.0 - float(norm.cdf(book_q1_total + 0.5, loc=expected_total, scale=_Q1_TOTAL_STD))
        raw_under  = float(norm.cdf(book_q1_total - 0.5, loc=expected_total, scale=_Q1_TOTAL_STD))
        denom      = max(1.0 - push_prob, 1e-6)
        prob_over  = raw_over  / denom
        prob_under = raw_under / denom
    else:
        prob_over  = 1.0 - float(norm.cdf(book_q1_total, loc=expected_total, scale=_Q1_TOTAL_STD))
        prob_under = float(norm.cdf(book_q1_total, loc=expected_total, scale=_Q1_TOTAL_STD))

    return {
        'home_win':   home_win,
        'away_win':   away_win,
        'home_cover': float(np.clip(home_cover, 0.0, 1.0)),
        'away_cover': float(np.clip(away_cover, 0.0, 1.0)),
        'over':       float(np.clip(prob_over,  0.0, 1.0)),
        'under':      float(np.clip(prob_under, 0.0, 1.0)),
    }


# ── Team totals projections ───────────────────────────────────────────────────
# Betting a single team's score isolates the edge and removes the opponent's
# offensive variance entirely.  Empirical NBA team-score std dev ≈ 8–9 pts.

_TEAM_TOTAL_STD = 8.5


def project_team_totals(
    team_expected_score: float,
    book_team_total: float,
) -> Dict[str, float]:
    """
    Project Over/Under probabilities for a single team's total.

    Parameters
    ----------
    team_expected_score : Pace-and-defense-adjusted projected score for the team.
    book_team_total     : The book's posted team total line.

    Returns {'over': p, 'under': p}.
    """
    is_whole = book_team_total > 0.0 and abs(book_team_total - round(book_team_total)) < 0.01
    if is_whole:
        push_prob  = float(
            norm.cdf(book_team_total + 0.5, loc=team_expected_score, scale=_TEAM_TOTAL_STD)
            - norm.cdf(book_team_total - 0.5, loc=team_expected_score, scale=_TEAM_TOTAL_STD)
        )
        raw_over   = 1.0 - float(norm.cdf(book_team_total + 0.5, loc=team_expected_score, scale=_TEAM_TOTAL_STD))
        raw_under  = float(norm.cdf(book_team_total - 0.5, loc=team_expected_score, scale=_TEAM_TOTAL_STD))
        denom      = max(1.0 - push_prob, 1e-6)
        prob_over  = raw_over  / denom
        prob_under = raw_under / denom
    else:
        prob_over  = 1.0 - float(norm.cdf(book_team_total, loc=team_expected_score, scale=_TEAM_TOTAL_STD))
        prob_under = float(norm.cdf(book_team_total, loc=team_expected_score, scale=_TEAM_TOTAL_STD))

    return {
        'over':  float(np.clip(prob_over,  0.0, 1.0)),
        'under': float(np.clip(prob_under, 0.0, 1.0)),
    }
