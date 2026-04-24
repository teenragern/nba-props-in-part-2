import pandas as pd
from typing import Any, Dict, Optional

# Priority 4: Home advantage and rest day factors
_HOME_ADVANTAGE = 1.02   # ~2% uplift when playing at home
_AWAY_PENALTY   = 0.98
_EXTENDED_REST_BOOST = 1.03  # 3+ days rest
_B2B_PENALTY_MULT    = 0.95  # back-to-back (rest_days == 0)

# Rest-day schedule: maps rest_days -> multiplier
_REST_SCHEDULE = {
    0: 0.95,   # B2B
    1: 1.00,   # normal rest
    2: 1.01,   # extra day
    3: 1.03,   # extended rest
    4: 1.025,  # diminishing returns
    5: 1.02,
    6: 1.015,
    7: 1.01,   # rust risk after a week off
}


def get_market_col(market: str) -> str:
    return {
        "player_points":   "PTS",
        "player_rebounds": "REB",
        "player_assists":  "AST",
        "player_threes":   "FG3M",
        "player_blocks":   "BLK",
        "player_steals":   "STL",
    }.get(market, "")


def get_bayesian_rate(sample_rate: float, prior_rate: float,
                      n_games: int, prior_weight: float = 8.0) -> float:
    return ((n_games * sample_rate) + (prior_weight * prior_rate)) / (n_games + prior_weight)


def get_home_away_factor(home_flag: bool) -> float:
    """Priority 4: Home court advantage multiplier."""
    return _HOME_ADVANTAGE if home_flag else _AWAY_PENALTY


def get_rest_days_factor(rest_days: int, b2b_flag: bool) -> float:
    """Priority 4: Granular rest-day schedule instead of binary B2B/extended."""
    if b2b_flag or rest_days == 0:
        return _B2B_PENALTY_MULT
    return _REST_SCHEDULE.get(min(rest_days, 7), 1.0)


# ---------------------------------------------------------------------------
# Sample weighting (playoff upweighting)
# ---------------------------------------------------------------------------

def compute_log_weights(
    logs: pd.DataFrame,
    playoff_mode: bool = False,
    current_opp_abbr: Optional[str] = None,
) -> pd.Series:
    """
    Per-row importance weights for game logs.
      - Regular-season rows: 1.0
      - Playoff rows (SEASON_ID starts with '4'): 1.75 when playoff_mode is True
      - Playoff rows vs current_opp_abbr (same series), once ≥2 such games exist: 2.5

    When playoff_mode is False or the necessary columns are missing, returns
    a Series of 1.0 so unweighted callers are unaffected.
    """
    if logs is None or logs.empty:
        return pd.Series(dtype=float)
    weights = pd.Series(1.0, index=logs.index)
    if not playoff_mode or 'SEASON_ID' not in logs.columns:
        return weights

    is_playoff = logs['SEASON_ID'].astype(str).str.startswith('4')
    weights = weights.where(~is_playoff, 1.75)

    if current_opp_abbr and 'MATCHUP' in logs.columns:
        opp_upper = str(current_opp_abbr).upper()
        same_series = is_playoff & logs['MATCHUP'].astype(str).str.upper().str.contains(
            opp_upper, na=False, regex=False
        )
        if int(same_series.sum()) >= 2:
            weights = weights.where(~same_series, 2.5)

    return weights


def compute_series_context(
    logs: pd.DataFrame,
    opp_abbr: Optional[str],
) -> Dict[str, int]:
    """
    Infer the current playoff-series state from a player's game log by
    counting consecutive playoff games vs opp_abbr.

    Returns a dict with 'games', 'wins', 'losses' reflecting the series
    W-L going INTO tonight's game (so a 3-0 lead means the team can close
    out tonight; a 1-3 hole means it's an elimination game).
    """
    ctx = {'games': 0, 'wins': 0, 'losses': 0}
    if logs is None or logs.empty or not opp_abbr or 'MATCHUP' not in logs.columns:
        return ctx
    opp_upper = str(opp_abbr).upper()
    for _, row in logs.iterrows():
        season_id = str(row.get('SEASON_ID', ''))
        matchup   = str(row.get('MATCHUP', '')).upper()
        if not season_id.startswith('4') or opp_upper not in matchup:
            break
        ctx['games'] += 1
        wl = str(row.get('WL', '')).upper()
        if wl == 'W':
            ctx['wins'] += 1
        elif wl == 'L':
            ctx['losses'] += 1
    return ctx


def classify_series_state(wins: int, losses: int) -> Dict[str, bool]:
    """
    Map a (wins, losses) series score to situational flags.

    must_win: facing elimination — opponent already has 3 wins AND team does not.
    closeout_opportunity: team has 3 wins AND opponent does not — closer game.
    """
    must_win = losses >= 3 and wins < 3
    closeout = wins >= 3 and losses < 3
    return {'must_win': must_win, 'closeout_opportunity': closeout}


# ---------------------------------------------------------------------------
# Rolling average helpers (exponential decay)
# ---------------------------------------------------------------------------

def _ewm_rate(logs: pd.DataFrame, col: str, n: int, alpha: float = 0.3) -> float:
    """Exponentially-weighted per-minute rate over the last n games (newest-first logs)."""
    recent = logs.head(n)
    if recent.empty or col not in recent.columns or 'MIN' not in recent.columns:
        return 0.0
    rates = []
    for _, row in recent.iterrows():
        m = float(row.get('MIN', 0) or 0)
        s = float(row.get(col, 0) or 0)
        if m > 0:
            rates.append(s / m)
    if not rates:
        return 0.0
    return float(pd.Series(rates[::-1]).ewm(alpha=alpha, adjust=False).mean().iloc[-1])


def _rolling_rate(
    logs: pd.DataFrame, col: str, n: int,
    weights: Optional[pd.Series] = None,
) -> float:
    """Weighted per-minute rate over the last n games. When weights is None,
    falls back to a simple mean."""
    subset = logs.head(n)
    if subset.empty or col not in subset.columns or 'MIN' not in subset.columns:
        return 0.0
    if weights is None:
        total_min = subset['MIN'].sum()
        if total_min <= 0:
            return 0.0
        return float(subset[col].sum() / total_min)
    w = weights.reindex(subset.index).fillna(1.0)
    total_min = float((subset['MIN'] * w).sum())
    if total_min <= 0:
        return 0.0
    return float((subset[col] * w).sum() / total_min)


def compute_rolling_rates(
    logs: pd.DataFrame, col: str,
    weights: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Compute rolling 5/10/20-game per-minute rates. The simple rate buckets are
    sample-weighted (playoff upweighting); the EWM buckets remain pure
    time-decayed recency.
    """
    return {
        'rate_5g':  _rolling_rate(logs, col, 5, weights=weights),
        'rate_10g': _rolling_rate(logs, col, 10, weights=weights),
        'rate_20g': _rolling_rate(logs, col, 20, weights=weights),
        'ewm_5g':   _ewm_rate(logs, col, 5, alpha=0.3),
        'ewm_10g':  _ewm_rate(logs, col, 10, alpha=0.25),
        'ewm_20g':  _ewm_rate(logs, col, 20, alpha=0.15),
    }


def blend_rolling_rates(rates: Dict[str, float], n_games: int) -> float:
    """
    Blend rolling rates with adaptive weighting based on sample size.
    Early-season (few games) leans on EWM; mid-season balances all windows.
    """
    if n_games < 8:
        # Small sample: lean on EWM of what we have
        return rates.get('ewm_5g', 0.0)
    if n_games < 15:
        # Growing sample: 60% EWM-5, 25% rate_10, 15% rate_season (not here)
        return 0.60 * rates.get('ewm_5g', 0.0) + 0.40 * rates.get('rate_10g', 0.0)
    # Full sample: blend across windows with recency bias
    return (
        0.35 * rates.get('ewm_5g', 0.0) +
        0.25 * rates.get('ewm_10g', 0.0) +
        0.15 * rates.get('rate_10g', 0.0) +
        0.15 * rates.get('ewm_20g', 0.0) +
        0.10 * rates.get('rate_20g', 0.0)
    )


# ---------------------------------------------------------------------------
# Minutes projection (refined)
# ---------------------------------------------------------------------------

def estimate_projected_minutes(recent_logs: pd.DataFrame, season_logs: pd.DataFrame,
                                injury_status: str, starter_flag: bool = False,
                                b2b_flag: bool = False, spread_magnitude: float = 0.0,
                                out_player_avg_mins: float = 0.0,
                                rest_days: int = 2) -> float:
    if recent_logs.empty and season_logs.empty:
        return 0.0

    recent_5_mins  = recent_logs['MIN'].head(5).mean()  if not recent_logs.empty else 0.0
    recent_10_mins = recent_logs['MIN'].head(10).mean() if not recent_logs.empty else recent_5_mins
    recent_20_mins = recent_logs['MIN'].head(20).mean() if not recent_logs.empty else recent_10_mins
    season_mins    = season_logs['MIN'].mean()           if not season_logs.empty else recent_5_mins

    if pd.isna(recent_5_mins):  recent_5_mins  = season_mins
    if pd.isna(recent_10_mins): recent_10_mins = season_mins
    if pd.isna(recent_20_mins): recent_20_mins = season_mins
    if pd.isna(season_mins):    season_mins    = recent_5_mins

    # Minutes volatility: high std signals unstable rotation
    min_std_5 = recent_logs['MIN'].head(5).std() if not recent_logs.empty and len(recent_logs) > 1 else 0.0
    if pd.isna(min_std_5):
        min_std_5 = 0.0

    # Trend detection: sharp upward trend signals a role change in progress
    mins_trend = recent_5_mins - recent_10_mins
    if mins_trend > 4.0:
        # Weight heavily toward recent games (rotation expanding)
        base_mins = (0.70 * recent_5_mins) + (0.15 * recent_10_mins) + (0.10 * recent_20_mins) + (0.05 * season_mins)
    elif mins_trend < -4.0:
        # Downward trend: role shrinking, still trust recent but hedge more
        base_mins = (0.55 * recent_5_mins) + (0.20 * recent_10_mins) + (0.15 * recent_20_mins) + (0.10 * season_mins)
    elif min_std_5 > 6.0:
        # High volatility: lean on 10/20 game windows for stability
        base_mins = (0.30 * recent_5_mins) + (0.30 * recent_10_mins) + (0.25 * recent_20_mins) + (0.15 * season_mins)
    else:
        base_mins = (0.45 * recent_5_mins) + (0.25 * recent_10_mins) + (0.20 * recent_20_mins) + (0.10 * season_mins)

    # Rotation-aware minutes boost
    if starter_flag:
        if out_player_avg_mins > base_mins + 5.0:
            base_mins = 0.45 * base_mins + 0.55 * out_player_avg_mins
        else:
            base_mins += 3.0
    elif out_player_avg_mins > 0:
        base_mins += out_player_avg_mins * 0.15

    # B2B minute reduction (scaled by rest)
    if b2b_flag:
        base_mins -= 1.5
    elif rest_days >= 4:
        # Extended rest can slightly increase minutes (coach plays starters longer)
        base_mins += 0.5

    if spread_magnitude > 15.0:
        base_mins -= 2.0
    elif spread_magnitude > 10.0:
        base_mins -= 1.0

    mult = 1.0
    status = injury_status.lower() if injury_status else "healthy"
    if "out"        in status: mult = 0.0
    elif "doubtful" in status: mult = 0.35
    elif "questionable" in status or "gtd" in status: mult = 0.75
    elif "probable" in status: mult = 0.95

    return max(0.0, base_mins * mult)


def calculate_rate(
    logs: pd.DataFrame, col: str,
    weights: Optional[pd.Series] = None,
) -> float:
    if logs.empty or col not in logs.columns:
        return 0.0
    if weights is None:
        total_min = logs['MIN'].sum()
        if total_min == 0:
            return 0.0
        return logs[col].sum() / total_min
    w = weights.reindex(logs.index).fillna(1.0)
    total_min = float((logs['MIN'] * w).sum())
    if total_min <= 0:
        return 0.0
    return float((logs[col] * w).sum() / total_min)


def calculate_pra_rate(
    logs: pd.DataFrame,
    weights: Optional[pd.Series] = None,
) -> float:
    if logs.empty:
        return 0.0
    if weights is None:
        total_min = logs['MIN'].sum()
        if total_min == 0:
            return 0.0
        total = logs['PTS'].sum() + logs['REB'].sum() + logs['AST'].sum()
        return total / total_min
    w = weights.reindex(logs.index).fillna(1.0)
    total_min = float((logs['MIN'] * w).sum())
    if total_min <= 0:
        return 0.0
    total = float(((logs['PTS'] + logs['REB'] + logs['AST']) * w).sum())
    return total / total_min


def get_market_variance_calibration(_market: str) -> float:
    # Phase 4: DB lookup for realized vs predicted variance scaling (stub -> 1.0 until data accumulates)
    return 1.0


def build_player_projection(player_id: str, market: str, line: float,
                             recent_logs: pd.DataFrame, season_logs: pd.DataFrame,
                             injury_status: str, team_pace: float, opp_pace: float,
                             opponent_multiplier: float = 1.0,
                             usage_shift: float = 0.0,
                             league_avg_pace: float = 99.0,
                             starter_flag: bool = False,
                             b2b_flag: bool = False,
                             spread_magnitude: float = 0.0,
                             prior_weight: float = 8.0,
                             home_flag: bool = False,
                             rest_days: int = 2,
                             out_player_avg_mins: float = 0.0,
                             projected_minutes_override: float = 0.0,
                             fatigue_multiplier: float = 1.0,
                             role_shift_rate: float = 0.0,
                             initiators_out: int = 0,
                             playoff_mode: bool = False,
                             opp_abbr: Optional[str] = None) -> Dict[str, Any]:
    """
    projected_minutes_override: when > 0, skips estimate_projected_minutes entirely
    and uses this value directly (e.g. from the RotationModel's slot-based projection).
    """
    if projected_minutes_override > 0:
        proj_mins = projected_minutes_override
    else:
        proj_mins = estimate_projected_minutes(
            recent_logs, season_logs, injury_status,
            starter_flag, b2b_flag, spread_magnitude, out_player_avg_mins,
            rest_days=rest_days,
        )

    if proj_mins <= 0:
        return {
            "player_id": player_id, "market": market, "line": line,
            "mean": 0.0, "projected_minutes": 0.0, "injury_status": injury_status,
        }

    n_sample_games = min(len(recent_logs), 30) if not recent_logs.empty else 0

    # Per-row playoff weighting (1.0 for RS; 1.75 for playoffs; 2.5 for same-series
    # once ≥2 head-to-head games exist). No-op when playoff_mode is False.
    recent_weights = compute_log_weights(recent_logs, playoff_mode=playoff_mode,
                                         current_opp_abbr=opp_abbr)
    season_weights = compute_log_weights(season_logs, playoff_mode=playoff_mode,
                                         current_opp_abbr=opp_abbr)

    # ── Rolling averages with exponential decay ────────────────────────────
    if market == "player_points_rebounds_assists":
        # Blend PRA from component rolling rates
        pts_rates = compute_rolling_rates(recent_logs, 'PTS', weights=recent_weights) if not recent_logs.empty else {}
        reb_rates = compute_rolling_rates(recent_logs, 'REB', weights=recent_weights) if not recent_logs.empty else {}
        ast_rates = compute_rolling_rates(recent_logs, 'AST', weights=recent_weights) if not recent_logs.empty else {}
        recent_rate = sum(blend_rolling_rates(r, n_sample_games) for r in [pts_rates, reb_rates, ast_rates])
        season_rate = calculate_pra_rate(season_logs, weights=season_weights)
    else:
        col = get_market_col(market)
        if not col:
            return {}
        rolling = compute_rolling_rates(recent_logs, col, weights=recent_weights) if not recent_logs.empty else {}
        recent_rate = blend_rolling_rates(rolling, n_sample_games) if rolling else 0.0
        season_rate = calculate_rate(season_logs, col, weights=season_weights)

    if season_rate == 0: season_rate = recent_rate
    if recent_rate == 0: recent_rate = season_rate

    # ── Role-shift override ────────────────────────────────────────────
    role_shifted = False
    if role_shift_rate > 0:
        blended_rate = role_shift_rate
        role_shifted = True
        if initiators_out >= 2:
            blended_rate *= 1.50
        adj_rate = blended_rate
    else:
        # Bayesian shrinkage (pulls rolling-blended rate toward season prior)
        blended_rate = get_bayesian_rate(recent_rate, season_rate, n_sample_games, prior_weight=prior_weight)
        adj_rate = blended_rate * (1 + usage_shift)

    # Pace adjustment
    pace_factor = 1.0
    if league_avg_pace > 0 and team_pace > 0 and opp_pace > 0:
        pace_factor = (team_pace + opp_pace) / (2 * league_avg_pace)
    adj_rate *= pace_factor

    # Priority 2: Opponent defensive adjustment
    adj_rate *= opponent_multiplier

    # Priority 4: Home/away factor
    adj_rate *= get_home_away_factor(home_flag)

    # Priority 4: Rest days factor
    rest_factor = get_rest_days_factor(rest_days, b2b_flag)
    adj_rate *= rest_factor

    # Travel fatigue reduces projected minutes
    if fatigue_multiplier < 1.0:
        proj_mins = proj_mins * fatigue_multiplier

    mean_proj = proj_mins * adj_rate
    variance_scale = get_market_variance_calibration(market)

    must_win = False
    closeout_opportunity = False
    series_games = 0
    if playoff_mode and opp_abbr:
        _ctx = compute_series_context(recent_logs, opp_abbr)
        series_games = _ctx['games']
        _state = classify_series_state(_ctx['wins'], _ctx['losses'])
        must_win = _state['must_win']
        closeout_opportunity = _state['closeout_opportunity']

    return {
        "player_id":          player_id,
        "market":             market,
        "line":               line,
        "mean":               mean_proj,
        "projected_minutes":  proj_mins,
        "injury_status":      injury_status,
        "usage_boost":        usage_shift,
        "variance_scale":     variance_scale,
        "home_flag":          home_flag,
        "rest_days":          rest_days,
        "fatigue_multiplier": fatigue_multiplier,
        "role_shifted":       role_shifted,
        "initiators_out":     initiators_out,
        # Factor breakdown: projected_rate = base_rate x pace x opp_def x rest x home
        "base_rate":          blended_rate,
        "pace_factor":        pace_factor,
        "opp_def_factor":     opponent_multiplier,
        "rest_factor":        rest_factor,
        "home_factor":        get_home_away_factor(home_flag),
        "series_games":       series_games,
        "must_win":           must_win,
        "closeout_opportunity": closeout_opportunity,
    }
