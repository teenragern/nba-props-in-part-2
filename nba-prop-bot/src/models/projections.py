import pandas as pd
from typing import Dict, Any

# Priority 4: Home advantage and rest day factors
_HOME_ADVANTAGE = 1.02   # ~2% uplift when playing at home
_AWAY_PENALTY   = 0.98
_EXTENDED_REST_BOOST = 1.03  # 3+ days rest
_B2B_PENALTY_MULT    = 0.95  # back-to-back (rest_days == 0)


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
    """Priority 4: Performance adjustment based on rest between games."""
    if b2b_flag or rest_days == 0:
        return _B2B_PENALTY_MULT
    if rest_days >= 3:
        return _EXTENDED_REST_BOOST
    return 1.0


def estimate_projected_minutes(recent_logs: pd.DataFrame, season_logs: pd.DataFrame,
                                injury_status: str, starter_flag: bool = False,
                                b2b_flag: bool = False, spread_magnitude: float = 0.0,
                                out_player_avg_mins: float = 0.0) -> float:
    if recent_logs.empty and season_logs.empty:
        return 0.0

    recent_5_mins  = recent_logs['MIN'].head(5).mean()  if not recent_logs.empty else 0.0
    recent_10_mins = recent_logs['MIN'].head(10).mean() if not recent_logs.empty else recent_5_mins
    season_mins    = season_logs['MIN'].mean()           if not season_logs.empty else recent_5_mins

    if pd.isna(recent_5_mins):  recent_5_mins  = season_mins
    if pd.isna(recent_10_mins): recent_10_mins = season_mins
    if pd.isna(season_mins):    season_mins    = recent_5_mins

    # Trend detection: sharp upward trend signals a role change in progress
    mins_trend = recent_5_mins - recent_10_mins
    if mins_trend > 4.0:
        # Weight heavily toward recent games (rotation expanding)
        base_mins = (0.70 * recent_5_mins) + (0.20 * recent_10_mins) + (0.10 * season_mins)
    else:
        base_mins = (0.50 * recent_5_mins) + (0.30 * recent_10_mins) + (0.20 * season_mins)

    # Rotation-aware minutes boost
    if starter_flag:
        if out_player_avg_mins > base_mins + 5.0:
            # Player is being thrust into a bigger role than their average suggests.
            # Primary backup absorbs ~55% of the missing starter's opportunity.
            base_mins = 0.45 * base_mins + 0.55 * out_player_avg_mins
        else:
            # Regular starter or minor role change: flat boost
            base_mins += 3.0
    elif out_player_avg_mins > 0:
        # Non-starter picks up overflow minutes
        base_mins += out_player_avg_mins * 0.15

    if b2b_flag:             base_mins -= 1.5
    if spread_magnitude > 15.0: base_mins -= 2.0

    mult = 1.0
    status = injury_status.lower() if injury_status else "healthy"
    if "out"        in status: mult = 0.0
    elif "doubtful" in status: mult = 0.35
    elif "questionable" in status or "gtd" in status: mult = 0.75
    elif "probable" in status: mult = 0.95

    return max(0.0, base_mins * mult)


def calculate_rate(logs: pd.DataFrame, col: str) -> float:
    if logs.empty or col not in logs.columns or logs['MIN'].sum() == 0:
        return 0.0
    return logs[col].sum() / logs['MIN'].sum()


def calculate_pra_rate(logs: pd.DataFrame) -> float:
    if logs.empty or logs['MIN'].sum() == 0:
        return 0.0
    total = logs['PTS'].sum() + logs['REB'].sum() + logs['AST'].sum()
    return total / logs['MIN'].sum()


def get_market_variance_calibration(_market: str) -> float:
    # Phase 4: DB lookup for realized vs predicted variance scaling (stub → 1.0 until data accumulates)
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
                             initiators_out: int = 0) -> Dict[str, Any]:
    """
    projected_minutes_override: when > 0, skips estimate_projected_minutes entirely
    and uses this value directly (e.g. from the RotationModel's slot-based projection).
    """
    if projected_minutes_override > 0:
        proj_mins = projected_minutes_override
    else:
        proj_mins = estimate_projected_minutes(
            recent_logs, season_logs, injury_status,
            starter_flag, b2b_flag, spread_magnitude, out_player_avg_mins
        )

    if proj_mins <= 0:
        return {
            "player_id": player_id, "market": market, "line": line,
            "mean": 0.0, "projected_minutes": 0.0, "injury_status": injury_status,
        }

    n_sample_games = min(5, len(recent_logs)) if not recent_logs.empty else 0

    if market == "player_points_rebounds_assists":
        recent_rate = calculate_pra_rate(recent_logs.head(5))
        season_rate = calculate_pra_rate(season_logs)
    else:
        col = get_market_col(market)
        if not col:
            return {}
        recent_rate = calculate_rate(recent_logs.head(5), col)
        season_rate = calculate_rate(season_logs, col)

    if season_rate == 0: season_rate = recent_rate
    if recent_rate == 0: recent_rate = season_rate

    # ── Role-shift override ────────────────────────────────────────────
    # When a primary initiator is OUT and we have reliable on/off split
    # data (rate_without), hard-override the Bayesian baseline with the
    # isolated rate. This prevents the season-long positional average
    # from anchoring the projection to the wrong role.
    role_shifted = False
    if role_shift_rate > 0:
        # Use the on/off rate_without directly — this IS the player's
        # per-minute production when the initiator is off the floor.
        blended_rate = role_shift_rate
        role_shifted = True
        # If BOTH initiators are out, apply a 1.50x multiplier on top —
        # the remaining guard absorbs even more creation responsibility.
        if initiators_out >= 2:
            blended_rate *= 1.50
        # Skip the generic usage_shift; it's already baked into rate_without
        adj_rate = blended_rate
    else:
        # Bayesian shrinkage (pulls sample toward season prior)
        blended_rate = get_bayesian_rate(recent_rate, season_rate, n_sample_games, prior_weight=prior_weight)
        # Lineup usage shift (key player out → +% to active players)
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

    # Priority 4: Rest days factor (applied to minutes proxy)
    rest_factor = get_rest_days_factor(rest_days, b2b_flag)
    adj_rate *= rest_factor

    # Travel fatigue reduces projected minutes (and thus the mean proportionally)
    if fatigue_multiplier < 1.0:
        proj_mins = proj_mins * fatigue_multiplier

    mean_proj = proj_mins * adj_rate
    variance_scale = get_market_variance_calibration(market)

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
    }
