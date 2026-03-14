"""
Priority 6: XGBoost ensemble model for per-minute stat rate prediction.

Trains one XGBRegressor per market (points/rebounds/assists/threes) on
historical player game logs. Predictions are blended 50/50 with the
Bayesian model in scan_props.py.

Feature vector per game prediction (18 features):
  Base rates:
    rate_5g          per-minute rate over last 5 games
    rate_10g         per-minute rate over last 10 games
    rate_season      season-long per-minute rate
    ewm_rate_5g      exponentially-weighted rate (alpha=0.3) over last 5 games
  Minutes:
    avg_min_5g       average minutes last 5 games
    std_min_5g       std-dev of minutes last 5 games (consistency)
  Situational:
    home_flag        1 = home, 0 = away
    rest_days        days since last game (capped at 7)
    b2b_flag         1 = back-to-back
    games_played     total season games (proxy for small-sample risk)
    streak_factor    rate last 3 games / rate games 4-10 (hot/cold momentum, clamped 0.5–2.0)
    home_rate_delta  historical home per-min rate minus away per-min rate
    usage_proxy_5g   (FGA + 0.44*FTA + TOV) / MIN over last 5 games (shot/possession volume)
  Matchup context:
    opp_def_rating   opponent defensive strength for this market (1.0 = league avg)
    pace_factor      (team_pace + opp_pace) / (2 * league_avg) — game tempo
    opp_pace         opponent team's raw pace normalized to league avg (1.0 = avg)
    opp_rebound_pct  opponent team's DREB_PCT normalized to league avg
    opp_pts_paint    opponent points allowed in paint per game, normalized

Target: per-minute rate for the specific stat.
Final projection = predicted_rate * projected_minutes.

NOTE: Adding features invalidates previously trained .pkl models.
      Run `train_ml` pipeline to retrain after any feature change.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'xgb')
MIN_TRAIN_SAMPLES = 50

_MARKET_COL = {
    'player_points':   'PTS',
    'player_rebounds': 'REB',
    'player_assists':  'AST',
    'player_threes':   'FG3M',
}

FEATURE_NAMES = [
    # Base rates
    'rate_5g', 'rate_10g', 'rate_season', 'ewm_rate_5g',
    # Minutes
    'avg_min_5g', 'std_min_5g',
    # Situational
    'home_flag', 'rest_days', 'b2b_flag', 'games_played',
    'streak_factor', 'home_rate_delta', 'usage_proxy_5g',
    # Matchup context
    'opp_def_rating', 'pace_factor',
    'opp_pace', 'opp_rebound_pct', 'opp_pts_paint',
]

# Opponent stat column per market (mirrors nba_stats._MARKET_OPP_COL)
_MARKET_OPP_STAT = {
    'player_points':   'OPP_PTS',
    'player_rebounds': 'OPP_REB',
    'player_assists':  'OPP_AST',
    'player_threes':   'OPP_FG3M',
}


def _extract_opp_abbr(matchup: str) -> str:
    """Extract opponent team abbreviation from a MATCHUP string like 'LAL vs. GSW'."""
    matchup = matchup.strip()
    if 'vs.' in matchup:
        return matchup.split('vs.')[-1].strip().upper()
    if '@' in matchup:
        return matchup.split('@')[-1].strip().upper()
    return ''


def _compute_pace_factor(matchup: str, opp_abbr: str,
                          pace_lookup: Optional[Dict[str, float]],
                          league_avg_pace: float) -> float:
    """Compute (team_pace + opp_pace) / (2 * league_avg). Falls back to 1.0."""
    if not pace_lookup or league_avg_pace <= 0:
        return 1.0
    # Own team abbreviation is the first token of MATCHUP
    own_abbr = matchup.split()[0].upper() if matchup else ''
    own_pace = pace_lookup.get(own_abbr, league_avg_pace)
    opp_pace = pace_lookup.get(opp_abbr, league_avg_pace)
    return (own_pace + opp_pace) / (2.0 * league_avg_pace)


class PropMLModel:
    """XGBoost model for a single prop market."""

    def __init__(self, market: str):
        self.market = market
        self.col = _MARKET_COL.get(market, 'PTS')
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model_path = os.path.join(MODEL_DIR, f'{market}_xgb.pkl')
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                import joblib
                self.model = joblib.load(self.model_path)
            except Exception:
                self.model = None

    def _safe_rate(self, df_slice: pd.DataFrame) -> float:
        if df_slice.empty or 'MIN' not in df_slice.columns:
            return 0.0
        total_min = df_slice['MIN'].sum()
        if total_min <= 0:
            return 0.0
        return float(df_slice[self.col].sum() / total_min) if self.col in df_slice.columns else 0.0

    def _ewm_rate(self, logs: pd.DataFrame, n: int = 5, alpha: float = 0.3) -> float:
        """Exponentially-weighted per-minute rate over the last n games (newest first)."""
        recent = logs.head(n)
        if recent.empty or self.col not in recent.columns or 'MIN' not in recent.columns:
            return 0.0
        rates = []
        for _, row in recent.iterrows():
            m = float(row.get('MIN', 0) or 0)
            s = float(row.get(self.col, 0) or 0)
            if m > 0:
                rates.append(s / m)
        if not rates:
            return 0.0
        # Newest first in logs → reverse to oldest-first for EWM then take last value
        return float(pd.Series(rates[::-1]).ewm(alpha=alpha, adjust=False).mean().iloc[-1])

    def _usage_proxy(self, logs: pd.DataFrame, n: int = 5) -> float:
        """(FGA + 0.44·FTA + TOV) / MIN averaged over last n games — possession volume proxy."""
        recent = logs.head(n)
        if recent.empty or 'MIN' not in recent.columns:
            return 0.0
        total_min = recent['MIN'].sum()
        if total_min <= 0:
            return 0.0
        fga = recent.get('FGA', pd.Series([0] * len(recent))).sum()
        fta = recent.get('FTA', pd.Series([0] * len(recent))).sum()
        tov = recent.get('TOV', pd.Series([0] * len(recent))).sum()
        return float((fga + 0.44 * fta + tov) / total_min)

    def _streak_factor(self, logs: pd.DataFrame) -> float:
        """Rate last 3 games / rate games 4-10. Captures hot/cold streaks. Clamped [0.5, 2.0]."""
        recent_3 = self._safe_rate(logs.head(3))
        prev_7   = self._safe_rate(logs.iloc[3:10])
        if prev_7 <= 0:
            return 1.0
        return float(max(0.5, min(2.0, recent_3 / prev_7)))

    def _home_rate_delta(self, logs: pd.DataFrame) -> float:
        """Historical per-minute rate in home games minus away games (up to last 30 of each)."""
        if 'MATCHUP' not in logs.columns:
            return 0.0
        home_logs = logs[logs['MATCHUP'].str.contains('vs\\.', na=False)].head(30)
        away_logs = logs[logs['MATCHUP'].str.contains('@', na=False)].head(30)
        return self._safe_rate(home_logs) - self._safe_rate(away_logs)

    def build_features(self, logs: pd.DataFrame,
                       home_flag: bool = False,
                       rest_days: int = 2,
                       opp_def_rating: float = 1.0,
                       pace_factor: float = 1.0,
                       opp_pace: float = 1.0,
                       opp_rebound_pct: float = 1.0,
                       opp_pts_paint: float = 1.0) -> Optional[Dict]:
        """Build feature dict from game logs. Returns None if insufficient data."""
        if logs.empty or len(logs) < 3:
            return None
        if self.col not in logs.columns or 'MIN' not in logs.columns:
            return None

        r5  = logs.head(5)
        r10 = logs.head(10)
        avg_min = r5['MIN'].mean()
        std_min = r5['MIN'].std() if len(r5) > 1 else 0.0

        return {
            # Base rates
            'rate_5g':          self._safe_rate(r5),
            'rate_10g':         self._safe_rate(r10),
            'rate_season':      self._safe_rate(logs),
            'ewm_rate_5g':      self._ewm_rate(logs, n=5),
            # Minutes
            'avg_min_5g':       float(avg_min) if not pd.isna(avg_min) else 0.0,
            'std_min_5g':       float(std_min) if not pd.isna(std_min) else 0.0,
            # Situational
            'home_flag':        int(home_flag),
            'rest_days':        min(rest_days, 7),
            'b2b_flag':         int(rest_days == 0),
            'games_played':     len(logs),
            'streak_factor':    self._streak_factor(logs),
            'home_rate_delta':  self._home_rate_delta(logs),
            'usage_proxy_5g':   self._usage_proxy(logs, n=5),
            # Matchup context
            'opp_def_rating':   float(opp_def_rating),
            'pace_factor':      float(pace_factor),
            'opp_pace':         float(opp_pace),
            'opp_rebound_pct':  float(opp_rebound_pct),
            'opp_pts_paint':    float(opp_pts_paint),
        }

    def build_training_data(self, logs: pd.DataFrame,
                            opp_def_lookup: Optional[Dict[str, float]] = None,
                            pace_lookup: Optional[Dict[str, float]] = None,
                            league_avg_pace: float = 99.0,
                            opp_rebound_pct_lookup: Optional[Dict[str, float]] = None,
                            opp_pts_paint_lookup: Optional[Dict[str, float]] = None,
                            ) -> Tuple[List, List]:
        """
        Generate (X, y) training samples from a player's chronological game logs.
        Requires at least 15 games. Only uses past games as features (no lookahead).

        Args:
            opp_def_lookup:         {team_abbr_upper: def_rating} for this market
                                    (normalized, 1.0 = league avg).
            pace_lookup:            {team_abbr_upper: pace_float} (raw values).
            opp_rebound_pct_lookup: {team_abbr_upper: dreb_pct_normalized}.
                                    Build from team_stats_df['DREB_PCT'].
            opp_pts_paint_lookup:   {team_abbr_upper: opp_pts_paint_normalized}.
                                    Build from def_stats_df['OPP_PTS_PAINT'].
        """
        if logs.empty or len(logs) < 15:
            return [], []
        if self.col not in logs.columns or 'MIN' not in logs.columns:
            return [], []

        # nba_api returns newest-first → reverse for chronological order
        chron = logs[::-1].reset_index(drop=True)

        X, y = [], []
        for i in range(10, len(chron)):
            current = chron.iloc[i]
            if float(current.get('MIN', 0) or 0) <= 0:
                continue

            target_rate = float(current[self.col]) / float(current['MIN'])
            history = chron.iloc[:i][::-1]  # newest-first slice of history

            matchup = str(current.get('MATCHUP', ''))
            is_home = 'vs.' in matchup
            opp_abbr = _extract_opp_abbr(matchup)

            try:
                cur_date  = pd.to_datetime(current['GAME_DATE'])
                prev_date = pd.to_datetime(chron.iloc[i - 1]['GAME_DATE'])
                rest = max(0, int((cur_date - prev_date).days) - 1)
            except Exception:
                rest = 2

            opp_def = opp_def_lookup.get(opp_abbr, 1.0) if opp_def_lookup else 1.0
            pace    = _compute_pace_factor(matchup, opp_abbr,
                                           pace_lookup, league_avg_pace)

            # Opponent pace normalized to league average
            if pace_lookup and league_avg_pace > 0:
                raw_opp_pace = pace_lookup.get(opp_abbr, league_avg_pace)
                opp_pace_norm = raw_opp_pace / league_avg_pace
            else:
                opp_pace_norm = 1.0

            opp_reb_pct = (opp_rebound_pct_lookup.get(opp_abbr, 1.0)
                           if opp_rebound_pct_lookup else 1.0)
            opp_paint   = (opp_pts_paint_lookup.get(opp_abbr, 1.0)
                           if opp_pts_paint_lookup else 1.0)

            feats = self.build_features(
                history, is_home, rest, opp_def, pace,
                opp_pace_norm, opp_reb_pct, opp_paint,
            )
            if feats is None:
                continue

            X.append([feats[k] for k in FEATURE_NAMES])
            y.append(target_rate)

        return X, y

    def train(self, X: List, y: List) -> bool:
        """Train XGBoost. Returns True on success."""
        if len(X) < MIN_TRAIN_SAMPLES:
            return False
        try:
            import xgboost as xgb
            import joblib
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )
            self.model.fit(np.array(X), np.array(y))
            joblib.dump(self.model, self.model_path)
            return True
        except Exception:
            self.model = None
            return False

    def predict_rate(self, features: Dict) -> Optional[float]:
        """Predict per-minute stat rate. Returns None if no model trained yet."""
        if self.model is None:
            return None
        try:
            feat_vec = np.array([[features[k] for k in FEATURE_NAMES]])
            return float(self.model.predict(feat_vec)[0])
        except Exception:
            return None


def get_ml_projection(market: str, logs: pd.DataFrame,
                      proj_minutes: float,
                      home_flag: bool = False,
                      rest_days: int = 2,
                      opp_def_rating: float = 1.0,
                      pace_factor: float = 1.0,
                      opp_pace: float = 1.0,
                      opp_rebound_pct: float = 1.0,
                      opp_pts_paint: float = 1.0) -> Optional[float]:
    """
    Public API: return ML mean projection for a player/market.
    Returns None if model is untrained or data insufficient.
    Caller blends this 50/50 with the Bayesian projection.

    Args:
        opp_def_rating:   opponent defensive strength for this market (1.0 = league avg).
        pace_factor:      (team_pace + opp_pace) / (2 * league_avg).
        opp_pace:         opponent team pace normalized to league avg.
        opp_rebound_pct:  opponent DREB_PCT normalized to league avg.
        opp_pts_paint:    opponent points allowed in paint normalized to league avg.
    """
    if proj_minutes <= 0:
        return None

    if market == 'player_points_rebounds_assists':
        pts = get_ml_projection('player_points',  logs, proj_minutes, home_flag, rest_days,
                                opp_def_rating, pace_factor, opp_pace, opp_rebound_pct, opp_pts_paint)
        reb = get_ml_projection('player_rebounds', logs, proj_minutes, home_flag, rest_days,
                                opp_def_rating, pace_factor, opp_pace, opp_rebound_pct, opp_pts_paint)
        ast = get_ml_projection('player_assists',  logs, proj_minutes, home_flag, rest_days,
                                opp_def_rating, pace_factor, opp_pace, opp_rebound_pct, opp_pts_paint)
        if any(v is None for v in [pts, reb, ast]):
            return None
        return pts + reb + ast

    model = PropMLModel(market)
    feats = model.build_features(logs, home_flag, rest_days, opp_def_rating, pace_factor,
                                 opp_pace, opp_rebound_pct, opp_pts_paint)
    if feats is None:
        return None

    rate = model.predict_rate(feats)
    if rate is None:
        return None

    return max(0.0, rate * proj_minutes)


def train_models_from_logs(player_logs_list: List[pd.DataFrame],
                           opp_stats_df: Optional[pd.DataFrame] = None,
                           team_stats_df: Optional[pd.DataFrame] = None,
                           def_stats_df: Optional[pd.DataFrame] = None,
                           league_avg_pace: float = 99.0) -> Dict[str, bool]:
    """
    Train XGBoost models for all markets using a list of player game log DataFrames.
    Call this from a pipeline (e.g., python main.py train_ml).
    Returns {market: success_bool}.

    Args:
        opp_stats_df:  DataFrame from NbaStatsClient.get_opponent_stats().
                       Must contain TEAM_ABBREVIATION + OPP_PTS/REB/AST/FG3M.
        team_stats_df: DataFrame from NbaStatsClient.get_team_stats() [Advanced].
                       Must contain TEAM_ABBREVIATION + PACE + DREB_PCT.
        def_stats_df:  DataFrame from NbaStatsClient.get_team_defense_stats() [Defense].
                       Must contain TEAM_ABBREVIATION + OPP_PTS_PAINT.
        league_avg_pace: League average pace (default 99.0).
    """
    from src.utils.logging_utils import get_logger
    log = get_logger(__name__)

    # Build pace lookup: {abbr: raw_pace}
    pace_lookup: Optional[Dict[str, float]] = None
    if team_stats_df is not None and not team_stats_df.empty and 'PACE' in team_stats_df.columns:
        pace_lookup = _build_abbr_lookup(team_stats_df, 'PACE')
        if pace_lookup:
            league_avg_pace = float(np.mean(list(pace_lookup.values())))

    # Build opp_rebound_pct lookup: {abbr: dreb_pct_normalized}
    opp_rebound_pct_lookup: Optional[Dict[str, float]] = None
    if team_stats_df is not None and not team_stats_df.empty and 'DREB_PCT' in team_stats_df.columns:
        raw = _build_abbr_lookup(team_stats_df, 'DREB_PCT')
        league_avg_dreb = float(np.mean(list(raw.values()))) if raw else 0.0
        if league_avg_dreb > 0:
            opp_rebound_pct_lookup = {k: v / league_avg_dreb for k, v in raw.items()}

    # Build opp_pts_paint lookup: {abbr: opp_pts_paint_normalized}
    opp_pts_paint_lookup: Optional[Dict[str, float]] = None
    if def_stats_df is not None and not def_stats_df.empty and 'OPP_PTS_PAINT' in def_stats_df.columns:
        raw = _build_abbr_lookup(def_stats_df, 'OPP_PTS_PAINT')
        league_avg_paint = float(np.mean(list(raw.values()))) if raw else 0.0
        if league_avg_paint > 0:
            opp_pts_paint_lookup = {k: v / league_avg_paint for k, v in raw.items()}

    results = {}
    for market in _MARKET_COL:
        model = PropMLModel(market)

        # Build opponent defensive lookup for this market: {abbr: def_rating}
        opp_def_lookup: Optional[Dict[str, float]] = None
        opp_col = _MARKET_OPP_STAT.get(market)
        if opp_stats_df is not None and not opp_stats_df.empty and opp_col and opp_col in opp_stats_df.columns:
            raw = _build_abbr_lookup(opp_stats_df, opp_col)
            league_avg_def = float(np.mean(list(raw.values()))) if raw else 1.0
            opp_def_lookup = {k: v / league_avg_def for k, v in raw.items()} if league_avg_def > 0 else raw

        all_X, all_y = [], []
        for logs in player_logs_list:
            X, y = model.build_training_data(
                logs, opp_def_lookup, pace_lookup, league_avg_pace,
                opp_rebound_pct_lookup, opp_pts_paint_lookup,
            )
            all_X.extend(X)
            all_y.extend(y)
        success = model.train(all_X, all_y)
        results[market] = success
        status = "trained" if success else f"skipped (need {MIN_TRAIN_SAMPLES} samples, got {len(all_X)})"
        log.info(f"ML model [{market}]: {status}")
    return results


def _build_abbr_lookup(df: pd.DataFrame, value_col: str) -> Dict[str, float]:
    """Build {team_abbr_upper: value} from a DataFrame that has TEAM_ABBREVIATION or TEAM_NAME."""
    result: Dict[str, float] = {}
    if 'TEAM_ABBREVIATION' in df.columns:
        for _, row in df.iterrows():
            abbr = str(row['TEAM_ABBREVIATION']).upper()
            result[abbr] = float(row[value_col])
    elif 'TEAM_NAME' in df.columns:
        try:
            from nba_api.stats.static import teams as _st
            name_to_abbr = {t['full_name'].lower(): t['abbreviation'].upper() for t in _st.get_teams()}
            for _, row in df.iterrows():
                abbr = name_to_abbr.get(str(row['TEAM_NAME']).lower(), '')
                if abbr:
                    result[abbr] = float(row[value_col])
        except Exception:
            pass
    return result
