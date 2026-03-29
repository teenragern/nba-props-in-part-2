"""
Priority 6: XGBoost ensemble model for per-minute stat rate prediction.

Trains one XGBRegressor per market (points/rebounds/assists/threes) on
historical player game logs. Predictions are blended 50/50 with the
Bayesian model in scan_props.py.

Feature vector per game prediction (32 features):
  Base rates:
    rate_5g               per-minute rate over last 5 games
    rate_10g              per-minute rate over last 10 games
    rate_season           season-long per-minute rate
    ewm_rate_5g           exponentially-weighted rate (alpha=0.3) over last 5 games
  Minutes:
    avg_min_5g            average minutes last 5 games
    std_min_5g            std-dev of minutes last 5 games (consistency)
  Situational:
    home_flag             1 = home, 0 = away
    rest_days             days since last game (capped at 7)
    b2b_flag              1 = back-to-back
    games_played          total season games (proxy for small-sample risk)
    streak_factor         rate last 3 games / rate games 4-10 (hot/cold momentum, clamped 0.5–2.0)
    home_rate_delta       historical home per-min rate minus away per-min rate
  Usage & efficiency (BDL real values at inference; box-score proxy at training):
    real_usage_pct        BDL true usage % (replaces (FGA+0.44*FTA+TOV)/MIN proxy)
    ts_pct_5g             true shooting % = PTS/(2*(FGA+0.44*FTA)); BDL general/advanced at inference
  Travel fatigue:
    travel_miles          straight-line miles traveled since last game
    tz_shift_hours        hours shifted east (+) or west (-) vs. last arena
    altitude_flag         1 if tonight's venue ≥ 4 000 ft (Denver/Utah)
  Matchup context:
    opp_def_rating        opponent defensive strength for this market (1.0 = league avg)
    pace_factor           (team_pace + opp_pace) / (2 * league_avg) — game tempo
    opp_pace              opponent team's raw pace normalized to league avg (1.0 = avg)
    opp_rebound_pct       opponent team's DREB_PCT normalized to league avg
    opp_pts_paint         opponent points allowed in paint per game, normalized
  BDL season profile — tracking + playtype (real at inference; 0 during training):
    avg_touches           per-game ball touches (BDL V2 advanced)
    pnr_bh_freq           PnR ball-handler play frequency (BDL playtype/prballhandler)
    iso_freq              isolation play frequency (BDL playtype/isolation)
    spotup_freq           spot-up play frequency (BDL playtype/spotup)
    transition_freq       transition play frequency (BDL playtype/transition)
  BDL V2 advanced tracking (real at inference; 0 during training):
    avg_speed             average court speed mph (BDL V2 advanced)
    avg_contested_fg_pct  contested FG% — shot-difficulty proxy (BDL V2 advanced)
    avg_deflections       deflections per game — defensive activity (BDL V2 advanced)
    avg_points_paint      points in the paint per game (BDL V2 advanced)
    avg_pct_pts_paint     % of points scored from the paint (BDL V2 advanced)

Target: per-minute rate for the specific stat.
Final projection = predicted_rate * projected_minutes.

NOTE: Adding features invalidates previously trained .pkl models.
      Run `train_ml` pipeline to retrain after any feature change.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

from src.clients.travel_fatigue import travel_features_for_game

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'xgb')
MIN_TRAIN_SAMPLES = 50

_MARKET_COL = {
    'player_points':   'PTS',
    'player_rebounds': 'REB',
    'player_assists':  'AST',
    'player_threes':   'FG3M',
    'player_blocks':   'BLK',
    'player_steals':   'STL',
}

FEATURE_NAMES = [
    # Base rates
    'rate_5g', 'rate_10g', 'rate_season', 'ewm_rate_5g',
    # Minutes
    'avg_min_5g', 'std_min_5g',
    # Situational
    'home_flag', 'rest_days', 'b2b_flag', 'games_played',
    'streak_factor', 'home_rate_delta',
    # Usage & efficiency (BDL real at inference; proxy/computed at training)
    'real_usage_pct', 'ts_pct_5g',
    # Travel fatigue
    'travel_miles', 'tz_shift_hours', 'altitude_flag',
    # Matchup context
    'opp_def_rating', 'pace_factor',
    'opp_pace', 'opp_rebound_pct', 'opp_pts_paint',
    # BDL season profile: tracking + playtype (real at inference, 0.0 at training)
    'avg_touches', 'pnr_bh_freq', 'iso_freq', 'spotup_freq', 'transition_freq',
    # BDL V2 advanced tracking (real at inference, 0.0 at training)
    'avg_speed', 'avg_contested_fg_pct', 'avg_deflections',
    'avg_points_paint', 'avg_pct_pts_paint',
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

    def _ts_pct(self, logs: pd.DataFrame, n: int = 5) -> float:
        """True shooting % = PTS / (2 * (FGA + 0.44 * FTA)) over last n games."""
        recent = logs.head(n)
        if recent.empty:
            return 0.0
        pts = recent.get('PTS', pd.Series([0] * len(recent))).sum()
        fga = recent.get('FGA', pd.Series([0] * len(recent))).sum()
        fta = recent.get('FTA', pd.Series([0] * len(recent))).sum()
        denom = 2.0 * (fga + 0.44 * fta)
        return float(pts / denom) if denom > 0 else 0.0

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
                       opp_pts_paint: float = 1.0,
                       travel_miles: float = 0.0,
                       tz_shift_hours: int = 0,
                       altitude_flag: bool = False,
                       # BDL season profile (real at inference; computed/0 at training)
                       real_usage_pct: float = 0.0,
                       avg_touches: float = 0.0,
                       pnr_bh_freq: float = 0.0,
                       iso_freq: float = 0.0,
                       spotup_freq: float = 0.0,
                       transition_freq: float = 0.0,
                       ts_pct: float = 0.0,
                       # BDL V2 advanced tracking (real at inference; 0 at training)
                       avg_speed: float = 0.0,
                       avg_contested_fg_pct: float = 0.0,
                       avg_deflections: float = 0.0,
                       avg_points_paint: float = 0.0,
                       avg_pct_pts_paint: float = 0.0) -> Optional[Dict]:
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
            # Usage & efficiency: real BDL value at inference; proxy at training
            'real_usage_pct':   real_usage_pct if real_usage_pct > 0 else self._usage_proxy(logs, n=5),
            'ts_pct_5g':        ts_pct if ts_pct > 0 else self._ts_pct(logs, n=5),
            # Travel fatigue
            'travel_miles':     float(travel_miles),
            'tz_shift_hours':   int(tz_shift_hours),
            'altitude_flag':    int(altitude_flag),
            # Matchup context
            'opp_def_rating':   float(opp_def_rating),
            'pace_factor':      float(pace_factor),
            'opp_pace':         float(opp_pace),
            'opp_rebound_pct':  float(opp_rebound_pct),
            'opp_pts_paint':    float(opp_pts_paint),
            # BDL season profile: 0.0 during training; real values at inference
            'avg_touches':           float(avg_touches),
            'pnr_bh_freq':           float(pnr_bh_freq),
            'iso_freq':              float(iso_freq),
            'spotup_freq':           float(spotup_freq),
            'transition_freq':       float(transition_freq),
            # BDL V2 advanced tracking: 0.0 during training; real values at inference
            'avg_speed':             float(avg_speed),
            'avg_contested_fg_pct':  float(avg_contested_fg_pct),
            'avg_deflections':       float(avg_deflections),
            'avg_points_paint':      float(avg_points_paint),
            'avg_pct_pts_paint':     float(avg_pct_pts_paint),
        }

    def build_training_data(self, logs: pd.DataFrame,
                            opp_def_lookup: Optional[Dict[str, float]] = None,
                            pace_lookup: Optional[Dict[str, float]] = None,
                            league_avg_pace: float = 99.0,
                            opp_rebound_pct_lookup: Optional[Dict[str, float]] = None,
                            opp_pts_paint_lookup: Optional[Dict[str, float]] = None,
                            player_name: str = "",
                            clv_weight_lookup: Optional[Dict] = None,
                            return_weights: bool = False,
                            ) -> Tuple:
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
            player_name:            Player's display name — used for CLV lookup.
            clv_weight_lookup:      {(player_norm, market, date_str): weight} from
                                    build_clv_weight_lookup().  When provided, each
                                    training sample is weighted by past CLV outcome.
            return_weights:         When True, returns (X, y, weights); when False
                                    (default) returns (X, y) for backward compatibility.
        """
        if logs.empty or len(logs) < 15:
            return ([], [], []) if return_weights else ([], [])
        if self.col not in logs.columns or 'MIN' not in logs.columns:
            return ([], [], []) if return_weights else ([], [])

        # nba_api returns newest-first → reverse for chronological order
        chron = logs[::-1].reset_index(drop=True)
        player_norm = _norm_player_name(player_name)

        X, y, weights = [], [], []
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

            # Travel features: current game arena vs. previous game arena
            own_abbr = str(current.get('TEAM_ABBREVIATION', '')).upper()
            prev_matchup = str(chron.iloc[i - 1].get('MATCHUP', ''))
            t_miles, t_tz, t_alt = travel_features_for_game(matchup, prev_matchup, own_abbr)

            feats = self.build_features(
                history, is_home, rest, opp_def, pace,
                opp_pace_norm, opp_reb_pct, opp_paint,
                travel_miles=t_miles, tz_shift_hours=t_tz, altitude_flag=t_alt,
            )
            if feats is None:
                continue

            # CLV-derived sample weight: past bets on this player/market/game
            # that beat the closing line are upweighted; those that missed are down.
            sample_weight = 1.0
            if clv_weight_lookup is not None and player_norm:
                try:
                    gdate = str(pd.to_datetime(current['GAME_DATE']).date())
                    sample_weight = clv_weight_lookup.get(
                        (player_norm, self.market, gdate), 1.0
                    )
                except Exception:
                    sample_weight = 1.0

            X.append([feats[k] for k in FEATURE_NAMES])
            y.append(target_rate)
            weights.append(sample_weight)

        if return_weights:
            return X, y, weights
        return X, y

    def train(self, X: List, y: List, sample_weight: Optional[List] = None) -> bool:
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
            sw = np.array(sample_weight) if sample_weight else None
            self.model.fit(np.array(X), np.array(y), sample_weight=sw)
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
                      opp_pts_paint: float = 1.0,
                      travel_miles: float = 0.0,
                      tz_shift_hours: int = 0,
                      altitude_flag: bool = False,
                      # BDL season profile features
                      real_usage_pct: float = 0.0,
                      avg_touches: float = 0.0,
                      pnr_bh_freq: float = 0.0,
                      iso_freq: float = 0.0,
                      spotup_freq: float = 0.0,
                      transition_freq: float = 0.0,
                      ts_pct: float = 0.0,
                      # BDL V2 advanced tracking features
                      avg_speed: float = 0.0,
                      avg_contested_fg_pct: float = 0.0,
                      avg_deflections: float = 0.0,
                      avg_points_paint: float = 0.0,
                      avg_pct_pts_paint: float = 0.0) -> Optional[float]:
    """
    Public API: return ML mean projection for a player/market.
    Returns None if model is untrained or data insufficient.
    Caller blends this 50/50 with the Bayesian projection.

    BDL season profile args (all default 0.0 — computed from logs when absent):
        real_usage_pct:   BDL true usage % (replaces box-score proxy).
        avg_touches:      per-game ball touches from BDL V2 advanced.
        pnr_bh_freq:      PnR ball-handler frequency from BDL playtype.
        iso_freq:         isolation play frequency from BDL playtype.
        spotup_freq:      spot-up play frequency from BDL playtype.
        transition_freq:  transition play frequency from BDL playtype.
        ts_pct:           true shooting % from BDL general/advanced.
    """
    if proj_minutes <= 0:
        return None

    _bdl_kwargs = dict(
        real_usage_pct=real_usage_pct, avg_touches=avg_touches,
        pnr_bh_freq=pnr_bh_freq, iso_freq=iso_freq,
        spotup_freq=spotup_freq, transition_freq=transition_freq,
        ts_pct=ts_pct,
        avg_speed=avg_speed, avg_contested_fg_pct=avg_contested_fg_pct,
        avg_deflections=avg_deflections, avg_points_paint=avg_points_paint,
        avg_pct_pts_paint=avg_pct_pts_paint,
    )
    _travel_kwargs = dict(travel_miles=travel_miles, tz_shift_hours=tz_shift_hours,
                          altitude_flag=altitude_flag)

    if market == 'player_points_rebounds_assists':
        pts = get_ml_projection('player_points',  logs, proj_minutes, home_flag, rest_days,
                                opp_def_rating, pace_factor, opp_pace, opp_rebound_pct, opp_pts_paint,
                                **_travel_kwargs, **_bdl_kwargs)
        reb = get_ml_projection('player_rebounds', logs, proj_minutes, home_flag, rest_days,
                                opp_def_rating, pace_factor, opp_pace, opp_rebound_pct, opp_pts_paint,
                                **_travel_kwargs, **_bdl_kwargs)
        ast = get_ml_projection('player_assists',  logs, proj_minutes, home_flag, rest_days,
                                opp_def_rating, pace_factor, opp_pace, opp_rebound_pct, opp_pts_paint,
                                **_travel_kwargs, **_bdl_kwargs)
        if any(v is None for v in [pts, reb, ast]):
            return None
        return pts + reb + ast

    model = PropMLModel(market)
    feats = model.build_features(logs, home_flag, rest_days, opp_def_rating, pace_factor,
                                 opp_pace, opp_rebound_pct, opp_pts_paint,
                                 **_travel_kwargs, **_bdl_kwargs)
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


# ── CLV Feedback Helpers ──────────────────────────────────────────────────────

def _norm_player_name(name: str) -> str:
    """Lowercase + strip accents + normalise whitespace for CLV key matching."""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', name)
    return ' '.join(nfkd.encode('ascii', 'ignore').decode('ascii').lower().split())


def build_clv_weight_lookup(db) -> Dict:
    """
    Query clv_tracking for all settled records and return a sample-weight dict:
        {(player_norm, market, date_str): weight_multiplier}

    CLV is the probability improvement over the alert price (implied_closing − implied_alert).
    We translate it to XGBoost sample weights that reinforce predictions the sharp market
    later confirmed (positive CLV) and discount those it moved against (negative CLV).

    Weight table:
        CLV >  0.02  → 1.8   (strong beat: closing line agreed with us)
        CLV >  0.00  → 1.2   (mild beat)
        CLV > −0.02  → 0.9   (near-miss: market barely moved against us)
        CLV ≤ −0.02  → 0.6   (market clearly disagreed: discount this prediction)
    """
    try:
        with db.get_conn() as conn:
            rows = conn.execute(
                """
                SELECT ct.player_id                                       AS player,
                       ct.market,
                       ct.clv,
                       COALESCE(a.game_date, DATE(ct.alert_time))         AS game_date
                FROM   clv_tracking ct
                LEFT   JOIN alerts_sent a
                            ON  a.player_name = ct.player_id
                            AND a.market      = ct.market
                            AND date(a.timestamp) = date(ct.alert_time)
                WHERE  ct.clv IS NOT NULL
                """
            ).fetchall()
    except Exception:
        return {}

    lookup: Dict = {}
    for row in rows:
        player_n = _norm_player_name(str(row['player']))
        market   = str(row['market'])
        gdate    = str(row['game_date'] or '')
        clv_val  = float(row['clv'])

        if clv_val > 0.02:
            weight = 1.8
        elif clv_val > 0.0:
            weight = 1.2
        elif clv_val > -0.02:
            weight = 0.9
        else:
            weight = 0.6

        lookup[(player_n, market, gdate)] = weight

    return lookup


def train_models_with_clv_feedback(
    player_named_logs: List[Tuple[str, pd.DataFrame]],
    db,
    opp_stats_df: Optional[pd.DataFrame] = None,
    team_stats_df: Optional[pd.DataFrame] = None,
    def_stats_df: Optional[pd.DataFrame] = None,
    league_avg_pace: float = 99.0,
) -> Dict[str, bool]:
    """
    CLV-weighted variant of train_models_from_logs.

    ``player_named_logs`` must be a list of ``(player_name, logs_df)`` tuples so
    each sample can be matched to its CLV record.

    Uses the same feature set as the standard model; the only difference is that
    XGBoost's ``sample_weight`` is drawn from the CLV history so predictions that
    historically beat the closing line receive higher training influence.

    Falls back to uniform weights when no CLV data is available (first run).
    """
    from src.utils.logging_utils import get_logger
    log = get_logger(__name__)

    clv_lookup = build_clv_weight_lookup(db)
    log.info(f"CLV feedback: {len(clv_lookup)} CLV weight records loaded.")

    # Build context lookups (same as train_models_from_logs)
    pace_lookup: Optional[Dict[str, float]] = None
    if team_stats_df is not None and not team_stats_df.empty and 'PACE' in team_stats_df.columns:
        pace_lookup = _build_abbr_lookup(team_stats_df, 'PACE')
        if pace_lookup:
            league_avg_pace = float(np.mean(list(pace_lookup.values())))

    opp_rebound_pct_lookup: Optional[Dict[str, float]] = None
    if team_stats_df is not None and not team_stats_df.empty and 'DREB_PCT' in team_stats_df.columns:
        raw = _build_abbr_lookup(team_stats_df, 'DREB_PCT')
        lg_avg = float(np.mean(list(raw.values()))) if raw else 0.0
        if lg_avg > 0:
            opp_rebound_pct_lookup = {k: v / lg_avg for k, v in raw.items()}

    opp_pts_paint_lookup: Optional[Dict[str, float]] = None
    if def_stats_df is not None and not def_stats_df.empty and 'OPP_PTS_PAINT' in def_stats_df.columns:
        raw = _build_abbr_lookup(def_stats_df, 'OPP_PTS_PAINT')
        lg_avg = float(np.mean(list(raw.values()))) if raw else 0.0
        if lg_avg > 0:
            opp_pts_paint_lookup = {k: v / lg_avg for k, v in raw.items()}

    results = {}
    for market in _MARKET_COL:
        model = PropMLModel(market)

        opp_def_lookup: Optional[Dict[str, float]] = None
        opp_col = _MARKET_OPP_STAT.get(market)
        if opp_stats_df is not None and not opp_stats_df.empty and opp_col and opp_col in opp_stats_df.columns:
            raw = _build_abbr_lookup(opp_stats_df, opp_col)
            lg_avg = float(np.mean(list(raw.values()))) if raw else 1.0
            opp_def_lookup = {k: v / lg_avg for k, v in raw.items()} if lg_avg > 0 else raw

        all_X, all_y, all_w = [], [], []
        for player_name, logs in player_named_logs:
            X, y, w = model.build_training_data(
                logs, opp_def_lookup, pace_lookup, league_avg_pace,
                opp_rebound_pct_lookup, opp_pts_paint_lookup,
                player_name=player_name,
                clv_weight_lookup=clv_lookup,
                return_weights=True,
            )
            all_X.extend(X)
            all_y.extend(y)
            all_w.extend(w)

        success = model.train(all_X, all_y, sample_weight=all_w if all_w else None)
        results[market] = success
        beats  = sum(1 for w in all_w if w > 1.0)
        misses = sum(1 for w in all_w if w < 1.0)
        status = (
            "trained" if success
            else f"skipped (need {MIN_TRAIN_SAMPLES} samples, got {len(all_X)})"
        )
        log.info(f"CLV model [{market}]: {status} | CLV beats={beats} misses={misses}")

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
