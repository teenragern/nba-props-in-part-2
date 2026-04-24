"""
Priority 6: Ensemble model for per-minute stat rate prediction.

Trains a stacking ensemble (XGBoost + RandomForest + Ridge) per market on
historical player game logs. Predictions are blended 50/50 with the
Bayesian Poisson/Normal model in scan_props.py.

Feature vector per game prediction (38 features):
  Base rates:
    rate_5g               per-minute rate over last 5 games
    rate_10g              per-minute rate over last 10 games
    rate_20g              per-minute rate over last 20 games
    rate_season           season-long per-minute rate
    ewm_rate_5g           exponentially-weighted rate (alpha=0.3) over last 5 games
    ewm_rate_10g          exponentially-weighted rate (alpha=0.25) over last 10 games
    ewm_rate_20g          exponentially-weighted rate (alpha=0.15) over last 20 games
  Minutes:
    avg_min_5g            average minutes last 5 games
    std_min_5g            std-dev of minutes last 5 games (consistency)
    max_min_10g           maximum minutes played in last 10 games (ceiling proxy)
  Situational:
    home_flag             1 = home, 0 = away
    rest_days             days since last game (capped at 7)
    b2b_flag              1 = back-to-back
    playoff_flag          1 = playoff game (target game)
    playoff_share_5g      fraction of last 5 games that were playoffs (RS→PO window composition)
    games_played          total season games (proxy for small-sample risk)
    streak_factor         rate last 3 games / rate games 4-10 (hot/cold momentum, clamped 0.5-2.0)
    home_rate_delta       historical home per-min rate minus away per-min rate
    series_game_num       consecutive games vs same opponent (playoff adaptation signal)
  Usage & efficiency (BDL real values at inference; box-score proxy at training):
    real_usage_pct        BDL true usage % (replaces (FGA+0.44*FTA+TOV)/MIN proxy)
    ts_pct_5g             true shooting % = PTS/(2*(FGA+0.44*FTA)); BDL general/advanced at inference
  Travel fatigue:
    travel_miles          straight-line miles traveled since last game
    tz_shift_hours        hours shifted east (+) or west (-) vs. last arena
    altitude_flag         1 if tonight's venue >= 4 000 ft (Denver/Utah)
  Matchup context:
    opp_def_rating        opponent defensive strength for this market (1.0 = league avg)
    pace_factor           (team_pace + opp_pace) / (2 * league_avg) -- game tempo
    opp_pace              opponent team's raw pace normalized to league avg (1.0 = avg)
    opp_rebound_pct       opponent team's DREB_PCT normalized to league avg
    opp_pts_paint         opponent points allowed in paint per game, normalized
  BDL season profile -- tracking + playtype (real at inference; 0 during training):
    avg_touches           per-game ball touches (BDL V2 advanced)
    pnr_bh_freq           PnR ball-handler play frequency (BDL playtype/prballhandler)
    pnr_roll_freq         PnR roll-man play frequency (BDL playtype/prrollman) -- bigs near basket
    iso_freq              isolation play frequency (BDL playtype/isolation)
    spotup_freq           spot-up play frequency (BDL playtype/spotup)
    transition_freq       transition play frequency (BDL playtype/transition)
    postup_freq           post-up play frequency (BDL playtype/postup) -- big men paint scoring
    drives_per_game       drives per game (BDL tracking/drives) -- penetration/FTA/AST proxy
  BDL V2 advanced tracking (real at inference; 0 during training):
    avg_speed             average court speed mph (BDL V2 advanced)
    avg_contested_fg_pct  contested FG% -- shot-difficulty proxy (BDL V2 advanced)
    avg_deflections       deflections per game -- defensive activity (BDL V2 advanced)
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
    # Base rates (expanded with 20-game windows)
    'rate_5g', 'rate_10g', 'rate_20g', 'rate_season',
    'ewm_rate_5g', 'ewm_rate_10g', 'ewm_rate_20g',
    # Minutes
    'avg_min_5g', 'std_min_5g', 'max_min_10g',
    # Situational
    'home_flag', 'rest_days', 'b2b_flag', 'playoff_flag', 'playoff_share_5g',
    'games_played',
    'streak_factor', 'home_rate_delta', 'series_game_num',
    # Usage & efficiency (BDL real at inference; proxy/computed at training)
    'real_usage_pct', 'ts_pct_5g',
    # Travel fatigue
    'travel_miles', 'tz_shift_hours', 'altitude_flag',
    # Matchup context
    'opp_def_rating', 'pace_factor',
    'opp_pace', 'opp_rebound_pct', 'opp_pts_paint',
    # BDL season profile: tracking + playtype (real at inference, 0.0 at training)
    'avg_touches', 'pnr_bh_freq', 'pnr_roll_freq', 'iso_freq', 'spotup_freq',
    'transition_freq', 'postup_freq', 'drives_per_game',
    # BDL V2 advanced tracking (real at inference, 0.0 at training)
    'avg_speed', 'avg_contested_fg_pct', 'avg_deflections',
    'avg_points_paint', 'avg_pct_pts_paint',
    # Foul rate: per-minute foul rate from recent logs (minutes variance signal)
    'player_foul_rate',
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
    own_abbr = matchup.split()[0].upper() if matchup else ''
    own_pace = pace_lookup.get(own_abbr, league_avg_pace)
    opp_pace = pace_lookup.get(opp_abbr, league_avg_pace)
    return (own_pace + opp_pace) / (2.0 * league_avg_pace)


class PropMLModel:
    """XGBoost model for a single prop market (base learner in ensemble)."""

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
        return float(pd.Series(rates[::-1]).ewm(alpha=alpha, adjust=False).mean().iloc[-1])

    def _usage_proxy(self, logs: pd.DataFrame, n: int = 5) -> float:
        """(FGA + 0.44*FTA + TOV) / MIN averaged over last n games."""
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
        """Rate last 3 games / rate games 4-10. Clamped [0.5, 2.0]."""
        recent_3 = self._safe_rate(logs.head(3))
        prev_7   = self._safe_rate(logs.iloc[3:10])
        if prev_7 <= 0:
            return 1.0
        return float(max(0.5, min(2.0, recent_3 / prev_7)))

    def _home_rate_delta(self, logs: pd.DataFrame) -> float:
        """Historical per-minute rate in home games minus away games."""
        if 'MATCHUP' not in logs.columns:
            return 0.0
        home_logs = logs[logs['MATCHUP'].str.contains('vs\\.', na=False)].head(30)
        away_logs = logs[logs['MATCHUP'].str.contains('@', na=False)].head(30)
        return self._safe_rate(home_logs) - self._safe_rate(away_logs)

    def _playoff_share(self, logs: pd.DataFrame, n: int = 5) -> float:
        """
        Fraction of the last n games that were playoff games (SEASON_ID starts '4').
        Signals how much of the rolling-rate window is playoff-composed so the
        model can learn the RS→PO drift implicit in rate_5g / ewm_rate_5g.
        """
        recent = logs.head(n)
        if recent.empty or 'SEASON_ID' not in recent.columns:
            return 0.0
        is_po = recent['SEASON_ID'].astype(str).str.startswith('4')
        return float(is_po.sum()) / float(len(recent))

    def build_features(self, logs: pd.DataFrame,
                       home_flag: bool = False,
                       rest_days: int = 2,
                       playoff_flag: bool = False,
                       opp_abbr: str = "",
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
                       pnr_roll_freq: float = 0.0,
                       iso_freq: float = 0.0,
                       spotup_freq: float = 0.0,
                       transition_freq: float = 0.0,
                       postup_freq: float = 0.0,
                       drives_per_game: float = 0.0,
                       ts_pct: float = 0.0,
                       # BDL V2 advanced tracking (real at inference; 0 at training)
                       avg_speed: float = 0.0,
                       avg_contested_fg_pct: float = 0.0,
                       avg_deflections: float = 0.0,
                       avg_points_paint: float = 0.0,
                       avg_pct_pts_paint: float = 0.0,
                       player_foul_rate: float = 0.0) -> Optional[Dict]:
        """Build feature dict from game logs. Returns None if insufficient data."""
        if logs.empty or len(logs) < 3:
            return None
        if self.col not in logs.columns or 'MIN' not in logs.columns:
            return None

        r5  = logs.head(5)
        r10 = logs.head(10)
        r20 = logs.head(20)
        avg_min = r5['MIN'].mean()
        std_min = r5['MIN'].std() if len(r5) > 1 else 0.0

        # Calculate series game number (consecutive games vs same opponent)
        series_count = 1
        if opp_abbr and 'MATCHUP' in logs.columns:
            for _, row in logs.iterrows():
                hist_opp = _extract_opp_abbr(str(row.get('MATCHUP', '')))
                if hist_opp == opp_abbr:
                    series_count += 1
                else:
                    break

        return {
            # Base rates (expanded)
            'rate_5g':          self._safe_rate(r5),
            'rate_10g':         self._safe_rate(r10),
            'rate_20g':         self._safe_rate(r20),
            'rate_season':      self._safe_rate(logs),
            'ewm_rate_5g':      self._ewm_rate(logs, n=5, alpha=0.3),
            'ewm_rate_10g':     self._ewm_rate(logs, n=10, alpha=0.25),
            'ewm_rate_20g':     self._ewm_rate(logs, n=20, alpha=0.15),
            # Minutes
            'avg_min_5g':       float(avg_min) if not pd.isna(avg_min) else 0.0,
            'std_min_5g':       float(std_min) if not pd.isna(std_min) else 0.0,
            'max_min_10g':      float(r10['MIN'].max()) if not r10.empty else 0.0,
            # Situational
            'home_flag':        int(home_flag),
            'rest_days':        min(rest_days, 7),
            'b2b_flag':         int(rest_days == 0),
            'playoff_flag':     int(playoff_flag),
            'playoff_share_5g': self._playoff_share(logs, n=5),
            'games_played':     len(logs),
            'streak_factor':    self._streak_factor(logs),
            'home_rate_delta':  self._home_rate_delta(logs),
            'series_game_num':  series_count,
            # Usage & efficiency
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
            # BDL season profile
            'avg_touches':           float(avg_touches),
            'pnr_bh_freq':           float(pnr_bh_freq),
            'pnr_roll_freq':         float(pnr_roll_freq),
            'iso_freq':              float(iso_freq),
            'spotup_freq':           float(spotup_freq),
            'transition_freq':       float(transition_freq),
            'postup_freq':           float(postup_freq),
            'drives_per_game':       float(drives_per_game),
            # BDL V2 advanced tracking
            'avg_speed':             float(avg_speed),
            'avg_contested_fg_pct':  float(avg_contested_fg_pct),
            'avg_deflections':       float(avg_deflections),
            'avg_points_paint':      float(avg_points_paint),
            'avg_pct_pts_paint':     float(avg_pct_pts_paint),
            # Foul rate
            'player_foul_rate':      float(player_foul_rate),
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
        """
        if logs.empty or len(logs) < 15:
            return ([], [], []) if return_weights else ([], [])
        if self.col not in logs.columns or 'MIN' not in logs.columns:
            return ([], [], []) if return_weights else ([], [])

        # nba_api returns newest-first -> reverse for chronological order
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

            if pace_lookup and league_avg_pace > 0:
                raw_opp_pace = pace_lookup.get(opp_abbr, league_avg_pace)
                opp_pace_norm = raw_opp_pace / league_avg_pace
            else:
                opp_pace_norm = 1.0

            opp_reb_pct = (opp_rebound_pct_lookup.get(opp_abbr, 1.0)
                           if opp_rebound_pct_lookup else 1.0)
            opp_paint   = (opp_pts_paint_lookup.get(opp_abbr, 1.0)
                           if opp_pts_paint_lookup else 1.0)

            own_abbr = str(current.get('TEAM_ABBREVIATION', '')).upper()
            prev_matchup = str(chron.iloc[i - 1].get('MATCHUP', ''))
            t_miles, t_tz, t_alt = travel_features_for_game(matchup, prev_matchup, own_abbr)

            is_playoff = str(current.get('SEASON_ID', '')).startswith('4')

            _hist_foul_rate = 0.0
            if 'PF' in history.columns and 'MIN' in history.columns:
                _h5 = history.head(5)
                _h5_min = _h5['MIN'].sum()
                if _h5_min > 0:
                    _h5_pf = _h5['PF'].fillna(0).sum()
                    _hist_foul_rate = float(_h5_pf / _h5_min)

            feats = self.build_features(
                history, is_home, rest, is_playoff, opp_abbr, opp_def, pace,
                opp_pace_norm, opp_reb_pct, opp_paint,
                travel_miles=t_miles, tz_shift_hours=t_tz, altitude_flag=t_alt,
                player_foul_rate=_hist_foul_rate,
            )
            if feats is None:
                continue

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
        except Exception as _e:
            import traceback as _tb
            from src.utils.logging_utils import get_logger as _gl
            _gl(__name__).error(f"XGB train failed for {self.market}: {_e}\n{_tb.format_exc()}")
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


# =========================================================================
# Ensemble model: XGBoost + RandomForest + Ridge stacking
# =========================================================================

class EnsembleModel:
    """
    Stacking ensemble for a single prop market.

    Base learners:
      1. XGBRegressor  (tree-based, handles interactions)
      2. RandomForest   (tree-based, reduces XGB overfitting)
      3. LinearRegression (linear, captures main effects XGB might overfit)

    Meta-learner: Ridge regression on base learner OOF predictions.
    """

    def __init__(self, market: str):
        self.market = market
        self.col = _MARKET_COL.get(market, 'PTS')
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.ensemble_path = os.path.join(MODEL_DIR, f'{market}_ensemble.pkl')
        self.ensemble = None
        self._load()

    def _load(self):
        if os.path.exists(self.ensemble_path):
            try:
                import joblib
                loaded = joblib.load(self.ensemble_path)
                stored_feats = loaded.get('feature_names', []) if isinstance(loaded, dict) else []
                if stored_feats != FEATURE_NAMES:
                    # Feature set changed since training → discard rather than
                    # feed a wrong-shape vector. Scan will fall back to the
                    # Bayesian projection until train_ml is re-run.
                    self.ensemble = None
                else:
                    self.ensemble = loaded
            except Exception:
                self.ensemble = None

    def train(self, X: List, y: List, sample_weight: Optional[List] = None) -> bool:
        """Train 3 base learners + Ridge meta-learner via 3-fold stacking."""
        if len(X) < MIN_TRAIN_SAMPLES:
            return False
        try:
            import joblib
            import xgboost as xgb
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression, Ridge
            from sklearn.model_selection import KFold

            X_arr = np.array(X)
            y_arr = np.array(y)
            sw = np.array(sample_weight) if sample_weight else None

            # Base learners
            xgb_model = xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
            )
            rf_model = RandomForestRegressor(
                n_estimators=150, max_depth=8, min_samples_leaf=5,
                max_features=0.7, random_state=42, n_jobs=-1,
            )
            lr_model = LinearRegression()

            base_learners = [
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('lr', lr_model),
            ]

            # Generate OOF predictions for meta-learner training
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            oof_preds = np.zeros((len(X_arr), len(base_learners)))

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_arr)):
                X_train, X_val = X_arr[train_idx], X_arr[val_idx]
                y_train = y_arr[train_idx]
                sw_train = sw[train_idx] if sw is not None else None

                for bl_idx, (name, model) in enumerate(base_learners):
                    model_clone = _clone_model(model)
                    if name == 'xgb' and sw_train is not None:
                        model_clone.fit(X_train, y_train, sample_weight=sw_train)
                    elif name == 'rf' and sw_train is not None:
                        model_clone.fit(X_train, y_train, sample_weight=sw_train)
                    else:
                        model_clone.fit(X_train, y_train)
                    oof_preds[val_idx, bl_idx] = model_clone.predict(X_val)

            # Train meta-learner on OOF predictions
            meta = Ridge(alpha=0.5)
            meta.fit(oof_preds, y_arr)

            # Retrain base learners on full data
            trained_bases = []
            for name, model in base_learners:
                if name in ('xgb', 'rf') and sw is not None:
                    model.fit(X_arr, y_arr, sample_weight=sw)
                else:
                    model.fit(X_arr, y_arr)
                trained_bases.append((name, model))

            self.ensemble = {
                'bases': trained_bases,
                'meta': meta,
                'feature_names': FEATURE_NAMES,
            }
            joblib.dump(self.ensemble, self.ensemble_path)
            return True
        except Exception as _e:
            import traceback as _tb
            from src.utils.logging_utils import get_logger as _gl
            _gl(__name__).error(f"Ensemble train failed for {self.market}: {_e}\n{_tb.format_exc()}")
            self.ensemble = None
            return False

    def predict_rate(self, features: Dict) -> Optional[float]:
        """Predict per-minute rate using ensemble."""
        if self.ensemble is None:
            return None
        try:
            feat_vec = np.array([[features[k] for k in FEATURE_NAMES]])
            base_preds = []
            for name, model in self.ensemble['bases']:
                base_preds.append(model.predict(feat_vec)[0])
            meta_input = np.array([base_preds])
            return float(self.ensemble['meta'].predict(meta_input)[0])
        except Exception:
            return None


def _clone_model(model):
    """Clone a sklearn/xgb model by re-instantiating with same params."""
    from sklearn.base import clone
    return clone(model)


# =========================================================================
# Public API
# =========================================================================

def get_ml_projection(market: str, logs: pd.DataFrame,
                      proj_minutes: float,
                      home_flag: bool = False,
                      rest_days: int = 2,
                      playoff_flag: bool = False,
                      opp_abbr: str = "",
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
                      pnr_roll_freq: float = 0.0,
                      iso_freq: float = 0.0,
                      spotup_freq: float = 0.0,
                      transition_freq: float = 0.0,
                      postup_freq: float = 0.0,
                      drives_per_game: float = 0.0,
                      ts_pct: float = 0.0,
                      # BDL V2 advanced tracking features
                      avg_speed: float = 0.0,
                      avg_contested_fg_pct: float = 0.0,
                      avg_deflections: float = 0.0,
                      avg_points_paint: float = 0.0,
                      avg_pct_pts_paint: float = 0.0,
                      player_foul_rate: float = 0.0) -> Optional[float]:
    """
    Public API: return ML mean projection for a player/market.
    Returns None if model is untrained or data insufficient.

    Tries ensemble first; falls back to standalone XGBoost.
    """
    if proj_minutes <= 0:
        return None

    _bdl_kwargs = dict(
        real_usage_pct=real_usage_pct, avg_touches=avg_touches,
        pnr_bh_freq=pnr_bh_freq, pnr_roll_freq=pnr_roll_freq,
        iso_freq=iso_freq, spotup_freq=spotup_freq,
        transition_freq=transition_freq, postup_freq=postup_freq,
        drives_per_game=drives_per_game, ts_pct=ts_pct,
        avg_speed=avg_speed, avg_contested_fg_pct=avg_contested_fg_pct,
        avg_deflections=avg_deflections, avg_points_paint=avg_points_paint,
        avg_pct_pts_paint=avg_pct_pts_paint,
        player_foul_rate=player_foul_rate,
    )
    _travel_kwargs = dict(travel_miles=travel_miles, tz_shift_hours=tz_shift_hours,
                          altitude_flag=altitude_flag)

    if market == 'player_points_rebounds_assists':
        pts = get_ml_projection('player_points',  logs, proj_minutes, home_flag, rest_days, playoff_flag, opp_abbr,
                                opp_def_rating, pace_factor, opp_pace, opp_rebound_pct, opp_pts_paint,
                                **_travel_kwargs, **_bdl_kwargs)
        reb = get_ml_projection('player_rebounds', logs, proj_minutes, home_flag, rest_days, playoff_flag, opp_abbr,
                                opp_def_rating, pace_factor, opp_pace, opp_rebound_pct, opp_pts_paint,
                                **_travel_kwargs, **_bdl_kwargs)
        ast = get_ml_projection('player_assists',  logs, proj_minutes, home_flag, rest_days, playoff_flag, opp_abbr,
                                opp_def_rating, pace_factor, opp_pace, opp_rebound_pct, opp_pts_paint,
                                **_travel_kwargs, **_bdl_kwargs)
        if any(v is None for v in [pts, reb, ast]):
            return None
        return pts + reb + ast

    # Build features once, try ensemble then XGB fallback
    prop_model = PropMLModel(market)
    feats = prop_model.build_features(logs, home_flag, rest_days, playoff_flag, opp_abbr, opp_def_rating, pace_factor,
                                      opp_pace, opp_rebound_pct, opp_pts_paint,
                                      **_travel_kwargs, **_bdl_kwargs)
    if feats is None:
        return None

    # Try ensemble first
    ens = EnsembleModel(market)
    rate = ens.predict_rate(feats)

    # Fallback to standalone XGB
    if rate is None:
        rate = prop_model.predict_rate(feats)

    if rate is None:
        return None

    return max(0.0, rate * proj_minutes)


# =========================================================================
# Training functions
# =========================================================================

def train_models_from_logs(player_logs_list: List[pd.DataFrame],
                           opp_stats_df: Optional[pd.DataFrame] = None,
                           team_stats_df: Optional[pd.DataFrame] = None,
                           def_stats_df: Optional[pd.DataFrame] = None,
                           league_avg_pace: float = 99.0) -> Dict[str, bool]:
    """
    Train ensemble models for all markets using a list of player game log DataFrames.
    Also trains standalone XGBoost as fallback.
    Returns {market: success_bool}.
    """
    from src.utils.logging_utils import get_logger
    log = get_logger(__name__)

    pace_lookup: Optional[Dict[str, float]] = None
    if team_stats_df is not None and not team_stats_df.empty and 'PACE' in team_stats_df.columns:
        pace_lookup = _build_abbr_lookup(team_stats_df, 'PACE')
        if pace_lookup:
            league_avg_pace = float(np.mean(list(pace_lookup.values())))

    opp_rebound_pct_lookup: Optional[Dict[str, float]] = None
    if team_stats_df is not None and not team_stats_df.empty and 'DREB_PCT' in team_stats_df.columns:
        raw = _build_abbr_lookup(team_stats_df, 'DREB_PCT')
        league_avg_dreb = float(np.mean(list(raw.values()))) if raw else 0.0
        if league_avg_dreb > 0:
            opp_rebound_pct_lookup = {k: v / league_avg_dreb for k, v in raw.items()}

    opp_pts_paint_lookup: Optional[Dict[str, float]] = None
    if def_stats_df is not None and not def_stats_df.empty and 'OPP_PTS_PAINT' in def_stats_df.columns:
        raw = _build_abbr_lookup(def_stats_df, 'OPP_PTS_PAINT')
        league_avg_paint = float(np.mean(list(raw.values()))) if raw else 0.0
        if league_avg_paint > 0:
            opp_pts_paint_lookup = {k: v / league_avg_paint for k, v in raw.items()}

    results = {}
    for market in _MARKET_COL:
        prop_model = PropMLModel(market)
        ens_model  = EnsembleModel(market)

        opp_def_lookup: Optional[Dict[str, float]] = None
        opp_col = _MARKET_OPP_STAT.get(market)
        if opp_stats_df is not None and not opp_stats_df.empty and opp_col and opp_col in opp_stats_df.columns:
            raw = _build_abbr_lookup(opp_stats_df, opp_col)
            league_avg_def = float(np.mean(list(raw.values()))) if raw else 1.0
            opp_def_lookup = {k: v / league_avg_def for k, v in raw.items()} if league_avg_def > 0 else raw

        all_X, all_y = [], []
        for logs in player_logs_list:
            X, y = prop_model.build_training_data(
                logs, opp_def_lookup, pace_lookup, league_avg_pace,
                opp_rebound_pct_lookup, opp_pts_paint_lookup,
            )
            all_X.extend(X)
            all_y.extend(y)

        # Train ensemble (primary)
        ens_ok = ens_model.train(all_X, all_y)
        # Train standalone XGB (fallback)
        xgb_ok = prop_model.train(all_X, all_y)

        success = ens_ok or xgb_ok
        results[market] = success
        log.info(f"ML model [{market}]: ensemble={'ok' if ens_ok else 'skip'} "
                 f"xgb={'ok' if xgb_ok else 'skip'} "
                 f"({len(all_X)} samples)")
    return results


# ── CLV Feedback Helpers ──────────────────────────────────────────────────────

def _norm_player_name(name: str) -> str:
    """Lowercase + strip accents + normalise whitespace for CLV key matching."""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', name)
    return ' '.join(nfkd.encode('ascii', 'ignore').decode('ascii').lower().split())


def build_clv_weight_lookup(db) -> Dict:
    """
    Query clv_tracking for all settled records and return a sample-weight dict.
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
    Trains both ensemble and standalone XGB with CLV sample weights.
    """
    from src.utils.logging_utils import get_logger
    log = get_logger(__name__)

    clv_lookup = build_clv_weight_lookup(db)
    log.info(f"CLV feedback: {len(clv_lookup)} CLV weight records loaded.")

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
        prop_model = PropMLModel(market)
        ens_model  = EnsembleModel(market)

        opp_def_lookup: Optional[Dict[str, float]] = None
        opp_col = _MARKET_OPP_STAT.get(market)
        if opp_stats_df is not None and not opp_stats_df.empty and opp_col and opp_col in opp_stats_df.columns:
            raw = _build_abbr_lookup(opp_stats_df, opp_col)
            lg_avg = float(np.mean(list(raw.values()))) if raw else 1.0
            opp_def_lookup = {k: v / lg_avg for k, v in raw.items()} if lg_avg > 0 else raw

        all_X, all_y, all_w = [], [], []
        for player_name, logs in player_named_logs:
            X, y, w = prop_model.build_training_data(
                logs, opp_def_lookup, pace_lookup, league_avg_pace,
                opp_rebound_pct_lookup, opp_pts_paint_lookup,
                player_name=player_name,
                clv_weight_lookup=clv_lookup,
                return_weights=True,
            )
            all_X.extend(X)
            all_y.extend(y)
            all_w.extend(w)

        sw = all_w if all_w else None
        ens_ok = ens_model.train(all_X, all_y, sample_weight=sw)
        xgb_ok = prop_model.train(all_X, all_y, sample_weight=sw)

        success = ens_ok or xgb_ok
        results[market] = success
        beats  = sum(1 for w in all_w if w > 1.0)
        misses = sum(1 for w in all_w if w < 1.0)
        log.info(f"CLV model [{market}]: ensemble={'ok' if ens_ok else 'skip'} "
                 f"xgb={'ok' if xgb_ok else 'skip'} | CLV beats={beats} misses={misses}")

    return results


def _build_abbr_lookup(df: pd.DataFrame, value_col: str) -> Dict[str, float]:
    """Build {team_abbr_upper: value} from a DataFrame."""
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
