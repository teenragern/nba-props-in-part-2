"""
Priority 6: XGBoost ensemble model for per-minute stat rate prediction.

Trains one XGBRegressor per market (points/rebounds/assists/threes) on
historical player game logs. Predictions are blended 50/50 with the
Bayesian model in scan_props.py.

Feature vector per game prediction:
  - rate_5g:      per-minute rate over last 5 games
  - rate_10g:     per-minute rate over last 10 games
  - rate_season:  season-long per-minute rate
  - avg_min_5g:   average minutes last 5 games
  - std_min_5g:   std-dev of minutes last 5 games (consistency)
  - home_flag:    1 = home, 0 = away
  - rest_days:    days since last game (capped at 7)
  - b2b_flag:     1 = back-to-back
  - games_played: total season games (proxy for small-sample risk)

Target: per-minute rate for the specific stat.
Final projection = predicted_rate * projected_minutes.
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
    'rate_5g', 'rate_10g', 'rate_season',
    'avg_min_5g', 'std_min_5g',
    'home_flag', 'rest_days', 'b2b_flag', 'games_played',
]


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

    def build_features(self, logs: pd.DataFrame,
                       home_flag: bool = False,
                       rest_days: int = 2) -> Optional[Dict]:
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
            'rate_5g':      self._safe_rate(r5),
            'rate_10g':     self._safe_rate(r10),
            'rate_season':  self._safe_rate(logs),
            'avg_min_5g':   float(avg_min)  if not pd.isna(avg_min)  else 0.0,
            'std_min_5g':   float(std_min)  if not pd.isna(std_min)  else 0.0,
            'home_flag':    int(home_flag),
            'rest_days':    min(rest_days, 7),
            'b2b_flag':     int(rest_days == 0),
            'games_played': len(logs),
        }

    def build_training_data(self, logs: pd.DataFrame) -> Tuple[List, List]:
        """
        Generate (X, y) training samples from a player's chronological game logs.
        Requires at least 15 games. Only uses past games as features (no lookahead).
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

            try:
                cur_date  = pd.to_datetime(current['GAME_DATE'])
                prev_date = pd.to_datetime(chron.iloc[i - 1]['GAME_DATE'])
                rest = max(0, int((cur_date - prev_date).days) - 1)
            except Exception:
                rest = 2

            feats = self.build_features(history, is_home, rest)
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
                      rest_days: int = 2) -> Optional[float]:
    """
    Public API: return ML mean projection for a player/market.
    Returns None if model is untrained or data insufficient.
    Caller blends this 50/50 with the Bayesian projection.
    """
    if proj_minutes <= 0:
        return None

    if market == 'player_points_rebounds_assists':
        pts = get_ml_projection('player_points',   logs, proj_minutes, home_flag, rest_days)
        reb = get_ml_projection('player_rebounds',  logs, proj_minutes, home_flag, rest_days)
        ast = get_ml_projection('player_assists',   logs, proj_minutes, home_flag, rest_days)
        if any(v is None for v in [pts, reb, ast]):
            return None
        return pts + reb + ast

    model = PropMLModel(market)
    feats = model.build_features(logs, home_flag, rest_days)
    if feats is None:
        return None

    rate = model.predict_rate(feats)
    if rate is None:
        return None

    return max(0.0, rate * proj_minutes)


def train_models_from_logs(player_logs_list: List[pd.DataFrame]) -> Dict[str, bool]:
    """
    Train XGBoost models for all markets using a list of player game log DataFrames.
    Call this from a pipeline (e.g., python main.py train_ml).
    Returns {market: success_bool}.
    """
    results = {}
    for market in _MARKET_COL:
        model = PropMLModel(market)
        all_X, all_y = [], []
        for logs in player_logs_list:
            X, y = model.build_training_data(logs)
            all_X.extend(X)
            all_y.extend(y)
        success = model.train(all_X, all_y)
        results[market] = success
        status = "trained" if success else f"skipped (need {MIN_TRAIN_SAMPLES} samples, got {len(all_X)})"
        from src.utils.logging_utils import get_logger
        get_logger(__name__).info(f"ML model [{market}]: {status}")
    return results
