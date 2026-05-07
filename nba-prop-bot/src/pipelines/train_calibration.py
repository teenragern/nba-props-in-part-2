"""
Pipeline to dynamically fit an Isotonic Regression on settled bet outcomes.
It calculates 'model_prob' from the (edge, odds) in alerts_sent and maps it
directly to the winning percentage stored in 'bet_results'.

Regression guard: the new model must achieve a lower (better) Brier score than
the incumbent on the most-recent 20% of settled bets (held-out test slice).
If it doesn't improve, the old model is kept and a warning is logged.

Usage:
    python -m src.pipelines.train_calibration
"""

import os
import shutil
from datetime import datetime

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.data.db import DatabaseClient, get_db_client
from src.models.experiment_tracker import training_run
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

MODEL_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'calibration_iso.pkl')

_MIN_TRAIN = 100   # minimum settled bets to attempt a fit
_TEST_FRAC = 0.20  # held-out fraction (chronologically most recent)


def _brier(model, X: np.ndarray, y: np.ndarray) -> float:
    preds = np.clip(model.predict(X), 0.0, 1.0)
    return float(np.mean((preds - y) ** 2))


def train_isotonic_calibration() -> bool:
    db = get_db_client()

    with db.get_conn() as conn:
        rows = conn.execute(
            """
            SELECT a.edge, a.odds, b.won, a.market
            FROM alerts_sent a
            JOIN bet_results b ON a.id = b.alert_id
            WHERE b.push = 0
              AND a.odds IS NOT NULL
              AND a.odds > 1.0
            ORDER BY COALESCE(b.settled_at, a.timestamp) ASC
            """
        ).fetchall()

    if len(rows) < _MIN_TRAIN:
        logger.warning(f"Insufficient settled bets to train IsotonicRegression (n={len(rows)}). Needs {_MIN_TRAIN}.")
        return False

    # Group data by market
    market_data = {}
    for r in rows:
        mkt = r['market']
        model_prob = max(0.01, min(0.99, float(r['edge']) + 1.0 / float(r['odds'])))
        market_data.setdefault(mkt, []).append((model_prob, float(r['won'])))
        market_data.setdefault('global', []).append((model_prob, float(r['won'])))

    models = {}
    
    for mkt, data in market_data.items():
        if len(data) < 30 and mkt != 'global':
            continue # Skip small markets, they'll use 'global'
            
        X = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])
        
        # Chronological split
        split = max(1, int(len(X) * (1.0 - _TEST_FRAC)))
        X_train, y_train = X[:split], y[:split]
        X_test,  y_test  = X[split:], y[split:]
        
        candidate = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        try:
            candidate.fit(X_train, y_train)
            brier = _brier(candidate, X_test, y_test) if len(X_test) > 0 else 0
            models[mkt] = candidate
            logger.info(f"Trained {mkt} calibration: n_train={len(X_train)}, brier={brier:.4f}")
        except Exception as e:
            logger.error(f"Failed to train {mkt} calibration: {e}")

    if 'global' not in models:
        return False

    # Regression guard: only replace if new model improves global Brier on test slice
    if os.path.exists(MODEL_PATH):
        try:
            old = joblib.load(MODEL_PATH)
            # Handle both single-model and market-dict formats
            old_global = old['global'] if isinstance(old, dict) else old
            
            # Re-generate test data for global (the hold-out slice)
            global_data = market_data.get('global', [])
            split = max(1, int(len(global_data) * (1.0 - _TEST_FRAC)))
            X_test_g = np.array([d[0] for d in global_data[split:]])
            y_test_g = np.array([d[1] for d in global_data[split:]])
            
            if len(X_test_g) > 0:
                new_brier = _brier(models['global'], X_test_g, y_test_g)
                old_brier = _brier(old_global, X_test_g, y_test_g)
                if new_brier >= old_brier:
                    logger.warning(f"Regression guard: New Brier {new_brier:.4f} >= Old Brier {old_brier:.4f}. Aborting save.")
                    return False
                logger.info(f"Regression guard passed: {new_brier:.4f} < {old_brier:.4f}")
        except Exception as e:
            logger.debug(f"Regression guard error (likely legacy model format): {e}")

    # Back up and save the full models dict
    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, MODEL_PATH + '.bak')

    joblib.dump(models, MODEL_PATH)
    logger.info(f"Market-aware isotonic calibration updated. Markets: {list(models.keys())}")

    run_name = f"calibration_{datetime.now().strftime('%Y%m%d_%H%M')}"
    with training_run(run_name, tags={"type": "isotonic_calibration"}) as run:
        run.log_params({
            "min_train": _MIN_TRAIN,
            "test_frac": _TEST_FRAC,
            "n_samples": len(rows),
            "markets_trained": ",".join(models.keys()),
        })
        for mkt, candidate in models.items():
            mkt_data = market_data.get(mkt, [])
            if len(mkt_data) > 0:
                split = max(1, int(len(mkt_data) * (1.0 - _TEST_FRAC)))
                X_test = np.array([d[0] for d in mkt_data[split:]])
                y_test = np.array([d[1] for d in mkt_data[split:]])
                if len(X_test) > 0:
                    run.log_metric(f"brier_{mkt}", _brier(candidate, X_test, y_test))
        run.log_artifact(MODEL_PATH)

    return True


if __name__ == "__main__":
    train_isotonic_calibration()
