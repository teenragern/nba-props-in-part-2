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

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.data.db import DatabaseClient
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
    db = DatabaseClient()

    with db.get_conn() as conn:
        rows = conn.execute(
            """
            SELECT a.edge, a.odds, b.won
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

    X_all, y_all = [], []
    for r in rows:
        model_prob = float(r['edge']) + 1.0 / float(r['odds'])
        model_prob = max(0.01, min(0.99, model_prob))
        X_all.append(model_prob)
        y_all.append(float(r['won']))

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # Chronological train/test split — test = most recent _TEST_FRAC rows
    split = max(1, int(len(X_all) * (1.0 - _TEST_FRAC)))
    X_train, y_train = X_all[:split], y_all[:split]
    X_test,  y_test  = X_all[split:], y_all[split:]

    # Fit candidate model on training slice
    candidate = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
    try:
        candidate.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"IsotonicRegression fit failed: {e}")
        return False

    candidate_brier = _brier(candidate, X_test, y_test)
    logger.info(f"Calibration candidate Brier on held-out {len(X_test)} bets: {candidate_brier:.4f}")

    # Regression guard: compare against incumbent if one exists
    if os.path.exists(MODEL_PATH) and len(X_test) >= 20:
        try:
            incumbent = joblib.load(MODEL_PATH)
            incumbent_brier = _brier(incumbent, X_test, y_test)
            logger.info(f"Calibration incumbent Brier on same slice: {incumbent_brier:.4f}")
            if candidate_brier >= incumbent_brier:
                logger.warning(
                    f"Calibration regression guard: candidate ({candidate_brier:.4f}) is not "
                    f"better than incumbent ({incumbent_brier:.4f}) — keeping old model."
                )
                return False
        except Exception as e:
            logger.warning(f"Could not load incumbent model for comparison: {e}")

    # Back up old model before overwriting
    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, MODEL_PATH + '.bak')

    joblib.dump(candidate, MODEL_PATH)
    logger.info(
        f"Isotonic calibration updated: trained on {len(X_train)}, "
        f"held-out Brier {candidate_brier:.4f} ({len(X_test)} bets)."
    )
    return True


if __name__ == "__main__":
    train_isotonic_calibration()
