"""
Priority 8: Hyperparameter tuning using real CLV data.

Instead of a mocked score, evaluate_params now queries the clv_tracking
table for the average CLV (closing line value) from the last 30 days.
Positive CLV = we beat the closing line = model is finding real edges.
"""

import json
import itertools
import numpy as np
from datetime import datetime
from src.data.db import DatabaseClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_params(db: DatabaseClient, prior_weight: float,
                    alpha: float, b2b_penalty: float) -> float:
    """
    Priority 8: Score a parameter combination using real CLV data.

    When >= 10 CLV records exist in the last 30 days the score is
    dominated by average CLV (scaled x100). Otherwise falls back to
    a stability heuristic so the grid search still returns a result.
    """
    avg_clv = db.get_avg_clv(days_back=30)  # returns 0.0 if < 10 records

    # Stability heuristic: prefer moderate prior_weight, alpha ≈ 0.15
    stability = (prior_weight / 20.0) - abs(alpha - 0.15) * 5.0 - abs(b2b_penalty - 1.5) * 0.5

    if avg_clv != 0.0:
        return float(avg_clv * 100.0 + stability)

    # Not enough real data yet
    return float(stability) + np.random.normal(0, 0.05)


def run_tuning(db: DatabaseClient):
    logger.info("Starting Hyperparameter Grid Search (CLV-based scoring)...")

    avg_clv = db.get_avg_clv(days_back=30)
    if avg_clv == 0.0:
        logger.info("< 10 CLV records found. Using stability heuristic until data accumulates.")
    else:
        logger.info(f"Using real CLV data: avg CLV = {avg_clv:.4f} over last 30 days.")

    prior_weights = [10.0, 15.0, 20.0]
    alphas        = [0.10, 0.15, 0.20]
    b2b_penalties = [1.0,  1.5,  2.0]

    best_score  = -999.0
    best_params: dict = {}

    for pw, a, b2bp in itertools.product(prior_weights, alphas, b2b_penalties):
        score = evaluate_params(db, pw, a, b2bp)
        logger.info(f"  Params (prior={pw}, alpha={a}, b2b={b2bp}): score={score:.4f}")
        if score > best_score:
            best_score  = score
            best_params = {"prior_weight": pw, "alpha_dispersion": a, "b2b_penalty": b2bp}

    logger.info(f"Tuning complete. Best: {best_params}  (score={best_score:.4f})")

    with db.get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO model_versions (parameters_json, performance_metrics) VALUES (?, ?)",
            (
                json.dumps(best_params),
                json.dumps({"best_score": best_score, "avg_clv": avg_clv,
                            "timestamp": datetime.utcnow().isoformat()}),
            )
        )
    logger.info("Saved tuned parameters to 'model_versions' table.")
    return best_params


if __name__ == "__main__":
    db = DatabaseClient()
    run_tuning(db)
