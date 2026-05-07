"""
Lightweight MLflow wrapper for the NBA props bot training pipelines.

Falls back to a no-op tracker when MLFLOW_TRACKING_URI is unset, so all
training pipelines work unchanged in environments without MLflow configured.

Usage:
    from src.models.experiment_tracker import training_run

    with training_run("train_ml_2026-05-05", tags={"type": "xgboost_weekly"}) as run:
        run.log_param("seasons", 3)
        run.log_metric("brier_points", 0.214)
        run.log_artifact("/path/to/model.pkl")
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tracker implementations
# ---------------------------------------------------------------------------

class _MlflowTracker:
    """Thin wrapper around an active MLflow run."""

    def __init__(self, active_run):
        self._run = active_run

    def log_param(self, key: str, value: Any):
        import mlflow
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]):
        import mlflow
        mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        import mlflow
        if value is None or (isinstance(value, float) and (value != value)):  # NaN guard
            return
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for k, v in metrics.items():
            self.log_metric(k, v, step=step)

    def log_artifact(self, local_path: str):
        import mlflow
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path)
        else:
            logger.debug(f"MLflow: artifact not found, skipping: {local_path}")

    def set_tag(self, key: str, value: str):
        import mlflow
        mlflow.set_tag(key, value)


class _NoOpTracker:
    """All methods are no-ops. Training pipelines work without any MLflow config."""

    def log_param(self, *a, **kw):     pass
    def log_params(self, *a, **kw):    pass
    def log_metric(self, *a, **kw):    pass
    def log_metrics(self, *a, **kw):   pass
    def log_artifact(self, *a, **kw):  pass
    def set_tag(self, *a, **kw):       pass


# ---------------------------------------------------------------------------
# Public context manager
# ---------------------------------------------------------------------------

@contextmanager
def training_run(run_name: str, tags: Optional[Dict[str, str]] = None):
    """
    Context manager yielding a tracker object.

    When MLFLOW_TRACKING_URI is set, opens a real MLflow run and logs to the
    configured server.  When unset, yields a _NoOpTracker — training pipelines
    continue to work with zero configuration overhead.

    Args:
        run_name: Human-readable name for the run (e.g. "train_ml_2026-05-05").
        tags:     Optional dict of string tags attached to the run.

    Yields:
        _MlflowTracker | _NoOpTracker
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        yield _NoOpTracker()
        return

    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_name=run_name, tags=tags) as active_run:
            logger.debug(f"MLflow run started: {run_name} (id={active_run.info.run_id})")
            yield _MlflowTracker(active_run)
            logger.debug(f"MLflow run finished: {run_name}")
    except Exception as e:
        logger.warning(f"MLflow unavailable ({e}) — falling back to no-op tracker.")
        yield _NoOpTracker()
