"""
Model drift monitor.

Runs daily after settlement.  Computes rolling Brier scores over the last 100
and 500 settled non-push bets (plus an all-time baseline) and compares them to
the historical mean stored in `model_health`.

Alarm logic:
  • If rolling_100 Brier is > (all_time_mean + 2 × all_time_std) → ALARM.
  • If rolling_500 Brier is > (all_time_mean + 1.5 × all_time_std) → WARNING.
  • If either alarm fires AND enough data exists → trigger an immediate retrain.

The model probability is reconstructed as: model_prob = edge + 1.0/odds
(the same formula used in train_calibration.py:57).  Pushes are excluded.

Per-market Brier is also computed for any market with ≥30 settled bets, giving
early warning of market-specific degradation (e.g. assists model going stale).
"""

import math
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.data.db import DatabaseClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Thresholds
_ALARM_SIGMA   = 2.0    # rolling_100 deviation to fire ALARM
_WARN_SIGMA    = 1.5    # rolling_500 deviation to fire WARNING
_MIN_BASELINE  = 30     # minimum all-time samples before comparing
_MIN_PER_MKT   = 30     # minimum settled bets per market for per-market Brier


# ---------------------------------------------------------------------------
# Core Brier helpers
# ---------------------------------------------------------------------------

def _brier(probs: List[float], outcomes: List[int]) -> float:
    """Mean squared error between predicted probs and 0/1 outcomes."""
    if not probs:
        return float('nan')
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def _fetch_settled(db: DatabaseClient, limit: Optional[int] = None,
                   market: str = None) -> List[Tuple[float, int]]:
    """
    Return [(model_prob, won)] from settled non-push alerts, newest-first.
    model_prob = edge + 1/odds  (reconstructed; same as train_calibration.py).
    """
    where_clauses = ["b.push = 0", "b.won IS NOT NULL", "a.odds > 1.0"]
    if market:
        where_clauses.append("a.market = ?")
    where = " AND ".join(where_clauses)

    sql = f"""
        SELECT (a.edge + 1.0 / a.odds) AS model_prob,
               CAST(b.won AS INTEGER)   AS outcome
        FROM alerts_sent a
        JOIN bet_results b ON a.id = b.alert_id
        WHERE {where}
        ORDER BY COALESCE(b.settled_at, a.timestamp) DESC
        {"LIMIT ?" if limit else ""}
    """
    params: list = []
    if market:
        params.append(market)
    if limit:
        params.append(limit)

    with db.get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [(float(r['model_prob']), int(r['outcome'])) for r in rows
            if 0.0 <= float(r['model_prob']) <= 1.0]


def _historical_brier_stats(db: DatabaseClient) -> Tuple[float, float, int]:
    """
    Return (mean_brier, std_brier, n_snapshots) from the all_time snapshots
    stored in model_health.  Used as the baseline for sigma comparison.
    """
    with db.get_conn() as conn:
        rows = conn.execute(
            """SELECT brier FROM model_health
               WHERE window = 'all_time' AND market = 'all'
               ORDER BY snapshot_date DESC
               LIMIT 90"""
        ).fetchall()
    vals = [float(r['brier']) for r in rows if not math.isnan(float(r['brier']))]
    if len(vals) < 2:
        return float('nan'), float('nan'), len(vals)
    return statistics.mean(vals), statistics.stdev(vals), len(vals)


# ---------------------------------------------------------------------------
# Snapshot writer
# ---------------------------------------------------------------------------

def _write_snapshot(db: DatabaseClient, window: str, market: str,
                    brier: float, n: int):
    today = datetime.now().strftime('%Y-%m-%d')
    with db.get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO model_health
                   (snapshot_date, window, market, brier, n_samples)
               VALUES (?, ?, ?, ?, ?)""",
            (today, window, market, brier, n),
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_drift(bot=None) -> Dict:
    """
    Compute drift metrics, write snapshots to DB, alert if alarmed.

    Returns:
        {
            'brier_100':  float,
            'brier_500':  float,
            'brier_all':  float,
            'alarm':      bool,   # immediate retrain warranted
            'warning':    bool,
            'n_100':      int,
            'per_market': {market: brier},
        }
    """
    db = DatabaseClient()
    result: Dict = {
        'brier_100': float('nan'), 'brier_500': float('nan'),
        'brier_all': float('nan'), 'alarm': False, 'warning': False,
        'n_100': 0, 'per_market': {},
    }

    # ── Rolling windows ───────────────────────────────────────────────────
    rows_100 = _fetch_settled(db, limit=100)
    rows_500 = _fetch_settled(db, limit=500)
    rows_all = _fetch_settled(db)

    n_100 = len(rows_100)
    n_all = len(rows_all)

    if n_100 < 10:
        logger.info(f"Drift monitor: only {n_100} settled bets — skipping alarm check.")
        return result

    p100, o100 = zip(*rows_100)
    brier_100 = _brier(list(p100), list(o100))
    result['brier_100'] = brier_100
    result['n_100'] = n_100
    _write_snapshot(db, 'last_100', 'all', brier_100, n_100)

    if rows_500:
        p500, o500 = zip(*rows_500)
        brier_500 = _brier(list(p500), list(o500))
        result['brier_500'] = brier_500
        _write_snapshot(db, 'last_500', 'all', brier_500, len(rows_500))

    if rows_all:
        pall, oall = zip(*rows_all)
        brier_all = _brier(list(pall), list(oall))
        result['brier_all'] = brier_all
        _write_snapshot(db, 'all_time', 'all', brier_all, n_all)

    # ── Per-market Brier ──────────────────────────────────────────────────
    markets = [
        'player_points', 'player_rebounds', 'player_assists',
        'player_threes', 'player_points_rebounds_assists',
        'player_blocks', 'player_steals',
    ]
    per_market: Dict[str, float] = {}
    for mkt in markets:
        rows = _fetch_settled(db, market=mkt)
        if len(rows) < _MIN_PER_MKT:
            continue
        pm, om = zip(*rows)
        b = _brier(list(pm), list(om))
        per_market[mkt] = b
        _write_snapshot(db, 'all_time', mkt, b, len(rows))
    result['per_market'] = per_market

    # ── Sigma comparison against historical baseline ───────────────────────
    baseline_mean, baseline_std, n_snapshots = _historical_brier_stats(db)

    if n_snapshots < 7 or math.isnan(baseline_mean):
        # Not enough history for a meaningful sigma comparison yet
        logger.info(
            f"Drift monitor: Brier_100={brier_100:.4f} | "
            f"building baseline ({n_snapshots} snapshots so far)."
        )
        _log_summary(result, per_market)
        return result

    sigma_100 = (brier_100 - baseline_mean) / baseline_std if baseline_std > 0 else 0.0
    sigma_500 = (result['brier_500'] - baseline_mean) / baseline_std \
        if (not math.isnan(result['brier_500']) and baseline_std > 0) else 0.0

    alarm   = sigma_100 > _ALARM_SIGMA
    warning = sigma_500 > _WARN_SIGMA and not alarm
    result['alarm']   = alarm
    result['warning'] = warning

    if alarm or warning:
        level = "🚨 ALARM" if alarm else "⚠️ WARNING"
        mkt_lines = "\n".join(
            f"  {m.replace('player_', '').title()}: {b:.4f}"
            for m, b in sorted(per_market.items(), key=lambda x: -x[1])
        ) or "  (insufficient per-market data)"

        msg = (
            f"{level} <b>Model drift detected</b>\n\n"
            f"Brier last 100: <b>{brier_100:.4f}</b> "
            f"({sigma_100:+.1f}σ vs baseline {baseline_mean:.4f}±{baseline_std:.4f})\n"
            f"Brier last 500: {result['brier_500']:.4f} ({sigma_500:+.1f}σ)\n"
            f"All-time: {result['brier_all']:.4f} ({n_all} bets)\n\n"
            f"Per-market Brier:\n{mkt_lines}"
        )
        if alarm:
            msg += "\n\n<b>Auto-retrain triggered.</b>"
        logger.warning(f"Drift {level}: σ_100={sigma_100:.2f}  σ_500={sigma_500:.2f}")
        if bot:
            try:
                bot.send_message(msg)
            except Exception:
                pass
    else:
        logger.info(
            f"Drift monitor: Brier_100={brier_100:.4f} ({sigma_100:+.1f}σ) — OK"
        )

    _log_summary(result, per_market)
    return result


def _log_summary(result: Dict, per_market: Dict[str, float]):
    mkt_str = "  " + "  ".join(
        f"{m.replace('player_','')[:4]}={b:.4f}" for m, b in per_market.items()
    )
    logger.info(
        f"Drift summary | "
        f"Brier[100]={result['brier_100']:.4f}  "
        f"Brier[500]={result['brier_500']:.4f}  "
        f"Brier[all]={result['brier_all']:.4f}\n"
        f"Per-market: {mkt_str}"
    )


if __name__ == "__main__":
    check_drift()
