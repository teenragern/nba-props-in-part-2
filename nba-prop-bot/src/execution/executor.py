"""
Bet executor — routes each alert to the appropriate execution path.

Modes (set via EXECUTION_MODE env var):
  paper  (default) — records the bet in placed_bets, simulates fill at the
                     best current line_history odds for this player/market/side.
                     Tracks slippage and session P&L.  No real money moves.
  live             — stub.  Real sportsbook adapters go here once paper-trade
                     validation shows consistent positive CLV over ≥ 2 weeks.

Kill-switch:
  EXECUTION_ENABLED=false  →  executor is a no-op.  Useful to pause all
  execution logic without touching code (e.g., during scheduled downtime).

Session loss limit:
  EXECUTION_SESSION_LOSS_LIMIT env var — fraction of daily bankroll.
  Default -0.20 (−20%).  When paper session P&L crosses this threshold the
  executor suspends itself for the rest of the session and sends a Telegram
  warning.  Resets at midnight.

Usage:
  from src.execution.executor import record_execution
  record_execution(alert_id, edge_data, stake, alerted_odds, db, bot)
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.config import BANKROLL
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

EXECUTION_MODE    = os.getenv('EXECUTION_MODE', 'paper').lower()
EXECUTION_ENABLED = os.getenv('EXECUTION_ENABLED', 'true').lower() != 'false'
_SESSION_LOSS_LIMIT: float = float(os.getenv('EXECUTION_SESSION_LOSS_LIMIT', '-0.20'))

# In-process kill-switch toggled when session loss limit is breached.
# Resets when the process restarts (i.e., at next scheduler boot / midnight).
_session_suspended: bool = False


def _session_id() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d')


# ── Fill-price lookup ─────────────────────────────────────────────────────────

def _best_fill_odds(db, player_name: str, market: str, side: str,
                    line: float) -> Optional[float]:
    """
    Return the best (highest) decimal odds currently available in line_history
    for this prop within the last 90 minutes.  This simulates the fill price
    the bot would have gotten if it placed the bet immediately after the alert.
    Returns None when no fresh data is available.
    """
    try:
        with db.get_conn() as conn:
            row = conn.execute(
                """
                SELECT MAX(odds) AS best_odds
                FROM line_history
                WHERE player_name = ?
                  AND market      = ?
                  AND side        = ?
                  AND ABS(line - ?) <= 0.5
                  AND timestamp  >= datetime('now', '-90 minutes')
                """,
                (player_name, market, side.upper(), line),
            ).fetchone()
        if row and row['best_odds']:
            return float(row['best_odds'])
    except Exception as e:
        logger.debug(f"Fill-odds lookup failed: {e}")
    return None


# ── Session P&L ───────────────────────────────────────────────────────────────

def _session_pnl(db) -> float:
    """
    Return today's total paper P&L from placed_bets + bet_results.
    Only counts settled paper bets (won/lost); open bets are ignored.
    """
    try:
        today = _session_id()
        with db.get_conn() as conn:
            row = conn.execute(
                """
                SELECT SUM(
                    CASE WHEN br.won = 1
                         THEN pb.stake * (pb.fill_odds - 1.0)
                         ELSE -pb.stake
                    END
                ) AS pnl
                FROM placed_bets pb
                JOIN bet_results br ON br.alert_id = pb.alert_id
                WHERE pb.session_id = ?
                  AND pb.mode       = 'paper'
                  AND br.push       = 0
                """,
                (today,),
            ).fetchone()
        return float(row['pnl'] or 0.0)
    except Exception:
        return 0.0


def _check_session_limit(db, bot) -> bool:
    """
    Return True (execution OK) or False (session suspended).
    Fires Telegram + sets _session_suspended when limit is breached.
    """
    global _session_suspended
    if _session_suspended:
        return False

    pnl = _session_pnl(db)
    limit = _SESSION_LOSS_LIMIT * BANKROLL
    if pnl <= limit:
        _session_suspended = True
        msg = (
            f"🛑 <b>Execution suspended — session loss limit reached</b>\n\n"
            f"Paper P&L today: <b>${pnl:.2f}</b>\n"
            f"Limit: ${limit:.2f} ({_SESSION_LOSS_LIMIT:.0%} of bankroll)\n\n"
            f"No further bets will be logged until the bot restarts tomorrow."
        )
        logger.warning(f"Session loss limit breached: P&L={pnl:.2f}, limit={limit:.2f}")
        if bot:
            try:
                bot.send_message(msg)
            except Exception:
                pass
        return False
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def record_execution(
    alert_id: int,
    edge_data: Dict[str, Any],
    stake: float,
    alerted_odds: float,
    db,
    bot=None,
) -> bool:
    """
    Record a bet execution (paper or live).

    Returns True if the bet was logged, False if skipped (kill-switch,
    session limit, or EXECUTION_ENABLED=false).
    """
    if not EXECUTION_ENABLED:
        logger.debug("Executor disabled (EXECUTION_ENABLED=false).")
        return False

    if not _check_session_limit(db, bot):
        return False

    player   = edge_data.get('player_id') or edge_data.get('player_name', 'Unknown')
    market   = edge_data.get('market', '')
    side     = edge_data.get('side', '')
    line     = float(edge_data.get('line', 0.0))
    book     = edge_data.get('book', '')
    game_date = edge_data.get('game_date')
    today    = _session_id()

    if EXECUTION_MODE == 'paper':
        fill_odds = _best_fill_odds(db, player, market, side, line)
        slippage  = (fill_odds - alerted_odds) if fill_odds else None
        logger.info(
            f"Paper fill: {player} {market} {side} — "
            f"alerted={alerted_odds:.3f}  fill={fill_odds or 'N/A'}  "
            f"slippage={slippage:+.4f}" if slippage is not None
            else f"Paper fill: {player} {market} {side} — "
                 f"alerted={alerted_odds:.3f}  fill=N/A (no line_history data)"
        )

        try:
            with db.get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO placed_bets
                        (alert_id, mode, player_name, market, side, line, book,
                         alerted_odds, fill_odds, slippage, stake, game_date, session_id)
                    VALUES (?, 'paper', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (alert_id, player, market, side, line, book,
                     alerted_odds, fill_odds, slippage, stake, game_date, today),
                )
            return True
        except Exception as e:
            logger.error(f"Paper bet record failed: {e}")
            return False

    # LIVE mode — stub.  Wire sportsbook adapter here.
    logger.warning(
        "LIVE execution mode is not yet implemented. "
        "Switch EXECUTION_MODE=paper until sportsbook adapters are wired."
    )
    return False


def session_summary(db) -> Dict[str, Any]:
    """
    Return today's paper-trade session stats for Telegram reporting.
    """
    today = _session_id()
    try:
        with db.get_conn() as conn:
            rows = conn.execute(
                """
                SELECT pb.stake, pb.alerted_odds, pb.fill_odds, pb.slippage,
                       br.won, br.push
                FROM placed_bets pb
                LEFT JOIN bet_results br ON br.alert_id = pb.alert_id
                WHERE pb.session_id = ? AND pb.mode = 'paper'
                """,
                (today,),
            ).fetchall()
    except Exception:
        return {}

    bets_placed    = len(rows)
    settled        = [r for r in rows if r['won'] is not None and not r['push']]
    wins           = sum(1 for r in settled if r['won'])
    pnl            = sum(
        r['stake'] * (r['fill_odds'] - 1.0) if r['won'] else -r['stake']
        for r in settled if r['fill_odds']
    )
    slippages      = [r['slippage'] for r in rows if r['slippage'] is not None]
    avg_slippage   = sum(slippages) / len(slippages) if slippages else 0.0

    return {
        'session':      today,
        'bets_placed':  bets_placed,
        'settled':      len(settled),
        'wins':         wins,
        'win_rate':     wins / len(settled) if settled else None,
        'pnl':          pnl,
        'avg_slippage': avg_slippage,
        'suspended':    _session_suspended,
    }
