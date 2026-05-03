"""
Credit-aware scheduler for the NBA Prop Bot.

Game-day logic:
  - job_sync() runs at 09:00 ET every day (1 Odds API credit).
    It counts today's games and sets _today_game_count.
  - scan_props / update_clv_lines only run on game days (>0 games)
    AND within the 11am–11pm ET window AND quota >= QUOTA_FLOOR.
  - check_steam runs on game days only (0 credits, reads DB).
  - All analytics / settlement / exposure jobs run every day (0 credits).

Configurable via env vars:
  SCAN_INTERVAL_MINUTES  (default 90)  — how often scan fires on game days
  QUOTA_FLOOR            (default 30)  — pause API jobs below this credit level
"""

import os
import time
import threading
import schedule
import dateutil.parser
from datetime import datetime
from dateutil import tz

from src.utils.logging_utils import get_logger
from src.clients.telegram_bot import TelegramBotClient
from src.clients.odds_api import OddsApiClient
from src.pipelines.sync_events import sync_events
from src.pipelines.scan_props import scan_props
from src.pipelines.update_clv import update_clv_lines
from src.pipelines.settle_results import settle_alerts
from src.pipelines.analytics import generate_analytics
from src.pipelines.calibration import check_calibration
from src.pipelines.tune import run_tuning
from src.data.db import DatabaseClient
from src.pipelines.market_stats import analyze_market_stats
from src.pipelines.steam import check_steam
from src.pipelines.exposure import check_exposure
from src.pipelines.timing_analysis import analyze_timing
from src.clients.twitter_monitor import TwitterNitterMonitor
from src.pipelines.breaking_news import check_breaking_news
from src.pipelines.train_ml import train_ml_models, train_ml_models_clv_feedback
from src.pipelines.train_calibration import train_isotonic_calibration
from src.pipelines.flush_alerts import flush_pending_alerts
from src.pipelines.drift_monitor import check_drift
from src.models.edge_ranker import reload_clv_thresholds
from src.models.calibration_model import reload_calibration_model
from src.pipelines.sync_injuries import sync_injuries
from src.pipelines.scout_lines import scout_lines
from src.execution.executor import session_summary
from src.pipelines.backup_db import backup_db
from src.config import SCAN_INTERVAL_MINUTES, QUOTA_FLOOR, TWITTER_POLL_INTERVAL, BDL_SHARP_SCAN_INTERVAL

_WATCHDOG_TIMEOUT_SEC = int(os.getenv('WATCHDOG_TIMEOUT_SEC', '300'))  # 5 min default

logger = get_logger(__name__)
bot    = TelegramBotClient()

# Shared Odds API client — used by sync so quota is updated each morning.
_odds_client: OddsApiClient = OddsApiClient()
# Number of NBA games scheduled for today (set by job_sync).
_today_game_count: int = 0
# Twitter/Nitter monitor — persists across polls to track seen tweet IDs.
_news_monitor: TwitterNitterMonitor = TwitterNitterMonitor()

ET = tz.gettz('America/New_York')

# Monotonic timestamp updated every scheduler tick — used by the watchdog.
_last_tick: float = 0.0


# ---------------------------------------------------------------------------
# In-process watchdog
# ---------------------------------------------------------------------------

def _start_watchdog():
    """
    Daemon thread: fires a Telegram alarm if the main scheduler loop has not
    ticked within WATCHDOG_TIMEOUT_SEC seconds.  Runs every 60 s.
    Railway will restart the process on failure, so the alert is informational
    (tells you the loop hung before Railway caught it).
    """
    def _watch():
        while True:
            time.sleep(60)
            if _last_tick == 0:
                continue  # scheduler not yet started
            stale = time.monotonic() - _last_tick
            if stale > _WATCHDOG_TIMEOUT_SEC:
                msg = (
                    f"⚠️ <b>Watchdog alert</b>: scheduler loop has not ticked "
                    f"in {int(stale)}s (threshold: {_WATCHDOG_TIMEOUT_SEC}s).\n"
                    f"Railway should restart automatically."
                )
                logger.error(f"Watchdog: loop stalled for {int(stale)}s")
                try:
                    bot.send_message(msg)
                except Exception:
                    pass

    t = threading.Thread(target=_watch, daemon=True, name='watchdog')
    t.start()
    logger.info(f"Watchdog started (timeout={_WATCHDOG_TIMEOUT_SEC}s).")


# ---------------------------------------------------------------------------
# Guard helpers
# ---------------------------------------------------------------------------

def _quota_ok() -> bool:
    """True when credits are sufficient (or quota not yet initialised)."""
    remaining = _odds_client.requests_remaining
    return remaining == 0 or remaining >= QUOTA_FLOOR


def _is_scan_window() -> bool:
    """True between 11 AM and 11 PM Eastern Time."""
    return 11 <= datetime.now(ET).hour < 23


def _has_games() -> bool:
    """True when today has at least one NBA game."""
    return _today_game_count > 0


# ---------------------------------------------------------------------------
# Generic notify wrapper
# ---------------------------------------------------------------------------

def notify(job_name, func, *args):
    logger.info(f"Executing scheduled job: {job_name}")
    bot.send_message(f"⏳ <b>Starting Scheduled Job:</b> {job_name}")
    try:
        func(*args)
        bot.send_message(f"✅ <b>Finished Scheduled Job:</b> {job_name}")
    except Exception as e:
        logger.error(f"Scheduled {job_name} failed: {e}")
        bot.send_message(f"❌ <b>Failed Scheduled Job:</b> {job_name}\n\nError: {e}")


# ---------------------------------------------------------------------------
# Individual job functions
# ---------------------------------------------------------------------------

def job_sync():
    """Fetch today's event list (1 credit) and update _today_game_count."""
    global _today_game_count
    fetched_events = []
    try:
        today_str = datetime.now(ET).strftime('%Y-%m-%d')
        fetched_events = _odds_client.get_events()
        _today_game_count = sum(
            1 for e in fetched_events
            if dateutil.parser.isoparse(e['commence_time'])
               .astimezone(ET).strftime('%Y-%m-%d') == today_str
        )
        logger.info(f"Sync: {_today_game_count} NBA game(s) today ({today_str}). "
                    f"Quota remaining: {_odds_client.requests_remaining}")
        if _today_game_count > 0:
            bot.send_message(
                f"📅 <b>Today:</b> {_today_game_count} NBA game(s). Scanning active.\n"
                f"Credits remaining: {_odds_client.requests_remaining}"
            )
        else:
            bot.send_message("🏀 No NBA games today. Scans suspended to save credits.")
    except Exception as e:
        logger.error(f"Sync quota check failed: {e}")

    # Pass the already-fetched events so sync_events doesn't need a second get_events() call.
    notify("Sync", sync_events, fetched_events or None)


def job_scan():
    if not _has_games():
        logger.info("Scan skipped: no NBA games today.")
        return
    if not _is_scan_window():
        logger.info("Scan skipped: outside 11am–11pm ET window.")
        return
    if not _quota_ok():
        msg = (f"⚠️ Odds API quota low "
               f"({_odds_client.requests_remaining} credits remaining, floor={QUOTA_FLOOR})"
               f" — scan suspended.")
        logger.warning(msg)
        bot.send_message(msg)
        return
    notify("Scan", scan_props)


def job_clv():
    if not _has_games():
        return
    if not _quota_ok():
        logger.warning("CLV update skipped: quota low.")
        return
    notify("Update CLV", update_clv_lines)


def job_steam():
    if not _has_games():
        return
    notify("Steam", check_steam, bot)


def job_breaking_news():
    """Poll Nitter/Twitter feeds every TWITTER_POLL_INTERVAL seconds. Trigger scan if news found."""
    if not _has_games():
        return
    found = check_breaking_news(_news_monitor, bot)
    if found and _quota_ok():
        logger.info("Breaking tweet → immediate scan triggered.")
        notify("Scan [BREAKING]", scan_props)


def job_scout_lines():
    """
    Lightweight line scout: fetches current Pinnacle/rec-book prop lines for
    games within 4h of tip and writes snapshots to line_history (1 credit/game).
    Immediately runs steam detection on the fresh data — no need to wait for
    the next scheduled steam job.
    """
    if not _has_games():
        return
    if not _is_scan_window():
        return
    if not _quota_ok():
        logger.warning("Scout skipped: quota low.")
        return
    try:
        result = scout_lines(_odds_client)
        if result['records_written'] > 0:
            # Run steam check right now — don't wait up to 20 min for it
            notify("Steam [post-scout]", check_steam, bot)
    except Exception as e:
        logger.error(f"Scout lines failed: {e}")


def job_sync_injuries():
    """
    Refresh injury feed (BDL + nba_api + Rotowire + CBS). Cheap (no Odds API
    credits). On game days, a *newly* OUT player triggers an immediate scan
    so late scratches surface before the next 90-min cycle.
    """
    try:
        summary = sync_injuries()
    except Exception as e:
        logger.error(f"Injury sync failed: {e}")
        return

    newly_out = summary.get('newly_out') or []
    if newly_out and _has_games():
        msg = "🚨 <b>Late scratch(es) detected:</b>\n  • " + "\n  • ".join(newly_out[:10])
        try:
            bot.send_message(msg)
        except Exception:
            pass
        if _is_scan_window() and _quota_ok():
            logger.info(f"Newly OUT ({len(newly_out)}) → immediate scan triggered.")
            notify("Scan [LATE-SCRATCH]", scan_props)


def job_flush_alerts():
    """Send Tier-2 digest (Props / Game Markets / SGPs)."""
    if not _has_games():
        logger.info("Digest flush skipped: no NBA games today.")
        return
    flush_pending_alerts(DatabaseClient(), bot)


def job_drift_monitor():
    """
    Run after daily settlement.  Computes rolling Brier scores, writes
    model_health snapshots, and fires a Telegram alarm if the model has
    degraded beyond 2σ.  On ALARM, immediately retrains calibration and
    ML models (with their own regression guards — fail closed if worse).
    """
    try:
        result = check_drift(bot)
    except Exception as e:
        logger.error(f"Drift monitor failed: {e}")
        return

    # Always refresh CLV-adaptive edge floors after settlement data is updated
    try:
        reload_clv_thresholds()
        logger.info("CLV-adaptive edge floors reloaded.")
    except Exception as e:
        logger.error(f"CLV floor reload failed: {e}")

    if result.get('alarm'):
        logger.warning("Drift ALARM — triggering immediate retrain.")
        notify("Train Calibration [drift alarm]", train_isotonic_calibration)
        reload_calibration_model()
        notify("Train ML [drift alarm]", train_ml_models)


def job_backup_db():
    """Daily hot backup of props.db → backups/ (+ optional S3 upload)."""
    try:
        result = backup_db()
        props = result.get('props', {})
        if 'error' in props:
            bot.send_message(f"❌ DB backup failed: {props['error']}")
        else:
            logger.info(
                f"DB backup OK: {props.get('path')} ({props.get('size_mb', 0):.1f} MB)"
            )
    except Exception as e:
        logger.error(f"DB backup job failed: {e}")


def job_execution_summary():
    """Send paper-trade session P&L recap after overnight settlement."""
    db = DatabaseClient()
    stats = session_summary(db)
    if not stats:
        return
    pnl      = stats.get('pnl', 0.0)
    placed   = stats.get('bets_placed', 0)
    settled  = stats.get('settled', 0)
    wins     = stats.get('wins', 0)
    win_rate = stats.get('win_rate')
    slip     = stats.get('avg_slippage', 0.0)
    susp     = stats.get('suspended', False)
    wr_str   = f"{win_rate:.0%}" if win_rate is not None else "—"
    msg = (
        f"📊 <b>Paper Session Recap ({stats.get('session', '—')})</b>\n\n"
        f"Bets placed:   {placed}  |  Settled: {settled}\n"
        f"Win rate:      {wr_str}  ({wins}/{settled})\n"
        f"P&L:           <b>${pnl:+.2f}</b>\n"
        f"Avg slippage:  {slip:+.4f} dec\n"
        + ("⛔ Execution suspended (session loss limit hit)\n" if susp else "")
    )
    try:
        bot.send_message(msg)
    except Exception:
        pass
    logger.info(f"Execution summary: P&L={pnl:.2f}, placed={placed}, win_rate={wr_str}")


def job_settle():        notify("Settle",          settle_alerts)
def job_stats():         notify("Stats",            generate_analytics)
def job_calibration():   notify("Calibration",      check_calibration)
def job_tune():          notify("Tune",             run_tuning, DatabaseClient())
def job_market_stats():  notify("Market Stats",     analyze_market_stats)
def job_exposure():      notify("Exposure",         check_exposure)
def job_timing_analysis(): notify("Timing Analysis", analyze_timing)
def job_train_ml():      notify("Train ML",                  train_ml_models)
def job_train_clv_ml():  notify("Train ML [CLV Feedback]",   train_ml_models_clv_feedback)
def job_train_calibration():
    notify("Train Calibration", train_isotonic_calibration)
    ok = reload_calibration_model()
    logger.info(f"Calibration model hot-reloaded: {'OK' if ok else 'no model on disk'}")


# ---------------------------------------------------------------------------
# Scheduler entry point
# ---------------------------------------------------------------------------

def start_scheduler():
    global _last_tick
    logger.info("Starting NBA Prop Bot scheduler (credit-aware mode)...")
    _start_watchdog()

    # --- Daily free jobs (run every day regardless of game schedule) ---
    schedule.every().day.at("01:00").do(job_train_calibration) # Nightly calibration model training
    schedule.every().day.at("03:00").do(job_backup_db)       # daily DB backup
    schedule.every().day.at("04:00").do(job_settle)         # settle previous night
    schedule.every().day.at("04:30").do(job_execution_summary)  # paper P&L recap
    schedule.every().day.at("05:00").do(job_drift_monitor)  # drift check after settlement
    schedule.every().day.at("09:00").do(job_sync)           # 1 credit; sets game count
    schedule.every().day.at("09:15").do(job_stats)
    schedule.every().day.at("09:30").do(job_calibration)
    schedule.every().day.at("09:45").do(job_tune)
    schedule.every().day.at("10:00").do(job_market_stats)
    schedule.every().day.at("10:15").do(job_timing_analysis)
    schedule.every(6).hours.do(job_exposure)
    schedule.every().sunday.at("03:00").do(job_train_ml)    # weekly ML retrain (off-peak)
    schedule.every(30).days.at("04:00").do(job_train_clv_ml)  # monthly CLV-weighted retrain

    # --- Game-day guarded jobs (skip when no games or outside window) ---
    schedule.every(SCAN_INTERVAL_MINUTES).minutes.do(job_scan)  # ~11 credits each
    schedule.every(120).minutes.do(job_clv)                     # ~11 credits each
    schedule.every(20).minutes.do(job_steam)                    # 0 credits (DB only)
    schedule.every(TWITTER_POLL_INTERVAL).seconds.do(job_breaking_news)  # 0 credits (Nitter)
    # Injury feed: free (BDL/scrape). Run frequently — late scratches move minutes hard.
    schedule.every(15).minutes.do(job_sync_injuries)
    # Fast line scout: 1 credit/game, fetches sharp lines between full scans.
    # Feeds steam detection with data every BDL_SHARP_SCAN_INTERVAL (default 30 min)
    # rather than every 90 min scan cycle.
    schedule.every(BDL_SHARP_SCAN_INTERVAL).minutes.do(job_scout_lines)

    # --- Tier-2 digest flushes (Props / Game Markets / SGPs) ---
    schedule.every().day.at("12:00").do(job_flush_alerts)
    schedule.every().day.at("15:00").do(job_flush_alerts)
    schedule.every().day.at("18:00").do(job_flush_alerts)

    # Run sync immediately so _today_game_count is set before first scan.
    job_sync()
    # Prime injury cache before first scan so projections see fresh status.
    job_sync_injuries()
    # Attempt an immediate scan if there are games today.
    job_scan()

    logger.info(
        f"Scheduler live. "
        f"Scan every {SCAN_INTERVAL_MINUTES}min | "
        f"Quota floor: {QUOTA_FLOOR} credits | "
        f"Twitter poll: every {TWITTER_POLL_INTERVAL}s | "
        f"Active window: 11am–11pm ET on game days only."
    )

    while True:
        schedule.run_pending()
        _last_tick = time.monotonic()
        time.sleep(1)


if __name__ == "__main__":
    start_scheduler()
