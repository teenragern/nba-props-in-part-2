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
from concurrent.futures import ThreadPoolExecutor
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
from src.data.db import get_db_client
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
from src.config import (
    SCAN_INTERVAL_MINUTES, QUOTA_FLOOR, TWITTER_POLL_INTERVAL,
    BDL_SHARP_SCAN_INTERVAL, OFF_SEASON, OFF_SEASON_AFTER_SYNCS,
)
from src.events.bus import get_bus, EventBus
from src.clients.ws_live_feed import start_live_feed
from src.pipelines.exchange_arb import run_exchange_arb
from src.pipelines.game_arb import run_game_arb
from src.pipelines.sync_sportradar import sync_sportradar_schedule, sync_sportradar_lineups, audit_finished_games

_WATCHDOG_TIMEOUT_SEC = int(os.getenv('WATCHDOG_TIMEOUT_SEC', '300'))  # 5 min default
# Minimum seconds between scan submissions to prevent duplicate triggers.
_SCAN_COOLDOWN_SEC = int(os.getenv('SCAN_COOLDOWN_SEC', '60'))

logger = get_logger(__name__)
bot    = TelegramBotClient()

# Thread pool for long-running jobs so they don't block the scheduler loop.
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='sched-worker')

# Shared Odds API client — used by sync so quota is updated each morning.
_odds_client: OddsApiClient = OddsApiClient()
# Number of NBA games scheduled for today (set by job_sync).
_today_game_count: int = 0
# Off-season hibernation state (see _update_hibernation).
_hibernating: bool = (OFF_SEASON == 'true')
# Consecutive successful syncs that returned zero upcoming events.
_empty_sync_streak: int = 0
# Twitter/Nitter monitor — persists across polls to track seen tweet IDs.
_news_monitor: TwitterNitterMonitor = TwitterNitterMonitor()

ET = tz.gettz('America/New_York')

# Monotonic timestamp updated every scheduler tick — used by the watchdog.
_last_tick: float = 0.0
# Monotonic timestamp of the last scan submission — used by cooldown guard.
_last_scan_submit: float = 0.0


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
    if not _odds_client._quota_fetched:
        return True
    return _odds_client.requests_remaining >= QUOTA_FLOOR


def _is_scan_window() -> bool:
    """True between 11 AM and 1 AM Eastern Time (next day).

    Extended past midnight for late playoff games (West Coast tips at
    ~10 PM ET routinely run past midnight).
    """
    h = datetime.now(ET).hour
    return h >= 11 or h < 1


def _has_games() -> bool:
    """True when today has at least one NBA game."""
    return _today_game_count > 0


# ---------------------------------------------------------------------------
# Off-season hibernation
# ---------------------------------------------------------------------------

def _update_hibernation(n_events: int):
    """
    Toggle hibernation from the upcoming-event count of a *successful* sync
    (API failures never advance the streak, so an outage can't hibernate us).

    Auto mode: the Odds API always lists games a few days out in-season —
    even across the All-Star break — so zero upcoming events for
    OFF_SEASON_AFTER_SYNCS consecutive daily syncs means the season is over.
    Wake as soon as events reappear (preseason in October).

    Note: the streak resets on process restart, so a mid-summer redeploy
    takes OFF_SEASON_AFTER_SYNCS days to re-hibernate. Set OFF_SEASON=true
    to force it immediately.
    """
    global _hibernating, _empty_sync_streak

    if OFF_SEASON == 'true':
        _hibernating = True
        return
    if OFF_SEASON == 'false':
        _hibernating = False
        return

    if n_events == 0:
        _empty_sync_streak += 1
        if not _hibernating and _empty_sync_streak >= OFF_SEASON_AFTER_SYNCS:
            _hibernating = True
            logger.info(
                f"Entering off-season hibernation "
                f"({_empty_sync_streak} consecutive syncs with no upcoming events)."
            )
            try:
                bot.send_message(
                    "😴 <b>Off-season hibernation</b>\n\n"
                    f"No upcoming NBA events for {_empty_sync_streak} days — season over.\n"
                    "Pausing scans, injury polling, retrains and daily analytics.\n"
                    "Daily 1-credit sync continues; the bot wakes itself when the "
                    "schedule returns. Weekly check-in every Monday."
                )
            except Exception:
                pass
    else:
        _empty_sync_streak = 0
        if _hibernating:
            _hibernating = False
            logger.info(f"Waking from hibernation: {n_events} upcoming event(s) found.")
            try:
                bot.send_message(
                    "🌅 <b>Season's back!</b>\n\n"
                    f"{n_events} upcoming NBA event(s) on the schedule.\n"
                    "Resuming full scanning, analytics and training cadence."
                )
            except Exception:
                pass


def _hibernate_heartbeat():
    """Weekly Monday check-in + DB backup while hibernating."""
    if datetime.now(ET).weekday() != 0:
        return
    try:
        bot.send_message(
            "😴 <b>Hibernation check-in</b>\n"
            f"No upcoming NBA events ({_empty_sync_streak} day streak).\n"
            f"Credits remaining: {_odds_client.requests_remaining}\n"
            "Will wake automatically when the schedule returns."
        )
    except Exception:
        pass
    submit_job("Backup [hibernate]", backup_db)


def unless_hibernating(func, job_name: str):
    """Wrap a scheduled job so it no-ops during off-season hibernation."""
    def _wrapped():
        if _hibernating:
            logger.debug(f"{job_name} skipped: off-season hibernation.")
            return
        return func()
    return _wrapped


# ---------------------------------------------------------------------------
# Generic notify wrapper  (used for lightweight / once-daily jobs)
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
# Thread-pool submit (used for long-running jobs to avoid blocking the loop)
# ---------------------------------------------------------------------------

def submit_job(job_name: str, func, *args):
    """Submit func to the thread pool so it doesn't block the scheduler loop."""
    def _run():
        logger.info(f"[pool] Starting {job_name}")
        try:
            func(*args)
            logger.info(f"[pool] Finished {job_name}")
        except Exception as e:
            logger.error(f"[pool] {job_name} failed: {e}")
            try:
                bot.send_message(f"❌ <b>Job failed:</b> {job_name}\n\nError: {e}")
            except Exception:
                pass
    return _pool.submit(_run)


def _maybe_submit_scan(label: str) -> bool:
    """
    Submit a scan job with a cooldown guard to prevent duplicate triggers
    (e.g., Redis event fires at the same time as the direct-call path).
    Returns True if submitted, False if cooldown is active.
    """
    global _last_scan_submit
    now = time.monotonic()
    if now - _last_scan_submit < _SCAN_COOLDOWN_SEC:
        remaining = int(_SCAN_COOLDOWN_SEC - (now - _last_scan_submit))
        logger.info(f"Scan [{label}] skipped: cooldown active ({remaining}s remaining).")
        return False
    _last_scan_submit = now
    submit_job(f"Scan [{label}]", scan_props)
    return True


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
        _update_hibernation(len(fetched_events))
        if _hibernating:
            logger.info("Sync: off-season hibernation active — daily jobs paused.")
            _hibernate_heartbeat()
            return
        if _today_game_count > 0:
            bot.send_message(
                f"📅 <b>Today:</b> {_today_game_count} NBA game(s). Scanning active.\n"
                f"Credits remaining: {_odds_client.requests_remaining}"
            )
        else:
            bot.send_message("🏀 No NBA games today. Scans suspended to save credits.")
    except Exception as e:
        logger.error(f"Sync quota check failed: {e}")

    if _hibernating:
        return
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
               f" — relying on Sportradar fallback.")
        logger.warning(msg)
        # We no longer return early here, so the Sportradar fallback can execute.

    def _run_and_summarise():
        scan_props()
        try:
            db = get_db_client()
            with db.get_conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM alerts_sent WHERE timestamp >= NOW() - INTERVAL '12 hours'"
                ).fetchone()
            n_today = int(row['n'] or 0) if row else 0
            summary_msg = (
                f"✅ <b>Scan complete</b> — {n_today} edge(s) found today.\n"
                f"Next scan in ~{SCAN_INTERVAL_MINUTES} min."
            )
            bot.send_message(summary_msg)
        except Exception as e:
            logger.debug(f"Scan summary message failed: {e}")

    # Use the cooldown logic from _maybe_submit_scan but with our custom wrapper
    global _last_scan_submit
    now = time.monotonic()
    if now - _last_scan_submit < _SCAN_COOLDOWN_SEC:
        remaining = int(_SCAN_COOLDOWN_SEC - (now - _last_scan_submit))
        logger.info(f"Scan [SCHEDULED] skipped: cooldown active ({remaining}s remaining).")
        return
    _last_scan_submit = now
    submit_job("Scan [SCHEDULED]", _run_and_summarise)


def job_clv():
    if not _has_games():
        return
    if not _quota_ok():
        logger.warning("CLV update: Odds API quota low — relying on Sportradar fallback.")
    submit_job("Update CLV", update_clv_lines)


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
        _maybe_submit_scan("BREAKING")


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
        logger.warning("Scout: Odds API quota low — relying on Sportradar fallback.")
    try:
        result = scout_lines(_odds_client)
        if result['records_written'] > 0:
            # Run steam check right now — don't wait up to 20 min for it
            submit_job("Steam [post-scout]", check_steam, bot)
    except Exception as e:
        logger.error(f"Scout lines failed: {e}")


def job_morning_briefing():
    """
    Broadcast a daily 'bot is live' message to all subscribers.
    Runs at 09:05 ET after job_sync() has populated _today_game_count.
    Admin-only when no games today so subscribers don't get noise.
    """
    db = get_db_client()
    if _today_game_count > 0:
        msg = (
            f"📅 <b>NBA Prop Scanner — Good morning!</b>\n\n"
            f"{_today_game_count} game(s) on the slate today.\n"
            f"Scanning props across all books. Edges will be sent instantly when found.\n\n"
            f"<i>Digests at 12 PM, 3 PM, and 6 PM ET.</i>"
        )
        try:
            bot.broadcast(msg, db=db)
        except Exception as e:
            logger.warning(f"Morning briefing broadcast failed: {e}")
    else:
        # No games — only notify admin, not subscribers.
        try:
            bot.send_message("🏀 No NBA games today. Prop scanner on standby.")
        except Exception:
            pass


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
        # When Redis is available the subscription in start_scheduler() handles
        # the scan trigger.  Fall back to a direct submit when Redis is offline.
        if not get_bus().is_available() and _is_scan_window() and _quota_ok():
            logger.info(f"Newly OUT ({len(newly_out)}) → immediate scan triggered (direct path).")
            _maybe_submit_scan("LATE-SCRATCH")


def job_flush_alerts():
    """Send Tier-2 digest (Props / Game Markets / SGPs)."""
    if not _has_games():
        logger.info("Digest flush skipped: no NBA games today.")
        return
    flush_pending_alerts(get_db_client(), bot)


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
    db = get_db_client()
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
def job_tune():          notify("Tune",             run_tuning, get_db_client())
def job_market_stats():  notify("Market Stats",     analyze_market_stats)
def job_exposure():      notify("Exposure",         check_exposure)
def job_timing_analysis(): notify("Timing Analysis", analyze_timing)
def job_train_ml():      notify("Train ML",                  train_ml_models)
def job_train_clv_ml():  notify("Train ML [CLV Feedback]",   train_ml_models_clv_feedback)
def job_train_calibration():
    notify("Train Calibration", train_isotonic_calibration)
    ok = reload_calibration_model()
    logger.info(f"Calibration model hot-reloaded: {'OK' if ok else 'no model on disk'}")

def job_exchange_arb():
    """Compare model probs vs Kalshi exchange prices. No-op when KALSHI_API_KEY unset."""
    if not _has_games():
        return
    submit_job("Exchange Arb", run_exchange_arb)


def job_sr_lineups():
    """Sync official lineups from Sportradar (once every 30m)."""
    if not _has_games() or not _is_scan_window():
        return
    submit_job("Sportradar Lineups", sync_sportradar_lineups)


def job_sr_audit():
    """Audit finished games via Sportradar boxscores/PBP."""
    notify("Sportradar Audit", audit_finished_games)


def job_game_arb():
    """Compare model win probs vs Kalshi game markets. No-op when KALSHI_API_KEY unset."""
    if not _has_games():
        return
    submit_job("Game Arb", run_game_arb)


# ---------------------------------------------------------------------------
# Scheduler entry point
# ---------------------------------------------------------------------------

def start_scheduler():
    global _last_tick
    logger.info("Starting NBA Prop Bot scheduler (credit-aware mode)...")
    _start_watchdog()
    bot.start_listener(db=get_db_client())

    # --- Phase 3A: WebSocket live feed (no-op when LIVE_FEED_WS_URL unset) ---
    start_live_feed()

    # --- Redis EventBus subscriptions (no-op when REDIS_URL is unset) ---
    bus = get_bus()
    if bus.is_available():
        def _on_injury_event(payload: dict):
            players = payload.get("players", [])
            if players and _has_games() and _is_scan_window() and _quota_ok():
                logger.info(f"Redis injury event → scan triggered. Players: {players}")
                _maybe_submit_scan("REDIS-INJURY")

        bus.subscribe(EventBus.INJURY_NEWLY_OUT, _on_injury_event)
        logger.info("Redis EventBus subscriptions active.")

    # --- Daily free jobs (run every day in-season; paused during hibernation,
    #     when there is no new data to settle, analyze or train on) ---
    def daily(t, func, name):
        schedule.every().day.at(t).do(unless_hibernating(func, name))

    daily("01:00", job_train_calibration, "Train Calibration")  # nightly calibration training
    daily("03:00", job_backup_db,         "Backup DB")          # daily DB backup (weekly while hibernating)
    daily("04:00", job_settle,            "Settle")             # settle previous night
    daily("04:15", job_sr_audit,          "Sportradar Audit")   # post-game audit
    daily("04:30", job_execution_summary, "Execution Summary")  # paper P&L recap
    daily("05:00", job_drift_monitor,     "Drift Monitor")      # drift check after settlement
    schedule.every().day.at("09:00").do(job_sync)               # 1 credit; sets game count + hibernation
    daily("09:00", sync_sportradar_schedule, "Sportradar Schedule")
    daily("09:05", job_morning_briefing,  "Morning Briefing")   # subscriber daily briefing
    daily("09:15", job_stats,             "Stats")
    daily("09:30", job_calibration,       "Calibration")
    daily("09:45", job_tune,              "Tune")
    daily("10:00", job_market_stats,      "Market Stats")
    daily("10:15", job_timing_analysis,   "Timing Analysis")
    schedule.every(6).hours.do(unless_hibernating(job_exposure, "Exposure"))
    schedule.every().sunday.at("03:00").do(unless_hibernating(job_train_ml, "Train ML"))  # weekly retrain
    schedule.every(30).days.at("04:00").do(unless_hibernating(job_train_clv_ml, "Train ML [CLV]"))

    # --- Game-day guarded jobs (skip when no games or outside window) ---
    schedule.every(SCAN_INTERVAL_MINUTES).minutes.do(job_scan)  # ~11 credits each
    schedule.every(120).minutes.do(job_clv)                     # ~11 credits each
    schedule.every(20).minutes.do(job_steam)                    # 0 credits (DB only)
    schedule.every(30).minutes.do(job_exchange_arb)             # 0 credits (Kalshi API, no-op when key unset)
    schedule.every(5).minutes.do(job_game_arb)                  # 0 credits (Kalshi game markets, no-op when key unset)
    schedule.every(TWITTER_POLL_INTERVAL).seconds.do(job_breaking_news)  # 0 credits (Nitter)
    # Injury feed: free (BDL/scrape). Run frequently — late scratches move minutes hard.
    schedule.every(15).minutes.do(unless_hibernating(job_sync_injuries, "Sync Injuries"))
    # Sportradar Lineup Verification (official starters)
    schedule.every(30).minutes.do(job_sr_lineups)
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
    if not _hibernating:
        # Prime injury cache before first scan so projections see fresh status.
        job_sync_injuries()
        # Attempt an immediate scan if there are games today (non-blocking).
        if _has_games() and _is_scan_window() and _quota_ok():
            _maybe_submit_scan("STARTUP")

    logger.info(
        f"Scheduler live. "
        f"Scan every {SCAN_INTERVAL_MINUTES}min | "
        f"Quota floor: {QUOTA_FLOOR} credits | "
        f"Twitter poll: every {TWITTER_POLL_INTERVAL}s | "
        f"Active window: 11am–11pm ET on game days only | "
        f"Off-season mode: {OFF_SEASON}"
        + (" (HIBERNATING)" if _hibernating else "")
    )

    while True:
        schedule.run_pending()
        _last_tick = time.monotonic()
        time.sleep(1)


if __name__ == "__main__":
    start_scheduler()
