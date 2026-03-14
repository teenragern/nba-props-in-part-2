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

import time
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
from src.config import SCAN_INTERVAL_MINUTES, QUOTA_FLOOR, TWITTER_POLL_INTERVAL

logger = get_logger(__name__)
bot    = TelegramBotClient()

# Shared Odds API client — used by sync so quota is updated each morning.
_odds_client: OddsApiClient = OddsApiClient()
# Number of NBA games scheduled for today (set by job_sync).
_today_game_count: int = 0
# Twitter/Nitter monitor — persists across polls to track seen tweet IDs.
_news_monitor: TwitterNitterMonitor = TwitterNitterMonitor()

ET = tz.gettz('America/New_York')


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
    try:
        today_str = datetime.now(ET).strftime('%Y-%m-%d')
        events    = _odds_client.get_events()
        _today_game_count = sum(
            1 for e in events
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

    # Also persist events to the games table.
    notify("Sync", sync_events)


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
    notify("Steam", check_steam)


def job_breaking_news():
    """Poll Nitter/Twitter feeds every TWITTER_POLL_INTERVAL seconds. Trigger scan if news found."""
    if not _has_games():
        return
    found = check_breaking_news(_news_monitor, bot)
    if found and _quota_ok():
        logger.info("Breaking tweet → immediate scan triggered.")
        notify("Scan [BREAKING]", scan_props)


def job_settle():      notify("Settle",          settle_alerts)
def job_stats():       notify("Stats",            generate_analytics)
def job_calibration(): notify("Calibration",      check_calibration)
def job_tune():        notify("Tune",             run_tuning, DatabaseClient())
def job_market_stats():notify("Market Stats",     analyze_market_stats)
def job_exposure():    notify("Exposure",         check_exposure)
def job_timing_analysis(): notify("Timing Analysis", analyze_timing)


# ---------------------------------------------------------------------------
# Scheduler entry point
# ---------------------------------------------------------------------------

def start_scheduler():
    logger.info("Starting NBA Prop Bot scheduler (credit-aware mode)...")

    # --- Daily free jobs (run every day regardless of game schedule) ---
    schedule.every().day.at("04:00").do(job_settle)         # settle previous night
    schedule.every().day.at("09:00").do(job_sync)           # 1 credit; sets game count
    schedule.every().day.at("09:15").do(job_stats)
    schedule.every().day.at("09:30").do(job_calibration)
    schedule.every().day.at("09:45").do(job_tune)
    schedule.every().day.at("10:00").do(job_market_stats)
    schedule.every().day.at("10:15").do(job_timing_analysis)
    schedule.every(6).hours.do(job_exposure)

    # --- Game-day guarded jobs (skip when no games or outside window) ---
    schedule.every(SCAN_INTERVAL_MINUTES).minutes.do(job_scan)  # ~11 credits each
    schedule.every(120).minutes.do(job_clv)                     # ~11 credits each
    schedule.every(20).minutes.do(job_steam)                    # 0 credits (DB only)
    schedule.every(TWITTER_POLL_INTERVAL).seconds.do(job_breaking_news)  # 0 credits (Nitter)

    # Run sync immediately so _today_game_count is set before first scan.
    job_sync()
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
        time.sleep(1)


if __name__ == "__main__":
    start_scheduler()
