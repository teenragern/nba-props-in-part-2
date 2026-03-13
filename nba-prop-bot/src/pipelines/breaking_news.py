from src.clients.news_monitor import BreakingNewsMonitor
from src.clients.telegram_bot import TelegramBotClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def check_breaking_news(monitor: BreakingNewsMonitor, bot: TelegramBotClient) -> bool:
    """
    Poll for new breaking injury news.
    Sends a Telegram alert for each new item.
    Returns True if any new items were found (caller should trigger immediate scan).
    """
    items = monitor.get_breaking_injuries()
    if not items:
        return False

    for item in items:
        player_label = item['player_name'] or 'NBA Player'
        msg = (
            f"⚡ <b>BREAKING: {item['title']}</b>\n\n"
            f"{item['summary']}\n\n"
            f"📰 {item['source']} · {item['published_at']}\n"
            f"<i>→ Triggering immediate prop scan...</i>"
        )
        bot.send_message(msg)
        logger.info(f"Breaking news alert sent: {item['title']}")

    return True
