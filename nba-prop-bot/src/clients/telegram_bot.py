import requests
from src.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from src.utils.retry import retry_with_backoff
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TelegramBotClient:
    """
    Two-tier Telegram client.

    Tier 1 — Instant (time-sensitive):
        bot.send_instant(msg)   ← explicit Tier 1 label
        bot.send_message(msg)   ← backward-compatible alias; same behaviour

        Use for: Arbitrage, Steam Chases, Middling, OTB Injury alerts.

    Tier 2 — Batched (digest):
        Queue via db.queue_pending_alert(...)
        Flush via flush_alerts.flush_pending_alerts(db, bot)

        Use for: Standard +EV Props, Game Markets, SGPs.
        Digests fire at 12 PM, 3 PM, and 6 PM on game days.
    """

    def __init__(self, token: str = TELEGRAM_BOT_TOKEN, chat_id: str = TELEGRAM_CHAT_ID):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    @retry_with_backoff(retries=3, backoff_in_seconds=2)
    def send_instant(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Tier 1: send immediately.  Use for time-sensitive, expiry-critical alerts.
        """
        if not self.token or not self.chat_id:
            logger.warning("Telegram credentials missing. Skipping alert.")
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            return response.json().get("ok", False)
        except requests.exceptions.Timeout:
            logger.warning("Telegram send_instant timed out — skipping.")
            return False

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Backward-compatible alias for send_instant."""
        return self.send_instant(text, parse_mode=parse_mode)
