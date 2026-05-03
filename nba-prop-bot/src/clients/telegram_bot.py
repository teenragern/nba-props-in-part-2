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

    def broadcast(self, text: str, db=None, parse_mode: str = "HTML") -> int:
        """
        Send `text` to all active subscribers in the `subscribers` table.
        Uses asyncio and aiohttp to send concurrently, strictly rate-limited
        to 25 messages per second to comply with Telegram's limits.
        Falls back to self.chat_id when no DB is provided or the table is empty.
        Returns the number of successful sends.
        """
        chat_ids: list[str] = []
        if db is not None:
            try:
                with db.get_conn() as conn:
                    rows = conn.execute(
                        "SELECT chat_id FROM subscribers WHERE active = 1"
                    ).fetchall()
                chat_ids = [r['chat_id'] for r in rows]
            except Exception as e:
                logger.warning(f"Could not load subscribers for broadcast: {e}")

        if not chat_ids:
            # Fallback: deliver to the configured single chat_id.
            return 1 if self.send_instant(text, parse_mode=parse_mode) else 0

        import asyncio
        import aiohttp

        async def _send_batch(session, url, batch, payload_template):
            tasks = []
            for cid in batch:
                payload = payload_template.copy()
                payload["chat_id"] = cid
                tasks.append(session.post(url, json=payload, timeout=10))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = 0
            for cid, resp in zip(batch, responses):
                if isinstance(resp, Exception):
                    logger.warning(f"Telegram broadcast error for chat_id {cid}: {resp}")
                else:
                    try:
                        data = await resp.json()
                        if data.get("ok"):
                            success_count += 1
                        else:
                            logger.warning(f"Telegram broadcast failed for chat_id {cid}: {data}")
                    except Exception as e:
                        logger.warning(f"Telegram broadcast json error for chat_id {cid}: {e}")
            return success_count

        async def _async_broadcast():
            url = f"{self.base_url}/sendMessage"
            payload_template = {"text": text, "parse_mode": parse_mode}
            total_sent = 0
            batch_size = 25 # Telegram allows max 30 msgs/sec overall
            
            async with aiohttp.ClientSession() as session:
                for i in range(0, len(chat_ids), batch_size):
                    batch = chat_ids[i:i+batch_size]
                    sent = await _send_batch(session, url, batch, payload_template)
                    total_sent += sent
                    # Strict 1 second wait between batches
                    if i + batch_size < len(chat_ids):
                        await asyncio.sleep(1.0)
            return total_sent

        try:
            return asyncio.run(_async_broadcast())
        except Exception as e:
            logger.error(f"Async broadcast failed: {e}")
            return 0

    def start_listener(self, db):
        """
        Starts a background daemon thread that polls Telegram for /start and /stats.
        """
        import threading
        import time
        import requests

        def _poll():
            offset = None
            url = f"{self.base_url}/getUpdates"
            logger.info("Telegram listener daemon started.")
            while True:
                try:
                    params = {"timeout": 30, "allowed_updates": ["message"]}
                    if offset:
                        params["offset"] = offset
                    resp = requests.get(url, params=params, timeout=40)
                    data = resp.json()
                    
                    for update in data.get("result", []):
                        offset = update["update_id"] + 1
                        msg = update.get("message")
                        if not msg:
                            continue
                            
                        text = msg.get("text", "").strip()
                        chat_id = str(msg["chat"]["id"])
                        username = msg["from"].get("username", "Unknown")
                        
                        if text == "/start":
                            with db.get_conn() as conn:
                                conn.execute(
                                    "INSERT OR IGNORE INTO subscribers (chat_id, username, tier, active) VALUES (?, ?, 'basic', 1)",
                                    (chat_id, username)
                                )
                                conn.execute(
                                    "UPDATE subscribers SET active = 1, username = ? WHERE chat_id = ?",
                                    (username, chat_id)
                                )
                            welcome = (
                                "👋 <b>Welcome to NBA Prop Bot!</b>\n\n"
                                "You are now subscribed to receive +EV prop alerts.\n\n"
                                "<i>Disclaimer: All alerts are for informational purposes only. "
                                "Sports betting involves financial risk. Please bet responsibly.</i>"
                            )
                            requests.post(f"{self.base_url}/sendMessage", json={"chat_id": chat_id, "text": welcome, "parse_mode": "HTML"})
                            
                        elif text == "/stats":
                            with db.get_conn() as conn:
                                row = conn.execute(
                                    "SELECT COUNT(*) as n, SUM(CASE WHEN won=1 THEN 1 ELSE 0 END) as w "
                                    "FROM bet_results"
                                ).fetchone()
                            n = row['n'] or 0
                            w = row['w'] or 0
                            win_rate = (w / n * 100) if n > 0 else 0
                            stats_msg = f"📊 <b>Bot Performance</b>\nTotal Settled: {n}\nWin Rate: {win_rate:.1f}%"
                            requests.post(f"{self.base_url}/sendMessage", json={"chat_id": chat_id, "text": stats_msg, "parse_mode": "HTML"})
                            
                except Exception as e:
                    logger.debug(f"Telegram listener error: {e}")
                    time.sleep(5)

        t = threading.Thread(target=_poll, daemon=True, name='telegram_listener')
        t.start()
