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
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(asyncio.run, _async_broadcast()).result(timeout=120)
            else:
                return asyncio.run(_async_broadcast())
        except Exception as e:
            logger.error(f"Async broadcast failed: {e}")
            return 0

    def start_listener(self, db, copilot=None):
        """
        Starts a background daemon thread that polls Telegram for commands and,
        when the Quant Edge Copilot is enabled, answers free-text questions
        about the current picks.

        Commands:
            /start            subscribe to alerts
            /stats            bot performance summary
            /ask <question>   ask the copilot (any non-command message also works)

        The copilot is NARRATION-ONLY (it explains picks the quant stack already
        produced — it never computes edges or invents stats). Access is gated to
        the owner chat and active subscribers, and rate-limited per chat. Pass
        `copilot` to inject a custom instance (e.g. in tests); otherwise one is
        built lazily from config when COPILOT_ENABLED.
        """
        import threading
        import time
        import requests
        from collections import defaultdict, deque

        from src.config import COPILOT_ENABLED, COPILOT_RATE_MAX, COPILOT_RATE_WINDOW_S

        # Build the copilot lazily; a missing API key simply leaves it disabled.
        if copilot is None and COPILOT_ENABLED:
            try:
                from src.clients.copilot import QuantCopilot
                copilot = QuantCopilot(db)
                logger.info("Quant Edge Copilot enabled for Telegram listener.")
            except Exception as e:
                logger.warning(f"Copilot disabled (init failed): {e}")
                copilot = None

        # In-memory per-chat sliding-window rate limiter for copilot questions.
        _rate: dict = defaultdict(deque)

        def _rate_ok(chat_id: str) -> bool:
            now = time.time()
            dq = _rate[chat_id]
            while dq and now - dq[0] > COPILOT_RATE_WINDOW_S:
                dq.popleft()
            if len(dq) >= COPILOT_RATE_MAX:
                return False
            dq.append(now)
            return True

        def _is_allowed(chat_id: str) -> bool:
            """Owner chat is always allowed; everyone else must be an active subscriber."""
            if self.chat_id and str(chat_id) == str(self.chat_id):
                return True
            try:
                # NB: '?' placeholder matches the rest of this listener (SQLite path).
                with db.get_conn() as conn:
                    row = conn.execute(
                        "SELECT active FROM subscribers WHERE chat_id = ?", (chat_id,)
                    ).fetchone()
                return bool(row and row["active"])
            except Exception:
                return False

        def _send(chat_id, text):
            try:
                requests.post(
                    f"{self.base_url}/sendMessage",
                    json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                    timeout=10,
                )
            except Exception as e:
                logger.debug(f"Telegram send error: {e}")

        def _handle_question(chat_id, question):
            if copilot is None:
                return  # feature disabled — stay silent on free text
            if not _is_allowed(chat_id):
                _send(chat_id, "Send /start to subscribe before using the copilot.")
                return
            if not _rate_ok(str(chat_id)):
                _send(chat_id, "⏳ Too many questions just now — give me a moment and try again.")
                return
            try:
                answer = copilot.answer(question)
            except Exception as e:
                logger.warning(f"Copilot answer failed: {e}")
                answer = "The copilot hit an error. Please try again shortly."
            _send(chat_id, answer)

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
                            copilot_line = (
                                "\n💬 Ask me about any pick — e.g. <i>\"why the edge on the "
                                "best assists prop?\"</i> — or use <code>/ask &lt;question&gt;</code>.\n"
                                if copilot is not None else "\n"
                            )
                            welcome = (
                                "👋 <b>Welcome to NBA Prop Bot!</b>\n\n"
                                "You are now subscribed to receive +EV prop alerts.\n"
                                f"{copilot_line}\n"
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

                        elif text.startswith("/ask"):
                            _handle_question(chat_id, text[len("/ask"):].strip())

                        elif text and not text.startswith("/"):
                            # Any non-command message is treated as a copilot question.
                            _handle_question(chat_id, text)

                except Exception as e:
                    logger.debug(f"Telegram listener error: {e}")
                    time.sleep(5)

        t = threading.Thread(target=_poll, daemon=True, name='telegram_listener')
        t.start()
