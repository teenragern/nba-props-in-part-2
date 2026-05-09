"""
Redis Pub/Sub EventBus for the NBA props bot.

Publishes and subscribes to structured events so pipelines can react
in near-real-time instead of waiting for the next scheduler tick.

Degrades gracefully to a no-op when REDIS_URL is unset — all existing
scheduler behavior is preserved without any configuration change.

Channels:
    nba:injury:newly_out      published by sync_injuries.py
    nba:steam:move            published by steam.py
    nba:line:snapshot         published by scout_lines.py (future)
    nba:scan:trigger          consumed by run_scheduler.py

Usage:
    from src.events.bus import get_bus, EventBus

    # Publisher
    get_bus().publish(EventBus.INJURY_NEWLY_OUT, {"players": [...], "game_date": "2026-05-05"})

    # Subscriber (returns daemon thread)
    def handler(payload: dict): ...
    get_bus().subscribe(EventBus.INJURY_NEWLY_OUT, handler)
"""

import json
import os
import threading
import time
from typing import Callable, Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class EventBus:
    # Channel name constants
    INJURY_NEWLY_OUT     = "nba:injury:newly_out"
    STEAM_MOVE_DETECTED  = "nba:steam:move"
    LINE_SNAPSHOT_READY  = "nba:line:snapshot"
    SCAN_TRIGGER         = "nba:scan:trigger"
    LIVE_STATE_UPDATE    = "nba:live:state_update"
    LIVE_EXECUTION_QUEUE = "nba:live:execution_queue"

    def __init__(self, redis_url: Optional[str] = None):
        self._redis_url = redis_url
        self._client = None
        self._available = False

        if not redis_url:
            logger.info("EventBus: REDIS_URL not set — running in no-op mode.")
            return

        try:
            import redis
            self._client = redis.from_url(redis_url, decode_responses=True, socket_timeout=5)
            self._client.ping()
            self._available = True
            logger.info(f"EventBus: connected to Redis at {redis_url.split('@')[-1]}")
        except Exception as e:
            logger.warning(f"EventBus: Redis unavailable ({e}) — running in no-op mode.")
            self._client = None
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def publish(self, channel: str, payload: dict) -> bool:
        """
        Publish a JSON payload to a channel.
        Returns True on success, False if Redis is unavailable or publish fails.
        """
        if not self._available or self._client is None:
            return False
        try:
            message = json.dumps({"channel": channel, "payload": payload, "ts": time.time()})
            self._client.publish(channel, message)
            logger.debug(f"EventBus: published to {channel}: {payload}")
            return True
        except Exception as e:
            logger.warning(f"EventBus: publish failed on {channel}: {e}")
            return False

    def subscribe(self, channel: str, callback: Callable[[dict], None]) -> Optional[threading.Thread]:
        """
        Subscribe to a channel. The callback receives the decoded payload dict.
        Runs in a daemon thread — does not block the scheduler loop.
        Returns the thread, or None if Redis is unavailable.
        """
        if not self._available or self._client is None:
            return None

        def _listener():
            try:
                import redis
                # Use a separate connection for blocking pubsub
                r = redis.from_url(self._redis_url, decode_responses=True, socket_timeout=None)
                pubsub = r.pubsub()
                pubsub.subscribe(channel)
                logger.info(f"EventBus: subscribed to {channel}")
                for raw in pubsub.listen():
                    if raw["type"] != "message":
                        continue
                    try:
                        envelope = json.loads(raw["data"])
                        callback(envelope.get("payload", {}))
                    except Exception as e:
                        logger.warning(f"EventBus: handler error on {channel}: {e}")
            except Exception as e:
                logger.error(f"EventBus: subscriber thread crashed on {channel}: {e}")

        t = threading.Thread(target=_listener, daemon=True, name=f"bus-sub-{channel.split(':')[-1]}")
        t.start()
        return t


# ---------------------------------------------------------------------------
# Module-level singleton — import get_bus() rather than EventBus() directly.
# ---------------------------------------------------------------------------
_bus: Optional[EventBus] = None
_bus_lock = threading.Lock()


def get_bus() -> EventBus:
    """Return the module-level EventBus singleton, initializing it on first call."""
    global _bus
    if _bus is None:
        with _bus_lock:
            if _bus is None:
                _bus = EventBus(redis_url=os.getenv("REDIS_URL"))
    return _bus
