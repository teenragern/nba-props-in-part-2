"""
WebSocket Live Feed client — Phase 3A.

Connects to a push-based sports data WebSocket (Sportradar, SportsFeed, etc.),
parses incoming events, and publishes them to the Redis EventBus so downstream
consumers react in real time rather than waiting for the next scheduler tick.

Published channels:
    nba:game:score_update    {"game_id", "home_score", "away_score", "period", "clock"}
    nba:game:player_event    {"game_id", "player_id", "player_name", "event_type", "value"}

Features:
    - Exponential-backoff auto-reconnect (cap 120 s)
    - Runs as a daemon thread — does not block the scheduler
    - No-op when LIVE_FEED_WS_URL is unset (or Redis is unavailable)

Environment:
    LIVE_FEED_WS_URL    wss://...   WebSocket endpoint (required to enable)
    LIVE_FEED_API_KEY   ...         Injected as ?api_key= query param if set

Usage:
    from src.clients.ws_live_feed import start_live_feed
    start_live_feed()   # call once at scheduler startup; returns daemon thread or None
"""

import json
import os
import threading
import time
from typing import Optional

from src.events.bus import EventBus, get_bus
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Channel names published by this client
CHANNEL_SCORE_UPDATE  = "nba:game:score_update"
CHANNEL_PLAYER_EVENT  = "nba:game:player_event"

_RECONNECT_BASE  = 2      # seconds before first retry
_RECONNECT_CAP   = 120    # maximum back-off ceiling (seconds)
_RECONNECT_MULT  = 2.0    # exponential multiplier


def _build_ws_url() -> Optional[str]:
    base = os.getenv("LIVE_FEED_WS_URL", "").strip()
    if not base:
        return None
    api_key = os.getenv("LIVE_FEED_API_KEY", "").strip()
    if api_key:
        sep = "&" if "?" in base else "?"
        base = f"{base}{sep}api_key={api_key}"
    return base


def _parse_message(raw: str) -> Optional[dict]:
    """
    Parse a raw WebSocket message string into a structured event dict.

    Expected provider JSON shapes (normalised to a common envelope):
        Score update:  {"type": "score", "game_id": "...", "home": N, "away": N,
                        "period": N, "clock": "MM:SS"}
        Player event:  {"type": "player_event", "game_id": "...", "player_id": "...",
                        "player_name": "...", "event": "points|rebound|assist",
                        "value": N}

    Returns None for unrecognised or unparseable messages.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    msg_type = data.get("type", "")

    if msg_type == "score":
        return {
            "event": CHANNEL_SCORE_UPDATE,
            "payload": {
                "game_id":    data.get("game_id", ""),
                "home_score": data.get("home", 0),
                "away_score": data.get("away", 0),
                "period":     data.get("period", 0),
                "clock":      data.get("clock", ""),
            },
        }

    if msg_type == "player_event":
        return {
            "event": CHANNEL_PLAYER_EVENT,
            "payload": {
                "game_id":     data.get("game_id", ""),
                "player_id":   str(data.get("player_id", "")),
                "player_name": data.get("player_name", ""),
                "event_type":  data.get("event", ""),
                "value":       data.get("value", 0),
            },
        }

    return None


def _run_feed(ws_url: str, bus: EventBus) -> None:
    """
    Main WebSocket loop. Runs forever, reconnecting on any error.
    Intended to be called from a daemon thread.
    """
    import websocket  # lazy import — only required when feed is active

    delay = _RECONNECT_BASE

    while True:
        logger.info(f"LiveFeed: connecting to {ws_url.split('?')[0]} ...")
        ws = None
        try:
            ws = websocket.create_connection(ws_url, timeout=30)
            delay = _RECONNECT_BASE  # reset back-off on successful connect
            logger.info("LiveFeed: connected.")

            while True:
                raw = ws.recv()
                if not raw:
                    continue
                parsed = _parse_message(raw)
                if parsed is None:
                    continue
                bus.publish(parsed["event"], parsed["payload"])

        except Exception as e:
            logger.warning(f"LiveFeed: disconnected ({type(e).__name__}: {e}). "
                           f"Reconnecting in {delay}s ...")
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

        time.sleep(delay)
        delay = min(delay * _RECONNECT_MULT, _RECONNECT_CAP)


def start_live_feed() -> Optional[threading.Thread]:
    """
    Start the WebSocket daemon thread.

    Returns the thread on success, None if disabled (no URL or no Redis).
    Safe to call multiple times — only the first call starts the thread.
    """
    ws_url = _build_ws_url()
    if not ws_url:
        logger.info("LiveFeed: LIVE_FEED_WS_URL not set — disabled.")
        return None

    bus = get_bus()
    if not bus.is_available():
        logger.info("LiveFeed: Redis unavailable — live feed disabled "
                    "(events have nowhere to go).")
        return None

    t = threading.Thread(
        target=_run_feed,
        args=(ws_url, bus),
        daemon=True,
        name="live-feed-ws",
    )
    t.start()
    logger.info("LiveFeed: daemon thread started.")
    return t
