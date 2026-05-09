"""
Async bridge utility — Phase 3 live trading.

Bridges the synchronous Redis EventBus (daemon thread + blocking pubsub.listen())
into an asyncio.Queue that async coroutines can await.

Usage:
    loop = asyncio.get_event_loop()
    queue, thread = make_event_bridge(EventBus.LIVE_STATE_UPDATE, loop)

    # In the async event loop:
    payload = await asyncio.wait_for(queue.get(), timeout=30.0)

The daemon thread puts messages onto the queue via loop.call_soon_threadsafe(),
which is the only safe way to schedule work on an asyncio loop from a non-async
thread. If the queue fills up (backpressure), messages are silently dropped with
a warning log rather than blocking the publisher thread.
"""

import asyncio
import threading
from typing import Optional

from src.events.bus import EventBus, get_bus
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def make_event_bridge(
    channel: str,
    loop: asyncio.AbstractEventLoop,
    maxsize: int = 100,
) -> tuple:
    """
    Wire a synchronous EventBus channel to an asyncio.Queue.

    Args:
        channel:  EventBus channel constant, e.g. EventBus.LIVE_STATE_UPDATE.
        loop:     The running event loop — must be obtained via
                  asyncio.get_event_loop() BEFORE any threads are started,
                  from within the async context that will own the queue.
        maxsize:  Maximum queue depth. When full, incoming messages are dropped
                  with a warning (backpressure signal).

    Returns:
        (queue, thread) where:
          queue  — asyncio.Queue safe to await within `loop`.
          thread — the EventBus daemon subscriber thread, or None if Redis is
                   unavailable (the queue will simply never receive items).
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    def _on_message(payload: dict) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, payload)
        except asyncio.QueueFull:
            logger.warning(
                f"async_bridge: queue full on {channel} (maxsize={maxsize}) — dropping message"
            )
        except RuntimeError:
            # Loop may be closing during shutdown — ignore silently.
            pass

    thread = get_bus().subscribe(channel, _on_message)
    if thread is None:
        logger.info(
            f"async_bridge: Redis unavailable — {channel} bridge is a no-op. "
            "The queue will not receive any messages."
        )
    else:
        logger.info(f"async_bridge: subscribed to {channel} (queue maxsize={maxsize})")

    return queue, thread
