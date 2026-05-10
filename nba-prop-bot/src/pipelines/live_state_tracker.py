"""
Live game state tracker — Phase 3 live trading.

Background asyncio service that polls the BallDontLie API every few seconds
while NBA games are in progress, extracts live game state, and publishes it to
the Redis EventBus so the live engine can react in near-real time.

Published channel:
    nba:live:state_update  (EventBus.LIVE_STATE_UPDATE)

Payload fields:
    game_id         str    BDL game ID (as string)
    home_team       str    Full team name, e.g. "Boston Celtics"
    away_team       str    Full team name, e.g. "New York Knicks"
    home_score      int
    away_score      int
    period          int    1-4 (regulation), 5+ (OT)
    clock           str    "MM:SS" remaining in the current period, or ""
    minutes_elapsed float  Total game minutes elapsed (0–48+)
    status          str    BDL status string, e.g. "in_progress"

Environment:
    BDL_API_KEY                 BallDontLie GOAT-tier key (required)
    REDIS_URL                   Redis connection URL (required for publishing)
    LIVE_POLL_INTERVAL_SECONDS  Polling interval in seconds (default: 5)
    LIVE_FEED_WS_URL            When set, prefer WebSocket feed over BDL polling

Run:
    python -m src.pipelines.live_state_tracker
"""

import asyncio
import dataclasses
import os
import signal
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.clients.bdl_client import BDLClient
from src.clients.ws_live_feed import start_live_feed, CHANNEL_SCORE_UPDATE, CHANNEL_PLAYER_EVENT
from src.events.bus import EventBus, get_bus
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_POLL_INTERVAL: int = int(os.getenv("LIVE_POLL_INTERVAL_SECONDS", "5"))
_USE_WEBSOCKET: bool = bool(os.getenv("LIVE_FEED_WS_URL", "").strip())

# BDL status values that indicate an in-progress game.
_LIVE_STATUSES = frozenset({"in_progress", "live", "halftime"})

# game_id → {home_team, away_team} — populated by a single BDL REST call
_game_meta_cache: Dict[str, dict] = {}

# game_id → team name that last had a possession-implying event
_last_possession: Dict[str, str] = {}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class LiveGameState:
    game_id:         str
    home_team:       str
    away_team:       str
    home_score:      int
    away_score:      int
    period:          int
    clock:           str    # "MM:SS" remaining in period, or ""
    minutes_elapsed: float  # total game minutes elapsed
    status:          str
    has_possession:  Optional[str] = None  # team name with current possession, or None


# ── Clock / time helpers ──────────────────────────────────────────────────────

def _parse_clock_remaining(clock_str: str) -> float:
    """
    Parse BDL's 'time' field ("MM:SS") into minutes remaining in the period.

    BDL sends "" or None at halftime / end of period — we treat those as 0.0.
    """
    if not clock_str:
        return 0.0
    try:
        parts = clock_str.split(":")
        return int(parts[0]) + int(parts[1]) / 60.0
    except (IndexError, ValueError):
        return 0.0


def _minutes_elapsed(period: int, clock_remaining: float) -> float:
    """
    Convert (period, clock_remaining) → total game minutes elapsed.

    NBA periods are 12 minutes each.  OT periods are 5 minutes, but for
    win-probability purposes we still treat them as having 12-min sigma
    (OT is very high-variance — the model will quickly converge).
    """
    period_length = 12.0
    return (period - 1) * period_length + (period_length - clock_remaining)


# ── State extraction ──────────────────────────────────────────────────────────

def _extract_state(game: dict) -> Optional[LiveGameState]:
    """
    Map a BDL game dict to a LiveGameState.

    Returns None when the game is not currently live.

    BDL game fields used:
        id, status, period, time,
        home_team_score, visitor_team_score,
        home_team.full_name, visitor_team.full_name
    """
    status = (game.get("status") or "").lower().strip()
    if status not in _LIVE_STATUSES:
        return None

    period = int(game.get("period") or 0)
    if period < 1:
        return None

    clock_str      = (game.get("time") or "").strip()
    clock_remaining = _parse_clock_remaining(clock_str)
    elapsed        = _minutes_elapsed(period, clock_remaining)

    home_team_obj  = game.get("home_team") or {}
    away_team_obj  = game.get("visitor_team") or {}

    home_name = (
        home_team_obj.get("full_name")
        or home_team_obj.get("name")
        or "Unknown"
    )
    away_name = (
        away_team_obj.get("full_name")
        or away_team_obj.get("name")
        or "Unknown"
    )

    return LiveGameState(
        game_id         = str(game.get("id", "")),
        home_team       = home_name,
        away_team       = away_name,
        home_score      = int(game.get("home_team_score") or 0),
        away_score      = int(game.get("visitor_team_score") or 0),
        period          = period,
        clock           = clock_str,
        minutes_elapsed = elapsed,
        status          = status,
    )


# ── Blocking BDL fetch (called via asyncio.to_thread) ────────────────────────

def _fetch_live_games(bdl: BDLClient, today: str) -> List[LiveGameState]:
    """
    Synchronous.  Fetch today's games from BDL and filter to live ones.
    Wrapped by asyncio.to_thread() in the async poll loop.
    """
    try:
        games = bdl.get_games_by_date(today)
    except Exception as e:
        logger.warning(f"live_state_tracker: BDL fetch failed: {e}")
        return []

    states = []
    for g in games:
        state = _extract_state(g)
        if state is not None:
            states.append(state)
    return states


def _publish_state(bus: EventBus, state: LiveGameState) -> bool:
    """Synchronous publish — called via asyncio.to_thread()."""
    return bus.publish(EventBus.LIVE_STATE_UPDATE, dataclasses.asdict(state))


# ── WebSocket consumer mode ───────────────────────────────────────────────────

def _build_game_meta_cache(bdl: BDLClient, today: str) -> None:
    """Fetch today's games from BDL and populate the team-name lookup cache."""
    try:
        games = bdl.get_games_by_date(today)
        for g in games:
            gid = str(g.get("id", ""))
            home_obj = g.get("home_team") or {}
            away_obj = g.get("visitor_team") or {}
            _game_meta_cache[gid] = {
                "home_team": home_obj.get("full_name") or home_obj.get("name") or "Unknown",
                "away_team": away_obj.get("full_name") or away_obj.get("name") or "Unknown",
            }
        logger.info(f"live_state_tracker: cached metadata for {len(_game_meta_cache)} games")
    except Exception as e:
        logger.warning(f"live_state_tracker: meta cache build failed: {e}")


def _convert_ws_score_update(payload: dict) -> Optional[LiveGameState]:
    """Convert a nba:game:score_update payload into a LiveGameState."""
    game_id = str(payload.get("game_id", ""))
    meta = _game_meta_cache.get(game_id)
    if meta is None:
        return None

    period = int(payload.get("period", 0))
    clock_str = payload.get("clock", "")
    clock_remaining = _parse_clock_remaining(clock_str)
    elapsed = _minutes_elapsed(period, clock_remaining)

    return LiveGameState(
        game_id=game_id,
        home_team=meta["home_team"],
        away_team=meta["away_team"],
        home_score=int(payload.get("home_score", 0)),
        away_score=int(payload.get("away_score", 0)),
        period=period,
        clock=clock_str,
        minutes_elapsed=elapsed,
        status="in_progress",
        has_possession=_last_possession.get(game_id),
    )


async def ws_consumer_loop(
    bdl: BDLClient,
    bus: EventBus,
    shutdown: asyncio.Event,
) -> None:
    """
    Consume nba:game:score_update from Redis (published by ws_live_feed daemon),
    convert to LiveGameState, and republish to nba:live:state_update.
    Falls back to poll_loop if subscription fails or no messages arrive.
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    def _on_score_update(payload: dict) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, payload)
        except asyncio.QueueFull:
            logger.warning("live_state_tracker: WS queue full — dropping update")
        except RuntimeError:
            pass

    def _on_player_event(payload: dict) -> None:
        game_id = str(payload.get("game_id", ""))
        team = payload.get("team", "")
        if game_id and team:
            _last_possession[game_id] = team

    sub_thread = bus.subscribe(CHANNEL_SCORE_UPDATE, _on_score_update)
    if sub_thread is None:
        logger.warning("live_state_tracker: cannot subscribe to WS channel — falling back to polling")
        return

    bus.subscribe(CHANNEL_PLAYER_EVENT, _on_player_event)

    # Pre-populate game metadata
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await asyncio.to_thread(_build_game_meta_cache, bdl, today)

    # Ensure the WS feed daemon is running
    start_live_feed()

    logger.info("live_state_tracker: WebSocket consumer mode active")

    while not shutdown.is_set():
        try:
            payload = await asyncio.wait_for(queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            # Refresh metadata cache for games starting after boot
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if not _game_meta_cache:
                await asyncio.to_thread(_build_game_meta_cache, bdl, today)
            continue
        except asyncio.CancelledError:
            break

        state = _convert_ws_score_update(payload)
        if state is None:
            # Try refreshing cache for unknown game_id
            game_id = str(payload.get("game_id", ""))
            if game_id and game_id not in _game_meta_cache:
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                await asyncio.to_thread(_build_game_meta_cache, bdl, today)
                state = _convert_ws_score_update(payload)
            if state is None:
                continue

        try:
            await asyncio.to_thread(_publish_state, bus, state)
        except Exception as e:
            logger.warning(f"live_state_tracker: WS publish failed for game {state.game_id}: {e}")


# ── Async event loop ──────────────────────────────────────────────────────────

async def poll_loop(
    bdl: BDLClient,
    bus: EventBus,
    shutdown: asyncio.Event,
) -> None:
    """
    Core polling loop.  Runs until shutdown is set.

    Each iteration:
      1. Fetch live game states from BDL (blocking call in thread).
      2. Publish each state to Redis EventBus (blocking call in thread).
      3. Sleep for LIVE_POLL_INTERVAL_SECONDS.
    """
    logger.info(
        f"live_state_tracker: starting poll loop "
        f"(interval={_POLL_INTERVAL}s, bus={bus.is_available()})"
    )

    while not shutdown.is_set():
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        states: List[LiveGameState] = await asyncio.to_thread(
            _fetch_live_games, bdl, today
        )

        if states:
            logger.info(
                f"live_state_tracker: {len(states)} live game(s) — "
                + ", ".join(
                    f"{s.home_team} {s.home_score}–{s.away_score} {s.away_team} "
                    f"(Q{s.period} {s.clock})"
                    for s in states
                )
            )
        else:
            logger.debug("live_state_tracker: no live games this tick")

        for state in states:
            try:
                await asyncio.to_thread(_publish_state, bus, state)
            except Exception as e:
                logger.warning(
                    f"live_state_tracker: publish failed for game {state.game_id}: {e}"
                )

        try:
            await asyncio.wait_for(shutdown.wait(), timeout=_POLL_INTERVAL)
        except asyncio.TimeoutError:
            pass   # normal — loop again


async def main() -> None:
    """
    Entry point.  Initialises clients, installs signal handlers, and runs
    the poll loop until SIGINT or SIGTERM is received.
    """
    if not os.getenv("BDL_API_KEY"):
        logger.error("live_state_tracker: BDL_API_KEY not set — exiting.")
        return

    bdl = BDLClient()
    bus = get_bus()

    if not bus.is_available():
        logger.warning(
            "live_state_tracker: Redis unavailable — state updates will not be "
            "published.  Set REDIS_URL to enable downstream consumers."
        )

    shutdown = asyncio.Event()

    def _handle_signal(sig, _frame):
        logger.info(f"live_state_tracker: received {signal.Signals(sig).name} — shutting down.")
        shutdown.set()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if _USE_WEBSOCKET and bus.is_available():
        logger.info("live_state_tracker: LIVE_FEED_WS_URL set — using WebSocket mode")
        await ws_consumer_loop(bdl, bus, shutdown)
        # If ws_consumer_loop returns (subscription failure), fall back to polling
        if not shutdown.is_set():
            logger.info("live_state_tracker: WebSocket unavailable — falling back to polling")
            await poll_loop(bdl, bus, shutdown)
    else:
        await poll_loop(bdl, bus, shutdown)

    logger.info("live_state_tracker: stopped.")


if __name__ == "__main__":
    asyncio.run(main())
