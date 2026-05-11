"""
Sub-Second Play-by-Play Ingestion Worker.

Connects to a tier-one WebSocket PBP feed (Sportradar / NBA raw / compatible
provider) and publishes enriched micro-state events to the Redis event bus at
sub-second latency.

Tracks per-game:
  - Exact game clock and shot clock
  - Active possession (state-machine driven)
  - Team fouls per period (bonus / double-bonus detection)
  - 5-man lineup on court for each team
  - Last play-by-play event type and actor

Publishes:  nba:live:micro_state  (EventBus.LIVE_MICRO_STATE)

Run:
    python -m src.clients.pbp_streamer

Environment:
    PBP_WS_URL      WebSocket endpoint for the PBP feed (required)
    PBP_API_KEY     API key appended as query param (optional)
    REDIS_URL       Redis connection for EventBus
"""

import asyncio
import json
import os
import signal
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]

from src.events.bus import EventBus, get_bus
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

_PBP_WS_URL = os.getenv("PBP_WS_URL", "")
_PBP_API_KEY = os.getenv("PBP_API_KEY", "")
_PBP_MODE = os.getenv("PBP_MODE", "auto")  # "auto", "stream", "rest_poll"
_PBP_POLL_INTERVAL = float(os.getenv("PBP_POLL_INTERVAL", "3.0"))  # seconds between REST polls
_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MULTIPLIER = 2.0
_RECONNECT_MAX_DELAY = 120.0

# Sportradar REST PBP base URL (trial-compatible)
_SPORTRADAR_REST_BASE = os.getenv(
    "SPORTRADAR_REST_BASE",
    "https://api.sportradar.com/nba/trial/v8/en"
)
# Sportradar daily schedule endpoint
_SPORTRADAR_SCHEDULE_URL = f"{_SPORTRADAR_REST_BASE}/games/{{date}}/schedule.json"
# Sportradar PBP endpoint
_SPORTRADAR_PBP_URL = f"{_SPORTRADAR_REST_BASE}/games/{{game_id}}/pbp.json"


# ── Data Structures ──────────────────────────────────────────────────────────

class PossessionState(str, Enum):
    HOME = "home"
    AWAY = "away"
    DEAD_BALL = "dead_ball"
    UNKNOWN = "unknown"


class PBPEventType(str, Enum):
    MADE_FG = "made_fg"
    MISSED_FG = "missed_fg"
    MADE_3PT = "made_3pt"
    MISSED_3PT = "missed_3pt"
    FREE_THROW_MADE = "free_throw_made"
    FREE_THROW_MISSED = "free_throw_missed"
    OFFENSIVE_REBOUND = "offensive_rebound"
    DEFENSIVE_REBOUND = "defensive_rebound"
    TURNOVER = "turnover"
    STEAL = "steal"
    FOUL = "foul"
    SUBSTITUTION = "substitution"
    TIMEOUT = "timeout"
    JUMP_BALL = "jump_ball"
    PERIOD_START = "period_start"
    PERIOD_END = "period_end"
    CLOCK_UPDATE = "clock_update"
    SHOT_CLOCK_RESET = "shot_clock_reset"
    UNKNOWN = "unknown"


@dataclass
class TeamFoulState:
    """Tracks fouls for a single team in the current period."""
    period_fouls: int = 0
    in_bonus: bool = False
    in_double_bonus: bool = False

    def add_foul(self) -> None:
        self.period_fouls += 1
        self.in_bonus = self.period_fouls >= 5
        self.in_double_bonus = self.period_fouls >= 10

    def reset_period(self) -> None:
        self.period_fouls = 0
        self.in_bonus = False
        self.in_double_bonus = False


@dataclass
class GameState:
    """Full in-memory state for a single live game."""
    game_id: str
    home_team: str
    away_team: str
    home_score: int = 0
    away_score: int = 0
    period: int = 1
    game_clock_seconds: float = 720.0  # 12:00 remaining in period
    shot_clock_seconds: float = 24.0
    possession: PossessionState = PossessionState.UNKNOWN
    home_fouls: TeamFoulState = field(default_factory=TeamFoulState)
    away_fouls: TeamFoulState = field(default_factory=TeamFoulState)
    players_on_court_home: Set[str] = field(default_factory=set)
    players_on_court_away: Set[str] = field(default_factory=set)
    last_event_type: str = ""
    last_event_player: str = ""
    # Free throw tracking
    _ft_shooting_team: Optional[str] = field(default=None, repr=False)
    _ft_attempts_remaining: int = field(default=0, repr=False)

    @property
    def minutes_elapsed(self) -> float:
        """Total game minutes elapsed (0-48+ for OT)."""
        completed_periods = max(0, self.period - 1)
        period_minutes = 12.0 if self.period <= 4 else 5.0
        elapsed_in_period = period_minutes - (self.game_clock_seconds / 60.0)
        return completed_periods * 12.0 + max(0.0, elapsed_in_period)

    @property
    def home_in_bonus(self) -> bool:
        return self.home_fouls.in_bonus

    @property
    def away_in_bonus(self) -> bool:
        return self.away_fouls.in_bonus

    def to_micro_state_dict(self) -> dict:
        """Serialize for Redis publish."""
        return {
            "game_id": self.game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "period": self.period,
            "game_clock_seconds": round(self.game_clock_seconds, 1),
            "shot_clock_seconds": round(self.shot_clock_seconds, 1),
            "possession": self.possession.value,
            "home_period_fouls": self.home_fouls.period_fouls,
            "away_period_fouls": self.away_fouls.period_fouls,
            "home_in_bonus": self.home_fouls.in_bonus,
            "away_in_bonus": self.away_fouls.in_bonus,
            "home_in_double_bonus": self.home_fouls.in_double_bonus,
            "away_in_double_bonus": self.away_fouls.in_double_bonus,
            "players_on_court_home": sorted(self.players_on_court_home),
            "players_on_court_away": sorted(self.players_on_court_away),
            "last_event_type": self.last_event_type,
            "last_event_player": self.last_event_player,
            "minutes_elapsed": round(self.minutes_elapsed, 2),
            "event_ts": time.time(),
        }


# ── PBP Event Parser ─────────────────────────────────────────────────────────

def _classify_event(msg_type: str) -> PBPEventType:
    """Map provider message types to our canonical event types."""
    mapping = {
        "made_shot": PBPEventType.MADE_FG,
        "made_fg": PBPEventType.MADE_FG,
        "missed_shot": PBPEventType.MISSED_FG,
        "missed_fg": PBPEventType.MISSED_FG,
        "made_3pt": PBPEventType.MADE_3PT,
        "three_made": PBPEventType.MADE_3PT,
        "missed_3pt": PBPEventType.MISSED_3PT,
        "three_missed": PBPEventType.MISSED_3PT,
        "free_throw_made": PBPEventType.FREE_THROW_MADE,
        "ft_made": PBPEventType.FREE_THROW_MADE,
        "free_throw_missed": PBPEventType.FREE_THROW_MISSED,
        "ft_missed": PBPEventType.FREE_THROW_MISSED,
        "offensive_rebound": PBPEventType.OFFENSIVE_REBOUND,
        "oreb": PBPEventType.OFFENSIVE_REBOUND,
        "defensive_rebound": PBPEventType.DEFENSIVE_REBOUND,
        "dreb": PBPEventType.DEFENSIVE_REBOUND,
        "turnover": PBPEventType.TURNOVER,
        "steal": PBPEventType.STEAL,
        "foul": PBPEventType.FOUL,
        "personal_foul": PBPEventType.FOUL,
        "shooting_foul": PBPEventType.FOUL,
        "flagrant_foul": PBPEventType.FOUL,
        "technical_foul": PBPEventType.FOUL,
        "substitution": PBPEventType.SUBSTITUTION,
        "sub": PBPEventType.SUBSTITUTION,
        "timeout": PBPEventType.TIMEOUT,
        "jump_ball": PBPEventType.JUMP_BALL,
        "period_start": PBPEventType.PERIOD_START,
        "period_end": PBPEventType.PERIOD_END,
        "clock": PBPEventType.CLOCK_UPDATE,
        "clock_update": PBPEventType.CLOCK_UPDATE,
        "shot_clock_reset": PBPEventType.SHOT_CLOCK_RESET,
        "shot_clock": PBPEventType.SHOT_CLOCK_RESET,
    }
    return mapping.get(msg_type, PBPEventType.UNKNOWN)


def parse_pbp_message(raw: str) -> Optional[dict]:
    """
    Parse raw provider WebSocket JSON into a normalized event dict.

    Expected provider message format (Sportradar-compatible):
    {
        "type": "event",
        "payload": {
            "game_id": "...",
            "event_type": "made_shot" | "foul" | "substitution" | ...,
            "team": "home" | "away",
            "player_id": "...",
            "player_name": "...",
            "clock": {"period": 2, "seconds_remaining": 345.2},
            "shot_clock": 14.0,
            "points": 2,  // for scoring events
            "foul_type": "shooting" | "personal" | "flagrant",
            "sub_in": "player_id",
            "sub_out": "player_id",
            // Game metadata (sent on connection/period start)
            "home_team": "...",
            "away_team": "...",
            "home_score": 55,
            "away_score": 52,
        }
    }
    """
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    # Support both envelope and flat formats
    if "payload" in msg:
        payload = msg["payload"]
    elif "event_type" in msg:
        payload = msg
    else:
        return None

    event_type_raw = payload.get("event_type", payload.get("type", ""))
    if not event_type_raw:
        return None

    return {
        "game_id": str(payload.get("game_id", "")),
        "event_type": event_type_raw,
        "team": payload.get("team", ""),  # "home" or "away"
        "player_id": str(payload.get("player_id", "")),
        "player_name": payload.get("player_name", ""),
        "period": payload.get("clock", {}).get("period", payload.get("period")),
        "seconds_remaining": payload.get("clock", {}).get(
            "seconds_remaining", payload.get("seconds_remaining")
        ),
        "shot_clock": payload.get("shot_clock"),
        "points": payload.get("points", 0),
        "foul_type": payload.get("foul_type", "personal"),
        "sub_in": payload.get("sub_in", ""),
        "sub_out": payload.get("sub_out", ""),
        "home_team": payload.get("home_team", ""),
        "away_team": payload.get("away_team", ""),
        "home_score": payload.get("home_score"),
        "away_score": payload.get("away_score"),
    }


# ── State Update Engine ──────────────────────────────────────────────────────

class PBPStateEngine:
    """
    Maintains per-game state from a stream of play-by-play events.
    Implements a possession state machine and foul/bonus tracking.
    """

    def __init__(self):
        self._games: Dict[str, GameState] = {}

    def get_or_create_game(self, game_id: str, home_team: str = "",
                           away_team: str = "") -> GameState:
        if game_id not in self._games:
            self._games[game_id] = GameState(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
            )
        game = self._games[game_id]
        if home_team and not game.home_team:
            game.home_team = home_team
        if away_team and not game.away_team:
            game.away_team = away_team
        return game

    def process_event(self, event: dict) -> Optional[GameState]:
        """
        Process a parsed PBP event and return updated GameState (or None).

        Returns the game state only when a meaningful state change occurred
        that warrants a Redis publish.
        """
        game_id = event.get("game_id", "")
        if not game_id:
            return None

        game = self.get_or_create_game(
            game_id,
            event.get("home_team", ""),
            event.get("away_team", ""),
        )

        event_type = _classify_event(event.get("event_type", ""))
        team = event.get("team", "")  # "home" or "away"

        # Update clock if provided
        if event.get("seconds_remaining") is not None:
            game.game_clock_seconds = float(event["seconds_remaining"])
        if event.get("period") is not None:
            game.period = int(event["period"])
        if event.get("shot_clock") is not None:
            game.shot_clock_seconds = float(event["shot_clock"])

        # Update scores if provided (absolute values from provider)
        if event.get("home_score") is not None:
            game.home_score = int(event["home_score"])
        if event.get("away_score") is not None:
            game.away_score = int(event["away_score"])

        game.last_event_type = event_type.value
        game.last_event_player = event.get("player_name", "")

        # ── Possession State Machine ──────────────────────────────────
        self._update_possession(game, event_type, team, event)

        # ── Foul Tracking ─────────────────────────────────────────────
        self._update_fouls(game, event_type, team, event)

        # ── Lineup Tracking ───────────────────────────────────────────
        self._update_lineup(game, event_type, team, event)

        # ── Score Tracking (from events, not just absolute) ───────────
        self._update_score_from_event(game, event_type, team, event)

        # ── Period Transitions ────────────────────────────────────────
        if event_type == PBPEventType.PERIOD_START:
            game.home_fouls.reset_period()
            game.away_fouls.reset_period()
            game.shot_clock_seconds = 24.0

        return game

    def _update_possession(self, game: GameState, event_type: PBPEventType,
                           team: str, event: dict) -> None:
        """Possession state machine transitions."""
        opponent = "away" if team == "home" else "home"

        if event_type in (PBPEventType.MADE_FG, PBPEventType.MADE_3PT):
            # After made basket: opponent inbounds (gets possession)
            game.possession = PossessionState(opponent)
            game.shot_clock_seconds = 24.0
            game._ft_shooting_team = None
            game._ft_attempts_remaining = 0

        elif event_type == PBPEventType.FREE_THROW_MADE:
            # Track FT sequence
            if game._ft_attempts_remaining > 0:
                game._ft_attempts_remaining -= 1
            if game._ft_attempts_remaining == 0:
                # Last FT made: opponent gets ball
                game.possession = PossessionState(opponent)
                game.shot_clock_seconds = 24.0
                game._ft_shooting_team = None

        elif event_type == PBPEventType.FREE_THROW_MISSED:
            if game._ft_attempts_remaining > 0:
                game._ft_attempts_remaining -= 1
            if game._ft_attempts_remaining == 0:
                # Last FT missed: live ball, possession TBD (rebound)
                game.possession = PossessionState.DEAD_BALL
                game._ft_shooting_team = None

        elif event_type == PBPEventType.DEFENSIVE_REBOUND:
            # Defensive rebound: rebounding team gets possession
            game.possession = PossessionState(team)
            game.shot_clock_seconds = 24.0

        elif event_type == PBPEventType.OFFENSIVE_REBOUND:
            # Offensive rebound: same team retains possession
            game.possession = PossessionState(team)
            game.shot_clock_seconds = 14.0  # NBA rule: 14s on offensive rebound

        elif event_type == PBPEventType.TURNOVER:
            # Turnover: opponent gets possession
            game.possession = PossessionState(opponent)
            game.shot_clock_seconds = 24.0

        elif event_type == PBPEventType.STEAL:
            # Steal: stealing team gets possession
            game.possession = PossessionState(team)
            game.shot_clock_seconds = 24.0

        elif event_type == PBPEventType.FOUL:
            foul_type = event.get("foul_type", "personal")
            if foul_type in ("shooting", "shooting_foul", "flagrant"):
                # Shooting foul: fouled team shoots FTs
                ft_team = opponent  # team that was fouled
                game._ft_shooting_team = ft_team
                # Determine FT count based on foul type
                if foul_type == "flagrant":
                    game._ft_attempts_remaining = 2
                elif event.get("points", 0) == 3:
                    game._ft_attempts_remaining = 3
                else:
                    game._ft_attempts_remaining = 2
                game.possession = PossessionState(ft_team)
            # Non-shooting fouls: possession doesn't change
            # (unless in bonus — handled by FT sequence above)

        elif event_type == PBPEventType.JUMP_BALL:
            if team:
                game.possession = PossessionState(team)
                game.shot_clock_seconds = 24.0

        elif event_type in (PBPEventType.MISSED_FG, PBPEventType.MISSED_3PT):
            # Missed shot: possession TBD (rebound decides)
            game.possession = PossessionState.DEAD_BALL

        elif event_type == PBPEventType.TIMEOUT:
            # Timeout: team that called it retains possession after
            if team:
                game.possession = PossessionState(team)

        elif event_type == PBPEventType.SHOT_CLOCK_RESET:
            game.shot_clock_seconds = 24.0

    def _update_fouls(self, game: GameState, event_type: PBPEventType,
                      team: str, event: dict) -> None:
        """Track team fouls per period for bonus detection."""
        if event_type != PBPEventType.FOUL:
            return
        foul_type = event.get("foul_type", "personal")
        # Technical fouls don't count toward team foul total for bonus
        if foul_type in ("technical", "technical_foul"):
            return
        if team == "home":
            game.home_fouls.add_foul()
        elif team == "away":
            game.away_fouls.add_foul()

    def _update_lineup(self, game: GameState, event_type: PBPEventType,
                       team: str, event: dict) -> None:
        """Track 5-man lineups via substitution events."""
        if event_type != PBPEventType.SUBSTITUTION:
            return

        sub_in = event.get("sub_in", "")
        sub_out = event.get("sub_out", "")

        if team == "home":
            lineup = game.players_on_court_home
        elif team == "away":
            lineup = game.players_on_court_away
        else:
            return

        if sub_out and sub_out in lineup:
            lineup.discard(sub_out)
        if sub_in:
            lineup.add(sub_in)

    def _update_score_from_event(self, game: GameState, event_type: PBPEventType,
                                 team: str, event: dict) -> None:
        """
        Increment score from scoring events (backup for when provider
        doesn't send absolute scores in every message).
        """
        points = 0
        if event_type == PBPEventType.MADE_FG:
            points = event.get("points", 2)
        elif event_type == PBPEventType.MADE_3PT:
            points = 3
        elif event_type == PBPEventType.FREE_THROW_MADE:
            points = 1

        if points == 0:
            return

        # Only apply if absolute scores weren't provided in this event
        if event.get("home_score") is not None:
            return  # already set from absolute values

        if team == "home":
            game.home_score += points
        elif team == "away":
            game.away_score += points


# ── Main Async Streamer ──────────────────────────────────────────────────────

class PBPStreamer:
    """
    Async consumer for play-by-play data.

    Supports two transport modes (auto-detected from URL):
      1. HTTP Chunked Streaming (Sportradar Push Feeds) — https:// URLs
      2. WebSocket — wss:// or ws:// URLs

    Publishes enriched MicroGameState to Redis on every meaningful event.
    """

    def __init__(self, ws_url: str, bus: EventBus, api_key: str = ""):
        self._base_url = ws_url
        self._api_key = api_key
        self._url = self._build_url(ws_url, api_key)
        self._bus = bus
        self._engine = PBPStateEngine()
        self._reconnect_delay = _RECONNECT_BASE_DELAY
        self._events_published = 0
        self._events_received = 0
        # Auto-detect transport mode
        self._use_http_stream = ws_url.startswith("http://") or ws_url.startswith("https://")

    @staticmethod
    def _build_url(base_url: str, api_key: str) -> str:
        if not api_key:
            return base_url
        sep = "&" if "?" in base_url else "?"
        return f"{base_url}{sep}api_key={api_key}"

    async def run_forever(self, shutdown: asyncio.Event) -> None:
        """Main loop: connect, receive, parse, update state, publish."""
        mode = _PBP_MODE

        if mode == "rest_poll":
            await self._run_rest_poll(shutdown)
        elif mode == "stream":
            if self._use_http_stream:
                await self._run_http_stream(shutdown)
            else:
                await self._run_websocket(shutdown)
        else:
            # Auto mode: try HTTP stream first, fall back to REST polling on 403
            if self._use_http_stream:
                success = await self._try_http_stream_once(shutdown)
                if not success and not shutdown.is_set():
                    logger.info(
                        "pbp_streamer: Push feed unavailable (trial key?). "
                        "Falling back to REST PBP polling mode."
                    )
                    await self._run_rest_poll(shutdown)
            else:
                await self._run_websocket(shutdown)

        logger.info(
            f"pbp_streamer: shutdown. Received={self._events_received} "
            f"Published={self._events_published}"
        )

    # ── HTTP Chunked Streaming (Sportradar Push Feed) ─────────────────

    async def _run_http_stream(self, shutdown: asyncio.Event) -> None:
        """
        Consume Sportradar-style HTTP chunked streaming (Push Feed).

        Sportradar Push feeds use HTTP with Transfer-Encoding: chunked.
        Each chunk is a JSON line (newline-delimited). Heartbeats are sent
        every 5 seconds to keep the connection alive.
        """
        if aiohttp is None:
            logger.error(
                "pbp_streamer: 'aiohttp' package not installed. "
                "Install with: pip install aiohttp"
            )
            return

        while not shutdown.is_set():
            try:
                logger.info(f"pbp_streamer: connecting to HTTP stream {self._url[:80]}...")
                timeout = aiohttp.ClientTimeout(total=None, sock_read=30)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(self._url) as resp:
                        if resp.status != 200:
                            logger.error(
                                f"pbp_streamer: HTTP {resp.status} from feed. "
                                f"Check your API key and access level."
                            )
                            await self._backoff_wait(shutdown)
                            continue

                        logger.info(
                            f"pbp_streamer: HTTP stream connected (status={resp.status})"
                        )
                        self._reconnect_delay = _RECONNECT_BASE_DELAY

                        # Read chunked response line by line
                        buffer = ""
                        async for chunk in resp.content.iter_any():
                            if shutdown.is_set():
                                break

                            text = chunk.decode("utf-8", errors="replace")
                            buffer += text

                            # Process complete JSON lines
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                line = line.strip()
                                if not line:
                                    continue
                                # Skip heartbeat messages
                                if line.startswith("{") and '"heartbeat"' in line:
                                    continue
                                await self._handle_message(line)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if shutdown.is_set():
                    break
                logger.warning(
                    f"pbp_streamer: HTTP stream lost ({type(e).__name__}: {e}). "
                    f"Reconnecting in {self._reconnect_delay:.1f}s..."
                )
                await self._backoff_wait(shutdown)

    # ── WebSocket Transport ───────────────────────────────────────────

    async def _run_websocket(self, shutdown: asyncio.Event) -> None:
        """Consume WebSocket play-by-play feed."""
        try:
            import websockets
        except ImportError:
            logger.error(
                "pbp_streamer: 'websockets' package not installed. "
                "Install with: pip install websockets"
            )
            return

        while not shutdown.is_set():
            try:
                logger.info(f"pbp_streamer: connecting to WS {self._url[:60]}...")
                async with websockets.connect(
                    self._url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info("pbp_streamer: WebSocket connected successfully")
                    self._reconnect_delay = _RECONNECT_BASE_DELAY

                    async for raw_message in ws:
                        if shutdown.is_set():
                            break
                        await self._handle_message(raw_message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if shutdown.is_set():
                    break
                logger.warning(
                    f"pbp_streamer: WS connection lost ({type(e).__name__}: {e}). "
                    f"Reconnecting in {self._reconnect_delay:.1f}s..."
                )
                await self._backoff_wait(shutdown)

    # ── Auto-detect: try push, fall back to REST ────────────────────

    async def _try_http_stream_once(self, shutdown: asyncio.Event) -> bool:
        """
        Attempt one HTTP stream connection. Returns True if it worked
        (stays running until shutdown), False if 403/unavailable.
        """
        if aiohttp is None:
            return False

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self._url) as resp:
                    if resp.status == 403:
                        return False
                    if resp.status != 200:
                        return False
                    # Stream is accessible — switch to full streaming mode
                    # Cancel this probe and re-run with the full streaming loop
                    pass
        except Exception:
            return False

        # If we got here, stream works — run the full loop
        await self._run_http_stream(shutdown)
        return True

    # ── REST PBP Polling (Trial Key Compatible) ───────────────────────

    async def _run_rest_poll(self, shutdown: asyncio.Event) -> None:
        """
        Poll Sportradar REST PBP endpoint every few seconds.

        Works with trial API keys. Fetches today's live games, then polls
        their PBP endpoints for new events.

        Flow:
          1. Fetch today's schedule to find live game IDs
          2. For each live game, fetch /games/{id}/pbp.json
          3. Parse new events since last poll
          4. Feed through the same state engine → publish to Redis
          5. Sleep PBP_POLL_INTERVAL seconds
        """
        if aiohttp is None:
            logger.error(
                "pbp_streamer: 'aiohttp' package not installed. "
                "Install with: pip install aiohttp"
            )
            return

        logger.info(
            f"pbp_streamer: starting REST PBP polling mode "
            f"(interval={_PBP_POLL_INTERVAL}s)"
        )

        # Track last seen event index per game to avoid re-processing
        last_event_idx: Dict[str, int] = {}

        while not shutdown.is_set():
            try:
                async with aiohttp.ClientSession() as session:
                    # 1. Get today's live games
                    live_game_ids = await self._fetch_live_game_ids(session)

                    if not live_game_ids:
                        # No live games — sleep longer
                        try:
                            await asyncio.wait_for(shutdown.wait(), timeout=30.0)
                            break
                        except asyncio.TimeoutError:
                            continue

                    # 2. Poll PBP for each live game
                    for game_id in live_game_ids:
                        if shutdown.is_set():
                            break

                        new_events = await self._fetch_pbp_events(
                            session, game_id, last_event_idx.get(game_id, 0)
                        )

                        for event in new_events:
                            await self._handle_message(json.dumps(event))

                        if new_events:
                            last_event_idx[game_id] = (
                                last_event_idx.get(game_id, 0) + len(new_events)
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"pbp_streamer: REST poll error: {e}")

            # Sleep between polls
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=_PBP_POLL_INTERVAL)
                break  # shutdown signaled
            except asyncio.TimeoutError:
                pass  # normal: poll again

    async def _fetch_live_game_ids(self, session) -> List[str]:
        """Fetch today's schedule from Sportradar and return live game IDs."""
        today = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        url = _SPORTRADAR_SCHEDULE_URL.format(date=today)
        url += f"?api_key={self._api_key}" if self._api_key else ""

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.debug(f"pbp_streamer: schedule fetch HTTP {resp.status}")
                    return []
                data = await resp.json()
        except Exception as e:
            logger.debug(f"pbp_streamer: schedule fetch failed: {e}")
            return []

        # Sportradar schedule response structure
        games = data.get("games", [])
        live_ids = []
        for game in games:
            status = game.get("status", "").lower()
            if status in ("inprogress", "halftime", "live"):
                game_id = game.get("id", "")
                if game_id:
                    live_ids.append(game_id)

        if live_ids:
            logger.debug(f"pbp_streamer: found {len(live_ids)} live game(s)")
        return live_ids

    async def _fetch_pbp_events(self, session, game_id: str,
                                skip_count: int) -> List[dict]:
        """
        Fetch PBP for a game and return only new events (after skip_count).

        Parses Sportradar PBP response into our normalized event format.
        """
        url = _SPORTRADAR_PBP_URL.format(game_id=game_id)
        url += f"?api_key={self._api_key}" if self._api_key else ""

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 429:
                    logger.warning("pbp_streamer: rate limited by Sportradar, backing off")
                    await asyncio.sleep(5.0)
                    return []
                if resp.status != 200:
                    return []
                data = await resp.json()
        except Exception as e:
            logger.debug(f"pbp_streamer: PBP fetch failed for {game_id}: {e}")
            return []

        # Parse Sportradar PBP structure
        return self._parse_sportradar_pbp(data, game_id, skip_count)

    def _parse_sportradar_pbp(self, data: dict, game_id: str,
                              skip_count: int) -> List[dict]:
        """
        Parse Sportradar PBP JSON into our normalized event list.

        Sportradar PBP structure:
        {
            "id": "game_uuid",
            "home": {"name": "Lakers", "id": "..."},
            "away": {"name": "Celtics", "id": "..."},
            "periods": [
                {
                    "number": 1,
                    "events": [
                        {
                            "id": "event_uuid",
                            "type": "turnover" | "twopointmade" | ...,
                            "clock": "05:32",
                            "description": "...",
                            "home_points": 55,
                            "away_points": 52,
                            "attribution": {"name": "Lakers", "id": "..."},
                            "player": {"full_name": "LeBron James", "id": "..."},
                        }, ...
                    ]
                }, ...
            ]
        }
        """
        home_team = data.get("home", {}).get("name", "")
        away_team = data.get("away", {}).get("name", "")
        home_id = data.get("home", {}).get("id", "")
        away_id = data.get("away", {}).get("id", "")

        all_events: List[dict] = []
        periods = data.get("periods", [])

        for period_data in periods:
            period_num = period_data.get("number", 1)
            events = period_data.get("events", [])

            for ev in events:
                # Map Sportradar event type to our format
                sr_type = ev.get("type", "").lower()
                normalized_type = self._map_sportradar_event_type(sr_type)
                if normalized_type is None:
                    continue

                # Determine team ("home" or "away")
                attribution = ev.get("attribution", {})
                attr_id = attribution.get("id", "")
                if attr_id == home_id:
                    team = "home"
                elif attr_id == away_id:
                    team = "away"
                else:
                    team = ""

                # Parse clock "MM:SS" to seconds remaining
                clock_str = ev.get("clock", "")
                seconds_remaining = self._parse_clock(clock_str)

                # Player info
                player = ev.get("player", {})
                player_name = player.get("full_name", "")
                player_id = player.get("id", "")

                # Determine points for scoring events
                points = 0
                if "threepoint" in sr_type and "made" in sr_type:
                    points = 3
                elif "twopoint" in sr_type and "made" in sr_type:
                    points = 2
                elif "freethrow" in sr_type and "made" in sr_type:
                    points = 1

                # Foul type
                foul_type = "personal"
                if "shooting" in sr_type:
                    foul_type = "shooting"
                elif "flagrant" in sr_type:
                    foul_type = "flagrant"
                elif "technical" in sr_type:
                    foul_type = "technical"

                all_events.append({
                    "game_id": game_id,
                    "event_type": normalized_type,
                    "team": team,
                    "player_id": player_id,
                    "player_name": player_name,
                    "period": period_num,
                    "seconds_remaining": seconds_remaining,
                    "shot_clock": None,
                    "points": points,
                    "foul_type": foul_type,
                    "sub_in": ev.get("player", {}).get("id", "") if "substitution" in sr_type else "",
                    "sub_out": "",
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": ev.get("home_points"),
                    "away_score": ev.get("away_points"),
                })

        # Return only events we haven't seen yet
        return all_events[skip_count:]

    @staticmethod
    def _map_sportradar_event_type(sr_type: str) -> Optional[str]:
        """Map Sportradar event type strings to our canonical types."""
        mapping = {
            "twopointmade": "made_fg",
            "twopointmiss": "missed_fg",
            "threepointmade": "made_3pt",
            "threepointmiss": "missed_3pt",
            "freethrowmade": "free_throw_made",
            "freethrowmiss": "free_throw_missed",
            "offensiverebound": "offensive_rebound",
            "defensiverebound": "defensive_rebound",
            "rebound": "defensive_rebound",
            "turnover": "turnover",
            "steal": "steal",
            "personalfoul": "foul",
            "shootingfoul": "foul",
            "flagrantfoul": "foul",
            "technicalfoul": "foul",
            "offensivefoul": "foul",
            "substitution": "substitution",
            "timeout": "timeout",
            "jumpball": "jump_ball",
            "periodstart": "period_start",
            "periodend": "period_end",
            "endperiod": "period_end",
        }
        # Handle compound types (e.g., "twopointmade", "personalfoul")
        for key, value in mapping.items():
            if key in sr_type.replace(" ", "").replace("_", ""):
                return value
        return None

    @staticmethod
    def _parse_clock(clock_str: str) -> float:
        """Parse 'MM:SS' or 'M:SS' clock string to seconds remaining."""
        if not clock_str or ":" not in clock_str:
            return 0.0
        try:
            parts = clock_str.split(":")
            minutes = int(parts[0])
            seconds = int(parts[1]) if len(parts) > 1 else 0
            return minutes * 60.0 + seconds
        except (ValueError, IndexError):
            return 0.0

    # ── Shared Helpers ────────────────────────────────────────────────

    async def _backoff_wait(self, shutdown: asyncio.Event) -> None:
        """Wait with exponential backoff, respecting shutdown signal."""
        try:
            await asyncio.wait_for(
                shutdown.wait(), timeout=self._reconnect_delay
            )
        except asyncio.TimeoutError:
            pass
        self._reconnect_delay = min(
            self._reconnect_delay * _RECONNECT_MULTIPLIER,
            _RECONNECT_MAX_DELAY,
        )

    async def _handle_message(self, raw: str) -> None:
        """Parse a raw WebSocket message and publish state update."""
        self._events_received += 1

        event = parse_pbp_message(raw)
        if event is None:
            return

        game_state = self._engine.process_event(event)
        if game_state is None:
            return

        # Publish to Redis
        micro_state = game_state.to_micro_state_dict()
        published = self._bus.publish(EventBus.LIVE_MICRO_STATE, micro_state)
        if published:
            self._events_published += 1
            if self._events_published % 100 == 0:
                logger.debug(
                    f"pbp_streamer: published {self._events_published} events "
                    f"(game {game_state.game_id}: "
                    f"{game_state.home_score}-{game_state.away_score} "
                    f"Q{game_state.period} {game_state.game_clock_seconds:.0f}s)"
                )


# ── Entry Point ──────────────────────────────────────────────────────────────

async def main() -> None:
    """Start the PBP streamer worker."""
    if not _PBP_WS_URL:
        logger.error(
            "pbp_streamer: PBP_WS_URL not set. Cannot start PBP ingestion."
        )
        return

    bus = get_bus()
    if not bus.is_available():
        logger.error("pbp_streamer: Redis unavailable. Cannot publish events.")
        return

    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler():
        logger.info("pbp_streamer: shutdown signal received")
        shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    streamer = PBPStreamer(
        ws_url=_PBP_WS_URL,
        bus=bus,
        api_key=_PBP_API_KEY,
    )

    logger.info("pbp_streamer: starting sub-second PBP ingestion worker")
    await streamer.run_forever(shutdown)


if __name__ == "__main__":
    asyncio.run(main())
