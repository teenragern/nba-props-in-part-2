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
from enum import Enum
from typing import Dict, Optional, Set

from src.events.bus import EventBus, get_bus
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

_PBP_WS_URL = os.getenv("PBP_WS_URL", "")
_PBP_API_KEY = os.getenv("PBP_API_KEY", "")
_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MULTIPLIER = 2.0
_RECONNECT_MAX_DELAY = 120.0


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
    Async WebSocket consumer for play-by-play data.
    Publishes enriched MicroGameState to Redis on every meaningful event.
    """

    def __init__(self, ws_url: str, bus: EventBus, api_key: str = ""):
        self._ws_url = self._build_url(ws_url, api_key)
        self._bus = bus
        self._engine = PBPStateEngine()
        self._reconnect_delay = _RECONNECT_BASE_DELAY
        self._events_published = 0
        self._events_received = 0

    @staticmethod
    def _build_url(base_url: str, api_key: str) -> str:
        if not api_key:
            return base_url
        sep = "&" if "?" in base_url else "?"
        return f"{base_url}{sep}api_key={api_key}"

    async def run_forever(self, shutdown: asyncio.Event) -> None:
        """Main loop: connect, receive, parse, update state, publish."""
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
                logger.info(f"pbp_streamer: connecting to {self._ws_url[:60]}...")
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info("pbp_streamer: connected successfully")
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
                    f"pbp_streamer: connection lost ({type(e).__name__}: {e}). "
                    f"Reconnecting in {self._reconnect_delay:.1f}s..."
                )
                try:
                    await asyncio.wait_for(
                        shutdown.wait(), timeout=self._reconnect_delay
                    )
                    break  # shutdown signaled during wait
                except asyncio.TimeoutError:
                    pass  # normal: timeout expired, retry connection
                self._reconnect_delay = min(
                    self._reconnect_delay * _RECONNECT_MULTIPLIER,
                    _RECONNECT_MAX_DELAY,
                )

        logger.info(
            f"pbp_streamer: shutdown. Received={self._events_received} "
            f"Published={self._events_published}"
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
