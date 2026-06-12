import asyncio
import os
import signal
import json
from typing import Dict, Set, List, Optional

from src.events.bus import EventBus, get_bus
from src.data.db import get_db_client
from src.models.live_usage_adjuster import LiveUsageAdjuster
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class LiveUsageWorker:
    """
    Service that listens to player events (sub_in/sub_out) and updates
    real-time usage adjustments in Redis.
    """
    def __init__(self):
        self.db = get_db_client()
        self.bus = get_bus()
        self.adjuster = LiveUsageAdjuster(self.db)
        # game_id -> {team_abbr -> set(player_ids)}
        self.rotations: Dict[str, Dict[str, Set[int]]] = {}
        # game_id -> {team_abbr -> bool}
        self.contexts_loaded: Dict[str, Dict[str, bool]] = {}

    async def _load_context_if_needed(self, game_id: str, team_abbr: str):
        """Ensure team context and starting rotation are loaded."""
        if game_id not in self.contexts_loaded:
            self.contexts_loaded[game_id] = {}
        
        if team_abbr not in self.contexts_loaded[game_id]:
            logger.info(f"Loading team context for {team_abbr} in game {game_id}")
            try:
                with self.db.get_conn() as conn:
                    # 1. Get all players for this team
                    rows = conn.execute(
                        """SELECT p.player_id 
                           FROM players p 
                           JOIN teams t ON p.team_id = t.team_id 
                           WHERE t.abbreviation = %s OR t.team_name = %s""",
                        (team_abbr, team_abbr)
                    ).fetchall()
                    roster_ids = [int(r[0]) for r in rows]
                    
                    # 2. Try to get starters from rotation_slots
                    rows = conn.execute(
                        """SELECT player_id FROM rotation_slots 
                           WHERE team_abbr = %s AND slot_key LIKE 'start%%' 
                           ORDER BY slot_probability DESC LIMIT 5""",
                        (team_abbr,)
                    ).fetchall()
                    starters = {int(r[0]) for r in rows}
                    
                    if not starters:
                        logger.warning(f"No starters found for {team_abbr}, will rely on sub events.")

                await self.adjuster.load_team_context(team_abbr, roster_ids, starters)
                self.contexts_loaded[game_id][team_abbr] = True
                
                if game_id not in self.rotations:
                    self.rotations[game_id] = {}
                if team_abbr not in self.rotations[game_id]:
                    # Initialize rotation with starters
                    self.rotations[game_id][team_abbr] = set(starters)
            except Exception as e:
                logger.error(f"Failed to load context for {team_abbr}: {e}")

    def handle_player_event(self, payload: dict):
        """Callback for EventBus.PLAYER_EVENT."""
        event_type = payload.get("event_type")
        if event_type not in ("sub_in", "sub_out", "substitution"):
            return

        game_id = payload.get("game_id")
        player_id_str = payload.get("player_id")
        if not player_id_str:
            return
        player_id = int(player_id_str)
        
        team = payload.get("team")
        if not game_id or not team:
            return

        # Handle sub_in/sub_out if they are in the payload separately (from pbp_streamer)
        # or if it's a single event_type="substitution" (from ws_live_feed)
        if event_type == "substitution":
            sub_in = payload.get("sub_in")
            sub_out = payload.get("sub_out")
            if sub_in:
                asyncio.create_task(self.process_event(game_id, team, int(sub_in), "sub_in"))
            if sub_out:
                asyncio.create_task(self.process_event(game_id, team, int(sub_out), "sub_out"))
        else:
            asyncio.create_task(self.process_event(game_id, team, player_id, event_type))

    async def process_event(self, game_id: str, team_abbr: str, player_id: int, event_type: str):
        """Update rotation and push adjustments to Redis."""
        await self._load_context_if_needed(game_id, team_abbr)
        
        if game_id not in self.rotations or team_abbr not in self.rotations[game_id]:
            return

        rotation = self.rotations[game_id][team_abbr]
        if event_type == "sub_in":
            rotation.add(player_id)
        elif event_type == "sub_out":
            rotation.discard(player_id)
            
        # Recalculate usage for all players on the court
        # Note: If rotation > 5, it means we missed a sub_out; if < 5, missed a sub_in.
        # The adjuster handles conservation regardless.
        
        redis_client = self.bus._client
        if not redis_client:
            return

        for pid in rotation:
            adj = self.adjuster.get_adjusted_features(pid, rotation, team_abbr)
            if adj:
                key = f"live_usage:{pid}"
                # Convert float values to strings for Redis hash
                mapping = {k: str(v) for k, v in adj.items()}
                redis_client.hset(key, mapping=mapping)
                redis_client.expire(key, 86400) # 24h TTL
                
        logger.debug(f"Updated live usage for {len(rotation)} players on {team_abbr}")

    async def run(self, shutdown: asyncio.Event):
        """Main loop."""
        logger.info("LiveUsageWorker: subscribing to player events...")
        self.bus.subscribe(EventBus.PLAYER_EVENT, self.handle_player_event)
        
        while not shutdown.is_set():
            await asyncio.sleep(1)
        
        logger.info("LiveUsageWorker: shutting down.")

async def main():
    worker = LiveUsageWorker()
    shutdown = asyncio.Event()
    
    # Windows-friendly signal handling
    def _stop():
        shutdown.set()
        
    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _stop)
    except NotImplementedError:
        pass # Windows
        
    await worker.run(shutdown)

if __name__ == "__main__":
    asyncio.run(main())
