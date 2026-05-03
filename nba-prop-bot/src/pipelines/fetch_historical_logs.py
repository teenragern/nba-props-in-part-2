"""
Massive historic training data ingestion pipeline.
Downloads BDL box scores for the previous 3 seasons for all active players
and caches them into SQLite for ML training.

Usage:
    python -m src.pipelines.fetch_historical_logs
"""

import time
from src.data.db import DatabaseClient
from src.clients.bdl_client import BDLClient
from src.clients.bdl_game_logs import BDLGameLogs
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# BDL is GOAT tier (600 rpm).
# Historical seasons to back-populate
SEASONS = [2022, 2023, 2024, 2025]

def run_ingestion():
    bdl = BDLClient()
    db = DatabaseClient()
    game_logs_client = BDLGameLogs(bdl, db=db)

    logger.info("Fetching active players from BDL...")
    players = bdl.get_active_players()
    if not players:
        logger.error("Failed to fetch active players.")
        return

    logger.info(f"Loaded {len(players)} active players. Beginning season ingestion...")
    
    total = len(players)
    for idx, player in enumerate(players):
        pid = player['id']
        name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
        
        for szn in SEASONS:
            # Using ignore_ttl=True to efficiently bypass the 12-hour expiration rule
            existing = db.get_cached_bdl_game_logs(pid, szn, ignore_ttl=True)
            
            # If we already have a reasonable number of games, we probably pulled this season.
            if existing and len(existing) > 5:
                logger.debug(f"[{idx+1}/{total}] Skip {name} - {szn} (Already {len(existing)} games in DB)")
                continue
            
            logger.info(f"[{idx+1}/{total}] Fetching {name} for {szn} season...")
            
            # This fetches from BDL and automatically persists to db.cache_bdl_game_logs
            game_logs_client.get_player_game_logs(pid, szn, ignore_ttl=True)
            
            # Keep pacing completely safe for GOAT tier (10 req/s max)
            time.sleep(0.05)

    logger.info("✅ Massive Historical Ingestion Complete!")

if __name__ == "__main__":
    run_ingestion()
