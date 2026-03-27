from typing import Any
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class BDLPbpAdapter:
    def __init__(self, bdl_client: Any):
        self.bdl = bdl_client

    def get_play_by_play(self, game_id: int):
        try:
            plays = self.bdl.get_plays(game_id)
            return plays
        except Exception as e:
            logger.warning(f"Failed to get PBP for game {game_id}: {e}")
            return []
