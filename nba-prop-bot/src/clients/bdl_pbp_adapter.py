"""
BDL play-by-play adapter.

Provides per-player shot quality metrics derived from recent game PBP data:
  foul_draw_rate  — foul-drawing plays per shooting play (FTA tendency)
  paint_shot_pct  — share of shots attempted from the paint
  ast_rate        — share of scoring plays where another player assisted

Used to modulate FTA-heavy and paint-scoring projections in scan_props.py.
Wire via: _bdl_pbp.get_player_shot_profile(bdl_pid, game_ids)
"""

import math
from typing import Dict, List
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Approximate paint radius in BDL coordinate units.
# BDL returns coordinate_x / coordinate_y relative to basket center.
# Values are typically in 1/10-foot units, so 7 ft ≈ 70 units.
_PAINT_RADIUS = 70


def _in_paint(x, y) -> bool:
    try:
        return math.sqrt(float(x) ** 2 + float(y) ** 2) <= _PAINT_RADIUS
    except (TypeError, ValueError):
        return False


class BDLPbpAdapter:
    """
    Fetches BDL play-by-play for recent games and derives shot quality
    and foul-rate features for prop projection adjustments.
    """

    def __init__(self, bdl_client):
        self.bdl = bdl_client
        self._game_plays_cache: Dict[int, List[dict]] = {}  # game_id → plays

    def _get_plays(self, game_id: int) -> List[dict]:
        if game_id not in self._game_plays_cache:
            try:
                plays = self.bdl.get_plays(game_id)
                self._game_plays_cache[game_id] = plays or []
            except Exception as e:
                logger.debug(f"BDL PBP fetch failed for game {game_id}: {e}")
                self._game_plays_cache[game_id] = []
        return self._game_plays_cache[game_id]

    def get_player_shot_profile(
        self,
        bdl_player_id: int,
        game_ids: List[int],
    ) -> Dict[str, float]:
        """
        Derive shot quality features from recent game PBP data.

        Args:
            bdl_player_id:  BDL player ID to filter plays by
            game_ids:       List of BDL game IDs to scan (most recent first)

        Returns dict with:
            foul_draw_rate  float  foul-drawing plays per shooting play (0–1+)
            paint_shot_pct  float  share of shooting plays from paint (0–1)
            ast_rate        float  share of scoring plays that were assisted (0–1)
        """
        defaults = {'foul_draw_rate': 0.0, 'paint_shot_pct': 0.0, 'ast_rate': 0.0}
        if not game_ids:
            return defaults

        shooting_plays = 0
        foul_plays = 0
        paint_shots = 0
        scoring_plays = 0
        assisted_scores = 0

        for gid in game_ids:
            plays = self._get_plays(gid)
            for play in plays:
                participants = play.get('participants') or []
                if not isinstance(participants, list):
                    participants = []

                involved = any(
                    p.get('player_id') == bdl_player_id for p in participants
                )
                if not involved:
                    continue

                play_type = (play.get('type') or '').lower()

                if play.get('shooting_play'):
                    shooting_plays += 1
                    x = play.get('coordinate_x')
                    y = play.get('coordinate_y')
                    if x is not None and y is not None and _in_paint(x, y):
                        paint_shots += 1

                # Foul drawn: foul play where the player is NOT the one committing
                if 'foul' in play_type:
                    foul_plays += 1

                if play.get('scoring_play'):
                    scoring_plays += 1
                    # Assisted if another participant is present (passer)
                    other_players = [
                        p for p in participants
                        if p.get('player_id') != bdl_player_id
                    ]
                    if other_players:
                        assisted_scores += 1

        result = dict(defaults)
        if shooting_plays > 0:
            result['foul_draw_rate'] = foul_plays / shooting_plays
            result['paint_shot_pct'] = paint_shots / shooting_plays
        if scoring_plays > 0:
            result['ast_rate'] = assisted_scores / scoring_plays

        logger.debug(
            f"BDL PBP profile player={bdl_player_id}: "
            f"shoots={shooting_plays} fouls={foul_plays} "
            f"paint={paint_shots} scored={scoring_plays} assisted={assisted_scores}"
        )
        return result
