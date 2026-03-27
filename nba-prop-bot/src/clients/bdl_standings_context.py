from typing import Any, Dict
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class BDLStandingsContext:
    def __init__(self, bdl_client: Any):
        self.bdl = bdl_client
        self._cache = {}

    def get_game_context(self, home_team: str, away_team: str, season: int) -> Dict[str, float]:
        if season not in self._cache:
            try:
                standings = self.bdl.get_standings(season)
                self._cache[season] = {t.get('team', {}).get('full_name', ''): t for t in standings}
            except Exception as e:
                logger.warning(f"Failed to fetch BDL standings for {season}: {e}")
                self._cache[season] = {}
        
        ctx = {"minutes_adj_home": 1.0, "minutes_adj_away": 1.0}
        
        home_st = self._cache[season].get(home_team)
        away_st = self._cache[season].get(away_team)
        
        if home_st and away_st:
            try:
                # Calculate Win PCT if not directly available (BDL returns wins/losses)
                hw_w = float(home_st.get('won', 0))
                hw_l = float(home_st.get('lost', 0))
                aw_w = float(away_st.get('won', 0))
                aw_l = float(away_st.get('lost', 0))
                
                hw = hw_w / (hw_w + hw_l) if (hw_w + hw_l) > 0 else 0.5
                aw = aw_w / (aw_w + aw_l) if (aw_w + aw_l) > 0 else 0.5
                
                # Blowout blowout limits
                if hw > 0.70 and aw < 0.35:
                    ctx['minutes_adj_home'] = 0.92
                    ctx['minutes_adj_away'] = 0.95
                elif aw > 0.70 and hw < 0.35:
                    ctx['minutes_adj_away'] = 0.92
                    ctx['minutes_adj_home'] = 0.95
            except Exception:
                pass

        return ctx
