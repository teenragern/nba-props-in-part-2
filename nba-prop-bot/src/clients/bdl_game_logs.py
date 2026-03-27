import pandas as pd
from typing import Any
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class BDLGameLogs:
    def __init__(self, bdl_client: Any):
        self.bdl = bdl_client

    def get_player_game_logs(self, bdl_player_id: int, season: int) -> pd.DataFrame:
        try:
            stats = self.bdl.get_game_stats(player_ids=[bdl_player_id], seasons=[season])
            if not stats:
                return pd.DataFrame()
            
            df = pd.DataFrame(stats)
            if 'min' in df.columns:
                rename_map = {
                    'min': 'MIN', 'fgm': 'FGM', 'fga': 'FGA', 'fg3m': 'FG3M', 
                    'ftm': 'FTM', 'reb': 'REB', 'ast': 'AST', 'stl': 'STL', 
                    'blk': 'BLK', 'turnover': 'TOV', 'pts': 'PTS', 'plus_minus': 'PLUS_MINUS',
                    'oreb': 'OREB', 'dreb': 'DREB', 'pf': 'PF', 'fg_pct': 'FG_PCT',
                    'fg3_pct': 'FG3_PCT', 'ft_pct': 'FT_PCT'
                }
                df.rename(columns=rename_map, inplace=True)
                
                def parse_min(m):
                    if not m: return 0.0
                    m_str = str(m)
                    if ':' in m_str:
                        parts = m_str.split(':')
                        try:
                            return float(parts[0]) + float(parts[1])/60.0
                        except:
                            return 0.0
                    try:
                        return float(m)
                    except:
                        return 0.0
                
                df['MIN'] = df['MIN'].apply(parse_min)
                
                if 'game' in df.columns:
                    df['GAME_DATE'] = df['game'].apply(lambda x: x.get('date') if isinstance(x, dict) else None)
                    df['_dt'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
                    df = df.sort_values('_dt', ascending=False).drop(columns=['_dt'])

            return df
        except Exception as e:
            logger.warning(f"BDLGameLogs error for {bdl_player_id}: {e}")
            return pd.DataFrame()
