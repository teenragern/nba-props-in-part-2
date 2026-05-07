import time
from src.utils.logging_utils import get_logger
from src.clients.nba_stats import NbaStatsClient
from src.data.db import get_db_client
from datetime import datetime

logger = get_logger(__name__)

def sync_team_stats():
    client = NbaStatsClient()
    db = get_db_client()
    
    logger.info("Syncing team stats...")
    try:
        df = client.get_team_stats()
        if df.empty:
            logger.warning("No team stats returned.")
            return
            
        today = datetime.now().strftime('%Y-%m-%d')
        
        with db.get_conn() as conn:
            cursor = conn.cursor()
            for _, row in df.iterrows():
                team_id = row['TEAM_ID']
                pac = row.get('PACE', 99.0)
                off = row.get('OFF_RATING', 110.0)
                def_r = row.get('DEF_RATING', 110.0)
                
                # Using overall defensive stats as proxy for positional splits in V1
                opp_pts = row.get('OPP_PTS', 110.0)
                opp_reb = row.get('OPP_REB', 40.0)
                opp_ast = row.get('OPP_AST', 25.0)
                
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO team_context_daily 
                    (team_id, game_date, pace_rating, offensive_rating, defensive_rating,
                     opponent_pts_allowed_per_pos, opponent_reb_allowed_per_pos, opponent_ast_allowed_per_pos)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (team_id, today, pac, off, def_r, opp_pts, opp_reb, opp_ast)
                )
        logger.info("Team stats synced successfully.")
    except Exception as e:
        logger.error(f"Error syncing team stats: {e}")

if __name__ == "__main__":
    sync_team_stats()
