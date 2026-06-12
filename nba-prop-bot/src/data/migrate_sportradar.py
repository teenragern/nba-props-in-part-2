from src.data.db import get_db_client
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def migrate():
    db = get_db_client()
    sql = """
    CREATE TABLE IF NOT EXISTS lineups_official (
        game_id     TEXT    NOT NULL,
        player_name TEXT    NOT NULL,
        team        TEXT    NOT NULL,
        is_starter  BOOLEAN NOT NULL DEFAULT FALSE,
        updated_at  TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (game_id, player_name)
    );
    
    CREATE TABLE IF NOT EXISTS sportradar_audit_logs (
        game_id      TEXT PRIMARY KEY,
        pbp_saved    BOOLEAN DEFAULT FALSE,
        box_saved    BOOLEAN DEFAULT FALSE,
        audited_at   TIMESTAMPTZ DEFAULT NOW()
    );
    """
    
    logger.info("Running Sportradar schema migration...")
    with db.get_conn() as conn:
        conn.execute(sql)
    logger.info("Migration complete.")

if __name__ == "__main__":
    migrate()
