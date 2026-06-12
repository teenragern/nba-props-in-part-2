from src.data.db import get_db_client
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def migrate():
    db = get_db_client()
    sql = """
    ALTER TABLE games ADD COLUMN IF NOT EXISTS sr_id TEXT;
    CREATE INDEX IF NOT EXISTS idx_games_sr_id ON games(sr_id);
    """
    
    logger.info("Adding sr_id column to games table...")
    with db.get_conn() as conn:
        conn.execute(sql)
    logger.info("Migration complete.")

if __name__ == "__main__":
    migrate()
