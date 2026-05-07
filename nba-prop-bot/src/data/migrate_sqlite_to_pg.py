"""
One-shot migration: SQLite props.db → PostgreSQL / TimescaleDB.

Run ONCE against a copy of props.db before switching PG_DSN in production.

Usage:
    PG_DSN="postgresql://user:pass@host:5432/railway" python -m src.data.migrate_sqlite_to_pg
    # or with a custom SQLite path:
    PG_DSN="..." SQLITE_PATH="/path/to/props.db" python -m src.data.migrate_sqlite_to_pg

Skipped tables:
  - line_history   (start fresh — historical tick data is not worth migrating)
  - schema_migrations (re-applied automatically by PostgresDatabaseClient._init_db)

Migration order respects FK dependencies.

Safety:
  - Reads from SQLite in READ-ONLY mode (no writes to source DB)
  - Uses INSERT ... ON CONFLICT DO NOTHING for idempotency (safe to re-run)
  - Prints a per-table summary; exits non-zero on critical error
"""

import os
import sqlite3
import sys
from datetime import datetime

import psycopg2
import psycopg2.extras

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

SQLITE_PATH = os.getenv("SQLITE_PATH", "props.db")
PG_DSN      = os.getenv("PG_DSN", "")

# Tables in FK-dependency order (parents before children).
# Each entry: (sqlite_table, pg_table, [columns to copy])
# Use None for columns to auto-detect from SQLite schema.
MIGRATION_ORDER = [
    "teams",
    "players",
    "games",
    "bookmaker_profiles",
    "referee_stats",
    "alerts_sent",
    "bet_results",
    "clv_tracking",
    "placed_bets",
    "injury_reports",
    "model_health",
    "steam_alerts",
    "sgp_correlations",
    "on_off_splits",
    "rotation_slots",
    "cross_player_correlations",
    "cross_team_correlations",
    "team_opponent_stats",
    "pending_alerts",
    "backtest_results",
    "bdl_defense_profiles",
    "bdl_game_log_cache",
    "prop_snapshots",
    "projections",
    "player_game_logs",
    "team_context_daily",
    "model_versions",
]

# These tables are intentionally skipped.
SKIP_TABLES = {
    "line_history",       # start fresh — tick data should not be migrated
    "schema_migrations",  # managed by _init_db()
}

# SQLite → PostgreSQL type coercions applied to individual values.
# Most types are compatible; this handles edge cases.
def _coerce(val):
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return val


def _get_sqlite_columns(sqlite_conn: sqlite3.Connection, table: str) -> list:
    cursor = sqlite_conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]


def _get_pg_columns(pg_conn, table: str) -> list:
    cur = pg_conn.cursor()
    cur.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """,
        (table,),
    )
    return [r[0] for r in cur.fetchall()]


def migrate_table(sqlite_conn: sqlite3.Connection, pg_conn,
                  table: str, batch_size: int = 500) -> int:
    """
    Copy all rows from SQLite table to PostgreSQL table.
    Uses INSERT ... ON CONFLICT DO NOTHING so re-runs are idempotent.
    Returns the number of rows inserted.
    """
    sqlite_cols = _get_sqlite_columns(sqlite_conn, table)
    if not sqlite_cols:
        logger.warning(f"  {table}: not found in SQLite — skipped.")
        return 0

    pg_cols = _get_pg_columns(pg_conn, table)
    if not pg_cols:
        logger.warning(f"  {table}: not found in PostgreSQL — skipped.")
        return 0

    # Only copy columns that exist in BOTH databases
    common = [c for c in sqlite_cols if c in pg_cols]
    if not common:
        logger.warning(f"  {table}: no common columns — skipped.")
        return 0

    col_str     = ", ".join(common)
    placeholder = ", ".join(["%s"] * len(common))
    insert_sql  = (
        f"INSERT INTO {table} ({col_str}) VALUES ({placeholder}) "
        f"ON CONFLICT DO NOTHING"
    )

    sqlite_cur = sqlite_conn.execute(f"SELECT {col_str} FROM {table}")
    pg_cur     = pg_conn.cursor()

    total = 0
    while True:
        rows = sqlite_cur.fetchmany(batch_size)
        if not rows:
            break
        coerced = [tuple(_coerce(v) for v in row) for row in rows]
        pg_cur.executemany(insert_sql, coerced)
        total += len(rows)

    pg_conn.commit()
    return total


def run_migration():
    if not PG_DSN:
        print("ERROR: PG_DSN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(SQLITE_PATH):
        print(f"ERROR: SQLite database not found at {SQLITE_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"SQLite → PostgreSQL migration")
    print(f"  Source:      {SQLITE_PATH}")
    print(f"  Destination: {PG_DSN.split('@')[-1]}")
    print(f"  Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Open SQLite in read-only URI mode
    sqlite_conn = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True)
    sqlite_conn.row_factory = sqlite3.Row

    pg_conn = psycopg2.connect(PG_DSN, cursor_factory=psycopg2.extras.RealDictCursor)

    # Ensure PostgreSQL schema is initialised before migrating
    print("Initialising PostgreSQL schema...")
    from src.data.db_postgres import PostgresDatabaseClient
    PostgresDatabaseClient(dsn=PG_DSN)  # _init_db() runs automatically
    print("Schema ready.\n")

    total_rows = 0
    errors = []

    for table in MIGRATION_ORDER:
        if table in SKIP_TABLES:
            print(f"  {table:<40} SKIPPED (intentional)")
            continue
        try:
            n = migrate_table(sqlite_conn, pg_conn, table)
            print(f"  {table:<40} {n:>6} rows")
            total_rows += n
        except Exception as e:
            pg_conn.rollback()
            msg = f"  {table:<40} ERROR: {e}"
            print(msg)
            logger.error(msg)
            errors.append((table, str(e)))

    sqlite_conn.close()
    pg_conn.close()

    print(f"\n{'='*60}")
    print(f"Migration complete: {total_rows} rows total")
    print(f"Errors: {len(errors)}")
    for t, err in errors:
        print(f"  {t}: {err}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    run_migration()
