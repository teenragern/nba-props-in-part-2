"""
One-time fix: reset all PostgreSQL sequences to max(id) after SQLite migration.
Run once: python fix_sequences.py
"""
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()
dsn = os.getenv("PG_DSN")
if not dsn:
    raise RuntimeError("PG_DSN not set in .env")

TABLES = [
    "prop_snapshots",
    "projections",
    "alerts_sent",
    "clv_tracking",
    "model_versions",
    "line_history",
    "backtest_results",
    "steam_alerts",
    "pending_alerts",
    "placed_bets",
    "model_health",
    "subscribers",
]

conn = psycopg2.connect(dsn)
conn.autocommit = True
cur = conn.cursor(cursor_factory=RealDictCursor)

for table in TABLES:
    try:
        cur.execute(f"SELECT MAX(id) AS max_id FROM {table}")
        row = cur.fetchone()
        max_id = row["max_id"] if row and row["max_id"] is not None else 0
        cur.execute(
            f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), %s)",
            (max(max_id, 1),)
        )
        print(f"  {table}: sequence reset to {max(max_id, 1)}")
    except Exception as e:
        print(f"  {table}: SKIP ({e})")

cur.close()
conn.close()
print("\nDone. All sequences reset.")
