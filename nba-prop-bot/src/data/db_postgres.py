"""
PostgreSQL / TimescaleDB drop-in replacement for DatabaseClient.

Exposes the same public interface as src/data/db.py::DatabaseClient so all
pipeline code can switch by changing the import (or via get_db_client()).

Key differences from the SQLite client:
  - Placeholders: %s  (not ?)
  - UPSERT: INSERT ... ON CONFLICT ... DO UPDATE  (not INSERT OR REPLACE)
  - Date math: NOW() - INTERVAL '...'  (not datetime('now', '...'))
  - RETURNING id  (not cursor.lastrowid)
  - Connection pool: ThreadedConnectionPool  (one connection per thread)
  - line_history: TimescaleDB hypertable with 30-day retention

Activated when PG_DSN is set in the environment.  Falls back to SQLite
automatically via get_db_client() in db.py.
"""

import os
from contextlib import contextmanager
from typing import Dict, Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _to_py(v):
    """
    Coerce numpy scalar types to native Python so psycopg2 can serialize them.
    numpy.float64 / int64 / bool_ are not registered as psycopg2 adapters and
    get stringified as 'np.float64(x)' which PostgreSQL rejects.
    """
    typ = type(v).__name__
    if typ in ("float64", "float32", "float16"):
        return float(v)
    if typ in ("int64", "int32", "int16", "int8", "uint64", "uint32"):
        return int(v)
    if typ == "bool_":
        return bool(v)
    return v

# Default weights for sharp books (mirrors db.py)
_SHARP_DEFAULT_WEIGHTS: Dict[str, float] = {
    'pinnacle': 1.00,
    'circa':    0.90,
    'bookmaker': 0.82,
}


def normalize_book(book: Optional[str]) -> str:
    if not book:
        return ''
    return str(book).strip().lower()


# ---------------------------------------------------------------------------
# Thin wrapper so code using conn.execute() works identically to SQLite
# ---------------------------------------------------------------------------

class _PgCursor:
    """Wraps a psycopg2 cursor to provide SQLite-compatible '?' placeholder support."""

    def __init__(self, raw_cursor):
        self._cursor = raw_cursor

    def execute(self, sql: str, params=()):
        if '?' in sql:
            sql = sql.replace('?', '%s')
        self._cursor.execute(sql, tuple(_to_py(p) for p in params))
        return self

    def executemany(self, sql: str, params_list):
        if '?' in sql:
            sql = sql.replace('?', '%s')
        self._cursor.executemany(sql, [tuple(_to_py(p) for p in row) for row in params_list])
        return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def __iter__(self):
        return iter(self._cursor)

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _PgConn:
    """Wraps a psycopg2 connection to provide a SQLite-compatible .execute() shortcut."""

    def __init__(self, raw_conn):
        self._conn = raw_conn

    def execute(self, sql: str, params=()):
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    def executemany(self, sql: str, params_list):
        cur = self.cursor()
        cur.executemany(sql, params_list)
        return cur

    def cursor(self):
        from psycopg2.extras import DictCursor
        return _PgCursor(self._conn.cursor(cursor_factory=DictCursor))


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class PostgresDatabaseClient:
    """
    Drop-in replacement for DatabaseClient using PostgreSQL / TimescaleDB.

    Args:
        dsn: PostgreSQL DSN string, e.g.
             "postgresql://user:password@host:5432/dbname"
             Defaults to the PG_DSN environment variable.
    """

    def __init__(self, dsn: str = None):
        self._dsn = dsn or os.getenv("PG_DSN")
        if not self._dsn:
            raise ValueError("PG_DSN must be set (env var or constructor arg)")

        # ThreadedConnectionPool
        self._pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=self._dsn,
        )
        self._init_db()

    @contextmanager
    def get_conn(self):
        """
        Context manager: yields a _PgConn wrapping a connection from the pool.
        Commits on success, rolls back on exception, always returns to pool.
        """
        conn = self._pool.getconn()
        try:
            yield _PgConn(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _init_db(self):
        schema_path = os.path.join(os.path.dirname(__file__), 'schema_postgres.sql')
        if not os.path.exists(schema_path):
            logger.warning(f"PostgreSQL schema not found at {schema_path}")
            return

        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        # Strip all single-line comments (-- ...) before splitting by semicolon
        # to avoid splitting on semicolons found inside comments.
        import re
        schema_sql = re.sub(r'--.*', '', schema_sql)

        # Run in autocommit mode — PostgreSQL DDL is transactional but
        # TimescaleDB create_hypertable requires its own transaction context.
        raw = self._pool.getconn()
        try:
            raw.autocommit = True
            cur = raw.cursor()
            
            for stmt in schema_sql.split(';'):
                clean_stmt = stmt.strip()
                if clean_stmt:
                    try:
                        cur.execute(clean_stmt)
                    except Exception as e:
                        logger.error(f"Schema stmt FAILED ({e}): {clean_stmt[:60]}")

            # Enable TimescaleDB extension and hypertable (idempotent)
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
                cur.execute(
                    "SELECT create_hypertable('line_history', 'timestamp', "
                    "chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE)"
                )
                cur.execute(
                    "SELECT add_retention_policy('line_history', INTERVAL '30 days', "
                    "if_not_exists => TRUE)"
                )
                logger.info("TimescaleDB hypertable configured for line_history.")
            except Exception as e:
                logger.warning(f"TimescaleDB setup skipped (may not be installed): {e}")

            # ── Migrations: add columns that may be missing on older DBs ──
            _migrations = [
                "ALTER TABLE games ADD COLUMN IF NOT EXISTS sr_id TEXT",
                "ALTER TABLE line_history ADD COLUMN IF NOT EXISTS game_id TEXT",
            ]
            for _m in _migrations:
                try:
                    cur.execute(_m)
                except Exception as _mig_err:
                    logger.debug(f"Migration skipped: {_mig_err}")

            # Indexes for new columns (safe to repeat)
            for _idx in [
                "CREATE INDEX IF NOT EXISTS idx_games_sr_id ON games(sr_id)",
                "CREATE INDEX IF NOT EXISTS idx_line_history_game ON line_history(game_id, timestamp DESC)",
            ]:
                try:
                    cur.execute(_idx)
                except Exception:
                    pass

            # Reset sequences so they stay ahead of existing max IDs.
            # Prevents duplicate key errors after data was inserted outside
            # the normal sequence path (e.g. manual inserts or migrations).
            for table, col in [('alerts_sent', 'id'), ('clv_tracking', 'id')]:
                try:
                    cur.execute(
                        f"SELECT setval("
                        f"pg_get_serial_sequence('{table}', '{col}'), "
                        f"COALESCE((SELECT MAX({col}) FROM {table}), 0) + 1, false)"
                    )
                except Exception as _seq_err:
                    logger.debug(f"Sequence reset skipped for {table}.{col}: {_seq_err}")

            logger.info("PostgreSQL schema initialized.")
        finally:
            raw.autocommit = False
            self._pool.putconn(raw)

    # ------------------------------------------------------------------
    # Alert management
    # ------------------------------------------------------------------

    def insert_alert(self, player_name: str, market: str, line: float, side: str,
                     edge: float, book: str, odds: float, stake: float = 0.0,
                     game_date: str = None, event_id: str = None,
                     home_away: str = None, rest_days: int = 2,
                     raw_model_prob: float = None, model_prob: float = None) -> int:
        book = normalize_book(book)
        line, edge, odds, stake = _to_py(line), _to_py(edge), _to_py(odds), _to_py(stake)
        raw_model_prob = _to_py(raw_model_prob) if raw_model_prob is not None else None
        model_prob = _to_py(model_prob) if model_prob is not None else None
        with self.get_conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO alerts_sent
                    (player_name, market, line, side, edge, book, odds, stake,
                     game_date, event_id, home_away, rest_days, raw_model_prob, model_prob)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (player_name, market, line, side, edge, book, odds, stake,
                 game_date, event_id, home_away, rest_days, raw_model_prob, model_prob),
            )
            alert_id = cur.fetchone()['id']

            conn.execute(
                """
                INSERT INTO clv_tracking (alert_id, player_id, market, side, alert_odds, alert_time)
                VALUES (%s, %s, %s, %s, %s, NOW())
                """,
                (alert_id, player_name, market, side, odds),
            )
            return alert_id

    def insert_alert_with_limits(self, player_name: str, market: str, line: float, side: str,
                                 edge: float, book: str, odds: float, stake: float,
                                 game_date: str = None, event_id: str = None,
                                 home_away: str = None, rest_days: int = 2,
                                 max_daily_risk: float = 1000.0,
                                 max_per_game_risk: float = 400.0,
                                 raw_model_prob: float = None,
                                 model_prob: float = None) -> Optional[int]:
        """Atomic check-and-insert for PostgreSQL."""
        line, edge, odds, stake = _to_py(line), _to_py(edge), _to_py(odds), _to_py(stake)
        raw_model_prob = _to_py(raw_model_prob) if raw_model_prob is not None else None
        model_prob = _to_py(model_prob) if model_prob is not None else None
        with self.get_conn() as conn:
            cur = conn.execute(
                """
                WITH daily_risk AS (
                    SELECT COALESCE(SUM(stake), 0) as total
                    FROM alerts_sent
                    WHERE timestamp >= NOW() - INTERVAL '12 hours'
                ),
                game_risk AS (
                    SELECT COALESCE(SUM(stake), 0) as total
                    FROM alerts_sent
                    WHERE event_id = %s AND timestamp >= NOW() - INTERVAL '12 hours'
                )
                INSERT INTO alerts_sent
                    (player_name, market, line, side, edge, book, odds, stake,
                     game_date, event_id, home_away, rest_days, raw_model_prob, model_prob)
                SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                WHERE (SELECT total FROM daily_risk) + %s <= %s
                  AND (SELECT total FROM game_risk) + %s <= %s
                RETURNING id
                """,
                (event_id, player_name, market, line, side, edge, book, odds, stake,
                 game_date, event_id, home_away, rest_days, raw_model_prob, model_prob,
                 stake, max_daily_risk,
                 stake, max_per_game_risk)
            )
            row = cur.fetchone()
            return row['id'] if row else None

    def check_recent_alert(self, player_name: str, market: str, line: float,
                           side: str, edge: float) -> bool:
        with self.get_conn() as conn:
            cur = conn.execute(
                """
                SELECT edge FROM alerts_sent
                WHERE player_name = %s AND market = %s AND side = %s
                  AND abs(line - %s) <= 0.5
                  AND timestamp::date = CURRENT_DATE
                ORDER BY timestamp DESC LIMIT 1
                """,
                (player_name, market, side, line),
            )
            row = cur.fetchone()
        if not row:
            return False

        return True  # Already alerted on this prop today — block duplicate

    # ------------------------------------------------------------------
    # CLV tracking
    # ------------------------------------------------------------------

    def get_unsettled_clv(self):
        with self.get_conn() as conn:
            cur = conn.execute("SELECT * FROM clv_tracking WHERE closing_odds IS NULL")
            return [dict(r) for r in cur.fetchall()]

    def update_clv_closing_line(self, track_id: int, closing_odds: float,
                                implied_closing: float, implied_alert: float):
        clv = implied_closing - implied_alert
        with self.get_conn() as conn:
            conn.execute(
                """
                UPDATE clv_tracking
                SET closing_odds = %s, implied_closing = %s, implied_alert = %s,
                    closing_time = NOW(), clv = %s
                WHERE id = %s
                """,
                (closing_odds, implied_closing, implied_alert, clv, track_id),
            )

    def get_per_market_clv(self, days_back: int = 30, min_samples: int = 10) -> Dict[str, float]:
        try:
            with self.get_conn() as conn:
                cur = conn.execute(
                    """
                    SELECT market, AVG(clv) AS avg_clv, COUNT(*) AS n
                    FROM clv_tracking
                    WHERE clv IS NOT NULL
                      AND alert_time >= NOW() - %s::interval
                    GROUP BY market
                    """,
                    (f'{days_back} days',),
                )
                rows = cur.fetchall()
            return {
                r['market']: float(r['avg_clv'])
                for r in rows
                if r['n'] and int(r['n']) >= min_samples
            }
        except Exception:
            return {}

    def get_clv_beat_rate(self, days_back: int = 30, min_samples: int = 10) -> Optional[float]:
        try:
            with self.get_conn() as conn:
                cur = conn.execute(
                    """
                    SELECT COUNT(*) AS total,
                           SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) AS beats
                    FROM clv_tracking
                    WHERE clv IS NOT NULL
                      AND alert_time >= NOW() - %s::interval
                    """,
                    (f'{days_back} days',),
                )
                row = cur.fetchone()
            if row and row['total'] and row['total'] >= min_samples:
                return float(row['beats']) / float(row['total'])
        except Exception:
            pass
        return None

    def get_avg_clv(self, days_back: int = 30) -> float:
        try:
            with self.get_conn() as conn:
                cur = conn.execute(
                    """
                    SELECT AVG(clv) AS avg_clv, COUNT(*) AS count
                    FROM clv_tracking
                    WHERE clv IS NOT NULL
                      AND alert_time >= NOW() - %s::interval
                    """,
                    (f'{days_back} days',),
                )
                row = cur.fetchone()
            if row and row['count'] and row['count'] >= 10:
                return float(row['avg_clv'] or 0.0)
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Line history
    # ------------------------------------------------------------------

    def insert_line_history(self, player_name: str, market: str, bookmaker: str,
                            line: float, side: str, odds: float, implied_prob: float):
        line, odds, implied_prob = _to_py(line), _to_py(odds), _to_py(implied_prob)
        with self.get_conn() as conn:
            conn.execute(
                """
                INSERT INTO line_history (player_name, market, bookmaker, line, side, odds, implied_prob)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (player_name, market, bookmaker, line, side, odds, implied_prob),
            )

    def insert_line_history_batch(self, records: list):
        with self.get_conn() as conn:
            conn.executemany(
                """
                INSERT INTO line_history (player_name, market, bookmaker, line, side, odds, implied_prob)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                records,
            )

    def get_market_metrics(self, player_name: str, market: str, line: float, side: str) -> dict:
        import pandas as pd
        raw = self._pool.getconn()
        try:
            from psycopg2.extras import DictCursor
            cur = raw.cursor(cursor_factory=DictCursor)
            cur.execute(
                """
                SELECT bookmaker, implied_prob, timestamp
                FROM line_history
                WHERE player_name = %s AND market = %s AND line = %s AND side = %s
                  AND timestamp >= NOW() - INTERVAL '60 minutes'
                ORDER BY timestamp ASC
                """,
                (player_name, market, line, side),
            )
            rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=['bookmaker', 'implied_prob', 'timestamp']) if rows else pd.DataFrame()
        except Exception:
            df = pd.DataFrame()
        finally:
            self._pool.putconn(raw)

        if df.empty or len(df) < 2:
            return {"steam_flag": False, "velocity": 0.0, "dispersion": 0.0}

        latest = df.groupby('bookmaker').last()
        dispersion = 0.0
        if len(latest) > 1:
            dispersion = float(latest['implied_prob'].std() or 0.0)

        first = df.groupby('bookmaker').first()
        changes = latest['implied_prob'] - first['implied_prob']
        velocity = float(changes.mean()) if not changes.empty else 0.0
        steam_flag = len(changes[changes > 0.02]) >= 3

        return {"steam_flag": steam_flag, "velocity": velocity, "dispersion": dispersion}

    def get_sharp_line_shift(self, player_name: str, market: str) -> dict:
        _empty = {'shift_detected': False}
        try:
            with self.get_conn() as conn:
                cur = conn.execute(
                    """
                    SELECT bookmaker, line, timestamp
                    FROM line_history
                    WHERE player_name = %s AND market = %s AND side = 'OVER'
                      AND LOWER(bookmaker) IN ('pinnacle', 'circa', 'bookmaker')
                      AND timestamp >= NOW() - INTERVAL '3 hours'
                    ORDER BY bookmaker, timestamp ASC
                    """,
                    (player_name, market),
                )
                rows = cur.fetchall()
        except Exception:
            return _empty

        if not rows:
            return _empty

        from collections import defaultdict
        by_book: dict = defaultdict(list)
        for row in rows:
            by_book[row['bookmaker']].append(float(row['line']))

        for book, lines in by_book.items():
            if len(lines) < 2:
                continue
            magnitude = lines[-1] - lines[0]
            if abs(magnitude) >= 1.0:
                return {
                    'shift_detected': True,
                    'sharp_book':     book,
                    'old_line':       lines[0],
                    'new_line':       lines[-1],
                    'direction':      'UP' if magnitude > 0 else 'DOWN',
                    'magnitude':      abs(magnitude),
                }
        return _empty

    # ------------------------------------------------------------------
    # Bookmaker profiles
    # ------------------------------------------------------------------

    def init_bookmaker_profiles(self):
        profiles = [
            ("pinnacle",    "sharp", _SHARP_DEFAULT_WEIGHTS['pinnacle']),
            ("circa",       "sharp", _SHARP_DEFAULT_WEIGHTS['circa']),
            ("bookmaker",   "sharp", _SHARP_DEFAULT_WEIGHTS['bookmaker']),
            ("draftkings",  "rec",   0.0),
            ("fanduel",     "rec",   0.0),
            ("betmgm",      "rec",   0.0),
            ("caesars",     "rec",   0.0),
            ("bovada",      "rec",   0.0),
            ("betrivers",   "rec",   0.0),
            ("pointsbetus", "rec",   0.0),
        ]
        with self.get_conn() as conn:
            for book, role, default_clv in profiles:
                conn.execute(
                    """
                    INSERT INTO bookmaker_profiles (bookmaker, role, historical_clv_score)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (bookmaker) DO NOTHING
                    """,
                    (book, role, default_clv),
                )

    def get_bookmaker_role(self, bookmaker: str) -> str:
        with self.get_conn() as conn:
            cur = conn.execute(
                "SELECT role FROM bookmaker_profiles WHERE LOWER(bookmaker) = LOWER(%s)",
                (bookmaker,),
            )
            row = cur.fetchone()
        return row['role'] if row else "neutral"

    def get_sharp_book_weights(self) -> Dict[str, float]:
        try:
            with self.get_conn() as conn:
                cur = conn.execute(
                    "SELECT bookmaker, historical_clv_score FROM bookmaker_profiles WHERE role = 'sharp'"
                )
                rows = cur.fetchall()
            if not rows:
                return dict(_SHARP_DEFAULT_WEIGHTS)
            weights: Dict[str, float] = {}
            for row in rows:
                book = str(row['bookmaker']).lower()
                score = float(row['historical_clv_score'] or 0.0)
                weights[book] = score if score > 0.10 else _SHARP_DEFAULT_WEIGHTS.get(book, 0.75)
            max_w = max(weights.values()) if weights else 1.0
            if max_w > 0:
                return {k: v / max_w for k, v in weights.items()}
        except Exception:
            pass
        return dict(_SHARP_DEFAULT_WEIGHTS)

    def update_sharp_book_clv_score(self, book: str, brier_delta: float) -> None:
        try:
            with self.get_conn() as conn:
                cur = conn.execute(
                    "SELECT historical_clv_score FROM bookmaker_profiles WHERE LOWER(bookmaker) = LOWER(%s)",
                    (book,),
                )
                row = cur.fetchone()
                if row is None:
                    return
                current = float(row['historical_clv_score'] or 0.0)
                if current <= 0.0:
                    current = _SHARP_DEFAULT_WEIGHTS.get(book.lower(), 0.75)
                new_score = max(0.10, min(2.00, current * 0.95 + brier_delta * 0.05))
                conn.execute(
                    "UPDATE bookmaker_profiles SET historical_clv_score = %s WHERE LOWER(bookmaker) = LOWER(%s)",
                    (new_score, book),
                )
        except Exception:
            pass

    def get_book_market_bias(self, book: str, market: str) -> float:
        try:
            with self.get_conn() as conn:
                cur = conn.execute(
                    """
                    SELECT AVG(CAST(b.won AS REAL)) AS win_rate, COUNT(*) AS count
                    FROM alerts_sent a
                    JOIN bet_results b ON a.id = b.alert_id
                    WHERE a.book = %s AND a.market = %s
                    """,
                    (book, market),
                )
                row = cur.fetchone()
            if row and row['count'] and row['count'] >= 20:
                win_rate = float(row['win_rate'] or 0.5)
                bias = (win_rate - 0.52) / 0.10
                return float(max(0.85, min(1.15, 1.0 + bias * 0.15)))
        except Exception:
            pass
        return 1.0

    # ------------------------------------------------------------------
    # Team / opponent stats
    # ------------------------------------------------------------------

    def upsert_team_opponent_stats(self, team_name: str, season: str,
                                   opp_pts: float, opp_reb: float, opp_ast: float,
                                   opp_fg3m: float, pace: float, def_rating: float):
        with self.get_conn() as conn:
            conn.execute(
                """
                INSERT INTO team_opponent_stats
                    (team_name, season, opp_pts_pg, opp_reb_pg, opp_ast_pg,
                     opp_fg3m_pg, pace, def_rating, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE)
                ON CONFLICT (team_name, season) DO UPDATE SET
                    opp_pts_pg   = EXCLUDED.opp_pts_pg,
                    opp_reb_pg   = EXCLUDED.opp_reb_pg,
                    opp_ast_pg   = EXCLUDED.opp_ast_pg,
                    opp_fg3m_pg  = EXCLUDED.opp_fg3m_pg,
                    pace         = EXCLUDED.pace,
                    def_rating   = EXCLUDED.def_rating,
                    last_updated = EXCLUDED.last_updated
                """,
                (team_name, season, opp_pts, opp_reb, opp_ast, opp_fg3m, pace, def_rating),
            )

    def get_team_opponent_stats(self, team_name: str, season: str) -> dict:
        with self.get_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM team_opponent_stats WHERE team_name = %s AND season = %s",
                (team_name, season),
            )
            row = cur.fetchone()
        return dict(row) if row else {}

    # ------------------------------------------------------------------
    # On/off splits
    # ------------------------------------------------------------------

    def get_on_off_split(self, player_id: int, absent_player_id: int,
                         market: str, season: str):
        with self.get_conn() as conn:
            cur = conn.execute(
                """
                SELECT * FROM on_off_splits
                WHERE player_id = %s AND absent_player_id = %s
                  AND market = %s AND season = %s
                """,
                (player_id, absent_player_id, market, season),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def upsert_on_off_split(self, player_id: int, absent_player_id: int,
                             market: str, season: str,
                             games_processed: int, minutes_with: float,
                             minutes_without: float, rate_with: float,
                             rate_without: float, usage_multiplier: float):
        with self.get_conn() as conn:
            conn.execute(
                """
                INSERT INTO on_off_splits
                    (player_id, absent_player_id, season, market,
                     games_processed, minutes_with, minutes_without,
                     rate_with, rate_without, usage_multiplier, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE::text)
                ON CONFLICT (player_id, absent_player_id, season, market) DO UPDATE SET
                    games_processed  = EXCLUDED.games_processed,
                    minutes_with     = EXCLUDED.minutes_with,
                    minutes_without  = EXCLUDED.minutes_without,
                    rate_with        = EXCLUDED.rate_with,
                    rate_without     = EXCLUDED.rate_without,
                    usage_multiplier = EXCLUDED.usage_multiplier,
                    last_updated     = EXCLUDED.last_updated
                """,
                (player_id, absent_player_id, season, market,
                 games_processed, minutes_with, minutes_without,
                 rate_with, rate_without, usage_multiplier),
            )

    def get_on_off_rate_without(self, player_id: int, absent_player_id: int,
                                 market: str, season: str) -> Optional[float]:
        with self.get_conn() as conn:
            cur = conn.execute(
                """
                SELECT rate_without, minutes_without FROM on_off_splits
                WHERE player_id = %s AND absent_player_id = %s
                  AND market = %s AND season = %s
                """,
                (player_id, absent_player_id, market, season),
            )
            row = cur.fetchone()
        if row and row['minutes_without'] >= 10.0:
            return float(row['rate_without'])
        return None

    # ------------------------------------------------------------------
    # Player role helpers
    # ------------------------------------------------------------------

    def get_team_initiator_ids(self, team_id: int) -> list:
        with self.get_conn() as conn:
            cur = conn.execute(
                "SELECT player_id FROM players WHERE team_id = %s AND is_primary_initiator = TRUE",
                (team_id,),
            )
            return [r['player_id'] for r in cur.fetchall()]

    def is_primary_initiator(self, player_id: int) -> bool:
        with self.get_conn() as conn:
            cur = conn.execute(
                "SELECT is_primary_initiator FROM players WHERE player_id = %s",
                (player_id,),
            )
            row = cur.fetchone()
        return bool(row and row['is_primary_initiator'])

    def set_primary_initiators(self, team_id: int, player_ids: list):
        with self.get_conn() as conn:
            conn.execute(
                "UPDATE players SET is_primary_initiator = FALSE WHERE team_id = %s",
                (team_id,),
            )
            for pid in player_ids:
                conn.execute(
                    "UPDATE players SET is_primary_initiator = TRUE WHERE player_id = %s AND team_id = %s",
                    (pid, team_id),
                )

    # ------------------------------------------------------------------
    # Rotation slots
    # ------------------------------------------------------------------

    def get_rotation_slots(self, team_abbr: str, season: str) -> dict:
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        with self.get_conn() as conn:
            cur = conn.execute(
                "SELECT player_id, slot_key, slot_probability, last_updated "
                "FROM rotation_slots WHERE team_abbr = %s AND season = %s",
                (team_abbr, season),
            )
            rows = cur.fetchall()
        if not rows or rows[0]['last_updated'] != today:
            return {}
        result: dict = {}
        for row in rows:
            pid = int(row['player_id'])
            result.setdefault(pid, {})[row['slot_key']] = float(row['slot_probability'])
        return result

    def upsert_rotation_slots(self, team_abbr: str, team_slots: dict,
                               games_processed: int, season: str, last_updated: str):
        with self.get_conn() as conn:
            conn.execute(
                "DELETE FROM rotation_slots WHERE team_abbr = %s AND season = %s",
                (team_abbr, season),
            )
            for player_id, slots in team_slots.items():
                for slot_key, prob in slots.items():
                    conn.execute(
                        """
                        INSERT INTO rotation_slots
                            (team_abbr, player_id, season, slot_key,
                             slot_probability, games_processed, last_updated)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (team_abbr, int(player_id), season, slot_key,
                         float(prob), games_processed, last_updated),
                    )

    # ------------------------------------------------------------------
    # SGP correlations
    # ------------------------------------------------------------------

    def upsert_sgp_correlation(self, player_name: str, market_a: str,
                                market_b: str, correlation: float, sample_size: int):
        with self.get_conn() as conn:
            conn.execute(
                """
                INSERT INTO sgp_correlations
                    (player_name, market_a, market_b, correlation, sample_size, last_updated)
                VALUES (%s, %s, %s, %s, %s, CURRENT_DATE)
                ON CONFLICT (player_name, market_a, market_b) DO UPDATE SET
                    correlation  = EXCLUDED.correlation,
                    sample_size  = EXCLUDED.sample_size,
                    last_updated = EXCLUDED.last_updated
                """,
                (player_name, market_a, market_b, correlation, sample_size),
            )

    def get_player_sgp_correlation(self, player_name: str, market_a: str,
                                    market_b: str, min_samples: int = 15) -> tuple:
        key_pairs = [(market_a, market_b), (market_b, market_a)]
        try:
            with self.get_conn() as conn:
                for ma, mb in key_pairs:
                    cur = conn.execute(
                        """
                        SELECT correlation, sample_size FROM sgp_correlations
                        WHERE player_name = %s AND market_a = %s AND market_b = %s
                          AND sample_size >= %s
                        """,
                        (player_name, ma, mb, min_samples),
                    )
                    row = cur.fetchone()
                    if row:
                        return (float(row['correlation']), int(row['sample_size']))
        except Exception:
            pass
        return (None, 0)

    # ------------------------------------------------------------------
    # Cross-player / cross-team correlations
    # ------------------------------------------------------------------

    def upsert_cross_player_correlation(self, team: str, player_a: str,
                                         player_b: str, market_a: str,
                                         market_b: str, correlation: float,
                                         n_games: int) -> None:
        with self.get_conn() as conn:
            conn.execute(
                """
                INSERT INTO cross_player_correlations
                    (team, player_a, player_b, market_a, market_b,
                     correlation, n_games, computed_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_DATE::text)
                ON CONFLICT (team, player_a, player_b, market_a, market_b) DO UPDATE SET
                    correlation   = EXCLUDED.correlation,
                    n_games       = EXCLUDED.n_games,
                    computed_date = EXCLUDED.computed_date
                """,
                (team, player_a, player_b, market_a, market_b, correlation, n_games),
            )

    def get_cross_player_correlation(self, team: str, player_a: str,
                                      player_b: str, market_a: str,
                                      market_b: str) -> Optional[float]:
        with self.get_conn() as conn:
            cur = conn.execute(
                """
                SELECT correlation FROM cross_player_correlations
                WHERE team = %s AND player_a = %s AND player_b = %s
                  AND market_a = %s AND market_b = %s
                  AND computed_date::date >= CURRENT_DATE - INTERVAL '7 days'
                """,
                (team, player_a, player_b, market_a, market_b),
            )
            row = cur.fetchone()
        return float(row['correlation']) if row else None

    def upsert_cross_team_correlation(self, matchup: str, player_a: str,
                                       player_b: str, market_a: str,
                                       market_b: str, correlation: float,
                                       n_games: int) -> None:
        with self.get_conn() as conn:
            conn.execute(
                """
                INSERT INTO cross_team_correlations
                    (matchup, player_a, player_b, market_a, market_b,
                     correlation, n_games, computed_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_DATE::text)
                ON CONFLICT (matchup, player_a, player_b, market_a, market_b) DO UPDATE SET
                    correlation   = EXCLUDED.correlation,
                    n_games       = EXCLUDED.n_games,
                    computed_date = EXCLUDED.computed_date
                """,
                (matchup, player_a, player_b, market_a, market_b, correlation, n_games),
            )

    def get_cross_team_correlation(self, matchup: str, player_a: str,
                                    player_b: str, market_a: str,
                                    market_b: str) -> Optional[float]:
        with self.get_conn() as conn:
            cur = conn.execute(
                """
                SELECT correlation FROM cross_team_correlations
                WHERE matchup = %s AND player_a = %s AND player_b = %s
                  AND market_a = %s AND market_b = %s
                  AND computed_date::date >= CURRENT_DATE - INTERVAL '7 days'
                """,
                (matchup, player_a, player_b, market_a, market_b),
            )
            row = cur.fetchone()
        return float(row['correlation']) if row else None

    def get_outcome_correlation(self, player_name: str, market_a: str,
                                 market_b: str, min_samples: int = 30) -> tuple:
        try:
            with self.get_conn() as conn:
                cur = conn.execute(
                    """
                    SELECT
                        CAST(br1.won AS INTEGER) AS won_a,
                        CAST(br2.won AS INTEGER) AS won_b
                    FROM alerts_sent s1
                    JOIN bet_results br1
                        ON br1.alert_id = s1.id AND br1.push = FALSE AND br1.won IS NOT NULL
                    JOIN alerts_sent s2
                        ON  s2.player_name = s1.player_name
                        AND s2.game_date   = s1.game_date
                        AND s2.market      = %s
                        AND s2.id          > s1.id
                    JOIN bet_results br2
                        ON br2.alert_id = s2.id AND br2.push = FALSE AND br2.won IS NOT NULL
                    WHERE s1.player_name = %s AND s1.market = %s
                    """,
                    (market_b, player_name, market_a),
                )
                rows = cur.fetchall()
        except Exception:
            return (None, 0)

        n = len(rows)
        if n < min_samples:
            return (None, 0)

        import math
        hits_a = [r['won_a'] for r in rows]
        hits_b = [r['won_b'] for r in rows]
        p_a   = sum(hits_a) / n
        p_b   = sum(hits_b) / n
        p_ab  = sum(a * b for a, b in zip(hits_a, hits_b)) / n
        denom = math.sqrt(p_a * (1 - p_a) * p_b * (1 - p_b))
        if denom < 1e-9:
            return (None, 0)
        return (float((p_ab - p_a * p_b) / denom), n)

    # ------------------------------------------------------------------
    # Steam detection
    # ------------------------------------------------------------------

    def detect_steam_moves(self, window_minutes: int = 120,
                            sharp_move_threshold: float = 0.04,
                            stale_threshold: float = 0.01,
                            sharp_books: tuple = ('pinnacle', 'circa', 'bookmaker'),
                            soft_books: tuple = ('draftkings', 'fanduel', 'betmgm', 'caesars'),
                            ) -> list:
        import pandas as pd
        import warnings
        raw = self._pool.getconn()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
                df = pd.read_sql_query(
                """
                SELECT player_name, market, side, line,
                       bookmaker, odds, implied_prob, timestamp
                FROM line_history
                WHERE timestamp >= NOW() - %s::interval
                ORDER BY timestamp ASC
                """,
                raw,
                params=(f'{window_minutes} minutes',),
            )
        except Exception:
            df = pd.DataFrame()
        finally:
            self._pool.putconn(raw)

        if df.empty or len(df) < 4:
            return []

        df['bookmaker'] = df['bookmaker'].str.lower().str.strip()
        moves = []

        for (player, market, line, side), prop_df in df.groupby(
            ['player_name', 'market', 'line', 'side'], sort=False
        ):
            book_stats: dict = {}
            for book, book_df in prop_df.groupby('bookmaker', sort=False):
                book_df = book_df.sort_values('timestamp')
                n = len(book_df)
                first_prob = float(book_df.iloc[0]['implied_prob'])
                last_prob  = float(book_df.iloc[-1]['implied_prob'])
                last_odds  = float(book_df.iloc[-1]['odds'])
                delta      = last_prob - first_prob
                elapsed    = 0.0
                if n > 1:
                    try:
                        elapsed = (
                            pd.to_datetime(book_df.iloc[-1]['timestamp'])
                            - pd.to_datetime(book_df.iloc[0]['timestamp'])
                        ).total_seconds() / 60.0
                    except Exception:
                        elapsed = 0.0
                book_stats[book] = {
                    'first_prob': first_prob, 'last_prob': last_prob,
                    'last_odds': last_odds, 'delta': delta,
                    'elapsed': elapsed, 'n_obs': n,
                }

            sharp_moves = {
                b: s for b, s in book_stats.items()
                if b in sharp_books and s['n_obs'] >= 2
                and abs(s['delta']) >= sharp_move_threshold
            }
            if not sharp_moves:
                continue

            stale_softs = {
                b: s for b, s in book_stats.items()
                if b in soft_books and abs(s['delta']) <= stale_threshold
            }
            if not stale_softs:
                continue

            best_sharp = max(sharp_moves, key=lambda b: abs(sharp_moves[b]['delta']))
            sharp_info = sharp_moves[best_sharp]
            direction  = 'OVER' if sharp_info['delta'] > 0 else 'UNDER'
            best_stale = min(stale_softs, key=lambda b: stale_softs[b]['last_prob'])
            stale_info = stale_softs[best_stale]

            moves.append({
                'player_name':        player,
                'market':             market,
                'side':               side,
                'line':               line,
                'sharp_book':         best_sharp,
                'sharp_delta':        round(sharp_info['delta'], 4),
                'sharp_first_prob':   round(sharp_info['first_prob'], 4),
                'sharp_current_prob': round(sharp_info['last_prob'], 4),
                'stale_book':         best_stale,
                'stale_odds':         stale_info['last_odds'],
                'stale_current_prob': round(stale_info['last_prob'], 4),
                'direction':          direction,
                'elapsed_minutes':    round(sharp_info['elapsed'], 1),
            })

        return moves

    def check_recent_steam_alert(self, player_name: str, market: str,
                                  side: str, minutes: int = 30) -> bool:
        with self.get_conn() as conn:
            cur = conn.execute(
                """
                SELECT id FROM steam_alerts
                WHERE player_name = %s AND market = %s AND side = %s
                  AND timestamp >= NOW() - %s::interval
                """,
                (player_name, market, side, f'{minutes} minutes'),
            )
            return cur.fetchone() is not None

    def insert_steam_alert(self, player_name: str, market: str, side: str,
                            line: float, sharp_book: str, sharp_delta: float,
                            sharp_current_prob: float, stale_book: str,
                            stale_odds: float, stale_current_prob: float,
                            direction: str) -> None:
        line = _to_py(line)
        sharp_delta, sharp_current_prob = _to_py(sharp_delta), _to_py(sharp_current_prob)
        stale_odds, stale_current_prob  = _to_py(stale_odds),  _to_py(stale_current_prob)
        with self.get_conn() as conn:
            conn.execute(
                """
                INSERT INTO steam_alerts
                    (player_name, market, side, line, sharp_book, sharp_delta,
                     sharp_current_prob, stale_book, stale_odds, stale_current_prob, direction)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (player_name, market, side, line, sharp_book, sharp_delta,
                 sharp_current_prob, stale_book, stale_odds, stale_current_prob, direction),
            )

    # ------------------------------------------------------------------
    # Referee stats
    # ------------------------------------------------------------------

    def get_referee_foul_rates(self, ref_names: list) -> list:
        if not ref_names:
            return []
        try:
            with self.get_conn() as conn:
                cur = conn.execute(
                    "SELECT avg_pfd_per_game FROM referee_stats "
                    "WHERE referee_name = ANY(%s) AND avg_pfd_per_game > 0",
                    (ref_names,),
                )
                return [float(r['avg_pfd_per_game']) for r in cur.fetchall()]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Two-tier alert batching
    # ------------------------------------------------------------------

    def queue_pending_alert(self, alert_type: str, title: str, body: str,
                             priority: float = 0.0, game_date: str = None) -> int:
        priority = _to_py(priority)
        with self.get_conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO pending_alerts (alert_type, title, body, priority, game_date)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (alert_type, title, body, priority, game_date),
            )
            return cur.fetchone()['id']

    def get_pending_alerts(self, unsent_only: bool = True) -> list:
        where = "WHERE sent_at IS NULL" if unsent_only else ""
        with self.get_conn() as conn:
            cur = conn.execute(
                f"""
                SELECT id, alert_type, title, body, priority, game_date, created_at
                FROM pending_alerts
                {where}
                ORDER BY alert_type, priority DESC
                """
            )
            return [dict(r) for r in cur.fetchall()]

    def mark_pending_alerts_sent(self, ids: list) -> None:
        if not ids:
            return
        with self.get_conn() as conn:
            conn.execute(
                "UPDATE pending_alerts SET sent_at = NOW() WHERE id = ANY(%s)",
                (ids,),
            )

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def insert_backtest_results_batch(self, records: list) -> None:
        with self.get_conn() as conn:
            conn.executemany(
                """
                INSERT INTO backtest_results
                    (player_name, season, market, game_date, simulated_line,
                     model_mean, model_prob_over, actual_stat, hit, edge)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                records,
            )

    def get_backtest_summary(self) -> dict:
        with self.get_conn() as conn:
            mkt_cur = conn.execute(
                """
                SELECT market, COUNT(*) AS bets, SUM(hit) AS hits,
                       AVG(hit) AS hit_rate,
                       SUM(CASE WHEN hit=1 THEN 0.909 ELSE -1.0 END) AS profit
                FROM backtest_results GROUP BY market ORDER BY market
                """
            )
            bucket_cur = conn.execute(
                """
                SELECT
                    CASE WHEN edge < 0.03 THEN '0-3%'
                         WHEN edge < 0.06 THEN '3-6%'
                         WHEN edge < 0.10 THEN '6-10%'
                         ELSE '10%+' END AS bucket,
                    COUNT(*) AS bets, SUM(hit) AS hits, AVG(hit) AS hit_rate
                FROM backtest_results WHERE edge > 0
                GROUP BY bucket ORDER BY bucket
                """
            )
            cal_cur = conn.execute(
                """
                SELECT ROUND(model_prob_over / 0.05) * 0.05 AS prob_bin,
                       AVG(hit) AS actual_hit_rate, COUNT(*) AS n
                FROM backtest_results GROUP BY prob_bin ORDER BY prob_bin
                """
            )
            return {
                'markets':     [dict(r) for r in mkt_cur.fetchall()],
                'buckets':     [dict(r) for r in bucket_cur.fetchall()],
                'calibration': [dict(r) for r in cal_cur.fetchall()],
            }

    # ------------------------------------------------------------------
    # BDL defense profiles
    # ------------------------------------------------------------------

    def upsert_bdl_defense_profile(self, team_key: str, season: int, profile: dict):
        with self.get_conn() as conn:
            conn.execute(
                """
                INSERT INTO bdl_defense_profiles
                    (team_key, season, opp_pts, opp_reb, opp_ast, opp_fg3m,
                     opp_fta, opp_pts_paint, def_rating, pace, blk, stl, fetched_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (team_key, season) DO UPDATE SET
                    opp_pts       = EXCLUDED.opp_pts,
                    opp_reb       = EXCLUDED.opp_reb,
                    opp_ast       = EXCLUDED.opp_ast,
                    opp_fg3m      = EXCLUDED.opp_fg3m,
                    opp_fta       = EXCLUDED.opp_fta,
                    opp_pts_paint = EXCLUDED.opp_pts_paint,
                    def_rating    = EXCLUDED.def_rating,
                    pace          = EXCLUDED.pace,
                    blk           = EXCLUDED.blk,
                    stl           = EXCLUDED.stl,
                    fetched_at    = EXCLUDED.fetched_at
                """,
                (team_key.lower(), season,
                 profile.get('opp_pts', 1.0), profile.get('opp_reb', 1.0),
                 profile.get('opp_ast', 1.0), profile.get('opp_fg3m', 1.0),
                 profile.get('opp_fta', 1.0), profile.get('opp_pts_paint', 1.0),
                 profile.get('def_rating', 1.0), profile.get('pace', 1.0),
                 profile.get('blk', 1.0), profile.get('stl', 1.0),
                 profile.get('fetched_at', '')),
            )

    def get_bdl_defense_profile(self, team_key: str, season: int) -> Optional[dict]:
        with self.get_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM bdl_defense_profiles WHERE team_key = %s AND season = %s",
                (team_key.lower(), season),
            )
            row = cur.fetchone()
            if not row:
                cur = conn.execute(
                    "SELECT * FROM bdl_defense_profiles WHERE season = %s",
                    (season,),
                )
                low = team_key.lower()
                for r in cur.fetchall():
                    k = r['team_key']
                    if k in low or low in k:
                        row = r
                        break
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # BDL game log cache
    # ------------------------------------------------------------------

    def cache_bdl_game_logs(self, player_id: int, season: int, logs: list):
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        with self.get_conn() as conn:
            for log in logs:
                try:
                    conn.execute(
                        """
                        INSERT INTO bdl_game_log_cache
                            (player_id, season, game_date, game_id, min, pts, reb,
                             ast, fg3m, blk, stl, fga, fta, tov,
                             team_abbr, matchup, wl, cached_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (player_id, season, game_date) DO UPDATE SET
                            min       = EXCLUDED.min,
                            pts       = EXCLUDED.pts,
                            reb       = EXCLUDED.reb,
                            ast       = EXCLUDED.ast,
                            fg3m      = EXCLUDED.fg3m,
                            blk       = EXCLUDED.blk,
                            stl       = EXCLUDED.stl,
                            fga       = EXCLUDED.fga,
                            fta       = EXCLUDED.fta,
                            tov       = EXCLUDED.tov,
                            team_abbr = EXCLUDED.team_abbr,
                            matchup   = EXCLUDED.matchup,
                            wl        = EXCLUDED.wl,
                            cached_at = EXCLUDED.cached_at
                        """,
                        (player_id, season, log.get('game_date', ''),
                         log.get('game_id'), log.get('min', 0),
                         log.get('pts', 0), log.get('reb', 0),
                         log.get('ast', 0), log.get('fg3m', 0),
                         log.get('blk', 0), log.get('stl', 0),
                         log.get('fga', 0), log.get('fta', 0),
                         log.get('tov', 0), log.get('team_abbr', ''),
                         log.get('matchup', ''), log.get('wl', ''), now),
                    )
                except Exception:
                    pass

    def upsert_game(self, game_id: str, home_team: str, away_team: str,
                    commence_time: str, status: str) -> None:
        with self.get_conn() as conn:
            conn.execute(
                """
                INSERT INTO games (game_id, home_team, away_team, commence_time, status)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (game_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    commence_time = EXCLUDED.commence_time
                """,
                (game_id, home_team, away_team, commence_time, status),
            )

    def upsert_injury_report(self, game_date: str, player_name: str, team: str,
                             status: str, description: str, return_date: str,
                             severity: int, source: str, updated_at: str) -> None:
        with self.get_conn() as conn:
            conn.execute(
                """
                INSERT INTO injury_reports
                    (game_date, player_name, team, status, description,
                     return_date, severity, source, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (game_date, player_name) DO UPDATE SET
                    team        = EXCLUDED.team,
                    status      = EXCLUDED.status,
                    description = EXCLUDED.description,
                    return_date = EXCLUDED.return_date,
                    severity    = EXCLUDED.severity,
                    source      = EXCLUDED.source,
                    updated_at  = EXCLUDED.updated_at
                """,
                (game_date, player_name, team, status, description,
                 return_date, severity, source, updated_at),
            )

    def get_cached_bdl_game_logs(self, player_id: int, season: int,
                                  ignore_ttl: bool = False) -> list:
        from datetime import datetime, timedelta
        with self.get_conn() as conn:
            if ignore_ttl:
                cur = conn.execute(
                    """
                    SELECT * FROM bdl_game_log_cache
                    WHERE player_id = %s AND season = %s
                    ORDER BY game_date DESC
                    """,
                    (player_id, season),
                )
            else:
                cutoff = (datetime.utcnow() - timedelta(hours=12)).isoformat()
                cur = conn.execute(
                    """
                    SELECT * FROM bdl_game_log_cache
                    WHERE player_id = %s AND season = %s AND cached_at > %s
                    ORDER BY game_date DESC
                    """,
                    (player_id, season, cutoff),
                )
            rows = cur.fetchall()
        return [dict(r) for r in rows] if rows else []
