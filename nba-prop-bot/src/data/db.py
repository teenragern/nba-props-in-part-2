import sqlite3
import os
from contextlib import contextmanager
from src.config import DB_PATH
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class DatabaseClient:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.commit()
            conn.close()

    def _init_db(self):
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        if not os.path.exists(schema_path):
            logger.warning(f"Schema not found at {schema_path}")
            return

        with open(schema_path, 'r') as f:
            schema = f.read()

        with self.get_conn() as conn:
            conn.executescript(schema)
            self._migrate_schema(conn)
            logger.info("Database schema initialized.")

    def _migrate_schema(self, conn):
        """Add new columns to existing tables without breaking existing data."""
        migrations = [
            # alerts_sent columns
            "ALTER TABLE alerts_sent ADD COLUMN game_date TEXT",
            "ALTER TABLE alerts_sent ADD COLUMN event_id TEXT",
            "ALTER TABLE alerts_sent ADD COLUMN home_away TEXT",
            "ALTER TABLE alerts_sent ADD COLUMN rest_days INTEGER DEFAULT 2",
            # clv_tracking columns — fix "no such column: implied_closing"
            "ALTER TABLE clv_tracking ADD COLUMN implied_closing REAL",
            "ALTER TABLE clv_tracking ADD COLUMN implied_alert REAL",
            "ALTER TABLE clv_tracking ADD COLUMN clv REAL",
        ]
        for sql in migrations:
            try:
                conn.execute(sql)
            except Exception:
                pass  # Column already exists

    def insert_alert(self, player_name: str, market: str, line: float, side: str,
                     edge: float, book: str, odds: float, stake: float = 0.0,
                     game_date: str = None, event_id: str = None,
                     home_away: str = None, rest_days: int = 2) -> int:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO alerts_sent
                    (player_name, market, line, side, edge, book, odds, stake,
                     game_date, event_id, home_away, rest_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (player_name, market, line, side, edge, book, odds, stake,
                 game_date, event_id, home_away, rest_days)
            )
            alert_id = cursor.lastrowid

            # Phase 3 CLV Tracking Link
            cursor.execute(
                """
                INSERT INTO clv_tracking (player_id, market, side, alert_odds, alert_time)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (player_name, market, side, odds)
            )

            return alert_id

    def check_recent_alert(self, player_name: str, market: str, line: float, side: str, edge: float) -> bool:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT edge FROM alerts_sent
                WHERE player_name = ? AND market = ? AND side = ?
                AND abs(line - ?) <= 0.5
                AND date(timestamp) = date('now', 'localtime')
                ORDER BY timestamp DESC LIMIT 1
                """,
                (player_name, market, side, line)
            )
            row = cursor.fetchone()
            if not row:
                return False

            last_edge = row['edge']
            if edge - last_edge > 0.01:
                return False  # Allow re-alert: edge improved > 1%

            return True

    def get_unsettled_clv(self):
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM clv_tracking WHERE closing_odds IS NULL")
            return [dict(r) for r in cursor.fetchall()]

    def update_clv_closing_line(self, track_id: int, closing_odds: float, implied_closing: float, implied_alert: float):
        clv = implied_closing - implied_alert
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE clv_tracking
                SET closing_odds = ?, implied_closing = ?, implied_alert = ?,
                    closing_time = CURRENT_TIMESTAMP, clv = ?
                WHERE id = ?
                """,
                (closing_odds, implied_closing, implied_alert, clv, track_id)
            )

    def insert_line_history(self, player_name: str, market: str, bookmaker: str,
                            line: float, side: str, odds: float, implied_prob: float):
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO line_history (player_name, market, bookmaker, line, side, odds, implied_prob)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (player_name, market, bookmaker, line, side, odds, implied_prob)
            )

    def init_bookmaker_profiles(self):
        profiles = [
            ("pinnacle", "sharp"),
            ("circa", "sharp"),
            ("draftkings", "rec"),
            ("fanduel", "rec"),
            ("betmgm", "rec"),
            ("caesars", "rec"),
            ("bovada", "rec"),
            ("betrivers", "rec"),
            ("pointsbetus", "rec"),
        ]
        with self.get_conn() as conn:
            cursor = conn.cursor()
            for book, role in profiles:
                cursor.execute(
                    "INSERT OR IGNORE INTO bookmaker_profiles (bookmaker, role) VALUES (?, ?)",
                    (book, role)
                )

    def get_bookmaker_role(self, bookmaker: str) -> str:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT role FROM bookmaker_profiles WHERE bookmaker = ? COLLATE NOCASE", (bookmaker,))
            row = cursor.fetchone()
            return row['role'] if row else "neutral"

    def insert_line_history_batch(self, records: list):
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO line_history (player_name, market, bookmaker, line, side, odds, implied_prob)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                records
            )

    def get_market_metrics(self, player_name: str, market: str, line: float, side: str) -> dict:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT bookmaker, implied_prob, timestamp
                FROM line_history
                WHERE player_name = ? AND market = ? AND line = ? AND side = ?
                  AND timestamp >= datetime('now', '-60 minute')
                ORDER BY timestamp ASC
                """,
                (player_name, market, line, side)
            )
            rows = cursor.fetchall()

        if not rows:
            return {"steam_flag": False, "velocity": 0.0, "dispersion": 0.0}

        import pandas as pd
        df = pd.DataFrame(rows, columns=['bookmaker', 'implied_prob', 'timestamp'])

        if df.empty or len(df) < 2:
            return {"steam_flag": False, "velocity": 0.0, "dispersion": 0.0}

        latest = df.groupby('bookmaker').last()
        dispersion = 0.0
        if len(latest) > 1:
            dispersion = latest['implied_prob'].std()
            if pd.isna(dispersion):
                dispersion = 0.0

        first = df.groupby('bookmaker').first()
        changes = latest['implied_prob'] - first['implied_prob']
        velocity = changes.mean() if not changes.empty else 0.0
        steam_books = changes[changes > 0.02]
        steam_flag = len(steam_books) >= 3

        return {
            "steam_flag": steam_flag,
            "velocity": float(velocity),
            "dispersion": float(dispersion)
        }

    def get_book_market_bias(self, book: str, market: str) -> float:
        """
        Priority 7: Return per-book/market bias correction factor based on
        historical bet results. 1.0 = no correction. < 1.0 = we over-estimate
        for this book+market. Requires >= 20 settled bets to activate.
        """
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT AVG(CAST(b.won AS REAL)) as win_rate, COUNT(*) as count
                    FROM alerts_sent a
                    JOIN bet_results b ON a.id = b.alert_id
                    WHERE a.book = ? AND a.market = ?
                    """,
                    (book, market)
                )
                row = cursor.fetchone()
                if row and row['count'] and row['count'] >= 20:
                    win_rate = float(row['win_rate'] or 0.5)
                    # Expected: 52% win rate from edges. Deviation → correction.
                    # Clamp factor between 0.85 and 1.15.
                    bias = (win_rate - 0.52) / 0.10
                    return float(max(0.85, min(1.15, 1.0 + bias * 0.15)))
        except Exception:
            pass
        return 1.0

    def get_avg_clv(self, days_back: int = 30) -> float:
        """Priority 8: Return average CLV for recent alerts (used by tune.py)."""
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT AVG(clv) as avg_clv, COUNT(*) as count
                    FROM clv_tracking
                    WHERE clv IS NOT NULL
                    AND alert_time >= datetime('now', ?)
                    """,
                    (f'-{days_back} days',)
                )
                row = cursor.fetchone()
                if row and row['count'] and row['count'] >= 10:
                    return float(row['avg_clv'] or 0.0)
        except Exception:
            pass
        return 0.0

    def upsert_team_opponent_stats(self, team_name: str, season: str,
                                   opp_pts: float, opp_reb: float, opp_ast: float,
                                   opp_fg3m: float, pace: float, def_rating: float):
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO team_opponent_stats
                    (team_name, season, opp_pts_pg, opp_reb_pg, opp_ast_pg,
                     opp_fg3m_pg, pace, def_rating, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, date('now'))
                """,
                (team_name, season, opp_pts, opp_reb, opp_ast, opp_fg3m, pace, def_rating)
            )

    def get_team_opponent_stats(self, team_name: str, season: str) -> dict:
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM team_opponent_stats WHERE team_name = ? AND season = ?",
                (team_name, season)
            )
            row = cursor.fetchone()
            return dict(row) if row else {}

    def upsert_sgp_correlation(self, player_name: str, market_a: str,
                                market_b: str, correlation: float, sample_size: int):
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO sgp_correlations
                    (player_name, market_a, market_b, correlation, sample_size, last_updated)
                VALUES (?, ?, ?, ?, ?, date('now'))
                """,
                (player_name, market_a, market_b, correlation, sample_size)
            )
