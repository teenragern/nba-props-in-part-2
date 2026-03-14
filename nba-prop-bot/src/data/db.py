import sqlite3
import os
from contextlib import contextmanager
from typing import Dict, Optional
from src.config import DB_PATH
from src.utils.logging_utils import get_logger

# Default accuracy weights for consensus sharp books.
# Based on industry research; overridden by DB once >= 20 samples accumulate.
_SHARP_DEFAULT_WEIGHTS: Dict[str, float] = {
    'pinnacle': 1.00,
    'circa':    0.90,
    'bookmaker': 0.82,
}

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
        # (bookmaker, role, default_clv_score)
        # Non-zero default CLV seeds the weighting system for sharp books.
        # INSERT OR IGNORE preserves any historically accumulated scores.
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
            cursor = conn.cursor()
            for book, role, default_clv in profiles:
                cursor.execute(
                    """INSERT OR IGNORE INTO bookmaker_profiles
                           (bookmaker, role, historical_clv_score)
                       VALUES (?, ?, ?)""",
                    (book, role, default_clv)
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

    def get_on_off_split(self, player_id: int, absent_player_id: int,
                         market: str, season: str):
        """Return cached on/off split row dict, or None if not yet computed."""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM on_off_splits
                WHERE player_id = ? AND absent_player_id = ?
                  AND market = ? AND season = ?
                """,
                (player_id, absent_player_id, market, season)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def upsert_on_off_split(self, player_id: int, absent_player_id: int,
                             market: str, season: str,
                             games_processed: int, minutes_with: float,
                             minutes_without: float, rate_with: float,
                             rate_without: float, usage_multiplier: float):
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO on_off_splits
                    (player_id, absent_player_id, season, market,
                     games_processed, minutes_with, minutes_without,
                     rate_with, rate_without, usage_multiplier, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, date('now'))
                """,
                (player_id, absent_player_id, season, market,
                 games_processed, minutes_with, minutes_without,
                 rate_with, rate_without, usage_multiplier)
            )

    def get_rotation_slots(self, team_abbr: str, season: str) -> dict:
        """
        Return cached rotation slot matrix for a team, or {} if stale/missing.
        Format: {player_id (int): {slot_key (str): probability (float)}}.
        """
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT player_id, slot_key, slot_probability, last_updated "
                "FROM rotation_slots WHERE team_abbr=? AND season=?",
                (team_abbr, season)
            )
            rows = cursor.fetchall()
        if not rows:
            return {}
        if rows[0]['last_updated'] != today:
            return {}  # stale — caller will rebuild
        result: dict = {}
        for row in rows:
            pid = int(row['player_id'])
            if pid not in result:
                result[pid] = {}
            result[pid][row['slot_key']] = float(row['slot_probability'])
        return result

    def upsert_rotation_slots(self, team_abbr: str,
                               team_slots: dict, games_processed: int,
                               season: str, last_updated: str):
        """Persist a full team rotation slot matrix to the DB."""
        with self.get_conn() as conn:
            cursor = conn.cursor()
            # Remove stale rows for this team/season first
            cursor.execute(
                "DELETE FROM rotation_slots WHERE team_abbr=? AND season=?",
                (team_abbr, season)
            )
            for player_id, slots in team_slots.items():
                for slot_key, prob in slots.items():
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO rotation_slots
                            (team_abbr, player_id, season, slot_key,
                             slot_probability, games_processed, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (team_abbr, int(player_id), season, slot_key,
                         float(prob), games_processed, last_updated)
                    )

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

    def get_sharp_book_weights(self) -> Dict[str, float]:
        """
        Return {bookmaker_lower: weight} for all sharp-role books.

        Uses DB historical_clv_score when the book has accumulated meaningful
        data (score > 0.10); otherwise falls back to _SHARP_DEFAULT_WEIGHTS.
        Weights are normalized so the highest-weight book = 1.0.
        """
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT bookmaker, historical_clv_score "
                    "FROM bookmaker_profiles WHERE role = 'sharp'"
                )
                rows = cursor.fetchall()

            if not rows:
                return dict(_SHARP_DEFAULT_WEIGHTS)

            weights: Dict[str, float] = {}
            for row in rows:
                book = str(row['bookmaker']).lower()
                score = float(row['historical_clv_score'] or 0.0)
                weights[book] = score if score > 0.10 else _SHARP_DEFAULT_WEIGHTS.get(book, 0.75)

            # Normalize: max weight → 1.0
            max_w = max(weights.values()) if weights else 1.0
            if max_w > 0:
                return {k: v / max_w for k, v in weights.items()}
            return dict(_SHARP_DEFAULT_WEIGHTS)
        except Exception:
            return dict(_SHARP_DEFAULT_WEIGHTS)

    def update_sharp_book_clv_score(self, book: str, brier_delta: float) -> None:
        """
        Update a sharp book's accuracy score via exponential moving average.

        Call this after prop results settle.  `brier_delta` should be positive
        when the book was well-calibrated and negative when it was off.
        A slow EMA (α = 0.05) ensures the score adapts over many samples.
        Score is clamped to [0.10, 2.00] to avoid extreme values.
        """
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT historical_clv_score FROM bookmaker_profiles "
                    "WHERE bookmaker = ? COLLATE NOCASE",
                    (book,)
                )
                row = cursor.fetchone()
                if row is None:
                    return
                current = float(row['historical_clv_score'] or 0.0)
                if current <= 0.0:
                    current = _SHARP_DEFAULT_WEIGHTS.get(book.lower(), 0.75)
                new_score = max(0.10, min(2.00, current * 0.95 + brier_delta * 0.05))
                cursor.execute(
                    "UPDATE bookmaker_profiles SET historical_clv_score = ? "
                    "WHERE bookmaker = ? COLLATE NOCASE",
                    (new_score, book)
                )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Cross-player SGP correlations                                       #
    # ------------------------------------------------------------------ #

    def upsert_cross_player_correlation(
        self, team: str, player_a: str, player_b: str,
        market_a: str, market_b: str, correlation: float, n_games: int,
    ) -> None:
        with self.get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cross_player_correlations
                   (team, player_a, player_b, market_a, market_b,
                    correlation, n_games, computed_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, date('now'))""",
                (team, player_a, player_b, market_a, market_b, correlation, n_games),
            )

    def get_cross_player_correlation(
        self, team: str, player_a: str, player_b: str,
        market_a: str, market_b: str,
    ) -> Optional[float]:
        """Return cached correlation if computed within the last 7 days, else None."""
        with self.get_conn() as conn:
            row = conn.execute(
                """SELECT correlation FROM cross_player_correlations
                   WHERE team = ? AND player_a = ? AND player_b = ?
                   AND market_a = ? AND market_b = ?
                   AND computed_date >= date('now', '-7 days')""",
                (team, player_a, player_b, market_a, market_b),
            ).fetchone()
        return float(row['correlation']) if row else None

    # ------------------------------------------------------------------ #
    #  Backtesting                                                         #
    # ------------------------------------------------------------------ #

    def insert_backtest_results_batch(self, records: list) -> None:
        """
        Bulk-insert backtest simulation rows.
        Each record: (player_name, season, market, game_date, simulated_line,
                      model_mean, model_prob_over, actual_stat, hit, edge)
        """
        with self.get_conn() as conn:
            conn.executemany(
                """
                INSERT INTO backtest_results
                    (player_name, season, market, game_date, simulated_line,
                     model_mean, model_prob_over, actual_stat, hit, edge)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )

    def get_backtest_summary(self) -> dict:
        """
        Return aggregate backtest metrics:
          - per-market: bets, hits, hit_rate, roi (at -110 odds)
          - per-edge-bucket: bets, hits, hit_rate
          - calibration: list of (prob_bin, actual_hit_rate, n)
        """
        with self.get_conn() as conn:
            # Per-market
            mkt_rows = conn.execute(
                """
                SELECT market,
                       COUNT(*)             AS bets,
                       SUM(hit)             AS hits,
                       AVG(hit)             AS hit_rate,
                       SUM(CASE WHEN hit=1 THEN 0.909 ELSE -1.0 END) AS profit
                FROM backtest_results
                GROUP BY market
                ORDER BY market
                """
            ).fetchall()

            # Per-edge-bucket (0-3%, 3-6%, 6-10%, 10%+)
            bucket_rows = conn.execute(
                """
                SELECT
                    CASE
                        WHEN edge < 0.03 THEN '0-3%'
                        WHEN edge < 0.06 THEN '3-6%'
                        WHEN edge < 0.10 THEN '6-10%'
                        ELSE '10%+'
                    END AS bucket,
                    COUNT(*) AS bets,
                    SUM(hit) AS hits,
                    AVG(hit) AS hit_rate
                FROM backtest_results
                WHERE edge > 0
                GROUP BY bucket
                ORDER BY bucket
                """
            ).fetchall()

            # Calibration: model_prob_over binned into 5% bands
            cal_rows = conn.execute(
                """
                SELECT
                    ROUND(model_prob_over / 0.05) * 0.05 AS prob_bin,
                    AVG(hit)  AS actual_hit_rate,
                    COUNT(*)  AS n
                FROM backtest_results
                GROUP BY prob_bin
                ORDER BY prob_bin
                """
            ).fetchall()

        return {
            'markets':     [dict(r) for r in mkt_rows],
            'buckets':     [dict(r) for r in bucket_rows],
            'calibration': [dict(r) for r in cal_rows],
        }
