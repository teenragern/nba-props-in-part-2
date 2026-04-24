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


def normalize_book(book: Optional[str]) -> str:
    """Canonicalize sportsbook names so BetMGM/betmgm dedupe to one entry."""
    if not book:
        return ''
    return str(book).strip().lower()

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
            # bet_results — push detection (settlement v2)
            "ALTER TABLE bet_results ADD COLUMN push BOOLEAN NOT NULL DEFAULT 0",
            # players — role shift: primary ball-handler tag
            "ALTER TABLE players ADD COLUMN is_primary_initiator BOOLEAN NOT NULL DEFAULT 0",
            # pending_alerts — two-tier alert batching (flush v1)
            # (table created by schema; migration ensures forward-compat if
            #  an older DB predates the CREATE TABLE statement above)
            """CREATE TABLE IF NOT EXISTS pending_alerts (
                id         INTEGER  PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT     NOT NULL,
                title      TEXT     NOT NULL,
                body       TEXT     NOT NULL,
                priority   REAL     NOT NULL DEFAULT 0.0,
                game_date  TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                sent_at    DATETIME
            )""",
            # BDL defense profiles cache
            """CREATE TABLE IF NOT EXISTS bdl_defense_profiles (
                team_key    TEXT    NOT NULL,
                season      INTEGER NOT NULL,
                opp_pts     REAL NOT NULL DEFAULT 1.0,
                opp_reb     REAL NOT NULL DEFAULT 1.0,
                opp_ast     REAL NOT NULL DEFAULT 1.0,
                opp_fg3m    REAL NOT NULL DEFAULT 1.0,
                opp_fta     REAL NOT NULL DEFAULT 1.0,
                opp_pts_paint REAL NOT NULL DEFAULT 1.0,
                def_rating  REAL NOT NULL DEFAULT 1.0,
                pace        REAL NOT NULL DEFAULT 1.0,
                blk         REAL NOT NULL DEFAULT 1.0,
                stl         REAL NOT NULL DEFAULT 1.0,
                fetched_at  TEXT NOT NULL,
                PRIMARY KEY (team_key, season)
            )""",
            # BDL game log cache
            """CREATE TABLE IF NOT EXISTS bdl_game_log_cache (
                player_id   INTEGER NOT NULL,
                season      INTEGER NOT NULL,
                game_date   TEXT    NOT NULL,
                game_id     INTEGER,
                min         REAL NOT NULL DEFAULT 0.0,
                pts         REAL NOT NULL DEFAULT 0.0,
                reb         REAL NOT NULL DEFAULT 0.0,
                ast         REAL NOT NULL DEFAULT 0.0,
                fg3m        REAL NOT NULL DEFAULT 0.0,
                blk         REAL NOT NULL DEFAULT 0.0,
                stl         REAL NOT NULL DEFAULT 0.0,
                fga         REAL NOT NULL DEFAULT 0.0,
                fta         REAL NOT NULL DEFAULT 0.0,
                tov         REAL NOT NULL DEFAULT 0.0,
                team_abbr   TEXT,
                matchup     TEXT,
                wl          TEXT,
                cached_at   TEXT NOT NULL,
                PRIMARY KEY (player_id, season, game_date)
            )""",
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
        book = normalize_book(book)
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

    def get_sharp_line_shift(self, player_name: str, market: str) -> dict:
        """
        Detect if any sharp book has made a whole-number line move (>= 1.0)
        within the last 3 hours for this player/market.

        Reads the Over side only — both sides are stored with the same line
        value so one side is sufficient to see the structural shift.

        Returns a dict with:
          shift_detected  – bool
          sharp_book      – the book that moved first
          old_line        – earliest line seen in the window
          new_line        – latest line seen in the window
          direction       – 'UP' (sharp money on Over) or 'DOWN' (sharp money on Under)
          magnitude       – |new_line - old_line|
        """
        _empty: dict = {'shift_detected': False}
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT bookmaker, line, timestamp
                    FROM line_history
                    WHERE player_name = ? AND market = ? AND side = 'OVER'
                      AND LOWER(bookmaker) IN ('pinnacle', 'circa', 'bookmaker')
                      AND timestamp >= datetime('now', '-3 hours')
                    ORDER BY bookmaker, timestamp ASC
                    """,
                    (player_name, market),
                )
                rows = cursor.fetchall()
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
            earliest, latest = lines[0], lines[-1]
            magnitude = latest - earliest
            if abs(magnitude) >= 1.0:
                return {
                    'shift_detected':    True,
                    'sharp_book':        book,
                    'old_line':          earliest,
                    'new_line':          latest,
                    'direction':         'UP' if magnitude > 0 else 'DOWN',
                    'magnitude':         abs(magnitude),
                }
        return _empty

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

    def get_clv_beat_rate(self, days_back: int = 30, min_samples: int = 10) -> Optional[float]:
        """
        Return the fraction of settled bets that beat the closing line (clv > 0)
        over the last `days_back` days.  Returns None when fewer than
        `min_samples` settled bets exist (insufficient data to act on).
        """
        try:
            with self.get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COUNT(*)                             AS total,
                           SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) AS beats
                    FROM clv_tracking
                    WHERE clv IS NOT NULL
                      AND alert_time >= datetime('now', ?)
                    """,
                    (f'-{days_back} days',)
                )
                row = cursor.fetchone()
                if row and row['total'] and row['total'] >= min_samples:
                    return float(row['beats']) / float(row['total'])
        except Exception:
            pass
        return None

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

    # ── Role-shift: primary initiator helpers ──────────────────────────

    def get_team_initiator_ids(self, team_id: int) -> list:
        """Return player_ids tagged as primary initiators for a team."""
        with self.get_conn() as conn:
            rows = conn.execute(
                "SELECT player_id FROM players WHERE team_id = ? AND is_primary_initiator = 1",
                (team_id,)
            ).fetchall()
            return [r['player_id'] for r in rows]

    def is_primary_initiator(self, player_id: int) -> bool:
        """Check if a player is tagged as a primary initiator."""
        with self.get_conn() as conn:
            row = conn.execute(
                "SELECT is_primary_initiator FROM players WHERE player_id = ?",
                (player_id,)
            ).fetchone()
            return bool(row and row['is_primary_initiator'])

    def get_on_off_rate_without(self, player_id: int, absent_player_id: int,
                                 market: str, season: str) -> Optional[float]:
        """Return the raw per-minute rate_without from on_off_splits, or None."""
        with self.get_conn() as conn:
            row = conn.execute(
                """SELECT rate_without, minutes_without FROM on_off_splits
                   WHERE player_id = ? AND absent_player_id = ?
                     AND market = ? AND season = ?""",
                (player_id, absent_player_id, market, season)
            ).fetchone()
            if row and row['minutes_without'] >= 10.0:
                return float(row['rate_without'])
            return None

    def set_primary_initiators(self, team_id: int, player_ids: list):
        """Tag players as primary initiators; clear old tags for the team first."""
        with self.get_conn() as conn:
            conn.execute(
                "UPDATE players SET is_primary_initiator = 0 WHERE team_id = ?",
                (team_id,)
            )
            for pid in player_ids:
                conn.execute(
                    "UPDATE players SET is_primary_initiator = 1 WHERE player_id = ? AND team_id = ?",
                    (pid, team_id)
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
    #  Cross-team (opposing-player) SGP correlations                      #
    # ------------------------------------------------------------------ #

    def upsert_cross_team_correlation(
        self, matchup: str, player_a: str, player_b: str,
        market_a: str, market_b: str, correlation: float, n_games: int,
    ) -> None:
        with self.get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cross_team_correlations
                   (matchup, player_a, player_b, market_a, market_b,
                    correlation, n_games, computed_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, date('now'))""",
                (matchup, player_a, player_b, market_a, market_b, correlation, n_games),
            )

    def get_cross_team_correlation(
        self, matchup: str, player_a: str, player_b: str,
        market_a: str, market_b: str,
    ) -> Optional[float]:
        """Return cached cross-team correlation if stored within the last 7 days."""
        with self.get_conn() as conn:
            row = conn.execute(
                """SELECT correlation FROM cross_team_correlations
                   WHERE matchup = ? AND player_a = ? AND player_b = ?
                   AND market_a = ? AND market_b = ?
                   AND computed_date >= date('now', '-7 days')""",
                (matchup, player_a, player_b, market_a, market_b),
            ).fetchone()
        return float(row['correlation']) if row else None

    # ------------------------------------------------------------------ #
    #  Steam detection                                                     #
    # ------------------------------------------------------------------ #

    def detect_steam_moves(
        self,
        window_minutes: int = 120,
        sharp_move_threshold: float = 0.04,
        stale_threshold: float = 0.01,
        sharp_books: tuple = ('pinnacle', 'circa', 'bookmaker'),
        soft_books: tuple = ('draftkings', 'fanduel', 'betmgm', 'caesars'),
    ) -> list:
        """
        Detect sharp steam moves: a sharp book has moved its implied probability
        by >= sharp_move_threshold within the window AND at least one soft book
        is still priced near its original level (<= stale_threshold change).

        Returns a list of dicts — one per detected steam opportunity — with keys:
            player_name, market, side, line,
            sharp_book, sharp_delta, sharp_first_prob, sharp_current_prob,
            stale_book, stale_odds, stale_current_prob,
            direction ('OVER' or 'UNDER'), elapsed_minutes
        """
        import pandas as pd

        with self.get_conn() as conn:
            try:
                df = pd.read_sql_query(
                    """
                    SELECT player_name, market, side, line,
                           bookmaker, odds, implied_prob, timestamp
                    FROM line_history
                    WHERE timestamp >= datetime('now', ?)
                    ORDER BY timestamp ASC
                    """,
                    conn,
                    params=(f'-{window_minutes} minutes',),
                )
            except Exception:
                return []

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
                    'first_prob': first_prob,
                    'last_prob':  last_prob,
                    'last_odds':  last_odds,
                    'delta':      delta,
                    'elapsed':    elapsed,
                    'n_obs':      n,
                }

            # Sharp books need ≥2 observations to confirm a real move
            sharp_moves = {
                b: s for b, s in book_stats.items()
                if b in sharp_books
                and s['n_obs'] >= 2
                and abs(s['delta']) >= sharp_move_threshold
            }
            if not sharp_moves:
                continue

            # Soft books: stale when their delta is ≤ stale_threshold
            stale_softs = {
                b: s for b, s in book_stats.items()
                if b in soft_books and abs(s['delta']) <= stale_threshold
            }
            if not stale_softs:
                continue

            best_sharp = max(sharp_moves, key=lambda b: abs(sharp_moves[b]['delta']))
            sharp_info = sharp_moves[best_sharp]
            direction  = 'OVER' if sharp_info['delta'] > 0 else 'UNDER'

            # Best stale soft book = lowest implied_prob (highest decimal odds)
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

    def check_recent_steam_alert(
        self, player_name: str, market: str, side: str, minutes: int = 30
    ) -> bool:
        """Return True if a steam alert for this player/market/side was already sent
        within the last `minutes` minutes (deduplication guard)."""
        with self.get_conn() as conn:
            row = conn.execute(
                """
                SELECT id FROM steam_alerts
                WHERE player_name = ? AND market = ? AND side = ?
                  AND timestamp >= datetime('now', ?)
                """,
                (player_name, market, side, f'-{minutes} minutes'),
            ).fetchone()
        return row is not None

    def insert_steam_alert(
        self, player_name: str, market: str, side: str, line: float,
        sharp_book: str, sharp_delta: float, sharp_current_prob: float,
        stale_book: str, stale_odds: float, stale_current_prob: float,
        direction: str,
    ) -> None:
        with self.get_conn() as conn:
            conn.execute(
                """
                INSERT INTO steam_alerts
                    (player_name, market, side, line, sharp_book, sharp_delta,
                     sharp_current_prob, stale_book, stale_odds, stale_current_prob,
                     direction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (player_name, market, side, line, sharp_book, sharp_delta,
                 sharp_current_prob, stale_book, stale_odds, stale_current_prob,
                 direction),
            )

    # ------------------------------------------------------------------ #
    #  Referee stats                                                       #
    # ------------------------------------------------------------------ #

    def get_referee_foul_rates(self, ref_names: list) -> list:
        """
        Return avg_pfd_per_game values for known referees.
        Skips refs with no data (avg_pfd_per_game = 0.0 means unknown).
        """
        if not ref_names:
            return []
        placeholders = ",".join("?" * len(ref_names))
        try:
            with self.get_conn() as conn:
                rows = conn.execute(
                    f"SELECT avg_pfd_per_game FROM referee_stats "
                    f"WHERE referee_name IN ({placeholders}) AND avg_pfd_per_game > 0",
                    ref_names,
                ).fetchall()
            return [float(r["avg_pfd_per_game"]) for r in rows]
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    #  Backtesting                                                         #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Two-tier alert batching                                            #
    # ------------------------------------------------------------------ #

    def queue_pending_alert(
        self,
        alert_type: str,
        title: str,
        body: str,
        priority: float = 0.0,
        game_date: str = None,
    ) -> int:
        """
        Store a Tier-2 alert for inclusion in the next digest flush.

        Returns the new row id.
        """
        with self.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO pending_alerts (alert_type, title, body, priority, game_date)
                VALUES (?, ?, ?, ?, ?)
                """,
                (alert_type, title, body, priority, game_date),
            )
            return cursor.lastrowid

    def get_pending_alerts(self, unsent_only: bool = True) -> list:
        """
        Return pending_alerts rows as dicts.

        When `unsent_only=True` (default), returns only rows where sent_at IS NULL.
        Rows are ordered by alert_type then priority descending (highest edge first).
        """
        with self.get_conn() as conn:
            cursor = conn.cursor()
            where = "WHERE sent_at IS NULL" if unsent_only else ""
            cursor.execute(
                f"""
                SELECT id, alert_type, title, body, priority, game_date, created_at
                FROM pending_alerts
                {where}
                ORDER BY alert_type, priority DESC
                """
            )
            return [dict(r) for r in cursor.fetchall()]

    def mark_pending_alerts_sent(self, ids: list) -> None:
        """Stamp sent_at = now for the given pending_alert ids."""
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        with self.get_conn() as conn:
            conn.execute(
                f"UPDATE pending_alerts SET sent_at = CURRENT_TIMESTAMP "
                f"WHERE id IN ({placeholders})",
                ids,
            )

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

    # ── BDL defense profile cache ─────────────────────────────────────

    def upsert_bdl_defense_profile(self, team_key: str, season: int,
                                    profile: dict):
        """Cache a normalised BDL defense profile for a team/season."""
        with self.get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO bdl_defense_profiles
                    (team_key, season, opp_pts, opp_reb, opp_ast, opp_fg3m,
                     opp_fta, opp_pts_paint, def_rating, pace, blk, stl,
                     fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (team_key.lower(), season,
                 profile.get('opp_pts', 1.0), profile.get('opp_reb', 1.0),
                 profile.get('opp_ast', 1.0), profile.get('opp_fg3m', 1.0),
                 profile.get('opp_fta', 1.0), profile.get('opp_pts_paint', 1.0),
                 profile.get('def_rating', 1.0), profile.get('pace', 1.0),
                 profile.get('blk', 1.0), profile.get('stl', 1.0),
                 profile.get('fetched_at', ''))
            )

    def get_bdl_defense_profile(self, team_key: str, season: int) -> Optional[dict]:
        """Retrieve a cached BDL defense profile, or None if not found."""
        with self.get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM bdl_defense_profiles WHERE team_key = ? AND season = ?",
                (team_key.lower(), season)
            ).fetchone()
            if not row:
                # Partial match fallback
                rows = conn.execute(
                    "SELECT * FROM bdl_defense_profiles WHERE season = ?",
                    (season,)
                ).fetchall()
                low = team_key.lower()
                for r in rows:
                    k = r['team_key']
                    if k in low or low in k:
                        row = r
                        break
            return dict(row) if row else None

    # ── BDL game log cache ────────────────────────────────────────────

    def cache_bdl_game_logs(self, player_id: int, season: int,
                             logs: list):
        """
        Bulk-insert game log rows into bdl_game_log_cache.
        Each item in logs is a dict with keys matching the table columns.
        """
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        with self.get_conn() as conn:
            for log in logs:
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO bdl_game_log_cache
                            (player_id, season, game_date, game_id, min, pts, reb,
                             ast, fg3m, blk, stl, fga, fta, tov,
                             team_abbr, matchup, wl, cached_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (player_id, season, log.get('game_date', ''),
                         log.get('game_id'), log.get('min', 0),
                         log.get('pts', 0), log.get('reb', 0),
                         log.get('ast', 0), log.get('fg3m', 0),
                         log.get('blk', 0), log.get('stl', 0),
                         log.get('fga', 0), log.get('fta', 0),
                         log.get('tov', 0), log.get('team_abbr', ''),
                         log.get('matchup', ''), log.get('wl', ''), now)
                    )
                except Exception:
                    pass

    def get_cached_bdl_game_logs(self, player_id: int, season: int) -> list:
        """
        Retrieve cached game logs for a player/season.
        Returns list of dicts sorted newest-first, or [] if stale/missing.
        Uses a 12-hour TTL.
        """
        from datetime import datetime, timedelta
        cutoff = (datetime.utcnow() - timedelta(hours=12)).isoformat()
        with self.get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM bdl_game_log_cache
                WHERE player_id = ? AND season = ? AND cached_at > ?
                ORDER BY game_date DESC
                """,
                (player_id, season, cutoff)
            ).fetchall()
        return [dict(r) for r in rows] if rows else []
