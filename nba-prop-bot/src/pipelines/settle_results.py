"""
Priority 3: Fixed settlement — filters player game logs by alert date
instead of always grabbing the most recent game (which may be wrong).
"""

import pandas as pd
from src.utils.logging_utils import get_logger
from src.data.db import DatabaseClient
from src.clients.nba_stats import NbaStatsClient
from src.clients.telegram_bot import TelegramBotClient

logger = get_logger(__name__)


def evaluate_result(market: str, line: float, side: str, actual_val: float) -> int:
    if side.upper() == "OVER":
        return 1 if actual_val > line else 0
    elif side.upper() == "UNDER":
        return 1 if actual_val < line else 0
    return 0


def _find_game_row(logs: pd.DataFrame, game_date: str) -> pd.Series:
    """
    Priority 3: Match the game log row to the alert's game_date.
    Falls back to most-recent game if date parsing fails.
    """
    if logs.empty:
        return pd.Series()

    if game_date and 'GAME_DATE' in logs.columns:
        try:
            target = pd.to_datetime(game_date).normalize()
            logs_copy = logs.copy()
            logs_copy['_parsed_date'] = pd.to_datetime(logs_copy['GAME_DATE'], errors='coerce').dt.normalize()
            match = logs_copy[logs_copy['_parsed_date'] == target]
            if not match.empty:
                return match.iloc[0]
        except Exception:
            pass

    return logs.iloc[0]  # fallback: latest game


def settle_alerts():
    db    = DatabaseClient()
    stats = NbaStatsClient()
    bot   = TelegramBotClient()

    logger.info("Initializing settlement engine...")

    with db.get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT a.id, a.player_name, a.market, a.line, a.side,
                   COALESCE(a.game_date, DATE(a.timestamp)) as game_date
            FROM alerts_sent a
            LEFT JOIN bet_results b ON a.id = b.alert_id
            WHERE b.alert_id IS NULL
            """
        )
        unsettled = cursor.fetchall()

    if not unsettled:
        logger.info("No unsettled bets to grade.")
        return

    logger.info(f"Attempting to settle {len(unsettled)} alerts.")

    for row in unsettled:
        alert_id    = row['id']
        player_name = row['player_name']
        market      = row['market']
        line        = row['line']
        side        = row['side']
        game_date   = row['game_date']

        try:
            from nba_api.stats.static import players
            found = players.find_players_by_full_name(player_name)
            if not found:
                continue
            pid = found[0]['id']

            logs = stats.get_player_game_logs(pid)
            if logs.empty:
                continue

            game_row = _find_game_row(logs, game_date)
            if game_row.empty:
                logger.warning(f"No game found on {game_date} for {player_name}")
                continue

            pts    = float(game_row.get('PTS',  0) or 0)
            ast    = float(game_row.get('AST',  0) or 0)
            reb    = float(game_row.get('REB',  0) or 0)
            threes = float(game_row.get('FG3M', 0) or 0)
            pra    = pts + ast + reb

            actual_val = 0.0
            if market == "player_points":                    actual_val = pts
            elif market == "player_rebounds":                actual_val = reb
            elif market == "player_assists":                 actual_val = ast
            elif market == "player_threes":                  actual_val = threes
            elif market == "player_points_rebounds_assists": actual_val = pra

            won = evaluate_result(market, line, side, actual_val)

            with db.get_conn() as wconn:
                wcursor = wconn.cursor()
                wcursor.execute(
                    "INSERT INTO bet_results (alert_id, actual_result, won) VALUES (?, ?, ?)",
                    (alert_id, float(actual_val), won)
                )

            logger.info(
                f"Settled Alert {alert_id} | {player_name} {side} {line} {market} "
                f"| Date: {game_date} | Actual: {actual_val} | Won: {won}"
            )

            result_emoji = "✅ WON" if won else "❌ LOST"
            bot.send_message(
                f"<b>Bet Settled: {result_emoji}</b>\n\n"
                f"🏀 <b>Player:</b> {player_name}\n"
                f"📊 <b>Market:</b> {market.replace('_', ' ').title()}\n"
                f"🎯 <b>Line:</b> {side} {line}\n"
                f"📈 <b>Actual Result:</b> {actual_val}\n"
                f"📅 <b>Game Date:</b> {game_date}"
            )

        except Exception as e:
            logger.error(f"Failed resolving settlement for {player_name}: {e}")
            continue

    logger.info("Settlement run complete.")


if __name__ == "__main__":
    settle_alerts()
