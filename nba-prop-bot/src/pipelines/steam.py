"""
Steam Detection Pipeline.

Runs every 20 minutes (0 Odds API credits — reads the local DB only).

Algorithm
─────────
1. Query the last 120 minutes of line_history.
2. Group by (player, market, line, side).
3. For each sharp book (Pinnacle / Circa / Bookmaker):
     delta = last_implied_prob - first_implied_prob
   If |delta| >= 4% AND the book has ≥2 snapshots → steam confirmed.
4. For the same prop, check soft books (DK / FD / BetMGM / Caesars):
     If |delta| <= 1%  → book is stale (hasn't priced the move yet).
5. Fire a high-priority STEAM CHASE alert to Telegram for each new hit.
   Deduplication: skip if same player/market/side was alerted in the last 30 min.
"""

from src.data.db import DatabaseClient
from src.clients.telegram_bot import TelegramBotClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_MARKET_LABELS = {
    'player_points':                  'Points',
    'player_rebounds':                'Rebounds',
    'player_assists':                 'Assists',
    'player_threes':                  'Threes',
    'player_points_rebounds_assists': 'PRA',
    'player_blocks':                  'Blocks',
    'player_steals':                  'Steals',
}


def _american(decimal_odds: float) -> str:
    if decimal_odds >= 2.0:
        return f"+{int((decimal_odds - 1) * 100)}"
    if decimal_odds <= 1.0:
        return "N/A"
    return f"-{int(100 / (decimal_odds - 1))}"


def _format_steam_alert(move: dict) -> str:
    market_label = _MARKET_LABELS.get(move['market'], move['market'].replace('_', ' ').title())
    sharp        = move['sharp_book'].title()
    stale        = move['stale_book'].title()
    direction    = move['direction']
    line         = move['line']
    player       = move['player_name']

    sharp_from   = f"{move['sharp_first_prob']:.1%}"
    sharp_to     = f"{move['sharp_current_prob']:.1%}"
    delta_pct    = f"{abs(move['sharp_delta']):.1%}"
    elapsed      = move['elapsed_minutes']
    stale_am     = _american(move['stale_odds'])
    stale_prob   = f"{move['stale_current_prob']:.1%}"

    elapsed_str  = f"{elapsed:.0f} min" if elapsed >= 1 else "< 1 min"
    arrow        = "↑" if direction == 'OVER' else "↓"

    return (
        f"🔥 <b>STEAM CHASE ALERT</b>\n\n"
        f"<b>{player}</b>  {direction}  {line}  {market_label}\n\n"
        f"<b>Sharp Signal ({sharp}):</b>\n"
        f"  {sharp_from}  →  {sharp_to}  ({arrow}{delta_pct} in {elapsed_str})\n\n"
        f"<b>Stale Book to Bet:</b>\n"
        f"  {stale}  {stale_am}  (implied {stale_prob}) ← Act NOW\n\n"
        f"Sharp money is piling in on {direction} — "
        f"<b>{stale} hasn't moved yet.</b>"
    )


def check_steam(bot: TelegramBotClient = None, db: DatabaseClient = None) -> None:
    """
    Detect steam moves in the local line_history DB and fire Telegram alerts
    for any new opportunities before soft books catch up.

    Called by run_scheduler.job_steam() every 20 minutes on game days.
    """
    if db is None:
        db = DatabaseClient()
    if bot is None:
        bot = TelegramBotClient()

    moves = db.detect_steam_moves()

    if not moves:
        logger.debug("Steam check: no steam moves detected.")
        return

    sent = 0
    for move in moves:
        player = move['player_name']
        market = move['market']
        side   = move['side']

        if db.check_recent_steam_alert(player, market, side, minutes=30):
            logger.debug(f"Steam dedup: skipping {player} {market} {side} (alerted < 30 min ago)")
            continue

        db.insert_steam_alert(
            player_name=player,
            market=market,
            side=side,
            line=move['line'],
            sharp_book=move['sharp_book'],
            sharp_delta=move['sharp_delta'],
            sharp_current_prob=move['sharp_current_prob'],
            stale_book=move['stale_book'],
            stale_odds=move['stale_odds'],
            stale_current_prob=move['stale_current_prob'],
            direction=move['direction'],
        )

        msg = _format_steam_alert(move)
        bot.broadcast(msg, db=db)
        logger.info(
            f"Steam alert: {player} {market} {side} {move['line']} | "
            f"{move['sharp_book']} moved {move['sharp_delta']:+.1%} | "
            f"stale book: {move['stale_book']} ({_american(move['stale_odds'])})"
        )
        sent += 1

    logger.info(f"Steam check complete: {sent} alert(s) sent, {len(moves)} move(s) detected.")


if __name__ == "__main__":
    check_steam()
