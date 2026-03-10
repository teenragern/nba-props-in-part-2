from typing import Dict, Any
from src.utils.logging_utils import get_logger
from src.clients.telegram_bot import TelegramBotClient
from src.data.db import DatabaseClient
from src.config import BANKROLL, KELLY_FRACTION

logger = get_logger(__name__)


def evaluate_and_alert(edge_data: Dict[str, Any], db: DatabaseClient, bot: TelegramBotClient):
    player    = edge_data.get('player_id', 'Unknown')
    market    = edge_data.get('market',    'Unknown')
    line      = edge_data.get('line',      0.0)
    side      = edge_data.get('side',      'Unknown')
    book      = edge_data.get('book',      'Unknown')
    edge      = edge_data.get('edge',      0.0)
    odds      = edge_data.get('odds',      0.0)
    game_date = edge_data.get('game_date', None)
    event_id  = edge_data.get('event_id',  None)
    home_away = edge_data.get('home_away', None)
    rest_days = edge_data.get('rest_days', 2)

    if db.check_recent_alert(player, market, line, side, edge):
        logger.info(f"Skipping duplicate alert for {player} {market} {side} {line}")
        return

    home = edge_data.get('home_team', 'Home')
    away = edge_data.get('away_team', 'Away')

    # Fractional Kelly stake sizing
    stake = 0.0
    if odds > 1:
        stake = BANKROLL * (edge / (odds - 1.0)) * KELLY_FRACTION
        stake = min(stake, BANKROLL * 0.05)  # hard cap: 5% per bet

    MAX_DAILY_RISK = BANKROLL * 0.25

    with db.get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT SUM(stake) as total_risk FROM alerts_sent WHERE date(timestamp) = date('now')"
        )
        row = cursor.fetchone()
        current_daily_risk = float(row['total_risk'] or 0.0) if row else 0.0

        if current_daily_risk + stake > MAX_DAILY_RISK:
            logger.warning(
                f"Skipping {player} — daily risk limit "
                f"({current_daily_risk:.2f}/{MAX_DAILY_RISK:.2f})"
            )
            return

    db.insert_alert(
        player_name=player, market=market, line=line, side=side,
        edge=edge, book=book, odds=odds, stake=stake,
        game_date=game_date, event_id=event_id,
        home_away=home_away, rest_days=rest_days,
    )

    ml_blend_note = " [ML+Bayesian blend]" if edge_data.get('ml_blend') else ""
    msg = (
        f"<b>🔥 NBA PROP EDGE</b>{ml_blend_note}\n\n"
        f"Game: {away} @ {home}  ({game_date})\n"
        f"Player: {player}  [{home_away}, {rest_days}d rest]\n"
        f"Market: {market.replace('_', ' ').title()}\n"
        f"Side: {side}  |  Line: {line}\n"
        f"Best Book: {book}  |  Odds: {odds}\n\n"
        f"Model Prob:   {edge_data.get('model_prob',   0.0):.3f}\n"
        f"Implied Prob: {edge_data.get('implied_prob', 0.0):.3f}\n"
        f"Edge:  {edge:.3%}\n"
        f"EV:    {edge_data.get('ev', 0.0):.3%}\n\n"
        f"Proj Mean:    {edge_data.get('mean',               0.0):.2f}\n"
        f"Proj Minutes: {edge_data.get('projected_minutes',  0.0):.1f}\n"
        f"Injury:       {edge_data.get('injury_status', 'Healthy')}\n"
        f"Usage Boost:  {edge_data.get('usage_boost',    0.0):.1%}\n"
        f"Book Bias:    {edge_data.get('feedback_factor_applied', 1.0):.2f}\n\n"
        f"<b>Suggested Stake (Kelly):</b> ${stake:.2f}"
    )

    bot.send_message(msg)
    logger.info(f"Alert sent: {player} {market} {side} {line} @ {book}")
