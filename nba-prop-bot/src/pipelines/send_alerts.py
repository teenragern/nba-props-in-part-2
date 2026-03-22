from math import prod
from typing import Dict, Any, List
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
    MAX_PER_GAME   = BANKROLL * 0.10

    with db.get_conn() as conn:
        cursor = conn.cursor()

        # Daily total risk check
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

        # Per-game risk check — group correlated bets within the same 48-minute window
        if event_id:
            cursor.execute(
                "SELECT SUM(stake) as game_risk FROM alerts_sent "
                "WHERE event_id = ? AND date(timestamp) = date('now')",
                (event_id,),
            )
            game_row = cursor.fetchone()
            current_game_risk = float(game_row['game_risk'] or 0.0) if game_row else 0.0

            if current_game_risk + stake > MAX_PER_GAME:
                logger.warning(
                    f"Skipping {player} — per-game risk limit for event {event_id} "
                    f"({current_game_risk:.2f}/{MAX_PER_GAME:.2f})"
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


_PARLAY_MARKET_LABELS: Dict[str, str] = {
    'player_points':                  'Points',
    'player_rebounds':                'Rebounds',
    'player_assists':                 'Assists',
    'player_threes':                  'Threes',
    'player_points_rebounds_assists': 'PRA',
    'player_blocks':                  'Blocks',
    'player_steals':                  'Steals',
}


def _parlay_american(dec: float) -> str:
    if dec >= 2.0:
        return f"+{int((dec - 1) * 100)}"
    if dec <= 1.0:
        return "N/A"
    return f"-{int(100 / (dec - 1))}"


def send_parlay_alert(
    legs: List[Dict[str, Any]],
    joint_true_prob: float,
    joint_book_prob: float,
    bot: TelegramBotClient,
) -> None:
    """
    Format and send a multi-leg high-probability parlay to Telegram.

    Displays each leg with its individual hit probability and game context,
    then shows combined book odds, true hit probability, edge, and EV.

    Args:
        legs:             List of edge dicts (player_id, market, side, line,
                          book, odds, model_prob, home_team, away_team).
        joint_true_prob:  Model's joint probability all legs hit.
        joint_book_prob:  Book's implied joint probability (product of vigs).
        bot:              Telegram client.
    """
    n = len(legs)
    icon = '4️⃣' if n == 4 else '8️⃣'

    combined_decimal = prod(leg['odds'] for leg in legs)
    edge = joint_true_prob - joint_book_prob
    ev   = joint_true_prob * (combined_decimal - 1) - (1 - joint_true_prob)

    # Hard cap: parlays are high-variance bets — even a model with real edge
    # can go on cold streaks.  Cap at 0.5% (4-leg) / 0.25% (8-leg) of bankroll
    # regardless of Kelly, to prevent ruin from model over-confidence.
    _PARLAY_MAX_PCT = {4: 0.005, 8: 0.0025}
    max_parlay_stake = BANKROLL * _PARLAY_MAX_PCT.get(n, 0.0025)

    lines = [f"{icon} <b>{n}-Leg High-Probability Parlay</b>\n"]
    for leg in legs:
        market = _PARLAY_MARKET_LABELS.get(leg.get('market', ''), leg.get('market', ''))
        away   = leg.get('away_team', '')
        home   = leg.get('home_team', '')
        game   = f"[{away} @ {home}]" if away and home else ""
        prob   = leg.get('model_prob', 0.0)
        lines.append(
            f"• <b>{leg.get('player_id', '?')}</b> "
            f"{leg.get('side', '')} {leg.get('line', '')} {market} "
            f"@ {leg.get('book', '')} ({_parlay_american(leg.get('odds', 1.0))}) "
            f"{game} — {prob:.0%}"
        )

    lines.append(
        f"\nCombined Odds:  {_parlay_american(combined_decimal)}\n"
        f"True Hit Prob:  {joint_true_prob:.2%}\n"
        f"Book Implied:   {joint_book_prob:.2%}\n"
        f"Edge:           +{edge:.2%}\n"
        f"EV:             {ev:+.2%}\n"
        f"Max Stake:      ${max_parlay_stake:.2f} ({_PARLAY_MAX_PCT.get(n, 0.0025):.2%} bankroll cap)"
    )

    bot.send_message("\n".join(lines))
    logger.info(
        f"{n}-leg high-prob parlay | "
        f"joint_prob={joint_true_prob:.2%} | edge={edge:.2%} | ev={ev:.2%}"
    )
