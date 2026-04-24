from math import prod
from typing import Dict, Any, List, Optional
from src.utils.logging_utils import get_logger
from src.clients.telegram_bot import TelegramBotClient
from src.data.db import DatabaseClient
from src.config import BANKROLL, KELLY_FRACTION

logger = get_logger(__name__)

# Compact market labels used in digest one-liners.
_MARKET_SHORT: Dict[str, str] = {
    'player_points':                  'Pts',
    'player_rebounds':                'Reb',
    'player_assists':                 'Ast',
    'player_threes':                  '3PM',
    'player_points_rebounds_assists': 'PRA',
    'player_blocks':                  'Blk',
    'player_steals':                  'Stl',
}


_CROSS_GAME_KELLY_MULTIPLIER = 0.5   # cross-game parlays have no structural edge


def _parlay_kelly_stake(
    legs: List[Dict[str, Any]],
    joint_true_prob: float,
    combined_decimal: float,
    max_pct: float = 0.005,
) -> float:
    """
    Kelly-sized stake for a multi-leg parlay.

    SGPs (all legs share one event_id) may have a Gaussian Copula edge baked in,
    so they use the full KELLY_FRACTION.  Cross-game parlays have no hidden
    correlation value — the book prices independent legs correctly — so their
    Kelly fraction is halved to account for the higher variance with no structural
    offset.
    """
    ev = joint_true_prob * (combined_decimal - 1) - (1 - joint_true_prob)
    if ev <= 0 or combined_decimal <= 1:
        return 0.0

    event_ids = {leg.get('event_id', '') for leg in legs if leg.get('event_id')}
    is_sgp = len(event_ids) <= 1
    effective_kelly = KELLY_FRACTION if is_sgp else KELLY_FRACTION * _CROSS_GAME_KELLY_MULTIPLIER

    stake = BANKROLL * (ev / (combined_decimal - 1)) * effective_kelly
    stake = min(stake, BANKROLL * max_pct)
    return _camouflage_stake(stake)


def _camouflage_stake(stake: float) -> float:
    """Round to natural recreational-bettor increments to avoid sharp-profiling flags."""
    if stake <= 0:
        return 0.0
    if stake < 50:
        return max(5.0, round(stake / 5) * 5)
    if stake < 200:
        return round(stake / 25) * 25
    return round(stake / 50) * 50


def evaluate_and_alert(edge_data: Dict[str, Any], db: DatabaseClient, _bot: TelegramBotClient):
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
    # Kelly formula: f* = (b*p - q) / b  =  ev / (odds - 1)
    # where ev = model_prob*(odds-1) - (1-model_prob) = model_prob*odds - 1.
    # Using raw `edge` (prob difference) in the numerator was wrong: it
    # under-stakes high-odds bets and over-stakes low-odds bets.
    ev    = edge_data.get('ev', 0.0)
    stake = 0.0
    if odds > 1 and ev > 0:
        stake = BANKROLL * (ev / (odds - 1.0)) * KELLY_FRACTION
        stake = min(stake, BANKROLL * 0.05)  # hard cap: 5% per bet
    stake = _camouflage_stake(stake)

    MAX_DAILY_RISK = BANKROLL * 1.00
    MAX_PER_GAME   = BANKROLL * 0.40

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
        f"Book Bias:    {edge_data.get('feedback_factor_applied', 1.0):.2f}\n"
        + (
            f"⚡ Line Shift: {edge_data['line_shift_old_line']} → "
            f"{edge_data['line_shift_new_line']} "
            f"@ {edge_data['line_shift_sharp_book']}\n"
            if edge_data.get('line_shift_flag') else ""
        )
        + f"\n<b>Suggested Stake (Kelly):</b> ${stake:.0f}"
    )

    # Tier 2: queue for next digest flush (12 PM / 3 PM / 6 PM).
    side_char  = 'O' if side.upper() == 'OVER' else 'U'
    mkt_short  = _MARKET_SHORT.get(market, market[:3].title())
    title = (
        f"{player} {side_char}{line} {mkt_short} "
        f"@{book} {_parlay_american(odds)} | {edge:.1%} | ${stake:.0f}"
    )
    db.queue_pending_alert('prop', title, msg, priority=edge, game_date=game_date)
    logger.info(f"Alert queued: {player} {market} {side} {line} @ {book}")


def send_line_disagreement_alert(
    player: str, market: str, sharp_book: str, sharp_line: float,
    soft_book: str, soft_line: float, side: str, soft_odds: float,
    game_date: str, event_id: str, home_team: str, away_team: str,
    db: DatabaseClient, _bot: TelegramBotClient,
):
    """
    Fire a priority alert when a sharp book's line differs from a soft book
    by >= 1 full point.  These are the highest-ROI bets in the market — the
    soft book hasn't adjusted yet and is offering a line the sharp book has
    already moved past.
    """
    line_gap = abs(sharp_line - soft_line)

    # Dedup: don't re-alert on the same disagreement within a scan
    if db.check_recent_alert(player, market, soft_line, side, edge=0.0):
        return

    mkt_short = _MARKET_SHORT.get(market, market[:3].title())

    msg = (
        f"<b>⚡ SHARP LINE DISAGREEMENT</b>\n\n"
        f"Game: {away_team} @ {home_team}  ({game_date})\n"
        f"Player: {player}\n"
        f"Market: {market.replace('_', ' ').title()}\n\n"
        f"<b>{sharp_book}</b> line: {sharp_line}\n"
        f"<b>{soft_book}</b> line:  {soft_line}  ({line_gap:+.1f} gap)\n\n"
        f"Action: <b>{side} {soft_line}</b> on {soft_book}\n"
        f"Odds: {soft_odds}\n\n"
        f"A {line_gap:.1f}-point line gap on a Poisson distribution is massive.\n"
        f"Hammer this before {soft_book} adjusts."
    )

    title = (
        f"⚡ {player} {side[0]}{soft_line} {mkt_short} "
        f"@{soft_book} | {sharp_book} @ {sharp_line} ({line_gap:+.1f})"
    )

    # Queue as highest-priority alert (priority = line_gap to sort above normal edges)
    db.queue_pending_alert('prop', title, msg, priority=line_gap, game_date=game_date)
    logger.info(
        f"LINE DISAGREEMENT: {player} {market} — {sharp_book} {sharp_line} vs "
        f"{soft_book} {soft_line} ({side})"
    )


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
    db: Optional[DatabaseClient] = None,
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

    _PARLAY_MAX_PCT = {4: 0.005, 8: 0.0025}
    max_pct = _PARLAY_MAX_PCT.get(n, 0.0025)
    kelly_stake = _parlay_kelly_stake(legs, joint_true_prob, combined_decimal, max_pct=max_pct)

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

    # Thresholds for comparing against the book's actual SGP price:
    # BET if book offers ≥ 85% of theoretical baseline (≤15% SGP penalty)
    # SKIP if book offers < 60% of theoretical baseline (>40% SGP penalty)
    bet_threshold  = _parlay_american(combined_decimal * 0.85)
    skip_threshold = _parlay_american(combined_decimal * 0.60)

    lines.append(
        f"\nTheoretical Baseline: {_parlay_american(combined_decimal)}\n"
        f"⚠️ Books apply hidden SGP penalties — verify the book's actual SGP price.\n"
        f"✅ BET if book offers ≥ {bet_threshold} | ❌ SKIP if book offers < {skip_threshold}\n"
        f"True Hit Prob:  {joint_true_prob:.2%}\n"
        f"Book Implied:   {joint_book_prob:.2%}\n"
        f"Edge:           +{edge:.2%}\n"
        f"EV:             {ev:+.2%}\n"
        f"Suggested Stake (Kelly): ${kelly_stake:.0f} (cap: {max_pct:.2%} bankroll)"
    )

    msg = "\n".join(lines)
    if db is not None:
        # Tier 2: queue for digest.
        title = f"{n}-leg SGP | True: {joint_true_prob:.0%} | Edge: {edge:+.1%} | EV: {ev:+.1%}"
        db.queue_pending_alert('parlay', title, msg, priority=ev)
    else:
        # Fallback: send instantly when no DB is available.
        bot.send_message(msg)
    logger.info(
        f"{n}-leg high-prob parlay | "
        f"joint_prob={joint_true_prob:.2%} | edge={edge:.2%} | ev={ev:.2%}"
    )


# ── Game market alerts (Moneyline / Spread / Total) ───────────────────────────

_GAME_MARKET_LABELS: Dict[str, str] = {
    'h2h':     'Moneyline',
    'spreads': 'Spread',
    'totals':  'Total',
}


def send_game_market_alert(
    home_team: str,
    away_team: str,
    home_score: float,
    away_score: float,
    market: str,
    side: str,
    edge: float,
    ev: float,
    model_prob: float,
    book_prob: float,
    book_odds: float,
    book: str,
    game_date: str,
    event_id: str,
    line: float,
    db: DatabaseClient,
    _bot: TelegramBotClient,
    home_abbr: str = '',
    away_abbr: str = '',
) -> None:
    """
    Evaluate and send a Telegram alert for a game market edge
    (Moneyline, Spread, or Total).

    `side` doubles as both the DB deduplication key and the display label
    (e.g. "Denver Nuggets", "Boston Celtics -6.5", "Over").
    `line` is 0.0 for moneyline, the spread value for spreads, and the
    combined total for totals.
    """
    matchup      = f"{away_team} @ {home_team}"
    market_label = _GAME_MARKET_LABELS.get(market, market.title())

    if db.check_recent_alert(matchup, market, line, side, edge):
        logger.info(f"Skipping duplicate game market alert: {matchup} {market_label} {side}")
        return

    # Kelly sizing — capped at 3 % of bankroll (game markets are more efficient
    # than player props; model uncertainty is higher for team-level projections).
    stake = 0.0
    if book_odds > 1.0 and ev > 0.0:
        stake = BANKROLL * (ev / (book_odds - 1.0)) * KELLY_FRACTION
        stake = min(stake, BANKROLL * 0.03)
    stake = _camouflage_stake(stake)

    MAX_DAILY_RISK = BANKROLL * 1.00
    MAX_PER_GAME   = BANKROLL * 0.40

    with db.get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT SUM(stake) as total_risk FROM alerts_sent WHERE date(timestamp) = date('now')"
        )
        row = cursor.fetchone()
        current_daily_risk = float(row['total_risk'] or 0.0) if row else 0.0
        if current_daily_risk + stake > MAX_DAILY_RISK:
            logger.warning(
                f"Skipping {matchup} — daily risk limit "
                f"({current_daily_risk:.2f}/{MAX_DAILY_RISK:.2f})"
            )
            return

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
                    f"Skipping {matchup} — per-game risk limit "
                    f"({current_game_risk:.2f}/{MAX_PER_GAME:.2f})"
                )
                return

    db.insert_alert(
        player_name=matchup, market=market, line=line, side=side,
        edge=edge, book=book, odds=book_odds, stake=stake,
        game_date=game_date, event_id=event_id,
    )

    _h_abbr = home_abbr or home_team.split()[-1][:3].upper()
    _a_abbr = away_abbr or away_team.split()[-1][:3].upper()
    line_str = f" {line:+.1f}" if market != 'h2h' else ''

    msg = (
        f"<b>⚡ NBA GAME MARKET EDGE</b>\n\n"
        f"Game: {_a_abbr} @ {_h_abbr}  ({game_date})\n"
        f"Proj: {_a_abbr} {away_score:.1f} – {_h_abbr} {home_score:.1f}\n\n"
        f"Market: {market_label}{line_str}\n"
        f"Side:   {side}\n"
        f"Best Book: {book}  |  Odds: {_parlay_american(book_odds)}\n\n"
        f"Model Prob:  {model_prob:.3f}\n"
        f"Book Prob:   {book_prob:.3f}\n"
        f"Edge:  {edge:+.2%}\n"
        f"EV:    {ev:+.2%}\n\n"
        f"<b>Suggested Stake (Kelly):</b> ${stake:.0f}"
    )

    # Tier 2: queue for digest.
    side_short = side if len(side) <= 14 else side[:13] + "…"
    title = (
        f"{_a_abbr}@{_h_abbr} {market_label}{line_str} {side_short} "
        f"@{book} {_parlay_american(book_odds)} | {edge:+.1%} | ${stake:.0f}"
    )
    db.queue_pending_alert('game_market', title, msg, priority=edge, game_date=game_date)
    logger.info(f"Game market alert queued: {matchup} {market_label} {side} edge={edge:.2%}")
