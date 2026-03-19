"""
Multi-leg combo (parlay) generator.

After single-leg edges are ranked, this module composes 2–4 leg combos
from the top actionable edges per game, computes joint probability (with
correlation adjustments for same-team combos), and sends the best combo
for each game on the slate.

Rules:
  • One parlay per game — the highest-edge combo across all leg counts.
  • All legs must be from different players (no intra-player SGPs).
  • Same-team 2-leg pairs use cross-player historical correlation from DB.
  • Opposing-team / 3+ player combos assume independence.
"""

from itertools import combinations
from math import prod
from typing import List, Dict, Any

from src.models.sgp_correlations import adjust_joint_probability
from src.clients.telegram_bot import TelegramBotClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

MAX_LEGS        = 4     # maximum legs per combo
MAX_INPUT_EDGES = 15    # top-N edges per game considered as candidates
COMBO_EDGE_MIN  = 0.02  # minimum joint edge to alert on

_MARKET_LABELS: Dict[str, str] = {
    'player_points':                  'Points',
    'player_rebounds':                'Rebounds',
    'player_assists':                 'Assists',
    'player_threes':                  'Threes',
    'player_points_rebounds_assists': 'PRA',
}


def _american(decimal_odds: float) -> str:
    """Convert decimal odds to American odds string."""
    if decimal_odds >= 2.0:
        return f"+{int((decimal_odds - 1) * 100)}"
    if decimal_odds <= 1.0:
        return "N/A"
    return f"-{int(100 / (decimal_odds - 1))}"


def _compatible(legs: List[Dict]) -> bool:
    """
    Return True only when all legs are from different players.
    Intra-player parlays (same player, different markets) are excluded —
    books price those in their SGP builder with hidden holds.
    """
    return len({leg['player_id'] for leg in legs}) == len(legs)


def _combo_edge(legs: List[Dict], db=None) -> Dict:
    """
    Compute joint edge for a combo.

    Same-team, different-player 2-leg combos: look up cross-player
      historical correlation from DB (PG assists ↔ C/PF points, etc.).
    All other multi-player combos: assume independence.
    """
    sgp_legs = [
        {
            'market':       leg['market'],
            'side':         leg['side'],
            'prob':         leg['model_prob'],
            'implied_prob': leg['implied_prob'],
            'mean':         leg.get('mean'),
            'line':         leg.get('line'),
        }
        for leg in legs
    ]

    player_ids = {leg['player_id'] for leg in legs}

    # Two legs from different players on the same team → cross-player lookup
    if db is not None and len(legs) == 2 and len(player_ids) == 2:
        team_a = legs[0].get('team_name', '')
        team_b = legs[1].get('team_name', '')
        if team_a and team_a == team_b:
            pa, pb = legs[0]['player_id'], legs[1]['player_id']
            ma, mb = legs[0]['market'],    legs[1]['market']
            corr = db.get_cross_player_correlation(team_a, pa, pb, ma, mb)
            if corr is None:
                corr = db.get_cross_player_correlation(team_a, pb, pa, mb, ma)
            if corr is not None:
                jt = adjust_joint_probability(
                    legs[0]['model_prob'], legs[1]['model_prob'], corr,
                    mean_a=legs[0].get('mean'), mean_b=legs[1].get('mean'),
                    line_a=legs[0].get('line'), line_b=legs[1].get('line'),
                )
                jb = prod(l['implied_prob'] for l in sgp_legs)
                return {
                    'joint_true_prob':     jt,
                    'joint_book_prob':     jb,
                    'sgp_edge':            jt - jb,
                    'correlation_applied': corr,
                }

    # Default: assume independence across different players / teams
    jt = prod(l['prob']         for l in sgp_legs)
    jb = prod(l['implied_prob'] for l in sgp_legs)
    return {
        'joint_true_prob':     jt,
        'joint_book_prob':     jb,
        'sgp_edge':            jt - jb,
        'correlation_applied': 0.0,
    }


def _format_combo(legs: List[Dict], edge: float, joint_prob: float,
                  away_team: str, home_team: str) -> str:
    combined = prod(leg['odds'] for leg in legs)
    header = f"🎯 <b>{len(legs)}-Leg Parlay — {away_team} @ {home_team}</b>\n"
    lines = [header]
    for leg in legs:
        market = _MARKET_LABELS.get(leg['market'], leg['market'])
        lines.append(
            f"• <b>{leg['player_id']}</b> {leg['side']} {leg['line']} {market}"
            f" @ {leg.get('book', '')} ({_american(leg['odds'])})"
        )
    lines.append(f"\nCombined: {_american(combined)} | Edge: {edge:.1%} | Hit Prob: {joint_prob:.1%}")
    return "\n".join(lines)


def generate_and_alert_combos(
    actionable: List[Dict[str, Any]],
    bot: TelegramBotClient,
    db=None,
) -> None:
    """
    Generate the best diverse parlay for each game on the slate.

    Groups actionable edges by event_id, then for each game finds the
    highest-edge 2–4 leg combo where every leg comes from a different player.
    Sends exactly one Telegram message per game (skips games with < 2 edges).

    Args:
        actionable: ranked list of edge dicts (from rank_edges), best first.
        bot:        Telegram client.
    """
    if len(actionable) < 2:
        return

    # Group edges by game
    games: Dict[str, List[Dict]] = {}
    for edge in actionable:
        eid = edge.get('event_id', 'unknown')
        games.setdefault(eid, []).append(edge)

    sent = 0
    for event_id, edges in games.items():
        if len(edges) < 2:
            continue

        pool = edges[:MAX_INPUT_EDGES]
        best: Dict = {}

        for size in range(2, MAX_LEGS + 1):
            for combo in combinations(pool, size):
                legs = list(combo)
                if not _compatible(legs):
                    continue
                result = _combo_edge(legs, db=db)
                combo_edge = result.get('sgp_edge', 0)
                if combo_edge < COMBO_EDGE_MIN:
                    continue
                if not best or combo_edge > best['edge']:
                    best = {
                        'legs':       legs,
                        'edge':       combo_edge,
                        'joint_prob': result.get('joint_true_prob', 0),
                    }

        if not best:
            continue

        away = best['legs'][0].get('away_team', '')
        home = best['legs'][0].get('home_team', '')
        msg = _format_combo(best['legs'], best['edge'], best['joint_prob'], away, home)
        bot.send_message(msg)
        logger.info(
            f"Game parlay sent: {away} @ {home} | "
            f"{len(best['legs'])}-leg | edge={best['edge']:.2%} | "
            f"joint_prob={best['joint_prob']:.2%}"
        )
        sent += 1

    logger.info(f"Game parlays sent: {sent}/{len(games)} games on slate.")
