"""
Multi-leg combo (parlay) generator.

After single-leg edges are ranked, this module composes 2–4 leg combos
from the top actionable edges, computes joint probability (with correlation
adjustments for same-player combos), and sends the best combos to Telegram.
"""

from itertools import combinations
from math import prod
from typing import List, Dict, Any

from src.models.sgp_correlations import get_sgp_edge, adjust_joint_probability
from src.clients.telegram_bot import TelegramBotClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

MAX_LEGS           = 4    # maximum legs per combo
MAX_INPUT_EDGES    = 12   # only consider top-N edges as candidates
MAX_COMBOS_TO_SEND = 5    # anti-spam: max Telegram messages per scan
COMBO_EDGE_MIN     = 0.03  # minimum joint edge to alert on

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
    """Return False if the same player+market appears on opposing sides."""
    seen: Dict = {}
    for leg in legs:
        key = (leg['player_id'], leg['market'])
        if key in seen and seen[key] != leg['side']:
            return False
        seen[key] = leg['side']
    return True


def _combo_edge(legs: List[Dict], db=None) -> Dict:
    """
    Compute joint edge for a combo.

    Same-player combos: use intra-player SGP correlation model.
    Same-team, different-player combos (2-leg): look up cross-player
      historical correlation from DB (PG assists ↔ C/PF points, etc.).
    All other multi-player combos: assume independence.
    """
    sgp_legs = [
        {
            'market':       leg['market'],
            'side':         leg['side'],
            'prob':         leg['model_prob'],
            'implied_prob': leg['implied_prob'],
        }
        for leg in legs
    ]

    player_ids = {leg['player_id'] for leg in legs}

    if len(player_ids) == 1:
        # Same player — intra-player correlation model
        return get_sgp_edge(sgp_legs, player_logs=None)

    # Two legs from different players on the same team → cross-player lookup
    if db is not None and len(legs) == 2 and len(player_ids) == 2:
        team_a = legs[0].get('team_name', '')
        team_b = legs[1].get('team_name', '')
        if team_a and team_a == team_b:
            pa, pb = legs[0]['player_id'], legs[1]['player_id']
            ma, mb = legs[0]['market'],    legs[1]['market']
            corr = db.get_cross_player_correlation(team_a, pa, pb, ma, mb)
            if corr is None:
                # Try swapped order (a/b symmetric in DB key)
                corr = db.get_cross_player_correlation(team_a, pb, pa, mb, ma)
            if corr is not None:
                jt = adjust_joint_probability(
                    legs[0]['model_prob'], legs[1]['model_prob'], corr)
                jb = prod(l['implied_prob'] for l in sgp_legs)
                return {
                    'joint_true_prob':     jt,
                    'joint_book_prob':     jb,
                    'sgp_edge':            jt - jb,
                    'correlation_applied': corr,
                }

    # Default: assume independence
    jt = prod(l['prob']         for l in sgp_legs)
    jb = prod(l['implied_prob'] for l in sgp_legs)
    return {
        'joint_true_prob':     jt,
        'joint_book_prob':     jb,
        'sgp_edge':            jt - jb,
        'correlation_applied': 0.0,
    }


def _format_combo(legs: List[Dict], edge: float, joint_prob: float) -> str:
    combined = prod(leg['odds'] for leg in legs)
    lines = [f"🎯 <b>{len(legs)}-Leg Combo</b>\n"]
    for leg in legs:
        market = _MARKET_LABELS.get(leg['market'], leg['market'])
        lines.append(
            f"• <b>{leg['player_id']}</b> {leg['side']} {leg['line']} {market}"
            f" @ {leg.get('book', '')} ({_american(leg['odds'])})"
        )
    lines.append(f"\nCombined Odds: {_american(combined)} | Edge: {edge:.1%}")
    lines.append(f"Joint Probability: {joint_prob:.1%}")
    return "\n".join(lines)


def generate_and_alert_combos(
    actionable: List[Dict[str, Any]],
    bot: TelegramBotClient,
    db=None,
) -> None:
    """
    Generate 2–4 leg combos from the top actionable edges and send to Telegram.

    Args:
        actionable: ranked list of edge dicts (from rank_edges), best first.
        bot:        Telegram client.
    """
    if len(actionable) < 2:
        return

    pool = actionable[:MAX_INPUT_EDGES]
    candidates = []

    for size in range(2, MAX_LEGS + 1):
        for combo in combinations(pool, size):
            legs = list(combo)
            if not _compatible(legs):
                continue
            result = _combo_edge(legs, db=db)
            combo_edge = result.get('sgp_edge', 0)
            if combo_edge < COMBO_EDGE_MIN:
                continue
            candidates.append({
                'legs':       legs,
                'edge':       combo_edge,
                'joint_prob': result.get('joint_true_prob', 0),
            })

    candidates.sort(key=lambda x: x['edge'], reverse=True)

    sent = 0
    for c in candidates:
        if sent >= MAX_COMBOS_TO_SEND:
            break
        msg = _format_combo(c['legs'], c['edge'], c['joint_prob'])
        bot.send_message(msg)
        logger.info(
            f"Combo alert sent: {len(c['legs'])}-leg, edge={c['edge']:.2%}, "
            f"joint_prob={c['joint_prob']:.2%}"
        )
        sent += 1

    if sent:
        logger.info(f"Total combo alerts sent this scan: {sent}")
    else:
        logger.info("No qualifying combos found this scan.")
