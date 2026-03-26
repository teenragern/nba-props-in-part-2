"""
Multi-leg combo (parlay) generator — V2 (calibrated).

Key changes from V1:
  • Every leg is calibrated through the empirical correction model before
    joint probability is computed. The 55-65% overconfidence band is
    compressed to match actual hit rates.
  • Per-leg minimums: each leg must independently show ≥5% edge AND
    ≥55% calibrated probability. No more stacking marginal props.
  • Max legs capped at 3. The math on 4+ leggers doesn't work at current
    calibration levels (0.55^4 = 9.1% — below break-even on any book).
  • Heavier reality tax: 0.90 for 2-leg, 0.82 for 3-leg.
  • Slate-wide high-prob parlays (4-leg, 8-leg) removed entirely —
    they were the primary source of parlay losses.
  • Cross-game only: legs must come from different games (Rainbet does
    not allow same-game stacking). One best parlay per scan.
  • Legs must come from different players AND different markets
    (no double-dipping on correlated outcomes like PTS + PRA).

Rules:
  • One best cross-game parlay per scan.
  • All legs must be from different games (enforced by _compatible).
  • All legs must be from different players.
  • All legs must be from different market families (PTS and PRA conflict).
  • Same-team 2-leg pairs use cross-player historical correlation from DB.
  • Opposing-team / 3-player combos assume independence.
"""

from itertools import combinations
from math import prod
from typing import List, Dict, Any, Optional, Set

from src.models.sgp_correlations import adjust_joint_probability
from src.models.calibration_model import calibrate_prob
from src.clients.telegram_bot import TelegramBotClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Tuning knobs ──────────────────────────────────────────────────────────────

MAX_LEGS        = 3      # hard cap: no 4+ leggers until calibration improves
MAX_INPUT_EDGES = 12     # top-N edges per game considered as candidates
COMBO_EDGE_MIN  = 0.08   # minimum joint edge to alert (was 0.04)

# Per-leg quality gates — each leg must pass BOTH independently
PER_LEG_EDGE_MIN    = 0.05   # each leg must show ≥5% edge after calibration
PER_LEG_PROB_MIN    = 0.60   # each leg must have ≥60% calibrated hit probability (was 0.55)
PER_LEG_IMPLIED_MAX = 0.55   # skip legs where book already prices >55% (no edge room)

# Reality tax: accounts for correlated blowouts, last-minute news,
# model miscalibration, and unknown unknowns.
_REALITY_TAX: Dict[int, float] = {2: 0.92, 3: 0.82}

# Market families — legs from the same family are correlated and shouldn't stack
_MARKET_FAMILY: Dict[str, str] = {
    'player_points':                  'scoring',
    'player_threes':                  'scoring',
    'player_points_rebounds_assists': 'scoring',  # PRA overlaps with PTS
    'player_rebounds':                'rebounds',
    'player_assists':                 'assists',
    'player_blocks':                  'defense',
    'player_steals':                  'defense',
}

_MARKET_LABELS: Dict[str, str] = {
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


def _leg_passes_quality_gate(leg: Dict) -> bool:
    """
    Return True only if this leg is independently strong enough to
    be included in a parlay. This is the single most important filter.
    """
    raw_prob = leg.get('model_prob', 0.0)
    implied  = leg.get('implied_prob', 0.0)

    # Use calibrated probability for the quality check
    cal_prob = calibrate_prob(raw_prob)

    if cal_prob < PER_LEG_PROB_MIN:
        return False

    # Edge must survive calibration
    cal_edge = cal_prob - implied
    if cal_edge < PER_LEG_EDGE_MIN:
        return False

    # Skip props where the book already prices this heavily —
    # no room for real edge, we're just riding vig
    if implied > PER_LEG_IMPLIED_MAX:
        return False

    # Skip fragile edges (already flagged by edge_ranker)
    if leg.get('fragile', False):
        return False

    return True


def _compatible(legs: List[Dict]) -> bool:
    """
    Return True only when:
      1. All legs are from different players.
      2. All legs are from different market families (no PTS + PRA stacking).
      3. All legs are from different games (Rainbet does not allow same-game stacking).
    """
    players = set()
    families: Set[str] = set()
    events: Set[str] = set()

    for leg in legs:
        pid = leg['player_id']
        if pid in players:
            return False
        players.add(pid)

        family = _MARKET_FAMILY.get(leg.get('market', ''), leg.get('market', ''))
        if family in families:
            return False
        families.add(family)

        eid = leg.get('event_id', '')
        if eid in events:
            return False
        events.add(eid)

    return True


def _combo_edge(legs: List[Dict], db=None) -> Dict:
    """
    Compute joint edge for a combo using CALIBRATED probabilities.

    Same-team, different-player 2-leg combos: look up cross-player
    historical correlation from DB.
    All other multi-player combos: assume independence.
    """
    # Calibrate each leg's probability before computing joint
    cal_probs = [calibrate_prob(leg.get('model_prob', 0.5)) for leg in legs]
    implied_probs = [leg['implied_prob'] for leg in legs]

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
                    cal_probs[0], cal_probs[1], corr,
                    mean_a=legs[0].get('mean'), mean_b=legs[1].get('mean'),
                    line_a=legs[0].get('line'), line_b=legs[1].get('line'),
                    side_a=legs[0].get('side', 'OVER'),
                    side_b=legs[1].get('side', 'OVER'),
                )
                jb = prod(implied_probs)
                return {
                    'joint_true_prob':     jt,
                    'joint_book_prob':     jb,
                    'sgp_edge':            jt - jb,
                    'correlation_applied': corr,
                }

    # Default: assume independence with calibrated probs
    jt = prod(cal_probs)
    jb = prod(implied_probs)
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
        raw_p  = leg.get('model_prob', 0)
        cal_p  = calibrate_prob(raw_p)
        lines.append(
            f"• <b>{leg['player_id']}</b> {leg['side']} {leg['line']} {market}"
            f" @ {leg.get('book', '')} ({_american(leg['odds'])})"
            f" — {cal_p:.0%} cal"
        )
    lines.append(
        f"\nCombined: {_american(combined)} | Edge: {edge:.1%}"
        f" | Hit Prob: {joint_prob:.1%}"
        f" | Reality-taxed"
    )
    return "\n".join(lines)


def generate_and_alert_combos(
    actionable: List[Dict[str, Any]],
    bot: TelegramBotClient,
    db=None,
) -> None:
    """
    Generate the best parlay for each game on the slate.

    V2 changes:
      - Each leg must independently pass quality gates (calibrated prob + edge).
      - Max 3 legs.
      - No slate-wide high-prob parlays (removed — they were money incinerators).
      - Market family diversification enforced.
    """
    if len(actionable) < 2:
        return

    # ── Step 1: Filter to parlay-quality legs only ────────────────────────
    parlay_eligible = [e for e in actionable if _leg_passes_quality_gate(e)]

    if len(parlay_eligible) < 2:
        logger.info(
            f"Parlay generation: only {len(parlay_eligible)} legs pass quality gates "
            f"(need ≥2). No parlays today."
        )
        return

    logger.info(
        f"Parlay pool: {len(parlay_eligible)}/{len(actionable)} edges "
        f"pass per-leg quality gates."
    )

    # ── Step 2: Build cross-game combos from the full pool ───────────────
    # Legs from the same game are blocked by _compatible (Rainbet restriction).
    pool = sorted(
        parlay_eligible,
        key=lambda e: e.get('risk_adjusted_ev', 0),
        reverse=True,
    )[:MAX_INPUT_EDGES]

    best: Dict = {}

    for size in range(2, MAX_LEGS + 1):
        for combo in combinations(pool, size):
            legs = list(combo)
            if not _compatible(legs):
                continue

            result = _combo_edge(legs, db=db)
            raw_joint = result.get('joint_true_prob', 0)

            dampening = _REALITY_TAX.get(size, 0.80)
            taxed_joint = raw_joint * dampening

            combo_edge = taxed_joint - result.get('joint_book_prob', 0)
            if combo_edge < COMBO_EDGE_MIN:
                continue

            if not best or combo_edge > best['edge']:
                best = {
                    'legs':       legs,
                    'edge':       combo_edge,
                    'joint_prob': taxed_joint,
                }

    if best:
        away = best['legs'][0].get('away_team', '')
        home = best['legs'][0].get('home_team', '')
        msg = _format_combo(best['legs'], best['edge'], best['joint_prob'], away, home)
        bot.send_message(msg)
        logger.info(
            f"Cross-game parlay sent: {len(best['legs'])}-leg | "
            f"edge={best['edge']:.2%} | joint_prob={best['joint_prob']:.2%}"
        )
    else:
        logger.info("No cross-game parlay met the edge threshold today.")
