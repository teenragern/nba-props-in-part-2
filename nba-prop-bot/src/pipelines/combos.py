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
  • SGPs allowed: same-game OVER/OVER and OVER/UNDER combos are valid.
    Rainbet only bans same-game double-unders (all legs UNDER same match).
  • Legs must come from different players AND different markets
    (no double-dipping on the same outcome like LeBron PTS + LeBron PRA).
  • Same-Player, Same-Team, and Cross-Team combinations all dynamically
    apply the Gaussian Copula correlation models.

Rules:
  • One best parlay per scan (SGP or cross-game).
  • Same-player legs must be from different market families.
  • Same-game double-unders blocked (Rainbet rule).
"""

import dateutil.parser
from itertools import combinations
from math import prod
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime

from src.models.sgp_correlations import (
    adjust_joint_probability,
    get_pairwise_correlation,
    get_cross_team_default_corr
)
from src.models.calibration_model import calibrate_prob
from src.clients.telegram_bot import TelegramBotClient
from src.pipelines.send_alerts import _parlay_kelly_stake
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Tuning knobs ──────────────────────────────────────────────────────────────

MAX_LEGS        = 3      # hard cap: no 4+ leggers until calibration improves
MAX_INPUT_EDGES = 12     # top-N edges per game considered as candidates
COMBO_EDGE_MIN  = 0.04   # minimum joint edge to alert (was 0.04)

# ── 4-Leg Diversified Parlays (max 2 non-overlapping tickets per day) ─────────
FOUR_LEG_MAX_TICKETS  = 5      # at most 5 tickets per scan/day
FOUR_LEG_MAX_INPUT    = 20     # wider pool — need enough for 2 full tickets
FOUR_LEG_REALITY_TAX  = 0.75  # between 3-leg (0.82) and 8-leg (0.65)
FOUR_LEG_KELLY_CAP    = 0.006  # hard cap: 0.6% bankroll (midpoint of 0.5–0.75%)

# ── Slate Ultimate (8-Leg Golden Ticket) ──────────────────────────────────────
SLATE_ULTIMATE_LEGS      = 8
SLATE_ULTIMATE_MIN_GAMES = 8      # silent on slates with < 8 qualifying games
SLATE_ULTIMATE_MAX_INPUT = 12     # top-N games considered in combo search
SLATE_ULTIMATE_REALITY   = 0.65  # 8-leg reality dampening (accounts for news, variance)
SLATE_ULTIMATE_KELLY_CAP = 0.0015 # hard cap: 0.15% bankroll (midpoint of 0.10–0.25%)

# Per-leg quality gates — each leg must pass BOTH independently
PER_LEG_EDGE_MIN    = 0.05   # each leg must show ≥5% edge after calibration
PER_LEG_PROB_MIN    = 0.55   # each leg must have ≥55% calibrated hit probability (was 0.55)
PER_LEG_IMPLIED_MAX = 0.55   # skip legs where book already prices >55% (no edge room)
PER_LEG_IMPLIED_MIN = 0.15   # skip lottery-ticket legs priced below ~15% implied (~+550);
                              # these are almost always alt-line data errors masquerading as edge

# Reality tax: accounts for correlated blowouts, last-minute news,
# model miscalibration, and unknown unknowns.
# Cross-game parlays get full tax; SGPs already have Gaussian Copula
# correlation adjustment so they get a lighter dampening.
_REALITY_TAX_CROSS: Dict[int, float] = {2: 0.92, 3: 0.82, 4: 0.75}
_REALITY_TAX_SGP:   Dict[int, float] = {2: 0.96, 3: 0.90, 4: 0.84}

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


def _leg_passes_quality_gate(leg: Dict, playoff_mode: bool = False) -> bool:
    """
    Return True only if this leg is independently strong enough to
    be included in a parlay. This is the single most important filter.
    """
    raw_prob = leg.get('model_prob', 0.0)
    implied  = leg.get('implied_prob', 0.0)

    # Use calibrated probability for the quality check
    cal_prob = calibrate_prob(raw_prob, playoff_mode=playoff_mode)

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

    # Skip lottery-ticket legs (e.g. +2400 alt lines) — if the book prices this
    # at <15% implied, the model's high probability is almost certainly a data
    # error (wrong line, alt-line confusion, wrong stat units).
    if implied < PER_LEG_IMPLIED_MIN:
        return False

    # Skip fragile edges (already flagged by edge_ranker)
    if leg.get('fragile', False):
        return False

    return True


_TIP_OFF_WINDOW_SECS = 5400  # 90 minutes


def _compatible(legs: List[Dict]) -> bool:
    """
    Return True only when passing structural parlay tests:
      1. Prevent double-dipping same-market families for the SAME player.
      2. Allow same-player different-markets, or different-players same-markets.
      3. No same-game double-unders (Rainbet rule: all legs UNDER in same match
         is disallowed; OVER/OVER and OVER/UNDER SGPs are fine).
      4. Cross-game legs must tip off within 90 minutes of each other —
         prevents late-scratch risk on games that start hours apart.
    """
    player_families: Set[Tuple[str, str]] = set()

    for leg in legs:
        pid = leg['player_id']
        family = _MARKET_FAMILY.get(leg.get('market', ''), leg.get('market', ''))
        
        # Disallow the same player stacking correlated markets (e.g. LeBron PTS and PRA)
        key = (pid, family)
        if key in player_families:
            return False
        player_families.add(key)

    # Group legs by event for the next two checks.
    event_legs: Dict[str, List] = {}
    for leg in legs:
        eid = leg.get('event_id', '')
        if eid:
            event_legs.setdefault(eid, []).append(leg)

    # Rainbet rule: same-game combos where every leg is UNDER are illegal.
    for ev_legs in event_legs.values():
        if len(ev_legs) >= 2 and all(l.get('side', '') == 'UNDER' for l in ev_legs):
            return False

    # Tip-off window: cross-game legs must start within 90 minutes of each other.
    # Skipped for SGPs (only one unique event_id).
    if len(event_legs) > 1:
        times: List[datetime] = []
        for leg in legs:
            ct = leg.get('commence_time')
            if isinstance(ct, datetime):
                times.append(ct)
            elif isinstance(ct, str) and ct:
                try:
                    times.append(dateutil.parser.isoparse(ct))
                except Exception:
                    pass
        if len(times) >= 2:
            spread = (max(times) - min(times)).total_seconds()
            if spread > _TIP_OFF_WINDOW_SECS:
                return False

    return True


def _combo_edge(legs: List[Dict], db=None, playoff_mode: bool = False) -> Dict:
    """
    Compute joint edge for a combo using CALIBRATED probabilities.

    Dynamically applies Gaussian Copula adjustments for Same-Player, 
    Same-Team, and Cross-Team SGP sequences. Assumes independence for 
    cross-game links.
    """
    cal_probs = [calibrate_prob(leg.get('model_prob', 0.5), playoff_mode=playoff_mode) for leg in legs]
    implied_probs = [leg['implied_prob'] for leg in legs]

    joint_true = cal_probs[0]
    corr_applied = 0.0

    for i in range(1, len(legs)):
        leg_a, leg_b = legs[i - 1], legs[i]
        corr = 0.0

        # Only inject Copula correlation modeling if both legs share a game context
        if leg_a.get('event_id') and leg_a.get('event_id') == leg_b.get('event_id'):
            pa, pb = leg_a['player_id'], leg_b['player_id']
            ma, mb = leg_a['market'], leg_b['market']
            team_a, team_b = leg_a.get('team_name', ''), leg_b.get('team_name', '')

            if pa == pb:
                # Same-player SGP (e.g. Points + Assists)
                corr = get_pairwise_correlation(ma, mb, playoff_mode=playoff_mode)
            elif team_a and team_a == team_b and db is not None:
                # Same-team, cross-player SGP (e.g. Point Guard Assists + Center Points)
                db_corr = db.get_cross_player_correlation(team_a, pa, pb, ma, mb)
                if db_corr is None:
                    db_corr = db.get_cross_player_correlation(team_a, pb, pa, mb, ma)
                if db_corr is not None:
                    corr = db_corr
            elif team_a and team_b and team_a != team_b:
                # Cross-team SGP (e.g. finite rebounding logic).
                # Prefer the matchup's DB record — in playoff_mode with ≥3 series
                # games, it holds an empirical Pearson. Fall back to the league
                # default when no record exists.
                corr = None
                if db is not None:
                    matchup_key = "|".join(sorted([team_a.lower(), team_b.lower()]))
                    corr = db.get_cross_team_correlation(matchup_key, pa, pb, ma, mb)
                    if corr is None:
                        corr = db.get_cross_team_correlation(matchup_key, pb, pa, mb, ma)
                if corr is None:
                    corr = get_cross_team_default_corr(ma, mb)

        # Copula applied to first pair; subsequent legs sequence via Bivariate Normal
        mean_a = leg_a.get('mean') if i == 1 else None
        line_a = leg_a.get('line') if i == 1 else None

        joint_true = adjust_joint_probability(
            joint_true, cal_probs[i], corr,
            mean_a=mean_a, mean_b=leg_b.get('mean'),
            line_a=line_a, line_b=leg_b.get('line'),
            side_a=leg_a.get('side', 'OVER'),
            side_b=leg_b.get('side', 'OVER'),
        )
        corr_applied += corr

    if len(legs) > 1:
        corr_applied /= (len(legs) - 1)

    jb = prod(implied_probs)
    return {
        'joint_true_prob':     joint_true,
        'joint_book_prob':     jb,
        'sgp_edge':            joint_true - jb,
        'correlation_applied': corr_applied,
    }


def _format_combo(legs: List[Dict], edge: float, joint_prob: float, playoff_mode: bool = False) -> str:
    combined = prod(leg['odds'] for leg in legs)
    event_ids = {leg.get('event_id', '') for leg in legs}
    if len(event_ids) == 1:
        away = legs[0].get('away_team', '')
        home = legs[0].get('home_team', '')
        title = f"SGP — {away} @ {home}"
    else:
        title = f"Cross-Game Parlay ({len(event_ids)} games)"
    header = f"🎯 <b>{len(legs)}-Leg {title}</b>\n"
    lines = [header]
    for leg in legs:
        market = _MARKET_LABELS.get(leg['market'], leg['market'])
        raw_p  = leg.get('model_prob', 0)
        cal_p  = calibrate_prob(raw_p, playoff_mode=playoff_mode)
        lines.append(
            f"• <b>{leg['player_id']}</b> {leg['side']} {leg['line']} {market}"
            f" @ {leg.get('book', '')} ({_american(leg['odds'])})"
            f" — {cal_p:.0%} cal"
        )
    stake = _parlay_kelly_stake(legs, joint_prob, combined)
    lines.append(
        f"\nCombined: {_american(combined)} | Edge: {edge:.1%}"
        f" | Hit Prob: {joint_prob:.1%} | Reality-taxed"
        f"\n<b>Suggested Stake (Kelly):</b> ${stake:.0f}"
    )
    return "\n".join(lines)


def _four_leg_sent_today_count(db) -> int:
    """Return how many 4-leg parlay tickets have been queued today."""
    if db is None:
        return 0
    with db.get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) FROM pending_alerts
            WHERE alert_type = 'four_leg_parlay'
              AND date(created_at) = date('now', 'localtime')
            """
        )
        row = cursor.fetchone()
        return row[0] if row else 0


def _format_four_leg_parlay(
    legs: List[Dict], edge: float, joint_prob: float, stake: float, ticket_num: int,
    playoff_mode: bool = False,
) -> str:
    combined = prod(leg['odds'] for leg in legs)
    n_events = len({leg.get('event_id', '') for leg in legs})
    title_tag = 'SGP' if n_events == 1 else f'{n_events} games'
    header    = f"🃏 <b>4-Leg Parlay #{ticket_num} ({title_tag})</b>\n"
    lines     = [header]
    for leg in legs:
        market = _MARKET_LABELS.get(leg['market'], leg['market'])
        cal_p  = calibrate_prob(leg.get('model_prob', 0), playoff_mode=playoff_mode)
        away   = leg.get('away_team', '')
        home   = leg.get('home_team', '')
        game   = f"[{away}@{home}]" if away and home else ''
        lines.append(
            f"• <b>{leg['player_id']}</b> {leg['side']} {leg['line']} {market}"
            f" @ {leg.get('book', '')} ({_american(leg['odds'])})"
            f" {game} — {cal_p:.0%} cal"
        )
    lines.append(
        f"\nCombined: {_american(combined)} | Hit Prob: {joint_prob:.1%} | Edge: {edge:+.2%}"
        f"\n<b>Suggested Stake (Kelly):</b> ${stake:.0f}"
    )
    return "\n".join(lines)


def generate_four_leg_parlays(
    actionable: List[Dict[str, Any]],
    bot: TelegramBotClient,
    db=None,
    playoff_mode: bool = False,
) -> None:
    """
    Generate up to 2 non-overlapping 4-leg parlays per day.

    Architecture:
      • Each leg must pass the standard quality gate (≥60% cal prob, ≥5% edge).
      • Ticket 1 uses the highest-EV legs; ticket 2 draws from the remainder.
      • No player may appear in more than one ticket (excluded_players set).
      • Uses existing _compatible() — SGPs allowed, tip-off window enforced.
      • Reality tax: 0.75 (tighter than 3-leg due to compounding variance).
      • Kelly staking: hard cap at 0.6% of bankroll.
      • At most FOUR_LEG_MAX_TICKETS tickets queued per calendar day.
    """
    already_sent = _four_leg_sent_today_count(db)
    remaining_slots = FOUR_LEG_MAX_TICKETS - already_sent
    if remaining_slots <= 0:
        logger.info("4-Leg Parlays: daily limit reached — skipping.")
        return

    parlay_eligible = [e for e in actionable if _leg_passes_quality_gate(e, playoff_mode=playoff_mode)]
    if len(parlay_eligible) < 4:
        logger.info(
            f"4-Leg Parlays: only {len(parlay_eligible)} eligible legs (need ≥4). Skipping."
        )
        return

    pool = sorted(
        parlay_eligible,
        key=lambda e: e.get('risk_adjusted_ev', 0),
        reverse=True,
    )[:FOUR_LEG_MAX_INPUT]

    excluded_players: set = set()
    tickets_generated = 0

    for ticket_num in range(already_sent + 1, already_sent + remaining_slots + 1):
        available = [l for l in pool if l['player_id'] not in excluded_players]
        if len(available) < 4:
            logger.info(
                f"4-Leg Parlays: only {len(available)} legs left after exclusions — stopping."
            )
            break

        best: Dict = {}
        for combo in combinations(available, 4):
            legs = list(combo)
            if not _compatible(legs):
                continue

            cal_probs     = [calibrate_prob(l.get('model_prob', 0.5), playoff_mode=playoff_mode) for l in legs]
            implied_probs = [l['implied_prob'] for l in legs]

            jt         = prod(cal_probs) * FOUR_LEG_REALITY_TAX
            jb         = prod(implied_probs)
            combo_edge = jt - jb

            if combo_edge < COMBO_EDGE_MIN:
                continue

            if not best or combo_edge > best['edge']:
                best = {'legs': legs, 'edge': combo_edge, 'joint_prob': jt}

        if not best:
            logger.info(f"4-Leg Parlays: no valid ticket #{ticket_num} above edge threshold.")
            break

        legs       = best['legs']
        joint_prob = best['joint_prob']
        joint_edge = best['edge']
        combined   = prod(l['odds'] for l in legs)

        stake = _parlay_kelly_stake(legs, joint_prob, combined, max_pct=FOUR_LEG_KELLY_CAP)
        msg   = _format_four_leg_parlay(legs, joint_edge, joint_prob, stake, ticket_num, playoff_mode=playoff_mode)

        n_events = len({l.get('event_id', '') for l in legs})
        kind     = 'SGP' if n_events == 1 else f'{n_events}-game'
        title    = (
            f"4-Leg #{ticket_num} {kind}"
            f" | Hit: {joint_prob:.1%} | Edge: {joint_edge:+.2%} | ${stake:.0f}"
        )

        if db is not None:
            db.queue_pending_alert('four_leg_parlay', title, msg, priority=joint_edge)
        else:
            bot.send_message(msg)

        logger.info(
            f"4-Leg Parlay #{ticket_num} {'queued' if db else 'sent'}: "
            f"{kind} | edge={joint_edge:.2%} | joint_prob={joint_prob:.1%} | stake=${stake:.0f}"
        )

        for leg in legs:
            excluded_players.add(leg['player_id'])
        tickets_generated += 1

    if tickets_generated == 0:
        logger.info("4-Leg Parlays: no tickets met the edge threshold today.")


def _slate_ultimate_sent_today(db) -> bool:
    """Return True if a Slate Ultimate has already been queued today."""
    if db is None:
        return False
    with db.get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) FROM pending_alerts
            WHERE alert_type = 'slate_ultimate'
              AND date(created_at) = date('now', 'localtime')
            """
        )
        row = cursor.fetchone()
        return (row[0] if row else 0) > 0


def _format_slate_ultimate(
    legs: List[Dict], edge: float, joint_prob: float, stake: float,
    playoff_mode: bool = False,
) -> str:
    combined = prod(leg['odds'] for leg in legs)
    n_games  = len({leg.get('event_id', '') for leg in legs})
    header   = f"🎯 <b>SLATE ULTIMATE — 8-Leg Golden Ticket ({n_games} games)</b>\n"
    lines    = [header]
    for leg in legs:
        market = _MARKET_LABELS.get(leg['market'], leg['market'])
        cal_p  = calibrate_prob(leg.get('model_prob', 0), playoff_mode=playoff_mode)
        away   = leg.get('away_team', '')
        home   = leg.get('home_team', '')
        game   = f"[{away}@{home}]" if away and home else ''
        lines.append(
            f"• <b>{leg['player_id']}</b> {leg['side']} {leg['line']} {market}"
            f" @ {leg.get('book', '')} ({_american(leg['odds'])})"
            f" {game} — {cal_p:.0%} cal"
        )
    lines.append(
        f"\nCombined: {_american(combined)} | Hit Prob: {joint_prob:.2%} | Edge: {edge:+.2%}"
        f"\n⚠️ Max wager: ${stake:.0f} (0.15% bankroll — lottery ticket sizing)"
    )
    return "\n".join(lines)


def generate_slate_ultimate(
    actionable: List[Dict[str, Any]],
    bot: TelegramBotClient,
    db=None,
    playoff_mode: bool = False,
) -> None:
    """
    Generate and queue exactly one 8-leg 'Slate Ultimate' Golden Ticket per day.

    Architecture:
      • One leg per game (event_id). True independence; no copula needed.
      • Each leg must pass the standard quality gate (≥60% cal prob, ≥5% edge).
      • Requires ≥8 games with qualifying legs — silent on low-volume slates.
      • Per-leg selection: best leg per game by risk_adjusted_ev.
      • Reality tax: 0.65 (more conservative than 2-3 leg combos).
      • Kelly staking: hard cap at 0.15% of bankroll.
      • Fires at most once per day (deduplication via pending_alerts).
    """
    if _slate_ultimate_sent_today(db):
        logger.info("Slate Ultimate: already queued today — skipping.")
        return

    parlay_eligible = [e for e in actionable if _leg_passes_quality_gate(e, playoff_mode=playoff_mode)]

    # One best leg per game
    best_per_game: Dict[str, Dict] = {}
    for leg in parlay_eligible:
        eid = leg.get('event_id', '')
        if not eid:
            continue
        existing = best_per_game.get(eid)
        if existing is None or leg.get('risk_adjusted_ev', 0) > existing.get('risk_adjusted_ev', 0):
            best_per_game[eid] = leg

    if len(best_per_game) < SLATE_ULTIMATE_MIN_GAMES:
        logger.info(
            f"Slate Ultimate: only {len(best_per_game)} qualifying games "
            f"(need ≥{SLATE_ULTIMATE_MIN_GAMES}). Silent today."
        )
        return

    # Sort by ev and take the top-N games for the combination search
    candidates = sorted(
        best_per_game.values(),
        key=lambda e: e.get('risk_adjusted_ev', 0),
        reverse=True,
    )[:SLATE_ULTIMATE_MAX_INPUT]

    best: Dict = {}
    for combo in combinations(candidates, SLATE_ULTIMATE_LEGS):
        legs = list(combo)

        # Unique-player guard (same player could theoretically appear in two games)
        if len({l['player_id'] for l in legs}) < len(legs):
            continue

        cal_probs     = [calibrate_prob(l.get('model_prob', 0.5), playoff_mode=playoff_mode) for l in legs]
        implied_probs = [l['implied_prob'] for l in legs]

        jt         = prod(cal_probs) * SLATE_ULTIMATE_REALITY
        jb         = prod(implied_probs)
        joint_edge = jt - jb

        if joint_edge <= 0:
            continue

        if not best or joint_edge > best['edge']:
            best = {'legs': legs, 'edge': joint_edge, 'joint_prob': jt}

    if not best:
        logger.info("Slate Ultimate: no 8-leg combo with positive edge today.")
        return

    legs       = best['legs']
    joint_prob = best['joint_prob']
    joint_edge = best['edge']
    combined   = prod(l['odds'] for l in legs)

    stake = _parlay_kelly_stake(legs, joint_prob, combined, max_pct=SLATE_ULTIMATE_KELLY_CAP)
    msg   = _format_slate_ultimate(legs, joint_edge, joint_prob, stake, playoff_mode=playoff_mode)

    n_games = len({l.get('event_id', '') for l in legs})
    title   = (
        f"8-Leg Golden Ticket | {n_games} games"
        f" | Hit: {joint_prob:.1%} | Edge: {joint_edge:+.2%} | ${stake:.0f}"
    )

    if db is not None:
        db.queue_pending_alert('slate_ultimate', title, msg, priority=joint_edge)
    else:
        bot.send_message(msg)

    logger.info(
        f"Slate Ultimate {'queued' if db else 'sent'}: {n_games} games | "
        f"edge={joint_edge:.3%} | joint_prob={joint_prob:.3%} | stake=${stake:.0f}"
    )


def generate_and_alert_combos(
    actionable: List[Dict[str, Any]],
    bot: TelegramBotClient,
    db=None,
    playoff_mode: bool = False,
) -> None:
    """
    Generate the best parlay from the slate (SGP or cross-game).

    V2 changes:
      - Each leg must independently pass quality gates (calibrated prob + edge).
      - Max 3 legs.
      - No slate-wide high-prob parlays (removed — they were money incinerators).
      - Market family diversification enforced.
      - SGPs enabled: OVER/OVER and OVER/UNDER same-game combos are valid.
        Only same-game double-unders are blocked (Rainbet rule).
    """
    if len(actionable) < 2:
        return

    # ── Step 1: Filter to parlay-quality legs only ────────────────────────
    parlay_eligible = [e for e in actionable if _leg_passes_quality_gate(e, playoff_mode=playoff_mode)]

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

    # ── Step 2: Build combos from the full pool ───────────────────────────
    # SGPs allowed; only same-game double-unders blocked by _compatible.
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

            result = _combo_edge(legs, db=db, playoff_mode=playoff_mode)
            raw_joint = result.get('joint_true_prob', 0)

            _is_sgp = len({l.get('event_id') for l in legs}) == 1
            _tax_table = _REALITY_TAX_SGP if _is_sgp else _REALITY_TAX_CROSS
            dampening = _tax_table.get(size, 0.80)
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
        msg = _format_combo(best['legs'], best['edge'], best['joint_prob'], playoff_mode=playoff_mode)
        bot.send_message(msg)
        n_events = len({l.get('event_id') for l in best['legs']})
        kind = 'SGP' if n_events == 1 else 'cross-game'
        logger.info(
            f"Parlay sent ({kind}): {len(best['legs'])}-leg | "
            f"edge={best['edge']:.2%} | joint_prob={best['joint_prob']:.2%}"
        )
    else:
        logger.info("No parlay met the edge threshold today.")
