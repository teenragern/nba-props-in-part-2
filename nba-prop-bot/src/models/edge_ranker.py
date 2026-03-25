from typing import List, Dict, Any, Optional
from src.config import MIN_PROJECTED_MINUTES
from src.models.distributions import get_probability_distribution

# Priority 7: db is optional — set once from scan_props
_state: Dict[str, Any] = {'db': None}

# Time-based edge thresholds
# Early-morning lines carry more pre-game uncertainty (injury news not yet priced in).
# Close to tip-off the line is efficient but late info (scratches, warm-up reports)
# can create real, low-latency edges that justify a looser threshold.
_EARLY_EDGE_MIN  = 0.05   # > 4 hours to tip
_LATE_EDGE_MIN   = 0.02   # < 1 hour to tip
_EARLY_HOURS     = 4.0
_LATE_HOURS      = 1.0


def compute_dynamic_edge_min(hours_to_tipoff: float) -> float:
    """
    Scale the minimum edge requirement based on how close tip-off is.

    Returns:
        0.05  when hours_to_tipoff >= 4  (early — require strong edge)
        0.02  when hours_to_tipoff <= 1  (pre-game — accept thin edge)
        Linear interpolation between 1 and 4 hours.
    """
    if hours_to_tipoff >= _EARLY_HOURS:
        return _EARLY_EDGE_MIN
    if hours_to_tipoff <= _LATE_HOURS:
        return _LATE_EDGE_MIN
    # Linear blend: 0.02 at 1hr, 0.05 at 4hr
    t = (hours_to_tipoff - _LATE_HOURS) / (_EARLY_HOURS - _LATE_HOURS)
    return round(_LATE_EDGE_MIN + t * (_EARLY_EDGE_MIN - _LATE_EDGE_MIN), 4)


def set_db(db):
    """Call once from scan_props to give edge_ranker access to DB for bias lookups."""
    _state['db'] = db


def get_market_feedback_factor(market: str, book: Optional[str] = None) -> float:
    """
    Priority 7: Per-book/market bias correction from historical bet results.
    Falls back to 1.0 when fewer than 20 settled bets exist for this combo.
    """
    db = _state['db']
    if db is None:
        return 1.0
    try:
        return db.get_book_market_bias(book or '', market)
    except Exception:
        return 1.0


def rank_edges(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = []

    for c in candidates:
        model_prob   = c.get('model_prob',  0)
        implied_prob = c.get('implied_prob', 0)
        odds         = c.get('odds',        0)
        proj_mins    = c.get('projected_minutes', 0)
        status       = (c.get('injury_status') or 'healthy').lower()
        market       = c.get('market', '')
        mean         = c.get('mean',   0)
        line         = c.get('line',   0)
        side         = c.get('side',   '')
        var_scale    = c.get('variance_scale', 1.0)
        book         = c.get('book', '')

        if proj_mins < MIN_PROJECTED_MINUTES:
            continue

        edge = model_prob - implied_prob

        # Phase 4: Edge Stability Filter
        c['fragile'] = False
        if mean > 0 and model_prob > 0 and side:
            dist_up   = get_probability_distribution(market, mean * 1.05, line, variance_scale=var_scale)
            dist_down = get_probability_distribution(market, mean * 0.95, line, variance_scale=var_scale)

            prob_up   = dist_up.get(f'prob_{side.lower()}',   model_prob)
            prob_down = dist_down.get(f'prob_{side.lower()}', model_prob)

            edge_up   = prob_up   - implied_prob
            edge_down = prob_down - implied_prob

            if (edge > 0 and (edge_up < 0 or edge_down < 0)) or \
               (edge < 0 and (edge_up > 0 or edge_down > 0)):
                edge *= 0.70  # fragile edge — reduce by 30%
                c['fragile'] = True

        # Phase 5: Market Microstructure Adjustments
        steam      = c.get('steam_flag', False)
        velocity   = c.get('velocity',   0.0)
        dispersion = c.get('dispersion', 0.0)
        book_role  = c.get('book_role', 'neutral')

        if steam and edge > 0:
            edge *= 1.10  # align with sharp steam
        elif velocity < -0.02 and edge > 0:
            edge *= 0.80  # fade anti-steam

        # Real-time stale line: sharp repriced within 2 min, retail hasn't caught up
        if c.get('timestamp_stale') and edge > 0:
            edge *= 1.20  # velocity premium — chase before retail wakes up

        if dispersion > 0.04:
            edge *= 1.05  # inefficient market
        elif 0.0 < dispersion < 0.015:
            edge *= 0.90  # too-tight market

        if book_role == 'sharp':
            edge *= 1.10  # validated by sharp line

        # Consensus confirmation / contradiction
        consensus_prob = c.get('consensus_prob')
        if consensus_prob is not None and edge > 0:
            # consensus_prob = true P(over); derive consensus edge for this side
            if side.upper() == 'OVER':
                consensus_edge = consensus_prob - implied_prob
            else:
                consensus_edge = (1.0 - consensus_prob) - implied_prob
            if consensus_edge > 0.02:
                edge *= 1.15   # consensus confirms — more confident
            elif consensus_edge < -0.02:
                edge *= 0.80   # consensus contradicts — reduce confidence

        # Priority 7: Per-book/market bias correction from historical results
        factor = get_market_feedback_factor(market, book)
        edge  *= factor
        ev     = ((model_prob * odds) - 1.0) * factor

        # Injury penalties
        if "questionable" in status or "gtd" in status:
            edge *= 0.8
            ev   *= 0.8
        elif "doubtful" in status:
            edge *= 0.5
            ev   *= 0.5
        elif "out" in status:
            continue

        c['edge']                    = edge
        c['ev']                      = ev
        c['feedback_factor_applied'] = factor
        c['edge_min_applied']        = compute_dynamic_edge_min(c.get('hours_to_tipoff', 4.0))

        # Risk-Adjusted EV (sort key)
        variance = (mean * 1.25) * var_scale if mean > 0 else 1.0
        c['risk_adjusted_ev'] = ev / variance if variance > 0 else ev

        # ──────────────────────────────────────────────────────────────────
        # Under Tax REMOVED.
        #
        # Previously applied a 15% penalty to Under risk_adjusted_ev to
        # offset right-skew in NBA stat distributions. Empirical data
        # shows the model OVER-projects means in the 55-65% band, meaning
        # Unders are mechanically more likely to hit than the model thinks.
        # The tax was filtering out profitable Under picks and contributing
        # to the parlay loss spiral. Calibration_model.py now handles this
        # correctly at the probability level.
        # ──────────────────────────────────────────────────────────────────

        ranked.append(c)

    ranked.sort(key=lambda x: x.get('risk_adjusted_ev', 0), reverse=True)
    return ranked
