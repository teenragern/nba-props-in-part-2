from typing import List, Dict, Any, Optional
from src.config import MIN_PROJECTED_MINUTES
from src.models.distributions import get_probability_distribution

# Priority 7: db is optional — set once from scan_props
_state: Dict[str, Any] = {'db': None}


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

        # Risk-Adjusted EV (sort key)
        variance = (mean * 1.25) * var_scale if mean > 0 else 1.0
        c['risk_adjusted_ev'] = ev / variance if variance > 0 else ev

        ranked.append(c)

    ranked.sort(key=lambda x: x.get('risk_adjusted_ev', 0), reverse=True)
    return ranked
