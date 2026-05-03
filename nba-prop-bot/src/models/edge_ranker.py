from typing import List, Dict, Any, Optional
from src.config import (
    MIN_PROJECTED_MINUTES,
    PLAYOFF_MIN_PROJECTED_MINUTES,
    PLAYOFF_EDGE_MIN,
    PER_MARKET_EDGE_MIN,
    ALT_TO_BASE_MARKET,
)
from src.models.distributions import get_probability_distribution

# Priority 7: db is optional — set once from scan_props
_state: Dict[str, Any] = {'db': None, 'playoff_mode': False}

# CLV-calibrated per-market edge minimums (loaded lazily from settled alerts)
_clv_edge_mins: Dict[str, float] = {}
_clv_loaded: bool = False

# Time-based edge thresholds
# Early-morning lines carry more pre-game uncertainty (injury news not yet priced in).
# Close to tip-off the line is efficient but late info (scratches, warm-up reports)
# can create real, low-latency edges that justify a looser threshold.
_EARLY_EDGE_MIN  = 0.05   # > 4 hours to tip
_LATE_EDGE_MIN   = 0.02   # < 1 hour to tip
_EARLY_HOURS     = 4.0
_LATE_HOURS      = 1.0


def _load_clv_thresholds() -> None:
    """
    Compute per-market edge floors from 30-day rolling CLV.

    Formula:
      base        = _EARLY_EDGE_MIN (0.05)
      clv_delta   = clamp(-avg_clv * 3, -0.02, +0.03)
                    — negative CLV raises floor, positive lowers it
      floor       = clamp(base + clv_delta, 0.02, 0.12)

    The result is stored in _clv_edge_mins[market].  The static
    PER_MARKET_EDGE_MIN overrides in config are applied on top in
    compute_dynamic_edge_min() (take max), so manually tuned hard
    floors are never relaxed below their configured minimum.

    Falls back silently when the DB is unavailable or has < 10 CLV
    records per market.
    """
    global _clv_loaded, _clv_edge_mins
    _clv_loaded = True
    db = _state['db']
    if db is None:
        return
    try:
        per_market_clv = db.get_per_market_clv(days_back=30, min_samples=10)
    except Exception:
        return

    from src.utils.logging_utils import get_logger as _log
    log = _log(__name__)
    for market, avg_clv in per_market_clv.items():
        clv_delta = max(-0.02, min(0.03, -avg_clv * 3.0))
        floor = max(0.02, min(0.12, _EARLY_EDGE_MIN + clv_delta))
        _clv_edge_mins[market] = floor
        log.info(
            f"CLV floor [{market.replace('player_', '')}]: "
            f"avg_clv={avg_clv:+.4f}  delta={clv_delta:+.3f}  floor={floor:.3f}"
        )


def reload_clv_thresholds() -> None:
    """Force a fresh reload of CLV-adaptive floors. Called nightly by the scheduler."""
    global _clv_loaded, _clv_edge_mins
    _clv_loaded = False
    _clv_edge_mins = {}
    _load_clv_thresholds()


def compute_dynamic_edge_min(hours_to_tipoff: float, market: str = '') -> float:
    """
    Scale the minimum edge requirement based on how close tip-off is,
    with optional per-market CLV calibration.

    Returns:
        0.05  when hours_to_tipoff >= 4  (early — require strong edge)
        0.02  when hours_to_tipoff <= 1  (pre-game — accept thin edge)
        Linear interpolation between 1 and 4 hours.

    If CLV data exists for this market, uses that as the early-edge floor instead.
    """
    global _clv_loaded
    if not _clv_loaded:
        _load_clv_thresholds()

    early = _clv_edge_mins.get(market, _EARLY_EDGE_MIN) if market else _EARLY_EDGE_MIN
    if _state.get('playoff_mode'):
        # Playoff lines are sharper; require a higher early-edge floor.
        early = max(early, PLAYOFF_EDGE_MIN)
    market_floor = PER_MARKET_EDGE_MIN.get(market, 0.0) if market else 0.0
    if market_floor:
        early = max(early, market_floor)
    late = max(_LATE_EDGE_MIN, market_floor)

    if hours_to_tipoff >= _EARLY_HOURS:
        return early
    if hours_to_tipoff <= _LATE_HOURS:
        return late
    t = (hours_to_tipoff - _LATE_HOURS) / (_EARLY_HOURS - _LATE_HOURS)
    return round(late + t * (early - late), 4)


def set_db(db):
    """Call once from scan_props to give edge_ranker access to DB for bias lookups."""
    _state['db'] = db


def set_playoff_mode(playoff_mode: bool) -> None:
    """Call once from scan_props to enable playoff-tightened edge/minutes gates."""
    _state['playoff_mode'] = bool(playoff_mode)


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
        raw_market   = c.get('market', '')
        market       = ALT_TO_BASE_MARKET.get(raw_market, raw_market)
        mean         = c.get('mean',   0)
        line         = c.get('line',   0)
        side         = c.get('side',   '')
        var_scale    = c.get('variance_scale', 1.0)
        book         = c.get('book', '')

        _min_minutes = PLAYOFF_MIN_PROJECTED_MINUTES if _state.get('playoff_mode') else MIN_PROJECTED_MINUTES
        if proj_mins < _min_minutes:
            continue

        edge = model_prob - implied_prob

        # Phase 4: Edge Stability Filter
        # Market-scaled perturbation: low-mean markets (blocks/steals) need
        # a wider % shift to produce a meaningful absolute change.
        _FRAGILE_SHIFT = {
            'player_points': 0.05, 'player_rebounds': 0.08, 'player_assists': 0.08,
            'player_threes': 0.12, 'player_blocks': 0.15, 'player_steals': 0.15,
            'player_points_rebounds_assists': 0.05,
        }
        c['fragile'] = False
        if mean > 0 and model_prob > 0 and side:
            _shift = _FRAGILE_SHIFT.get(market, 0.08)
            dist_up   = get_probability_distribution(market, mean * (1 + _shift), line, variance_scale=var_scale)
            dist_down = get_probability_distribution(market, mean * (1 - _shift), line, variance_scale=var_scale)

            prob_up   = dist_up.get(f'prob_{side.lower()}',   model_prob)
            prob_down = dist_down.get(f'prob_{side.lower()}', model_prob)

            edge_up   = prob_up   - implied_prob
            edge_down = prob_down - implied_prob

            if (edge > 0 and (edge_up < 0 or edge_down < 0)) or \
               (edge < 0 and (edge_up > 0 or edge_down > 0)):
                edge *= 0.70  # fragile edge — reduce by 30%
                c['fragile'] = True

        # Sharp line shift: boost when sharp books moved in our direction,
        # penalize when they moved against us.
        _sharp_shift = c.get('sharp_line_shift')
        if _sharp_shift and _sharp_shift.get('shift_detected') and edge > 0:
            _shift_dir = _sharp_shift.get('direction', '')
            if (side.upper() == 'OVER' and _shift_dir == 'UP') or \
               (side.upper() == 'UNDER' and _shift_dir == 'DOWN'):
                edge *= 1.15  # sharp move aligns with our edge
            elif (side.upper() == 'OVER' and _shift_dir == 'DOWN') or \
                 (side.upper() == 'UNDER' and _shift_dir == 'UP'):
                edge *= 0.75  # sharp move contradicts our edge

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
        c['edge_min_applied']        = compute_dynamic_edge_min(c.get('hours_to_tipoff', 4.0), market=market)

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
