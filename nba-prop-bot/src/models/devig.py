import math
from typing import Tuple, List, Dict


def decimal_to_implied_prob(odds: float) -> float:
    if odds <= 1.0: return 0.0
    return 1.0 / odds


def get_theoretical_hold(prob_over_raw: float, prob_under_raw: float) -> float:
    """
    Return the theoretical hold (overround) for a two-way market.
    Normal sharp-book range: ~0.02–0.04 on props.
    A spike to 0.07+ signals the book is protecting itself (injury news,
    stale line) — the price is no longer a reliable signal.
    """
    return (prob_over_raw + prob_under_raw) - 1.0


def devig_two_way(prob_over_raw: float, prob_under_raw: float) -> Tuple[float, float]:
    """Simple additive (proportional) devigging. Kept for backward compatibility."""
    total = prob_over_raw + prob_under_raw
    if total == 0: return 0.0, 0.0
    return prob_over_raw / total, prob_under_raw / total


def devig_shin(prob_over_raw: float, prob_under_raw: float) -> Tuple[float, float]:
    """
    Shin's method for removing sportsbook hold from a two-way market.

    Background — Favourite-Longshot Bias:
    Books embed proportionally MORE margin in the underdog/longshot price
    because recreational bettors over-bet long shots, inflating that side's
    raw implied probability beyond a proportional share of the hold. Simple
    additive devigging (dividing each raw implied by their sum) treats both
    sides symmetrically and therefore OVER-estimates the underdog's true
    probability.

    Shin's model corrects for this by estimating the fraction z of total
    betting volume from insiders who know the true outcome.  Insiders
    concentrate bets on the underdog, forcing the book to inflate the
    underdog's price to break even — exactly the asymmetry that additive
    devigging misses.

    Correction direction vs additive devigging:
      • Favourite (higher raw implied): true prob moves UP slightly.
      • Underdog  (lower raw implied):  true prob moves DOWN slightly.

    For near-50/50 props the adjustment is < 0.5 pp.  For a 70/30 asymmetry
    at 5% hold the shift reaches ~1.5 pp, meaningfully improving the accuracy
    of consensus sharp-line probability estimates.

    The two-outcome solution exploits the analytical constraint p_1+p_2 = 1:
    find z ∈ (0, 1) satisfying
        √(z²+4(1−z)q₁²/S) + √(z²+4(1−z)q₂²/S) = 2
    then compute  p_i = (√(z²+4(1−z)qᵢ²/S) − z) / (2(1−z))

    where S = q_1 + q_2 = 1 + H (the raw overround).

    Reference: Shin (1993), 'Measuring the Incidence of Insider Trading in a
    Market for State-Contingent Claims', Economic Journal 103(420).
    """
    S = prob_over_raw + prob_under_raw
    if S <= 0:
        return 0.0, 0.0
    if S <= 1.0:
        # No overround — normalise only (same as additive)
        return prob_over_raw / S, prob_under_raw / S

    q1, q2 = prob_over_raw, prob_under_raw

    def _f(z: float) -> float:
        """Residual: equals 0 at the Shin z that satisfies p1+p2=1."""
        t1 = math.sqrt(z * z + 4.0 * (1.0 - z) * q1 * q1 / S)
        t2 = math.sqrt(z * z + 4.0 * (1.0 - z) * q2 * q2 / S)
        return t1 + t2 - 2.0

    # _f(0) = 2*sqrt(S) - 2 > 0 for S > 1
    # _f(0.5) < 0 for all realistic overrounds (verified across 0–30% hold)
    # Binary-search the zero in (0, 0.5); 56 iterations → precision < 1e-16
    lo, hi = 0.0, 0.5
    if _f(hi) >= 0.0:
        # Degenerate input (extreme overround) — fall back to additive
        return prob_over_raw / S, prob_under_raw / S

    for _ in range(56):
        mid = (lo + hi) * 0.5
        if _f(mid) > 0.0:
            lo = mid
        else:
            hi = mid

    z = (lo + hi) * 0.5

    denom = 2.0 * (1.0 - z)
    if denom < 1e-12:
        return prob_over_raw / S, prob_under_raw / S

    p1 = (math.sqrt(z * z + 4.0 * (1.0 - z) * q1 * q1 / S) - z) / denom
    p2 = (math.sqrt(z * z + 4.0 * (1.0 - z) * q2 * q2 / S) - z) / denom

    total = p1 + p2
    if total <= 0.0:
        return prob_over_raw / S, prob_under_raw / S

    return float(p1 / total), float(p2 / total)


def build_consensus_true_prob(
    book_probs: List[Dict],
) -> Tuple[float, float, str]:
    """
    Build a weighted consensus true probability from multiple independently
    deviggged sharp-book prices.

    Each element of `book_probs` must be a dict with keys:
        book   (str)   — display name of the book
        over   (float) — deviggged true P(over) from that book
        under  (float) — deviggged true P(under) from that book
        weight (float) — accuracy weight (higher = more trusted)
        hold   (float) — raw theoretical hold before devigging (informational)

    Books with elevated holds should already be filtered by the caller
    (see CONSENSUS_HOLD_MAX in config.py) before being passed here.

    Returns
    -------
    consensus_over  : float   weighted-average P(over), clamped to [0.01, 0.99]
    consensus_under : float   1 - consensus_over
    label           : str     e.g. "Pinnacle+Circa (hold≈3.1%)" — books that contributed
    """
    if not book_probs:
        return 0.5, 0.5, ''

    total_weight = sum(b['weight'] for b in book_probs)
    if total_weight <= 0:
        return 0.5, 0.5, ''

    consensus_over = sum(b['over'] * b['weight'] for b in book_probs) / total_weight
    consensus_over = float(max(0.01, min(0.99, consensus_over)))

    avg_hold = sum(b.get('hold', 0.0) for b in book_probs) / len(book_probs)
    book_names = '+'.join(b['book'] for b in book_probs)
    label = f"{book_names} (hold≈{avg_hold:.1%})"
    return consensus_over, 1.0 - consensus_over, label
