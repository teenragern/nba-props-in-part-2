from typing import Tuple, List, Dict


def decimal_to_implied_prob(odds: float) -> float:
    if odds <= 1.0: return 0.0
    return 1.0 / odds


def devig_two_way(prob_over_raw: float, prob_under_raw: float) -> Tuple[float, float]:
    total = prob_over_raw + prob_under_raw
    if total == 0: return 0.0, 0.0
    return prob_over_raw / total, prob_under_raw / total


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

    Returns
    -------
    consensus_over  : float   weighted-average P(over), clamped to [0.01, 0.99]
    consensus_under : float   1 - consensus_over
    label           : str     e.g. "Pinnacle+Circa" — books that contributed
    """
    if not book_probs:
        return 0.5, 0.5, ''

    total_weight = sum(b['weight'] for b in book_probs)
    if total_weight <= 0:
        return 0.5, 0.5, ''

    consensus_over = sum(b['over'] * b['weight'] for b in book_probs) / total_weight
    consensus_over = float(max(0.01, min(0.99, consensus_over)))

    label = '+'.join(b['book'] for b in book_probs)
    return consensus_over, 1.0 - consensus_over, label
