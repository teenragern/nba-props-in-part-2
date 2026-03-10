from src.models.devig import decimal_to_implied_prob, devig_two_way


def test_decimal_to_implied_even_odds():
    assert abs(decimal_to_implied_prob(2.0) - 0.5) < 1e-9


def test_decimal_to_implied_favorite():
    # -200 American = 1.5 decimal → implied = 66.7%
    assert abs(decimal_to_implied_prob(1.5) - 0.6667) < 0.001


def test_devig_two_way_sums_to_one():
    # Typical -110/-110 market: decimal odds ≈ 1.909
    prob_o = decimal_to_implied_prob(1.909)
    prob_u = decimal_to_implied_prob(1.909)
    fair_o, fair_u = devig_two_way(prob_o, prob_u)
    assert abs(fair_o + fair_u - 1.0) < 1e-9, "Devigged probs must sum to 1.0"


def test_devig_two_way_asymmetric():
    # Favorite: 1.5 decimal (~66.7%), dog: 2.8 (~35.7%)
    prob_o = decimal_to_implied_prob(1.5)
    prob_u = decimal_to_implied_prob(2.8)
    fair_o, fair_u = devig_two_way(prob_o, prob_u)
    assert abs(fair_o + fair_u - 1.0) < 1e-9
    assert fair_o > fair_u, "Favorite should have higher devigged probability"


def test_devig_removes_juice():
    # Both sides at -110 → raw implied = 52.4% each, sum = 104.8% (vig)
    prob_o = decimal_to_implied_prob(1.909)
    prob_u = decimal_to_implied_prob(1.909)
    fair_o, _ = devig_two_way(prob_o, prob_u)
    assert fair_o < prob_o, "Devigged probability must be less than raw implied (vig removed)"
