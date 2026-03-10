import pandas as pd
from src.models.distributions import get_probability_distribution


def test_probabilities_sum_to_one():
    for market in ['player_points', 'player_rebounds', 'player_assists', 'player_threes']:
        result = get_probability_distribution(market, mean=20.0, line=19.5)
        total = result.get('prob_over', 0) + result.get('prob_under', 0)
        assert abs(total - 1.0) < 0.01, f"{market}: probs sum to {total}, expected ~1.0"


def test_high_mean_favours_over():
    result = get_probability_distribution('player_points', mean=30.0, line=20.5)
    assert result['prob_over'] > 0.75, "Mean far above line should give >75% over probability"


def test_low_mean_favours_under():
    result = get_probability_distribution('player_points', mean=10.0, line=20.5)
    assert result['prob_under'] > 0.75, "Mean far below line should give >75% under probability"


def test_pra_market_handled():
    result = get_probability_distribution('player_points_rebounds_assists', mean=35.0, line=32.5)
    assert 'prob_over' in result and 'prob_under' in result


def test_bootstrap_uses_logs_when_available():
    logs = pd.DataFrame({'PTS': [20, 22, 18, 25, 19, 21, 23, 20, 17, 22,
                                  24, 20, 18, 21, 19, 23, 25, 20, 22, 21]})
    result = get_probability_distribution('player_points', mean=21.0, line=20.5, logs=logs)
    assert 0.0 < result['prob_over'] < 1.0


def test_variance_scale_affects_probability():
    # Higher variance → probabilities closer to 50/50
    r_tight = get_probability_distribution('player_points', mean=22.0, line=20.5, variance_scale=0.5)
    r_wide  = get_probability_distribution('player_points', mean=22.0, line=20.5, variance_scale=2.0)
    # Wide variance → over prob closer to 0.5
    assert abs(r_wide['prob_over'] - 0.5) < abs(r_tight['prob_over'] - 0.5), \
        "Higher variance scale should pull probability toward 0.5"
