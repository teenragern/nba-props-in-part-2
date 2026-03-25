"""Tests for calibration_model.py"""

from src.models.calibration_model import calibrate_prob, calibrate_edge_candidate


def test_extremes_preserved():
    """Well-calibrated extremes should pass through roughly unchanged."""
    # >65% bucket: 81.7% predicted, 81.8% actual
    assert abs(calibrate_prob(0.82) - 0.82) < 0.02, \
        "High-confidence bucket should be near-unchanged"

    # <45% bucket: 41.8% predicted, 40.0% actual
    low = calibrate_prob(0.42)
    assert abs(low - 0.40) < 0.03, \
        f"Low bucket should calibrate near 40%, got {low:.1%}"


def test_overconfident_middle_compressed():
    """The 55-65% band should be pulled down significantly."""
    # 58% model → should calibrate to ~48% (actual was 46.2%)
    cal_58 = calibrate_prob(0.58)
    assert cal_58 < 0.52, \
        f"58% model prob should calibrate below 52%, got {cal_58:.1%}"

    # 63% model → should calibrate to ~50% (actual was 50.0%)
    cal_63 = calibrate_prob(0.63)
    assert cal_63 < 0.55, \
        f"63% model prob should calibrate below 55%, got {cal_63:.1%}"


def test_calibration_monotonic_at_extremes():
    """Higher raw prob should generally give higher calibrated prob outside the problem zone."""
    # Below 50% and above 70% should be monotonic
    assert calibrate_prob(0.30) < calibrate_prob(0.40)
    assert calibrate_prob(0.75) < calibrate_prob(0.85)
    assert calibrate_prob(0.85) < calibrate_prob(0.95)


def test_calibrate_edge_candidate_preserves_raw():
    """calibrate_edge_candidate should store the raw value and overwrite model_prob."""
    c = {'model_prob': 0.60, 'implied_prob': 0.50}
    calibrate_edge_candidate(c)

    assert c['raw_model_prob'] == 0.60, "Raw prob should be preserved"
    assert c['model_prob'] != 0.60, "model_prob should be overwritten with calibrated value"
    assert c['calibrated'] is True


def test_boundary_values():
    """Edge cases: 0, 1, and near-boundary values."""
    assert calibrate_prob(0.0) >= 0.0
    assert calibrate_prob(1.0) <= 1.0
    assert 0.0 < calibrate_prob(0.01) < 1.0
    assert 0.0 < calibrate_prob(0.99) <= 1.0
