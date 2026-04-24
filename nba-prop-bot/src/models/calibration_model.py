"""
Post-hoc probability calibration.

Built from empirical calibration data (206 settled alerts):

    Predicted   Actual    Gap
    ─────────   ──────    ────
    41.8%       40.0%     -1.8%   (good)
    47.8%       60.0%     +12.2%  (under-confident — rare bucket)
    52.2%       58.3%     +6.1%   (slight under-confidence)
    58.0%       46.2%     -11.8%  ← PROBLEM ZONE
    62.9%       50.0%     -12.9%  ← PROBLEM ZONE
    81.7%       81.8%     +0.1%   (good)

The 55–65% band is where the model is most dangerous: it's
overconfident by ~12 percentage points. This module applies a
piecewise-linear correction before probabilities are used for
parlay construction or edge calculation.

The correction is conservative — it only compresses the known
problem zone and leaves the well-calibrated extremes alone.

Usage:
    from src.models.calibration_model import calibrate_prob
    true_prob = calibrate_prob(model_prob)
"""

from typing import List, Dict, Any


# ── Piecewise calibration knots ────────────────────────────────────────────
# Format: (model_predicted, empirical_actual)
# Derived from the bucket breakdown above.
# Between knots we linearly interpolate.
_KNOTS = [
    (0.00, 0.00),
    (0.42, 0.40),   # <45% bucket: well-calibrated
    (0.48, 0.55),   # 45-50% bucket: model slightly under-confident
    (0.52, 0.55),   # 50-55% bucket: slight under-confidence
    (0.58, 0.48),   # 55-60% bucket: OVERCONFIDENT — actual is 46.2%
    (0.63, 0.50),   # 60-65% bucket: OVERCONFIDENT — actual is 50.0%
    (0.70, 0.65),   # transition back toward calibrated
    (0.82, 0.82),   # >65% bucket: near-perfect
    (1.00, 1.00),
]


# Fraction of the RS-fit correction to apply when playoff_mode=True.
# 0.50 → blend 50/50 raw↔calibrated. The curve was fit on 206 regular-season
# settled alerts; until playoff outcomes are tagged and re-fit, we damp the
# correction to avoid overapplying a misfit shape to a different distribution.
_PLAYOFF_CALIBRATION_STRENGTH = 0.50


def calibrate_prob(raw_prob: float, playoff_mode: bool = False) -> float:
    """
    Map a raw model probability to a calibrated probability using
    piecewise-linear interpolation through empirical knots.

    Args:
        raw_prob: model's predicted P(side hits), range [0, 1].
        playoff_mode: when True, blend the RS-fit correction with the raw
            probability at _PLAYOFF_CALIBRATION_STRENGTH (default 0.50). The
            knots were fit on regular-season alerts and are not validated
            for playoff outcomes yet.

    Returns:
        Calibrated probability, clamped to [0.01, 0.99].
    """
    if raw_prob <= _KNOTS[0][0]:
        calibrated = _KNOTS[0][1]
    elif raw_prob >= _KNOTS[-1][0]:
        calibrated = _KNOTS[-1][1]
    else:
        calibrated = raw_prob
        for i in range(len(_KNOTS) - 1):
            x0, y0 = _KNOTS[i]
            x1, y1 = _KNOTS[i + 1]
            if x0 <= raw_prob <= x1:
                if x1 == x0:
                    calibrated = y0
                else:
                    t = (raw_prob - x0) / (x1 - x0)
                    calibrated = y0 + t * (y1 - y0)
                break

    if playoff_mode:
        calibrated = (
            _PLAYOFF_CALIBRATION_STRENGTH * calibrated
            + (1.0 - _PLAYOFF_CALIBRATION_STRENGTH) * raw_prob
        )

    return float(max(0.01, min(0.99, calibrated)))


def calibrate_edge_candidate(candidate: Dict[str, Any],
                             playoff_mode: bool = False) -> Dict[str, Any]:
    """
    Apply calibration to a single edge candidate dict (in-place).
    Stores both raw and calibrated values for transparency.

    Adds/modifies keys:
        raw_model_prob   — original model probability (preserved)
        model_prob       — overwritten with calibrated value
        calibrated       — True flag
    """
    raw = candidate.get('model_prob', 0.5)
    candidate['raw_model_prob'] = raw
    candidate['model_prob'] = calibrate_prob(raw, playoff_mode=playoff_mode)
    candidate['calibrated'] = True
    return candidate


def calibrate_candidates(candidates: List[Dict[str, Any]],
                         playoff_mode: bool = False) -> List[Dict[str, Any]]:
    """Calibrate a list of edge candidates. Mutates in place and returns the list."""
    for c in candidates:
        calibrate_edge_candidate(c, playoff_mode=playoff_mode)
    return candidates
