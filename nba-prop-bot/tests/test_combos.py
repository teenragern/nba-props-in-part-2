"""Tests for the V2 calibrated combo generator."""

from unittest.mock import MagicMock
from src.pipelines.combos import (
    generate_and_alert_combos,
    COMBO_EDGE_MIN,
    MAX_LEGS,
    PER_LEG_EDGE_MIN,
    PER_LEG_PROB_MIN,
    _leg_passes_quality_gate,
    _compatible,
)


def _edge(player='Player A', market='player_points', side='OVER',
          model_prob=0.65, implied_prob=0.48, odds=2.10, line=20.5,
          event_id='ev1', home_team='Lakers', away_team='Celtics',
          team_name='Lakers', risk_adjusted_ev=0.10, fragile=False):
    return dict(
        player_id=player, market=market, side=side,
        model_prob=model_prob, implied_prob=implied_prob,
        odds=odds, line=line, book='draftkings',
        event_id=event_id, home_team=home_team, away_team=away_team,
        team_name=team_name, risk_adjusted_ev=risk_adjusted_ev,
        fragile=fragile,
    )


# ── Per-leg quality gate tests ──────────────────────────────────────────────

def test_strong_leg_passes_gate():
    """A leg with high calibrated prob and edge should pass."""
    leg = _edge(model_prob=0.72, implied_prob=0.48)
    assert _leg_passes_quality_gate(leg), "Strong leg should pass quality gate"


def test_marginal_leg_rejected():
    """A leg in the overconfident 55-65% band with thin edge should fail."""
    # model_prob=0.58 → calibrates to ~0.48 → below PER_LEG_PROB_MIN
    leg = _edge(model_prob=0.58, implied_prob=0.50)
    assert not _leg_passes_quality_gate(leg), \
        "58% model prob calibrates to ~48% — should be rejected"


def test_fragile_leg_rejected():
    """Fragile edges should not enter parlays."""
    leg = _edge(model_prob=0.70, implied_prob=0.48, fragile=True)
    leg['fragile'] = True
    assert not _leg_passes_quality_gate(leg), "Fragile legs should be rejected"


def test_high_implied_prob_rejected():
    """When the book already prices a heavy favorite, there's no edge room."""
    leg = _edge(model_prob=0.72, implied_prob=0.60)
    assert not _leg_passes_quality_gate(leg), \
        "Implied prob > 55% means no edge room for parlays"


# ── Compatibility tests ─────────────────────────────────────────────────────

def test_same_player_incompatible():
    legs = [
        _edge('Harden', 'player_points'),
        _edge('Harden', 'player_rebounds'),
    ]
    assert not _compatible(legs), "Same player should be incompatible"


def test_same_market_family_incompatible():
    """PTS and PRA are in the 'scoring' family — shouldn't stack."""
    legs = [
        _edge('Harden', 'player_points'),
        _edge('Westbrook', 'player_points_rebounds_assists'),
    ]
    assert not _compatible(legs), "PTS + PRA from same family should be incompatible"


def test_diverse_legs_compatible():
    legs = [
        _edge('Harden', 'player_points'),
        _edge('Westbrook', 'player_rebounds'),
    ]
    assert _compatible(legs), "Different players + different families should be compatible"


# ── Integration tests ────────────────────────────────────────────────────────

def test_strong_two_leg_combo_sent():
    """Two strong, diverse legs should produce a combo alert."""
    bot = MagicMock()
    edges = [
        _edge('Harden',    'player_assists',  'OVER', 0.72, 0.48),
        _edge('Westbrook', 'player_rebounds', 'OVER', 0.70, 0.47),
    ]
    generate_and_alert_combos(edges, bot)
    assert bot.send_message.called, "Expected a combo message for two strong legs"


def test_marginal_edges_no_combo():
    """Two marginal legs (58% model → ~48% calibrated) should NOT produce a combo."""
    bot = MagicMock()
    edges = [
        _edge('A', 'player_points',   'OVER', model_prob=0.58, implied_prob=0.50),
        _edge('B', 'player_rebounds', 'OVER', model_prob=0.57, implied_prob=0.50),
    ]
    generate_and_alert_combos(edges, bot)
    assert not bot.send_message.called, \
        "Marginal calibrated edges should not produce a combo"


def test_max_legs_enforced():
    """Even with many strong edges, combos should not exceed MAX_LEGS."""
    bot = MagicMock()
    edges = [
        _edge(f'P{i}', market, 'OVER', 0.75, 0.45)
        for i, market in enumerate([
            'player_points', 'player_rebounds', 'player_assists',
            'player_steals', 'player_blocks',
        ])
    ]
    generate_and_alert_combos(edges, bot)
    if bot.send_message.called:
        msg = bot.send_message.call_args[0][0]
        # Should say "2-Leg" or "3-Leg", never "4-Leg" or higher
        assert '4-Leg' not in msg and '5-Leg' not in msg, \
            f"Combo should not exceed {MAX_LEGS} legs"


def test_no_slate_wide_parlays():
    """Slate-wide 4-leg and 8-leg parlays should no longer be generated."""
    bot = MagicMock()
    # 8 strong edges across different games
    edges = [
        _edge(f'P{i}', 'player_points', 'OVER', 0.75, 0.45,
              event_id=f'ev{i}')
        for i in range(8)
    ]
    generate_and_alert_combos(edges, bot)
    if bot.send_message.called:
        for call in bot.send_message.call_args_list:
            msg = call[0][0]
            assert '4-Leg High-Probability' not in msg, \
                "Slate-wide 4-leg parlays should be removed"
            assert '8-Leg High-Probability' not in msg, \
                "Slate-wide 8-leg parlays should be removed"
