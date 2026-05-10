"""Tests for the V2 calibrated combo generator."""

from unittest.mock import MagicMock
from src.pipelines.combos import (
    generate_and_alert_combos,
    generate_four_leg_parlays,
    MAX_LEGS,
    _leg_passes_quality_gate,
    _compatible,
)


def _edge(player='Player A', market='player_points', side='OVER',
          model_prob=0.65, implied_prob=0.48, odds=2.10, line=20.5,
          event_id='ev1', home_team='Lakers', away_team='Celtics',
          team_name='Lakers', risk_adjusted_ev=0.10, fragile=False,
          edge=0.17, calibrated=True):
    return dict(
        player_id=player, market=market, side=side,
        model_prob=model_prob, implied_prob=implied_prob,
        odds=odds, line=line, book='draftkings',
        event_id=event_id, home_team=home_team, away_team=away_team,
        team_name=team_name, risk_adjusted_ev=risk_adjusted_ev,
        fragile=fragile, edge=edge, calibrated=calibrated,
    )


# ── Per-leg quality gate tests ──────────────────────────────────────────────

def test_strong_leg_passes_gate():
    """A leg with high calibrated prob and edge should pass."""
    leg = _edge(model_prob=0.72, implied_prob=0.48)
    assert _leg_passes_quality_gate(leg), "Strong leg should pass quality gate"


def test_marginal_leg_rejected():
    """A leg in the overconfident 55-65% band with thin edge should fail."""
    # model_prob=0.58 → calibrates to ~0.48 → below PER_LEG_PROB_MIN
    leg = _edge(model_prob=0.58, implied_prob=0.50, calibrated=False)
    assert not _leg_passes_quality_gate(leg), \
        "58% model prob calibrates to ~48% — should be rejected"


def test_fragile_leg_rejected():
    """Fragile edges should not enter parlays."""
    leg = _edge(model_prob=0.70, implied_prob=0.48, fragile=True)
    assert not _leg_passes_quality_gate(leg), "Fragile legs should be rejected"


def test_high_implied_prob_rejected():
    """When the book already prices a heavy favorite, there's no edge room."""
    leg = _edge(model_prob=0.72, implied_prob=0.60)
    assert not _leg_passes_quality_gate(leg), \
        "Implied prob > 55% means no edge room for parlays"


# ── Compatibility tests ─────────────────────────────────────────────────────

def test_same_family_same_player_incompatible():
    """PTS and PRA are in the 'scoring' family — shouldn't stack for same player."""
    legs = [
        _edge('Harden', 'player_points'),
        _edge('Harden', 'player_points_rebounds_assists'),
    ]
    assert not _compatible(legs), "Same player + same family should be incompatible"


def test_diverse_legs_compatible():
    legs = [
        _edge('Harden', 'player_points', edge=0.10),
        _edge('Westbrook', 'player_rebounds', edge=0.10),
    ]
    assert _compatible(legs), "Different players + different families should be compatible"


def test_compatible_default_caps_sgp_at_2():
    """Default max_sgp_legs=2 should reject 3 legs from the same game."""
    legs = [
        _edge('Harden', 'player_points', edge=0.10),
        _edge('Westbrook', 'player_rebounds', edge=0.10),
        _edge('Davis', 'player_assists', edge=0.10),
    ]
    # All from ev1 (same game) — 3 > default cap of 2
    assert not _compatible(legs), "Default SGP cap should reject 3 same-game legs"


def test_compatible_max_sgp_legs_allows_3():
    """max_sgp_legs=4 should allow 3 legs from the same game."""
    legs = [
        _edge('Harden', 'player_points', edge=0.10),
        _edge('Westbrook', 'player_rebounds', edge=0.10),
        _edge('Davis', 'player_assists', edge=0.10),
    ]
    assert _compatible(legs, max_sgp_legs=4), \
        "max_sgp_legs=4 should allow 3 same-game legs"


def test_compatible_max_sgp_legs_allows_4():
    """max_sgp_legs=4 should allow 4 legs from the same game."""
    legs = [
        _edge('Harden', 'player_points', edge=0.10),
        _edge('Westbrook', 'player_rebounds', edge=0.10),
        _edge('Davis', 'player_assists', edge=0.10),
        _edge('James', 'player_steals', edge=0.10),
    ]
    assert _compatible(legs, max_sgp_legs=4), \
        "max_sgp_legs=4 should allow 4 same-game legs"


def test_compatible_rejects_5_same_game_with_cap_4():
    """max_sgp_legs=4 should reject 5 legs from one game."""
    legs = [
        _edge('Harden', 'player_points', edge=0.10),
        _edge('Westbrook', 'player_rebounds', edge=0.10),
        _edge('Davis', 'player_assists', edge=0.10),
        _edge('James', 'player_steals', edge=0.10),
        _edge('Curry', 'player_blocks', edge=0.10),
    ]
    assert not _compatible(legs, max_sgp_legs=4), \
        "5 same-game legs should exceed max_sgp_legs=4"


# ── Integration tests ────────────────────────────────────────────────────────

def test_strong_two_leg_combo_sent():
    """Two strong, diverse legs should produce a combo alert."""
    bot = MagicMock()
    edges = [
        _edge('Harden',    'player_assists',  'OVER', 0.72, 0.48, edge=0.24),
        _edge('Westbrook', 'player_rebounds', 'OVER', 0.70, 0.47, edge=0.23),
    ]
    generate_and_alert_combos(edges, bot)
    assert bot.send_message.called, "Expected a combo message for two strong legs"


def test_marginal_edges_no_combo():
    """Two marginal legs (58% model -> ~48% calibrated) should NOT produce a combo."""
    bot = MagicMock()
    edges = [
        _edge('A', 'player_points',   'OVER', model_prob=0.58, implied_prob=0.50, calibrated=False),
        _edge('B', 'player_rebounds', 'OVER', model_prob=0.57, implied_prob=0.50, calibrated=False),
    ]
    generate_and_alert_combos(edges, bot)
    assert not bot.send_message.called, \
        "Marginal calibrated edges should not produce a combo"


def test_max_legs_enforced():
    """Even with many strong edges, combos should not exceed MAX_LEGS."""
    bot = MagicMock()
    edges = [
        _edge(f'P{i}', market, 'OVER', 0.75, 0.45, edge=0.30,
              event_id=f'ev{i}')
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


# ── 4-Leg multi-SGP tests ───────────────────────────────────────────────────

def test_four_leg_single_game_produced():
    """4 legs from 1 game (different players, different families) should produce a ticket."""
    bot = MagicMock()
    edges = [
        _edge('Harden',    'player_points',   'OVER', 0.72, 0.40, odds=2.50, edge=0.32, risk_adjusted_ev=0.15),
        _edge('Westbrook', 'player_rebounds',  'OVER', 0.70, 0.40, odds=2.50, edge=0.30, risk_adjusted_ev=0.14),
        _edge('Davis',     'player_assists',   'OVER', 0.68, 0.40, odds=2.50, edge=0.28, risk_adjusted_ev=0.13),
        _edge('James',     'player_steals',    'OVER', 0.66, 0.40, odds=2.50, edge=0.26, risk_adjusted_ev=0.12),
    ]
    generate_four_leg_parlays(edges, bot)
    assert bot.send_message.called, \
        "4-leg parlay should be produced from 1 game with 4 diverse players"


def test_four_leg_two_games_produced():
    """4 legs from 2 games should produce a ticket."""
    bot = MagicMock()
    edges = [
        _edge('Harden',    'player_points',   'OVER', 0.72, 0.40, odds=2.50, edge=0.32, event_id='ev1', risk_adjusted_ev=0.15),
        _edge('Westbrook', 'player_rebounds',  'OVER', 0.70, 0.40, odds=2.50, edge=0.30, event_id='ev1', risk_adjusted_ev=0.14),
        _edge('Davis',     'player_assists',   'OVER', 0.68, 0.40, odds=2.50, edge=0.28, event_id='ev2', risk_adjusted_ev=0.13),
        _edge('James',     'player_steals',    'OVER', 0.66, 0.40, odds=2.50, edge=0.26, event_id='ev2', risk_adjusted_ev=0.12),
    ]
    generate_four_leg_parlays(edges, bot)
    assert bot.send_message.called, \
        "4-leg parlay should be produced from 2 games"
