from unittest.mock import MagicMock
from src.pipelines.combos import (
    generate_and_alert_combos,
    COMBO_EDGE_MIN,
    MAX_COMBOS_TO_SEND,
)


def _edge(player='Player A', market='player_points', side='OVER',
          model_prob=0.62, implied_prob=0.50, odds=2.0, line=20.5):
    return dict(
        player_id=player, market=market, side=side,
        model_prob=model_prob, implied_prob=implied_prob,
        odds=odds, line=line, book='draftkings',
    )


def test_two_leg_combo_sent():
    bot = MagicMock()
    edges = [
        _edge('Harden',   'player_threes',  'OVER',  0.65, 0.50),
        _edge('Westbrook','player_points',  'OVER',  0.63, 0.50),
    ]
    generate_and_alert_combos(edges, bot)
    assert bot.send_message.called, "Expected at least one combo message"


def test_opposing_sides_rejected():
    """Same player + same market on OVER and UNDER must not produce a combo."""
    bot = MagicMock()
    edges = [
        _edge('Harden', 'player_points', 'OVER',  0.65, 0.50),
        _edge('Harden', 'player_points', 'UNDER', 0.65, 0.50),
    ]
    generate_and_alert_combos(edges, bot)
    assert not bot.send_message.called, "Opposing sides on same player+market should be rejected"


def test_low_edge_combo_skipped():
    """Combos whose joint edge < COMBO_EDGE_MIN should not be sent."""
    bot = MagicMock()
    # Both legs have model_prob barely above implied — joint edge will be tiny
    edges = [
        _edge('A', 'player_points',   'OVER', model_prob=0.51, implied_prob=0.50),
        _edge('B', 'player_rebounds', 'OVER', model_prob=0.51, implied_prob=0.50),
    ]
    generate_and_alert_combos(edges, bot)
    assert not bot.send_message.called, "Near-zero-edge combo should be filtered out"


def test_max_combos_to_send_respected():
    """No more than MAX_COMBOS_TO_SEND messages should be sent."""
    bot = MagicMock()
    # 8 strong edges → many combinations above threshold
    edges = [
        _edge(f'Player{i}', 'player_points', 'OVER', model_prob=0.70, implied_prob=0.50)
        for i in range(8)
    ]
    generate_and_alert_combos(edges, bot)
    assert bot.send_message.call_count <= MAX_COMBOS_TO_SEND, \
        f"Should send at most {MAX_COMBOS_TO_SEND} combo messages"
