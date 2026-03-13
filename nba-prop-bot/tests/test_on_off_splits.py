"""Tests for OnOffSplitsClient."""

import pandas as pd
from unittest.mock import MagicMock, patch
from src.clients.on_off_splits import OnOffSplitsClient, FALLBACK, CLAMP_HI, MIN_MINUTES


def _make_client():
    return OnOffSplitsClient()


def _mock_db(row=None):
    db = MagicMock()
    db.get_on_off_split.return_value = row
    return db


# ── Tests ────────────────────────────────────────────────────────────────────

def test_get_usage_multiplier_fallback_when_no_data():
    """When DB returns None, multiplier should equal FALLBACK."""
    client = _make_client()
    db = _mock_db(row=None)

    # Patch _build so no real API calls happen
    with patch.object(client, '_build', return_value=None):
        result = client.get_usage_multiplier(
            target_player_id=201935,
            absent_player_id=203954,
            market='player_assists',
            db=db,
        )
    assert result == FALLBACK, f"Expected {FALLBACK}, got {result}"


def test_get_usage_multiplier_uses_cached_data():
    """Cached row with sufficient minutes should return rate_without/rate_with."""
    client = _make_client()
    row = {
        'last_updated':   '2099-01-01',   # always fresh
        'minutes_without': 80.0,
        'rate_with':       0.50,
        'rate_without':    0.75,
        'usage_multiplier': 1.50,
    }
    db = _mock_db(row=row)

    result = client.get_usage_multiplier(
        target_player_id=201935,
        absent_player_id=203954,
        market='player_assists',
        db=db,
    )
    assert abs(result - 1.50) < 0.01, f"Expected ≈1.50, got {result}"


def test_multiplier_is_clamped_to_ceiling():
    """An extreme ratio should be clamped to CLAMP_HI."""
    client = _make_client()
    row = {
        'last_updated':   '2099-01-01',
        'minutes_without': 50.0,
        'rate_with':       0.10,
        'rate_without':    5.00,   # 50× — obviously spurious
        'usage_multiplier': 50.0,
    }
    db = _mock_db(row=row)

    result = client.get_usage_multiplier(
        target_player_id=1,
        absent_player_id=2,
        market='player_points',
        db=db,
    )
    assert result <= CLAMP_HI, f"Expected ≤{CLAMP_HI}, got {result}"


def test_fallback_when_insufficient_without_minutes():
    """If minutes_without < MIN_MINUTES, return FALLBACK even with valid row."""
    client = _make_client()
    row = {
        'last_updated':    '2099-01-01',
        'minutes_without': MIN_MINUTES - 1.0,   # just under threshold
        'rate_with':       0.50,
        'rate_without':    0.75,
        'usage_multiplier': 1.50,
    }
    db = _mock_db(row=row)

    result = client.get_usage_multiplier(
        target_player_id=1,
        absent_player_id=2,
        market='player_rebounds',
        db=db,
    )
    assert result == FALLBACK, f"Expected FALLBACK {FALLBACK}, got {result}"
