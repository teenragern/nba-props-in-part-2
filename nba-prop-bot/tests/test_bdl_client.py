"""Tests for BDL client and bridge."""

from unittest.mock import MagicMock, patch

from src.clients.bdl_client import BDLClient
from src.clients.bdl_bridge import BDLBridge


# ── Static method tests ───────────────────────────────────────────────────────

def test_normalize_prop_type_points():
    assert BDLClient.normalize_prop_type("points") == "player_points"


def test_normalize_prop_type_rebounds():
    assert BDLClient.normalize_prop_type("rebounds") == "player_rebounds"


def test_normalize_prop_type_assists():
    assert BDLClient.normalize_prop_type("assists") == "player_assists"


def test_normalize_prop_type_threes():
    assert BDLClient.normalize_prop_type("threes") == "player_threes"


def test_normalize_prop_type_pra():
    assert BDLClient.normalize_prop_type("points_rebounds_assists") == "player_points_rebounds_assists"


def test_normalize_prop_type_unsupported():
    assert BDLClient.normalize_prop_type("double_double") is None
    assert BDLClient.normalize_prop_type("triple_double") is None
    assert BDLClient.normalize_prop_type("points_rebounds") is None


def test_normalize_injury_status():
    assert BDLClient.normalize_injury_status("Out") == "Out"
    assert BDLClient.normalize_injury_status("Doubtful") == "Doubtful"
    assert BDLClient.normalize_injury_status("Questionable") == "Questionable"
    assert BDLClient.normalize_injury_status("Probable") == "Probable"
    assert BDLClient.normalize_injury_status("Day-To-Day") == "Questionable"
    assert BDLClient.normalize_injury_status("") == "Unknown"
    assert BDLClient.normalize_injury_status("Game Time Decision") == "Questionable"


def test_american_to_decimal_positive():
    assert abs(BDLClient.american_to_decimal(150) - 2.50) < 0.01


def test_american_to_decimal_negative():
    result = BDLClient.american_to_decimal(-110)
    assert abs(result - 1.909) < 0.01


def test_american_to_decimal_heavy_favorite():
    result = BDLClient.american_to_decimal(-200)
    assert abs(result - 1.50) < 0.01


def test_american_to_decimal_even():
    result = BDLClient.american_to_decimal(100)
    assert abs(result - 2.0) < 0.01


def test_american_to_decimal_big_underdog():
    result = BDLClient.american_to_decimal(300)
    assert abs(result - 4.0) < 0.01


# ── extract_props_for_scan tests ──────────────────────────────────────────────

def _make_client() -> BDLClient:
    """Build a BDLClient with mocked session (no real HTTP)."""
    client = BDLClient.__new__(BDLClient)
    client.api_key = "test"
    client._session = MagicMock()
    client._request_count = 0
    return client


def test_extract_props_for_scan_returns_over_and_under():
    """Both OVER and UNDER rows emitted for over_under markets."""
    client = _make_client()
    raw = [{
        "player_id": 1,
        "prop_type": "points",
        "vendor": "draftkings",
        "line_value": 24.5,
        "market": {"type": "over_under", "over_odds": -110, "under_odds": -110},
    }]
    with patch.object(client, "get_player_props", return_value=raw):
        result = client.extract_props_for_scan(game_id=99)

    assert len(result) == 2
    sides = {r["side"] for r in result}
    assert sides == {"OVER", "UNDER"}
    assert all(r["market"] == "player_points" for r in result)
    assert all(r["line"] == 24.5 for r in result)
    assert all(r["book"] == "draftkings" for r in result)


def test_extract_props_for_scan_filters_milestone():
    """Milestone markets are skipped."""
    client = _make_client()
    raw = [{
        "player_id": 1,
        "prop_type": "points",
        "vendor": "draftkings",
        "line_value": 20.0,
        "market": {"type": "milestone", "odds": -150},
    }]
    with patch.object(client, "get_player_props", return_value=raw):
        result = client.extract_props_for_scan(game_id=99)
    assert result == []


def test_extract_props_for_scan_filters_unsupported_prop():
    """Unsupported prop types (double_double etc.) are skipped."""
    client = _make_client()
    raw = [{
        "player_id": 1,
        "prop_type": "double_double",
        "vendor": "fanduel",
        "line_value": 0.5,
        "market": {"type": "over_under", "over_odds": -200, "under_odds": 160},
    }]
    with patch.object(client, "get_player_props", return_value=raw):
        result = client.extract_props_for_scan(game_id=99)
    assert result == []


def test_extract_props_for_scan_decimal_odds_conversion():
    """Decimal odds are correctly converted from American."""
    client = _make_client()
    raw = [{
        "player_id": 1,
        "prop_type": "rebounds",
        "vendor": "fanduel",
        "line_value": 8.5,
        "market": {"type": "over_under", "over_odds": -130, "under_odds": 110},
    }]
    with patch.object(client, "get_player_props", return_value=raw):
        result = client.extract_props_for_scan(game_id=99)

    over = next(r for r in result if r["side"] == "OVER")
    under = next(r for r in result if r["side"] == "UNDER")
    assert abs(over["decimal_odds"] - BDLClient.american_to_decimal(-130)) < 0.001
    assert abs(under["decimal_odds"] - BDLClient.american_to_decimal(110)) < 0.001


def test_extract_props_for_scan_vendors_passed():
    """Vendors argument is forwarded to get_player_props."""
    client = _make_client()
    with patch.object(client, "get_player_props", return_value=[]) as mock_props:
        client.extract_props_for_scan(game_id=5, vendors=["draftkings", "fanduel"])
    mock_props.assert_called_once()
    _, kwargs = mock_props.call_args
    assert kwargs.get("vendors") == ["draftkings", "fanduel"]


def test_extract_props_for_scan_multiple_props():
    """Multiple players and markets are all returned."""
    client = _make_client()
    raw = [
        {
            "player_id": 1, "prop_type": "points", "vendor": "dk",
            "line_value": 25.5,
            "market": {"type": "over_under", "over_odds": -115, "under_odds": -105},
        },
        {
            "player_id": 2, "prop_type": "assists", "vendor": "fd",
            "line_value": 7.5,
            "market": {"type": "over_under", "over_odds": -110, "under_odds": -110},
        },
    ]
    with patch.object(client, "get_player_props", return_value=raw):
        result = client.extract_props_for_scan(game_id=1)
    assert len(result) == 4  # 2 props × 2 sides


# ── BDLBridge helper ──────────────────────────────────────────────────────────

def _make_bridge(active_players=None, extract_props=None) -> BDLBridge:
    bdl = MagicMock()
    bdl.get_active_players.return_value = active_players or []
    bdl.extract_props_for_scan.return_value = extract_props or []
    bridge = BDLBridge.__new__(BDLBridge)
    bridge.bdl = bdl
    bridge._player_cache = {}
    bridge._team_cache = {}
    return bridge


# ── BDLBridge.get_props_for_game tests ───────────────────────────────────────

def test_bridge_get_props_for_game_structure():
    """Returns dict with all expected keys."""
    players = [{"id": 42, "first_name": "LeBron", "last_name": "James"}]
    props = [{
        "bdl_player_id": 42, "market": "player_points", "line": 25.5,
        "side": "OVER", "book": "draftkings",
        "decimal_odds": 1.909, "american_odds": -110, "vendor": "draftkings",
    }]
    bridge = _make_bridge(active_players=players, extract_props=props)

    result = bridge.get_props_for_game(bdl_game_id=1234)

    for key in ("players_in_event", "prices_by_market", "best_odds",
                "line_records", "player_id_map"):
        assert key in result, f"Missing key: {key}"

    assert "LeBron James" in result["players_in_event"]
    assert "player_points" in result["prices_by_market"]
    assert result["player_id_map"]["LeBron James"] == 42


def test_bridge_get_props_for_game_best_odds_tracks_highest():
    """best_odds records the highest price per (player, market, line, side)."""
    players = [{"id": 1, "first_name": "A", "last_name": "B"}]
    props = [
        {
            "bdl_player_id": 1, "market": "player_points", "line": 20.5,
            "side": "OVER", "book": "dk", "decimal_odds": 1.87,
            "american_odds": -115, "vendor": "dk",
        },
        {
            "bdl_player_id": 1, "market": "player_points", "line": 20.5,
            "side": "OVER", "book": "fd", "decimal_odds": 1.95,
            "american_odds": -105, "vendor": "fd",
        },
    ]
    bridge = _make_bridge(active_players=players, extract_props=props)
    result = bridge.get_props_for_game(bdl_game_id=1)

    key = ("A B", "player_points", 20.5, "OVER")
    assert result["best_odds"][key]["price"] == 1.95
    assert result["best_odds"][key]["book"] == "fd"


def test_bridge_get_props_for_game_passes_vendors():
    """vendors param is forwarded to extract_props_for_scan."""
    bridge = _make_bridge()
    bridge.get_props_for_game(bdl_game_id=7, vendors=["caesars", "betmgm"])
    bridge.bdl.extract_props_for_scan.assert_called_once_with(7, vendors=["caesars", "betmgm"])


def test_bridge_get_props_skips_unknown_players():
    """Players not in active-player cache and not fetchable individually are skipped."""
    bridge = _make_bridge(
        active_players=[{"id": 99, "first_name": "Known", "last_name": "Player"}],
        extract_props=[{
            "bdl_player_id": 1,  # not in active players
            "market": "player_points", "line": 10.5, "side": "OVER",
            "book": "dk", "decimal_odds": 1.9, "american_odds": -111, "vendor": "dk",
        }],
    )
    # Individual-fetch fallback returns no data for unknown player
    bridge.bdl._get.return_value = {"data": None}
    result = bridge.get_props_for_game(bdl_game_id=1)
    assert result["players_in_event"] == set()


# ── BDLBridge.get_injuries_for_date tests ────────────────────────────────────

def test_bridge_get_injuries_normalizes_records():
    bridge = _make_bridge()
    bridge.bdl.get_injuries.return_value = [{
        "player": {"id": 10, "first_name": "Kevin", "last_name": "Durant"},
        "status": "Out",
        "description": "Achilles",
        "return_date": "",
    }]
    injuries = bridge.get_injuries_for_date()
    assert len(injuries) == 1
    assert injuries[0]["player_name"] == "Kevin Durant"
    assert injuries[0]["status"] == "Out"
    assert injuries[0]["description"] == "Achilles"


def test_bridge_get_injuries_skips_empty_names():
    bridge = _make_bridge()
    bridge.bdl.get_injuries.return_value = [
        {"player": {"first_name": "", "last_name": ""}, "status": "Out", "description": ""},
    ]
    assert bridge.get_injuries_for_date() == []


def test_bridge_get_injuries_maps_status():
    bridge = _make_bridge()
    bridge.bdl.get_injuries.return_value = [
        {"player": {"first_name": "X", "last_name": "Y"},
         "status": "Day-To-Day", "description": "", "return_date": ""},
    ]
    injuries = bridge.get_injuries_for_date()
    assert injuries[0]["status"] == "Questionable"


# ── BDLBridge.get_confirmed_starters tests ───────────────────────────────────

def test_bridge_get_confirmed_starters():
    bridge = _make_bridge()
    bridge.bdl.get_starters_for_game.return_value = {
        "LAL": [{"name": "LeBron James", "position": "SF", "bdl_player_id": 1, "player_id": 1}]
    }
    starters = bridge.get_confirmed_starters(bdl_game_id=999)
    assert starters == {"LeBron James": True}


def test_bridge_get_confirmed_starters_multiple_teams():
    bridge = _make_bridge()
    bridge.bdl.get_starters_for_game.return_value = {
        "LAL": [{"name": "LeBron James", "position": "SF", "bdl_player_id": 1, "player_id": 1}],
        "BOS": [{"name": "Jayson Tatum", "position": "SF", "bdl_player_id": 2, "player_id": 2}],
    }
    starters = bridge.get_confirmed_starters(bdl_game_id=1)
    assert starters["LeBron James"] is True
    assert starters["Jayson Tatum"] is True


def test_bridge_get_confirmed_starters_empty_when_no_game():
    bridge = _make_bridge()
    bridge.bdl.get_starters_for_game.return_value = {}
    assert bridge.get_confirmed_starters(bdl_game_id=0) == {}


# ── BDLBridge.get_game_context_odds tests ────────────────────────────────────

def test_bridge_get_game_context_odds_median():
    """Median spread and total are computed across vendors."""
    bridge = _make_bridge()
    bridge.bdl.get_betting_odds.return_value = [
        {"game_id": 1, "spread_home_value": -5.0, "total_value": 219.0},
        {"game_id": 1, "spread_home_value": -5.5, "total_value": 220.0},
        {"game_id": 1, "spread_home_value": -6.0, "total_value": 221.0},
    ]
    result = bridge.get_game_context_odds(date="2026-03-24")
    assert 1 in result
    assert abs(result[1]["spread_home"] - -5.5) < 0.01
    assert abs(result[1]["total"] - 220.0) < 0.01


def test_bridge_get_game_context_odds_even_count():
    """Even number of values → average of two middle values."""
    bridge = _make_bridge()
    bridge.bdl.get_betting_odds.return_value = [
        {"game_id": 1, "spread_home_value": -4.0, "total_value": 218.0},
        {"game_id": 1, "spread_home_value": -6.0, "total_value": 222.0},
    ]
    result = bridge.get_game_context_odds(date="2026-03-24")
    assert abs(result[1]["spread_home"] - -5.0) < 0.01
    assert abs(result[1]["total"] - 220.0) < 0.01


def test_bridge_get_game_context_odds_multiple_games():
    bridge = _make_bridge()
    bridge.bdl.get_betting_odds.return_value = [
        {"game_id": 1, "spread_home_value": -3.0, "total_value": 210.0},
        {"game_id": 2, "spread_home_value": 2.5, "total_value": 225.0},
    ]
    result = bridge.get_game_context_odds(date="2026-03-24")
    assert 1 in result and 2 in result
    assert abs(result[2]["spread_home"] - 2.5) < 0.01


# ── BDLBridge.get_player_advanced_features tests ─────────────────────────────

def test_bridge_advanced_features_defaults_on_empty():
    bridge = _make_bridge()
    bridge.bdl.get_advanced_stats.return_value = []
    feats = bridge.get_player_advanced_features(bdl_player_id=1, season=2025)
    assert feats["avg_usage_pct"] == 0.0
    assert feats["avg_touches"] == 0.0
    assert feats["avg_distance"] == 0.0
    assert feats["avg_speed"] == 0.0


def test_bridge_advanced_features_averages():
    """Returns per-field averages over last n_games."""
    bridge = _make_bridge()
    bridge.bdl.get_advanced_stats.return_value = [
        {
            "usage_percentage": 0.30, "touches": 80, "speed": 4.5,
            "distance": 3.2, "contested_fg_pct": 0.45, "deflections": 2.0,
            "points_paint": 10, "pct_pts_paint": 0.30, "pct_pts_3pt": 0.25,
        },
        {
            "usage_percentage": 0.28, "touches": 76, "speed": 4.3,
            "distance": 3.0, "contested_fg_pct": 0.42, "deflections": 1.8,
            "points_paint": 8, "pct_pts_paint": 0.28, "pct_pts_3pt": 0.27,
        },
    ]
    feats = bridge.get_player_advanced_features(bdl_player_id=1, season=2025, n_games=10)
    assert abs(feats["avg_usage_pct"] - 0.29) < 0.001
    assert abs(feats["avg_touches"] - 78.0) < 0.001
    assert abs(feats["avg_distance"] - 3.1) < 0.001


def test_bridge_advanced_features_respects_n_games():
    """Only the first n_games records are averaged (BDL returns newest first)."""
    bridge = _make_bridge()
    # Provide 5 records; only the first 2 should be used with n_games=2
    bridge.bdl.get_advanced_stats.return_value = [
        {"usage_percentage": 0.40, "touches": 0, "speed": 0, "distance": 0,
         "contested_fg_pct": 0, "deflections": 0, "points_paint": 0,
         "pct_pts_paint": 0, "pct_pts_3pt": 0},
        {"usage_percentage": 0.20, "touches": 0, "speed": 0, "distance": 0,
         "contested_fg_pct": 0, "deflections": 0, "points_paint": 0,
         "pct_pts_paint": 0, "pct_pts_3pt": 0},
        # these should be ignored
        {"usage_percentage": 0.10, "touches": 0, "speed": 0, "distance": 999,
         "contested_fg_pct": 0, "deflections": 0, "points_paint": 0,
         "pct_pts_paint": 0, "pct_pts_3pt": 0},
    ]
    feats = bridge.get_player_advanced_features(bdl_player_id=1, season=2025, n_games=2)
    assert abs(feats["avg_usage_pct"] - 0.30) < 0.001
    assert feats["avg_distance"] == 0.0  # third record not included
