import pandas as pd
from src.models.projections import build_player_projection, estimate_projected_minutes

REQUIRED_KEYS = {'player_id', 'market', 'line', 'mean', 'projected_minutes',
                 'injury_status', 'usage_boost', 'variance_scale', 'home_flag', 'rest_days'}

def _make_logs(n: int = 15) -> pd.DataFrame:
    return pd.DataFrame({
        'MIN': [30.0] * n,
        'PTS': [20.0] * n,
        'REB': [5.0]  * n,
        'AST': [5.0]  * n,
        'FG3M': [3.0] * n,
    })

def test_projection_returns_all_keys():
    logs = _make_logs()
    result = build_player_projection(
        player_id='Test Player', market='player_points', line=20.5,
        recent_logs=logs, season_logs=logs,
        injury_status='Healthy', team_pace=100.0, opp_pace=98.0,
    )
    assert REQUIRED_KEYS.issubset(result.keys()), f"Missing keys: {REQUIRED_KEYS - result.keys()}"

def test_projection_zero_for_out_player():
    logs = _make_logs()
    result = build_player_projection(
        player_id='Injured Player', market='player_points', line=20.5,
        recent_logs=logs, season_logs=logs,
        injury_status='Out', team_pace=100.0, opp_pace=98.0,
    )
    assert result['mean'] == 0.0
    assert result['projected_minutes'] == 0.0

def test_home_flag_inflates_projection():
    logs = _make_logs()
    proj_home = build_player_projection(
        player_id='P', market='player_points', line=20.0,
        recent_logs=logs, season_logs=logs,
        injury_status='Healthy', team_pace=99.0, opp_pace=99.0,
        home_flag=True,
    )
    proj_away = build_player_projection(
        player_id='P', market='player_points', line=20.0,
        recent_logs=logs, season_logs=logs,
        injury_status='Healthy', team_pace=99.0, opp_pace=99.0,
        home_flag=False,
    )
    assert proj_home['mean'] > proj_away['mean'], "Home projection should exceed away projection"

def test_rest_days_extended_boost():
    logs = _make_logs()
    proj_rested = build_player_projection(
        player_id='P', market='player_points', line=20.0,
        recent_logs=logs, season_logs=logs,
        injury_status='Healthy', team_pace=99.0, opp_pace=99.0,
        rest_days=4,
    )
    proj_b2b = build_player_projection(
        player_id='P', market='player_points', line=20.0,
        recent_logs=logs, season_logs=logs,
        injury_status='Healthy', team_pace=99.0, opp_pace=99.0,
        rest_days=0, b2b_flag=True,
    )
    assert proj_rested['mean'] > proj_b2b['mean'], "Extended rest should boost vs back-to-back"

def test_doubtful_reduces_minutes():
    logs = _make_logs()
    mins_healthy   = estimate_projected_minutes(logs, logs, 'Healthy')
    mins_doubtful  = estimate_projected_minutes(logs, logs, 'Doubtful')
    assert mins_doubtful < mins_healthy * 0.5, "Doubtful should cut minutes by >50%"

def test_empty_logs_returns_zero():
    empty = pd.DataFrame()
    result = build_player_projection(
        player_id='P', market='player_points', line=20.0,
        recent_logs=empty, season_logs=empty,
        injury_status='Healthy', team_pace=99.0, opp_pace=99.0,
    )
    assert result['mean'] == 0.0
