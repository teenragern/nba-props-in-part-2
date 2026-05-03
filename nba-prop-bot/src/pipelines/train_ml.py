"""
Multi-season XGBoost training pipeline.

Fetches game logs for all active players across the last N seasons,
builds the full training dataset, and trains one XGBRegressor per market.
Existing models are overwritten on success.

Usage:
    python -m src.pipelines.train_ml
    TRAIN_SEASONS=2 python -m src.pipelines.train_ml

Environment:
    TRAIN_SEASONS   number of seasons of data to use (default 3)
    TRAIN_MIN_GP    minimum games played filter (default 20)
"""

import os
import time
import pandas as pd
from typing import List, Optional

from src.utils.logging_utils import get_logger
from src.clients.nba_stats import NbaStatsClient
from src.clients.bdl_client import BDLClient
from src.clients.bdl_game_logs import BDLGameLogs
from src.data.db import DatabaseClient
from src.clients.telegram_bot import TelegramBotClient
from src.models.ml_model import train_models_from_logs, train_models_with_clv_feedback

logger = get_logger(__name__)

_DEFAULT_SEASONS = ['2025-26', '2024-25', '2023-24']
_N_SEASONS       = int(os.getenv('TRAIN_SEASONS', '3'))
_MIN_GP          = int(os.getenv('TRAIN_MIN_GP',  '20'))


def train_ml_models(seasons: Optional[List[str]] = None) -> dict:
    """
    Fetch multi-season logs for all active players and retrain XGBoost models.
    Returns {market: success_bool}.
    """
    if seasons is None:
        seasons = _DEFAULT_SEASONS[:_N_SEASONS]

    stats = NbaStatsClient()
    bot   = TelegramBotClient()
    bdl_logs = BDLGameLogs(BDLClient(), db=DatabaseClient())

    logger.info(f"ML training started: seasons={seasons}")
    bot.send_message(
        f"🤖 <b>ML Training started</b>\nSeasons: {', '.join(seasons)}\n"
        f"Fetching player list..."
    )

    player_ids = stats.get_all_active_player_ids(min_gp=_MIN_GP)
    logger.info(f"Training on {len(player_ids)} players across {len(seasons)} seasons.")

    # Fetch matchup context once (current-season team stats for feature lookups)
    try:
        opp_stats_df  = stats.get_opponent_stats()
        team_stats_df = stats.get_team_stats()
        def_stats_df  = stats.get_team_defense_stats()
    except Exception as e:
        logger.warning(f"Could not fetch team stats for training context: {e}")
        opp_stats_df = team_stats_df = def_stats_df = None

    # Compute league-average pace for normalization
    league_avg_pace = 99.0
    if team_stats_df is not None and not team_stats_df.empty and 'PACE' in team_stats_df.columns:
        league_avg_pace = float(team_stats_df['PACE'].mean())

    # Collect all logs
    player_logs: List[pd.DataFrame] = []
    for idx, pid in enumerate(player_ids):
        season_dfs = []
        for season in seasons:
            szn_int = int(season.split('-')[0])
            df = bdl_logs.get_player_game_logs(pid, szn_int, ignore_ttl=True)
            if not df.empty:
                season_dfs.append(df)

        if not season_dfs:
            continue

        combined = pd.concat(season_dfs, ignore_index=True)
        if not combined.empty and 'MIN' in combined.columns:
            # Sort newest-first (nba_api convention expected by build_training_data)
            if 'GAME_DATE' in combined.columns:
                try:
                    combined['_dt'] = pd.to_datetime(combined['GAME_DATE'], errors='coerce')
                    combined = combined.sort_values('_dt', ascending=False).drop(columns=['_dt'])
                except Exception:
                    pass
            player_logs.append(combined)

        if (idx + 1) % 50 == 0:
            logger.info(f"Fetched logs: {idx + 1}/{len(player_ids)} players")

    logger.info(f"Collected logs for {len(player_logs)} players. Training XGBoost...")

    results = train_models_from_logs(
        player_logs_list=player_logs,
        opp_stats_df=opp_stats_df,
        team_stats_df=team_stats_df,
        def_stats_df=def_stats_df,
        league_avg_pace=league_avg_pace,
    )

    trained  = [m for m, ok in results.items() if ok]
    skipped  = [m for m, ok in results.items() if not ok]
    msg = (
        f"✅ <b>ML Training Complete</b>\n\n"
        f"Trained: {', '.join(trained) or 'none'}\n"
        f"Skipped (insufficient data): {', '.join(skipped) or 'none'}\n"
        f"Players: {len(player_logs)} | Seasons: {', '.join(seasons)}"
    )
    bot.send_message(msg)
    logger.info(f"ML training results: {results}")
    return results


def train_ml_models_clv_feedback(seasons: Optional[List[str]] = None) -> dict:
    """
    Monthly CLV-feedback retraining.

    Identical log-collection pipeline as ``train_ml_models``, but passes
    (player_name, logs) tuples to ``train_models_with_clv_feedback`` so
    XGBoost sample weights are derived from historical CLV outcomes.

    Scheduled once a month (first Sunday).  Falls back to uniform weights
    when the clv_tracking table has insufficient data.
    """
    if seasons is None:
        seasons = _DEFAULT_SEASONS[:_N_SEASONS]

    from src.data.db import DatabaseClient
    stats = NbaStatsClient()
    bot   = TelegramBotClient()
    db    = DatabaseClient()
    bdl_logs = BDLGameLogs(BDLClient(), db=db)

    logger.info(f"CLV feedback training: seasons={seasons}")
    bot.send_message(
        f"🧠 <b>CLV Feedback Training started</b>\nSeasons: {', '.join(seasons)}\n"
        f"Fetching player list..."
    )

    player_ids = stats.get_all_active_player_ids(min_gp=_MIN_GP)

    try:
        opp_stats_df  = stats.get_opponent_stats()
        team_stats_df = stats.get_team_stats()
        def_stats_df  = stats.get_team_defense_stats()
    except Exception as e:
        logger.warning(f"CLV training: could not fetch team stats: {e}")
        opp_stats_df = team_stats_df = def_stats_df = None

    league_avg_pace = 99.0
    if team_stats_df is not None and not team_stats_df.empty and 'PACE' in team_stats_df.columns:
        league_avg_pace = float(team_stats_df['PACE'].mean())

    # Collect (player_name, logs) — name required for CLV lookup
    named_logs: list = []
    for idx, pid in enumerate(player_ids):
        try:
            from nba_api.stats.static import players as _nba_players
            info = _nba_players.find_player_by_id(pid)
            player_name = info['full_name'] if info else str(pid)
        except Exception:
            player_name = str(pid)

        season_dfs = []
        for season in seasons:
            szn_int = int(season.split('-')[0])
            df = bdl_logs.get_player_game_logs(pid, szn_int, ignore_ttl=True)
            if not df.empty:
                season_dfs.append(df)

        if not season_dfs:
            continue

        combined = pd.concat(season_dfs, ignore_index=True)
        if not combined.empty and 'MIN' in combined.columns:
            if 'GAME_DATE' in combined.columns:
                try:
                    combined['_dt'] = pd.to_datetime(combined['GAME_DATE'], errors='coerce')
                    combined = combined.sort_values('_dt', ascending=False).drop(columns=['_dt'])
                except Exception:
                    pass
            named_logs.append((player_name, combined))

        if (idx + 1) % 50 == 0:
            logger.info(f"CLV training logs: {idx + 1}/{len(player_ids)} players")

    logger.info(f"CLV training: {len(named_logs)} players. Running weighted fit...")

    results = train_models_with_clv_feedback(
        player_named_logs=named_logs,
        db=db,
        opp_stats_df=opp_stats_df,
        team_stats_df=team_stats_df,
        def_stats_df=def_stats_df,
        league_avg_pace=league_avg_pace,
    )

    trained = [m for m, ok in results.items() if ok]
    skipped = [m for m, ok in results.items() if not ok]
    bot.send_message(
        f"✅ <b>CLV Feedback Training Complete</b>\n\n"
        f"Trained: {', '.join(trained) or 'none'}\n"
        f"Skipped: {', '.join(skipped) or 'none'}\n"
        f"Players: {len(named_logs)} | Seasons: {', '.join(seasons)}"
    )
    return results


if __name__ == '__main__':
    train_ml_models()
