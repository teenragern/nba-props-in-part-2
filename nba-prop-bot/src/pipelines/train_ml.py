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
from src.clients.telegram_bot import TelegramBotClient
from src.models.ml_model import train_models_from_logs

logger = get_logger(__name__)

_DEFAULT_SEASONS = ['2024-25', '2023-24', '2022-23']
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
            df = stats.get_player_game_logs_season(pid, season)
            if not df.empty:
                season_dfs.append(df)
            time.sleep(0.15)

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


if __name__ == '__main__':
    train_ml_models()
