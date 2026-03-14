"""
Full historical backtester.

For each active player, pulls 2-3 seasons of game logs from nba_api and
replays every past game chronologically:

  Simulated line  = player's rolling 10-game average for that stat
                    (what a sharp book would roughly hang as the number)
  Model mean      = build_player_projection() using only past games as features
  Model prob      = Poisson/Normal P(actual > line) via get_probability_distribution()
  Edge            = model_prob_over - 0.5   (fair line implied is 0.50)
  Hit             = 1 if actual stat > simulated line

Results are inserted into the backtest_results DB table.
A Telegram summary is sent when complete.

Usage:
    python -m src.pipelines.backtest          # uses BACKTEST_SEASONS env var or 3-season default
    BACKTEST_SEASONS=2 python -m src.pipelines.backtest

Environment:
    BACKTEST_MIN_GP   minimum games played to include a player (default 25)
    BACKTEST_SEASONS  number of past seasons to include (default 3)
    BACKTEST_EDGE_MIN minimum edge to count a bet (default 0.03)
"""

import os
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from src.utils.logging_utils import get_logger
from src.data.db import DatabaseClient
from src.clients.nba_stats import NbaStatsClient
from src.clients.telegram_bot import TelegramBotClient
from src.models.projections import build_player_projection, get_market_col
from src.models.distributions import get_probability_distribution

logger = get_logger(__name__)

# Market → stat column
_MARKET_COL = {
    'player_points':   'PTS',
    'player_rebounds': 'REB',
    'player_assists':  'AST',
    'player_threes':   'FG3M',
}

# Seasons to backtest (most recent first)
_DEFAULT_SEASONS = ['2024-25', '2023-24', '2022-23']

_MIN_HISTORY_GAMES = 15   # need this many prior games to simulate a bet
_LINE_WINDOW       = 10   # rolling average window used as the simulated line
_EDGE_MIN_BET      = float(os.getenv('BACKTEST_EDGE_MIN', '0.03'))
_MIN_GP            = int(os.getenv('BACKTEST_MIN_GP',    '25'))
_N_SEASONS         = int(os.getenv('BACKTEST_SEASONS',   '3'))


def _rolling_avg(series: pd.Series, window: int) -> Optional[float]:
    """Simple rolling average of last `window` values. Returns None if insufficient data."""
    vals = series.dropna().values
    if len(vals) < 3:
        return None
    return float(vals[:window].mean())


def _simulate_player(logs_chron: pd.DataFrame, market: str) -> List[Tuple]:
    """
    Replay a player's game log chronologically. Returns list of result tuples:
      (player_name, season, market, game_date, sim_line, model_mean,
       model_prob_over, actual_stat, hit, edge)
    """
    col = _MARKET_COL.get(market)
    if col not in logs_chron.columns or 'MIN' not in logs_chron.columns:
        return []

    results = []
    n = len(logs_chron)

    for i in range(_MIN_HISTORY_GAMES, n):
        current  = logs_chron.iloc[i]
        actual   = float(current.get(col, 0) or 0)
        act_mins = float(current.get('MIN', 0) or 0)

        # Skip DNPs
        if act_mins <= 0:
            continue

        # History = all games before this one (oldest-to-newest slice)
        # Build newest-first slice for model (nba_api convention)
        history_newest_first = logs_chron.iloc[:i][::-1].reset_index(drop=True)

        # Simulated line = rolling 10-game average of the stat
        sim_line = _rolling_avg(history_newest_first[col], _LINE_WINDOW)
        if sim_line is None or sim_line <= 0:
            continue

        # Projection (no injury/rotation adjustments — testing base rate model)
        try:
            proj = build_player_projection(
                player_id=str(current.get('PLAYER_NAME', '')),
                market=market,
                line=sim_line,
                recent_logs=history_newest_first,
                season_logs=history_newest_first,
                injury_status='Healthy',
                team_pace=99.0,
                opp_pace=99.0,
                opponent_multiplier=1.0,
            )
        except Exception:
            continue

        if not proj or proj.get('mean', 0) <= 0:
            continue

        model_mean = proj['mean']
        proj_mins  = proj.get('projected_minutes', 0)

        # Probability from distribution model
        try:
            dists = get_probability_distribution(
                market, model_mean, sim_line,
                logs=history_newest_first,
                variance_scale=1.0,
                proj_minutes=proj_mins,
            )
        except Exception:
            continue

        prob_over = float(dists.get('prob_over', 0.5))
        edge      = prob_over - 0.5
        hit       = int(actual > sim_line)

        try:
            season     = str(current.get('SEASON_YEAR', '')).strip()
            game_date  = str(current.get('GAME_DATE', '')).strip()
            player_name = str(current.get('PLAYER_NAME', '')).strip()
        except Exception:
            continue

        results.append((
            player_name, season, market, game_date,
            round(sim_line, 2), round(model_mean, 3),
            round(prob_over, 4), round(actual, 2),
            hit, round(edge, 4),
        ))

    return results


def run_backtest(seasons: Optional[List[str]] = None) -> Dict:
    """
    Main entry point. Returns summary dict with ROI / calibration metrics.
    Inserts all results into backtest_results table.
    """
    if seasons is None:
        seasons = _DEFAULT_SEASONS[:_N_SEASONS]

    db     = DatabaseClient()
    stats  = NbaStatsClient()
    bot    = TelegramBotClient()

    logger.info(f"Starting backtest: seasons={seasons}")
    bot.send_message(f"🔬 <b>Backtest started</b>\nSeasons: {', '.join(seasons)}\n"
                     f"Fetching active players...")

    player_ids = stats.get_all_active_player_ids(min_gp=_MIN_GP)
    logger.info(f"Backtesting {len(player_ids)} active players across {len(seasons)} seasons.")

    total_rows   = 0
    player_count = 0

    for pid in player_ids:
        # Fetch and concatenate logs across seasons (newest season first)
        season_dfs = []
        for season in seasons:
            df = stats.get_player_game_logs_season(pid, season)
            if not df.empty:
                season_dfs.append(df)
            time.sleep(0.15)   # light throttle

        if not season_dfs:
            continue

        # Combine: newest-first within each season, then stack seasons newest-first
        combined = pd.concat(season_dfs, ignore_index=True)
        if combined.empty or 'MIN' not in combined.columns:
            continue

        # Sort chronologically (oldest first) for the replay loop
        try:
            combined['_parsed_date'] = pd.to_datetime(combined['GAME_DATE'], errors='coerce')
            combined = combined.sort_values('_parsed_date').reset_index(drop=True)
        except Exception:
            continue

        # Run simulation for each market
        batch: List[Tuple] = []
        for market in _MARKET_COL:
            rows = _simulate_player(combined, market)
            batch.extend(rows)

        if batch:
            db.insert_backtest_results_batch(batch)
            total_rows   += len(batch)
            player_count += 1

        if player_count % 25 == 0:
            logger.info(f"Backtest progress: {player_count}/{len(player_ids)} players, "
                        f"{total_rows} simulated bets so far.")

    # Retrieve and format summary
    summary = db.get_backtest_summary()
    _send_summary(bot, summary, total_rows, player_count, seasons)
    return summary


def _send_summary(bot: TelegramBotClient, summary: Dict,
                  total_rows: int, player_count: int,
                  seasons: List[str]) -> None:
    lines = [
        f"📊 <b>Backtest Complete</b>",
        f"Players: {player_count} | Simulated bets: {total_rows:,}",
        f"Seasons: {', '.join(seasons)}",
        "",
        "<b>ROI by Market (−110 odds):</b>",
    ]

    for m in summary.get('markets', []):
        roi = m['profit'] / max(m['bets'], 1)
        lines.append(
            f"  {m['market'].replace('player_', '').replace('_', ' ').title()}: "
            f"{m['bets']:,} bets | {m['hit_rate']:.1%} hit | ROI {roi:+.2%}"
        )

    lines += ["", "<b>Edge Bucket Hit Rates:</b>"]
    for b in summary.get('buckets', []):
        lines.append(
            f"  {b['bucket']}: {b['bets']:,} bets | {b['hit_rate']:.1%} hit"
        )

    lines += ["", "<b>Calibration (model prob → actual hit rate):</b>"]
    for c in summary.get('calibration', []):
        if c['n'] >= 20:
            lines.append(
                f"  {c['prob_bin']:.0%}: model={c['prob_bin']:.1%} "
                f"actual={c['actual_hit_rate']:.1%} (n={c['n']})"
            )

    bot.send_message('\n'.join(lines))
    logger.info("Backtest summary sent.")


if __name__ == '__main__':
    run_backtest()
