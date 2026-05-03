"""
Backtester to mathematically prove expected ROI on historical closing lines.
Uses an exact replay simulator over the `line_history` table, matching 
historical game logs strictly WITHOUT lookahead bias.
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
import os

from src.data.db import DatabaseClient
from src.models.ml_model import get_ml_projection, _MARKET_COL
from src.models.distributions import get_probability_distribution, classify_bench_tier
from src.models.calibration_model import calibrate_prob
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Fast offline mapping (reduces reliance on nba_api to translate names -> PIDs)
def build_nba_api_lookup() -> dict:
    logger.info("Initializing NBA static player dict...")
    from nba_api.stats.static import players as nba_players
    mapping = {}
    for p in nba_players.get_active_players():
        mapping[p['full_name']] = p['id']
    return mapping

def run_backtest():
    db = DatabaseClient()
    
    # Preload NBA API names to IDs
    name_to_id = build_nba_api_lookup()

    logger.info("Querying historical lines for chronological backtesting...")
    # One row per (player, market, day) = the last line snapshot of the day.
    query_lines = """
        SELECT lh.player_name, lh.market, date(lh.timestamp) AS game_date, lh.line
        FROM line_history lh
        WHERE lh.side = 'OVER'
          AND lh.timestamp = (
              SELECT MAX(lh2.timestamp)
              FROM line_history lh2
              WHERE lh2.player_name = lh.player_name
                AND lh2.market     = lh.market
                AND date(lh2.timestamp) = date(lh.timestamp)
                AND lh2.side = 'OVER'
          )
    """
    
    with db.get_conn() as conn:
        lines_df = pd.read_sql(query_lines, conn)
        logger.info(f"Loaded {len(lines_df)} unique player/market/lines to backtest.")
        
        # Load entirely into memory for blazing fast offline simulation
        bdl_df = pd.read_sql("SELECT * FROM bdl_game_log_cache ORDER BY game_date ASC", conn)
        logger.info(f"Loaded {len(bdl_df)} BDL game logs for outcome resolution and ML training.")

    # Rename DB snake_case columns → nba_api-compatible uppercase so
    # get_ml_projection / get_probability_distribution don't KeyError.
    bdl_df = bdl_df.rename(columns={
        'min':        'MIN',
        'pts':        'PTS',
        'reb':        'REB',
        'ast':        'AST',
        'fg3m':       'FG3M',
        'blk':        'BLK',
        'stl':        'STL',
        'fga':        'FGA',
        'fta':        'FTA',
        'tov':        'TOV',
        'team_abbr':  'TEAM_ABBREVIATION',
        'matchup':    'MATCHUP',
        'game_date':  'GAME_DATE',
        'wl':         'WL',
    })
    
    total_simulated = 0
    hits = 0
    expected_hits = 0.0
    batch_results = []
    
    logger.info("Starting Simulation Loop (Strict No-Lookahead)...")
    start_time = time.time()
    
    # Iterate dynamically
    for idx, row in lines_df.iterrows():
        pname = row['player_name']
        mkt = row['market']
        gdate = row['game_date']
        line = row['line']
        
        pid = name_to_id.get(pname, 0)
        if pid == 0:
            continue
            
        # Extract logs: STRICT NO-LOOKAHEAD (past games only)
        # Using string comparison for dates works if YYYY-MM-DD
        player_bdl = bdl_df[bdl_df['player_id'] == pid]
        past_logs = player_bdl[player_bdl['GAME_DATE'] < gdate].sort_values('GAME_DATE', ascending=False)

        if len(past_logs) < 15:
            continue  # Insufficient ML sampling history

        # Reveal actual outcome: The Ground Truth
        today_log = player_bdl[player_bdl['GAME_DATE'] == gdate]
        if today_log.empty:
            continue  # Void bet (DNP or PPD)

        actual_record = today_log.iloc[0]

        # Map market → uppercase column name (matches nba_api-compatible schema)
        mkt_col = _MARKET_COL.get(mkt, 'PTS')
        actual_val = actual_record.get(mkt_col, 0)
        actual_stat = float(actual_val) if actual_val is not None else 0.0

        # Determine logical Minutes ceiling (unbiased backtest utilizes recent trailing mean)
        recent = past_logs.head(5)
        proj_min = recent['MIN'].mean() if 'MIN' in recent.columns else 24.0
        
        # Inference XGBoost
        model_mean = get_ml_projection(
            market=mkt,
            logs=past_logs,
            proj_minutes=proj_min,
            home_flag=False, # Neutral defaults for high-speed simulation lacking schedule matrix
        )
        
        if model_mean is None:
            continue

        bench_tier = classify_bench_tier(proj_min)
            
        # Bayesian Dist Curve Generation
        try:
            dists = get_probability_distribution(
                market=mkt,
                mean_proj=model_mean,
                line=line,
                logs=past_logs,
                bench_tier=bench_tier,
                proj_minutes=proj_min
            )
            raw_prob_over = dists.get('prob_over', 0.5)
        except TypeError:
            continue # Function signature fallback failed
        
        calibrated_prob = calibrate_prob(raw_prob_over)
        
        is_hit = 1 if actual_stat > line else 0
        edge = calibrated_prob - 0.50  # Flat 50% baseline for theoretical math
        
        yr = int(gdate[:4])
        season_label = f"{yr}-{str(yr + 1)[2:]}"
        batch_results.append((
            pname, season_label, mkt, gdate, line, model_mean, calibrated_prob,
            actual_stat, is_hit, edge
        ))
        
        total_simulated += 1
        hits += is_hit
        expected_hits += calibrated_prob
        
        if total_simulated % 500 == 0:
            logger.info(f"Simulated {total_simulated} props in {time.time()-start_time:.1f}s... WinRate: {hits/total_simulated:.1%} | Exp: {expected_hits/total_simulated:.1%}")

    with db.get_conn() as conn:
        conn.execute("DELETE FROM backtest_results") 
        conn.executemany("""
            INSERT INTO backtest_results 
            (player_name, season, market, game_date, simulated_line, model_mean, model_prob_over, actual_stat, hit, edge)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch_results)

    logger.info("=================================")
    logger.info("       BACKTEST COMPLETE         ")
    logger.info("=================================")
    logger.info(f"Total Bets Simulated: {total_simulated}")
    if total_simulated > 0:
        logger.info(f"Actual Accuracy:     {hits / total_simulated:.2%}")
        logger.info(f"Expected Accuracy:   {expected_hits / total_simulated:.2%}")
        logger.info(f"Execution Time:      {time.time()-start_time:.1f}s")

if __name__ == '__main__':
    run_backtest()
