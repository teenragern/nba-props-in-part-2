"""
Priority 9: Referee stats database from historical boxscore data.

Builds a profile of each referee's impact on pace and foul rates by
aggregating BoxScoreSummaryV2 data from past games. When referee
assignments are known before a game (requires scraping official.nba.com),
the historical profile is used to adjust pace projections.

Usage:
  from src.models.referee_stats import get_game_referee_factor
  factor = get_game_referee_factor(game_id)  # applied to pace in projections
"""

import time
import pandas as pd
from typing import List, Dict
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_referees_for_game(game_id: str) -> List[str]:
    """
    Extract referee names from BoxScoreSummaryV2 for a completed game.
    Returns a list of full names, e.g. ['Scott Foster', 'Tony Brothers'].
    """
    try:
        from nba_api.stats.endpoints import boxscoresummaryv2
        summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
        time.sleep(0.6)
        dfs = summary.get_data_frames()
        # DataFrame index 2 is the officials table
        if len(dfs) <= 2:
            return []
        officials = dfs[2]
        if officials.empty:
            return []
        names = (officials['FIRST_NAME'].str.strip() + ' ' + officials['LAST_NAME'].str.strip()).tolist()
        return [n for n in names if n.strip()]
    except Exception as e:
        logger.debug(f"Could not fetch referees for game {game_id}: {e}")
        return []


def build_referee_profiles(game_ids: List[str], db) -> Dict[str, Dict]:
    """
    Build referee pace profiles from a list of historical game IDs.
    Stores results in the referee_stats table via `db`.

    Run once per week/month to refresh profiles.
    Returns {referee_name: {avg_pace, games_tracked}}.
    """
    referee_games: Dict[str, List[float]] = {}

    for game_id in game_ids:
        refs = get_referees_for_game(game_id)
        if not refs:
            continue

        # Get game pace from box score summary
        try:
            from nba_api.stats.endpoints import boxscoresummaryv2
            summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id)
            time.sleep(0.6)
            dfs = summary.get_data_frames()
            # DataFrame 1 is game info with scores — use total possessions proxy
            # Simpler: use total PTS as pace proxy (higher pts ≈ faster pace)
            game_info = dfs[0] if len(dfs) > 0 else pd.DataFrame()
            if game_info.empty:
                continue

            # Use average PTS as pace signal (normalise later)
            total_pts = game_info['PTS'].sum() if 'PTS' in game_info.columns else 0
            if total_pts <= 0:
                continue

            for ref in refs:
                referee_games.setdefault(ref, []).append(float(total_pts))
        except Exception:
            continue

    profiles = {}
    if not referee_games:
        return profiles

    # League-wide average across all games sampled
    all_pts = [p for pts_list in referee_games.values() for p in pts_list]
    league_avg = sum(all_pts) / len(all_pts) if all_pts else 220.0

    for ref_name, pts_list in referee_games.items():
        avg = sum(pts_list) / len(pts_list)
        profiles[ref_name] = {
            'avg_pace_proxy': avg,
            'pace_multiplier': avg / league_avg if league_avg > 0 else 1.0,
            'games_tracked': len(pts_list),
        }

        # Persist to DB
        try:
            with db.get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO referee_stats
                        (referee_name, avg_pace, games_tracked, last_updated)
                    VALUES (?, ?, ?, date('now'))
                    """,
                    (ref_name, avg / league_avg if league_avg > 0 else 1.0, len(pts_list))
                )
        except Exception as e:
            logger.warning(f"Failed to persist referee stats for {ref_name}: {e}")

    logger.info(f"Built referee profiles for {len(profiles)} officials.")
    return profiles


def get_game_referee_factor(game_id: str, db=None) -> float:
    """
    Return a pace adjustment factor for the referees assigned to a game.

    If referee data is in the DB and the game's officials are known,
    returns avg(their pace multiplier). Falls back to 1.0 if unknown.

    Note: Pre-game referee assignments require scraping official.nba.com.
    This function is most useful post-game or for live-game contexts.
    """
    refs = get_referees_for_game(game_id)
    if not refs or db is None:
        return 1.0

    factors = []
    try:
        with db.get_conn() as conn:
            for ref_name in refs:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT avg_pace FROM referee_stats WHERE referee_name = ?",
                    (ref_name,)
                )
                row = cursor.fetchone()
                if row and row['avg_pace']:
                    factors.append(float(row['avg_pace']))
    except Exception:
        return 1.0

    if not factors:
        return 1.0

    avg_factor = sum(factors) / len(factors)
    # Clamp to reasonable range [0.95, 1.05]
    return float(max(0.95, min(1.05, avg_factor)))
