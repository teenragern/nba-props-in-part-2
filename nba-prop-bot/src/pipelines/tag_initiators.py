"""
Auto-tag primary initiators: top 2 assist-men per team.

Queries nba_api LeagueLeaders for AST, groups by team, and tags
the top 2 players on each team as is_primary_initiator in the DB.

Run via:  python main.py tag_initiators
"""

import time
from typing import Dict, List, Tuple

from src.data.db import DatabaseClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# nba_api team abbreviation → team_id mapping lives in the DB (teams table).
# We resolve team_id from the players table itself.


def tag_initiators():
    """
    Fetch league assist leaders, pick top 2 per team, tag them in the DB.
    """
    from nba_api.stats.endpoints import leagueleaders
    from nba_api.stats.static import teams as nba_teams

    db = DatabaseClient()

    # Fetch league leaders sorted by AST (current season, regular season)
    logger.info("Fetching league assist leaders...")
    time.sleep(0.6)
    leaders = leagueleaders.LeagueLeaders(
        stat_category_abbreviation='AST',
        season_type_all_star='Regular Season',
    )
    df = leaders.get_data_frames()[0]

    if df.empty:
        logger.warning("No league leaders data returned.")
        return

    # df has PLAYER_ID, PLAYER, TEAM_ID, TEAM, AST (and others)
    # Group by TEAM_ID, take top 2 by AST
    tagged_count = 0
    teams_processed = 0

    team_groups = df.groupby('TEAM_ID')
    for team_id, group in team_groups:
        # Already sorted by AST descending from the API
        top_2 = group.head(2)
        player_ids = top_2['PLAYER_ID'].tolist()
        player_names = top_2['PLAYER'].tolist()

        db.set_primary_initiators(int(team_id), [int(p) for p in player_ids])
        tagged_count += len(player_ids)
        teams_processed += 1

        logger.info(
            f"Tagged initiators for team {team_id}: "
            f"{', '.join(f'{n} ({pid})' for n, pid in zip(player_names, player_ids))}"
        )

    logger.info(
        f"Done: tagged {tagged_count} primary initiators across {teams_processed} teams."
    )
    print(f"Tagged {tagged_count} primary initiators across {teams_processed} teams.")
