import os
import json
from datetime import datetime, timezone, timedelta
from src.clients.sportradar_client import SportradarClient
from src.data.db import get_db_client
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def sync_sportradar_schedule():
    """Sync today's schedule from Sportradar and link to existing games (Suggestion 5)."""
    client = SportradarClient()
    db = get_db_client()
    
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = client.get_prematch_schedule(date_str)
    if not data or 'schedules' not in data:
        logger.warning("Sportradar: No schedule data returned.")
        return
        
    games = data['schedules']
    count = 0
    with db.get_conn() as conn:
        for g_item in games:
            g = g_item.get('sport_event', {})
            sr_id = g.get('id')
            
            # Find home and away team names from competitors
            home = "Unknown"
            away = "Unknown"
            for comp in g.get('competitors', []):
                if comp.get('qualifier') == 'home':
                    home = comp.get('name')
                elif comp.get('qualifier') == 'away':
                    away = comp.get('name')
                    
            start = g.get('start_time')
            status = g.get('status')
            
            # Try to find existing game by home team and date to avoid duplicates
            # Odds API IDs are already in the table from job_sync. 
            # Use a +/- 4 hour window to handle slight differences in scheduled start times.
            logger.debug(f"Sportradar: Attempting to link {away} @ {home} ({start})...")
            cur = conn.execute(
                """
                SELECT game_id, home_team, commence_time FROM games 
                WHERE home_team ILIKE %s 
                  AND commence_time >= %s::timestamptz - INTERVAL '4 hours'
                  AND commence_time <= %s::timestamptz + INTERVAL '4 hours'
                  AND game_id NOT LIKE '%%-%%-%%'
                LIMIT 1
                """,
                (f"%{home}%", start, start)
            )
            row = cur.fetchone()
            
            if row:
                logger.info(f"Sportradar: Linked {home} to existing game {row['game_id']}")
                # Update existing row with sr_id
                conn.execute(
                    "UPDATE games SET sr_id = %s, status = %s WHERE game_id = %s",
                    (sr_id, status, row['game_id'])
                )
                # Cleanup: if there's a Sportradar-only row for this same game, delete it
                if row['game_id'] != sr_id:
                    conn.execute(
                        "DELETE FROM games WHERE game_id = %s AND sr_id = %s",
                        (sr_id, sr_id)
                    )
            else:
                logger.info(f"Sportradar: No existing match for {home} - creating new row.")
                logger.info(f"Sportradar: No existing match for {home} - creating new row.")
                # Insert new row (Sportradar-only game for now)
                conn.execute(
                    """
                    INSERT INTO games (game_id, sr_id, home_team, away_team, commence_time, status)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (game_id) DO UPDATE SET
                        sr_id = EXCLUDED.sr_id,
                        status = EXCLUDED.status
                    """,
                    (sr_id, sr_id, home, away, start, status)
                )
            count += 1
            
    logger.info(f"Sportradar: Synced and linked {count} games.")

def sync_sportradar_lineups():
    """Fetch official lineups for active/upcoming games using sr_id (Suggestion 1)."""
    client = SportradarClient()
    db = get_db_client()
    
    # Get sr_id for games today that aren't finished yet
    with db.get_conn() as conn:
        cur = conn.execute(
            """
            SELECT sr_id, home_team, away_team FROM games 
            WHERE status NOT IN ('closed', 'complete') 
              AND commence_time::date = CURRENT_DATE
              AND sr_id IS NOT NULL
            """
        )
        games_to_sync = [dict(r) for r in cur.fetchall()]
        
    if not games_to_sync:
        logger.info("Sportradar: No active games with Sportradar IDs to sync lineups for.")
        return
        
    logger.info(f"Sportradar: Syncing lineups for {len(games_to_sync)} games...")
    for g in games_to_sync:
        sr_id = g['sr_id']
        logger.info(f"Sportradar: Fetching summary for {g['away_team']} @ {g['home_team']} ({sr_id})...")
        summary = client.get_game_summary(sr_id)
        if not summary:
            continue
            
        starters_found = 0
        for side in ['home', 'away']:
            team_name = summary.get(side, {}).get('name')
            players = summary.get(side, {}).get('players', [])
            
            for p in players:
                if p.get('starter'):
                    pname = p.get('full_name')
                    starters_found += 1
                    with db.get_conn() as conn:
                        conn.execute(
                            """
                            INSERT INTO lineups_official (game_id, player_name, team, is_starter)
                            VALUES (%s, %s, %s, TRUE)
                            ON CONFLICT (game_id, player_name) DO UPDATE SET
                                is_starter = TRUE,
                                updated_at = NOW()
                            """,
                            (sr_id, pname, team_name)
                        )
        logger.info(f"Sportradar: Found {starters_found} starters for {sr_id}")
                        
    logger.info("Sportradar: Lineup sync complete.")

def audit_finished_games():
    """Fetch boxscores and PBP for recently finished games using sr_id (Suggestion 2)."""
    client = SportradarClient()
    db = get_db_client()
    
    # Find games that finished but haven't been audited
    with db.get_conn() as conn:
        cur = conn.execute(
            """
            SELECT g.sr_id FROM games g
            LEFT JOIN sportradar_audit_logs l ON g.sr_id = l.game_id
            WHERE g.status IN ('closed', 'complete')
              AND (l.game_id IS NULL OR l.pbp_saved = FALSE OR l.box_saved = FALSE)
              AND g.commence_time >= NOW() - INTERVAL '48 hours'
              AND g.sr_id IS NOT NULL
            """
        )
        sr_ids = [r['sr_id'] for r in cur.fetchall()]
        
    if not sr_ids:
        logger.info("Sportradar: No new finished games with Sportradar IDs to audit.")
        return
        
    logger.info(f"Sportradar: Auditing {len(sr_ids)} games...")
    os.makedirs("data/sportradar_audit", exist_ok=True)
    
    for sid in sr_ids:
        # 1. Boxscore
        box = client.get_game_boxscore(sid)
        if box:
            with open(f"data/sportradar_audit/box_{sid}.json", 'w') as f:
                json.dump(box, f)
            
        # 2. PBP
        pbp = client.get_game_pbp(sid)
        if pbp:
            with open(f"data/sportradar_audit/pbp_{sid}.json", 'w') as f:
                json.dump(pbp, f)
                
        # Update audit log
        with db.get_conn() as conn:
            conn.execute(
                """
                INSERT INTO sportradar_audit_logs (game_id, pbp_saved, box_saved)
                VALUES (%s, %s, %s)
                ON CONFLICT (game_id) DO UPDATE SET
                    pbp_saved = EXCLUDED.pbp_saved,
                    box_saved = EXCLUDED.box_saved,
                    audited_at = NOW()
                """,
                (sid, pbp is not None, box is not None)
            )
            
    logger.info("Sportradar: Audit complete.")

def sync_synergy_stats():
    """Fetch advanced Synergy play-type stats for all active teams to use in ML features."""
    client = SportradarClient()
    db = get_db_client()
    
    # We will fetch team-level synergy stats to keep API usage low on the trial key.
    # To get player-level, we would iterate over player UUIDs.
    # Sportradar team UUIDs can be extracted from game summaries.
    with db.get_conn() as conn:
        # Just an example list of a few team UUIDs for demonstration. 
        # In a full system, you'd map your SQLite team_ids to Sportradar UUIDs.
        cur = conn.execute("SELECT DISTINCT home_team, sr_id FROM games WHERE sr_id IS NOT NULL LIMIT 1")
        row = cur.fetchone()
        
    if not row:
        logger.warning("Sportradar: No games with sr_id found to extract team UUIDs for Synergy sync.")
        return
        
    logger.info("Sportradar: Syncing Synergy Team Stats...")
    # Get the game summary to extract the team UUID
    summary = client.get_game_summary(row['sr_id'])
    if not summary:
        return
        
    home_team_id = summary.get('home', {}).get('id')
    away_team_id = summary.get('away', {}).get('id')
    
    for tid in [home_team_id, away_team_id]:
        if not tid: continue
        stats = client.get_synergy_team_stats(tid)
        if not stats or 'play_types' not in stats:
            continue
            
        season = stats.get('season', {}).get('year', '2024')
        team_name = stats.get('name', 'Unknown')
        
        with db.get_conn() as conn:
            for pt in stats['play_types']:
                play_type = pt.get('type', 'Unknown')
                ppp = pt.get('points_per_possession', 0.0)
                freq = pt.get('frequency', 0.0)
                percentile = pt.get('percentile', 0.0)
                
                conn.execute(
                    """
                    INSERT INTO synergy_stats (player_id, player_name, team_id, season, play_type, points_per_poss, frequency_pct, percentile)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (player_id) DO UPDATE SET
                        points_per_poss = EXCLUDED.points_per_poss,
                        frequency_pct = EXCLUDED.frequency_pct,
                        percentile = EXCLUDED.percentile,
                        last_updated = NOW()
                    """,
                    (f"team_{tid}_{play_type}", "TEAM", tid, season, play_type, ppp, freq, percentile)
                )
    logger.info("Sportradar: Synergy stats sync complete.")

def sync_futures():
    """Fetch futures markets (e.g. NBA Championship, MVP)."""
    client = SportradarClient()
    db = get_db_client()
    
    # Typically, futures are attached to a tournament/season UUID.
    # For trial, we might use a known season UUID or just fetch the general schedule.
    # We will use a placeholder season UUID for the current NBA season.
    # e.g., 'sr:tournament:132' for NBA
    season_urn = "sr:tournament:132"
    
    logger.info("Sportradar: Syncing Futures Odds...")
    data = client.get_futures_markets(season_urn)
    if not data or 'markets' not in data:
        logger.warning("Sportradar: No futures markets returned.")
        return
        
    with db.get_conn() as conn:
        for market in data['markets']:
            market_name = market.get('name')
            for book in market.get('books', []):
                book_name = book.get('name')
                for outcome in book.get('outcomes', []):
                    sel_name = outcome.get('name')
                    odds = outcome.get('odds')
                    prob = 1.0 / odds if odds and odds > 0 else 0.0
                    
                    conn.execute(
                        """
                        INSERT INTO futures_odds (market_name, selection_name, odds, implied_prob, bookmaker)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (market_name, sel_name, odds, prob, book_name)
                    )
    logger.info("Sportradar: Futures sync complete.")

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if cmd == "schedule":
        sync_sportradar_schedule()
    elif cmd == "lineups":
        sync_sportradar_lineups()
    elif cmd == "audit":
        audit_finished_games()
    elif cmd == "synergy":
        sync_synergy_stats()
    elif cmd == "futures":
        sync_futures()
    else:
        sync_sportradar_schedule()
        sync_sportradar_lineups()
        audit_finished_games()
        sync_synergy_stats()
        sync_futures()
