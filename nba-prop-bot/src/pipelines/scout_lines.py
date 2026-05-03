"""
Fast line scout — decoupled from the full scan pipeline.

Runs every BDL_SHARP_SCAN_INTERVAL (default 30 min).  Fetches current
Pinnacle / Circa / Bookmaker + rec-book odds for games within
_SCOUT_HOURS_TO_TIP of tip-off and writes snapshots to `line_history`.

This feeds the steam-detection engine with data every 30 min rather than
every 90 min, so sharp line moves surface quickly without waiting for the
next full scan+projection cycle.

Credit cost: 1 Odds API credit per game scouted.  On a 6-game slate running
for 3 hours before tip = 18 credits total across the day — well within budget.
"""

import dateutil.parser
from datetime import datetime, timezone
from typing import Dict, List

from src.clients.odds_api import OddsApiClient
from src.config import PROP_MARKETS, BDL_SHARP_SCAN_INTERVAL
from src.data.db import DatabaseClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Only scout games within this many hours of tip-off; beyond that Pinnacle
# barely moves and the credit spend isn't justified.
_SCOUT_HOURS_TO_TIP: float = 4.0


def _today_upcoming_events(db: DatabaseClient) -> List[Dict]:
    """Read today's games from the local DB — 0 API credits."""
    today = datetime.now().strftime('%Y-%m-%d')
    with db.get_conn() as conn:
        rows = conn.execute(
            """
            SELECT game_id, home_team, away_team, commence_time
            FROM games
            WHERE date(commence_time) = ?
              AND status = 'upcoming'
            ORDER BY commence_time ASC
            """,
            (today,),
        ).fetchall()
    return [dict(r) for r in rows]


def _hours_to_tip(commence_time_iso: str) -> float:
    """Return hours until the game tips off. Negative = already started."""
    try:
        tip = dateutil.parser.isoparse(commence_time_iso)
        if tip.tzinfo is None:
            tip = tip.replace(tzinfo=timezone.utc)
        delta = (tip - datetime.now(timezone.utc)).total_seconds() / 3600.0
        return delta
    except Exception:
        return float('inf')


def _extract_line_records(event_id: str, bookmakers: List[Dict]) -> List[tuple]:
    """
    Parse the bookmakers list from get_event_odds() into (player, market,
    bookmaker, line, side, odds, implied_prob) tuples ready for
    insert_line_history_batch().
    """
    records: List[tuple] = []
    for book in bookmakers:
        book_name = (book.get('title') or '').lower().strip()
        for mkt in book.get('markets', []):
            market_key = mkt.get('key', '')
            if market_key not in PROP_MARKETS:
                continue
            for outcome in mkt.get('outcomes', []):
                player = outcome.get('description') or outcome.get('name')
                if not player:
                    continue
                line = outcome.get('point')
                if line is None:
                    continue
                side = (outcome.get('name') or '').upper()
                price = outcome.get('price')
                if not price or price <= 1.0:
                    continue
                implied = 1.0 / price
                records.append((player, market_key, book_name, line, side, price, implied))
    return records


def scout_lines(odds_client: OddsApiClient = None) -> Dict:
    """
    Fetch current lines for games near tip-off and write to line_history.

    Returns:
        {
            'games_scouted': int,
            'records_written': int,
            'credits_used': int,
            'skipped': int,   # games outside the scout window
        }
    """
    db = DatabaseClient()
    if odds_client is None:
        odds_client = OddsApiClient()

    events = _today_upcoming_events(db)
    if not events:
        logger.info("Scout: no upcoming games in DB today — run sync_events first.")
        return {'games_scouted': 0, 'records_written': 0, 'credits_used': 0, 'skipped': 0}

    credits_before = odds_client.requests_remaining
    games_scouted = 0
    skipped = 0
    total_records = 0

    for event in events:
        hours = _hours_to_tip(event['commence_time'])
        if hours < 0:
            # Already tipped — game should be flipped to 'started' by sync_events
            skipped += 1
            continue
        if hours > _SCOUT_HOURS_TO_TIP:
            logger.debug(
                f"Scout: skipping {event['home_team']} vs {event['away_team']} "
                f"({hours:.1f}h to tip)"
            )
            skipped += 1
            continue

        try:
            data = odds_client.get_event_odds(event['game_id'], markets=PROP_MARKETS)
            bookmakers = data.get('bookmakers', [])
            records = _extract_line_records(event['game_id'], bookmakers)
            if records:
                db.insert_line_history_batch(records)
                total_records += len(records)
            games_scouted += 1
            logger.info(
                f"Scout: {event['home_team']} vs {event['away_team']} "
                f"({hours:.1f}h to tip) — {len(records)} line records written"
            )
        except Exception as e:
            logger.warning(f"Scout: failed to fetch {event['game_id']}: {e}")

    credits_used = max(0, credits_before - odds_client.requests_remaining) if credits_before else games_scouted
    logger.info(
        f"Scout complete: {games_scouted} game(s) scouted, "
        f"{total_records} records written, "
        f"~{credits_used} credit(s) used, "
        f"{skipped} game(s) skipped (outside {_SCOUT_HOURS_TO_TIP}h window)"
    )
    return {
        'games_scouted': games_scouted,
        'records_written': total_records,
        'credits_used': credits_used,
        'skipped': skipped,
    }


if __name__ == "__main__":
    scout_lines()
