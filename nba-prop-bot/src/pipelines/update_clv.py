"""
CLV Closing Line Capture.

For each unsettled CLV tracker whose game is in the tip-off window
(up to 90 minutes past scheduled start), fetches the Pinnacle / Circa /
Bookmaker closing price and saves the CLV delta.

Key improvements over v1:
  • Event-targeted  — one API call per event (not one per player).
  • Pinnacle-first  — sharp books in priority order; Pinnacle is the gold
                      standard closing line signal.
  • Tip-off gate    — only processes events within a 10-min pre-game to
                      90-min post-start window; ignores events far in
                      the future or clearly finished.
  • JOIN enrichment — joins clv_tracking with alerts_sent to retrieve the
                      event_id without a schema change.

Credits per run: 1 (get_events) + N per tipping event (typically 1–3).

Schedule: every 30 minutes on game days (see run_scheduler.py).
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

from src.data.db import DatabaseClient
from src.clients.odds_api import OddsApiClient
from src.models.devig import decimal_to_implied_prob
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Timing window ─────────────────────────────────────────────────────────────
# Capture closing lines from 10 minutes before scheduled tip-off until 90
# minutes after.  After 90 minutes the game is usually well into the second
# half and Pinnacle may have suspended prop markets.
_WINDOW_FUTURE = timedelta(minutes=10)   # how far ahead of tip-off we may grab
_WINDOW_PAST   = timedelta(minutes=90)   # how far past tip-off we still try

# Sharp books in priority order.  Pinnacle is the only globally-accepted
# closing-line benchmark; Circa and Bookmaker are used as fallbacks.
_SHARP_PRIORITY = ['pinnacle', 'circa', 'bookmaker', 'betonlineag']


def _norm(s: str) -> str:
    """Lowercase + strip for matching player names / side labels."""
    return s.strip().lower()


def update_clv_lines() -> None:
    """
    Snapshot closing lines for all unsettled CLV trackers in the tip-off window.
    """
    db          = DatabaseClient()
    odds_client = OddsApiClient()

    # ── 1. Pull unsettled trackers + event context ────────────────────────
    # JOIN with alerts_sent so we can group by event_id without a schema change.
    with db.get_conn() as conn:
        rows = conn.execute(
            """
            SELECT ct.id              AS track_id,
                   ct.player_id       AS player,
                   ct.market,
                   ct.side,
                   ct.alert_odds,
                   a.event_id,
                   a.game_date
            FROM   clv_tracking ct
            LEFT   JOIN alerts_sent a
                        ON  a.player_name = ct.player_id
                        AND a.market      = ct.market
                        AND a.side        = ct.side
                        AND date(a.timestamp) = date(ct.alert_time)
            WHERE  ct.closing_odds IS NULL
            ORDER  BY ct.alert_time
            """
        ).fetchall()

    if not rows:
        logger.info("CLV: no unsettled trackers.")
        return

    # ── 2. Group by event_id ──────────────────────────────────────────────
    by_event: Dict[str, List[dict]] = {}
    for row in rows:
        eid = row['event_id']
        if eid:
            by_event.setdefault(eid, []).append(dict(row))
        else:
            logger.debug(f"CLV: no event_id for {row['player']} {row['market']} — skipping.")

    if not by_event:
        logger.info("CLV: no trackers with event_id found.")
        return

    # ── 3. Fetch upcoming events to check commence times (1 credit) ───────
    try:
        events_list = odds_client.get_events()
    except Exception as exc:
        logger.error(f"CLV: get_events failed: {exc}")
        return

    commence_map: Dict[str, datetime] = {}
    for ev in events_list:
        try:
            dt = datetime.fromisoformat(ev['commence_time'].replace('Z', '+00:00'))
            commence_map[ev['id']] = dt
        except Exception:
            pass

    now     = datetime.now(timezone.utc)
    updated = 0

    # ── 4. Process each event in the tip-off window ───────────────────────
    for event_id, trackers in by_event.items():
        commence = commence_map.get(event_id)

        if commence is None:
            # Event not in upcoming list — game is likely already over.
            logger.debug(f"CLV: event {event_id} not in upcoming events — skipping.")
            continue

        age = now - commence  # positive = game started; negative = in future

        if age < -_WINDOW_FUTURE:
            # Too far in future — market still moving; wait for closing line
            logger.debug(
                f"CLV: event {event_id} tips in "
                f"{-age.total_seconds()/60:.0f}min — skipping."
            )
            continue

        if age > _WINDOW_PAST:
            # Game well underway or finished — sharp books have likely
            # suspended props; odds are stale or unavailable.
            logger.info(
                f"CLV: event {event_id} started "
                f"{age.total_seconds()/60:.0f}min ago — may be too late."
            )
            # Still attempt in case the API still has data.

        # One API call per event (not per player)
        markets_needed = list({t['market'] for t in trackers})
        try:
            odds_data = odds_client.get_event_odds(
                event_id=event_id,
                markets=markets_needed,
            )
        except Exception as exc:
            logger.warning(f"CLV: odds fetch failed for event {event_id}: {exc}")
            continue

        bookmakers = odds_data.get('bookmakers', [])

        # Build lookup: {(book_lower, market_key, player_lower, side_lower): price}
        prices: Dict[Tuple, float] = {}
        for book in bookmakers:
            book_key = _norm(book.get('title', ''))
            for mkt in book.get('markets', []):
                mkt_key = mkt.get('key', '')
                for outcome in mkt.get('outcomes', []):
                    desc  = _norm(outcome.get('description', ''))  # player name
                    name  = _norm(outcome.get('name', ''))          # OVER/UNDER
                    price = float(outcome.get('price', 0) or 0)
                    if price > 1.0:
                        prices[(book_key, mkt_key, desc, name)] = price

        # ── Grade each tracker against the closing price ──────────────────
        for tracker in trackers:
            player_n   = _norm(tracker['player'])
            market     = tracker['market']
            side_n     = _norm(tracker['side'])
            track_id   = tracker['track_id']
            alert_odds = float(tracker['alert_odds'])

            # Find sharpest available closing price
            closing_price = 0.0
            source_book   = ''
            for sharp in _SHARP_PRIORITY:
                key = (sharp, market, player_n, side_n)
                if key in prices and prices[key] > 1.0:
                    closing_price = prices[key]
                    source_book   = sharp
                    break

            if closing_price <= 1.0:
                logger.debug(
                    f"CLV: no sharp price for "
                    f"{tracker['player']} {market} {tracker['side']}"
                )
                continue

            implied_closing = decimal_to_implied_prob(closing_price)
            implied_alert   = decimal_to_implied_prob(alert_odds)
            db.update_clv_closing_line(
                track_id, closing_price, implied_closing, implied_alert
            )
            updated += 1
            logger.info(
                f"CLV: {tracker['player']} {market} {tracker['side']} "
                f"alert={alert_odds:.3f} → close={closing_price:.3f} "
                f"({source_book}) | CLV={implied_closing - implied_alert:+.4f}"
            )

    logger.info(f"CLV update complete: {updated} closing line(s) recorded.")


if __name__ == "__main__":
    update_clv_lines()
