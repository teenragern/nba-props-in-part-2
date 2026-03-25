"""
Tier-2 alert digest — flushes all queued pending_alerts into a single
beautifully formatted Telegram message.

Scheduled at 12:00 PM, 3:00 PM, and 6:00 PM ET on game days by
run_scheduler.py.  Can also be called manually:

    python -m src.pipelines.flush_alerts
"""

from datetime import datetime
from typing import List

from src.data.db import DatabaseClient
from src.clients.telegram_bot import TelegramBotClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Maximum prop rows shown per digest before truncation.
_MAX_PROPS_SHOWN    = 15
_MAX_GAME_MKT_SHOWN = 10
_MAX_PARLAYS_SHOWN  = 5


def flush_pending_alerts(db: DatabaseClient, bot: TelegramBotClient) -> int:
    """
    Pull all unsent pending_alerts, format a grouped digest, send it,
    then mark all included rows as sent.

    Returns the number of alerts included in the digest (0 = nothing sent).
    """
    rows = db.get_pending_alerts(unsent_only=True)
    if not rows:
        logger.info("flush_pending_alerts: nothing to send.")
        return 0

    props     = [r for r in rows if r['alert_type'] == 'prop']
    game_mkts = [r for r in rows if r['alert_type'] == 'game_market']
    parlays   = [r for r in rows if r['alert_type'] == 'parlay']

    total = len(rows)
    now   = datetime.now().strftime("%-I:%M %p")   # e.g. "3:00 PM"

    lines: List[str] = [
        f"🏀 <b>NBA Prop Slate — {now} Update</b>",
        f"<i>{total} edge{'s' if total != 1 else ''} queued since last digest</i>",
    ]

    # ── Player Props ──────────────────────────────────────────────────────
    if props:
        lines.append(f"\n📈 <b>Player Props ({len(props)})</b>")
        for r in props[:_MAX_PROPS_SHOWN]:
            lines.append(f"• {r['title']}")
        overflow = len(props) - _MAX_PROPS_SHOWN
        if overflow > 0:
            lines.append(f"  <i>… and {overflow} more</i>")

    # ── Game Markets ──────────────────────────────────────────────────────
    if game_mkts:
        lines.append(f"\n⚡ <b>Game Markets ({len(game_mkts)})</b>")
        for r in game_mkts[:_MAX_GAME_MKT_SHOWN]:
            lines.append(f"• {r['title']}")
        overflow = len(game_mkts) - _MAX_GAME_MKT_SHOWN
        if overflow > 0:
            lines.append(f"  <i>… and {overflow} more</i>")

    # ── SGPs / Parlays ────────────────────────────────────────────────────
    if parlays:
        lines.append(f"\n🔗 <b>SGPs ({len(parlays)})</b>")
        for r in parlays[:_MAX_PARLAYS_SHOWN]:
            lines.append(f"• {r['title']}")

    bot.send_instant("\n".join(lines))
    db.mark_pending_alerts_sent([r['id'] for r in rows])

    logger.info(
        f"Digest sent: {len(props)} props, {len(game_mkts)} game markets, "
        f"{len(parlays)} parlays"
    )
    return total


if __name__ == "__main__":
    flush_pending_alerts(DatabaseClient(), TelegramBotClient())
