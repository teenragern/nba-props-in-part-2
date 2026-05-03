"""
Injury sync pipeline.

Pulls per-player availability from every available source (BDL, NBA stats API,
Rotowire, CBS), merges with most-pessimistic-wins, and upserts to
`injury_reports` keyed on (game_date, player_name).

Returns a summary dict so the scheduler can react to *newly* OUT players
(late scratches) by triggering an immediate scan instead of waiting for the
next 90-min cycle.
"""

from datetime import datetime, timezone
from typing import Dict, List, Tuple

from src.utils.logging_utils import get_logger
from src.clients.injuries import InjuryClient, _STATUS_RANK
from src.data.db import DatabaseClient

logger = get_logger(__name__)


def _bdl_injuries() -> List[Dict[str, str]]:
    """Pull BDL injuries via the bridge. Returns [] on failure (BDL may be down/quota'd)."""
    try:
        from src.clients.bdl_bridge import BDLBridge
        from src.clients.bdl_client import BDLClient
        return BDLBridge(BDLClient()).get_injuries_for_date()
    except Exception as e:
        logger.warning(f"BDL injuries unavailable: {e}")
        return []


def _merge(*sources: Tuple[str, List[Dict[str, str]]]) -> Dict[str, Dict]:
    """
    Merge records from multiple named sources keyed by lower-cased player name.
    Most pessimistic status wins; description is concatenated; source list tracked.
    """
    merged: Dict[str, Dict] = {}
    for source_name, records in sources:
        for r in records:
            name = (r.get('player_name') or r.get('player') or '').strip()
            if not name:
                continue
            key = name.lower()
            status = r.get('status') or 'Unknown'
            rank = _STATUS_RANK.get(status, 0)
            existing = merged.get(key)
            if existing is None:
                merged[key] = {
                    'player_name': name,
                    'team':        (r.get('team') or '').strip(),
                    'status':      status,
                    'severity':    rank,
                    'description': (r.get('description') or '').strip(),
                    'return_date': (r.get('return_date') or '').strip(),
                    'sources':     {source_name},
                }
                continue
            existing['sources'].add(source_name)
            if rank > existing['severity']:
                existing['status']   = status
                existing['severity'] = rank
            new_desc = (r.get('description') or '').strip()
            if new_desc and new_desc not in existing['description']:
                existing['description'] = (
                    f"{existing['description']}; {new_desc}".strip('; ')
                )
            new_return = (r.get('return_date') or '').strip()
            if new_return and not existing['return_date']:
                existing['return_date'] = new_return
            if not existing['team'] and r.get('team'):
                existing['team'] = r['team'].strip()
    return merged


def sync_injuries() -> Dict:
    """
    Run a full injury sync. Returns:
      {
        'records':   int,        # total players with availability data today
        'newly_out': List[str],  # players whose status moved up to Out vs prior snapshot
        'sources':   {name: count},
      }
    """
    today = datetime.now().strftime('%Y-%m-%d')
    db = DatabaseClient()

    bdl_records       = _bdl_injuries()
    multi_records     = []
    try:
        multi_records = InjuryClient().get_injuries()
    except Exception as e:
        logger.warning(f"Multi-source InjuryClient failed: {e}")

    counts = {'bdl': len(bdl_records), 'multi': len(multi_records)}
    if not bdl_records and not multi_records:
        logger.error("Injury sync: ALL sources returned empty. Skipping write to avoid wiping yesterday's snapshot.")
        return {'records': 0, 'newly_out': [], 'sources': counts}

    merged = _merge(('bdl', bdl_records), ('multi', multi_records))

    # Snapshot of prior-known status for today (so we can detect *newly* OUT players)
    prior: Dict[str, str] = {}
    with db.get_conn() as conn:
        for row in conn.execute(
            "SELECT player_name, status FROM injury_reports WHERE game_date = ?",
            (today,),
        ).fetchall():
            prior[row['player_name'].lower()] = row['status'] or 'Unknown'

    newly_out: List[str] = []
    now_iso = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with db.get_conn() as conn:
        cur = conn.cursor()
        for key, rec in merged.items():
            old_status = prior.get(key, 'Unknown')
            if rec['status'] == 'Out' and old_status != 'Out':
                newly_out.append(rec['player_name'])
            cur.execute(
                """
                INSERT OR REPLACE INTO injury_reports
                    (game_date, player_name, team, status, description,
                     return_date, severity, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    today, rec['player_name'], rec['team'], rec['status'],
                    rec['description'], rec['return_date'], rec['severity'],
                    ','.join(sorted(rec['sources'])), now_iso,
                ),
            )

    out_count = sum(1 for r in merged.values() if r['status'] == 'Out')
    logger.info(
        f"Injury sync: wrote {len(merged)} rows for {today} "
        f"(bdl={counts['bdl']}, multi={counts['multi']}, "
        f"out={out_count}, newly_out={len(newly_out)})"
    )
    if newly_out:
        logger.warning(f"Newly OUT: {', '.join(newly_out)}")

    return {'records': len(merged), 'newly_out': newly_out, 'sources': counts}


if __name__ == "__main__":
    sync_injuries()
