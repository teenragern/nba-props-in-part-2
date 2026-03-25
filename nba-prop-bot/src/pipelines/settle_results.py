"""
BDL-powered settlement engine.

Runs daily (suggested 8:00 AM). Grades both player props and game markets
(Moneyline / Spread / Total) against BDL final box scores, detects pushes,
and sends a daily Telegram P&L summary.

Usage:
    python -m src.pipelines.settle_results              # grade yesterday
    python -m src.pipelines.settle_results 2026-03-20   # backfill a date
"""

import re
import unicodedata
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from src.data.db import DatabaseClient
from src.clients.bdl_client import BDLClient
from src.clients.telegram_bot import TelegramBotClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Market → BDL stat field ───────────────────────────────────────────────────

_MARKET_TO_STAT: Dict[str, str] = {
    'player_points':                  'pts',
    'player_rebounds':                'reb',
    'player_assists':                 'ast',
    'player_threes':                  'fg3m',
    'player_points_rebounds_assists': 'pra',   # computed: pts + reb + ast
    'player_blocks':                  'blk',
    'player_steals':                  'stl',
}

_GAME_MARKETS = {'h2h', 'spreads', 'totals'}

# ── Name normalisation ────────────────────────────────────────────────────────


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents and extra whitespace for fuzzy matching."""
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_str = nfkd.encode('ascii', 'ignore').decode('ascii')
    return ' '.join(ascii_str.lower().split())


# ── Grading helpers ───────────────────────────────────────────────────────────


def _grade_prop(
    market: str,
    line: float,
    side: str,
    stat_row: dict,
) -> Tuple[Optional[int], float]:
    """
    Grade a player prop.

    Returns (won, actual_value) where won is:
      1    — winning bet
      0    — losing bet
      None — push (actual == line)
    """
    stat_key = _MARKET_TO_STAT.get(market)
    if not stat_key:
        return None, 0.0

    if stat_key == 'pra':
        actual = (
            float(stat_row.get('pts', 0) or 0) +
            float(stat_row.get('reb', 0) or 0) +
            float(stat_row.get('ast', 0) or 0)
        )
    else:
        actual = float(stat_row.get(stat_key, 0) or 0)

    s = side.upper()
    if actual == line:
        return None, actual
    if s == 'OVER':
        return (1 if actual > line else 0), actual
    if s == 'UNDER':
        return (1 if actual < line else 0), actual
    return None, actual


def _grade_h2h(
    side: str,
    home_full_name: str,
    home_score: float,
    visitor_score: float,
) -> Tuple[int, float]:
    """
    Grade a moneyline. `side` is the team name we bet on.
    NBA has no ties (OT decides), so push is impossible.

    Returns (won: 0/1, bet_team_margin).
    """
    home_lower = home_full_name.lower()
    side_lower = side.lower()

    if side_lower in home_lower or home_lower in side_lower:
        return (1 if home_score > visitor_score else 0), home_score - visitor_score
    else:
        return (1 if visitor_score > home_score else 0), visitor_score - home_score


def _grade_spread(
    side: str,
    home_full_name: str,
    home_score: float,
    visitor_score: float,
) -> Tuple[Optional[int], float]:
    """
    Grade a point spread.

    `side` format: "Team Name +/-spread"  (e.g., "Denver Nuggets -6.5").
    The spread value is the handicap given to that team; positive = underdog.

    Returns (won, covered_by) where won=None is a push (covered_by == 0).
    """
    m = re.match(r'^(.+?)\s+([+-]?\d+(?:\.\d+)?)\s*$', side.strip())
    if not m:
        logger.warning(f"Cannot parse spread side: {side!r}")
        return None, 0.0

    team_str = m.group(1).strip()
    spread   = float(m.group(2))

    home_lower = home_full_name.lower()
    team_lower = team_str.lower()

    if team_lower in home_lower or home_lower in team_lower:
        actual_margin = home_score - visitor_score
    else:
        actual_margin = visitor_score - home_score

    covered_by = actual_margin + spread  # > 0 → cover; == 0 → push; < 0 → loss
    if covered_by == 0.0:
        return None, 0.0
    return (1 if covered_by > 0 else 0), covered_by


def _grade_total(
    side: str,
    line: float,
    home_score: float,
    visitor_score: float,
) -> Tuple[Optional[int], float]:
    """
    Grade a game total. `side` is 'Over' or 'Under'.
    Returns (won, actual_total) where won=None is a push.
    """
    actual = home_score + visitor_score
    if actual == line:
        return None, actual
    s = side.lower()
    if s == 'over':
        return (1 if actual > line else 0), actual
    if s == 'under':
        return (1 if actual < line else 0), actual
    return None, actual


# ── Index builders ────────────────────────────────────────────────────────────


def _build_stat_index(stats: List[dict]) -> Dict[str, dict]:
    """
    Build {normalized_player_name: stat_row} from BDL game stats.
    Last entry wins when a player appears multiple times (shouldn't happen
    when filtered to a single date, but defensive).
    """
    index: Dict[str, dict] = {}
    for s in stats:
        p = s.get('player', {})
        if not p:
            continue
        full = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        index[_normalize_name(full)] = s
    return index


def _build_game_index(games: List[dict]) -> Dict[str, dict]:
    """
    Build {"visitor @ home" (normalized): game_dict} from BDL games.
    """
    index: Dict[str, dict] = {}
    for g in games:
        home = g.get('home_team', {}) or {}
        away = g.get('visitor_team', {}) or {}
        home_name = home.get('full_name', '')
        away_name  = away.get('full_name', '')
        if not home_name or not away_name:
            continue
        key = f"{_normalize_name(away_name)} @ {_normalize_name(home_name)}"
        index[key] = g
    return index


# ── Main entry point ──────────────────────────────────────────────────────────


def settle_alerts(target_date: Optional[str] = None) -> None:
    """
    Grade all unsettled alerts for `target_date` (default: yesterday).

    Prop stats and game scores are fetched in two batched BDL calls.
    Results are written to bet_results; a Telegram summary is sent at the end.
    """
    db  = DatabaseClient()
    bdl = BDLClient()
    bot = TelegramBotClient()

    if target_date is None:
        target_date = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    logger.info(f"Settlement run — target date: {target_date}")

    # ── 1. Pull unsettled alerts for this date ────────────────────────────
    with db.get_conn() as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.player_name, a.market, a.line, a.side,
                   a.odds, a.stake,
                   COALESCE(a.game_date, DATE(a.timestamp)) AS game_date,
                   a.event_id
            FROM   alerts_sent a
            LEFT   JOIN bet_results b ON a.id = b.alert_id
            WHERE  b.alert_id IS NULL
              AND  COALESCE(a.game_date, DATE(a.timestamp)) = ?
            ORDER  BY a.id
            """,
            (target_date,),
        ).fetchall()

    if not rows:
        logger.info(f"No unsettled bets for {target_date} — nothing to do.")
        return

    logger.info(f"Grading {len(rows)} alert(s) for {target_date}")

    # ── 2. Batch-fetch BDL data (2 API calls total) ───────────────────────
    raw_games = bdl.get_games_by_date(target_date)
    raw_stats = bdl.get_game_stats(dates=[target_date])

    # Only grade against Final games
    final_ids = {
        str(g.get('id'))
        for g in raw_games
        if 'final' in str(g.get('status', '')).lower()
    }
    final_games = [g for g in raw_games if str(g.get('id')) in final_ids]
    final_stats = [
        s for s in raw_stats
        if str((s.get('game') or {}).get('id', '')) in final_ids
    ]

    stat_index = _build_stat_index(final_stats)
    game_index = _build_game_index(final_games)

    logger.info(
        f"BDL: {len(final_games)} final game(s), "
        f"{len(final_stats)} player stat line(s) available"
    )

    # ── 3. Grade each alert ───────────────────────────────────────────────
    wins = losses = pushes = skipped = 0
    pnl = 0.0
    detail_lines: List[str] = []

    for row in rows:
        alert_id    = row['id']
        player_name = row['player_name']
        market      = row['market']
        line        = float(row['line'])
        side        = row['side']
        odds        = float(row['odds'] or 2.0)
        stake       = float(row['stake'] or 0.0)

        try:
            won_val: Optional[int] = None
            actual: float = 0.0

            if market in _GAME_MARKETS:
                # Game market — player_name holds "Away @ Home" matchup
                game = game_index.get(_normalize_name(player_name))
                if not game:
                    logger.warning(f"Game not found in index for: {player_name!r}")
                    skipped += 1
                    continue

                home_score = float(game.get('home_team_score') or 0)
                away_score = float(game.get('visitor_team_score') or 0)
                home_name  = (game.get('home_team') or {}).get('full_name', '')

                if market == 'h2h':
                    won_val, actual = _grade_h2h(side, home_name, home_score, away_score)
                elif market == 'spreads':
                    won_val, actual = _grade_spread(side, home_name, home_score, away_score)
                else:  # totals
                    won_val, actual = _grade_total(side, line, home_score, away_score)

            else:
                # Player prop — player_name is the full player name
                norm = _normalize_name(player_name)
                stat_row = stat_index.get(norm)

                # Last-name fallback (handles "K. Caldwell-Pope" vs "Kentavious Caldwell-Pope")
                if stat_row is None:
                    last = norm.split()[-1] if norm else ''
                    stat_row = next(
                        (v for k, v in stat_index.items() if k.split()[-1] == last),
                        None,
                    )

                if stat_row is None:
                    logger.warning(f"No BDL stat line for: {player_name!r}")
                    skipped += 1
                    continue

                won_val, actual = _grade_prop(market, line, side, stat_row)

            # ── Write to bet_results ──────────────────────────────────────
            is_push = (won_val is None)
            won_int = 0 if is_push else int(won_val)

            with db.get_conn() as wconn:
                wconn.execute(
                    """
                    INSERT INTO bet_results (alert_id, actual_result, won, push)
                    VALUES (?, ?, ?, ?)
                    """,
                    (alert_id, actual, won_int, int(is_push)),
                )

            # Accumulate P&L
            if is_push:
                pushes   += 1
                pnl_delta = 0.0
            elif won_int:
                wins     += 1
                pnl_delta = stake * (odds - 1.0)
            else:
                losses   += 1
                pnl_delta = -stake
            pnl += pnl_delta

            icon = '✅' if (won_int and not is_push) else ('⚠️' if is_push else '❌')
            detail_lines.append(
                f"{icon} {player_name} | {side} {line} "
                f"{market.replace('_', ' ').title()} | "
                f"Actual: {actual:.1f} | ${pnl_delta:+.0f}"
            )
            logger.info(
                f"#{alert_id} settled — {player_name} {side} {line} {market}: "
                f"actual={actual:.1f}  won={won_int}  push={is_push}"
            )

        except Exception as exc:
            logger.error(f"Error grading alert #{alert_id} ({player_name}): {exc}")
            skipped += 1
            continue

    # ── 4. Daily Telegram summary ─────────────────────────────────────────
    total_graded = wins + losses + pushes
    if total_graded == 0:
        logger.info("No bets successfully graded — skipping Telegram summary.")
        return

    win_rate  = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    pnl_sign  = '+' if pnl >= 0 else ''

    header = (
        f"<b>📊 Daily Settlement — {target_date}</b>\n\n"
        f"Record: {wins}W – {losses}L – {pushes}P  "
        f"(Win Rate: {win_rate:.1%})\n"
        f"P&L: {pnl_sign}${pnl:.0f}"
        + (f"  |  Skipped: {skipped}" if skipped else '')
        + "\n\n"
    )

    # Telegram message cap: ~4096 chars; truncate detail at 30 lines
    visible = detail_lines[:30]
    overflow = len(detail_lines) - len(visible)
    detail = "\n".join(visible)
    if overflow > 0:
        detail += f"\n… and {overflow} more"

    bot.send_message(header + detail)
    logger.info(
        f"Settlement complete — {wins}W/{losses}L/{pushes}P  "
        f"P&L={pnl_sign}${pnl:.0f}  skipped={skipped}"
    )


if __name__ == "__main__":
    import sys
    _date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    settle_alerts(target_date=_date_arg)
