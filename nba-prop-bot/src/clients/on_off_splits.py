"""
On/Off Court Splits Client — play-by-play based usage multipliers.

Replaces the flat +15% usage bump with exact per-minute rate ratios derived
from real play-by-play data. For player X when teammate Y is injured/out:

    multiplier = per_min_rate_without_Y / per_min_rate_with_Y

A multiplier of 1.30 means player X produces 30% more per minute without Y.
A multiplier of 0.85 means player X produces 15% less without Y.
Falls back to 1.15 when insufficient data exists (< MIN_MINUTES of "without" time).

Data is cached in SQLite and refreshed every CACHE_TTL_H hours.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

GAMES_BACK  = 20     # look back this many shared games
MIN_MINUTES = 10.0   # minimum "without" minutes for data to be trusted
FALLBACK    = 1.05   # multiplier when data is sparse
CACHE_TTL_H = 20     # hours before cached result is considered stale
CLAMP_LO    = 0.70   # minimum plausible multiplier
CLAMP_HI    = 2.00   # maximum plausible multiplier

MARKETS = [
    'player_points',
    'player_rebounds',
    'player_assists',
    'player_threes',
    'player_points_rebounds_assists',
]

# nba_api season string — mirrors NbaStatsClient convention
def _current_season() -> str:
    now = datetime.now()
    y = now.year
    if now.month >= 10:
        return f"{y}-{str(y + 1)[2:]}"
    return f"{y - 1}-{str(y)[2:]}"


# ── Time helpers ─────────────────────────────────────────────────────────────

def _pct_to_seconds(pctimestring: str, period: int) -> int:
    """Convert 'MM:SS' remaining + period number to absolute game-seconds elapsed."""
    try:
        parts = pctimestring.strip().split(':')
        mm, ss = int(parts[0]), int(parts[1])
    except Exception:
        return 0
    period_len = 720 if period <= 4 else 300   # regulation = 12 min, OT = 5 min
    remaining  = mm * 60 + ss
    elapsed    = period_len - remaining
    # Sum all prior periods
    prior_secs = sum(720 if p <= 4 else 300 for p in range(1, period))
    return prior_secs + elapsed


# ── Core client ──────────────────────────────────────────────────────────────

class OnOffSplitsClient:
    """Compute and cache per-minute stat rates for on/off court contexts."""

    def get_usage_multiplier(
        self,
        target_player_id: int,
        absent_player_id: int,
        market: str,
        db,
    ) -> float:
        """
        Return the per-minute rate ratio (without / with) for target_player
        when absent_player is not on the floor.

        Args:
            target_player_id:  NBA Stats player ID of the player we are projecting.
            absent_player_id:  NBA Stats player ID of the injured / out teammate.
            market:            One of the PROP_MARKETS strings.
            db:                DatabaseClient instance.

        Returns:
            Float multiplier clamped to [CLAMP_LO, CLAMP_HI].
            Returns FALLBACK (1.05) if data is insufficient.
        """
        season = _current_season()

        # Check cache
        row = db.get_on_off_split(target_player_id, absent_player_id, market, season)
        if row and self._cache_fresh(row['last_updated']):
            return self._row_to_multiplier(row)

        # Build / refresh
        logger.info(
            f"Computing on/off splits: player={target_player_id} "
            f"absent={absent_player_id} season={season}"
        )
        try:
            self._build(target_player_id, absent_player_id, season, db)
        except Exception as e:
            logger.warning(f"on_off_splits build failed: {e}")
            return FALLBACK

        row = db.get_on_off_split(target_player_id, absent_player_id, market, season)
        if row:
            return self._row_to_multiplier(row)
        return FALLBACK

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _cache_fresh(last_updated_str: str) -> bool:
        try:
            updated = datetime.strptime(last_updated_str, '%Y-%m-%d')
            return datetime.now() - updated < timedelta(hours=CACHE_TTL_H)
        except Exception:
            return False

    @staticmethod
    def _row_to_multiplier(row: dict) -> float:
        if row.get('minutes_without', 0) < MIN_MINUTES:
            return FALLBACK
        m = row.get('usage_multiplier')
        if m is None:
            return FALLBACK
        return float(max(CLAMP_LO, min(CLAMP_HI, m)))

    def _build(self, target_id: int, absent_id: int, season: str, db) -> None:
        """Fetch shared games, parse PBP, aggregate stats, write to DB."""
        game_ids = self._get_shared_games(target_id, absent_id, season)
        if not game_ids:
            logger.warning(f"No shared games found for {target_id}/{absent_id}")
            return

        # Accumulate across games: {market: [stats_with, min_with, stats_without, min_without]}
        totals: Dict[str, List[float]] = {m: [0.0, 0.0, 0.0, 0.0] for m in MARKETS}
        games_processed = 0

        for gid in game_ids[:GAMES_BACK]:
            try:
                game_result = self._process_game(gid, target_id, absent_id)
                for mkt, vals in game_result.items():
                    # vals = (stats_with, min_with, stats_without, min_without)
                    for i, v in enumerate(vals):
                        totals[mkt][i] += v
                games_processed += 1
                time.sleep(0.6)   # respect nba_api rate limit
            except Exception as e:
                logger.debug(f"Skipping game {gid}: {e}")
                continue

        if games_processed == 0:
            return

        for mkt in MARKETS:
            sw, mw, swo, mwo = totals[mkt]
            rate_with    = (sw  / mw)  if mw  > 0 else 0.0
            rate_without = (swo / mwo) if mwo > 0 else 0.0
            if rate_with > 0:
                multiplier = rate_without / rate_with
                multiplier = max(CLAMP_LO, min(CLAMP_HI, multiplier))
            else:
                multiplier = FALLBACK

            db.upsert_on_off_split(
                player_id=target_id,
                absent_player_id=absent_id,
                market=mkt,
                season=season,
                games_processed=games_processed,
                minutes_with=mw,
                minutes_without=mwo,
                rate_with=rate_with,
                rate_without=rate_without,
                usage_multiplier=multiplier,
            )
            logger.debug(
                f"on_off {target_id} w/o {absent_id} {mkt}: "
                f"{rate_with:.3f}→{rate_without:.3f} × {multiplier:.3f} "
                f"({mwo:.0f} min without)"
            )

    def _get_shared_games(
        self, player_a: int, player_b: int, season: str
    ) -> List[str]:
        """Return game IDs (sorted descending by date) where both players appeared."""
        from nba_api.stats.endpoints import leaguegamelog

        def fetch_game_ids(pid: int) -> set:
            time.sleep(0.6)
            log = leaguegamelog.LeagueGameLog(
                player_id_nullable=pid,
                season=season,
                season_type_all_star='Regular Season',
            )
            df = log.get_data_frames()[0]
            return set(df['GAME_ID'].astype(str).tolist()) if not df.empty else set()

        ids_a = fetch_game_ids(player_a)
        ids_b = fetch_game_ids(player_b)
        shared = sorted(ids_a & ids_b, reverse=True)   # newest first
        logger.debug(f"Shared games for {player_a}/{player_b}: {len(shared)}")
        return shared

    def _process_game(
        self, game_id: str, target_id: int, absent_id: int
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Parse play-by-play for one game.

        Returns for each market:
            (stats_with, minutes_with, stats_without, minutes_without)
        where "with" = absent_id was on the floor,
              "without" = absent_id was NOT on the floor.
        """
        from nba_api.stats.endpoints import playbyplayv2

        time.sleep(0.6)
        pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
        df  = pbp.get_data_frames()[0]

        if df.empty:
            return {m: (0.0, 0.0, 0.0, 0.0) for m in MARKETS}

        # Normalise column names
        df.columns = [c.upper() for c in df.columns]

        current_lineup: set = set()
        last_second:   int  = 0
        # Per-market accumulators: [with_stat, with_min, without_stat, without_min]
        acc: Dict[str, List[float]] = {m: [0.0, 0.0, 0.0, 0.0] for m in MARKETS}

        # ── Infer starting lineup from first events before any sub ──────
        pre_sub_players: set = set()
        for _, row in df.iterrows():
            etype = int(row.get('EVENTMSGTYPE', 0) or 0)
            if etype == 8:   # first substitution → stop collecting
                break
            for col in ('PLAYER1_ID', 'PLAYER2_ID'):
                pid = row.get(col)
                if pid and int(pid) > 0:
                    pre_sub_players.add(int(pid))

        current_lineup = set(pre_sub_players)

        # ── Walk events ──────────────────────────────────────────────────
        for _, row in df.iterrows():
            etype  = int(row.get('EVENTMSGTYPE', 0) or 0)
            period = int(row.get('PERIOD', 1)        or 1)
            pct    = str(row.get('PCTIMESTRING', '0:00') or '0:00')
            cur_s  = _pct_to_seconds(pct, period)

            p1 = int(row.get('PLAYER1_ID', 0) or 0)
            p2 = int(row.get('PLAYER2_ID', 0) or 0)

            # Accumulate minutes for the segment just ended
            if cur_s > last_second:
                seg_min = (cur_s - last_second) / 60.0
                if target_id in current_lineup:
                    if absent_id in current_lineup:
                        for m in MARKETS:
                            acc[m][1] += seg_min   # minutes_with
                    else:
                        for m in MARKETS:
                            acc[m][3] += seg_min   # minutes_without
            last_second = cur_s

            # ── Substitution ───────────────────────────────────────
            if etype == 8:
                # PLAYER1 enters, PLAYER2 leaves
                if p2 > 0:
                    current_lineup.discard(p2)
                if p1 > 0:
                    current_lineup.add(p1)
                continue

            # ── Stat events (only credit target_id) ────────────────
            if target_id not in current_lineup:
                continue

            absent_on = absent_id in current_lineup
            idx_offset = 0 if absent_on else 2   # stats_with vs stats_without

            # Made field goal (EVENTMSGTYPE == 1)
            if etype == 1:
                desc = str(row.get('HOMEDESCRIPTION') or '') + \
                       str(row.get('VISITORDESCRIPTION') or '')
                is_three = '3PT' in desc.upper()

                if p1 == target_id:
                    acc['player_points'][idx_offset]                    += 3 if is_three else 2
                    acc['player_points_rebounds_assists'][idx_offset]   += 3 if is_three else 2
                    if is_three:
                        acc['player_threes'][idx_offset] += 1

                if p2 == target_id:   # assist
                    acc['player_assists'][idx_offset]                   += 1
                    acc['player_points_rebounds_assists'][idx_offset]   += 1

            # Rebound (EVENTMSGTYPE == 4)
            elif etype == 4 and p1 == target_id:
                acc['player_rebounds'][idx_offset]                      += 1
                acc['player_points_rebounds_assists'][idx_offset]       += 1

        return {
            m: (acc[m][0], acc[m][1], acc[m][2], acc[m][3])
            for m in MARKETS
        }
