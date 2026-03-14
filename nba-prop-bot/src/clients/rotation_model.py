"""
Markov Chain Coach Rotation Model.

Builds a per-team rotation template from play-by-play substitution events.
For each 2-minute game slot (Q1_0 … Q4_5), tracks which players are on the
floor using exact substitution timestamps — not flat averages.

When a player is ruled OUT, the model:
  1. Identifies every slot they typically occupy (slot_probability >= 0.4).
  2. Finds the "next man up" for each slot — the non-absent player with the
     highest historical probability in that slot.
  3. Adds those absorbed slot-minutes to the target player's base projection.

This gives per-coach, per-slot minute projections rather than a blanket
"55% of starter minutes" heuristic.
"""
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

# ── Slot geometry ─────────────────────────────────────────────────────────────
SLOT_DURATION_MIN  = 2            # 2-minute resolution
QUARTER_DURATION_S = 12 * 60      # 720 seconds per quarter
SLOTS_PER_QUARTER  = QUARTER_DURATION_S // (SLOT_DURATION_MIN * 60)  # 6
N_QUARTERS         = 4
TOTAL_SLOTS        = N_QUARTERS * SLOTS_PER_QUARTER  # 24


def _slot_key(absolute_slot: int) -> str:
    """Convert linear slot index (0-23) to human-readable key like 'Q2_3'."""
    q    = (absolute_slot // SLOTS_PER_QUARTER) + 1
    s    = absolute_slot % SLOTS_PER_QUARTER
    return f"Q{q}_{s}"


def _pct_to_game_seconds(pct_string: str, period: int) -> int:
    """Convert PCTIMESTRING 'MM:SS' remaining + period → absolute game seconds elapsed."""
    try:
        m, s = pct_string.strip().split(':')
        remaining  = int(m) * 60 + int(s)
        period_start = (min(period, 4) - 1) * QUARTER_DURATION_S
        return period_start + (QUARTER_DURATION_S - remaining)
    except Exception:
        return (min(period, 4) - 1) * QUARTER_DURATION_S


def _int_id(val) -> int:
    """Safely convert a possibly-float player ID to int. Returns 0 on failure."""
    try:
        v = int(float(val))
        return v if v > 0 else 0
    except Exception:
        return 0


class RotationModel:
    """
    Builds and queries a coach rotation model from PlayByPlayV2 data.

    Usage:
        model = RotationModel(nba_stats_client)
        proj_mins = model.get_projected_minutes(
            target_player_id=203076,
            absent_player_ids=[203507],
            logs=target_player_logs_df,
            season="2024-25",
            db=db_client,
        )
    """

    GAMES_BACK      = 15
    MIN_GAMES       = 5   # need at least 5 PBP games to trust the model
    NEXT_MAN_THRESHOLD = 0.40  # slot_prob >= 40% = "this is their typical slot"

    def __init__(self, nba_client):
        self._client = nba_client

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_projected_minutes(
        self,
        target_player_id: int,
        absent_player_ids: List[int],
        logs: pd.DataFrame,
        season: str,
        db,
    ) -> float:
        """
        Return slot-based projected minutes for target_player when absent players
        are OUT. Returns 0.0 if insufficient PBP data (caller should fall back).
        """
        if not absent_player_ids or logs.empty:
            return 0.0

        team_abbr  = str(logs.iloc[0].get('TEAM_ABBREVIATION', ''))
        game_ids   = [str(g) for g in logs['GAME_ID'].head(self.GAMES_BACK).tolist()
                      if str(g).strip()]
        if not game_ids or not team_abbr:
            return 0.0

        team_slots = self._get_team_slots(team_abbr, game_ids, season, db)
        if not team_slots:
            return 0.0

        return self._compute_projection(target_player_id, absent_player_ids, team_slots)

    # ── Core computation ───────────────────────────────────────────────────────

    def _compute_projection(
        self,
        target_id: int,
        absent_ids: List[int],
        team_slots: Dict[int, Dict[str, float]],
    ) -> float:
        absent_set    = set(absent_ids)
        target_slots  = team_slots.get(target_id, {})

        if not target_slots:
            return 0.0

        # Base minutes from own typical rotation slots
        base_mins = sum(target_slots.values()) * SLOT_DURATION_MIN

        # Absorbed minutes from each absent player's vacated slots
        bonus_mins = 0.0
        for absent_id in absent_ids:
            absent_slots = team_slots.get(absent_id, {})
            for slot_key, absent_prob in absent_slots.items():
                if absent_prob < self.NEXT_MAN_THRESHOLD:
                    continue
                if not self._is_next_man_up(target_id, slot_key, team_slots, absent_set):
                    continue
                own_prob    = target_slots.get(slot_key, 0.0)
                # Credit the delta so we don't double-count slots the target
                # already plays even when the absent player is healthy.
                bonus_mins += max(0.0, absent_prob - own_prob) * SLOT_DURATION_MIN

        total = base_mins + bonus_mins
        return total if total > 1.0 else 0.0   # ignore noise < 1 minute

    def _is_next_man_up(
        self,
        player_id: int,
        slot_key: str,
        team_slots: Dict[int, Dict[str, float]],
        absent_set: Set[int],
    ) -> bool:
        """
        True if player_id has the highest slot probability for this slot
        among all players not in absent_set.
        """
        my_prob = team_slots.get(player_id, {}).get(slot_key, 0.0)
        for pid, slots in team_slots.items():
            if pid in absent_set or pid == player_id:
                continue
            if slots.get(slot_key, 0.0) > my_prob:
                return False
        return True

    # ── Caching ────────────────────────────────────────────────────────────────

    def _get_team_slots(
        self,
        team_abbr: str,
        game_ids: List[str],
        season: str,
        db,
    ) -> Dict[int, Dict[str, float]]:
        """Return cached team rotation or rebuild from PBP."""
        cached = db.get_rotation_slots(team_abbr, season)
        if cached:
            return cached

        team_slots = self._build_from_pbp(game_ids)
        if team_slots:
            db.upsert_rotation_slots(
                team_abbr, team_slots,
                min(self.GAMES_BACK, len(game_ids)),
                season,
                datetime.now().strftime('%Y-%m-%d'),
            )
        return team_slots

    # ── PBP Processing ─────────────────────────────────────────────────────────

    def _build_from_pbp(self, game_ids: List[str]) -> Dict[int, Dict[str, float]]:
        """
        Aggregate slot presence counts across all games, then normalise to
        probabilities. Returns {} if fewer than MIN_GAMES could be parsed.
        """
        # player_id → {slot_key → game-presence count}
        player_slot_counts: Dict[int, Dict[str, int]] = {}
        games_ok = 0

        for game_id in game_ids[:self.GAMES_BACK]:
            try:
                pbp = self._fetch_pbp(game_id)
                if pbp is None or pbp.empty:
                    continue
                game_slots = self._parse_game_slots(pbp)
                for pid, slots in game_slots.items():
                    if pid not in player_slot_counts:
                        player_slot_counts[pid] = {}
                    for slot in slots:
                        player_slot_counts[pid][slot] = \
                            player_slot_counts[pid].get(slot, 0) + 1
                games_ok += 1
            except Exception as exc:
                logger.debug(f"Rotation PBP parse failed for {game_id}: {exc}")

        if games_ok < self.MIN_GAMES:
            return {}

        return {
            pid: {slot: count / games_ok for slot, count in slots.items()}
            for pid, slots in player_slot_counts.items()
        }

    def _parse_game_slots(self, pbp: pd.DataFrame) -> Dict[int, Set[str]]:
        """
        Use substitution events to build an exact on-floor timeline.

        Strategy:
          • Non-sub events reveal starters (player appears before first sub
            that involves them → must have been on floor from quarter start).
          • Substitution events (EVENTMSGTYPE=8) transition the state:
              PLAYER2_ID exits → record their stint → pop from on_floor.
              PLAYER1_ID enters → add to on_floor with current time.
          • At game end (4 × 720 s) flush all remaining on-floor players.

        Returns {player_id: set_of_slot_keys}.
        """
        on_floor: Dict[int, int] = {}   # player_id → game-second of entry
        player_slots: Dict[int, Set[str]] = {}

        for _, row in pbp.iterrows():
            period = int(row.get('PERIOD', 1) or 1)
            if period > N_QUARTERS:          # skip overtime
                continue

            pct        = str(row.get('PCTIMESTRING', '12:00') or '12:00')
            now_secs   = _pct_to_game_seconds(pct, period)
            event_type = int(row.get('EVENTMSGTYPE', 0) or 0)

            # Non-sub events: if player not tracked yet, assume starter for this Q
            if event_type != 8:
                for col in ('PLAYER1_ID', 'PLAYER2_ID'):
                    pid = _int_id(row.get(col, 0))
                    if pid and pid not in on_floor:
                        # They were already on the floor from the start of this quarter
                        on_floor[pid] = (period - 1) * QUARTER_DURATION_S

            # Substitution event
            elif event_type == 8:
                player_out = _int_id(row.get('PLAYER2_ID', 0))
                player_in  = _int_id(row.get('PLAYER1_ID', 0))

                if player_out and player_out in on_floor:
                    entry = on_floor.pop(player_out)
                    self._record_slots(player_slots, player_out, entry, now_secs)

                if player_in:
                    on_floor[player_in] = now_secs

        # Flush players still on floor at game end
        game_end = N_QUARTERS * QUARTER_DURATION_S
        for pid, entry in on_floor.items():
            self._record_slots(player_slots, pid, entry, game_end)

        return player_slots

    def _record_slots(
        self,
        player_slots: Dict[int, Set[str]],
        player_id: int,
        start_secs: int,
        end_secs: int,
    ) -> None:
        """Mark every 2-minute slot whose midpoint falls within [start, end)."""
        if player_id not in player_slots:
            player_slots[player_id] = set()
        slot_dur = SLOT_DURATION_MIN * 60
        for abs_slot in range(TOTAL_SLOTS):
            slot_start = abs_slot * slot_dur
            slot_mid   = slot_start + slot_dur // 2
            if start_secs <= slot_mid < end_secs:
                player_slots[player_id].add(_slot_key(abs_slot))

    def _fetch_pbp(self, game_id: str) -> Optional[pd.DataFrame]:
        """Fetch PlayByPlayV2 for a game. No in-model caching — relies on daily cadence."""
        try:
            from nba_api.stats.endpoints import playbyplayv2
            pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
            time.sleep(0.6)
            df = pbp.get_data_frames()[0]
            return df if not df.empty else None
        except Exception as exc:
            logger.warning(f"PBP fetch failed for game {game_id}: {exc}")
            return None
