"""
Dynamic "On/Off" Usage Matrix — Live Feature Adjustment.

Adjusts player feature vectors (real_usage_pct, ts_pct_5g) in real-time
based on the exact 5-man rotation currently on the floor.

When a star player exits the game (foul trouble, injury, rest), the
remaining players' usage rates shift upward. This module quantifies
that shift using historical on/off split data and applies it to the
ML model's feature vector for live re-prediction.

Key concepts:
  - Usage absorption: when a high-usage player exits, his teammates
    absorb that usage proportionally to their historical splits.
  - Conservation constraint: team-wide usage must sum to ~1.0 across
    the 5 on-court players to prevent over-projection.
  - Confidence gating: adjustments are only applied when sufficient
    historical minutes-with/without data exists.

Data source:
  - on_off_splits table: (player_id, absent_player_id, season, market)
    → minutes_with, minutes_without, rate_with, rate_without, usage_multiplier

Usage:
    from src.models.live_usage_adjuster import LiveUsageAdjuster

    adjuster = LiveUsageAdjuster(db)
    await adjuster.load_team_context("BOS", starter_ids)
    adjusted = adjuster.recalculate_live_usage(
        target_player_id=123,
        players_on_court={101, 102, 103, 104, 123},
        market="player_points",
    )
    # adjusted = {"real_usage_pct": 0.32, "ts_pct_5g": 0.58, "confidence": 0.85}
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

_MULTIPLIER_FLOOR = 0.70       # Never reduce usage below 70% of base
_MULTIPLIER_CEILING = 2.00     # Never boost usage above 200% of base
_MIN_MINUTES_WITHOUT = 50.0    # Minimum minutes of split data for confidence
_HIGH_CONFIDENCE_MINUTES = 200.0  # Full confidence at this threshold
_TS_MULTIPLIER_DAMPENING = 0.5  # TS% shifts less than usage (efficiency is stickier)
_DEFAULT_MULTIPLIER = 1.0       # When no data available


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class PlayerProfile:
    """Pre-loaded player context for a single team member."""
    player_id: int
    player_name: str
    base_usage_pct: float               # Season-average real_usage_pct
    base_ts_pct: float                  # Season-average ts_pct_5g
    is_starter: bool = False
    # on_off_multipliers[absent_player_id][market] = usage_multiplier
    on_off_multipliers: Dict[int, Dict[str, float]] = field(default_factory=dict)
    # Minutes data for confidence scoring
    minutes_without: Dict[int, float] = field(default_factory=dict)


@dataclass
class UsageAdjustment:
    """Output: adjusted features for a single player."""
    player_id: int
    adjusted_usage_pct: float
    adjusted_ts_pct: float
    raw_multiplier: float           # Combined multiplier before conservation
    final_multiplier: float         # After conservation constraint
    confidence: float               # 0-1, based on data quality
    absent_players: List[int]       # Who is missing that drove the adjustment
    reason: str                     # Human-readable explanation


# ── Core Engine ──────────────────────────────────────────────────────────────

class LiveUsageAdjuster:
    """
    Computes real-time usage/efficiency adjustments based on current lineup.

    Lifecycle:
      1. load_team_context() — called once per game at tip-off
      2. recalculate_live_usage() — called on every micro_state lineup change
    """

    def __init__(self, db):
        self.db = db
        # team_name → {player_id: PlayerProfile}
        self._team_profiles: Dict[str, Dict[int, PlayerProfile]] = {}
        # team_name → set of expected starter IDs
        self._team_starters: Dict[str, Set[int]] = {}

    async def load_team_context(
        self,
        team: str,
        roster_player_ids: List[int],
        starters: Optional[Set[int]] = None,
        season: str = "2024-25",
    ) -> None:
        """
        Pre-load on_off_splits and base stats for all players on a team.
        Called once per game at tip-off.

        Args:
            team: Team name or abbreviation.
            roster_player_ids: All player IDs on the roster.
            starters: Expected starting 5 IDs (if known).
            season: Season string for splits lookup.
        """
        profiles: Dict[int, PlayerProfile] = {}

        for pid in roster_player_ids:
            profile = await self._load_player_profile(pid, team, season)
            if profile:
                if starters and pid in starters:
                    profile.is_starter = True
                profiles[pid] = profile

        self._team_profiles[team] = profiles
        self._team_starters[team] = starters or set(roster_player_ids[:5])

        logger.info(
            f"live_usage: loaded context for {team}: "
            f"{len(profiles)} players, {len(starters or set())} starters"
        )

    async def _load_player_profile(
        self, player_id: int, team: str, season: str
    ) -> Optional[PlayerProfile]:
        """Load a single player's profile from the database."""
        try:
            # Load on_off_splits for this player (as the target)
            splits = self._query_on_off_splits(player_id, season)

            # Load base usage/ts from player's recent game logs
            base_stats = self._query_base_stats(player_id)

            profile = PlayerProfile(
                player_id=player_id,
                player_name=base_stats.get("name", str(player_id)),
                base_usage_pct=base_stats.get("usage_pct", 0.20),
                base_ts_pct=base_stats.get("ts_pct", 0.55),
            )

            # Populate on_off multipliers
            for split in splits:
                absent_id = split["absent_player_id"]
                market = split["market"]
                multiplier = split.get("usage_multiplier", _DEFAULT_MULTIPLIER)
                mins_without = split.get("minutes_without", 0.0)

                if absent_id not in profile.on_off_multipliers:
                    profile.on_off_multipliers[absent_id] = {}
                profile.on_off_multipliers[absent_id][market] = multiplier
                profile.minutes_without[absent_id] = mins_without

            return profile

        except Exception as e:
            logger.warning(f"live_usage: failed to load profile for {player_id}: {e}")
            return None

    def _query_on_off_splits(self, player_id: int, season: str) -> List[dict]:
        """Query on_off_splits table for a target player."""
        try:
            with self.db.get_conn() as conn:
                cursor = conn.execute(
                    """SELECT absent_player_id, market, usage_multiplier,
                              minutes_with, minutes_without, rate_with, rate_without
                       FROM on_off_splits
                       WHERE player_id = %s AND season = %s""",
                    (player_id, season),
                )
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception:
            return []

    def _query_base_stats(self, player_id: int) -> dict:
        """Query recent base stats for a player."""
        try:
            with self.db.get_conn() as conn:
                row = conn.execute(
                    """SELECT p.target_name as name,
                              COALESCE(AVG(
                                  (g.fga + 0.44 * g.fta + g.tov)::float /
                                  NULLIF(g.min, 0)
                              ), 0.20) as usage_pct,
                              COALESCE(AVG(
                                  g.pts::float /
                                  NULLIF(2.0 * (g.fga + 0.44 * g.fta), 0)
                              ), 0.55) as ts_pct
                       FROM bdl_game_log_cache g
                       JOIN players p ON p.player_id = g.player_id
                       WHERE g.player_id = %s
                       ORDER BY g.game_date DESC
                       LIMIT 10""",
                    (player_id,),
                ).fetchone()
                if row:
                    return {"name": row[0], "usage_pct": float(row[1]), "ts_pct": float(row[2])}
        except Exception:
            pass
        return {"name": str(player_id), "usage_pct": 0.20, "ts_pct": 0.55}

    def recalculate_live_usage(
        self,
        target_player_id: int,
        players_on_court: Set[int],
        team: str,
        market: str = "player_points",
    ) -> UsageAdjustment:
        """
        Compute adjusted usage for a player given who is currently on court.

        Algorithm:
          1. Identify who is absent from the expected rotation
          2. For each absent player: lookup usage_multiplier from on_off_splits
          3. Combine multipliers multiplicatively (capped at ceiling)
          4. Apply usage conservation constraint
          5. Compute confidence score from data quality

        Args:
            target_player_id: Player whose features we're adjusting.
            players_on_court: Set of player IDs currently on the floor.
            team: Team name/abbreviation for lookup.
            market: Market type (player_points, player_rebounds, etc.)

        Returns:
            UsageAdjustment with adjusted values.
        """
        profiles = self._team_profiles.get(team, {})
        target_profile = profiles.get(target_player_id)

        if target_profile is None:
            return UsageAdjustment(
                player_id=target_player_id,
                adjusted_usage_pct=0.20,
                adjusted_ts_pct=0.55,
                raw_multiplier=1.0,
                final_multiplier=1.0,
                confidence=0.0,
                absent_players=[],
                reason="no profile data loaded",
            )

        starters = self._team_starters.get(team, set())
        # Identify absent players from the expected rotation
        absent_players = starters - players_on_court

        if not absent_players:
            # Everyone is on court: no adjustment needed
            return UsageAdjustment(
                player_id=target_player_id,
                adjusted_usage_pct=target_profile.base_usage_pct,
                adjusted_ts_pct=target_profile.base_ts_pct,
                raw_multiplier=1.0,
                final_multiplier=1.0,
                confidence=1.0,
                absent_players=[],
                reason="full rotation on court",
            )

        # Compute combined multiplier from all absent players
        combined_mult = 1.0
        confidence_scores: List[float] = []
        absent_list: List[int] = []

        for absent_id in absent_players:
            if absent_id == target_player_id:
                continue  # target player is absent from their own splits

            mult = target_profile.on_off_multipliers.get(
                absent_id, {}
            ).get(market, _DEFAULT_MULTIPLIER)

            combined_mult *= mult
            absent_list.append(absent_id)

            # Confidence from minutes of data
            mins = target_profile.minutes_without.get(absent_id, 0.0)
            if mins >= _HIGH_CONFIDENCE_MINUTES:
                confidence_scores.append(1.0)
            elif mins >= _MIN_MINUTES_WITHOUT:
                confidence_scores.append(mins / _HIGH_CONFIDENCE_MINUTES)
            else:
                confidence_scores.append(0.3)  # low confidence

        # Clamp combined multiplier
        raw_multiplier = combined_mult
        combined_mult = max(_MULTIPLIER_FLOOR, min(_MULTIPLIER_CEILING, combined_mult))

        # Apply usage conservation constraint
        # All on-court players' adjusted usages should sum to ~1.0
        final_multiplier = self._apply_conservation(
            target_player_id, combined_mult, players_on_court, team, market
        )

        # Compute adjusted features
        adjusted_usage = target_profile.base_usage_pct * final_multiplier
        # TS% adjusts with dampening (efficiency is stickier than volume)
        ts_mult = 1.0 + (final_multiplier - 1.0) * _TS_MULTIPLIER_DAMPENING
        adjusted_ts = target_profile.base_ts_pct * ts_mult

        # Aggregate confidence
        confidence = min(confidence_scores) if confidence_scores else 0.0

        # Build reason string
        if absent_list:
            absent_names = [
                profiles.get(pid, PlayerProfile(pid, str(pid), 0, 0)).player_name
                for pid in absent_list
            ]
            reason = f"absent: {', '.join(absent_names)} → mult={final_multiplier:.2f}"
        else:
            reason = "no relevant absences"

        return UsageAdjustment(
            player_id=target_player_id,
            adjusted_usage_pct=adjusted_usage,
            adjusted_ts_pct=adjusted_ts,
            raw_multiplier=raw_multiplier,
            final_multiplier=final_multiplier,
            confidence=confidence,
            absent_players=absent_list,
            reason=reason,
        )

    def _apply_conservation(
        self,
        target_id: int,
        target_mult: float,
        players_on_court: Set[int],
        team: str,
        market: str,
    ) -> float:
        """
        Apply usage conservation constraint.

        Total adjusted usage across all 5 on-court players should sum to ~1.0.
        If boosting one player pushes the total above 1.0, dampen proportionally.
        """
        profiles = self._team_profiles.get(team, {})
        starters = self._team_starters.get(team, set())
        absent = starters - players_on_court

        # Calculate total projected usage for all on-court players
        total_usage = 0.0
        target_adjusted = 0.0

        for pid in players_on_court:
            profile = profiles.get(pid)
            if profile is None:
                total_usage += 0.15  # assume bench minimum
                continue

            if pid == target_id:
                adjusted = profile.base_usage_pct * target_mult
                target_adjusted = adjusted
                total_usage += adjusted
            else:
                # Estimate this player's adjustment too
                player_mult = 1.0
                for absent_id in absent:
                    if absent_id == pid:
                        continue
                    m = profile.on_off_multipliers.get(absent_id, {}).get(market, 1.0)
                    player_mult *= m
                player_mult = max(_MULTIPLIER_FLOOR, min(_MULTIPLIER_CEILING, player_mult))
                total_usage += profile.base_usage_pct * player_mult

        # If total exceeds 1.0, normalize proportionally
        if total_usage > 1.0 and target_adjusted > 0:
            normalization_factor = 1.0 / total_usage
            return target_mult * normalization_factor

        return target_mult

    def get_adjusted_features(
        self,
        target_player_id: int,
        players_on_court: Set[int],
        team: str,
        market: str = "player_points",
    ) -> Dict[str, float]:
        """
        Convenience method: return adjusted ML features as a dict.

        Returns:
            {"real_usage_pct": float, "ts_pct_5g": float, "confidence": float}
            Empty dict if insufficient data for adjustment.
        """
        adj = self.recalculate_live_usage(
            target_player_id, players_on_court, team, market
        )
        if adj.confidence < 0.3:
            return {}  # insufficient data, don't adjust

        return {
            "real_usage_pct": adj.adjusted_usage_pct,
            "ts_pct_5g": adj.adjusted_ts_pct,
            "confidence": adj.confidence,
        }


# ── Module-Level Convenience Function ────────────────────────────────────────

def recalculate_live_usage(
    target_player_id: int,
    players_on_court: Set[int],
    team_starters: Set[int],
    on_off_data: Dict[int, Dict[str, float]],
    base_usage: float,
    base_ts: float,
    market: str = "player_points",
) -> Dict[str, float]:
    """
    Stateless convenience function for quick usage adjustment.

    Args:
        target_player_id: Player to adjust.
        players_on_court: Set of IDs currently on floor.
        team_starters: Expected rotation player IDs.
        on_off_data: {absent_player_id: {market: usage_multiplier}}
        base_usage: Player's baseline real_usage_pct.
        base_ts: Player's baseline ts_pct_5g.
        market: Market type for multiplier lookup.

    Returns:
        {"real_usage_pct": adjusted, "ts_pct_5g": adjusted}
        Empty dict if no adjustment needed.
    """
    absent = team_starters - players_on_court
    if not absent:
        return {"real_usage_pct": base_usage, "ts_pct_5g": base_ts}

    combined_mult = 1.0
    for absent_id in absent:
        if absent_id == target_player_id:
            continue
        mult = on_off_data.get(absent_id, {}).get(market, _DEFAULT_MULTIPLIER)
        combined_mult *= mult

    combined_mult = max(_MULTIPLIER_FLOOR, min(_MULTIPLIER_CEILING, combined_mult))

    adjusted_usage = base_usage * combined_mult
    ts_mult = 1.0 + (combined_mult - 1.0) * _TS_MULTIPLIER_DAMPENING
    adjusted_ts = base_ts * ts_mult

    return {
        "real_usage_pct": adjusted_usage,
        "ts_pct_5g": adjusted_ts,
    }
