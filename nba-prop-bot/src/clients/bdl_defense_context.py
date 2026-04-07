"""
BDL defense context — full opponent defensive profiles from BDL team season averages.

Pulls three BDL stat categories per team and merges them into a single
defense profile that feeds directly into projections:

  1. general/opponent  → OPP_PTS, OPP_REB, OPP_AST, OPP_FG3M, OPP_FTA
  2. general/advanced  → DEF_RATING, PACE, NET_RATING
  3. general/defense   → OPP_PTS_PAINT, BLK, STL, CONTESTED_FG_PCT

Each stat is normalised to a league-average factor (1.0 = average).
Position-specific DRTG factors are derived from the opponent's defensive
profile using per-position weights (guards give up more AST, centers face
tighter paint D, etc.).

Everything is cached in SQLite with a 12-hour TTL so a full scan cycle
costs 3 BDL requests per team at most once per day.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Position weights: how sensitive each position is to each defensive dimension.
# Values > 1.0 = position is MORE impacted; < 1.0 = less impacted.
_POSITION_DEF_WEIGHTS = {
    'Guard': {
        'opp_pts': 1.10,   # guards score more from perimeter → opp PTS allowed matters
        'opp_ast': 1.25,   # playmakers absorb assist opportunity from weak defenses
        'opp_reb': 0.60,   # guards grab few rebounds regardless
        'opp_fg3m': 1.30,  # perimeter D matters most for guards
        'opp_fta': 0.90,
        'opp_pts_paint': 0.60,
        'def_rating': 1.00,
    },
    'Forward': {
        'opp_pts': 1.00,
        'opp_ast': 0.90,
        'opp_reb': 1.10,
        'opp_fg3m': 1.00,
        'opp_fta': 1.00,
        'opp_pts_paint': 1.10,
        'def_rating': 1.00,
    },
    'Center': {
        'opp_pts': 0.90,
        'opp_ast': 0.70,
        'opp_reb': 1.40,   # centers dominate the glass → opp DREB weakness exploitable
        'opp_fg3m': 0.40,  # most centers don't shoot 3s
        'opp_fta': 1.15,   # centers draw fouls in paint
        'opp_pts_paint': 1.40,
        'def_rating': 1.05,
    },
}

# Market → which defensive dimensions matter most
_MARKET_DEF_DIMS = {
    'player_points':   ['opp_pts', 'opp_pts_paint', 'opp_fta', 'def_rating'],
    'player_rebounds':  ['opp_reb', 'def_rating'],
    'player_assists':   ['opp_ast', 'def_rating'],
    'player_threes':    ['opp_fg3m', 'def_rating'],
    'player_blocks':    ['opp_pts_paint', 'def_rating'],
    'player_steals':    ['def_rating'],
    'player_points_rebounds_assists': ['opp_pts', 'opp_reb', 'opp_ast', 'def_rating'],
}

_CACHE_TTL_HOURS = 12


class BDLDefenseContext:
    """
    Fetches and caches full opponent defensive profiles from BDL.

    Main entry point: get_opponent_profile(team_name, season, db)
    Returns a dict of league-normalised factors.

    get_position_def_factor(profile, position, market)
    Returns a single multiplier for projection adjustment.
    """

    def __init__(self, bdl_client):
        self.bdl = bdl_client
        # In-memory cache: season → {team_key: profile_dict}
        self._mem_cache: Dict[int, Dict[str, dict]] = {}

    def get_opponent_profile(
        self, team_name: str, season: int, db=None
    ) -> Dict[str, float]:
        """
        Return a normalised defensive profile for the opponent team.

        Keys returned:
            opp_pts, opp_reb, opp_ast, opp_fg3m, opp_fta,
            opp_pts_paint, def_rating, pace, blk, stl,
            contested_fg_pct
        All normalised so league average = 1.0.
        """
        # 1. Check in-memory cache
        if season in self._mem_cache:
            hit = self._find_team(self._mem_cache[season], team_name)
            if hit:
                return hit

        # 2. Check SQLite cache
        if db:
            cached = db.get_bdl_defense_profile(team_name, season)
            if cached and self._is_fresh(cached):
                # Populate mem cache
                self._mem_cache.setdefault(season, {})[team_name.lower()] = cached
                return cached

        # 3. Fetch from BDL and build profiles for ALL teams (one batch)
        profiles = self._fetch_all_profiles(season)
        self._mem_cache[season] = profiles

        # 4. Persist to SQLite
        if db:
            for team_key, profile in profiles.items():
                db.upsert_bdl_defense_profile(team_key, season, profile)

        return self._find_team(profiles, team_name) or self._default_profile()

    def get_position_def_factor(
        self, profile: Dict[str, float], position: str, market: str
    ) -> float:
        """
        Compute a single multiplicative factor for a position+market combo
        using the opponent's full defensive profile.

        > 1.0 = opponent is weak at defending this → boost projection.
        < 1.0 = opponent is strong defensively → reduce projection.

        Clamped to [0.80, 1.20] to prevent extreme adjustments.
        """
        dims = _MARKET_DEF_DIMS.get(market, ['def_rating'])
        pos_weights = _POSITION_DEF_WEIGHTS.get(position, {})

        if not dims:
            return 1.0

        weighted_sum = 0.0
        weight_total = 0.0
        for dim in dims:
            factor = profile.get(dim, 1.0)
            pos_w = pos_weights.get(dim, 1.0)
            weighted_sum += factor * pos_w
            weight_total += pos_w

        if weight_total <= 0:
            return 1.0

        raw = weighted_sum / weight_total
        return round(max(0.80, min(1.20, raw)), 4)

    def _fetch_all_profiles(self, season: int) -> Dict[str, dict]:
        """Fetch opponent, advanced, and defense stats and merge into profiles."""
        profiles: Dict[str, dict] = {}

        # Fetch all three stat categories
        opp_rows = self.bdl.get_team_season_averages(
            season=season, category="general", stat_type="opponent"
        )
        adv_rows = self.bdl.get_team_season_averages(
            season=season, category="general", stat_type="advanced"
        )
        def_rows = self.bdl.get_team_season_averages(
            season=season, category="general", stat_type="defense"
        )

        # Index by team key
        opp_by_team = self._index_rows(opp_rows)
        adv_by_team = self._index_rows(adv_rows)
        def_by_team = self._index_rows(def_rows)

        # Compute league averages for normalisation
        opp_avgs = self._compute_averages(opp_rows, [
            'pts', 'reb', 'ast', 'fg3m', 'fta',
        ])
        adv_avgs = self._compute_averages(adv_rows, [
            'def_rating', 'pace',
        ])
        def_avgs = self._compute_averages(def_rows, [
            'opp_pts_paint', 'blk', 'stl', 'def_rim_fgm',
        ])

        # Build normalised profiles
        all_keys = set(opp_by_team) | set(adv_by_team) | set(def_by_team)
        for key in all_keys:
            opp = opp_by_team.get(key, {})
            adv = adv_by_team.get(key, {})
            dfn = def_by_team.get(key, {})

            profile = {
                'opp_pts':     self._norm(opp, 'pts', opp_avgs),
                'opp_reb':     self._norm(opp, 'reb', opp_avgs),
                'opp_ast':     self._norm(opp, 'ast', opp_avgs),
                'opp_fg3m':    self._norm(opp, 'fg3m', opp_avgs),
                'opp_fta':     self._norm(opp, 'fta', opp_avgs),
                'opp_pts_paint': self._norm(dfn, 'opp_pts_paint', def_avgs),
                'def_rating':  self._norm_inv(adv, 'def_rating', adv_avgs),
                'pace':        self._norm(adv, 'pace', adv_avgs),
                'blk':         self._norm(dfn, 'blk', def_avgs),
                'stl':         self._norm(dfn, 'stl', def_avgs),
                'fetched_at':  datetime.utcnow().isoformat(),
            }
            profiles[key] = profile

        logger.info(
            f"BDL defense context: built {len(profiles)} team profiles "
            f"for season {season}"
        )
        return profiles

    @staticmethod
    def _index_rows(rows: List[dict]) -> Dict[str, dict]:
        """Index API rows by lowercase team abbreviation and full name."""
        idx: Dict[str, dict] = {}
        for row in (rows or []):
            team = row.get('team') or {}
            abbr = (team.get('abbreviation') or '').lower()
            full = (team.get('full_name') or '').lower()
            if abbr:
                idx[abbr] = row
            if full:
                idx[full] = row
        return idx

    @staticmethod
    def _compute_averages(rows: List[dict], fields: List[str]) -> Dict[str, float]:
        """Compute league-average for each field across all teams."""
        avgs: Dict[str, float] = {}
        for field in fields:
            vals = [float(r.get(field, 0) or 0) for r in (rows or []) if r.get(field)]
            avgs[field] = sum(vals) / len(vals) if vals else 1.0
        return avgs

    @staticmethod
    def _norm(row: dict, field: str, avgs: Dict[str, float]) -> float:
        """Normalise: team_val / league_avg. >1.0 = allows more than avg."""
        val = float(row.get(field, 0) or 0)
        avg = avgs.get(field, 1.0)
        if avg <= 0:
            return 1.0
        return round(val / avg, 4)

    @staticmethod
    def _norm_inv(row: dict, field: str, avgs: Dict[str, float]) -> float:
        """
        Inverse normalise for defensive rating: lower DRTG = better defense.
        We invert so that > 1.0 = opponent allows more (weaker D) — consistent
        with the other dimensions.

        Formula: league_avg / team_val  (team w/ DRTG 105 vs avg 110 → 110/105 = 1.048 → strong D → lower factor)
        Wait — we want >1.0 = allows more scoring. Low DRTG = good defense = should produce <1.0.

        So: team_DRTG / league_avg. High DRTG → >1.0 → allows more → boost.
        """
        val = float(row.get(field, 0) or 0)
        avg = avgs.get(field, 1.0)
        if avg <= 0 or val <= 0:
            return 1.0
        return round(val / avg, 4)

    @staticmethod
    def _is_fresh(profile: dict) -> bool:
        """Check if a cached profile is within the TTL window."""
        fetched = profile.get('fetched_at')
        if not fetched:
            return False
        try:
            ts = datetime.fromisoformat(fetched)
            return datetime.utcnow() - ts < timedelta(hours=_CACHE_TTL_HOURS)
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _find_team(profiles: Dict[str, dict], team_name: str) -> Optional[dict]:
        """Find a team profile by exact or partial match."""
        low = team_name.lower()
        if low in profiles:
            return profiles[low]
        for k, v in profiles.items():
            if k and (k in low or low in k):
                return v
        return None

    @staticmethod
    def _default_profile() -> Dict[str, float]:
        return {
            'opp_pts': 1.0, 'opp_reb': 1.0, 'opp_ast': 1.0,
            'opp_fg3m': 1.0, 'opp_fta': 1.0, 'opp_pts_paint': 1.0,
            'def_rating': 1.0, 'pace': 1.0, 'blk': 1.0, 'stl': 1.0,
            'fetched_at': datetime.utcnow().isoformat(),
        }
