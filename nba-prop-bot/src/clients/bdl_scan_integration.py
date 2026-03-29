"""
BDL playtype × opponent boost integration.

Provides init_bdl_boost() and get_bdl_boost() for scan_props.py.

The boost is a multiplicative adjustment (default 1.0) computed from:
  1. Player's dominant playtype frequencies (pnr_bh, iso, spotup, transition)
     from their season profile.
  2. Opponent's defensive quality relative to league average, proxied by
     the opponent's points-allowed from BDL team season averages.

Boost range is clamped to [0.93, 1.07] to avoid over-correction.
"""

from typing import Dict, Optional
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_BOOST_CLAMP = 0.07         # max ±7% adjustment
_MIN_FREQ_TOTAL = 0.15      # player must have ≥15% total playtype signal

# Per-market weight over each playtype dimension
_MARKET_WEIGHTS: Dict[str, Dict[str, float]] = {
    'player_points': {
        'iso': 0.40, 'pnr_bh': 0.30, 'spotup': 0.20, 'transition': 0.10,
    },
    'player_assists': {
        'pnr_bh': 0.50, 'transition': 0.30, 'iso': 0.10, 'spotup': 0.10,
    },
    'player_rebounds': {
        'spotup': 0.40, 'pnr_bh': 0.20, 'iso': 0.20, 'transition': 0.20,
    },
    'player_threes': {
        'spotup': 0.60, 'transition': 0.20, 'pnr_bh': 0.10, 'iso': 0.10,
    },
    'player_points_rebounds_assists': {
        'iso': 0.30, 'pnr_bh': 0.35, 'spotup': 0.20, 'transition': 0.15,
    },
    'player_blocks': {
        'iso': 0.40, 'pnr_bh': 0.30, 'transition': 0.20, 'spotup': 0.10,
    },
    'player_steals': {
        'transition': 0.50, 'pnr_bh': 0.30, 'iso': 0.10, 'spotup': 0.10,
    },
}
_DEFAULT_WEIGHTS = {'iso': 0.30, 'pnr_bh': 0.30, 'spotup': 0.20, 'transition': 0.20}


class BDLBooster:
    """
    Holds opponent defensive context and computes per-player/market boost factors.
    Lazily creates a BDLClient on first use to avoid double-init at import time.
    """

    def __init__(self):
        self._bdl = None
        self._opp_cache: Dict[int, Dict[str, float]] = {}  # season → {key: norm_factor}

    def _ensure_client(self):
        if self._bdl is None:
            from src.clients.bdl_client import BDLClient
            self._bdl = BDLClient()

    def get_opp_def_factor(self, opponent_team: str, season: int) -> float:
        """
        Return opponent defensive quality relative to league average.
        > 1.0 → opponent allows more scoring than average (boost up).
        < 1.0 → opponent is a strong defense (boost down).
        """
        self._ensure_client()
        if season not in self._opp_cache:
            self._load_opp_def(season)
        idx = self._opp_cache.get(season, {})
        key = opponent_team.lower()
        if key in idx:
            return idx[key]
        # Partial match
        for k, v in idx.items():
            if k and (k in key or key in k):
                return v
        return 1.0

    def _load_opp_def(self, season: int):
        """Fetch team opponent averages and normalize to league average."""
        try:
            rows = self._bdl.get_team_season_averages(
                season=season, category="general", stat_type="opponent"
            )
            if not rows:
                self._opp_cache[season] = {}
                return

            pts_by_key: Dict[str, float] = {}
            for row in rows:
                team = row.get('team') or {}
                abbr = (team.get('abbreviation') or '').lower()
                full = (team.get('full_name') or '').lower()
                pts = float(row.get('pts', 0) or 0)
                if abbr:
                    pts_by_key[abbr] = pts
                if full:
                    pts_by_key[full] = pts

            if not pts_by_key:
                self._opp_cache[season] = {}
                return

            league_avg = sum(pts_by_key.values()) / len(pts_by_key)
            if league_avg <= 0:
                self._opp_cache[season] = {}
                return

            self._opp_cache[season] = {k: v / league_avg for k, v in pts_by_key.items()}
            logger.debug(
                f"BDL booster: loaded opp-def for {len(pts_by_key)} teams, "
                f"season {season}, league_avg_pts={league_avg:.1f}"
            )
        except Exception as e:
            logger.debug(f"BDL booster opp-def load failed (season={season}): {e}")
            self._opp_cache[season] = {}


def init_bdl_boost() -> BDLBooster:
    """Initialize the BDL booster. Call once at startup."""
    logger.info("BDL playtype booster initialized.")
    return BDLBooster()


def get_bdl_boost(
    booster: BDLBooster,
    bdl_player_id: int,
    opponent_team: str,
    market: str,
    season: int,
    player_profile: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute a multiplicative projection boost in [0.93, 1.07].

    Uses player's playtype frequencies (from player_profile) weighted by
    the market's playtype importance, scaled by opponent's defensive quality
    relative to league average.

    Args:
        booster:        BDLBooster from init_bdl_boost()
        bdl_player_id:  BDL player ID (unused; reserved for future profile fetch)
        opponent_team:  opponent team name or abbreviation
        market:         prop market key (e.g. 'player_points')
        season:         NBA season year (e.g. 2025)
        player_profile: dict from bdl_bridge.get_player_season_profile()

    Returns:
        float multiplier; 1.0 means no adjustment.
    """
    try:
        weights = _MARKET_WEIGHTS.get(market, _DEFAULT_WEIGHTS)
        profile = player_profile or {}

        pnr    = float(profile.get('pnr_bh_freq', 0.0))
        iso    = float(profile.get('iso_freq', 0.0))
        spotup = float(profile.get('spotup_freq', 0.0))
        trans  = float(profile.get('transition_freq', 0.0))

        playtype_total = pnr + iso + spotup + trans
        if playtype_total < _MIN_FREQ_TOTAL:
            return 1.0

        opp_def = booster.get_opp_def_factor(opponent_team, season)
        opp_dev = opp_def - 1.0  # positive = weak defense (allows more scoring)

        # Weighted alignment: how much does this player lean into the playtypes
        # most relevant for this market?
        alignment = (
            weights['pnr_bh']     * pnr +
            weights['iso']        * iso +
            weights['spotup']     * spotup +
            weights['transition'] * trans
        ) / max(playtype_total, 0.01)

        adj = alignment * opp_dev
        adj = max(-_BOOST_CLAMP, min(_BOOST_CLAMP, adj))

        if adj != 0.0:
            logger.debug(
                f"BDL boost: player={bdl_player_id} market={market} "
                f"opp={opponent_team} opp_def={opp_def:.3f} "
                f"alignment={alignment:.3f} adj={adj:+.3f}"
            )
        return 1.0 + adj

    except Exception as e:
        logger.debug(f"BDL boost error: {e}")
        return 1.0
