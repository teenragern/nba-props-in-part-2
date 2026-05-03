"""
BDL Integration Bridge.

Translates BDL API responses into the data shapes that scan_props.py
already expects. This allows minimal changes to the main pipeline while
swapping the underlying data source from Odds API → BDL for props,
injuries, and lineups.

The Odds API is preserved exclusively for Pinnacle/sharp book prices.

Usage:
    from src.clients.bdl_bridge import BDLBridge

    bridge = BDLBridge(bdl_client)
    props_by_game = bridge.get_all_props_for_date("2026-03-24")
    injuries = bridge.get_injuries_for_teams(["Lakers", "Celtics"])
    starters = bridge.get_confirmed_starters(bdl_game_id)
"""

import time
from typing import Dict, List, Optional, Set, Any
from src.clients.bdl_client import BDLClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# BDL team name → abbreviation lookup (built lazily on first use)
_TEAM_LOOKUP: Dict[str, dict] = {}


class BDLBridge:
    """Bridge between BDL API and the existing scan_props pipeline."""

    _ODDS_CACHE_TTL = 86400  # 24 h — spreads/totals don't move enough to re-fetch intraday

    def __init__(self, bdl: BDLClient = None):
        self.bdl = bdl or BDLClient()
        self._player_cache: Dict[int, dict] = {}  # bdl_player_id → player info
        self._team_cache: Dict[int, dict] = {}     # bdl_team_id → team info
        self._odds_cache: Dict[str, tuple] = {}    # date -> (result, monotonic_time)
        self._profile_cache: Dict[int, dict] = {}  # bdl_player_id -> profile

    # ── Team resolution ──────────────────────────────────────────────────

    def _ensure_team_cache(self):
        """Fetch teams once and build lookup tables."""
        if self._team_cache:
            return
        teams = self.bdl.get_teams()
        for t in teams:
            tid = t.get("id")
            self._team_cache[tid] = t
            # Also index by full_name (lowercase) for matching
            full = t.get("full_name", "").lower()
            _TEAM_LOOKUP[full] = t
            # And by abbreviation
            abbr = t.get("abbreviation", "").lower()
            _TEAM_LOOKUP[abbr] = t

    def resolve_team_id(self, team_name: str) -> Optional[int]:
        """Resolve a team name/abbreviation to BDL team ID."""
        self._ensure_team_cache()
        t = _TEAM_LOOKUP.get(team_name.lower())
        if t:
            return t["id"]
        # Partial match
        low = team_name.lower()
        for key, val in _TEAM_LOOKUP.items():
            if low in key or key in low:
                return val["id"]
        return None

    # ── Games ────────────────────────────────────────────────────────────

    def get_today_games(self, date: str) -> List[dict]:
        """
        Fetch today's games from BDL.

        Returns list of dicts with keys matching what scan_props expects:
            bdl_game_id, home_team, away_team, commence_time, status,
            home_team_score, visitor_team_score
        """
        games = self.bdl.get_games_by_date(date)
        result = []
        for g in games:
            home = g.get("home_team", {})
            away = g.get("visitor_team", {})
            result.append({
                "bdl_game_id": g.get("id"),
                "home_team": home.get("full_name", ""),
                "away_team": away.get("full_name", ""),
                "home_team_abbr": home.get("abbreviation", ""),
                "away_team_abbr": away.get("abbreviation", ""),
                "home_team_id": home.get("id"),
                "away_team_id": away.get("id"),
                "commence_time": g.get("datetime", ""),
                "status": g.get("status", ""),
                "season": g.get("season"),
            })
        return result

    # ── Player Props ─────────────────────────────────────────────────────

    def get_props_for_game(
        self, bdl_game_id: int, vendors: List[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch all player props for a game and organize them for scanning.

        Returns:
        {
            "players_in_event": set of player names,
            "prices_by_market": {market: {player_name: set of lines}},
            "best_odds": {(player, market, line, side): {price, book}},
            "line_records": [(player, market, book, line, side, price, implied)],
            "player_id_map": {player_name: bdl_player_id},
        }
        """
        raw_props = self.bdl.extract_props_for_scan(bdl_game_id, vendors=vendors)

        players_in_event: Set[str] = set()
        prices_by_market: Dict[str, Dict[str, set]] = {}
        best_odds: Dict[tuple, dict] = {}
        line_records: List[tuple] = []
        player_id_map: Dict[str, int] = {}

        # We need player names — resolve from BDL player cache
        player_ids_needed = {p["bdl_player_id"] for p in raw_props}
        self._resolve_player_names(player_ids_needed)

        for prop in raw_props:
            pid = prop["bdl_player_id"]
            player_info = self._player_cache.get(pid)
            if not player_info:
                continue

            name = f"{player_info.get('first_name', '')} {player_info.get('last_name', '')}".strip()
            if not name:
                continue

            market = prop["market"]
            line = prop["line"]
            side = prop["side"]
            book = prop["book"]
            price = prop["decimal_odds"]

            players_in_event.add(name)
            player_id_map[name] = pid

            if market not in prices_by_market:
                prices_by_market[market] = {}
            prices_by_market[market].setdefault(name, set()).add(line)

            # Track best odds per (player, market, line, side)
            key = (name, market, line, side)
            if key not in best_odds or price > best_odds[key]["price"]:
                best_odds[key] = {"price": price, "book": book}

            if price > 0:
                line_records.append(
                    (name, market, book, line, side, price, 1.0 / price)
                )

        return {
            "players_in_event": players_in_event,
            "prices_by_market": prices_by_market,
            "best_odds": best_odds,
            "line_records": line_records,
            "player_id_map": player_id_map,
        }

    def _resolve_player_names(self, player_ids: Set[int]):
        """Batch-resolve BDL player IDs to names (cached)."""
        missing = player_ids - set(self._player_cache.keys())
        if not missing:
            return

        # BDL doesn't have a batch player endpoint, but active_players
        # returns all ~500 active players — fetch once and cache
        if not self._player_cache:
            active = self.bdl.get_active_players()
            for p in active:
                self._player_cache[p["id"]] = p
            logger.info(f"BDL: cached {len(self._player_cache)} active players")

        # Any still missing? Fetch individually
        still_missing = missing - set(self._player_cache.keys())
        for pid in still_missing:
            try:
                result = self.bdl._get(f"https://api.balldontlie.io/v1/players/{pid}")
                data = result.get("data")
                if data:
                    self._player_cache[pid] = data
            except Exception:
                pass

    def get_player_name(self, bdl_player_id: int) -> str:
        """Get player full name from cache."""
        p = self._player_cache.get(bdl_player_id)
        if p:
            return f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        return ""

    # ── Injuries ─────────────────────────────────────────────────────────

    def get_injuries_for_date(self) -> List[dict]:
        """
        Fetch all injuries and normalize to scan_props format:
            [{player_name, team, status, description}, ...]
        """
        raw = self.bdl.get_injuries()
        injuries = []
        for inj in raw:
            player = inj.get("player", {})
            team = player.get("team", {}) if isinstance(player.get("team"), dict) else {}
            # Some injury responses nest team under player, some don't
            team_name = team.get("full_name", "")
            if not team_name:
                # Try to resolve from team_id
                team_id = player.get("team_id")
                if team_id and team_id in self._team_cache:
                    team_name = self._team_cache[team_id].get("full_name", "")

            name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
            if not name:
                continue

            injuries.append({
                "player_name": name,
                "team": team_name,
                "status": BDLClient.normalize_injury_status(inj.get("status", "")),
                "description": inj.get("description", ""),
                "return_date": inj.get("return_date", ""),
            })

        logger.info(f"BDL Bridge: {len(injuries)} normalized injury records")
        return injuries

    # ── Confirmed Starters ───────────────────────────────────────────────

    def get_confirmed_starters(self, bdl_game_id: int) -> Dict[str, bool]:
        """
        Return {player_name: True} for confirmed starters.

        Only available once the game begins — returns empty dict for
        future games (caller falls back to infer_starter_flag).
        """
        starters_by_team = self.bdl.get_starters_for_game(bdl_game_id)
        result: Dict[str, bool] = {}
        for abbr, players in starters_by_team.items():
            for p in players:
                name = p.get("name", "")
                if name:
                    result[name] = True
        return result

    # ── Betting Odds (spreads/totals for game context) ───────────────────

    def get_game_context_odds(
        self, date: str
    ) -> Dict[int, Dict[str, float]]:
        """
        Fetch spreads and totals from BDL for all games on a date.

        Returns: {bdl_game_id: {
            "spread_home": float,  # negative = home favored
            "total": float,
        }}

        Computes median across all vendors for robustness.
        """
        cached = self._odds_cache.get(date)
        if cached and (time.monotonic() - cached[1]) < self._ODDS_CACHE_TTL:
            return cached[0]

        raw_odds = self.bdl.get_betting_odds(dates=[date])

        # Group by game_id
        by_game: Dict[int, List[dict]] = {}
        for odd in raw_odds:
            gid = odd.get("game_id")
            if gid:
                by_game.setdefault(gid, []).append(odd)

        result: Dict[int, Dict[str, float]] = {}
        for gid, odds_list in by_game.items():
            spreads = []
            totals = []
            for o in odds_list:
                sv = o.get("spread_home_value")
                tv = o.get("total_value")
                if sv is not None:
                    try:
                        spreads.append(float(sv))
                    except (ValueError, TypeError):
                        pass
                if tv is not None:
                    try:
                        totals.append(float(tv))
                    except (ValueError, TypeError):
                        pass

            ctx: Dict[str, float] = {}
            if spreads:
                spreads.sort()
                mid = len(spreads) // 2
                ctx["spread_home"] = (
                    (spreads[mid - 1] + spreads[mid]) / 2.0
                    if len(spreads) % 2 == 0
                    else spreads[mid]
                )
            if totals:
                totals.sort()
                mid = len(totals) // 2
                ctx["total"] = (
                    (totals[mid - 1] + totals[mid]) / 2.0
                    if len(totals) % 2 == 0
                    else totals[mid]
                )

            if ctx:
                result[gid] = ctx

        self._odds_cache[date] = (result, time.monotonic())
        return result

    # ── Comprehensive season profile (advanced V2 + playtype + general/advanced) ──

    def get_player_season_profile(
        self, bdl_player_id: int, season: int
    ) -> Dict[str, float]:
        """
        Fetch the full season-level player profile used by XGBoost and the
        fatigue model. Combines three BDL data layers:

          1. V2 per-game advanced stats (last 10 games) → usage%, touches,
             court distance, pct_pts_3pt.
          2. Season averages general/advanced → ts_pct.
          3. Season averages playtype/* → PnR ball-handler, isolation,
             spot-up, and transition frequencies.

        Returns a flat dict with keys:
            avg_distance          — court miles/game (fatigue model)
            real_usage_pct        — true usage % (replaces box-score proxy)
            avg_touches           — per-game ball touches
            pct_pts_3pt           — % of points from 3s (scoring profile)
            ts_pct                — true shooting % (general/advanced)
            pnr_bh_freq           — PnR ball-handler play frequency
            iso_freq              — isolation play frequency
            spotup_freq           — spot-up play frequency
            transition_freq       — transition play frequency
            avg_speed             — avg court speed (mph) from tracking
            avg_contested_fg_pct  — contested FG% (shot difficulty proxy)
            avg_deflections       — deflections per game (defensive activity)
            avg_points_paint      — points in the paint per game
            avg_pct_pts_paint     — % of points scored from the paint
        """
        defaults: Dict[str, float] = {
            "avg_distance":         0.0,
            "real_usage_pct":       0.0,
            "avg_touches":          0.0,
            "pct_pts_3pt":          0.0,
            "ts_pct":               0.0,
            "pnr_bh_freq":          0.0,
            "pnr_roll_freq":        0.0,
            "iso_freq":             0.0,
            "spotup_freq":          0.0,
            "transition_freq":      0.0,
            "postup_freq":          0.0,
            "drives_per_game":      0.0,
            "avg_speed":            0.0,
            "avg_contested_fg_pct": 0.0,
            "avg_deflections":      0.0,
            "avg_points_paint":     0.0,
            "avg_pct_pts_paint":    0.0,
        }
        result = dict(defaults)

        if bdl_player_id in self._profile_cache:
            return self._profile_cache[bdl_player_id]

        # ── Layer 1: V2 per-game advanced stats ──────────────────────────
        adv = self.get_player_advanced_features(bdl_player_id, season, n_games=10)
        
        # ── Layer 2: Season averages general/advanced → ts_pct ───────────
        gen_adv = None
        try:
            rows = self.bdl.get_season_averages(
                season=season,
                player_ids=[bdl_player_id],
                category="general",
                stat_type="advanced",
            )
            if rows:
                gen_adv = rows[0]
        except Exception:
            pass

        # ── Layer 3: Playtype season averages ────────────────────────────
        playtypes = ["prballhandler", "prrollman", "isolation", "spotup", "transition", "postup"]
        pt_data = {}
        for pt in playtypes:
            try:
                rows = self.bdl.get_season_averages(
                    season=season,
                    player_ids=[bdl_player_id],
                    category="playtype",
                    stat_type=pt,
                )
                if rows:
                    pt_data[pt] = rows[0]
            except Exception:
                pass

        # ── Layer 4: Tracking drives ──────────────────────────────────────
        drives = None
        try:
            rows = self.bdl.get_season_averages(
                season=season,
                player_ids=[bdl_player_id],
                category="tracking",
                stat_type="drives",
            )
            if rows:
                drives = rows[0]
        except Exception:
            pass

        profile = self._build_profile_from_data(
            bdl_player_id, 
            adv_rows_recent=[adv], # Wrap in list to match expected format
            gen_adv_row=gen_adv,
            pt_rows=pt_data,
            drives_row=drives
        )
        self._profile_cache[bdl_player_id] = profile
        return profile

    def prefetch_player_profiles(self, player_ids: List[int], season: int):
        """
        Batch-fetch season profiles for multiple players and store in cache.
        Reduces API calls from O(N) to O(1) per scan.
        """
        if not player_ids:
            return
        
        # Filter out already cached players
        needed = [pid for pid in player_ids if pid not in self._profile_cache]
        if not needed:
            return

        logger.info(f"BDL: Prefetching profiles for {len(needed)} players (season {season})")
        import requests

        # Helper to check for 429 and abort prefetch
        def _is_rate_limited(e):
            return "429" in str(e) or isinstance(e, requests.exceptions.RetryError)

        # 1. Advanced stats V2
        adv_by_p = {}
        try:
            adv_rows = self.bdl.get_advanced_stats(player_ids=needed, seasons=[season], period=0)
            for row in adv_rows:
                pid = row.get("player_id")
                if pid:
                    adv_by_p.setdefault(pid, []).append(row)
        except Exception as e:
            logger.warning(f"BDL prefetch: advanced stats failed: {e}")
            if _is_rate_limited(e): return # Abort prefetch if rate limited

        time.sleep(1.0) # Grace period

        # 2. General/Advanced
        gen_adv_by_p = {}
        try:
            gen_adv_rows = self.bdl.get_season_averages(season=season, player_ids=needed, category="general", stat_type="advanced")
            gen_adv_by_p = {row.get("player_id"): row for row in gen_adv_rows if row.get("player_id")}
        except Exception as e:
            if _is_rate_limited(e): return

        time.sleep(1.0)

        # 3. Playtypes
        playtypes = ["prballhandler", "prrollman", "isolation", "spotup", "transition", "postup"]
        pt_data_by_p = {pid: {} for pid in needed}
        for pt in playtypes:
            try:
                rows = self.bdl.get_season_averages(season=season, player_ids=needed, category="playtype", stat_type=pt)
                for row in rows:
                    pid = row.get("player_id")
                    if pid in pt_data_by_p:
                        pt_data_by_p[pid][pt] = row
                time.sleep(0.5) # Breath between playtypes
            except Exception as e:
                if _is_rate_limited(e): return
                continue

        # 4. Tracking/Drives
        drives_by_p = {}
        try:
            drives_rows = self.bdl.get_season_averages(season=season, player_ids=needed, category="tracking", stat_type="drives")
            drives_by_p = {row.get("player_id"): row for row in drives_rows if row.get("player_id")}
        except Exception as e:
            if _is_rate_limited(e): return

        # Build and cache
        for pid in needed:
            profile = self._build_profile_from_data(
                pid,
                adv_rows_recent=adv_by_p.get(pid, [])[:10],
                gen_adv_row=gen_adv_by_p.get(pid),
                pt_rows=pt_data_by_p.get(pid, {}),
                drives_row=drives_by_p.get(pid)
            )
            self._profile_cache[pid] = profile

    def _build_profile_from_data(
        self, bdl_player_id: int, 
        adv_rows_recent: List[dict],
        gen_adv_row: Optional[dict],
        pt_rows: Dict[str, dict],
        drives_row: Optional[dict]
    ) -> Dict[str, float]:
        """Helper to assemble a profile dict from raw BDL components."""
        defaults: Dict[str, float] = {
            "avg_distance":         0.0, "real_usage_pct":       0.0,
            "avg_touches":          0.0, "pct_pts_3pt":          0.0,
            "ts_pct":               0.0, "pnr_bh_freq":          0.0,
            "pnr_roll_freq":        0.0, "iso_freq":             0.0,
            "spotup_freq":          0.0, "transition_freq":      0.0,
            "postup_freq":          0.0, "drives_per_game":      0.0,
            "avg_speed":            0.0, "avg_contested_fg_pct": 0.0,
            "avg_deflections":      0.0, "avg_points_paint":     0.0,
            "avg_pct_pts_paint":    0.0,
        }
        result = dict(defaults)

        # Layer 1: Advanced (recent)
        if adv_rows_recent:
            n = len(adv_rows_recent)
            def _avg(key: str) -> float:
                return sum(float(s.get(key, 0) or 0) for s in adv_rows_recent) / n
            
            # Note: get_player_advanced_features uses specific keys.
            # BDL V2 stats use 'usage_percentage', 'touches', etc.
            # If adv_rows_recent came from get_player_advanced_features, it's already a single dict.
            # If it's a list from prefetch, we need to average them.
            if isinstance(adv_rows_recent[0], dict) and "avg_usage_pct" in adv_rows_recent[0]:
                # It's already the averaged dict from get_player_advanced_features
                r = adv_rows_recent[0]
                result.update({
                    "avg_distance": r.get("avg_distance", 0.0),
                    "real_usage_pct": r.get("avg_usage_pct", 0.0),
                    "avg_touches": r.get("avg_touches", 0.0),
                    "pct_pts_3pt": r.get("avg_pct_pts_3pt", 0.0),
                    "avg_speed": r.get("avg_speed", 0.0),
                    "avg_contested_fg_pct": r.get("avg_contested_fg_pct", 0.0),
                    "avg_deflections": r.get("avg_deflections", 0.0),
                    "avg_points_paint": r.get("avg_points_paint", 0.0),
                    "avg_pct_pts_paint": r.get("avg_pct_pts_paint", 0.0),
                })
            else:
                # It's raw rows from prefetch
                result["avg_distance"]         = _avg("distance")
                result["real_usage_pct"]       = _avg("usage_percentage")
                result["avg_touches"]          = _avg("touches")
                result["pct_pts_3pt"]          = _avg("pct_pts_3pt")
                result["avg_speed"]            = _avg("speed")
                result["avg_contested_fg_pct"] = _avg("contested_fg_pct")
                result["avg_deflections"]      = _avg("deflections")
                result["avg_points_paint"]     = _avg("points_paint")
                result["avg_pct_pts_paint"]    = _avg("pct_pts_paint")

        # Layer 2: General/Advanced
        if gen_adv_row:
            result["ts_pct"] = float(gen_adv_row.get("ts_pct", 0.0) or 0.0)

        # Layer 3: Playtype
        _pt_map = {
            "prballhandler": "pnr_bh_freq", "prrollman": "pnr_roll_freq",
            "isolation": "iso_freq", "spotup": "spotup_freq",
            "transition": "transition_freq", "postup": "postup_freq",
        }
        for bdl_key, res_key in _pt_map.items():
            row = pt_rows.get(bdl_key)
            if row:
                freq = row.get("percent_of_plays") or row.get("frequency") or 0.0
                result[res_key] = float(freq or 0.0)

        # Layer 4: Tracking
        if drives_row:
            result["drives_per_game"] = float(drives_row.get("drives", 0.0) or 0.0)

        return result

    # ── Advanced Stats for XGBoost features ──────────────────────────────

    def get_player_advanced_features(
        self, bdl_player_id: int, season: int, n_games: int = 10
    ) -> Dict[str, float]:
        """
        Fetch recent advanced stats for a player and return
        aggregated features for XGBoost input.

        Returns dict with keys:
            avg_usage_pct, avg_touches, avg_speed, avg_distance,
            avg_contested_fg_pct, avg_deflections, avg_points_paint,
            avg_pct_pts_paint, avg_pct_pts_3pt
        """
        defaults = {
            "avg_usage_pct": 0.0,
            "avg_touches": 0.0,
            "avg_speed": 0.0,
            "avg_distance": 0.0,
            "avg_contested_fg_pct": 0.0,
            "avg_deflections": 0.0,
            "avg_points_paint": 0.0,
            "avg_pct_pts_paint": 0.0,
            "avg_pct_pts_3pt": 0.0,
        }

        try:
            stats = self.bdl.get_advanced_stats(
                player_ids=[bdl_player_id],
                seasons=[season],
                period=0,  # full game only
            )
        except Exception:
            return defaults

        if not stats:
            return defaults

        # Take last n_games entries (BDL returns newest first)
        recent = stats[:n_games]
        n = len(recent)
        if n == 0:
            return defaults

        def _avg(key: str) -> float:
            vals = [float(s.get(key, 0) or 0) for s in recent]
            return sum(vals) / n if n > 0 else 0.0

        return {
            "avg_usage_pct": _avg("usage_percentage"),
            "avg_touches": _avg("touches"),
            "avg_speed": _avg("speed"),
            "avg_distance": _avg("distance"),
            "avg_contested_fg_pct": _avg("contested_fg_pct"),
            "avg_deflections": _avg("deflections"),
            "avg_points_paint": _avg("points_paint"),
            "avg_pct_pts_paint": _avg("pct_pts_paint"),
            "avg_pct_pts_3pt": _avg("pct_pts_3pt"),
        }
