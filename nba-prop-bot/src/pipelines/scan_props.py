"""
Main edge-detection pipeline — V2 (BDL-powered).

Data source strategy:
  PRIMARY (BDL GOAT):
    - Player props (live, 600 req/min, 9+ rec books)
    - Player injuries (structured, no scraping)
    - Confirmed lineups (pre-game when available)
    - Game context odds (spreads/totals from 13 vendors)
    - Advanced stats V2 (usage%, touches, tracking)

  SECONDARY (Odds API — sharp books only):
    - Pinnacle/Circa/Bookmaker lines for consensus true probability
    - Sharp vs. rec spread detection (steam chases)

  FALLBACK (nba_api — free):
    - Player game logs (historical rates, XGBoost features)
    - Team stats (pace, defense, opponent allowed)
    - Play-by-play (on_off_splits, rotation model)

Credit savings: ~90% reduction in Odds API usage.
  Before: 11 credits per scan × 16 scans/day = 176 credits/day
  After:  1-2 credits per scan (sharp books only) × 16 = 32 credits/day
"""

import time
from datetime import datetime, timezone
import dateutil.parser
from dateutil import tz
from typing import List, Dict, Any, Tuple, Optional

from src.utils.logging_utils import get_logger
from src.data.db import DatabaseClient
from src.clients.odds_api import OddsApiClient
from src.clients.nba_stats import NbaStatsClient
from src.clients.injuries import InjuryClient
from src.clients.telegram_bot import TelegramBotClient
from src.config import (
    PROP_MARKETS, EDGE_MIN, REC_BOOKS, SHARP_EDGE_MIN,
    CONSENSUS_BOOKS, CONSENSUS_HOLD_MAX,
    BDL_ENABLED, BDL_PROP_VENDORS,
)
from src.models.projections import build_player_projection
from src.models.distributions import (
    get_probability_distribution,
    compute_player_foul_rate,
    classify_bench_tier,
)
from src.models.devig import (
    decimal_to_implied_prob, devig_two_way,
    build_consensus_true_prob, get_theoretical_hold,
)
from src.models.edge_ranker import rank_edges, set_db as set_ranker_db
from src.models.ml_model import get_ml_projection
from src.pipelines.send_alerts import evaluate_and_alert
from src.pipelines.combos import generate_and_alert_combos
from src.models.sgp_correlations import build_team_correlation_matrix
from src.clients.on_off_splits import OnOffSplitsClient
from src.clients.rotation_model import RotationModel
from src.clients.travel_fatigue import compute_travel_fatigue, TEAM_NAME_TO_ABBR
from src.clients.referee_client import fetch_today_assignments, match_event_refs
from src.models.referee_stats import get_crew_foul_factor

logger = get_logger(__name__)
_PROJECTIONS_CACHE: Dict[str, Any] = {}

_SHARP_TS_BOOKS      = {'pinnacle', 'circa', 'bookmaker'}
_SOFT_TS_BOOKS       = {'draftkings', 'fanduel', 'betmgm', 'caesars'}
_STALE_THRESHOLD_SEC = 60
_SHARP_RECENT_SEC    = 120

# ─── BDL integration (conditional import) ─────────────────────────────

_bdl_bridge = None
if BDL_ENABLED:
    try:
        from src.clients.bdl_bridge import BDLBridge
        from src.clients.bdl_client import BDLClient
        _bdl_bridge = BDLBridge(BDLClient())
        logger.info("BDL GOAT tier enabled — using BDL for props, injuries, lineups")
    except Exception as _bdl_err:
        logger.warning(f"BDL import failed, falling back to Odds API: {_bdl_err}")
        BDL_ENABLED_RUNTIME = False
else:
    BDL_ENABLED_RUNTIME = False

BDL_ENABLED_RUNTIME = BDL_ENABLED and _bdl_bridge is not None


# ─── Helper functions (unchanged from V1) ─────────────────────────────

def _check_timestamp_staleness(
    bookmakers: List[Dict], market_key: str, soft_book: Optional[str]
) -> Dict[str, Any]:
    _no_signal = {'timestamp_stale': False, 'lag_seconds': 0.0}
    if not soft_book or soft_book.lower() not in _SOFT_TS_BOOKS:
        return _no_signal
    now = datetime.now(timezone.utc)
    sharp_ts: Optional[datetime] = None
    for book in bookmakers:
        if book.get('title', '').lower() not in _SHARP_TS_BOOKS:
            continue
        for mkt in book.get('markets', []):
            if mkt.get('key') != market_key:
                continue
            ts_str = mkt.get('last_update') or book.get('last_update')
            if not ts_str:
                continue
            try:
                ts = dateutil.parser.isoparse(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if sharp_ts is None or ts > sharp_ts:
                    sharp_ts = ts
            except Exception:
                pass
    if sharp_ts is None:
        return _no_signal
    if (now - sharp_ts).total_seconds() > _SHARP_RECENT_SEC:
        return _no_signal
    soft_ts: Optional[datetime] = None
    soft_lower = soft_book.lower()
    for book in bookmakers:
        if book.get('title', '').lower() != soft_lower:
            continue
        for mkt in book.get('markets', []):
            if mkt.get('key') != market_key:
                continue
            ts_str = mkt.get('last_update') or book.get('last_update')
            if not ts_str:
                continue
            try:
                soft_ts = dateutil.parser.isoparse(ts_str)
                if soft_ts.tzinfo is None:
                    soft_ts = soft_ts.replace(tzinfo=timezone.utc)
            except Exception:
                pass
        if soft_ts:
            break
    if soft_ts is None:
        return _no_signal
    lag = (sharp_ts - soft_ts).total_seconds()
    return {
        'timestamp_stale': lag >= _STALE_THRESHOLD_SEC,
        'lag_seconds': float(max(0.0, lag)),
    }


def get_best_odds(bookmakers: List[Dict], player_name: str,
                  market_key: str, line: float) -> Tuple[Dict, Dict]:
    best_over  = {"price": 0.0, "book": None}
    best_under = {"price": 0.0, "book": None}
    for book in bookmakers:
        for mkt in book.get('markets', []):
            if mkt['key'] != market_key:
                continue
            for outcome in mkt.get('outcomes', []):
                if outcome.get('description') != player_name or outcome.get('point') != line:
                    continue
                side  = outcome.get('name', '').lower()
                price = outcome.get('price', 0.0)
                if side == 'over' and price > best_over['price']:
                    best_over = {"price": price, "book": book['title']}
                elif side == 'under' and price > best_under['price']:
                    best_under = {"price": price, "book": book['title']}
    return best_over, best_under


def _american(decimal_odds: float) -> str:
    if decimal_odds >= 2.0:
        return f"+{int((decimal_odds - 1) * 100)}"
    if decimal_odds <= 1.0:
        return "N/A"
    return f"-{int(100 / (decimal_odds - 1))}"


def get_consensus_true_prob(
    bookmakers: List[Dict], player_name: str, market_key: str, line: float,
    book_weights: Dict,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    consensus_lower = {b.lower() for b in CONSENSUS_BOOKS}
    book_probs = []
    for book in bookmakers:
        title_lower = book.get('title', '').lower()
        if title_lower not in consensus_lower:
            continue
        over_price = under_price = None
        for mkt in book.get('markets', []):
            if mkt.get('key') != market_key:
                continue
            for outcome in mkt.get('outcomes', []):
                if outcome.get('description', '').lower() != player_name.lower():
                    continue
                if abs(float(outcome.get('point', 0)) - line) > 0.01:
                    continue
                side = outcome.get('name', '').lower()
                if side == 'over':
                    over_price = float(outcome['price'])
                elif side == 'under':
                    under_price = float(outcome['price'])
        if over_price and under_price:
            raw_o = decimal_to_implied_prob(over_price)
            raw_u = decimal_to_implied_prob(under_price)
            hold  = get_theoretical_hold(raw_o, raw_u)
            if hold > CONSENSUS_HOLD_MAX:
                continue
            true_o, true_u = devig_two_way(raw_o, raw_u)
            weight = book_weights.get(title_lower, 0.5)
            book_probs.append({
                'book': book['title'], 'over': true_o, 'under': true_u,
                'weight': weight, 'hold': hold,
            })
    if not book_probs:
        return None, None, None
    cons_o, cons_u, label = build_consensus_true_prob(book_probs)
    return cons_o, cons_u, label


def get_best_rec_price(
    bookmakers: List[Dict], player_name: str,
    market_key: str, line: float, side: str
) -> Tuple[Optional[float], Optional[str]]:
    rec_lower  = [b.lower() for b in REC_BOOKS]
    best_price = None
    best_book  = None
    for book in bookmakers:
        if book.get('title', '').lower() not in rec_lower:
            continue
        for mkt in book.get('markets', []):
            if mkt.get('key') != market_key:
                continue
            for outcome in mkt.get('outcomes', []):
                if outcome.get('description', '').lower() != player_name.lower():
                    continue
                if abs(float(outcome.get('point', 0)) - line) > 0.01:
                    continue
                if outcome.get('name', '').lower() != side.lower():
                    continue
                price = float(outcome['price'])
                if best_price is None or price > best_price:
                    best_price = price
                    best_book  = book['title']
    return best_price, best_book


# ─── BDL-specific helpers ─────────────────────────────────────────────

def _get_best_bdl_odds(
    best_odds: Dict[tuple, dict], player_name: str,
    market: str, line: float
) -> Tuple[Dict, Dict]:
    """Extract best over/under from BDL bridge best_odds dict."""
    over_key = (player_name, market, line, "OVER")
    under_key = (player_name, market, line, "UNDER")
    best_over = best_odds.get(over_key, {"price": 0.0, "book": None})
    best_under = best_odds.get(under_key, {"price": 0.0, "book": None})
    return best_over, best_under


def _sync_injuries_bdl(db: DatabaseClient, game_date: str):
    """Sync injuries from BDL (structured API — no scraping)."""
    injuries = _bdl_bridge.get_injuries_for_date()
    if not injuries:
        logger.warning("BDL: No injury data — falling back to scraping.")
        return False

    with db.get_conn() as conn:
        cursor = conn.cursor()
        for inj in injuries:
            try:
                cursor.execute(
                    """INSERT OR REPLACE INTO injury_reports
                       (game_date, player_name, team, status)
                       VALUES (?, ?, ?, ?)""",
                    (game_date, inj['player_name'], inj['team'], inj['status'])
                )
            except Exception:
                continue
    logger.info(f"BDL: Persisted {len(injuries)} injury records for {game_date}.")
    return True


def _sync_injuries_legacy(db: DatabaseClient, injury_client: InjuryClient, game_date: str):
    """Legacy injury sync via scraping (fallback)."""
    injuries = injury_client.get_injuries()
    if not injuries:
        logger.warning("No injury data fetched — all players treated as Healthy.")
        return
    with db.get_conn() as conn:
        cursor = conn.cursor()
        for inj in injuries:
            try:
                cursor.execute(
                    """INSERT OR REPLACE INTO injury_reports
                       (game_date, player_name, team, status)
                       VALUES (?, ?, ?, ?)""",
                    (game_date, inj['player_name'], inj['team'], inj['status'])
                )
            except Exception:
                continue
    logger.info(f"Legacy: Persisted {len(injuries)} injury records for {game_date}.")


# ─── Main scan pipeline ──────────────────────────────────────────────

def scan_props():
    logger.info(f"Initializing scan pipeline ({'BDL+Sharp' if BDL_ENABLED_RUNTIME else 'Odds API only'})...")

    db             = DatabaseClient()
    odds_client    = OddsApiClient()
    stats_client   = NbaStatsClient()
    injury_client  = InjuryClient()
    bot            = TelegramBotClient()
    on_off_client  = OnOffSplitsClient()
    rotation_model = RotationModel(stats_client)

    set_ranker_db(db)
    db.init_bookmaker_profiles()
    _book_weights = db.get_sharp_book_weights()

    today      = datetime.now().strftime('%Y-%m-%d')
    local_zone = tz.tzlocal()

    # ── 1. Sync injuries ─────────────────────────────────────────────
    if BDL_ENABLED_RUNTIME:
        if not _sync_injuries_bdl(db, today):
            _sync_injuries_legacy(db, injury_client, today)
    else:
        _sync_injuries_legacy(db, injury_client, today)

    # ── 2. Get today's games ─────────────────────────────────────────
    if BDL_ENABLED_RUNTIME:
        bdl_games = _bdl_bridge.get_today_games(today)
        today_events = []
        for g in bdl_games:
            if g["status"] == "Final":
                continue  # skip completed games
            today_events.append(g)
        logger.info(f"BDL: {len(today_events)} active games today")

        # Also fetch BDL game context (spreads/totals) in one batch
        bdl_odds_ctx = _bdl_bridge.get_game_context_odds(today)
    else:
        bdl_games = []
        bdl_odds_ctx = {}
        try:
            events = odds_client.get_events()
            today_events = [
                e for e in events
                if dateutil.parser.isoparse(e['commence_time'])
                   .astimezone(local_zone).strftime('%Y-%m-%d') == today
            ]
        except Exception as e:
            logger.error(f"Failed to fetch events: {e}")
            return

    # BDL season year (used for advanced-stats lookups)
    _season_int = int(stats_client.season.split('-')[0]) if BDL_ENABLED_RUNTIME else 0

    # Referee assignments (0 credits)
    _all_ref_assignments = fetch_today_assignments()

    candidates = []

    for event in today_events:
        # ── Resolve game identifiers ──────────────────────────────────
        if BDL_ENABLED_RUNTIME:
            bdl_game_id = event.get("bdl_game_id")
            home_team   = event.get("home_team", "")
            away_team   = event.get("away_team", "")
            event_id    = str(bdl_game_id)
            commence_str = event.get("commence_time", "")
        else:
            bdl_game_id = None
            event_id    = event['id']
            home_team   = event['home_team']
            away_team   = event['away_team']
            commence_str = event.get('commence_time', '')

        game_date = today

        # ── 3. Fetch prop lines ───────────────────────────────────────
        if BDL_ENABLED_RUNTIME and bdl_game_id:
            # BDL: get all rec-book props (0 Odds API credits)
            bdl_data = _bdl_bridge.get_props_for_game(bdl_game_id, vendors=BDL_PROP_VENDORS)
            players_in_event = bdl_data["players_in_event"]
            prices_by_market = bdl_data["prices_by_market"]
            bdl_best_odds    = bdl_data["best_odds"]
            bdl_player_map   = bdl_data["player_id_map"]

            # Insert line history
            if bdl_data["line_records"]:
                db.insert_line_history_batch(bdl_data["line_records"])

            # Fetch sharp books from Odds API (1 credit per game — Pinnacle only)
            sharp_bookmakers = []
            try:
                sharp_odds = odds_client.get_event_odds(
                    event_id=event_id,
                    markets=[*PROP_MARKETS, 'spreads', 'totals']
                )
                sharp_bookmakers = sharp_odds.get('bookmakers', [])
            except Exception as _se:
                logger.debug(f"Sharp book fetch skipped for {event_id}: {_se}")

            # Game context from BDL odds (spreads/totals)
            _ctx = bdl_odds_ctx.get(bdl_game_id, {})
            _home_spread = _ctx.get("spread_home", 0.0)
            _game_total  = _ctx.get("total", 0.0)

            # If BDL didn't have odds, try Odds API bookmakers
            if not _home_spread and sharp_bookmakers:
                _home_spread = OddsApiClient.extract_consensus_spread(
                    sharp_bookmakers, home_team) or 0.0
            if not _game_total and sharp_bookmakers:
                _game_total = OddsApiClient.extract_consensus_total(
                    sharp_bookmakers) or 0.0

            # Confirmed starters from BDL
            bdl_starters = _bdl_bridge.get_confirmed_starters(bdl_game_id)
        else:
            # Legacy path: everything from Odds API
            bdl_best_odds = {}
            bdl_player_map = {}
            bdl_starters = {}

            try:
                odds_data = odds_client.get_event_odds(
                    event_id=event_id,
                    markets=[*PROP_MARKETS, 'spreads', 'totals']
                )
            except Exception:
                continue

            sharp_bookmakers = odds_data.get('bookmakers', [])
            bookmakers = sharp_bookmakers  # same thing in legacy mode

            if not bookmakers:
                continue

            _home_spread = OddsApiClient.extract_consensus_spread(bookmakers, home_team)
            _game_total  = OddsApiClient.extract_consensus_total(bookmakers)

            players_in_event = set()
            prices_by_market = {}

            for mkt in PROP_MARKETS:
                prices_by_market[mkt] = {}
                line_records = []
                for book in bookmakers:
                    for book_mkt in book.get('markets', []):
                        if book_mkt['key'] != mkt:
                            continue
                        for outcome in book_mkt.get('outcomes', []):
                            player = outcome.get('description')
                            line   = outcome.get('point')
                            if not player or line is None:
                                continue
                            players_in_event.add(player)
                            prices_by_market[mkt].setdefault(player, set()).add(line)
                            side  = outcome.get('name', '').upper()
                            price = outcome.get('price', 0.0)
                            if price > 0:
                                line_records.append(
                                    (player, mkt, book.get('title'), line, side, price, 1.0 / price)
                                )
                if line_records:
                    db.insert_line_history_batch(line_records)

        # ── Hours to tip-off ──────────────────────────────────────────
        _hours_to_tip = 4.0
        if commence_str:
            try:
                _commence_dt = dateutil.parser.isoparse(commence_str)
                if _commence_dt.tzinfo is None:
                    _commence_dt = _commence_dt.replace(tzinfo=timezone.utc)
                _hours_to_tip = max(
                    0.0,
                    (_commence_dt - datetime.now(timezone.utc)).total_seconds() / 3600.0
                )
            except Exception:
                pass

        # ── Referee whistle factor ────────────────────────────────────
        _event_refs  = match_event_refs(home_team, away_team, _all_ref_assignments)
        _crew_factor = get_crew_foul_factor(_event_refs, db)

        # ── Cross-player correlations ─────────────────────────────────
        for _team in (home_team, away_team):
            try:
                build_team_correlation_matrix(_team, stats_client, db)
            except Exception:
                pass

        # ── OUT players ───────────────────────────────────────────────
        out_players: List[str] = []
        with db.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT player_name FROM injury_reports
                   WHERE game_date = ? AND status = 'Out'
                   AND (team = ? OR team = ?)""",
                (today, home_team, away_team)
            )
            out_players = [r['player_name'] for r in cursor.fetchall()]

        absent_ids: List[int] = []
        _out_player_avg_mins: float = 0.0
        if out_players:
            for _nm in out_players:
                _aid = NbaStatsClient.resolve_player_id(_nm)
                if _aid:
                    absent_ids.append(_aid)
                    try:
                        _out_logs = stats_client.get_player_game_logs(_aid)
                        if not _out_logs.empty and 'MIN' in _out_logs.columns:
                            _avg = _out_logs['MIN'].head(10).mean()
                            if _avg == _avg:
                                _out_player_avg_mins += float(_avg)
                    except Exception:
                        pass
            _out_player_avg_mins = min(_out_player_avg_mins, 40.0)

        # ── Pace data ─────────────────────────────────────────────────
        pace_info = stats_client.get_team_pace(home_team, away_team)

        sharp_alerted: set = set()

        # ── Per-player loop ───────────────────────────────────────────
        for player_name in players_in_event:
            # Injury status
            injury_status = "Healthy"
            if player_name in out_players:
                injury_status = "Out"
            else:
                with db.get_conn() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT status FROM injury_reports WHERE player_name = ? AND game_date = ?",
                        (player_name, today)
                    )
                    inj_row = cursor.fetchone()
                    if inj_row:
                        injury_status = inj_row['status'] or 'Unknown'

            if "out" in (injury_status or 'healthy').lower():
                continue

            # Fetch game logs (nba_api — still needed for projection model)
            cache_key = f"{player_name}_{today}"
            if cache_key not in _PROJECTIONS_CACHE:
                try:
                    from nba_api.stats.static import players as nba_players
                    found = nba_players.find_players_by_full_name(player_name)
                    if not found:
                        continue
                    player_id = found[0]['id']
                    logs = stats_client.get_player_game_logs(player_id)
                    time.sleep(0.6)
                    _PROJECTIONS_CACHE[cache_key] = {"logs": logs, "pid": player_id}
                except Exception:
                    continue

            p_data        = _PROJECTIONS_CACHE[cache_key]
            logs          = p_data["logs"]
            player_id_int = p_data.get("pid")

            # Home/away + rest days
            player_team_abbr = logs.iloc[0]['TEAM_ABBREVIATION'] if not logs.empty else None
            home_flag  = stats_client.is_home_team(player_team_abbr or '', home_team)
            rest_days  = NbaStatsClient.calculate_rest_days(logs)
            b2b_flag   = rest_days == 0

            # Travel fatigue
            _home_abbr = TEAM_NAME_TO_ABBR.get(home_team, '')
            _fatigue   = compute_travel_fatigue(
                player_team_abbr=player_team_abbr or '',
                today_arena_abbr=_home_abbr,
                logs=logs, b2b_flag=b2b_flag,
            )

            # Starter flag: BDL confirmed > inferred
            starter_flag = bdl_starters.get(player_name, False)
            if not starter_flag:
                starter_flag = NbaStatsClient.infer_starter_flag(logs)

            opp_team = away_team if home_flag else home_team
            _opp_ctx = stats_client.get_opponent_matchup_context(opp_team)
            _player_foul_rate = compute_player_foul_rate(logs)

            # Rotation model
            _rotation_mins = 0.0
            if absent_ids and player_id_int:
                try:
                    _rotation_mins = rotation_model.get_projected_minutes(
                        target_player_id=int(player_id_int),
                        absent_player_ids=absent_ids,
                        logs=logs, season=stats_client.season, db=db,
                    )
                except Exception:
                    pass

            _position_group = NbaStatsClient.infer_position_group(logs)

            # BDL advanced features: court-distance supplement to fatigue model.
            # avg_distance > 3.0 miles/game signals high accumulated playing effort
            # beyond what travel_fatigue captures (travel vs. on-court exertion).
            _adj_fatigue_mult = _fatigue['fatigue_multiplier']
            if BDL_ENABLED_RUNTIME and player_name in bdl_player_map:
                _adv_ck = f"bdl_adv_{bdl_player_map[player_name]}"
                if _adv_ck not in _PROJECTIONS_CACHE:
                    _PROJECTIONS_CACHE[_adv_ck] = _bdl_bridge.get_player_advanced_features(
                        bdl_player_map[player_name], season=_season_int
                    )
                _bdl_adv = _PROJECTIONS_CACHE[_adv_ck]
                if _bdl_adv.get("avg_distance", 0.0) > 3.0:
                    _adj_fatigue_mult *= 0.98

            for mkt in PROP_MARKETS:
                if player_name not in prices_by_market.get(mkt, {}):
                    continue

                opp_multiplier = stats_client.get_positional_def_multiplier(
                    opp_team, mkt, _position_group
                )

                for line in prices_by_market[mkt][player_name]:
                    # On/off usage shift
                    if absent_ids and player_id_int:
                        shifts = [
                            on_off_client.get_usage_multiplier(
                                target_player_id=int(player_id_int),
                                absent_player_id=_aid, market=mkt, db=db,
                            ) - 1.0
                            for _aid in absent_ids
                        ]
                        _usage_shift = min(sum(shifts), 0.50)
                    else:
                        _usage_shift = 0.0

                    proj = build_player_projection(
                        player_id=player_name, market=mkt, line=line,
                        recent_logs=logs, season_logs=logs,
                        injury_status=injury_status,
                        team_pace=pace_info['home_pace'] if home_flag else pace_info['away_pace'],
                        opp_pace=pace_info['away_pace'] if home_flag else pace_info['home_pace'],
                        opponent_multiplier=opp_multiplier,
                        usage_shift=_usage_shift,
                        starter_flag=starter_flag,
                        b2b_flag=b2b_flag, home_flag=home_flag,
                        rest_days=rest_days,
                        out_player_avg_mins=_out_player_avg_mins,
                        projected_minutes_override=_rotation_mins,
                        fatigue_multiplier=_adj_fatigue_mult,
                    )

                    if not proj or proj.get('mean', 0) == 0:
                        continue

                    if proj.get('projected_minutes', 0) < 10:
                        continue

                    # XGBoost blend
                    _league_avg_pace = 99.0
                    _pace_factor = (
                        (pace_info['home_pace'] + pace_info['away_pace'])
                        / (2.0 * _league_avg_pace)
                    )
                    ml_mean = get_ml_projection(
                        mkt, logs, proj['projected_minutes'], home_flag, rest_days,
                        opp_def_rating=opp_multiplier,
                        pace_factor=_pace_factor,
                        opp_pace=_opp_ctx['opp_pace'],
                        opp_rebound_pct=_opp_ctx['opp_rebound_pct'],
                        opp_pts_paint=_opp_ctx['opp_pts_paint'],
                        travel_miles=_fatigue['miles_traveled'],
                        tz_shift_hours=_fatigue['tz_shift_hours'],
                        altitude_flag=_fatigue['altitude_flag'],
                    )
                    if ml_mean is not None and ml_mean > 0:
                        proj['mean'] = 0.5 * proj['mean'] + 0.5 * ml_mean
                        proj['ml_blend'] = True

                    # Tight whistle
                    if _crew_factor["tight_whistle"] and mkt in {
                        "player_points", "player_points_rebounds_assists"
                    }:
                        proj['mean'] *= 1.02

                    # Altitude fatigue
                    if (b2b_flag and _fatigue['altitude_flag']
                        and _fatigue['miles_traveled'] > 800 and not home_flag):
                        if mkt == 'player_threes':
                            proj['mean'] *= 0.95
                        elif mkt == 'player_rebounds':
                            proj['mean'] *= 0.97

                    _bench_tier = classify_bench_tier(proj['projected_minutes'])

                    dists = get_probability_distribution(
                        mkt, proj['mean'], line, logs=logs,
                        variance_scale=proj.get('variance_scale', 1.0),
                        proj_minutes=proj['projected_minutes'],
                        spread=_home_spread or 0.0,
                        total=_game_total or 0.0,
                        player_foul_rate=_player_foul_rate,
                        opp_foul_rate=_opp_ctx['opp_fta_rate'] * _crew_factor["foul_rate_multiplier"],
                        bench_tier=_bench_tier,
                    )

                    # ── Get best odds (BDL primary, Odds API for sharp) ───
                    if BDL_ENABLED_RUNTIME:
                        best_over, best_under = _get_best_bdl_odds(
                            bdl_best_odds, player_name, mkt, line
                        )
                        # Also check sharp bookmakers for better price
                        if sharp_bookmakers:
                            s_over, s_under = get_best_odds(
                                sharp_bookmakers, player_name, mkt, line
                            )
                            if s_over['price'] > best_over.get('price', 0):
                                best_over = s_over
                            if s_under['price'] > best_under.get('price', 0):
                                best_under = s_under
                    else:
                        best_over, best_under = get_best_odds(
                            sharp_bookmakers, player_name, mkt, line
                        )

                    # Devig
                    imp_over = imp_under = 0.0
                    if best_over.get('price', 0) > 0 and best_under.get('price', 0) > 0:
                        raw_imp_o = decimal_to_implied_prob(best_over['price'])
                        raw_imp_u = decimal_to_implied_prob(best_under['price'])
                        imp_over, imp_under = devig_two_way(raw_imp_o, raw_imp_u)
                    elif best_over.get('price', 0) > 0:
                        imp_over = decimal_to_implied_prob(best_over['price'])
                    elif best_under.get('price', 0) > 0:
                        imp_under = decimal_to_implied_prob(best_under['price'])

                    # Consensus from sharp bookmakers (Odds API)
                    _consensus_o, _consensus_u, _consensus_label = None, None, None
                    if sharp_bookmakers:
                        _consensus_o, _consensus_u, _consensus_label = get_consensus_true_prob(
                            sharp_bookmakers, player_name, mkt, line, _book_weights
                        )

                    common = {
                        **proj,
                        "home_team": home_team, "away_team": away_team,
                        "game_date": game_date, "event_id": event_id,
                        "home_away": "HOME" if home_flag else "AWAY",
                        "rest_days": rest_days,
                        "team_name": home_team if home_flag else away_team,
                        "consensus_prob": _consensus_o,
                        "consensus_label": _consensus_label,
                        "hours_to_tipoff": _hours_to_tip,
                    }

                    if best_over.get('price', 0) > 0 and imp_over > 0:
                        over_metrics = db.get_market_metrics(player_name, mkt, line, "OVER")
                        _ts_over = _check_timestamp_staleness(
                            sharp_bookmakers, mkt, best_over.get('book')
                        ) if sharp_bookmakers else {'timestamp_stale': False, 'lag_seconds': 0.0}
                        candidates.append({
                            **common,
                            "side": "OVER", "book": best_over['book'],
                            "book_role": db.get_bookmaker_role(best_over.get('book', '')),
                            "odds": best_over['price'],
                            "model_prob": dists['prob_over'],
                            "implied_prob": imp_over,
                            **over_metrics, **_ts_over,
                        })

                    if best_under.get('price', 0) > 0 and imp_under > 0:
                        under_metrics = db.get_market_metrics(player_name, mkt, line, "UNDER")
                        _ts_under = _check_timestamp_staleness(
                            sharp_bookmakers, mkt, best_under.get('book')
                        ) if sharp_bookmakers else {'timestamp_stale': False, 'lag_seconds': 0.0}
                        candidates.append({
                            **common,
                            "side": "UNDER", "book": best_under['book'],
                            "book_role": db.get_bookmaker_role(best_under.get('book', '')),
                            "odds": best_under['price'],
                            "model_prob": dists['prob_under'],
                            "implied_prob": imp_under,
                            **under_metrics, **_ts_under,
                        })

                    # Sharp vs rec alerts (from Odds API sharp bookmakers only)
                    if _consensus_o is not None and sharp_bookmakers:
                        mkt_label = mkt.replace('player_', '').replace('_', ' ').title()
                        _sharp_sides = [('Over', _consensus_o)]
                        if _consensus_u is not None:
                            _sharp_sides.append(('Under', _consensus_u))
                        for chk_side, true_prob in _sharp_sides:
                            rec_price, rec_book = get_best_rec_price(
                                sharp_bookmakers, player_name, mkt, line, chk_side
                            )
                            if rec_price is None:
                                continue
                            rec_implied = decimal_to_implied_prob(rec_price)
                            sharp_gap = true_prob - rec_implied
                            if sharp_gap < SHARP_EDGE_MIN:
                                continue
                            dedup_key = (player_name, mkt, line, chk_side)
                            if dedup_key in sharp_alerted:
                                continue
                            sharp_alerted.add(dedup_key)
                            msg = (
                                f"🔪 <b>Sharp Alert: {player_name} "
                                f"{chk_side} {line} {mkt_label}</b>\n\n"
                                f"Consensus ({_consensus_label}) True: {true_prob:.1%}"
                                f" → {rec_book.title()}: {rec_implied:.1%}\n"
                                f"Gap: <b>+{sharp_gap:.1%}</b>"
                                f" | {rec_book.title()} {_american(rec_price)}"
                            )
                            bot.send_message(msg)

    # ── Rank and alert ────────────────────────────────────────────────
    ranked_edges = rank_edges(candidates)
    actionable = [e for e in ranked_edges if e.get('edge', 0) >= e.get('edge_min_applied', EDGE_MIN)]

    logger.info(
        f"Scan complete: {len(candidates)} candidates, {len(actionable)} actionable edges. "
        f"{'[BDL+Sharp mode]' if BDL_ENABLED_RUNTIME else '[Odds API mode]'}"
    )

    for edge in actionable:
        evaluate_and_alert(edge, db, bot)

    generate_and_alert_combos(actionable, bot, db=db)

    if BDL_ENABLED_RUNTIME and _bdl_bridge:
        logger.info(f"BDL requests this scan: {_bdl_bridge.bdl.requests_made}")


if __name__ == "__main__":
    scan_props()
