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
import pandas as pd

from src.utils.logging_utils import get_logger
from src.data.db import DatabaseClient
from src.clients.odds_api import OddsApiClient
from src.clients.nba_stats import NbaStatsClient
from src.clients.injuries import InjuryClient
from src.clients.telegram_bot import TelegramBotClient
from src.config import (
    PROP_MARKETS, ALT_PROP_MARKETS, ALT_TO_BASE_MARKET,
    EDGE_MIN, REC_BOOKS, SHARP_EDGE_MIN,
    PLAYOFF_EDGE_MIN, PLAYOFF_SHARP_EDGE_MIN,
    CONSENSUS_BOOKS, CONSENSUS_HOLD_MAX,
    BDL_ENABLED, BDL_PROP_VENDORS,
)
from src.models.projections import build_player_projection
from src.models.distributions import (
    get_probability_distribution,
    compute_player_foul_rate,
    classify_bench_tier,
    project_game_markets,
    project_q1_markets,
    project_team_totals,
)
from src.models.devig import (
    decimal_to_implied_prob, devig_two_way, devig_shin,
    build_consensus_true_prob, get_theoretical_hold,
)
from src.models.edge_ranker import (
    rank_edges,
    set_db as set_ranker_db,
    set_playoff_mode as set_ranker_playoff_mode,
)
from src.models.ml_model import get_ml_projection
from src.pipelines.send_alerts import evaluate_and_alert, send_game_market_alert, send_line_disagreement_alert
from src.pipelines.combos import generate_and_alert_combos, generate_slate_ultimate, generate_four_leg_parlays
from src.models.sgp_correlations import build_team_correlation_matrix
from src.clients.on_off_splits import OnOffSplitsClient
from src.clients.rotation_model import RotationModel
from src.clients.travel_fatigue import (
    compute_travel_fatigue, TEAM_NAME_TO_ABBR, ARENAS, haversine_miles, _HIGH_ALTITUDE_FT,
)
from src.clients.referee_client import fetch_today_assignments, match_event_refs
from src.models.referee_stats import get_crew_foul_factor
from src.clients.bdl_scan_integration import init_bdl_boost, get_bdl_boost
from src.clients.bdl_game_logs import BDLGameLogs
from src.clients.bdl_pbp_adapter import BDLPbpAdapter
from src.clients.bdl_standings_context import BDLStandingsContext
from src.clients.bdl_defense_context import BDLDefenseContext

logger = get_logger(__name__)
_PROJECTIONS_CACHE: Dict[str, Any] = {}

_SHARP_TS_BOOKS      = {'pinnacle', 'circa', 'bookmaker'}
_SOFT_TS_BOOKS       = {'draftkings', 'fanduel', 'betmgm', 'caesars'}
_STALE_THRESHOLD_SEC = 60
_SHARP_RECENT_SEC    = 120

# Game markets are highly efficient — require a stricter edge threshold.
_GAME_MARKET_EDGE_MIN = 0.015  # 1.5 %

# Toggle this to True during the playoffs to tighten rotations and disable regular-season rest logic
PLAYOFF_MODE = True

# ── Credit-conservation: events TTL cache ─────────────────────────────────────
# get_events() costs 1 credit. Between 90-min scans the event list is
# essentially static. Cache it for up to 45 minutes so consecutive scans
# (or breaking-news triggered scans) reuse the same fetch.
_EVENTS_CACHE: Dict[str, Any] = {'data': [], 'fetched_at': None}
_EVENTS_CACHE_TTL_SEC = 5400  # 90 minutes — matches scan interval

# Only fetch sharp Pinnacle odds when the game tips off within this window.
# Saves 1 Odds API credit per out-of-window game per scan.
_TIP_OFF_SHARP_HOURS = 4  # was 6 — Pinnacle barely moves 4-6h out


def _get_team_adv_row(df, team_name: str):
    """
    Extract OFF_RATING, DEF_RATING, PACE from the advanced-stats DataFrame
    for `team_name`.  Returns a dict or None if not found.
    """
    import pandas as pd
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    low = team_name.lower()
    row = df[df['TEAM_NAME'].str.lower() == low]
    if row.empty:
        last = low.split()[-1]
        row = df[df['TEAM_NAME'].str.lower().str.contains(last, na=False)]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        'off_rating': float(r.get('OFF_RATING', 110.0)),
        'def_rating': float(r.get('DEF_RATING', 110.0)),
        'pace':       float(r.get('PACE',       99.0)),
    }


def _team_fatigue_mult(team_abbr: str, home_arena_abbr: str) -> float:
    """
    Simplified travel-fatigue multiplier for team-level expected-score projection.

    Home team (team_abbr == home_arena_abbr) never incurs a penalty.
    Away team is penalised for distance, east-bound timezone shift, and altitude —
    using ½ the per-player B2B rate because the team isn't necessarily on a B2B.

    Returns a multiplier in [0.93, 1.0].
    """
    game_arena = ARENAS.get(home_arena_abbr.upper(), {})
    altitude   = game_arena.get('elev_ft', 0) >= _HIGH_ALTITUDE_FT

    if not team_abbr or team_abbr.upper() == home_arena_abbr.upper():
        return 1.0  # Home team: no travel penalty, altitude is their home turf

    team_arena = ARENAS.get(team_abbr.upper())
    if not team_arena:
        return 0.97 if altitude else 1.0

    miles    = haversine_miles(
        team_arena['lat'], team_arena['lon'],
        game_arena.get('lat', team_arena['lat']),
        game_arena.get('lon', team_arena['lon']),
    )
    tz_shift = game_arena.get('tz_offset', 0) - team_arena['tz_offset']

    if miles < 150:
        # Local road game (e.g. LAL at LAC) — treat as near-home
        return 0.97 if altitude else 1.0

    # Scale at ½ the B2B player rate: 1 % per 1 000 mi, 0.8 % per hour east
    penalty = (miles / 1000.0) * 0.01
    if tz_shift > 0:
        penalty += tz_shift * 0.008
    if altitude:
        penalty += 0.02

    return round(max(0.93, 1.0 - penalty), 4)


# ── Rest asymmetry ────────────────────────────────────────────────────────
# NBA fatigue is relative: a B2B team vs. a 3-day-rest team is far worse
# than two B2B teams facing each other.  The schedule delta drives edges
# the current travel-fatigue model misses entirely.

_REST_CACHE: Dict[str, int] = {}   # team_name → rest_days (per-scan)


def _get_team_rest_days(team_name: str) -> int:
    """
    Return days of rest for a team before tonight's game.
    Uses nba_api TeamGameLog (cached per scan to avoid repeated calls).
    0 = back-to-back, 1 = one day off, etc.  Default 2 if lookup fails.
    """
    if team_name in _REST_CACHE:
        return _REST_CACHE[team_name]

    rest = 2  # safe default
    try:
        import pandas as pd
        from nba_api.stats.endpoints import teamgamelog
        from nba_api.stats.static import teams as nba_teams

        # Resolve team ID from name
        matches = [t for t in nba_teams.get_teams()
                   if t['full_name'].lower() == team_name.lower()]
        if not matches:
            last = team_name.lower().split()[-1]
            matches = [t for t in nba_teams.get_teams()
                       if last in t['full_name'].lower()]
        if not matches:
            _REST_CACHE[team_name] = rest
            return rest

        time.sleep(0.6)
        tgl_rs = teamgamelog.TeamGameLog(
            team_id=matches[0]['id'],
            season_type_all_star='Regular Season',
        )
        df_rs = tgl_rs.get_data_frames()[0]

        time.sleep(0.6)
        tgl_po = teamgamelog.TeamGameLog(
            team_id=matches[0]['id'],
            season_type_all_star='Playoffs',
        )
        df_po = tgl_po.get_data_frames()[0]

        df = pd.concat([df_rs, df_po], ignore_index=True)
        if not df.empty and 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE', ascending=False)
            last_game = df.iloc[0]['GAME_DATE']
            today = pd.Timestamp.today().normalize()
            rest = max(0, int((today - last_game).days) - 1)
    except Exception:
        pass

    _REST_CACHE[team_name] = rest
    return rest


def compute_rest_asymmetry(
    home_team: str, away_team: str,
) -> Dict[str, Any]:
    """
    Compute the rest advantage for game-market and player-prop adjustments.

    Returns:
        home_rest:     int   — home team rest days
        away_rest:     int   — away team rest days
        rest_delta:    int   — home_rest - away_rest (positive = home more rested)
        home_off_mult: float — offensive rating multiplier for home team
        home_def_mult: float — defensive rating multiplier for home team
        away_off_mult: float — offensive rating multiplier for away team
        away_def_mult: float — defensive rating multiplier for away team
    """
    home_rest = _get_team_rest_days(home_team)
    away_rest = _get_team_rest_days(away_team)
    delta = home_rest - away_rest   # positive = home advantage

    # Base: no adjustment when rest is symmetric
    h_off, h_def = 1.0, 1.0
    a_off, a_def = 1.0, 1.0

    # Asymmetry tiers.  Each expected-score line absorbs ONE offensive
    # multiplier and ONE defensive multiplier (they compound), so keep
    # each factor modest.  Historical NBA B2B-vs-rested data shows a
    # ~3-5 point net swing.  With two ~110-pt teams:
    #   |delta| >= 3  →  ±2.5% each → ~5.4 pt swing per team, ~10.8 net
    #   |delta| == 2  →  ±1.5% each → ~3.3 pt swing per team, ~6.6 net
    #   |delta| == 1  →  ±0.7% each → ~1.5 pt swing per team, ~3.1 net
    abs_delta = abs(delta)

    if abs_delta >= 3:
        off_boost, def_degrade = 1.025, 1.025
    elif abs_delta == 2:
        off_boost, def_degrade = 1.015, 1.015
    elif abs_delta == 1:
        off_boost, def_degrade = 1.007, 1.007
    else:
        return {
            'home_rest': home_rest, 'away_rest': away_rest,
            'rest_delta': delta,
            'home_off_mult': 1.0, 'home_def_mult': 1.0,
            'away_off_mult': 1.0, 'away_def_mult': 1.0,
        }

    if delta > 0:
        # Home is more rested
        h_off = off_boost           # home offense sharper
        a_def = def_degrade         # away defense degraded (higher = worse)
        a_off = 1.0 / off_boost    # tired away offense sluggish
        h_def = 1.0 / def_degrade  # rested home defense tighter (lower = better)
    else:
        # Away is more rested — mirror
        a_off = off_boost
        h_def = def_degrade
        h_off = 1.0 / off_boost
        a_def = 1.0 / def_degrade

    return {
        'home_rest': home_rest, 'away_rest': away_rest,
        'rest_delta': delta,
        'home_off_mult': round(h_off, 4),
        'home_def_mult': round(h_def, 4),
        'away_off_mult': round(a_off, 4),
        'away_def_mult': round(a_def, 4),
    }


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

# BDL full potential modules
_bdl_booster = init_bdl_boost() if BDL_ENABLED else None
_bdl_game_logs = None
_bdl_pbp = None
_bdl_standings = None
_bdl_defense = None
if BDL_ENABLED_RUNTIME:
    _bdl_game_logs = BDLGameLogs(_bdl_bridge.bdl)
    _bdl_pbp = BDLPbpAdapter(_bdl_bridge.bdl)
    _bdl_standings = BDLStandingsContext(_bdl_bridge.bdl)
    _bdl_defense = BDLDefenseContext(_bdl_bridge.bdl)
    logger.info("BDL full potential: game logs, PBP, standings, defense context, booster initialized")


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


# ─── Sharp line disagreement detection ────────────────────────────────

_SHARP_LINE_BOOKS = {'pinnacle', 'circa', 'bookmaker'}
_SOFT_LINE_BOOKS  = {'draftkings', 'fanduel', 'betmgm', 'caesars',
                     'pointsbet', 'betrivers', 'unibet'}
_MIN_LINE_GAP     = 1.0   # full integer (e.g. 9.5 vs 10.5)


def _build_book_line_map(
    line_records: List[tuple],
    sharp_bookmakers: Optional[List[Dict]] = None,
) -> Dict[tuple, Dict[str, set]]:
    """
    Build {(player, market) → {book_lower: set_of_lines}} from all sources.

    Each book can offer multiple lines for the same player/market (main +
    alternates).  We collect all of them so the disagreement detector can
    identify the primary line per book.
    """
    book_lines: Dict[tuple, Dict[str, set]] = {}

    for rec in line_records:
        player, market, book, line_val = rec[0], rec[1], rec[2], rec[3]
        key = (player, market)
        book_lines.setdefault(key, {}).setdefault(book.lower(), set()).add(line_val)

    if sharp_bookmakers:
        for book in sharp_bookmakers:
            bname = book.get('title', '').lower()
            if bname not in _SHARP_LINE_BOOKS:
                continue
            for mkt in book.get('markets', []):
                mkt_key = mkt.get('key', '')
                for outcome in mkt.get('outcomes', []):
                    player = outcome.get('description')
                    line_val = outcome.get('point')
                    if player and line_val is not None:
                        key = (player, mkt_key)
                        book_lines.setdefault(key, {}).setdefault(bname, set()).add(line_val)

    return book_lines


def detect_line_disagreements(
    line_records: List[tuple],
    sharp_bookmakers: Optional[List[Dict]],
    home_team: str, away_team: str, game_date: str, event_id: str,
    db, bot,
):
    """
    Scan for cases where a sharp book's line differs from a soft book's by
    >= 1 full point.  Fire a priority alert for each disagreement.

    Example: Pinnacle has Rebounds @ 9.5, DraftKings has Rebounds @ 10.5
             → alert UNDER 10.5 on DraftKings.
    """
    book_lines = _build_book_line_map(line_records, sharp_bookmakers)
    alerted: set = set()

    for (player, market), books in book_lines.items():
        sharp_books = {b: lines for b, lines in books.items() if b in _SHARP_LINE_BOOKS}
        soft_books  = {b: lines for b, lines in books.items() if b in _SOFT_LINE_BOOKS}

        if not sharp_books or not soft_books:
            continue

        # Reference sharp line: use Pinnacle if available.
        # Take the line closest to the center of the market (most books
        # will converge here) — i.e. pick the line that also appears in
        # the most soft books.  If ambiguous, take the lowest (sharp
        # books move down first when adjusting).
        if 'pinnacle' in sharp_books:
            ref_sharp_book = 'pinnacle'
            ref_sharp_lines = sharp_books['pinnacle']
        else:
            ref_sharp_book = next(iter(sharp_books))
            ref_sharp_lines = sharp_books[ref_sharp_book]

        # If Pinnacle offers multiple lines (main + alts), use the one
        # that most soft books also offer — that's the "main" line.
        # Fallback: the minimum (sharps tend to shade low).
        all_soft_lines = set()
        for sl in soft_books.values():
            all_soft_lines.update(sl)

        shared = ref_sharp_lines & all_soft_lines
        ref_sharp_line = min(shared) if shared else min(ref_sharp_lines)

        for soft_book, soft_line_set in soft_books.items():
            for soft_line in soft_line_set:
                gap = soft_line - ref_sharp_line

                if abs(gap) < _MIN_LINE_GAP:
                    continue

                # If soft_line > sharp_line → UNDER on soft book
                # If soft_line < sharp_line → OVER on soft book
                side = "UNDER" if gap > 0 else "OVER"

                dedup_key = (player, market, soft_book, side)
                if dedup_key in alerted:
                    continue
                alerted.add(dedup_key)

                # Look up the best price on the soft book for this line+side
                soft_odds = 1.91  # default -110 if we can't find exact price
                for rec in line_records:
                    r_player, r_mkt, r_book, r_line, r_side, r_price, _ = rec
                    if (r_player == player and r_mkt == market
                            and r_book.lower() == soft_book
                            and abs(r_line - soft_line) < 0.01
                            and r_side.upper() == side
                            and r_price > soft_odds):
                        soft_odds = r_price

                send_line_disagreement_alert(
                    player=player, market=market,
                    sharp_book=ref_sharp_book.title(), sharp_line=ref_sharp_line,
                    soft_book=soft_book.title(), soft_line=soft_line,
                    side=side, soft_odds=soft_odds,
                    game_date=game_date, event_id=event_id,
                    home_team=home_team, away_team=away_team,
                    db=db, _bot=bot,
                )


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
    _REST_CACHE.clear()  # fresh team-rest lookups each scan cycle
    logger.info(f"Initializing scan pipeline ({'BDL+Sharp' if BDL_ENABLED_RUNTIME else 'Odds API only'})...")

    db             = DatabaseClient()
    odds_client    = OddsApiClient()
    stats_client   = NbaStatsClient()
    injury_client  = InjuryClient()
    bot            = TelegramBotClient()
    on_off_client  = OnOffSplitsClient()
    rotation_model = RotationModel(stats_client)

    set_ranker_db(db)
    set_ranker_playoff_mode(PLAYOFF_MODE)
    db.init_bookmaker_profiles()
    _book_weights = db.get_sharp_book_weights()

    # Inject DB into BDL modules for SQLite caching
    if _bdl_game_logs:
        _bdl_game_logs.db = db

    today      = datetime.now().strftime('%Y-%m-%d')
    local_zone = tz.tzlocal()

    # ── 1. Sync injuries ─────────────────────────────────────────────
    if BDL_ENABLED_RUNTIME:
        if not _sync_injuries_bdl(db, today):
            _sync_injuries_legacy(db, injury_client, today)
    else:
        _sync_injuries_legacy(db, injury_client, today)

    # ── 2. Get today's games ─────────────────────────────────────────
    odds_events = []
    odds_map = {}
    try:
        # Use cached event list if still fresh — saves 1 credit per scan.
        now_utc = datetime.now(timezone.utc)
        cache_age = (
            (now_utc - _EVENTS_CACHE['fetched_at']).total_seconds()
            if _EVENTS_CACHE['fetched_at'] else float('inf')
        )
        if cache_age < _EVENTS_CACHE_TTL_SEC and _EVENTS_CACHE['data']:
            events = _EVENTS_CACHE['data']
            logger.info(f"Events cache hit ({cache_age:.0f}s old) — skipping get_events() call.")
        else:
            events = odds_client.get_events()
            _EVENTS_CACHE['data'] = events
            _EVENTS_CACHE['fetched_at'] = now_utc
            logger.info(f"Events cache refreshed ({len(events)} events).")
        odds_events = [
            e for e in events
            if dateutil.parser.isoparse(e['commence_time'])
               .astimezone(local_zone).strftime('%Y-%m-%d') == today
        ]
        for e in events:
            odds_map[e['home_team'].lower()] = e['id']
            odds_map[e['away_team'].lower()] = e['id']
    except Exception as e:
        logger.error(f"Failed to fetch Odds API events: {e}")
        if not BDL_ENABLED_RUNTIME:
            return

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
        today_events = odds_events

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
            odds_eid    = odds_map.get(home_team.lower()) or odds_map.get(away_team.lower())
            event_id    = str(odds_eid) if odds_eid else str(bdl_game_id)
            commence_str = event.get("commence_time", "")
        else:
            bdl_game_id = None
            event_id    = event['id']
            home_team   = event['home_team']
            away_team   = event['away_team']
            commence_str = event.get('commence_time', '')

        # Pre-match only — skip any game that has already tipped off
        if commence_str:
            try:
                _cd = dateutil.parser.isoparse(commence_str)
                if _cd.tzinfo is None:
                    _cd = _cd.replace(tzinfo=timezone.utc)
                if _cd <= datetime.now(timezone.utc):
                    logger.info(f"Skipping in-progress game: {home_team} vs {away_team} (tipped at {_cd})")
                    continue
            except Exception:
                pass

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
            _all_line_records = bdl_data["line_records"]
            if _all_line_records:
                db.insert_line_history_batch(_all_line_records)

            # Fetch sharp books from Odds API (1 credit per game — Pinnacle only).
            # Skip if the game is more than _TIP_OFF_SHARP_HOURS away — sharp
            # lines barely move that far out, and this is the largest single
            # source of credit waste on morning/afternoon scans.
            sharp_bookmakers = []
            _tip_hours_away = float('inf')
            if commence_str:
                try:
                    _tip_dt = dateutil.parser.isoparse(commence_str).astimezone(local_zone)
                    _tip_hours_away = (
                        _tip_dt - datetime.now(local_zone)
                    ).total_seconds() / 3600
                except Exception:
                    pass
            if _tip_hours_away <= _TIP_OFF_SHARP_HOURS:
                try:
                    sharp_odds = odds_client.get_event_odds(
                        event_id=event_id,
                        markets=[*PROP_MARKETS, 'h2h', 'spreads', 'totals', 'team_totals']
                    )
                    sharp_bookmakers = sharp_odds.get('bookmakers', [])
                except Exception as _se:
                    logger.debug(f"Sharp book fetch skipped for {event_id}: {_se}")
            else:
                logger.info(
                    f"Skipping Odds API fetch for {home_team} vs {away_team} "
                    f"({_tip_hours_away:.1f}h to tip — outside {_TIP_OFF_SHARP_HOURS}h window)."
                )

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
                    markets=[*PROP_MARKETS, 'h2h', 'spreads', 'totals', 'team_totals']
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
            _all_line_records = []

            for mkt in PROP_MARKETS:
                prices_by_market[mkt] = {}
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
                                _all_line_records.append(
                                    (player, mkt, book.get('title'), line, side, price, 1.0 / price)
                                )
            if _all_line_records:
                db.insert_line_history_batch(_all_line_records)

        # ── Collect alt-lines from sharp bookmakers ───────────────────
        # Alt-lines are sourced from Pinnacle (already fetched above at 0 extra
        # credits).  We only process alt-lines for players already known to be
        # in the event — avoids pulling game-logs for players with no main-line prop.
        if sharp_bookmakers:
            for _alt_mkt in ALT_PROP_MARKETS:
                if _alt_mkt not in prices_by_market:
                    prices_by_market[_alt_mkt] = {}
                for _bk in sharp_bookmakers:
                    for _bk_mkt in _bk.get('markets', []):
                        if _bk_mkt.get('key') != _alt_mkt:
                            continue
                        for _oc in _bk_mkt.get('outcomes', []):
                            _p   = _oc.get('description')
                            _ln  = _oc.get('point')
                            if not _p or _ln is None:
                                continue
                            # Only attach alt-lines for players already in event
                            if _p not in players_in_event:
                                continue
                            prices_by_market[_alt_mkt].setdefault(_p, set()).add(float(_ln))

        # ── Sharp line disagreement scan ──────────────────────────────
        # Detect when Pinnacle/Circa have moved to a different line than
        # DraftKings/FanDuel etc. A full-integer gap is free money.
        if _all_line_records or sharp_bookmakers:
            try:
                detect_line_disagreements(
                    _all_line_records, sharp_bookmakers,
                    home_team, away_team, game_date, event_id,
                    db=db, bot=bot,
                )
            except Exception as _ld_err:
                logger.debug(f"Line disagreement scan failed: {_ld_err}")

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

        # ── Foul-trouble cascade signal ───────────────────────────────
        # When the opponent draws fouls at a top-5 rate (opp_fta_rate ≥ 1.08)
        # OR the crew has a tight whistle, the starting Center faces elevated
        # foul-trouble risk.  Cascade: starter C minutes × 0.85, backup C × 1.30.
        #
        # Compute per-side: a home player's opponent is the away team and v.v.
        _home_opp_ctx = stats_client.get_opponent_matchup_context(away_team, playoff_blend=PLAYOFF_MODE)
        _away_opp_ctx = stats_client.get_opponent_matchup_context(home_team, playoff_blend=PLAYOFF_MODE)
        _crew_foul_mult = _crew_factor.get("foul_rate_multiplier", 1.0)
        _tight_whistle  = _crew_factor.get("tight_whistle", False)

        # Combined foul-pressure signal: opponent foul-draw rate × crew whistle
        _home_foul_pressure = _home_opp_ctx.get('opp_fta_rate', 1.0) * _crew_foul_mult
        _away_foul_pressure = _away_opp_ctx.get('opp_fta_rate', 1.0) * _crew_foul_mult

        # Cascade activates when combined pressure ≥ 1.08 (top ~5 teams)
        # OR when tight whistle alone pushes it over threshold.
        _FOUL_CASCADE_THRESHOLD = 1.08
        _home_foul_cascade = _home_foul_pressure >= _FOUL_CASCADE_THRESHOLD
        _away_foul_cascade = _away_foul_pressure >= _FOUL_CASCADE_THRESHOLD

        if _home_foul_cascade or _away_foul_cascade:
            logger.info(
                f"Foul-cascade signal: "
                f"home={_home_foul_pressure:.3f} ({'ACTIVE' if _home_foul_cascade else 'off'}), "
                f"away={_away_foul_pressure:.3f} ({'ACTIVE' if _away_foul_cascade else 'off'}) "
                f"| crew_mult={_crew_foul_mult:.3f} tight={_tight_whistle} "
                f"| {home_team} vs {away_team}"
            )

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

        # ── Role-shift: identify OUT primary initiators ──────────────
        _out_initiator_ids: List[int] = []
        for _aid in absent_ids:
            if db.is_primary_initiator(_aid):
                _out_initiator_ids.append(_aid)
        _initiators_out = len(_out_initiator_ids)
        if _out_initiator_ids:
            logger.info(
                f"Role-shift active: {_initiators_out} primary initiator(s) OUT "
                f"(ids={_out_initiator_ids}) for {home_team} vs {away_team}"
            )

        # ── Pace data ─────────────────────────────────────────────────
        pace_info = stats_client.get_team_pace(home_team, away_team, playoff_blend=PLAYOFF_MODE)

        # ── Rest asymmetry ────────────────────────────────────────────
        _rest_asym = compute_rest_asymmetry(home_team, away_team)
        if _rest_asym['rest_delta'] != 0:
            logger.info(
                f"Rest asymmetry: {home_team} {_rest_asym['home_rest']}d vs "
                f"{away_team} {_rest_asym['away_rest']}d "
                f"(delta={_rest_asym['rest_delta']:+d})"
            )

        # BDL standings context (rest risk + blowout risk)
        _standings_ctx = {}
        if PLAYOFF_MODE:
            logger.debug("Playoff mode active: ignoring regular season standings blowout risk.")
        elif _bdl_standings:
            _standings_ctx = _bdl_standings.get_game_context(
                home_team, away_team, season=_season_int
            )

        # BDL full defense profiles (opponent DRTG, pace, position-specific factors)
        _home_def_profile = {}
        _away_def_profile = {}
        if _bdl_defense and _season_int:
            try:
                _home_def_profile = _bdl_defense.get_opponent_profile(
                    away_team, _season_int, db=db
                )
                _away_def_profile = _bdl_defense.get_opponent_profile(
                    home_team, _season_int, db=db
                )
                logger.debug(
                    f"BDL defense: home opp def_rating={_home_def_profile.get('def_rating', 1.0):.3f} "
                    f"away opp def_rating={_away_def_profile.get('def_rating', 1.0):.3f}"
                )
            except Exception as _def_err:
                logger.debug(f"BDL defense profile fetch failed: {_def_err}")

        sharp_alerted: set = set()

        # ── Game Markets Scan ─────────────────────────────────────────
        # Run before the player loop so team-level edges are reported
        # independently of prop availability.  Requires sharp book odds
        # (h2h + spreads + totals) from the Odds API.
        _home_exp = 0.0
        _away_exp = 0.0
        if sharp_bookmakers:
            _gm_home_abbr = TEAM_NAME_TO_ABBR.get(home_team, home_team.split()[-1][:3].upper())
            _gm_away_abbr = TEAM_NAME_TO_ABBR.get(away_team, away_team.split()[-1][:3].upper())
            _gm_matchup   = f"{away_team} @ {home_team}"
            try:
                _adv_df   = stats_client.get_team_stats()
                _home_row = _get_team_adv_row(_adv_df, home_team)
                _away_row = _get_team_adv_row(_adv_df, away_team)

                if _home_row and _away_row and not _adv_df.empty:
                    _lg_def   = float(_adv_df['DEF_RATING'].mean()) or 114.0
                    _gm_pace  = (_home_row['pace'] + _away_row['pace']) / 2.0

                    # Expected scores: pace-adjusted, opponent-defense-adjusted
                    _home_exp = (
                        _home_row['off_rating'] * (_away_row['def_rating'] / _lg_def)
                        * _gm_pace / 100.0
                    )
                    _away_exp = (
                        _away_row['off_rating'] * (_home_row['def_rating'] / _lg_def)
                        * _gm_pace / 100.0
                    )

                    # Sanity-clamp before applying fatigue
                    _home_exp = max(85.0, min(135.0, _home_exp))
                    _away_exp = max(85.0, min(135.0, _away_exp))

                    # Team travel-fatigue multipliers
                    _home_exp *= _team_fatigue_mult(_gm_home_abbr, _gm_home_abbr)
                    _away_exp *= _team_fatigue_mult(_gm_away_abbr, _gm_home_abbr)

                    # Rest asymmetry: rested team's offense sharpens,
                    # tired team's defense degrades.
                    _home_exp *= _rest_asym['home_off_mult'] * _rest_asym['away_def_mult']
                    _away_exp *= _rest_asym['away_off_mult'] * _rest_asym['home_def_mult']

                    _book_spread = _home_spread or 0.0
                    _book_total  = _game_total  or 0.0

                    _gm_probs = project_game_markets(
                        _home_exp, _away_exp, _book_spread, _book_total
                    )

                    # ── Moneyline (h2h) ───────────────────────────────
                    _h2h = OddsApiClient.extract_h2h_odds(
                        sharp_bookmakers, home_team, away_team
                    )
                    if _h2h:
                        _h2h_hp, _h2h_ap, _h2h_book = _h2h
                        _true_h, _true_a = devig_shin(
                            decimal_to_implied_prob(_h2h_hp),
                            decimal_to_implied_prob(_h2h_ap),
                        )
                        for _gm_model_p, _gm_book_p, _gm_odds, _gm_side in [
                            (_gm_probs['home_win'], _true_h, _h2h_hp, home_team),
                            (_gm_probs['away_win'], _true_a, _h2h_ap, away_team),
                        ]:
                            _gm_edge = _gm_model_p - _gm_book_p
                            if _gm_edge >= _GAME_MARKET_EDGE_MIN:
                                _gm_ev = _gm_model_p * (_gm_odds - 1) - (1 - _gm_model_p)
                                send_game_market_alert(
                                    home_team=home_team, away_team=away_team,
                                    home_score=_home_exp, away_score=_away_exp,
                                    market='h2h', side=_gm_side,
                                    edge=_gm_edge, ev=_gm_ev,
                                    model_prob=_gm_model_p, book_prob=_gm_book_p,
                                    book_odds=_gm_odds, book=_h2h_book,
                                    game_date=game_date, event_id=event_id,
                                    line=0.0, db=db, _bot=bot,
                                    home_abbr=_gm_home_abbr, away_abbr=_gm_away_abbr,
                                )

                    # ── Spread ────────────────────────────────────────
                    if _book_spread != 0.0:
                        _sp = OddsApiClient.extract_spread_odds_at_line(
                            sharp_bookmakers, home_team, _book_spread
                        )
                        if _sp:
                            _sp_hp, _sp_ap, _sp_book = _sp
                            _true_h, _true_a = devig_shin(
                                decimal_to_implied_prob(_sp_hp),
                                decimal_to_implied_prob(_sp_ap),
                            )
                            for _gm_model_p, _gm_book_p, _gm_odds, _gm_side in [
                                (_gm_probs['home_cover'], _true_h, _sp_hp,
                                 f"{home_team} {_book_spread:+.1f}"),
                                (_gm_probs['away_cover'], _true_a, _sp_ap,
                                 f"{away_team} {-_book_spread:+.1f}"),
                            ]:
                                _gm_edge = _gm_model_p - _gm_book_p
                                if _gm_edge >= _GAME_MARKET_EDGE_MIN:
                                    _gm_ev = _gm_model_p * (_gm_odds - 1) - (1 - _gm_model_p)
                                    send_game_market_alert(
                                        home_team=home_team, away_team=away_team,
                                        home_score=_home_exp, away_score=_away_exp,
                                        market='spreads', side=_gm_side,
                                        edge=_gm_edge, ev=_gm_ev,
                                        model_prob=_gm_model_p, book_prob=_gm_book_p,
                                        book_odds=_gm_odds, book=_sp_book,
                                        game_date=game_date, event_id=event_id,
                                        line=_book_spread, db=db, _bot=bot,
                                        home_abbr=_gm_home_abbr, away_abbr=_gm_away_abbr,
                                    )

                    # ── Total ─────────────────────────────────────────
                    if _book_total > 0.0:
                        _tot = OddsApiClient.extract_total_odds_at_line(
                            sharp_bookmakers, _book_total
                        )
                        if _tot:
                            _tot_op, _tot_up, _tot_book = _tot
                            _true_o, _true_u = devig_shin(
                                decimal_to_implied_prob(_tot_op),
                                decimal_to_implied_prob(_tot_up),
                            )
                            for _gm_model_p, _gm_book_p, _gm_odds, _gm_side in [
                                (_gm_probs['over'],  _true_o, _tot_op, 'Over'),
                                (_gm_probs['under'], _true_u, _tot_up, 'Under'),
                            ]:
                                _gm_edge = _gm_model_p - _gm_book_p
                                if _gm_edge >= _GAME_MARKET_EDGE_MIN:
                                    _gm_ev = _gm_model_p * (_gm_odds - 1) - (1 - _gm_model_p)
                                    send_game_market_alert(
                                        home_team=home_team, away_team=away_team,
                                        home_score=_home_exp, away_score=_away_exp,
                                        market='totals', side=_gm_side,
                                        edge=_gm_edge, ev=_gm_ev,
                                        model_prob=_gm_model_p, book_prob=_gm_book_p,
                                        book_odds=_gm_odds, book=_tot_book,
                                        game_date=game_date, event_id=event_id,
                                        line=_book_total, db=db, _bot=bot,
                                        home_abbr=_gm_home_abbr, away_abbr=_gm_away_abbr,
                                    )

            except Exception as _gm_err:
                logger.warning(f"Game markets scan failed for {_gm_matchup}: {_gm_err}")

        # ── Q1 Markets Scan ───────────────────────────────────────────
        # Uses the same expected-score math but scaled to 12 minutes (×0.25)
        # with tighter std devs — no blowout risk, no foul-trouble variance.
        # Q1 markets are fetched for free inside the existing get_event_odds call.
        if sharp_bookmakers and _home_exp > 0:
            try:
                _q1_home_exp = _home_exp * 0.25
                _q1_away_exp = _away_exp * 0.25

                # Extract Q1 consensus lines for spread/total anchor
                _q1_spread = OddsApiClient.extract_consensus_spread(
                    [b for b in sharp_bookmakers
                     if any(m.get('key') == 'spreads_q1'
                            for m in b.get('markets', []))],
                    home_team
                ) or 0.0
                _q1_total = OddsApiClient.extract_consensus_total(
                    [b for b in sharp_bookmakers
                     if any(m.get('key') == 'totals_q1'
                            for m in b.get('markets', []))]
                ) or 0.0

                if _q1_home_exp > 0 and _q1_away_exp > 0:
                    _q1_probs = project_q1_markets(
                        _q1_home_exp, _q1_away_exp, _q1_spread, _q1_total
                    )

                    # ── Q1 Moneyline ──────────────────────────────────
                    _q1_h2h = OddsApiClient.extract_q1_h2h_odds(
                        sharp_bookmakers, home_team, away_team
                    )
                    if _q1_h2h:
                        _q1_hp, _q1_ap, _q1_book = _q1_h2h
                        _q1_true_h, _q1_true_a = devig_shin(
                            decimal_to_implied_prob(_q1_hp),
                            decimal_to_implied_prob(_q1_ap),
                        )
                        for _q1_model_p, _q1_book_p, _q1_odds, _q1_side in [
                            (_q1_probs['home_win'], _q1_true_h, _q1_hp, home_team),
                            (_q1_probs['away_win'], _q1_true_a, _q1_ap, away_team),
                        ]:
                            _q1_edge = _q1_model_p - _q1_book_p
                            if _q1_edge >= _GAME_MARKET_EDGE_MIN:
                                _q1_ev = _q1_model_p * (_q1_odds - 1) - (1 - _q1_model_p)
                                send_game_market_alert(
                                    home_team=home_team, away_team=away_team,
                                    home_score=_q1_home_exp, away_score=_q1_away_exp,
                                    market='h2h_q1', side=_q1_side,
                                    edge=_q1_edge, ev=_q1_ev,
                                    model_prob=_q1_model_p, book_prob=_q1_book_p,
                                    book_odds=_q1_odds, book=_q1_book,
                                    game_date=game_date, event_id=event_id,
                                    line=0.0, db=db, _bot=bot,
                                    home_abbr=_gm_home_abbr, away_abbr=_gm_away_abbr,
                                )

                    # ── Q1 Spread ─────────────────────────────────────
                    if _q1_spread != 0.0:
                        _q1_sp = OddsApiClient.extract_q1_spread_odds_at_line(
                            sharp_bookmakers, home_team, _q1_spread
                        )
                        if _q1_sp:
                            _q1_sp_hp, _q1_sp_ap, _q1_sp_book = _q1_sp
                            _q1_true_h, _q1_true_a = devig_shin(
                                decimal_to_implied_prob(_q1_sp_hp),
                                decimal_to_implied_prob(_q1_sp_ap),
                            )
                            for _q1_model_p, _q1_book_p, _q1_odds, _q1_side in [
                                (_q1_probs['home_cover'], _q1_true_h, _q1_sp_hp,
                                 f"{home_team} Q1 {_q1_spread:+.1f}"),
                                (_q1_probs['away_cover'], _q1_true_a, _q1_sp_ap,
                                 f"{away_team} Q1 {-_q1_spread:+.1f}"),
                            ]:
                                _q1_edge = _q1_model_p - _q1_book_p
                                if _q1_edge >= _GAME_MARKET_EDGE_MIN:
                                    _q1_ev = _q1_model_p * (_q1_odds - 1) - (1 - _q1_model_p)
                                    send_game_market_alert(
                                        home_team=home_team, away_team=away_team,
                                        home_score=_q1_home_exp, away_score=_q1_away_exp,
                                        market='spreads_q1', side=_q1_side,
                                        edge=_q1_edge, ev=_q1_ev,
                                        model_prob=_q1_model_p, book_prob=_q1_book_p,
                                        book_odds=_q1_odds, book=_q1_sp_book,
                                        game_date=game_date, event_id=event_id,
                                        line=_q1_spread, db=db, _bot=bot,
                                        home_abbr=_gm_home_abbr, away_abbr=_gm_away_abbr,
                                    )

                    # ── Q1 Total ──────────────────────────────────────
                    if _q1_total > 0.0:
                        _q1_tot = OddsApiClient.extract_q1_total_odds_at_line(
                            sharp_bookmakers, _q1_total
                        )
                        if _q1_tot:
                            _q1_tot_op, _q1_tot_up, _q1_tot_book = _q1_tot
                            _q1_true_o, _q1_true_u = devig_shin(
                                decimal_to_implied_prob(_q1_tot_op),
                                decimal_to_implied_prob(_q1_tot_up),
                            )
                            for _q1_model_p, _q1_book_p, _q1_odds, _q1_side in [
                                (_q1_probs['over'],  _q1_true_o, _q1_tot_op, 'Q1 Over'),
                                (_q1_probs['under'], _q1_true_u, _q1_tot_up, 'Q1 Under'),
                            ]:
                                _q1_edge = _q1_model_p - _q1_book_p
                                if _q1_edge >= _GAME_MARKET_EDGE_MIN:
                                    _q1_ev = _q1_model_p * (_q1_odds - 1) - (1 - _q1_model_p)
                                    send_game_market_alert(
                                        home_team=home_team, away_team=away_team,
                                        home_score=_q1_home_exp, away_score=_q1_away_exp,
                                        market='totals_q1', side=_q1_side,
                                        edge=_q1_edge, ev=_q1_ev,
                                        model_prob=_q1_model_p, book_prob=_q1_book_p,
                                        book_odds=_q1_odds, book=_q1_tot_book,
                                        game_date=game_date, event_id=event_id,
                                        line=_q1_total, db=db, _bot=bot,
                                        home_abbr=_gm_home_abbr, away_abbr=_gm_away_abbr,
                                    )

            except Exception as _q1_err:
                logger.warning(f"Q1 markets scan failed for {_gm_matchup}: {_q1_err}")

        # ── Team Totals Scan ──────────────────────────────────────────
        # Isolates each team's expected score independently — removes the
        # opponent's offensive variance that contaminates game total bets.
        # Team total std dev (~8.5 pts) is much tighter than game total (14.0).
        if sharp_bookmakers and _home_exp > 0:
            try:
                for _tt_team, _tt_exp in (
                    (home_team, _home_exp),
                    (away_team, _away_exp),
                ):
                    _tt_line = OddsApiClient.extract_consensus_team_total(
                        sharp_bookmakers, _tt_team
                    ) or 0.0
                    if not _tt_line:
                        continue

                    _tt_probs = project_team_totals(_tt_exp, _tt_line)
                    _tt = OddsApiClient.extract_team_total_odds(
                        sharp_bookmakers, _tt_team, _tt_line
                    )
                    if not _tt:
                        continue

                    _tt_op, _tt_up, _tt_book = _tt
                    _tt_true_o, _tt_true_u = devig_shin(
                        decimal_to_implied_prob(_tt_op),
                        decimal_to_implied_prob(_tt_up),
                    )
                    for _tt_model_p, _tt_book_p, _tt_odds, _tt_side in [
                        (_tt_probs['over'],  _tt_true_o, _tt_op,
                         f"{_tt_team} Over {_tt_line}"),
                        (_tt_probs['under'], _tt_true_u, _tt_up,
                         f"{_tt_team} Under {_tt_line}"),
                    ]:
                        _tt_edge = _tt_model_p - _tt_book_p
                        if _tt_edge >= _GAME_MARKET_EDGE_MIN:
                            _tt_ev = _tt_model_p * (_tt_odds - 1) - (1 - _tt_model_p)
                            send_game_market_alert(
                                home_team=home_team, away_team=away_team,
                                home_score=_home_exp, away_score=_away_exp,
                                market='team_totals', side=_tt_side,
                                edge=_tt_edge, ev=_tt_ev,
                                model_prob=_tt_model_p, book_prob=_tt_book_p,
                                book_odds=_tt_odds, book=_tt_book,
                                game_date=game_date, event_id=event_id,
                                line=_tt_line, db=db, _bot=bot,
                                home_abbr=_gm_home_abbr, away_abbr=_gm_away_abbr,
                            )
            except Exception as _tt_err:
                logger.warning(f"Team totals scan failed for {_gm_matchup}: {_tt_err}")

        # ── Per-player loop ───────────────────────────────────────────
        team_player_logs: Dict[str, Dict[str, pd.DataFrame]] = {}
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

            # Fetch game logs (BDL primary → nba_api fallback)
            cache_key = f"{player_name}_{today}"
            _bdl_pid = bdl_player_map.get(player_name) if BDL_ENABLED_RUNTIME else None
            if cache_key not in _PROJECTIONS_CACHE:

                # Try BDL first (no sleep needed, 600 req/min)
                if _bdl_game_logs and _bdl_pid:
                    try:
                        logs = _bdl_game_logs.get_player_game_logs(
                            bdl_player_id=_bdl_pid, season=_season_int
                        )
                        if not logs.empty and len(logs) >= 5:
                            _PROJECTIONS_CACHE[cache_key] = {"logs": logs, "pid": _bdl_pid}
                        else:
                            _bdl_pid = None  # Fall through to nba_api
                    except Exception:
                        _bdl_pid = None

                # Fallback to nba_api
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
            _pt_team = home_team if home_flag else away_team
            team_player_logs.setdefault(_pt_team, {})[player_name] = logs

            # Standings-based minutes adjustment
            _standings_min_adj = 1.0
            if _standings_ctx:
                _standings_min_adj = (
                    _standings_ctx.get('minutes_adj_home', 1.0) if home_flag
                    else _standings_ctx.get('minutes_adj_away', 1.0)
                )

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
            _opp_ctx = stats_client.get_opponent_matchup_context(opp_team, playoff_blend=PLAYOFF_MODE)
            _player_foul_rate = compute_player_foul_rate(logs)

            # Revenge game: player previously played for tonight's opponent
            _revenge_game = False
            if not logs.empty and 'TEAM_ABBREVIATION' in logs.columns:
                _opp_abbr = TEAM_NAME_TO_ABBR.get(opp_team, '')
                if _opp_abbr:
                    _current_abbr = str(logs.iloc[0]['TEAM_ABBREVIATION']).upper()
                    _all_abbrs = {str(a).upper() for a in logs['TEAM_ABBREVIATION'].unique()}
                    _revenge_game = _opp_abbr.upper() in _all_abbrs and _opp_abbr.upper() != _current_abbr

            # Next-opponent win%: opponent's season win% from standings context
            _next_opp_win_pct = 0.0
            if _standings_ctx:
                _next_opp_win_pct = (
                    _standings_ctx.get('away_win_pct', 0.0) if home_flag
                    else _standings_ctx.get('home_win_pct', 0.0)
                )

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

            # BDL season profile: usage%, touches, playtype frequencies, ts_pct.
            # avg_distance > 3.0 mi/game signals on-court exertion beyond travel fatigue.
            _adj_fatigue_mult = _fatigue['fatigue_multiplier']
            _bdl_profile: Dict[str, float] = {}
            if BDL_ENABLED_RUNTIME and player_name in bdl_player_map:
                _prof_ck = f"bdl_profile_{bdl_player_map[player_name]}"
                if _prof_ck not in _PROJECTIONS_CACHE:
                    _PROJECTIONS_CACHE[_prof_ck] = _bdl_bridge.get_player_season_profile(
                        bdl_player_map[player_name], season=_season_int
                    )
                _bdl_profile = _PROJECTIONS_CACHE[_prof_ck]
                if _bdl_profile.get("avg_distance", 0.0) > 3.0:
                    _adj_fatigue_mult *= 0.98

            # BDL PBP shot quality profile (foul-draw rate, paint shot %, assist rate)
            _pbp_profile: Dict[str, float] = {}
            if _bdl_pbp and _bdl_game_logs and _bdl_pid:
                try:
                    _recent_gids = _bdl_game_logs.get_recent_game_ids(
                        _bdl_pid, season=_season_int, n=3
                    )
                    if _recent_gids:
                        _pbp_profile = _bdl_pbp.get_player_shot_profile(
                            _bdl_pid, _recent_gids
                        )
                except Exception:
                    pass

            for mkt in [*PROP_MARKETS, *ALT_PROP_MARKETS]:
                if player_name not in prices_by_market.get(mkt, {}):
                    continue
                # base_mkt: used for all stat/model calls (strips "_alternate" suffix)
                base_mkt = ALT_TO_BASE_MARKET.get(mkt, mkt)

                opp_multiplier = stats_client.get_positional_def_multiplier(
                    opp_team, base_mkt, _position_group, playoff_blend=PLAYOFF_MODE
                )

                # BDL defense context: position-specific DRTG-weighted factor.
                # Blend 50/50 with nba_api multiplier when available — BDL
                # provides richer opponent/advanced/defense dimensions.
                _def_profile = _home_def_profile if home_flag else _away_def_profile
                if _def_profile:
                    _bdl_def_factor = _bdl_defense.get_position_def_factor(
                        _def_profile, _position_group, base_mkt
                    )
                    opp_multiplier = 0.50 * opp_multiplier + 0.50 * _bdl_def_factor

                for line in prices_by_market[mkt][player_name]:
                    # On/off usage shift
                    if absent_ids and player_id_int:
                        shifts = [
                            on_off_client.get_usage_multiplier(
                                target_player_id=int(player_id_int),
                                absent_player_id=_aid, market=base_mkt, db=db,
                            ) - 1.0
                            for _aid in absent_ids
                        ]
                        _usage_shift = min(sum(shifts), 0.50)
                    else:
                        _usage_shift = 0.0

                    # ── Role-shift override ──────────────────────────
                    # When a primary initiator is OUT, look up the
                    # player's isolated rate_without from on_off_splits
                    # and hard-override the Bayesian baseline.
                    _role_shift_rate = 0.0
                    if _out_initiator_ids and player_id_int:
                        for _init_id in _out_initiator_ids:
                            _rw = db.get_on_off_rate_without(
                                int(player_id_int), _init_id,
                                base_mkt, stats_client.season,
                            )
                            if _rw is not None and _rw > 0:
                                # Use the highest rate_without if multiple
                                # initiators are out (most impactful split)
                                _role_shift_rate = max(_role_shift_rate, _rw)
                        if _role_shift_rate > 0:
                            logger.info(
                                f"Role-shift: {player_name} {base_mkt} "
                                f"rate_without={_role_shift_rate:.4f} "
                                f"(initiators_out={_initiators_out})"
                            )

                    _opp_abbr_proj = TEAM_NAME_TO_ABBR.get(opp_team, opp_team.split()[-1][:3].upper())
                    proj = build_player_projection(
                        player_id=player_name, market=base_mkt, line=line,
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
                        role_shift_rate=_role_shift_rate,
                        initiators_out=_initiators_out,
                        playoff_mode=PLAYOFF_MODE,
                        opp_abbr=_opp_abbr_proj,
                    )

                    if not proj or proj.get('mean', 0) == 0:
                        continue

                    if proj.get('projected_minutes', 0) < 10:
                        continue

                    # ── Playoff Rotation Adjustments ──────────────────────────
                    if PLAYOFF_MODE:
                        if starter_flag:
                            # Starters play heavier minutes in the playoffs (boost by ~10%, cap at 43)
                            proj['projected_minutes'] = min(43.0, proj['projected_minutes'] * 1.10)
                            proj['mean'] *= 1.10
                            # Minutes are more predictable for starters, reducing variance
                            proj['variance_scale'] = proj.get('variance_scale', 1.0) * 0.85
                        elif proj['projected_minutes'] < 16.0:
                            # Deep bench gets squeezed out of the rotation
                            proj['projected_minutes'] *= 0.50
                            proj['mean'] *= 0.50
                            # High volatility for deep bench (dependent on garbage time/fouls)
                            proj['variance_scale'] = proj.get('variance_scale', 1.0) * 1.25
                        else:
                            # Regular bench rotation also sees increased volatility in playoffs
                            proj['variance_scale'] = proj.get('variance_scale', 1.0) * 1.15

                    # XGBoost blend
                    _league_avg_pace = pace_info.get('league_avg', 99.0)
                    if PLAYOFF_MODE:
                        # Playoff pace runs ~3% slower than regular season.
                        _league_avg_pace = min(_league_avg_pace, 96.0)
                    _pace_factor = (
                        (pace_info['home_pace'] + pace_info['away_pace'])
                        / (2.0 * _league_avg_pace)
                    )
                    _opp_abbr_ml = TEAM_NAME_TO_ABBR.get(opp_team, opp_team.split()[-1][:3].upper())
                    ml_mean = get_ml_projection(
                        base_mkt, logs, proj['projected_minutes'], home_flag, rest_days,
                            playoff_flag=PLAYOFF_MODE,
                        opp_abbr=_opp_abbr_ml,
                        opp_def_rating=opp_multiplier,
                        pace_factor=_pace_factor,
                        opp_pace=_opp_ctx['opp_pace'],
                        opp_rebound_pct=_opp_ctx['opp_rebound_pct'],
                        opp_pts_paint=_opp_ctx['opp_pts_paint'],
                        travel_miles=_fatigue['miles_traveled'],
                        tz_shift_hours=_fatigue['tz_shift_hours'],
                        altitude_flag=_fatigue['altitude_flag'],
                        real_usage_pct=_bdl_profile.get('real_usage_pct', 0.0),
                        avg_touches=_bdl_profile.get('avg_touches', 0.0),
                        pnr_bh_freq=_bdl_profile.get('pnr_bh_freq', 0.0),
                        pnr_roll_freq=_bdl_profile.get('pnr_roll_freq', 0.0),
                        iso_freq=_bdl_profile.get('iso_freq', 0.0),
                        spotup_freq=_bdl_profile.get('spotup_freq', 0.0),
                        transition_freq=_bdl_profile.get('transition_freq', 0.0),
                        postup_freq=_bdl_profile.get('postup_freq', 0.0),
                        drives_per_game=_bdl_profile.get('drives_per_game', 0.0),
                        ts_pct=_bdl_profile.get('ts_pct', 0.0),
                        avg_speed=_bdl_profile.get('avg_speed', 0.0),
                        avg_contested_fg_pct=_bdl_profile.get('avg_contested_fg_pct', 0.0),
                        avg_deflections=_bdl_profile.get('avg_deflections', 0.0),
                        avg_points_paint=_bdl_profile.get('avg_points_paint', 0.0),
                        avg_pct_pts_paint=_bdl_profile.get('avg_pct_pts_paint', 0.0),
                        player_foul_rate=_player_foul_rate,
                    )
                    if ml_mean is not None and ml_mean > 0:
                        proj['mean'] = 0.5 * proj['mean'] + 0.5 * ml_mean
                        proj['ml_blend'] = True

                    # BDL playtype × opponent boost
                    if _bdl_booster and player_name in bdl_player_map:
                        _bdl_mult = get_bdl_boost(
                            _bdl_booster,
                            bdl_player_id=bdl_player_map[player_name],
                            opponent_team=opp_team,
                            market=base_mkt,
                            season=_season_int,
                            player_profile=_bdl_profile,
                        )
                        if _bdl_mult != 1.0:
                            proj['mean'] *= _bdl_mult
                            proj['bdl_boost'] = _bdl_mult
                            logger.debug(
                                f"BDL boost: {player_name} {base_mkt} × {_bdl_mult:.3f} "
                                f"vs {opp_team}"
                            )

                    # BDL PBP foul-draw boost: high foul-draw rate → more FTAs → more pts
                    if _pbp_profile and base_mkt in {
                        'player_points', 'player_points_rebounds_assists'
                    }:
                        _fdr = _pbp_profile.get('foul_draw_rate', 0.0)
                        if _fdr >= 0.30:
                            # Every 0.10 above 0.30 threshold → ~1% points boost (capped at 4%)
                            _pbp_mult = 1.0 + min((_fdr - 0.30) / 0.10 * 0.01, 0.04)
                            proj['mean'] *= _pbp_mult
                            proj['pbp_fdr_boost'] = round(_pbp_mult, 4)

                    # Standings minutes adjustment
                    if _standings_min_adj != 1.0:
                        proj['projected_minutes'] *= _standings_min_adj

                    # Tight whistle
                    if _crew_factor["tight_whistle"] and base_mkt in {
                        "player_points", "player_points_rebounds_assists"
                    }:
                        proj['mean'] *= 1.02

                    # Altitude fatigue
                    if (b2b_flag and _fatigue['altitude_flag']
                        and _fatigue['miles_traveled'] > 800 and not home_flag):
                        if base_mkt == 'player_threes':
                            proj['mean'] *= 0.95
                        elif base_mkt == 'player_rebounds':
                            proj['mean'] *= 0.97

                    # Rest asymmetry: player-level adjustment.
                    # The rested team's players get an offensive boost;
                    # the tired team's players get a penalty — this is the
                    # schedule delta the market consistently underprices.
                    if _rest_asym['rest_delta'] != 0:
                        _ra_off = (_rest_asym['home_off_mult']
                                   if home_flag else _rest_asym['away_off_mult'])
                        if _ra_off != 1.0:
                            proj['mean'] *= _ra_off
                            proj['rest_asymmetry'] = _ra_off

                    # ── Foul-trouble cascade ──────────────────────────
                    # When opponent draws fouls at a top-5 rate + tight
                    # whistle, Centers face predictable foul trouble.
                    # Slash starter C minutes, boost backup C minutes.
                    _foul_cascade_active = (
                        _home_foul_cascade if home_flag else _away_foul_cascade
                    )
                    if _foul_cascade_active and _position_group == 'Center':
                        _old_mins = proj['projected_minutes']
                        if starter_flag:
                            # Starter C: expected to sit ~5 min with
                            # early fouls → 0.85x minutes
                            proj['projected_minutes'] *= 0.85
                            proj['mean'] *= 0.85
                            proj['foul_cascade'] = 'starter_C_cut'
                        else:
                            # Backup C: absorbs the starter's lost
                            # minutes → 1.30x minutes
                            proj['projected_minutes'] *= 1.30
                            proj['mean'] *= 1.30
                            proj['foul_cascade'] = 'backup_C_boost'
                        logger.info(
                            f"Foul-cascade: {player_name} ({proj['foul_cascade']}) "
                            f"mins {_old_mins:.1f} → {proj['projected_minutes']:.1f}"
                        )

                    _bench_tier = classify_bench_tier(proj['projected_minutes'])

                    dists = get_probability_distribution(
                        base_mkt, proj['mean'], line, logs=logs,
                        variance_scale=proj.get('variance_scale', 1.0),
                        proj_minutes=proj['projected_minutes'],
                        spread=_home_spread or 0.0,
                        total=_game_total or 0.0,
                        player_foul_rate=_player_foul_rate,
                        opp_foul_rate=_opp_ctx['opp_fta_rate'] * _crew_factor["foul_rate_multiplier"],
                        bench_tier=_bench_tier,
                        next_opp_win_pct=_next_opp_win_pct,
                        revenge_game=_revenge_game,
                        playoff_mode=PLAYOFF_MODE,
                        must_win=bool(proj.get('must_win', False)),
                        closeout_game=bool(proj.get('closeout_opportunity', False)),
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

                    # Sharp line shift: detect if sharp books moved this line
                    _sharp_shift = db.get_sharp_line_shift(player_name, base_mkt)

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
                        "sharp_line_shift": _sharp_shift,
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
                            _sharp_edge_min = PLAYOFF_SHARP_EDGE_MIN if PLAYOFF_MODE else SHARP_EDGE_MIN
                            if sharp_gap < _sharp_edge_min:
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

        # ── Build Full Team Correlation Matrix for SGPs ───────────────
        for _t_name, _t_logs in team_player_logs.items():
            try:
                from src.models.sgp_correlations import build_full_team_correlation_matrix
                build_full_team_correlation_matrix(_t_name, _t_logs, db)
            except Exception as e:
                logger.warning(f"Failed to build full team correlation matrix for {_t_name}: {e}")

        # ── Cross-Team Correlations (big-vs-big, empirical in playoffs) ──
        try:
            from src.models.sgp_correlations import build_cross_team_correlation_matrix
            _away_abbr_corr = TEAM_NAME_TO_ABBR.get(away_team, '')
            build_cross_team_correlation_matrix(
                home_team, away_team, stats_client, db,
                home_player_logs=team_player_logs.get(home_team),
                away_player_logs=team_player_logs.get(away_team),
                away_abbr=_away_abbr_corr,
                playoff_mode=PLAYOFF_MODE,
            )
        except Exception as e:
            logger.warning(f"Failed to build cross-team correlation matrix for {home_team} vs {away_team}: {e}")

    # ── Rank and alert ────────────────────────────────────────────────
    ranked_edges = rank_edges(candidates)
    _edge_min_floor = PLAYOFF_EDGE_MIN if PLAYOFF_MODE else EDGE_MIN
    actionable = [e for e in ranked_edges if e.get('edge', 0) >= e.get('edge_min_applied', _edge_min_floor)]

    logger.info(
        f"Scan complete: {len(candidates)} candidates, {len(actionable)} actionable edges. "
        f"{'[BDL+Sharp mode]' if BDL_ENABLED_RUNTIME else '[Odds API mode]'}"
    )

    for edge in actionable:
        evaluate_and_alert(edge, db, bot)

    generate_and_alert_combos(actionable, bot, db=db, playoff_mode=PLAYOFF_MODE)
    generate_four_leg_parlays(actionable, bot, db=db, playoff_mode=PLAYOFF_MODE)
    generate_slate_ultimate(actionable, bot, db=db, playoff_mode=PLAYOFF_MODE)

    if BDL_ENABLED_RUNTIME and _bdl_bridge:
        logger.info(f"BDL requests this scan: {_bdl_bridge.bdl.requests_made}")


if __name__ == "__main__":
    scan_props()
