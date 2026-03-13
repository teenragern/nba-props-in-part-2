"""
Main edge-detection pipeline.

Wires up all Priority 1-6 improvements:
  P1  Real injury feed  → populates injury_reports table
  P2  Opponent defensive multiplier per market
  P3  game_date stored on each alert for correct settlement
  P4  Home/away + rest day features passed to projections
  P5  Starter flag inferred from average minutes
  P6  XGBoost projection blended 50/50 with Bayesian mean
  P7  Per-book bias feeds through edge_ranker.set_db()
"""

import time
from datetime import datetime
import dateutil.parser
from dateutil import tz
from typing import List, Dict, Any, Tuple, Optional

from src.utils.logging_utils import get_logger
from src.data.db import DatabaseClient
from src.clients.odds_api import OddsApiClient
from src.clients.nba_stats import NbaStatsClient
from src.clients.injuries import InjuryClient
from src.clients.telegram_bot import TelegramBotClient
from src.config import PROP_MARKETS, EDGE_MIN, SHARP_BOOKS, REC_BOOKS, SHARP_EDGE_MIN
from src.models.projections import build_player_projection
from src.models.distributions import get_probability_distribution
from src.models.devig import decimal_to_implied_prob, devig_two_way
from src.models.edge_ranker import rank_edges, set_db as set_ranker_db
from src.models.ml_model import get_ml_projection
from src.pipelines.send_alerts import evaluate_and_alert
from src.pipelines.combos import generate_and_alert_combos
from src.clients.on_off_splits import OnOffSplitsClient

logger = get_logger(__name__)
_PROJECTIONS_CACHE: Dict[str, Any] = {}


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
    """Convert decimal odds to American odds string."""
    if decimal_odds >= 2.0:
        return f"+{int((decimal_odds - 1) * 100)}"
    if decimal_odds <= 1.0:
        return "N/A"
    return f"-{int(100 / (decimal_odds - 1))}"


def get_sharp_true_prob(
    bookmakers: List[Dict], player_name: str, market_key: str, line: float
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Find the first sharp-book (e.g. Pinnacle) price for this player/market/line,
    devig it, and return (true_over_prob, true_under_prob, book_name).
    Returns (None, None, None) when no sharp book carries this line.
    """
    sharp_lower = [b.lower() for b in SHARP_BOOKS]
    for book in bookmakers:
        if book.get('title', '').lower() not in sharp_lower:
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
            true_o, true_u = devig_two_way(raw_o, raw_u)
            return true_o, true_u, book['title']
    return None, None, None


def get_best_rec_price(
    bookmakers: List[Dict], player_name: str,
    market_key: str, line: float, side: str
) -> Tuple[Optional[float], Optional[str]]:
    """
    Return (best_decimal_price, book_name) for side across REC_BOOKS.
    Returns (None, None) when no rec book carries this line.
    """
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


def _sync_injuries(db: DatabaseClient, injury_client: InjuryClient, game_date: str):
    """Priority 1: Fetch and persist today's injury report."""
    injuries = injury_client.get_injuries()
    if not injuries:
        logger.warning("No injury data fetched — all players will be treated as Healthy.")
        return

    with db.get_conn() as conn:
        cursor = conn.cursor()
        for inj in injuries:
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO injury_reports (game_date, player_name, team, status)
                    VALUES (?, ?, ?, ?)
                    """,
                    (game_date, inj['player_name'], inj['team'], inj['status'])
                )
            except Exception:
                continue
    logger.info(f"Persisted {len(injuries)} injury records for {game_date}.")


def scan_props():
    logger.info("Initializing scan pipeline (P1–P7 active)...")
    db             = DatabaseClient()
    odds_client    = OddsApiClient()
    stats_client   = NbaStatsClient()
    injury_client  = InjuryClient()
    bot            = TelegramBotClient()
    on_off_client  = OnOffSplitsClient()

    # Priority 7: give edge_ranker access to DB for per-book bias
    set_ranker_db(db)
    db.init_bookmaker_profiles()

    today      = datetime.now().strftime('%Y-%m-%d')
    local_zone = tz.tzlocal()

    # Priority 1: sync injuries at scan start
    _sync_injuries(db, injury_client, today)

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

    candidates = []

    for event in today_events:
        event_id   = event['id']
        home_team  = event['home_team']
        away_team  = event['away_team']
        game_date  = today

        try:
            odds_data = odds_client.get_event_odds(event_id=event_id, markets=PROP_MARKETS)
        except Exception:
            continue

        bookmakers = odds_data.get('bookmakers', [])
        if not bookmakers:
            continue

        players_in_event: set = set()
        prices_by_market: Dict[str, Dict] = {}

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

        # Identify OUT players from injury table
        out_players: List[str] = []
        with db.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT player_name FROM injury_reports
                WHERE game_date = ? AND status = 'Out'
                AND (team = ? OR team = ?)
                """,
                (today, home_team, away_team)
            )
            out_players = [r['player_name'] for r in cursor.fetchall()]

        # Resolve absent player IDs once per event for on/off lookup
        absent_ids: List[int] = []
        if out_players:
            for _nm in out_players:
                _aid = NbaStatsClient.resolve_player_id(_nm)
                if _aid:
                    absent_ids.append(_aid)
            logger.info(f"{away_team} @ {home_team}: OUT players {out_players}. Exact on/off splits active.")

        # Priority 2: fetch pace data for this matchup once per event
        pace_info = stats_client.get_team_pace(home_team, away_team)

        sharp_alerted: set = set()   # dedup sharp alerts within this event scan

        for player_name in players_in_event:
            # ---- injury status ----
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

            # ---- fetch / cache player game logs ----
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

            p_data    = _PROJECTIONS_CACHE[cache_key]
            logs      = p_data["logs"]

            # ---- Priority 4: home/away + rest days ----
            player_team_abbr = logs.iloc[0]['TEAM_ABBREVIATION'] if not logs.empty else None
            home_flag  = stats_client.is_home_team(player_team_abbr or '', home_team)
            rest_days  = NbaStatsClient.calculate_rest_days(logs)
            b2b_flag   = rest_days == 0

            # ---- Priority 5: starter flag from minutes ----
            starter_flag = NbaStatsClient.infer_starter_flag(logs)

            # ---- determine opponent team for defensive multiplier ----
            opp_team = away_team if home_flag else home_team

            for mkt in PROP_MARKETS:
                if player_name not in prices_by_market[mkt]:
                    continue

                # Priority 2: get market-specific opponent def multiplier
                if mkt == 'player_points_rebounds_assists':
                    opp_multiplier = stats_client.get_opponent_def_multiplier_pra(opp_team)
                else:
                    opp_multiplier = stats_client.get_opponent_def_multiplier(opp_team, mkt)

                for line in prices_by_market[mkt][player_name]:

                    # Exact per-market on/off usage shift (replaces flat 15%)
                    player_id_int = p_data.get("pid")
                    if absent_ids and player_id_int:
                        shifts = [
                            on_off_client.get_usage_multiplier(
                                target_player_id=int(player_id_int),
                                absent_player_id=_aid,
                                market=mkt,
                                db=db,
                            ) - 1.0
                            for _aid in absent_ids
                        ]
                        _usage_shift = min(sum(shifts), 0.50)
                    else:
                        _usage_shift = 0.0

                    proj = build_player_projection(
                        player_id=player_name,
                        market=mkt,
                        line=line,
                        recent_logs=logs,
                        season_logs=logs,
                        injury_status=injury_status,
                        team_pace=pace_info['home_pace'] if home_flag else pace_info['away_pace'],
                        opp_pace=pace_info['away_pace'] if home_flag else pace_info['home_pace'],
                        opponent_multiplier=opp_multiplier,
                        usage_shift=_usage_shift,
                        starter_flag=starter_flag,
                        b2b_flag=b2b_flag,
                        home_flag=home_flag,
                        rest_days=rest_days,
                    )

                    if not proj or proj.get('mean', 0) == 0:
                        continue

                    # Priority 6: blend Bayesian mean with XGBoost projection
                    ml_mean = get_ml_projection(
                        mkt, logs, proj['projected_minutes'], home_flag, rest_days
                    )
                    if ml_mean is not None and ml_mean > 0:
                        proj['mean'] = 0.5 * proj['mean'] + 0.5 * ml_mean
                        proj['ml_blend'] = True

                    dists = get_probability_distribution(
                        mkt, proj['mean'], line,
                        logs=logs,
                        variance_scale=proj.get('variance_scale', 1.0)
                    )

                    best_over, best_under = get_best_odds(bookmakers, player_name, mkt, line)

                    if best_over['price'] > 0 and best_under['price'] > 0:
                        raw_imp_o = decimal_to_implied_prob(best_over['price'])
                        raw_imp_u = decimal_to_implied_prob(best_under['price'])
                        imp_over, _ = devig_two_way(raw_imp_o, raw_imp_u)
                    else:
                        imp_over  = decimal_to_implied_prob(best_over['price'])
                        imp_under = decimal_to_implied_prob(best_under['price'])

                    common = {
                        **proj,
                        "home_team": home_team,
                        "away_team": away_team,
                        "game_date": game_date,
                        "event_id":  event_id,
                        "home_away": "HOME" if home_flag else "AWAY",
                        "rest_days": rest_days,
                    }

                    if best_over['price'] > 0:
                        over_metrics = db.get_market_metrics(player_name, mkt, line, "OVER")
                        candidates.append({
                            **common,
                            "side":         "OVER",
                            "book":         best_over['book'],
                            "book_role":    db.get_bookmaker_role(best_over['book']),
                            "odds":         best_over['price'],
                            "model_prob":   dists['prob_over'],
                            "implied_prob": imp_over,
                            **over_metrics,
                        })

                    # Unders intentionally excluded — OVER only strategy

                    # ── Top-down sharp vs rec comparison ─────────────────
                    true_o, true_u, sharp_book = get_sharp_true_prob(
                        bookmakers, player_name, mkt, line)

                    if true_o is not None:
                        mkt_label = mkt.replace('player_', '').replace('_', ' ').title()
                        for chk_side, true_prob in (('Over', true_o),):
                            rec_price, rec_book = get_best_rec_price(
                                bookmakers, player_name, mkt, line, chk_side)
                            if rec_price is None:
                                continue
                            rec_implied = decimal_to_implied_prob(rec_price)
                            sharp_gap   = true_prob - rec_implied
                            if sharp_gap < SHARP_EDGE_MIN:
                                continue
                            dedup_key = (player_name, mkt, line, chk_side)
                            if dedup_key in sharp_alerted:
                                continue
                            sharp_alerted.add(dedup_key)
                            msg = (
                                f"🔪 <b>Sharp Alert: {player_name} "
                                f"{chk_side} {line} {mkt_label}</b>\n\n"
                                f"{sharp_book.title()} True: {true_prob:.1%}"
                                f" → {rec_book.title()}: {rec_implied:.1%}\n"
                                f"Gap: <b>+{sharp_gap:.1%}</b>"
                                f" | {rec_book.title()} {_american(rec_price)}"
                            )
                            bot.send_message(msg)
                            logger.info(
                                f"Sharp alert: {player_name} {chk_side} {line} {mkt} "
                                f"{sharp_book} {true_prob:.1%} vs "
                                f"{rec_book} {rec_implied:.1%} gap={sharp_gap:.1%}"
                            )

    ranked_edges = rank_edges(candidates)
    actionable   = [e for e in ranked_edges if e.get('edge', 0) >= EDGE_MIN]

    logger.info(f"Scan complete: {len(candidates)} candidates, {len(actionable)} actionable edges.")

    for edge in actionable:
        logger.info(
            f"EDGE: {edge['player_id']} {edge['market']} {edge['side']} {edge['line']} "
            f"@ {edge['book']} ({edge['odds']}) edge={edge['edge']:.2%}"
        )
        evaluate_and_alert(edge, db, bot)

    # Multi-leg combo alerts (2–4 legs)
    generate_and_alert_combos(actionable, bot)


if __name__ == "__main__":
    scan_props()
