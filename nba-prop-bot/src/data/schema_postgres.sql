-- PostgreSQL / TimescaleDB schema for the NBA props bot.
-- Mirrors schema.sql with PostgreSQL-native types.
-- line_history is converted to a TimescaleDB hypertable after table creation.
--
-- Run once against a fresh database (idempotent due to IF NOT EXISTS guards).
-- TimescaleDB extension must be enabled: CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS games (
    game_id     TEXT PRIMARY KEY,
    home_team   TEXT,
    away_team   TEXT,
    commence_time TIMESTAMPTZ,
    status      TEXT
);

CREATE TABLE IF NOT EXISTS teams (
    team_id     BIGINT PRIMARY KEY,
    team_name   TEXT,
    abbreviation TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id            BIGINT PRIMARY KEY,
    target_name          TEXT,
    team_id              BIGINT REFERENCES teams(team_id),
    position             TEXT,
    is_primary_initiator BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS injury_reports (
    game_date   DATE    NOT NULL,
    player_name TEXT    NOT NULL,
    team        TEXT,
    status      TEXT,
    description TEXT,
    return_date TEXT,
    severity    INTEGER DEFAULT 0,
    source      TEXT,
    updated_at  TIMESTAMPTZ,
    PRIMARY KEY (game_date, player_name)
);

CREATE INDEX IF NOT EXISTS idx_injury_reports_date_status
    ON injury_reports(game_date, status);

CREATE TABLE IF NOT EXISTS player_game_logs (
    player_id   BIGINT  NOT NULL,
    game_id     TEXT    NOT NULL,
    game_date   DATE,
    minutes     INTEGER,
    points      INTEGER,
    rebounds    INTEGER,
    assists     INTEGER,
    threes      INTEGER,
    PRIMARY KEY (player_id, game_id)
);

CREATE TABLE IF NOT EXISTS team_context_daily (
    team_id                      BIGINT NOT NULL,
    game_date                    DATE   NOT NULL,
    pace_rating                  REAL,
    offensive_rating             REAL,
    defensive_rating             REAL,
    opponent_pts_allowed_per_pos REAL,
    opponent_reb_allowed_per_pos REAL,
    opponent_ast_allowed_per_pos REAL,
    PRIMARY KEY (team_id, game_date)
);

CREATE TABLE IF NOT EXISTS prop_snapshots (
    id           BIGSERIAL PRIMARY KEY,
    game_id      TEXT,
    player_name  TEXT,
    market       TEXT,
    line         REAL,
    over_odds    REAL,
    under_odds   REAL,
    implied_over REAL,
    implied_under REAL,
    best_book    TEXT,
    timestamp    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS projections (
    id               BIGSERIAL PRIMARY KEY,
    player_name      TEXT,
    market           TEXT,
    projected_mean   REAL,
    model_prob_over  REAL,
    model_prob_under REAL,
    timestamp        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS alerts_sent (
    id          BIGSERIAL PRIMARY KEY,
    player_name TEXT,
    market      TEXT,
    line        REAL,
    side        TEXT,
    edge        REAL,
    book        TEXT,
    odds        REAL,
    stake       REAL    DEFAULT 0.0,
    timestamp   TIMESTAMPTZ DEFAULT NOW(),
    game_date   TEXT,
    event_id    TEXT,
    home_away   TEXT,
    rest_days   INTEGER DEFAULT 2
);

CREATE INDEX IF NOT EXISTS idx_alerts_sent_timestamp ON alerts_sent(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_sent_game_date ON alerts_sent(game_date);

CREATE TABLE IF NOT EXISTS bet_results (
    alert_id    BIGINT PRIMARY KEY REFERENCES alerts_sent(id),
    actual_result REAL,
    won         BOOLEAN,
    push        BOOLEAN NOT NULL DEFAULT FALSE,
    settled_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS clv_tracking (
    id              BIGSERIAL PRIMARY KEY,
    player_id       TEXT,
    market          TEXT,
    side            TEXT,
    alert_odds      REAL,
    alert_time      TIMESTAMPTZ,
    closing_odds    REAL,
    implied_closing REAL,
    implied_alert   REAL,
    clv             REAL,
    closing_time    TIMESTAMPTZ,
    alert_id        BIGINT REFERENCES alerts_sent(id)
);

CREATE TABLE IF NOT EXISTS model_versions (
    id                  BIGSERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ DEFAULT NOW(),
    parameters_json     TEXT,
    performance_metrics TEXT
);

-- line_history: high-frequency prop snapshots.
-- Created as a standard table here; converted to TimescaleDB hypertable below.
CREATE TABLE IF NOT EXISTS line_history (
    id           BIGSERIAL,
    player_name  TEXT,
    market       TEXT,
    bookmaker    TEXT,
    line         REAL,
    side         TEXT,
    odds         REAL,
    implied_prob REAL,
    timestamp    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, timestamp)   -- composite PK required for hypertable
);

CREATE INDEX IF NOT EXISTS idx_line_history_ts
    ON line_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_line_history_player_market
    ON line_history(player_name, market, timestamp DESC);

CREATE TABLE IF NOT EXISTS bookmaker_profiles (
    bookmaker              TEXT PRIMARY KEY,
    role                   TEXT,
    historical_clv_score   REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS referee_stats (
    referee_name    TEXT PRIMARY KEY,
    avg_pace        REAL DEFAULT 99.0,
    avg_pfd_per_game REAL DEFAULT 0.0,
    games_tracked   INTEGER DEFAULT 0,
    last_updated    DATE
);

CREATE TABLE IF NOT EXISTS sgp_correlations (
    player_name TEXT,
    market_a    TEXT,
    market_b    TEXT,
    correlation REAL,
    sample_size INTEGER,
    last_updated DATE,
    PRIMARY KEY (player_name, market_a, market_b)
);

CREATE TABLE IF NOT EXISTS on_off_splits (
    player_id        BIGINT  NOT NULL,
    absent_player_id BIGINT  NOT NULL,
    season           TEXT    NOT NULL,
    market           TEXT    NOT NULL,
    games_processed  INTEGER NOT NULL DEFAULT 0,
    minutes_with     REAL    NOT NULL DEFAULT 0.0,
    minutes_without  REAL    NOT NULL DEFAULT 0.0,
    rate_with        REAL    NOT NULL DEFAULT 0.0,
    rate_without     REAL    NOT NULL DEFAULT 0.0,
    usage_multiplier REAL,
    last_updated     TEXT    NOT NULL,
    PRIMARY KEY (player_id, absent_player_id, season, market)
);

CREATE TABLE IF NOT EXISTS rotation_slots (
    team_abbr        TEXT    NOT NULL,
    player_id        BIGINT  NOT NULL,
    season           TEXT    NOT NULL,
    slot_key         TEXT    NOT NULL,
    slot_probability REAL    NOT NULL,
    games_processed  INTEGER NOT NULL DEFAULT 0,
    last_updated     TEXT    NOT NULL,
    PRIMARY KEY (team_abbr, player_id, season, slot_key)
);

CREATE TABLE IF NOT EXISTS cross_player_correlations (
    team          TEXT NOT NULL,
    player_a      TEXT NOT NULL,
    player_b      TEXT NOT NULL,
    market_a      TEXT NOT NULL,
    market_b      TEXT NOT NULL,
    correlation   REAL NOT NULL,
    n_games       INTEGER DEFAULT 0,
    computed_date TEXT NOT NULL,
    PRIMARY KEY (team, player_a, player_b, market_a, market_b)
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id              BIGSERIAL PRIMARY KEY,
    player_name     TEXT    NOT NULL,
    season          TEXT    NOT NULL,
    market          TEXT    NOT NULL,
    game_date       TEXT    NOT NULL,
    simulated_line  REAL    NOT NULL,
    model_mean      REAL    NOT NULL,
    model_prob_over REAL    NOT NULL,
    actual_stat     REAL    NOT NULL,
    hit             INTEGER NOT NULL,
    edge            REAL    NOT NULL,
    run_timestamp   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS team_opponent_stats (
    team_name    TEXT,
    season       TEXT,
    opp_pts_pg   REAL,
    opp_reb_pg   REAL,
    opp_ast_pg   REAL,
    opp_fg3m_pg  REAL,
    pace         REAL,
    def_rating   REAL,
    last_updated DATE,
    PRIMARY KEY (team_name, season)
);

CREATE TABLE IF NOT EXISTS steam_alerts (
    id                 BIGSERIAL PRIMARY KEY,
    player_name        TEXT    NOT NULL,
    market             TEXT    NOT NULL,
    side               TEXT    NOT NULL,
    line               REAL    NOT NULL,
    sharp_book         TEXT    NOT NULL,
    sharp_delta        REAL    NOT NULL,
    sharp_current_prob REAL    NOT NULL,
    stale_book         TEXT    NOT NULL,
    stale_odds         REAL    NOT NULL,
    stale_current_prob REAL    NOT NULL,
    direction          TEXT    NOT NULL,
    timestamp          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_steam_alerts_dedup
    ON steam_alerts(player_name, market, side, timestamp DESC);

CREATE TABLE IF NOT EXISTS cross_team_correlations (
    matchup       TEXT NOT NULL,
    player_a      TEXT NOT NULL,
    player_b      TEXT NOT NULL,
    market_a      TEXT NOT NULL,
    market_b      TEXT NOT NULL,
    correlation   REAL NOT NULL,
    n_games       INTEGER DEFAULT 0,
    computed_date TEXT NOT NULL,
    PRIMARY KEY (matchup, player_a, player_b, market_a, market_b)
);

CREATE TABLE IF NOT EXISTS pending_alerts (
    id          BIGSERIAL PRIMARY KEY,
    alert_type  TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    body        TEXT    NOT NULL,
    priority    REAL    NOT NULL DEFAULT 0.0,
    game_date   TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    sent_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_pending_alerts_unsent
    ON pending_alerts(sent_at NULLS FIRST, created_at);

CREATE TABLE IF NOT EXISTS bdl_defense_profiles (
    team_key      TEXT    NOT NULL,
    season        INTEGER NOT NULL,
    opp_pts       REAL    NOT NULL DEFAULT 1.0,
    opp_reb       REAL    NOT NULL DEFAULT 1.0,
    opp_ast       REAL    NOT NULL DEFAULT 1.0,
    opp_fg3m      REAL    NOT NULL DEFAULT 1.0,
    opp_fta       REAL    NOT NULL DEFAULT 1.0,
    opp_pts_paint REAL    NOT NULL DEFAULT 1.0,
    def_rating    REAL    NOT NULL DEFAULT 1.0,
    pace          REAL    NOT NULL DEFAULT 1.0,
    blk           REAL    NOT NULL DEFAULT 1.0,
    stl           REAL    NOT NULL DEFAULT 1.0,
    fetched_at    TEXT    NOT NULL,
    PRIMARY KEY (team_key, season)
);

CREATE TABLE IF NOT EXISTS bdl_game_log_cache (
    player_id   BIGINT  NOT NULL,
    season      INTEGER NOT NULL,
    game_date   TEXT    NOT NULL,
    game_id     BIGINT,
    min         REAL    NOT NULL DEFAULT 0.0,
    pts         REAL    NOT NULL DEFAULT 0.0,
    reb         REAL    NOT NULL DEFAULT 0.0,
    ast         REAL    NOT NULL DEFAULT 0.0,
    fg3m        REAL    NOT NULL DEFAULT 0.0,
    blk         REAL    NOT NULL DEFAULT 0.0,
    stl         REAL    NOT NULL DEFAULT 0.0,
    fga         REAL    NOT NULL DEFAULT 0.0,
    fta         REAL    NOT NULL DEFAULT 0.0,
    tov         REAL    NOT NULL DEFAULT 0.0,
    team_abbr   TEXT,
    matchup     TEXT,
    wl          TEXT,
    cached_at   TEXT    NOT NULL,
    PRIMARY KEY (player_id, season, game_date)
);

CREATE TABLE IF NOT EXISTS placed_bets (
    id           BIGSERIAL PRIMARY KEY,
    alert_id     BIGINT  NOT NULL REFERENCES alerts_sent(id),
    mode         TEXT    NOT NULL DEFAULT 'paper',
    player_name  TEXT    NOT NULL,
    market       TEXT    NOT NULL,
    side         TEXT    NOT NULL,
    line         REAL    NOT NULL,
    book         TEXT,
    alerted_odds REAL    NOT NULL,
    fill_odds    REAL,
    slippage     REAL,
    stake        REAL    NOT NULL,
    game_date    TEXT,
    executed_at  TIMESTAMPTZ DEFAULT NOW(),
    session_id   TEXT
);

CREATE INDEX IF NOT EXISTS idx_placed_bets_session
    ON placed_bets(session_id, executed_at DESC);

CREATE TABLE IF NOT EXISTS model_health (
    id            BIGSERIAL PRIMARY KEY,
    snapshot_date DATE    NOT NULL,
    "window"      TEXT    NOT NULL,
    market        TEXT    NOT NULL DEFAULT 'all',
    brier         REAL    NOT NULL,
    n_samples     INTEGER NOT NULL,
    recorded_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_health_unique
    ON model_health(snapshot_date, "window", market);

CREATE TABLE IF NOT EXISTS subscribers (
    id        BIGSERIAL PRIMARY KEY,
    chat_id   TEXT    NOT NULL UNIQUE,
    username  TEXT,
    tier      TEXT    NOT NULL DEFAULT 'basic',
    active    INTEGER NOT NULL DEFAULT 1,
    joined_at TIMESTAMPTZ DEFAULT NOW()
);

-- Schema migrations tracking (mirrors SQLite version)
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    description TEXT,
    applied_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── Live In-Game Trading ──────────────────────────────────────────────────────

-- live_positions: one row per open Kalshi position (keyed by ticker).
-- Upserted on BUY, deleted on SELL.
CREATE TABLE IF NOT EXISTS live_positions (
    ticker          TEXT        PRIMARY KEY,
    game_id         TEXT        NOT NULL,
    team            TEXT        NOT NULL,
    side            TEXT        NOT NULL CHECK (side IN ('yes', 'no')),
    contracts       INTEGER     NOT NULL DEFAULT 0,
    avg_price_cents INTEGER     NOT NULL,
    total_stake_usd REAL        NOT NULL DEFAULT 0.0,
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    order_id        TEXT
);

CREATE INDEX IF NOT EXISTS idx_live_positions_game_id
    ON live_positions(game_id);

-- live_trades: immutable audit log of every execution attempt.
-- action='rejected' rows have zero contracts/price and a rejection_reason.
-- realized_pnl is NULL on BUY rows, populated when the matching SELL fills.
CREATE TABLE IF NOT EXISTS live_trades (
    id               BIGSERIAL   PRIMARY KEY,
    ticker           TEXT        NOT NULL,
    game_id          TEXT        NOT NULL,
    team             TEXT        NOT NULL,
    side             TEXT        NOT NULL CHECK (side IN ('yes', 'no')),
    action           TEXT        NOT NULL CHECK (action IN ('buy', 'sell', 'rejected')),
    contracts        INTEGER     NOT NULL DEFAULT 0,
    price_cents      INTEGER     NOT NULL DEFAULT 0,
    stake_usd        REAL        NOT NULL DEFAULT 0.0,
    model_prob       REAL        NOT NULL,
    kalshi_price     REAL        NOT NULL,
    edge             REAL        NOT NULL,
    order_id         TEXT,
    session_id       TEXT        NOT NULL,
    rejection_reason TEXT,
    executed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    realized_pnl     REAL
);

CREATE INDEX IF NOT EXISTS idx_live_trades_ticker_session
    ON live_trades(ticker, session_id, executed_at DESC);
CREATE INDEX IF NOT EXISTS idx_live_trades_game_id
    ON live_trades(game_id, executed_at DESC);
