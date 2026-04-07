CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    home_team TEXT,
    away_team TEXT,
    commence_time DATETIME,
    status TEXT
);

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    team_name TEXT,
    abbreviation TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    target_name TEXT,
    team_id INTEGER,
    position TEXT,
    is_primary_initiator BOOLEAN NOT NULL DEFAULT 0,
    FOREIGN KEY(team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS injury_reports (
    game_date DATE,
    player_name TEXT,
    team TEXT,
    status TEXT,
    PRIMARY KEY (game_date, player_name)
);

CREATE TABLE IF NOT EXISTS player_game_logs (
    player_id INTEGER,
    game_id TEXT,
    game_date DATE,
    minutes INTEGER,
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    threes INTEGER,
    PRIMARY KEY(player_id, game_id)
);

CREATE TABLE IF NOT EXISTS team_context_daily (
    team_id INTEGER,
    game_date DATE,
    pace_rating REAL,
    offensive_rating REAL,
    defensive_rating REAL,
    opponent_pts_allowed_per_pos REAL,
    opponent_reb_allowed_per_pos REAL,
    opponent_ast_allowed_per_pos REAL,
    PRIMARY KEY(team_id, game_date)
);

CREATE TABLE IF NOT EXISTS prop_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT,
    player_name TEXT,
    market TEXT,
    line REAL,
    over_odds REAL,
    under_odds REAL,
    implied_over REAL,
    implied_under REAL,
    best_book TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS projections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT,
    market TEXT,
    projected_mean REAL,
    model_prob_over REAL,
    model_prob_under REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alerts_sent (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT,
    market TEXT,
    line REAL,
    side TEXT,
    edge REAL,
    book TEXT,
    odds REAL,
    stake REAL DEFAULT 0.0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bet_results (
    alert_id INTEGER PRIMARY KEY,
    actual_result REAL,
    won BOOLEAN,
    push BOOLEAN NOT NULL DEFAULT 0,
    settled_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(alert_id) REFERENCES alerts_sent(id)
);

CREATE TABLE IF NOT EXISTS clv_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id TEXT,
    market TEXT,
    side TEXT,
    alert_odds REAL,
    alert_time DATETIME,
    closing_odds REAL,
    implied_closing REAL,
    implied_alert REAL,
    clv REAL,
    closing_time DATETIME
);

CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    parameters_json TEXT,
    performance_metrics TEXT
);

CREATE TABLE IF NOT EXISTS line_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT,
    market TEXT,
    bookmaker TEXT,
    line REAL,
    side TEXT,
    odds REAL,
    implied_prob REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bookmaker_profiles (
    bookmaker TEXT PRIMARY KEY,
    role TEXT,
    historical_clv_score REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS referee_stats (
    referee_name TEXT PRIMARY KEY,
    avg_pace REAL DEFAULT 99.0,
    avg_pfd_per_game REAL DEFAULT 0.0,
    games_tracked INTEGER DEFAULT 0,
    last_updated DATE
);

CREATE TABLE IF NOT EXISTS sgp_correlations (
    player_name TEXT,
    market_a TEXT,
    market_b TEXT,
    correlation REAL,
    sample_size INTEGER,
    last_updated DATE,
    PRIMARY KEY (player_name, market_a, market_b)
);

CREATE TABLE IF NOT EXISTS on_off_splits (
    player_id        INTEGER NOT NULL,
    absent_player_id INTEGER NOT NULL,
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
    player_id        INTEGER NOT NULL,
    season           TEXT    NOT NULL,
    slot_key         TEXT    NOT NULL,  -- "Q1_0" .. "Q4_5" (2-minute buckets)
    slot_probability REAL    NOT NULL,  -- fraction of games player was on floor in slot
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
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT NOT NULL,
    season TEXT NOT NULL,
    market TEXT NOT NULL,
    game_date TEXT NOT NULL,
    simulated_line REAL NOT NULL,
    model_mean REAL NOT NULL,
    model_prob_over REAL NOT NULL,
    actual_stat REAL NOT NULL,
    hit INTEGER NOT NULL,       -- 1 if actual > simulated_line
    edge REAL NOT NULL,         -- model_prob_over - 0.5
    run_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS team_opponent_stats (
    team_name TEXT,
    season TEXT,
    opp_pts_pg REAL,
    opp_reb_pg REAL,
    opp_ast_pg REAL,
    opp_fg3m_pg REAL,
    pace REAL,
    def_rating REAL,
    last_updated DATE,
    PRIMARY KEY (team_name, season)
);

CREATE TABLE IF NOT EXISTS steam_alerts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name         TEXT    NOT NULL,
    market              TEXT    NOT NULL,
    side                TEXT    NOT NULL,
    line                REAL    NOT NULL,
    sharp_book          TEXT    NOT NULL,
    sharp_delta         REAL    NOT NULL,   -- prob change at sharp book (signed)
    sharp_current_prob  REAL    NOT NULL,   -- latest implied_prob at sharp book
    stale_book          TEXT    NOT NULL,   -- soft book that hasn't moved yet
    stale_odds          REAL    NOT NULL,   -- decimal odds still on offer
    stale_current_prob  REAL    NOT NULL,   -- implied_prob at stale book
    direction           TEXT    NOT NULL,   -- 'OVER' or 'UNDER'
    timestamp           DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Opposing-player finite-resource correlations (cross-team SGP adjustment).
-- Books assume independence across teams; negative rebounding / scoring
-- correlations create structural pricing edges in OVER/UNDER pairs.
CREATE TABLE IF NOT EXISTS cross_team_correlations (
    matchup       TEXT NOT NULL,  -- "|"-joined sorted team names (canonical)
    player_a      TEXT NOT NULL,
    player_b      TEXT NOT NULL,
    market_a      TEXT NOT NULL,
    market_b      TEXT NOT NULL,
    correlation   REAL NOT NULL,
    n_games       INTEGER DEFAULT 0,
    computed_date TEXT NOT NULL,
    PRIMARY KEY (matchup, player_a, player_b, market_a, market_b)
);

-- Two-tier alert batching: Tier 2 edges queue here until the next digest flush.
CREATE TABLE IF NOT EXISTS pending_alerts (
    id          INTEGER  PRIMARY KEY AUTOINCREMENT,
    alert_type  TEXT     NOT NULL,            -- 'prop' | 'game_market' | 'parlay'
    title       TEXT     NOT NULL,            -- compact one-liner shown in digest
    body        TEXT     NOT NULL,            -- full HTML body (for reference / future standalone send)
    priority    REAL     NOT NULL DEFAULT 0.0, -- sort key: higher = shown first (use edge/ev value)
    game_date   TEXT,                         -- YYYY-MM-DD, for grouping
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    sent_at     DATETIME                      -- NULL = pending; populated when included in a digest
);

CREATE INDEX IF NOT EXISTS idx_pending_alerts_unsent
    ON pending_alerts(sent_at, created_at);

-- Indexes to accelerate the steam detection query (scans last 120 min)
CREATE INDEX IF NOT EXISTS idx_line_history_ts
    ON line_history(timestamp);

CREATE INDEX IF NOT EXISTS idx_steam_alerts_dedup
    ON steam_alerts(player_name, market, side, timestamp);
