# Live In-Game Trading Pipeline

Three async services that work together to identify and execute +EV in-game trades on Kalshi.

**Strategy:** When the model projects a team's win probability at 71% and Kalshi is pricing them at 60%, buy the contract. If the market overreacts and prices them at 88% while the model only sees 82%, sell immediately to lock in profit — rather than holding until the final buzzer.

---

## Architecture

```
[BDL API]  ──poll every 5s──▶  live_state_tracker  ──▶  Redis nba:live:state_update
                                                                     │
                                                          live_engine (win prob model)
                                                          + Kalshi price polling
                                                                     │
                                                         Redis nba:live:execution_queue
                                                                     │
                                                          kalshi_trader (circuit breakers)
                                                          ──▶  Kalshi BUY / SELL orders
                                                          ──▶  Postgres live_trades table
```

---

## Prerequisites

1. **PostgreSQL** running and accessible via `PG_DSN`.
2. **Redis** running and accessible via `REDIS_URL`.
3. **BDL GOAT-tier API key** — required for live box score polling.
4. **Kalshi API credentials** — required for price fetching and order placement.

### Apply the schema migration

```bash
psql $PG_DSN -f nba-prop-bot/src/data/schema_postgres.sql
```

This is idempotent (`IF NOT EXISTS` guards on all tables), so it is safe to run against an existing database.

---

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `REDIS_URL` | Redis connection URL, e.g. `redis://localhost:6379` |
| `PG_DSN` | PostgreSQL DSN, e.g. `postgresql://user:pass@host:5432/db` |
| `BDL_API_KEY` | BallDontLie GOAT-tier API key |
| `KALSHI_API_KEY` | Kalshi API key |
| `KALSHI_API_SECRET` | Kalshi RSA private key (PEM block or raw base64) |

### Bankroll / Sizing (shared with pre-game arb)

| Variable | Default | Description |
|----------|---------|-------------|
| `BANKROLL` | `1000.0` | Total bankroll in USD |
| `KELLY_FRACTION` | `0.25` | Fractional Kelly multiplier (0.25 = quarter-Kelly) |
| `KALSHI_MAX_STAKE` | `5.00` | Maximum USD per single order |

### Circuit Breakers

| Variable | Default | Description |
|----------|---------|-------------|
| `LIVE_MAX_EXPOSURE_PER_TEAM` | `100.0` | Max total USD exposure per team per game |
| `LIVE_SIGNAL_LATENCY_LIMIT` | `1.5` | Reject signals older than this many seconds |
| `LIVE_WASH_COOLDOWN` | `2.0` | Min seconds between actions on the same ticker |

### Engine Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `LIVE_BUY_THRESHOLD` | `0.07` | Min (model_prob − kalshi_price) to trigger a BUY |
| `LIVE_SELL_THRESHOLD` | `0.03` | Min (kalshi_price − model_prob) to trigger a SELL |

### Tracker Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `LIVE_POLL_INTERVAL_SECONDS` | `5` | Seconds between BDL polls |

### Execution Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `EXECUTION_MODE` | `paper` | `paper` simulates fills and logs to DB; `live` places real Kalshi orders |

---

## Startup Order

**Always start services in this order.** Each downstream service depends on the upstream Redis channel being populated.

### Step 1 — State Tracker

Polls BDL for live game scores and publishes state updates to Redis.

```bash
python -m src.pipelines.live_state_tracker
```

Expected output when games are live:
```
INFO live_state_tracker: 2 live game(s) — Celtics 54–48 Knicks (Q3 7:14), Lakers 41–39 Warriors (Q2 3:55)
```

Verify with Redis:
```bash
redis-cli SUBSCRIBE "nba:live:state_update"
```

### Step 2 — Live Engine

Consumes state updates, computes win probabilities, and emits BUY/SELL signals.

```bash
python -m src.pipelines.live_engine
```

Expected output on a BUY signal:
```
INFO live_engine: BUY signal — Celtics model=71.0% kalshi=60.0% edge=+11.0% stake=$5.00
```

Verify signals:
```bash
redis-cli SUBSCRIBE "nba:live:execution_queue"
```

### Step 3 — Kalshi Trader

Consumes signals, runs circuit breakers, and executes orders.

**Start in paper mode first:**

```bash
EXECUTION_MODE=paper python -m src.execution.kalshi_trader
```

Check the `live_trades` table to confirm paper trades are being logged:
```sql
SELECT action, ticker, team, contracts, price_cents, stake_usd, realized_pnl, executed_at
FROM live_trades
ORDER BY executed_at DESC
LIMIT 20;
```

**Switch to live mode** only after validating paper behaviour:
```bash
EXECUTION_MODE=live python -m src.execution.kalshi_trader
```

---

## Running as a Procfile (Railway / Heroku)

```
live_tracker: python -m src.pipelines.live_state_tracker
live_engine:  python -m src.pipelines.live_engine
live_trader:  python -m src.execution.kalshi_trader
```

Or with `honcho`:
```bash
honcho start live_tracker live_engine live_trader
```

---

## Monitoring

### Open positions
```sql
SELECT ticker, team, side, contracts, avg_price_cents, total_stake_usd, opened_at
FROM live_positions
ORDER BY opened_at DESC;
```

### Today's realized P&L
```sql
SELECT
    team,
    COUNT(*) FILTER (WHERE action = 'sell')    AS sells,
    SUM(realized_pnl)                          AS realized_pnl,
    COUNT(*) FILTER (WHERE action = 'rejected') AS rejections
FROM live_trades
WHERE DATE(executed_at) = CURRENT_DATE
GROUP BY team
ORDER BY realized_pnl DESC NULLS LAST;
```

### Circuit breaker rejection breakdown
```sql
SELECT rejection_reason, COUNT(*) AS n
FROM live_trades
WHERE action = 'rejected'
  AND DATE(executed_at) = CURRENT_DATE
GROUP BY rejection_reason
ORDER BY n DESC;
```

---

## Win Probability Model

The engine uses a **Normal-approximation** anchored to the pre-game devigged sharp moneyline:

```
sigma     = 11.0 × sqrt(minutes_remaining / 48)
prior_adj = Φ⁻¹(pre_game_prob) × sigma
win_prob  = Φ((score_diff + prior_adj) / sigma)
```

- At tip-off: `win_prob == pre_game_prob` (the prior is the entire signal).
- As time runs out: `sigma → 0`, so `win_prob → 1.0` or `0.0` based purely on the score.
- `sigma = 11.0` at tip-off matches the empirical NBA game-score standard deviation.

The pre-game probability is fetched once per game from the sharpest available Odds API book (Pinnacle → Circa → Bookmaker) and cached for the duration of the game.

---

## Circuit Breakers (in execution order)

| Breaker | Check | Cost |
|---------|-------|------|
| Latency | Signal age > `LIVE_SIGNAL_LATENCY_LIMIT` s | In-memory |
| Wash trade | Same ticker acted on within `LIVE_WASH_COOLDOWN` s | In-memory |
| Exposure cap | Sum of open stake for team > `LIVE_MAX_EXPOSURE_PER_TEAM` | DB query |

Cheaper checks always run first. All rejected trades are logged to `live_trades` with `action='rejected'` and a `rejection_reason` for auditability.
