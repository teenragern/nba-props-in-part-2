"""
Tests for the five institutional-grade upgrades:
  1. PBP Streamer — event parsing, possession state machine, foul tracking
  2. Possession Win Model — convergence, bonus effects, MC simulation
  3. L2 Execution Engine — strategy selection, TWAP planning
  4. Live Usage Adjuster — multiplier calculation, conservation constraint
  5. Portfolio Risk Manager — covariance dampening, concentration limits
"""

import asyncio
import math
import time
from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADE 1: PBP Streamer Tests
# ═══════════════════════════════════════════════════════════════════════════════

from src.clients.pbp_streamer import (
    PBPStateEngine,
    parse_pbp_message,
    PossessionState,
    GameState,
    TeamFoulState,
)


class TestPBPParser:
    """Test WebSocket message parsing."""

    def test_parse_valid_scoring_event(self):
        raw = '{"type": "event", "payload": {"game_id": "g1", "event_type": "made_shot", "team": "home", "player_id": "p1", "player_name": "LeBron James", "clock": {"period": 2, "seconds_remaining": 345.2}, "points": 2}}'
        event = parse_pbp_message(raw)
        assert event is not None
        assert event["game_id"] == "g1"
        assert event["event_type"] == "made_shot"
        assert event["team"] == "home"
        assert event["points"] == 2
        assert event["seconds_remaining"] == 345.2

    def test_parse_flat_format(self):
        raw = '{"game_id": "g2", "event_type": "foul", "team": "away", "foul_type": "shooting"}'
        event = parse_pbp_message(raw)
        assert event is not None
        assert event["game_id"] == "g2"
        assert event["foul_type"] == "shooting"

    def test_parse_invalid_json(self):
        assert parse_pbp_message("not json") is None
        assert parse_pbp_message("") is None
        assert parse_pbp_message("{}") is None  # no event_type


class TestPBPStateEngine:
    """Test possession state machine and foul tracking."""

    def setup_method(self):
        self.engine = PBPStateEngine()

    def test_made_basket_flips_possession(self):
        state = self.engine.process_event({
            "game_id": "g1", "event_type": "made_fg", "team": "home",
            "home_team": "Lakers", "away_team": "Celtics",
            "player_name": "LeBron", "points": 2,
        })
        assert state is not None
        assert state.possession == PossessionState.AWAY

    def test_defensive_rebound_gives_possession(self):
        self.engine.process_event({
            "game_id": "g1", "event_type": "missed_fg", "team": "home",
            "home_team": "Lakers", "away_team": "Celtics",
        })
        state = self.engine.process_event({
            "game_id": "g1", "event_type": "defensive_rebound", "team": "away",
        })
        assert state.possession == PossessionState.AWAY

    def test_offensive_rebound_retains_possession(self):
        self.engine.process_event({
            "game_id": "g1", "event_type": "missed_fg", "team": "home",
            "home_team": "Lakers", "away_team": "Celtics",
        })
        state = self.engine.process_event({
            "game_id": "g1", "event_type": "offensive_rebound", "team": "home",
        })
        assert state.possession == PossessionState.HOME

    def test_turnover_gives_opponent_possession(self):
        state = self.engine.process_event({
            "game_id": "g1", "event_type": "turnover", "team": "home",
            "home_team": "Lakers", "away_team": "Celtics",
        })
        assert state.possession == PossessionState.AWAY

    def test_foul_tracking_triggers_bonus(self):
        """5 fouls in a period should trigger bonus."""
        for i in range(5):
            state = self.engine.process_event({
                "game_id": "g1", "event_type": "foul", "team": "home",
                "home_team": "Lakers", "away_team": "Celtics",
                "foul_type": "personal",
            })
        assert state.home_fouls.period_fouls == 5
        assert state.home_fouls.in_bonus is True
        assert state.home_fouls.in_double_bonus is False

    def test_foul_tracking_triggers_double_bonus(self):
        """10 fouls should trigger double bonus."""
        for i in range(10):
            self.engine.process_event({
                "game_id": "g1", "event_type": "foul", "team": "away",
                "home_team": "Lakers", "away_team": "Celtics",
                "foul_type": "personal",
            })
        game = self.engine._games["g1"]
        assert game.away_fouls.in_double_bonus is True

    def test_period_start_resets_fouls(self):
        for i in range(5):
            self.engine.process_event({
                "game_id": "g1", "event_type": "foul", "team": "home",
                "home_team": "Lakers", "away_team": "Celtics",
                "foul_type": "personal",
            })
        self.engine.process_event({
            "game_id": "g1", "event_type": "period_start", "team": "",
            "period": 2, "seconds_remaining": 720,
        })
        game = self.engine._games["g1"]
        assert game.home_fouls.period_fouls == 0
        assert game.home_fouls.in_bonus is False

    def test_substitution_updates_lineup(self):
        state = self.engine.process_event({
            "game_id": "g1", "event_type": "substitution", "team": "home",
            "home_team": "Lakers", "away_team": "Celtics",
            "sub_in": "player_new", "sub_out": "",
        })
        assert "player_new" in state.players_on_court_home

    def test_minutes_elapsed_calculation(self):
        game = GameState(game_id="g1", home_team="A", away_team="B", period=2, game_clock_seconds=360.0)
        # Period 2, 6:00 remaining → 12 min (Q1) + 6 min (into Q2) = 18 min
        assert abs(game.minutes_elapsed - 18.0) < 0.01

    def test_offensive_rebound_14s_shot_clock(self):
        """NBA rule: offensive rebound resets to 14s, not 24s."""
        self.engine.process_event({
            "game_id": "g1", "event_type": "missed_fg", "team": "home",
            "home_team": "Lakers", "away_team": "Celtics",
        })
        state = self.engine.process_event({
            "game_id": "g1", "event_type": "offensive_rebound", "team": "home",
        })
        assert state.shot_clock_seconds == 14.0


# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADE 2: Possession Win Model Tests
# ═══════════════════════════════════════════════════════════════════════════════

from src.models.possession_model import (
    PossessionWinModel,
    PossessionModelParams,
    GameMicroState,
    compute_live_win_prob,
    _build_scoring_distribution,
)


class TestPossessionModel:
    """Test Markov possession win probability model."""

    def setup_method(self):
        params = PossessionModelParams(pre_game_prob=0.55)
        self.model = PossessionWinModel(params)

    def test_tipoff_equals_pregame_prob(self):
        """At tip-off (score=0, full time), win_prob should match pre_game_prob."""
        state = GameMicroState(score_diff=0, minutes_remaining=48.0)
        prob = self.model.compute_win_prob(state)
        assert abs(prob - 0.55) < 0.03  # within 3% tolerance

    def test_large_lead_near_buzzer_approaches_1(self):
        """Home up 20 with 1 minute left → near certainty."""
        state = GameMicroState(score_diff=20, minutes_remaining=1.0)
        prob = self.model.compute_win_prob(state)
        assert prob > 0.95

    def test_large_deficit_near_buzzer_approaches_0(self):
        """Home down 20 with 1 minute left → near certain loss."""
        state = GameMicroState(score_diff=-20, minutes_remaining=1.0)
        prob = self.model.compute_win_prob(state)
        assert prob < 0.05

    def test_possession_value(self):
        """Team with possession should have slightly higher win prob."""
        state_home = GameMicroState(score_diff=0, minutes_remaining=24.0, possession="home")
        state_away = GameMicroState(score_diff=0, minutes_remaining=24.0, possession="away")
        state_unknown = GameMicroState(score_diff=0, minutes_remaining=24.0)

        p_home = self.model.compute_win_prob(state_home)
        p_away = self.model.compute_win_prob(state_away)
        p_unknown = self.model.compute_win_prob(state_unknown)

        assert p_home > p_unknown > p_away

    def test_bonus_state_helps_fouling_team_opponent(self):
        """If away is in bonus, home gets more FTs → higher home win prob."""
        state_no_bonus = GameMicroState(score_diff=0, minutes_remaining=24.0)
        state_away_bonus = GameMicroState(
            score_diff=0, minutes_remaining=24.0, away_in_bonus=True
        )
        p_no = self.model.compute_win_prob(state_no_bonus)
        p_bonus = self.model.compute_win_prob(state_away_bonus)
        assert p_bonus > p_no  # home benefits from away's fouls

    def test_symmetry(self):
        """Symmetric game (50/50) with tied score should be ~0.50."""
        params = PossessionModelParams(pre_game_prob=0.50)
        model = PossessionWinModel(params)
        state = GameMicroState(score_diff=0, minutes_remaining=24.0)
        prob = model.compute_win_prob(state)
        assert abs(prob - 0.50) < 0.03

    def test_endgame_simulation_runs(self):
        """Monte Carlo path activates below 6 minutes."""
        state = GameMicroState(score_diff=3, minutes_remaining=4.0, possession="home")
        prob = self.model.compute_win_prob(state)
        assert 0.02 < prob < 0.98

    def test_scoring_distribution_sums_to_1(self):
        """Transition probabilities must sum to 1."""
        dist = _build_scoring_distribution(1.10, False, False)
        assert abs(np.sum(dist) - 1.0) < 0.001

    def test_scoring_distribution_bonus_boost(self):
        """Bonus should increase expected points."""
        dist_normal = _build_scoring_distribution(1.10, False, False)
        dist_bonus = _build_scoring_distribution(1.10, True, False)
        ev_normal = np.sum(dist_normal * np.array([0, 1, 2, 3]))
        ev_bonus = np.sum(dist_bonus * np.array([0, 1, 2, 3]))
        assert ev_bonus > ev_normal

    def test_backward_compat_wrapper_fallback(self):
        """compute_live_win_prob without enriched state falls back to Normal."""
        prob = compute_live_win_prob(0, 0.0, 0.55, 0.0)
        assert abs(prob - 0.55) < 0.02

    def test_backward_compat_wrapper_enriched(self):
        """compute_live_win_prob with enriched state uses possession model."""
        prob = compute_live_win_prob(
            10, 36.0, 0.55, 0.0,
            possession="home", home_in_bonus=True,
        )
        assert prob > 0.55  # leading + possession + bonus

    def test_from_pre_game_factory(self):
        """Factory method constructs a valid model."""
        model = PossessionWinModel.from_pre_game(
            pre_game_prob=0.60,
            home_pace=102.0,
            away_pace=98.0,
            home_off_rating=112.0,
            away_off_rating=108.0,
        )
        state = GameMicroState(score_diff=0, minutes_remaining=48.0)
        prob = model.compute_win_prob(state)
        assert abs(prob - 0.60) < 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADE 3: L2 Execution Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

from src.execution.l2_execution import (
    L2ExecutionEngine,
    L2Snapshot,
    OrderBookLevel,
    ExecutionStrategy,
    TWAPPlan,
)


class TestL2Execution:
    """Test L2 order book execution strategy selection and TWAP planning."""

    def setup_method(self):
        self.mock_exchange = MagicMock()
        self.engine = L2ExecutionEngine(self.mock_exchange)

    def _make_l2(self, best_bid=55, best_ask=58, depth=50):
        return L2Snapshot(
            ticker="TEST",
            yes_bids=[OrderBookLevel(best_bid, depth)],
            yes_asks=[OrderBookLevel(best_ask, depth)],
        )

    def test_strategy_selection_locked_market(self):
        """Locked market (spread=0) → aggressive."""
        l2 = self._make_l2(best_bid=55, best_ask=56)  # 1c spread
        strategy = self.engine._select_strategy(10, l2, 0.07)
        assert strategy == ExecutionStrategy.AGGRESSIVE

    def test_strategy_selection_small_order(self):
        """Small order with reasonable spread → passive."""
        l2 = self._make_l2(best_bid=50, best_ask=55)  # 5c spread
        strategy = self.engine._select_strategy(3, l2, 0.07)
        assert strategy == ExecutionStrategy.PASSIVE

    def test_strategy_selection_medium_order(self):
        """Medium order → TWAP."""
        l2 = self._make_l2(best_bid=50, best_ask=55)
        strategy = self.engine._select_strategy(15, l2, 0.07)
        assert strategy == ExecutionStrategy.TWAP

    def test_strategy_selection_very_strong_edge(self):
        """Very strong edge (>15%) → aggressive regardless of size."""
        l2 = self._make_l2(best_bid=50, best_ask=55)
        strategy = self.engine._select_strategy(15, l2, 0.20)
        assert strategy == ExecutionStrategy.AGGRESSIVE

    def test_twap_plan_small(self):
        """TWAP plan for 10 contracts."""
        l2 = self._make_l2()
        plan = self.engine._build_twap_plan(10, l2, 0.07, 0.60)
        assert plan.n_tranches == 2
        assert plan.tranche_size == 5
        assert plan.total_contracts == 10

    def test_twap_plan_large(self):
        """TWAP plan for 60 contracts → 5 tranches."""
        l2 = self._make_l2()
        plan = self.engine._build_twap_plan(60, l2, 0.07, 0.60)
        assert plan.n_tranches == 5
        assert plan.tranche_size == 12

    def test_twap_plan_strong_edge_faster(self):
        """Strong edge should produce shorter duration."""
        l2 = self._make_l2()
        plan_strong = self.engine._build_twap_plan(20, l2, 0.12, 0.60)
        plan_weak = self.engine._build_twap_plan(20, l2, 0.04, 0.60)
        assert plan_strong.max_duration_seconds < plan_weak.max_duration_seconds

    def test_l2_snapshot_properties(self):
        """Test L2Snapshot computed properties."""
        l2 = L2Snapshot(
            ticker="TEST",
            yes_bids=[OrderBookLevel(55, 100), OrderBookLevel(54, 50)],
            yes_asks=[OrderBookLevel(58, 80), OrderBookLevel(59, 40)],
        )
        assert l2.best_yes_bid == 55
        assert l2.best_yes_ask == 58
        assert l2.spread_cents == 3
        assert l2.mid_price_cents == 56
        assert l2.bid_depth == 150
        assert l2.ask_depth == 120


# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADE 4: Live Usage Adjuster Tests
# ═══════════════════════════════════════════════════════════════════════════════

from src.models.live_usage_adjuster import (
    recalculate_live_usage,
    LiveUsageAdjuster,
    PlayerProfile,
    _MULTIPLIER_CEILING,
    _MULTIPLIER_FLOOR,
)


class TestLiveUsageAdjuster:
    """Test dynamic on/off usage adjustment."""

    def test_no_absences_no_adjustment(self):
        """When all starters are on court, no adjustment needed."""
        result = recalculate_live_usage(
            target_player_id=1,
            players_on_court={1, 2, 3, 4, 5},
            team_starters={1, 2, 3, 4, 5},
            on_off_data={},
            base_usage=0.25,
            base_ts=0.55,
        )
        assert result["real_usage_pct"] == 0.25
        assert result["ts_pct_5g"] == 0.55

    def test_star_absence_boosts_usage(self):
        """When a star is absent, teammates get boosted usage."""
        result = recalculate_live_usage(
            target_player_id=1,
            players_on_court={1, 2, 3, 4, 6},  # player 5 (star) is out
            team_starters={1, 2, 3, 4, 5},
            on_off_data={5: {"player_points": 1.30}},  # 30% boost when 5 is out
            base_usage=0.20,
            base_ts=0.55,
        )
        assert result["real_usage_pct"] == pytest.approx(0.20 * 1.30, rel=0.01)
        assert result["ts_pct_5g"] > 0.55  # TS also boosted (dampened)

    def test_multiple_absences_compound(self):
        """Multiple absences multiply together."""
        result = recalculate_live_usage(
            target_player_id=1,
            players_on_court={1, 2, 3, 6, 7},  # players 4 and 5 out
            team_starters={1, 2, 3, 4, 5},
            on_off_data={
                4: {"player_points": 1.15},
                5: {"player_points": 1.25},
            },
            base_usage=0.20,
            base_ts=0.55,
        )
        expected_mult = 1.15 * 1.25  # = 1.4375
        assert result["real_usage_pct"] == pytest.approx(0.20 * expected_mult, rel=0.01)

    def test_multiplier_ceiling(self):
        """Combined multiplier should be capped at ceiling."""
        result = recalculate_live_usage(
            target_player_id=1,
            players_on_court={1, 6, 7, 8, 9},  # 4 starters out
            team_starters={1, 2, 3, 4, 5},
            on_off_data={
                2: {"player_points": 1.50},
                3: {"player_points": 1.50},
                4: {"player_points": 1.50},
                5: {"player_points": 1.50},
            },
            base_usage=0.20,
            base_ts=0.55,
        )
        # 1.5^4 = 5.06 → capped at 2.0
        assert result["real_usage_pct"] == pytest.approx(0.20 * _MULTIPLIER_CEILING, rel=0.01)

    def test_ts_dampening(self):
        """TS% adjustment should be dampened vs. usage adjustment."""
        result = recalculate_live_usage(
            target_player_id=1,
            players_on_court={1, 2, 3, 4, 6},
            team_starters={1, 2, 3, 4, 5},
            on_off_data={5: {"player_points": 1.40}},
            base_usage=0.25,
            base_ts=0.55,
        )
        # Usage boost = 40%, TS boost should be 20% (dampened by 0.5)
        usage_boost = result["real_usage_pct"] / 0.25 - 1.0
        ts_boost = result["ts_pct_5g"] / 0.55 - 1.0
        assert ts_boost < usage_boost
        assert ts_boost == pytest.approx(usage_boost * 0.5, rel=0.1)

    def test_missing_absent_data_uses_default(self):
        """When no on_off data for an absent player, multiplier = 1.0."""
        result = recalculate_live_usage(
            target_player_id=1,
            players_on_court={1, 2, 3, 4, 6},  # player 5 out but no data
            team_starters={1, 2, 3, 4, 5},
            on_off_data={},  # no data for anyone
            base_usage=0.25,
            base_ts=0.55,
        )
        assert result["real_usage_pct"] == 0.25  # no change


# ═══════════════════════════════════════════════════════════════════════════════
# UPGRADE 5: Portfolio Risk Manager Tests
# ═══════════════════════════════════════════════════════════════════════════════

from src.models.portfolio_risk import (
    PortfolioRiskManager,
    ProposedTrade,
    OpenPosition,
    _DEFAULT_CORRELATIONS,
)


class TestPortfolioRiskManager:
    """Test covariance-adjusted portfolio sizing."""

    def setup_method(self):
        self.mock_db = MagicMock()
        # Mock get_conn to return empty results
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_conn.execute.return_value.fetchone.return_value = None
        self.mock_db.get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        self.mock_db.get_conn.return_value.__exit__ = MagicMock(return_value=False)

        self.risk_mgr = PortfolioRiskManager(self.mock_db, bankroll=10000.0)

    def test_single_position_no_dampening(self):
        """With no existing positions, raw Kelly should pass through."""
        # Override _load_open_positions to return empty
        self.risk_mgr._load_open_positions = MagicMock(return_value=[])

        proposed = ProposedTrade(
            ticker="KXNBAGAME-TEST", game_id="g1", team="Lakers",
            side="yes", market_type="game_winner", player_name=None,
            model_prob=0.60, kalshi_price=0.53, edge=0.07,
            raw_kelly_stake=5.00,
        )
        result = asyncio.run(self.risk_mgr.size_position(proposed))
        assert result.adjusted_stake == 5.00
        assert result.dampening_factor == pytest.approx(1.0)

    def test_concentration_per_game_cap(self):
        """Per-game cap should limit exposure."""
        existing = [
            OpenPosition(
                ticker="KXNBAGAME-A", game_id="g1", team="Lakers",
                side="yes", contracts=100, avg_price_cents=55,
                total_stake_usd=900.0,  # already at 9% of $10k bankroll
                market_type="game_winner", current_price=0.55,
            )
        ]
        self.risk_mgr._load_open_positions = MagicMock(return_value=existing)

        proposed = ProposedTrade(
            ticker="KXNBAGAME-B", game_id="g1", team="Lakers",
            side="yes", market_type="game_winner", player_name=None,
            model_prob=0.62, kalshi_price=0.55, edge=0.07,
            raw_kelly_stake=500.0,  # wants $500 but per-game cap is $1000 (10%)
        )
        result = asyncio.run(self.risk_mgr.size_position(proposed))
        # Per-game cap = $1000, existing = $900, headroom = $100
        assert result.adjusted_stake <= 100.0

    def test_correlated_positions_dampened(self):
        """Same-game same-team positions should be dampened by covariance."""
        existing = [
            OpenPosition(
                ticker="KXNBAPTS-A", game_id="g1", team="Lakers",
                side="yes", contracts=10, avg_price_cents=50,
                total_stake_usd=50.0,
                market_type="player_points", current_price=0.50,
            )
        ]
        self.risk_mgr._load_open_positions = MagicMock(return_value=existing)

        proposed = ProposedTrade(
            ticker="KXNBAPTS-B", game_id="g1", team="Lakers",
            side="yes", market_type="player_points", player_name="Player B",
            model_prob=0.60, kalshi_price=0.53, edge=0.07,
            raw_kelly_stake=5.00,
        )
        result = asyncio.run(self.risk_mgr.size_position(proposed))
        # Should be dampened due to same-game same-team correlation
        assert result.adjusted_stake < 5.00
        assert result.covariance_penalty > 0

    def test_uncorrelated_positions_undampened(self):
        """Different-game different-team positions should not be dampened."""
        existing = [
            OpenPosition(
                ticker="KXNBAGAME-X", game_id="g2", team="Celtics",
                side="yes", contracts=10, avg_price_cents=60,
                total_stake_usd=50.0,
                market_type="game_winner", current_price=0.60,
            )
        ]
        self.risk_mgr._load_open_positions = MagicMock(return_value=existing)

        proposed = ProposedTrade(
            ticker="KXNBAGAME-Y", game_id="g3", team="Warriors",
            side="yes", market_type="game_winner", player_name=None,
            model_prob=0.58, kalshi_price=0.51, edge=0.07,
            raw_kelly_stake=5.00,
        )
        result = asyncio.run(self.risk_mgr.size_position(proposed))
        # Independent positions: no covariance penalty
        assert result.covariance_penalty == pytest.approx(0.0, abs=0.01)

    def test_blowout_dampening(self):
        """Over props in blowout (Q3+, diff>15) should be dampened."""
        self.risk_mgr._load_open_positions = MagicMock(return_value=[])

        proposed = ProposedTrade(
            ticker="KXNBAPTS-BLOWOUT", game_id="g1", team="Lakers",
            side="yes", market_type="player_points", player_name="Star",
            model_prob=0.60, kalshi_price=0.53, edge=0.07,
            raw_kelly_stake=5.00,
            score_diff=20,  # blowout
            period=3,       # Q3
        )
        result = asyncio.run(self.risk_mgr.size_position(proposed))
        assert result.blowout_penalty > 0
        assert result.adjusted_stake < 5.00

    def test_blowout_no_dampening_early_game(self):
        """Blowout dampening should NOT trigger in Q1/Q2."""
        self.risk_mgr._load_open_positions = MagicMock(return_value=[])

        proposed = ProposedTrade(
            ticker="KXNBAPTS-EARLY", game_id="g1", team="Lakers",
            side="yes", market_type="player_points", player_name="Star",
            model_prob=0.60, kalshi_price=0.53, edge=0.07,
            raw_kelly_stake=5.00,
            score_diff=20,  # big lead...
            period=2,       # ...but only Q2
        )
        result = asyncio.run(self.risk_mgr.size_position(proposed))
        assert result.blowout_penalty == 0.0

    def test_market_type_inference(self):
        """Ticker prefix should correctly infer market type."""
        assert PortfolioRiskManager._infer_market_type("KXNBAGAME-LAL-BOS") == "game_winner"
        assert PortfolioRiskManager._infer_market_type("KXNBAPTS-LEBRON-25") == "player_points"
        assert PortfolioRiskManager._infer_market_type("KXNBAREB-AD-10") == "player_rebounds"
        assert PortfolioRiskManager._infer_market_type("KXNBAAST-HALIBURTON") == "player_assists"
        assert PortfolioRiskManager._infer_market_type("KXNBA3PT-CURRY-5") == "player_threes"

    def test_total_exposure_cap(self):
        """Total exposure cap should reject when portfolio is full."""
        # Fill portfolio to 19% of $10k = $1900
        existing = [
            OpenPosition(
                ticker=f"KXNBAGAME-{i}", game_id=f"g{i}", team=f"Team{i}",
                side="yes", contracts=10, avg_price_cents=50,
                total_stake_usd=380.0,
                market_type="game_winner", current_price=0.50,
            )
            for i in range(5)  # 5 * $380 = $1900
        ]
        self.risk_mgr._load_open_positions = MagicMock(return_value=existing)

        proposed = ProposedTrade(
            ticker="KXNBAGAME-NEW", game_id="g99", team="NewTeam",
            side="yes", market_type="game_winner", player_name=None,
            model_prob=0.60, kalshi_price=0.53, edge=0.07,
            raw_kelly_stake=500.0,  # wants $500 but only $100 headroom
        )
        result = asyncio.run(self.risk_mgr.size_position(proposed))
        # Total cap = $2000 (20%), existing = $1900, headroom = $100
        assert result.adjusted_stake <= 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: End-to-End Signal Flow
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegrationSignalFlow:
    """Test that the possession model + portfolio risk produce valid signals."""

    def test_enriched_state_produces_different_prob_than_basic(self):
        """Micro_state enrichment should shift probability vs basic Normal."""
        basic_prob = compute_live_win_prob(5, 30.0, 0.55, 0.0)
        enriched_prob = compute_live_win_prob(
            5, 30.0, 0.55, 0.0,
            possession="home", home_in_bonus=True,
        )
        # Enriched should be slightly higher (home leading + possession + bonus)
        assert enriched_prob > basic_prob

    def test_model_output_range(self):
        """Model output should always be in (0.02, 0.98)."""
        for diff in range(-50, 51, 5):
            for minutes in [1.0, 6.0, 12.0, 24.0, 36.0, 47.0]:
                prob = compute_live_win_prob(diff, 48.0 - minutes, 0.55, 0.0)
                assert 0.02 <= prob <= 0.98, f"Out of range: diff={diff}, min={minutes}, prob={prob}"
