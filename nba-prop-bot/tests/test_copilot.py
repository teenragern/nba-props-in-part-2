"""
Tests for the Quant Edge Copilot wiring (the narration-only LLM layer).

None of these hit the network. The copilot tests inject a fake OpenRouter
client; the OpenRouterClient guard tests exercise the short-circuit paths
(missing key / exhausted budget) that return before any `openai` import — so
the suite runs even when `openai` is not installed.
"""

import json

from src.clients.copilot import QuantCopilot, SYSTEM_PROMPT, _DISCLAIMER
from src.clients.openrouter_client import OpenRouterClient


# ── Test doubles ──────────────────────────────────────────────────────────

def _alert(**override):
    base = dict(
        id=1, player_name="Bam Adebayo", market="player_assists", side="OVER",
        line=3.5, edge=0.0612, book="draftkings", odds=1.91, stake=12.0,
        raw_model_prob=0.58, model_prob=0.5852,
        timestamp="2026-06-13 18:00:00", won=None, push=None, actual_result=None,
    )
    base.update(override)
    return base


class FakeDB:
    def __init__(self, rows=None, raise_exc=False):
        self._rows = rows if rows is not None else [_alert()]
        self._raise = raise_exc

    def get_recent_alerts(self, limit, hours_back):
        if self._raise:
            raise RuntimeError("db down")
        return list(self._rows)


class FakeClient:
    def __init__(self, reply="canned reply"):
        self.reply = reply
        self.calls = []

    def complete(self, system_prompt, user_prompt, **kwargs):
        self.calls.append((system_prompt, user_prompt))
        return self.reply


def _context_json(user_prompt):
    """Extract the JSON context array embedded in the user prompt."""
    body = user_prompt.split("JSON):\n", 1)[1].split("\n\nUSER", 1)[0]
    return json.loads(body)


# ── Context building ──────────────────────────────────────────────────────

def test_context_includes_authoritative_fields_verbatim():
    client = FakeClient()
    QuantCopilot(FakeDB(), client=client).answer("why this pick?")
    _, user_prompt = client.calls[0]
    assert "Bam Adebayo" in user_prompt
    assert "0.5852" in user_prompt, "model_prob must be in the context"
    assert "0.0612" in user_prompt, "edge must be passed verbatim, not recomputed"


def test_implied_prob_is_one_over_odds():
    client = FakeClient()
    QuantCopilot(FakeDB([_alert(odds=2.0)]), client=client).answer("q")
    ctx = _context_json(client.calls[0][1])
    assert ctx[0]["implied_prob_from_odds"] == 0.5, "implied prob must be 1/odds"


def test_settled_status_mapping():
    rows = [
        _alert(id=1, won=1, push=0),
        _alert(id=2, won=0, push=0),
        _alert(id=3, won=0, push=1),
        _alert(id=4, won=None),
    ]
    client = FakeClient()
    QuantCopilot(FakeDB(rows), client=client).answer("q")
    results = [r["result"] for r in _context_json(client.calls[0][1])]
    assert results == ["won", "lost", "push", "pending"]


# ── Guardrails ─────────────────────────────────────────────────────────────

def test_system_prompt_enforces_narration_only():
    assert "Use ONLY the numbers" in SYSTEM_PROMPT
    assert "Never do probability or edge math" in SYSTEM_PROMPT


def test_disclaimer_appended_when_model_drops_it():
    client = FakeClient(reply="Explanation with no footer.")
    ans = QuantCopilot(FakeDB(), client=client).answer("q")
    assert ans.endswith(_DISCLAIMER)


def test_disclaimer_not_duplicated_when_model_includes_it():
    client = FakeClient(reply=f"Explanation. {_DISCLAIMER}")
    ans = QuantCopilot(FakeDB(), client=client).answer("q")
    assert ans.count(_DISCLAIMER) == 1


# ── Degraded paths ───────────────────────────────────────────────────────

def test_empty_context_skips_llm():
    client = FakeClient()
    ans = QuantCopilot(FakeDB(rows=[]), client=client).answer("q")
    assert client.calls == [], "LLM must not be called when there are no picks"
    assert _DISCLAIMER in ans


def test_db_error_degrades_gracefully():
    client = FakeClient()
    ans = QuantCopilot(FakeDB(raise_exc=True), client=client).answer("q")
    assert client.calls == [], "LLM must not be called when the DB read fails"
    assert _DISCLAIMER in ans


def test_blank_question_skips_llm():
    client = FakeClient()
    ans = QuantCopilot(FakeDB(), client=client).answer("   ")
    assert client.calls == []
    assert _DISCLAIMER in ans


def test_client_failure_returns_graceful_message():
    client = FakeClient(reply=None)  # primary + all fallbacks failed
    ans = QuantCopilot(FakeDB(), client=client).answer("q")
    assert "unavailable" in ans.lower()
    assert _DISCLAIMER in ans


# ── OpenRouterClient guard paths (no network) ───────────────────────────────

def test_client_without_api_key_returns_none():
    assert OpenRouterClient(api_key="").complete("sys", "user") is None


def test_client_exhausted_budget_returns_none():
    # spend (0.0) >= limit (0.0) short-circuits before any API import/call.
    c = OpenRouterClient(api_key="test-key", daily_limit_usd=0.0)
    assert c.complete("sys", "user") is None


def test_model_chain_orders_primary_then_unique_fallbacks():
    c = OpenRouterClient(api_key="x", model="m1", fallback_models=["m1", "m2", "m3"])
    assert c._model_chain() == ["m1", "m2", "m3"]
