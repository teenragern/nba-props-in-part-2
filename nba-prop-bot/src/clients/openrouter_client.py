"""
Synchronous OpenRouter chat client for the Quant Edge Copilot.

OpenRouter is OpenAI-API compatible, so we drive it with the `openai` SDK
pointed at OpenRouter's base URL. This client is deliberately synchronous:
the Telegram listener (telegram_bot.start_listener) runs in a blocking daemon
thread, so a sync call there is simpler and correct — no asyncio bridge needed.

Mirrors the good patterns of the async unified-sports-bot client (model
fallback, approximate cost tracking, a hard daily spend cap) without dragging
in its `settings` / async coupling.
"""

import time
from datetime import date
from typing import List, Optional

from src.config import (
    COPILOT_DAILY_USD_LIMIT,
    COPILOT_FALLBACK_MODELS,
    COPILOT_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Approximate USD per 1K tokens, used ONLY for the daily budget guard.
# Exact pricing is not required — this is a safety cap, not billing.
MODEL_PRICING = {
    "google/gemini-2.5-flash": {"in": 0.0003,  "out": 0.0025},
    "anthropic/claude-sonnet-4": {"in": 0.003,   "out": 0.015},
    "deepseek/deepseek-r1":      {"in": 0.0008,  "out": 0.002},
}
_DEFAULT_PRICE = {"in": 0.001, "out": 0.003}

_RETRYABLE = ("429", "timeout", "rate limit", "502", "503", "504", "overloaded", "server error")


class OpenRouterClient:
    """Sync OpenRouter client with model fallback and a hard daily spend cap."""

    MAX_RETRIES_PER_MODEL = 2
    BASE_BACKOFF = 1.0
    REQUEST_TIMEOUT = 30.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        daily_limit_usd: Optional[float] = None,
        base_url: Optional[str] = None,
    ):
        # `None` means "fall back to config"; an explicit "" disables the client
        # (the missing-key guard in complete() short-circuits before any network).
        self.api_key = api_key if api_key is not None else OPENROUTER_API_KEY
        self.model = model or COPILOT_MODEL
        self.fallback_models = (
            fallback_models if fallback_models is not None else list(COPILOT_FALLBACK_MODELS)
        )
        self.daily_limit = (
            daily_limit_usd if daily_limit_usd is not None else COPILOT_DAILY_USD_LIMIT
        )
        self.base_url = base_url or OPENROUTER_BASE_URL

        self._spend_date = date.today()
        self._spend_usd = 0.0
        self._client = None  # lazy — avoids importing openai unless used

    # ── client / budget helpers ────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy import
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.REQUEST_TIMEOUT,
                max_retries=0,  # we handle retries/fallback ourselves
            )
        return self._client

    def _roll_day(self) -> None:
        today = date.today()
        if today != self._spend_date:
            self._spend_date = today
            self._spend_usd = 0.0

    def budget_remaining(self) -> float:
        self._roll_day()
        return max(0.0, self.daily_limit - self._spend_usd)

    def _track_cost(self, model: str, in_tok: int, out_tok: int) -> float:
        p = MODEL_PRICING.get(model, _DEFAULT_PRICE)
        cost = (in_tok / 1000.0) * p["in"] + (out_tok / 1000.0) * p["out"]
        self._spend_usd += cost
        return cost

    def _model_chain(self) -> List[str]:
        chain = [self.model]
        for m in self.fallback_models:
            if m not in chain:
                chain.append(m)
        return chain

    # ── public ──────────────────────────────────────────────────────────

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 600,
        temperature: float = 0.0,
    ) -> Optional[str]:
        """
        Return the assistant's text, or None if the client is disabled,
        the daily budget is exhausted, or every model in the chain fails.
        temperature defaults to 0.0 for reproducible narration.
        """
        if not self.api_key:
            logger.warning("OpenRouter API key missing — copilot disabled.")
            return None

        self._roll_day()
        if self._spend_usd >= self.daily_limit:
            logger.warning(
                "Copilot daily budget (~$%.2f) exhausted — skipping LLM call.",
                self.daily_limit,
            )
            return None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        client = self._get_client()
        last_exc: Optional[Exception] = None

        for model in self._model_chain():
            for attempt in range(self.MAX_RETRIES_PER_MODEL):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    content = (
                        resp.choices[0].message.content
                        if resp.choices and resp.choices[0].message
                        else None
                    )
                    if not content:
                        raise ValueError(f"Empty response from {model}")

                    usage = getattr(resp, "usage", None)
                    in_tok = getattr(usage, "prompt_tokens", 0) or 0
                    out_tok = getattr(usage, "completion_tokens", 0) or 0
                    cost = self._track_cost(model, in_tok, out_tok)
                    logger.info(
                        "Copilot completion via %s (%d tok, ~$%.4f; day ~$%.4f/$%.2f)",
                        model, in_tok + out_tok, cost, self._spend_usd, self.daily_limit,
                    )
                    return content.strip()

                except Exception as exc:  # noqa: BLE001 — log and fall through
                    last_exc = exc
                    s = str(exc).lower()
                    if any(k in s for k in _RETRYABLE) and attempt < self.MAX_RETRIES_PER_MODEL - 1:
                        time.sleep(self.BASE_BACKOFF * (2 ** attempt))
                        continue
                    logger.warning("Copilot model %s failed: %s", model, exc)
                    break  # move to next model in the chain

        logger.error("All copilot models failed. Last error: %s", last_exc)
        return None
