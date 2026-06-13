"""
Quant Edge Copilot — a NARRATION-ONLY assistant over the bot's own picks.

The copilot answers natural-language questions in Telegram ("why do we like
Bam over 3.5 assists?") by feeding the LLM the structured alerts_sent rows the
quant stack already produced and asking it to *explain* them. It is forbidden
from computing edges or inventing statistics: every number it cites must come
verbatim from the context we hand it. The edge math lives in edge_ranker.py /
devig.py and stays there.
"""

import json
from typing import List, Optional

from src.config import COPILOT_CONTEXT_ALERTS, COPILOT_CONTEXT_HOURS
from src.clients.openrouter_client import OpenRouterClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_DISCLAIMER = "Informational only — bet responsibly."

SYSTEM_PROMPT = f"""You are the Quant Edge Copilot for an NBA player-prop betting bot.

You explain bets the quant stack (XGBoost projections + probability \
calibration + an edge ranker) has ALREADY produced. You are a narration layer, \
NOT a model and NOT a decision-maker.

HARD RULES — follow exactly:
1. Use ONLY the numbers present in the CONTEXT JSON. Never invent, estimate, or \
recall any statistic (player averages, on/off splits, usage rates, pace, injury \
news, etc.). If a figure is not in the context, say: "I don't have that in the \
current pick data."
2. Never do probability or edge math yourself. The `edge`, `model_prob`, and \
`implied_prob_from_odds` fields are authoritative and already computed — quote \
them, do not recompute or "sanity check" them.
3. If the user asks about a player, market, or game not present in the context, \
say it isn't in the current alerts. Do not guess.
4. Be concise: 2-5 sentences. Field meanings: `model_prob` is the calibrated \
model probability; `implied_prob_from_odds` is 1/decimal_odds (raw, INCLUDES \
the book's vig); `edge` is the ranker's FINAL edge after its adjustments \
(fragility, steam, consensus, bias) — so `edge` will NOT exactly equal \
`model_prob - implied_prob_from_odds`, and that is expected. Quote `edge` as-is; \
never present a hand-computed difference as the edge.
5. Never tell the user how much to bet beyond repeating the `stake` field if \
present. Do not give financial advice.
6. Always end your reply with exactly this sentence: "{_DISCLAIMER}"

You may explain what the fields mean and rank/summarize what is in the context, \
but every concrete figure must come straight from the context."""


class QuantCopilot:
    """Narration layer that grounds an LLM strictly on recent alerts_sent rows."""

    def __init__(self, db, client: Optional[OpenRouterClient] = None):
        self.db = db
        self.client = client or OpenRouterClient()

    def _build_context(self) -> List[dict]:
        """Pull recent picks and reshape into the minimal JSON the LLM may cite."""
        try:
            rows = self.db.get_recent_alerts(
                limit=COPILOT_CONTEXT_ALERTS, hours_back=COPILOT_CONTEXT_HOURS
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Copilot could not load recent alerts: %s", exc)
            return []

        context: List[dict] = []
        for r in rows:
            odds = r.get("odds") or 0
            # Raw implied prob = 1/odds (still includes the book's vig). This is a
            # deterministic conversion of the real odds, computed here so the model
            # never does arithmetic. NOTE: the stored `edge` is the ranker's final
            # adjusted edge and intentionally won't equal model_prob - this value.
            implied = round(1.0 / odds, 4) if odds else None
            model_prob = r.get("model_prob")
            edge = r.get("edge")
            won = r.get("won")
            if won is None:
                settled = "pending"
            elif won:
                settled = "won"
            elif r.get("push"):
                settled = "push"
            else:
                settled = "lost"

            context.append({
                "player": r.get("player_name"),
                "market": r.get("market"),
                "side": r.get("side"),
                "line": r.get("line"),
                "book": r.get("book"),
                "odds_decimal": odds or None,
                "implied_prob_from_odds": implied,
                "model_prob": round(model_prob, 4) if model_prob is not None else None,
                "edge": round(edge, 4) if edge is not None else None,
                "stake": r.get("stake"),
                "alerted_at": str(r.get("timestamp")),
                "result": settled,
            })
        return context

    def answer(self, question: str) -> str:
        """Answer a user question grounded only on current picks. Always returns text."""
        question = (question or "").strip()
        if not question:
            return f"Ask me about a current pick (player, market, or the edge). {_DISCLAIMER}"

        context = self._build_context()
        if not context:
            return (
                "There are no picks in the current window, so there's nothing for me "
                f"to explain yet. {_DISCLAIMER}"
            )

        user_prompt = (
            "CONTEXT (current picks — the ONLY data you may cite, as JSON):\n"
            f"{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
            f"USER QUESTION:\n{question}"
        )

        reply = self.client.complete(SYSTEM_PROMPT, user_prompt)
        if not reply:
            return (
                "The copilot is unavailable right now (model error or daily budget "
                f"reached). Try again later. {_DISCLAIMER}"
            )

        # Defensive: guarantee the disclaimer is present even if the model drops it.
        if _DISCLAIMER not in reply:
            reply = f"{reply}\n\n{_DISCLAIMER}"
        return reply
