"""Stateless post-run chat: build a chat payload from a finished run's data.

No storage: the client sends the rolling history each turn. One router call per
message. Works identically for in-memory jobs and DB-loaded (saved/public) runs.
"""

from __future__ import annotations

from typing import Any

from leanswarm.engine.llm import LiteLLMRouter
from leanswarm.engine.models import TaskType

MAX_HISTORY_TURNS = 8
MAX_MESSAGE_CHARS = 1000


def _find_agent(result: dict[str, Any], agent_id: str) -> dict[str, Any] | None:
    for agent in result.get("world", {}).get("agents", []):
        if agent.get("id") == agent_id:
            return agent
    return None


async def run_chat(
    router: LiteLLMRouter,
    result: dict[str, Any],
    agent_id: str | None,
    message: str,
    history: list[dict[str, str]],
    live: bool,
) -> dict[str, Any]:
    """Returns {"reply": str, "prompt_tokens": int, "completion_tokens": int}."""
    message = message[:MAX_MESSAGE_CHARS]
    trimmed_history = [
        {"role": str(t.get("role", "user"))[:16], "content": str(t.get("content", ""))[:500]}
        for t in history[-MAX_HISTORY_TURNS:]
    ]
    report = result.get("report", {})
    profile = result.get("world", {}).get("profile", {}) or {}

    before_prompt = router.prompt_tokens_total
    before_completion = router.completion_tokens_total

    if agent_id:
        agent = _find_agent(result, agent_id)
        if agent is None:
            raise KeyError(agent_id)
        response = await router.route(
            TaskType.AGENT_CHAT,
            {
                "use_llm": live,
                "question": report.get("question", ""),
                "agent_id": agent_id,
                "agent_name": agent.get("name", ""),
                "archetype": agent.get("archetype", ""),
                "persona": agent.get("persona") or "",
                "stance": agent.get("stance") or "",
                "mood": agent.get("mood", 0.5),
                "memories": (agent.get("memory", {}).get("episodic") or [])[:6],
                "world_summary": profile.get("summary", ""),
                "history": trimmed_history,
                "message": message,
            },
        )
    else:
        response = await router.route(
            TaskType.REPORT_CHAT,
            {
                "use_llm": live,
                "question": report.get("question", ""),
                "prediction": report.get("prediction", ""),
                "confidence": report.get("confidence", 0.5),
                "rationale": report.get("rationale", []),
                "tick_count": report.get("tick_count", 0),
                "key_events": report.get("key_events", []),
                "tick_events": [
                    event
                    for tick in result.get("ticks", [])[:8]
                    for event in tick.get("events", [])[:3]
                ],
                "world_summary": profile.get("summary", ""),
                "history": trimmed_history,
                "message": message,
            },
        )
    return {
        "reply": str(response.get("reply", "")),
        "prompt_tokens": router.prompt_tokens_total - before_prompt,
        "completion_tokens": router.completion_tokens_total - before_completion,
    }
