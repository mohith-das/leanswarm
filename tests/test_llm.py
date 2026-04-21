from __future__ import annotations

import asyncio

from lean_swarm.engine.config import RuntimeSettings
from lean_swarm.engine.llm import LiteLLMRouter
from lean_swarm.engine.models import TaskType


def test_router_records_cache_hit_rate(tmp_path) -> None:
    settings = RuntimeSettings(
        dry_run=True,
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
    )
    router = LiteLLMRouter(settings=settings)
    payload = {
        "question": "Will approval improve?",
        "agents": [{"id": "agent-1", "name": "Aster", "archetype": "optimistic-analyst-policy"}],
    }

    first = asyncio.run(router.route(TaskType.AGENT_BATCH, payload))
    second = asyncio.run(router.route(TaskType.AGENT_BATCH, payload))

    assert first == second
    assert router.cache_hits == 1
    assert router.cache_hit_rate > 0

