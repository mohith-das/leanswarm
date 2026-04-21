from __future__ import annotations

import asyncio

from leanswarm.engine.config import RuntimeSettings
from leanswarm.engine.simulator import LeanSwarmEngine


def test_smoke_simulation_returns_world(tmp_path) -> None:
    settings = RuntimeSettings(
        dry_run=True,
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
    )
    engine = LeanSwarmEngine(settings=settings)
    result = asyncio.run(engine.smoke_test())

    assert 80 <= len(engine.build_archetype_pool()) <= 150
    assert result.report.tick_count >= 1
    assert len(result.world.agents) <= 50
    assert result.report.llm_calls >= 1
