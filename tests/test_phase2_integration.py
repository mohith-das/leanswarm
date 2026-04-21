from __future__ import annotations

import asyncio
import json
import math
import random
from pathlib import Path
from typing import Any, cast

import pytest

from leanswarm.engine.config import RuntimeSettings
from leanswarm.engine.memory import HierarchicalMemoryManager
from leanswarm.engine.models import (
    ActivationMode,
    AgentState,
    SentimentLabel,
    SentimentSignal,
    SimulationRequest,
    TaskType,
    TickRecord,
    WorldProfile,
)
from leanswarm.engine.simulator import LeanSwarmEngine


def _make_engine(tmp_path: Path) -> LeanSwarmEngine:
    settings = RuntimeSettings(
        dry_run=True,
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
    )
    return LeanSwarmEngine(settings=settings)


def test_seed_document_flows_into_world_and_agent_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _make_engine(tmp_path)
    captured_calls: list[tuple[TaskType, dict[str, Any]]] = []
    original_route = engine.router.route

    async def tracking_route(task_type: TaskType | str, payload: dict[str, Any]) -> dict[str, Any]:
        resolved_task = TaskType(task_type)
        captured_calls.append((resolved_task, dict(payload)))
        return await original_route(task_type, payload)

    monkeypatch.setattr(engine.router, "route", tracking_route)

    request = SimulationRequest(
        seed_document=(
            "A city council rollout triggered mixed coverage, a sharp rise in complaint "
            "volume, and cautious optimism among local organizers."
        ),
        question="Will public sentiment improve by the next reporting cycle?",
        rounds=4,
        max_agents=16,
        active_agent_fraction=0.25,
        group_size=4,
        random_seed=19,
    )
    result = asyncio.run(engine.simulate(request))

    assert result.request == request
    assert result.report.question == request.question
    assert result.report.tick_count == len(result.ticks)
    assert 0 <= result.report.cache_hit_rate <= 1
    assert 0 < len(result.world.agents) <= request.max_agents
    assert len(result.world.edges) > 0
    assert result.report.llm_calls >= 3
    assert {task for task, _ in captured_calls} >= {
        TaskType.WORLD_BOOTSTRAP,
        TaskType.AGENT_BATCH,
        TaskType.PREDICTION_SYNTHESIS,
    }

    serialized_payloads = "\n".join(
        json.dumps(payload, sort_keys=True, default=str) for _, payload in captured_calls
    )
    assert "city council rollout" in serialized_payloads
    assert "public sentiment improve" in serialized_payloads


def test_population_shows_diversity_and_state_variation(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path)
    request = SimulationRequest(
        seed_document=(
            "Economic reporting is split across labor, housing, and consumer confidence "
            "with no single dominant narrative."
        ),
        question="Will confidence stabilize after the next news cycle?",
        rounds=5,
        max_agents=18,
        active_agent_fraction=0.3,
        group_size=5,
        random_seed=7,
    )
    result = asyncio.run(engine.simulate(request))

    archetypes = {agent.archetype for agent in result.world.agents}
    state_signatures = {
        (round(agent.mood, 3), round(agent.energy, 3), round(agent.attention, 3))
        for agent in result.world.agents
    }

    assert 0 < len(result.world.agents) <= request.max_agents
    assert len(archetypes) >= 3
    assert len(state_signatures) > 1
    assert any(edge.weight != 0 for edge in result.world.edges)
    assert all(
        0.0 <= value <= 1.0
        for agent in result.world.agents
        for value in (agent.mood, agent.energy, agent.attention)
    )


def test_phase2_ready_request_contract_roundtrips(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path)
    request = SimulationRequest(
        seed_document="Baseline coverage remains noisy but structurally stable.",
        question="Will the baseline stay stable through the month?",
        rounds=3,
        max_agents=10,
        active_agent_fraction=0.2,
        group_size=3,
        random_seed=11,
        use_llm=False,
    )
    result = asyncio.run(engine.simulate(request))

    assert result.request == request
    assert result.report.tick_count <= request.rounds
    assert len(result.world.agents) <= request.max_agents
    assert result.report.prediction
    assert isinstance(result.report.converged, bool)
    assert request.use_llm is False


def test_naive_activation_mode_activates_all_agents_each_tick(
    tmp_path: Path,
) -> None:
    engine = _make_engine(tmp_path)
    request = SimulationRequest(
        seed_document=(
            "Neighborhood updates remain contentious with recurring concerns and routine "
            "follow-up from local organizers."
        ),
        question="Will organizer engagement reduce complaints next week?",
        rounds=4,
        max_agents=14,
        active_agent_fraction=0.2,
        activation_mode=ActivationMode.NAIVE,
        random_seed=13,
        use_llm=False,
    )

    result = asyncio.run(engine.simulate(request))
    assert result.ticks
    assert all(len(tick.active_agent_ids) == len(result.world.agents) for tick in result.ticks)


def test_lean_activation_mode_stays_within_sparse_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _make_engine(tmp_path)
    agents = [
        AgentState(
            id=f"agent-{index}",
            name=f"Agent {index}",
            archetype="optimistic-watchdog-civic",
            mood=0.4 + (index % 3) * 0.1,
            energy=0.45 + (index % 2) * 0.08,
            attention=0.35 + (index % 4) * 0.05,
        )
        for index in range(20)
    ]
    request = SimulationRequest(
        seed_document="Calm updates continue with only minor attention shifts.",
        question="Will the headline remain stable?",
        rounds=1,
        max_agents=20,
        active_agent_fraction=0.18,
        activation_mode=ActivationMode.LEAN,
        random_seed=21,
        use_llm=False,
    )
    world_profile = WorldProfile(
        document_id="doc-1",
        question=request.question,
        summary="Calm coverage with moderate uncertainty and low volatility.",
        tone=SentimentLabel.NEUTRAL,
        sentiment=SentimentSignal(
            label=SentimentLabel.NEUTRAL,
            score=0.0,
            confidence=0.5,
        ),
        focus_terms=["calm", "coverage", "stability"],
        uncertainty=0.25,
        salience=0.2,
        complexity=0.3,
    )

    def fixed_fraction(*args: object, **kwargs: object) -> float:
        return 0.2

    monkeypatch.setattr(engine, "_activation_trigger_fraction", fixed_fraction)

    selected, activation_profile = engine._select_active_agents(
        agents,
        request=request,
        rng=random.Random(request.random_seed),
        world_profile=world_profile,
        ticks=[],
    )

    assert 0 < len(selected) < len(agents)
    assert int(activation_profile["target_count"]) == math.ceil(len(agents) * 0.2)
    assert len(selected) == int(activation_profile["target_count"])
    assert len(selected) <= math.ceil(len(agents) * 0.25)
    assert all(agent.id.startswith("agent-") for agent in selected)


def test_convergence_threshold_controls_stable_streak_break(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _make_engine(tmp_path)

    def always_stable(
        tick_index: int,
        active_agents: list[AgentState],
        actions: list[object],
        world_profile: object,
        rng: random.Random,
        **kwargs: Any,
    ) -> TickRecord:
        return TickRecord(
            tick=tick_index,
            active_agent_ids=[agent.id for agent in active_agents],
            events=[],
            mean_delta=0.0,
            stable=True,
        )

    monkeypatch.setattr(engine, "_apply_actions", always_stable)

    request = SimulationRequest(
        seed_document="Baseline signals are calm with minor fluctuations.",
        question="Will conditions remain steady?",
        rounds=6,
        max_agents=10,
        active_agent_fraction=0.25,
        convergence_threshold=3,
        random_seed=5,
        use_llm=False,
    )

    result = asyncio.run(engine.simulate(request))
    assert result.report.converged is True
    assert result.report.tick_count == request.convergence_threshold
    assert len(result.ticks) == request.convergence_threshold


def test_semantic_retrieval_and_memory_summary_payload_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = HierarchicalMemoryManager()
    agent = AgentState(id="agent-1", name="Ava", archetype="optimistic-watchdog-civic")
    memory.add_semantic_hint(agent, "city council reform")
    memory.add_semantic_hint(agent, "complaint triage process")
    agent.memory.episodic.extend(
        [
            "tick 1: complaints surged after rollout",
            "tick 2: reform messaging improved trust",
        ]
    )

    context = memory.retrieve_semantic_context(
        agent,
        query="Will city council reform reduce complaints?",
        limit=3,
    )
    assert context
    assert "city council reform" in context
    assert any("complaint" in item.lower() for item in context)

    engine = _make_engine(tmp_path)
    memory_payloads: list[dict[str, Any]] = []
    original_route = engine.router.route

    async def tracking_route(task_type: TaskType | str, payload: dict[str, Any]) -> dict[str, Any]:
        resolved = TaskType(task_type)
        if resolved is TaskType.MEMORY_SUMMARY:
            memory_payloads.append(dict(payload))
        return await original_route(task_type, payload)

    def always_summarize(_agent: AgentState) -> bool:
        return True

    monkeypatch.setattr(engine.memory, "should_summarize", always_summarize)
    monkeypatch.setattr(engine.router, "route", tracking_route)

    request = SimulationRequest(
        seed_document=(
            "Civic feedback remains mixed as city council reform plans face implementation "
            "questions and frequent complaint reviews."
        ),
        question="Will city council reform reduce complaints this month?",
        rounds=2,
        max_agents=10,
        active_agent_fraction=0.3,
        random_seed=17,
        use_llm=False,
    )
    asyncio.run(engine.simulate(request))

    assert memory_payloads
    semantic_contexts = [
        cast(list[object], payload["semantic_context"]) for payload in memory_payloads
    ]
    assert all("semantic_context" in payload for payload in memory_payloads)
    assert all(isinstance(context, list) for context in semantic_contexts)
    assert any(context for context in semantic_contexts)
