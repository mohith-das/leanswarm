from __future__ import annotations

import importlib.util

from leanswarm.engine.memory import HierarchicalMemoryManager
from leanswarm.engine.models import AgentState


def _vector_stack_available() -> bool:
    return all(
        importlib.util.find_spec(module) is not None
        for module in ("sentence_transformers", "sqlite_vss")
    )


def test_semantic_retrieval_prefers_relevant_context_and_handles_fallback() -> None:
    memory = HierarchicalMemoryManager()
    agent = AgentState(id="agent-1", name="Ava", archetype="optimistic-watchdog-civic")

    memory.add_semantic_hint(agent, "city council reform")
    memory.add_semantic_hint(agent, "complaint triage process")
    memory.add_semantic_hint(agent, "parking enforcement")
    agent.memory.episodic.extend(
        [
            "tick 1: complaints surged after rollout",
            "tick 2: quieter follow-up kept response steady",
        ]
    )

    ranked_context = memory.retrieve_semantic_context(
        agent,
        query="Will city council reform reduce complaints?",
        limit=3,
    )
    assert ranked_context
    assert len(ranked_context) <= 3
    assert len(ranked_context) == len(set(ranked_context))
    assert ranked_context[0] == "city council reform"
    assert any("complaint" in item.lower() for item in ranked_context)

    fallback_context = memory.retrieve_semantic_context(
        agent,
        query="Weather and transit updates are the only things moving today.",
        limit=5,
    )
    assert fallback_context[:3] == [
        "parking enforcement",
        "complaint triage process",
        "city council reform",
    ]
    assert fallback_context[3:] == [
        "tick 2: quieter follow-up kept response steady",
        "tick 1: complaints surged after rollout",
    ]

    if _vector_stack_available():
        assert ranked_context[0] in {
            "city council reform",
            "complaint triage process",
            "parking enforcement",
        }
