import asyncio
import json
from pathlib import Path
from typing import Any

from leanswarm.engine.config import RuntimeSettings
from leanswarm.engine.enrichment import apply_extraction
from leanswarm.engine.llm import _RESPONSE_MODELS, LiteLLMRouter
from leanswarm.engine.models import (
    ModelTier,
    SimulationRequest,
    TaskType,
    WorldEdgeKind,
    WorldExtractionResponse,
)
from leanswarm.engine.pricing import estimate_run
from leanswarm.engine.prompts import system_prompt
from leanswarm.engine.simulator import LeanSwarmEngine
from leanswarm.engine.world import build_seed_world

SEED_DOC = (
    "A recent memo from the Treasury Department outlines a split in public opinion "
    "regarding the new tax policy proposal. Older voters in the Midwest broadly oppose "
    "the measure, citing concerns about retirement income. The Treasury Department "
    "defends the proposal, arguing it will simplify the tax code. Meanwhile, the "
    "American Association of Retired Persons has launched a media campaign against it."
)


def test_extraction_response_tolerance():
    data = {
        "summary": "Test summary",
        "sentiment": "garbage",
        "topics": [
            {"label": "Tax Policy", "keywords": ["tax", "reform"], "salience": 0.8},
            {"label": "Public Opinion", "salience": "junk"},
        ],
        "entities": [
            {"label": "Treasury Department", "entity_type": "PERSON", "salience": "2.7"},
            {"label": "Older Voters", "entity_type": "alien", "salience": 0.9},
            "AARP",
            {"label": "Midwest", "entity_type": "location", "extra_key": True},
        ],
        "relations": [
            {"source": "Treasury Department", "target": "Tax Policy", "relation": "LOVES", "strength": 0.7},
            {"source": "Older Voters", "target": "Tax Policy", "relation": "opposes", "strength": 0.9},
            {"target": "Missing Source", "relation": "supports"},
        ],
    }
    parsed = WorldExtractionResponse.model_validate(data)

    assert parsed.sentiment.label.value == "neutral"
    assert parsed.entities[0].entity_type == "person"
    assert parsed.entities[0].salience == 1.0
    assert parsed.entities[1].entity_type == "concept"
    assert parsed.entities[2].label == "AARP"
    assert parsed.entities[3].entity_type == "location"

    assert parsed.relations[0].relation == "relates_to"
    assert parsed.relations[1].relation == "opposes"
    assert len(parsed.relations) == 2  # the missing-source one was dropped


def test_apply_extraction_enriches_world():
    seed_world = build_seed_world(SEED_DOC, question="Will the tax policy pass?")

    extraction = {
        "summary": "A memo about tax policy opposition from older voters.",
        "sentiment": {"label": "negative", "score": -0.4, "confidence": 0.8},
        "topics": [
            {"label": "Tax Policy", "keywords": ["tax", "reform"], "salience": 0.9},
            {"label": "Public Opinion", "keywords": ["voters"], "salience": 0.7},
        ],
        "entities": [
            {"label": "Treasury Department", "entity_type": "organization", "salience": 0.9, "evidence": "defends the proposal"},
            {"label": "Older Voters", "entity_type": "group", "salience": 0.8, "evidence": "broadly oppose"},
            {"label": "Tax Policy Proposal", "entity_type": "policy", "salience": 0.95, "evidence": "new tax policy"},
        ],
        "relations": [
            {"source": "Treasury Department", "target": "Tax Policy Proposal", "relation": "supports", "strength": 0.8, "evidence": "defends"},
            {"source": "Older Voters", "target": "Tax Policy Proposal", "relation": "opposes", "strength": 0.9, "evidence": "broadly oppose"},
            {"source": "Older Voters", "target": "Nonexistent Entity", "relation": "influences", "strength": 0.5},
        ],
    }

    result = apply_extraction(seed_world, extraction)

    assert result.profile.extraction_source == "llm"
    entity_labels = [e.label for e in result.profile.entities]
    assert entity_labels == ["Treasury Department", "Older Voters", "Tax Policy Proposal"]

    supports_edges = [e for e in result.graph.edges if e.relation == WorldEdgeKind.SUPPORTS]
    assert len(supports_edges) == 1

    nonexistent_edges = [
        e for e in result.graph.edges
        if e.relation == WorldEdgeKind.INFLUENCES and "Nonexistent" in str(e.evidence)
    ]
    assert len(nonexistent_edges) == 0

    assert 0.0 <= result.graph.density <= 1.0

    doc_nodes = [n for n in result.graph.nodes if n.kind.value == "document"]
    assert len(doc_nodes) == 1
    assert result.graph.root_id == doc_nodes[0].id

    question_nodes = [n for n in result.graph.nodes if n.kind.value == "question"]
    assert len(question_nodes) >= 1


def test_apply_extraction_empty_keeps_deterministic_world():
    seed_world = build_seed_world(SEED_DOC, question="Will the tax policy pass?")

    sentinel = {
        "summary": "",
        "sentiment": {"label": "neutral", "score": 0.0, "confidence": 0.0},
        "topics": [],
        "entities": [],
        "relations": [],
    }

    result = apply_extraction(seed_world, sentinel)
    assert result.model_dump() == seed_world.model_dump()
    assert result.profile.extraction_source == "deterministic"


def test_extraction_task_wiring():
    router = LiteLLMRouter(RuntimeSettings(dry_run=True))

    assert router._tier_for_task(TaskType.WORLD_EXTRACTION) is ModelTier.CHEAP
    assert TaskType.WORLD_EXTRACTION in _RESPONSE_MODELS

    prompt = system_prompt(TaskType.WORLD_EXTRACTION)
    assert "entities" in prompt
    assert "relations" in prompt
    assert "supports|opposes" in prompt

    mock_resp = router._mock_response(TaskType.WORLD_EXTRACTION, {"question": "q"})
    assert mock_resp == {
        "summary": "",
        "sentiment": {"label": "neutral", "score": 0.0, "confidence": 0.0},
        "topics": [],
        "entities": [],
        "relations": [],
    }


def test_dry_run_makes_no_extraction_call(tmp_path):
    settings = RuntimeSettings(
        dry_run=True,
        cache_dir=tmp_path / "c",
        log_dir=tmp_path / "l",
        semantic_store_path=tmp_path / "s.sqlite3",
    )
    engine = LeanSwarmEngine(settings=settings)
    result = asyncio.run(engine.smoke_test())

    log_path = settings.llm_log_path
    log_text = log_path.read_text() if log_path.exists() else ""
    for line in log_text.strip().splitlines():
        if line.strip():
            entry = json.loads(line)
            assert entry.get("task_type") != "world_extraction", "Dry run made an extraction call!"

    assert result.world.profile.extraction_source == "deterministic"


class RecordingRouter:
    def __init__(self, extraction_response: dict[str, Any], tmp_path: Path) -> None:
        self.calls: list[str] = []
        self.route_calls = 0
        self.cache_hits = 0
        self.cache_hit_rate = 0.0
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0
        self.prompt_tokens_by_model: dict[str, int] = {}
        self.completion_tokens_by_model: dict[str, int] = {}
        self._extraction = extraction_response
        self._mock = LiteLLMRouter(
            RuntimeSettings(
                dry_run=True,
                cache_dir=tmp_path / "c",
                log_dir=tmp_path / "l",
                semantic_store_path=tmp_path / "s.sqlite3",
            )
        )

    async def route(self, task_type: Any, payload: Any) -> dict[str, Any]:
        resolved = TaskType(task_type)
        self.calls.append(resolved.value)
        self.route_calls += 1
        if resolved is TaskType.WORLD_EXTRACTION:
            return dict(self._extraction)
        return self._mock._mock_response(resolved, payload)


def test_live_run_extracts_once_before_bootstrap(tmp_path):
    extraction = {
        "summary": "A memo about tax policy.",
        "sentiment": {"label": "negative", "score": -0.3, "confidence": 0.7},
        "topics": [{"label": "Tax Reform", "keywords": ["tax"], "salience": 0.9}],
        "entities": [
            {"label": "Treasury Dept", "entity_type": "organization", "salience": 0.9, "evidence": "memo"},
            {"label": "Older Voters", "entity_type": "group", "salience": 0.8, "evidence": "oppose"},
        ],
        "relations": [
            {"source": "Treasury Dept", "target": "Tax Reform", "relation": "supports", "strength": 0.8},
            {"source": "Older Voters", "target": "Tax Reform", "relation": "opposes", "strength": 0.9},
        ],
    }

    settings = RuntimeSettings(
        dry_run=False,
        cache_dir=tmp_path / "c",
        log_dir=tmp_path / "l",
        semantic_store_path=tmp_path / "s.sqlite3",
    )
    router = RecordingRouter(extraction, tmp_path)
    engine = LeanSwarmEngine(settings=settings, router=router)

    request = SimulationRequest(
        seed_document=SEED_DOC,
        question="Will the tax policy pass?",
        rounds=2,
        max_agents=6,
        use_llm=True,
    )
    result = asyncio.run(engine.simulate(request))

    assert router.calls.count("world_extraction") == 1
    assert router.calls.index("world_extraction") < router.calls.index("world_bootstrap")
    assert result.world.profile.extraction_source == "llm"

    entity_labels = [e.label for e in result.world.profile.entities]
    assert "Treasury Dept" in entity_labels


def test_estimate_run_includes_extraction():
    base_kwargs = dict(
        rounds=2,
        max_agents=8,
        active_agent_fraction=0.2,
        group_size=4,
        flagship_model="deepseek/deepseek-reasoner",
        standard_model="deepseek/deepseek-chat",
        cheap_model="deepseek/deepseek-chat",
    )

    est_zero = estimate_run(seed_chars=0, **base_kwargs)
    est_6k = estimate_run(seed_chars=6000, **base_kwargs)
    est_12k = estimate_run(seed_chars=12000, **base_kwargs)
    est_120k = estimate_run(seed_chars=120000, **base_kwargs)

    assert est_6k["calls_min"] == est_zero["calls_min"]
    assert est_6k["calls_max"] == est_zero["calls_max"]

    assert est_6k["prompt_tokens_est"] > est_zero["prompt_tokens_est"]
    assert est_6k["cost_max_usd"] > est_zero["cost_max_usd"]

    assert est_120k["prompt_tokens_est"] == est_12k["prompt_tokens_est"]
    assert est_120k["cost_max_usd"] == est_12k["cost_max_usd"]
