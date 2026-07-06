"""Map a validated WORLD_EXTRACTION response onto the deterministic seed world.

Pure functions only: no I/O, no LLM calls. If the extraction is empty (the mock
sentinel, or a model that found nothing), the deterministic seed world is returned
unchanged so live runs degrade gracefully to today's behavior.
"""

from __future__ import annotations

import hashlib
from typing import Any

from leanswarm.engine.models import (
    SeedEntity,
    SeedTopic,
    SeedWorld,
    SentimentSignal,
    WorldEdge,
    WorldEdgeKind,
    WorldExtractionResponse,
    WorldGraph,
    WorldNode,
    WorldNodeKind,
    WorldProfile,
)

# The extraction call reads at most this many characters of the seed document.
# Single pass, no chunking — this bound is what keeps the feature ~1 cheap call.
EXTRACTION_MAX_CHARS = 6000

_RELATION_EDGE_KINDS: dict[str, WorldEdgeKind] = {
    "supports": WorldEdgeKind.SUPPORTS,
    "opposes": WorldEdgeKind.OPPOSES,
    "influences": WorldEdgeKind.INFLUENCES,
    "part_of": WorldEdgeKind.PART_OF,
    "reports_on": WorldEdgeKind.REPORTS_ON,
    "causes": WorldEdgeKind.CAUSES,
    "targets": WorldEdgeKind.TARGETS,
    "relates_to": WorldEdgeKind.RELATES_TO,
}


def _stable_id(prefix: str, label: str) -> str:
    digest = hashlib.sha256(label.strip().lower().encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _dedupe(labels: list[str], limit: int) -> list[str]:
    seen: list[str] = []
    for label in labels:
        if label and label not in seen:
            seen.append(label)
        if len(seen) >= limit:
            break
    return seen


def apply_extraction(seed_world: SeedWorld, extraction: dict[str, Any]) -> SeedWorld:
    parsed = WorldExtractionResponse.model_validate(extraction)
    if not parsed.entities:
        return seed_world

    profile = seed_world.profile

    entities = [
        SeedEntity(
            id=_stable_id("entity", item.label),
            label=item.label,
            entity_type=item.entity_type,
            mentions=1,
            score=item.salience,
            evidence=[item.evidence] if item.evidence else [],
        )
        for item in parsed.entities[:16]
    ]
    topics = [
        SeedTopic(
            id=_stable_id("topic", item.label),
            label=item.label,
            score=item.salience,
            keywords=item.keywords or [item.label.lower()],
            mentions=1,
        )
        for item in parsed.topics[:8]
    ] or list(profile.topics)

    sentiment = SentimentSignal(
        label=parsed.sentiment.label,
        score=parsed.sentiment.score,
        confidence=parsed.sentiment.confidence,
        scope="document",
    )

    focus_terms = _dedupe(
        [topic.label for topic in topics]
        + [entity.label for entity in entities]
        + [term for term in profile.focus_terms if term.islower()],
        limit=16,
    )

    new_profile: WorldProfile = profile.model_copy(
        update={
            "summary": parsed.summary or profile.summary,
            "tone": parsed.sentiment.label,
            "sentiment": sentiment,
            "sentiment_signals": [sentiment],
            "topics": topics,
            "entities": entities,
            "focus_terms": focus_terms,
            "extraction_source": "llm",
        }
    )

    new_graph = _build_graph(seed_world.graph, parsed, entities, topics, sentiment)
    return seed_world.model_copy(update={"profile": new_profile, "graph": new_graph})


def _find_node(graph: WorldGraph, kind: WorldNodeKind) -> WorldNode | None:
    for node in graph.nodes:
        if node.kind is kind:
            return node
    return None


def _build_graph(
    old_graph: WorldGraph,
    parsed: WorldExtractionResponse,
    entities: list[SeedEntity],
    topics: list[SeedTopic],
    sentiment: SentimentSignal,
) -> WorldGraph:
    doc_node = _find_node(old_graph, WorldNodeKind.DOCUMENT) or WorldNode(
        id="node-document", label="Seed document", kind=WorldNodeKind.DOCUMENT, weight=1.0
    )
    question_node = _find_node(old_graph, WorldNodeKind.QUESTION)

    sentiment_node = WorldNode(
        id=_stable_id("node", f"sentiment:{sentiment.label.value}"),
        label=sentiment.label.value.capitalize(),
        kind=WorldNodeKind.SENTIMENT,
        weight=max(0.0, min(1.0, abs(sentiment.score))),
        attributes={"score": sentiment.score, "confidence": sentiment.confidence},
    )

    label_to_node: dict[str, WorldNode] = {}
    nodes: list[WorldNode] = [doc_node]
    if question_node is not None:
        nodes.append(question_node)
    nodes.append(sentiment_node)

    for entity in entities:
        node = WorldNode(
            id=_stable_id("node", f"entity:{entity.label}"),
            label=entity.label,
            kind=WorldNodeKind.ENTITY,
            weight=entity.score,
            evidence=entity.evidence,
            attributes={"entity_type": entity.entity_type},
        )
        nodes.append(node)
        label_to_node[entity.label.strip().lower()] = node

    for topic in topics:
        node = WorldNode(
            id=_stable_id("node", f"topic:{topic.label}"),
            label=topic.label,
            kind=WorldNodeKind.TOPIC,
            weight=topic.score,
            attributes={"keywords": list(topic.keywords)},
        )
        nodes.append(node)
        label_to_node.setdefault(topic.label.strip().lower(), node)

    edges: list[WorldEdge] = []
    for entity in entities:
        edges.append(
            WorldEdge(
                source=doc_node.id,
                target=_stable_id("node", f"entity:{entity.label}"),
                weight=entity.score,
                relation=WorldEdgeKind.MENTIONS,
                evidence=entity.evidence,
            )
        )
    for topic in topics:
        edges.append(
            WorldEdge(
                source=doc_node.id,
                target=_stable_id("node", f"topic:{topic.label}"),
                weight=topic.score,
                relation=WorldEdgeKind.CONTAINS,
            )
        )
        if question_node is not None:
            edges.append(
                WorldEdge(
                    source=question_node.id,
                    target=_stable_id("node", f"topic:{topic.label}"),
                    weight=0.9,
                    relation=WorldEdgeKind.FOCUSES_ON,
                )
            )
    edges.append(
        WorldEdge(source=doc_node.id, target=sentiment_node.id, weight=1.0, relation=WorldEdgeKind.FRAMES)
    )

    for relation in parsed.relations[:24]:
        source = label_to_node.get(relation.source.strip().lower())
        target = label_to_node.get(relation.target.strip().lower())
        if source is None or target is None or source.id == target.id:
            continue
        edges.append(
            WorldEdge(
                source=source.id,
                target=target.id,
                weight=relation.strength,
                relation=_RELATION_EDGE_KINDS.get(relation.relation, WorldEdgeKind.RELATES_TO),
                evidence=[relation.evidence] if relation.evidence else [],
            )
        )

    node_count = len(nodes)
    edge_count = len(edges)
    density = 0.0
    if node_count > 1:
        density = min(1.0, (2 * edge_count) / (node_count * (node_count - 1)))
    average_degree = (2 * edge_count / node_count) if node_count else 0.0

    return WorldGraph(
        nodes=nodes,
        edges=edges,
        root_id=doc_node.id,
        density=round(density, 3),
        average_degree=round(average_degree, 3),
    )
