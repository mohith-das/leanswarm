from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

from leanswarm.engine.models import (
    SeedDocumentProfile,
    SeedEntity,
    SeedTopic,
    SeedWorld,
    SentimentLabel,
    SentimentSignal,
    WorldEdge,
    WorldEdgeKind,
    WorldGraph,
    WorldNode,
    WorldNodeKind,
    WorldProfile,
)

__all__ = [
    "build_seed_world",
    "build_world_graph",
    "build_world_profile",
    "extract_entities",
    "extract_sentiment_signals",
    "extract_topics",
    "ingest_seed_document",
]

_ARTICLE_WORDS = {"a", "an", "the"}
_QUESTION_WORDS = {
    "how",
    "what",
    "when",
    "where",
    "why",
    "who",
    "which",
    "will",
    "would",
    "could",
    "should",
    "can",
}
_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "must",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "won't",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}
_POSITIVE_TERMS = {
    "advance",
    "benefit",
    "better",
    "boost",
    "bright",
    "build",
    "calm",
    "clear",
    "confidence",
    "constructive",
    "continue",
    "encourage",
    "expand",
    "favorable",
    "gain",
    "growth",
    "improve",
    "improving",
    "increase",
    "launch",
    "optimistic",
    "opportunity",
    "positive",
    "progress",
    "resilient",
    "stabilize",
    "stable",
    "support",
    "upbeat",
    "win",
}
_NEGATIVE_TERMS = {
    "against",
    "alarm",
    "anxious",
    "bad",
    "breakdown",
    "concern",
    "crisis",
    "decline",
    "delay",
    "disagree",
    "down",
    "drop",
    "failure",
    "fear",
    "fragile",
    "harm",
    "loss",
    "negative",
    "oppose",
    "pressure",
    "problem",
    "risk",
    "rough",
    "slip",
    "slow",
    "struggle",
    "uncertain",
    "uncertainty",
    "weak",
    "worry",
}
_ORG_SUFFIXES = {
    "agency",
    "association",
    "board",
    "center",
    "centre",
    "committee",
    "company",
    "corp",
    "corporation",
    "department",
    "foundation",
    "group",
    "inc",
    "institute",
    "llc",
    "ltd",
    "ministry",
    "organization",
    "party",
    "society",
    "team",
    "union",
    "university",
}
_LOCATION_SUFFIXES = {
    "bay",
    "borough",
    "city",
    "county",
    "district",
    "harbor",
    "heights",
    "island",
    "lake",
    "province",
    "river",
    "state",
    "street",
    "town",
    "valley",
}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][\w'&.-]*|[A-Z]{2,})(?:\s+(?:[A-Z][\w'&.-]*|[A-Z]{2,}))*\b")
_WHITESPACE_RE = re.compile(r"\s+")


class _EntityRecord(TypedDict):
    label: str
    aliases: set[str]
    mentions: int
    evidence: list[str]
    entity_type: str


class _EdgeBucket(TypedDict):
    weight: float
    count: int
    evidence: list[str]


def _new_entity_record(label: str) -> _EntityRecord:
    return {
        "label": label,
        "aliases": set(),
        "mentions": 0,
        "evidence": [],
        "entity_type": _infer_entity_type(label),
    }


def ingest_seed_document(seed_document: str, question: str = "") -> SeedDocumentProfile:
    source_text, source_path = _load_seed_text(seed_document)
    paragraph_source = source_text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [
        segment.strip() for segment in re.split(r"\n{2,}", paragraph_source) if segment.strip()
    ]
    normalized_text = _normalize_text(source_text)
    sentences = _split_sentences(normalized_text)
    tokens = _tokenize(normalized_text)
    question_tokens = [token for token in _tokenize(question) if token not in _STOPWORDS]
    token_counts = Counter(token for token in tokens if token not in _STOPWORDS and len(token) > 2)
    top_terms = [term for term, _ in token_counts.most_common(12)]
    document_id = _stable_id("seed", f"{normalized_text}\n{question}")
    return SeedDocumentProfile(
        document_id=document_id,
        source_text=source_text,
        normalized_text=normalized_text,
        question=question,
        source_path=source_path,
        paragraphs=paragraphs,
        sentences=sentences,
        tokens=tokens,
        question_tokens=question_tokens,
        token_count=len(tokens),
        unique_token_count=len(set(tokens)),
        paragraph_count=len(paragraphs),
        sentence_count=len(sentences),
        top_terms=top_terms,
    )


def extract_entities(profile: SeedDocumentProfile, limit: int = 12) -> list[SeedEntity]:
    records: dict[str, _EntityRecord] = {}
    sentences = profile.sentences or _split_sentences(profile.normalized_text)

    for match in _ENTITY_RE.finditer(profile.source_text):
        label = _clean_entity_label(match.group(0))
        if not label:
            continue
        if len(label.split()) == 1 and label.lower() in _STOPWORDS:
            continue
        key = label.lower()
        record = records.setdefault(key, _new_entity_record(label))
        record["mentions"] += 1
        if record["label"] != label:
            record["aliases"].add(label)
        evidence = _sentence_hits(sentences, label, limit=2)
        record["evidence"] = _merge_evidence(record["evidence"], evidence, limit=2)

    entities = [
        SeedEntity(
            id=_stable_id("entity", key),
            label=record["label"],
            entity_type=record["entity_type"],
            mentions=record["mentions"],
            score=_entity_score(record["mentions"], record["label"]),
            aliases=sorted(record["aliases"]),
            evidence=record["evidence"],
        )
        for key, record in records.items()
    ]
    entities.sort(key=lambda entity: (-entity.mentions, -entity.score, entity.label.lower()))
    return entities[:limit]


def extract_topics(
    profile: SeedDocumentProfile,
    entities: Iterable[SeedEntity] | None = None,
    limit: int = 8,
) -> list[SeedTopic]:
    entity_list = list(entities) if entities is not None else []
    question_terms = set(profile.question_tokens)
    content_tokens = [
        token for token in profile.tokens if token not in _STOPWORDS and len(token) > 2
    ]
    unigram_counts = Counter(content_tokens)
    phrase_counts: Counter[str] = Counter()
    phrase_evidence: defaultdict[str, list[str]] = defaultdict(list)

    for sentence in profile.sentences or _split_sentences(profile.normalized_text):
        sentence_tokens = [
            token for token in _tokenize(sentence) if token not in _STOPWORDS and len(token) > 2
        ]
        for size in (2, 3):
            for index in range(0, max(0, len(sentence_tokens) - size + 1)):
                phrase = " ".join(sentence_tokens[index : index + size])
                if phrase in _STOPWORDS:
                    continue
                phrase_counts[phrase] += 1
                if len(phrase_evidence[phrase]) < 2:
                    phrase_evidence[phrase].append(sentence.strip())

    candidate_scores: dict[str, float] = {}
    for token, count in unigram_counts.items():
        if token not in _STOPWORDS:
            candidate_scores[token] = candidate_scores.get(token, 0.0) + float(count)
            if token in question_terms:
                candidate_scores[token] += 0.75

    for phrase, count in phrase_counts.items():
        phrase_score = float(count) * (1.4 if " " in phrase else 1.0)
        if question_terms.intersection(phrase.split()):
            phrase_score *= 1.2
        candidate_scores[phrase] = candidate_scores.get(phrase, 0.0) + phrase_score

    for entity in entity_list:
        for token in entity.label.lower().split():
            if token in candidate_scores:
                candidate_scores[token] += min(0.75, entity.score * 0.4)

    if not candidate_scores:
        return []

    max_score = max(candidate_scores.values()) or 1.0
    entity_labels = [entity.label for entity in entity_list]
    topics: list[SeedTopic] = []

    sorted_candidates = sorted(
        candidate_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    for label, raw_score in sorted_candidates[:limit]:
        evidence = _sentence_hits(profile.sentences, label, limit=2)
        if not evidence:
            evidence = phrase_evidence.get(label, [])
        related_entities = [
            entity_label
            for entity_label in entity_labels
            if _label_matches_text(entity_label, label) or _label_matches_any(entity_label, [label])
        ]
        topics.append(
            SeedTopic(
                id=_stable_id("topic", label),
                label=_title_case_label(label),
                score=round(min(1.0, raw_score / max_score), 3),
                keywords=_topic_keywords(label),
                mentions=max(1, int(round(raw_score))),
                evidence=evidence[:2],
                related_entities=sorted(dict.fromkeys(related_entities))[:4],
            )
        )

    return topics


def extract_sentiment_signals(profile: SeedDocumentProfile) -> list[SentimentSignal]:
    sentences = profile.sentences or _split_sentences(profile.normalized_text)
    signals: list[SentimentSignal] = []
    aggregate_score = 0.0
    aggregate_weight = 0.0

    for index, sentence in enumerate(sentences):
        tokens = _tokenize(sentence)
        positive_terms = [token for token in tokens if token in _POSITIVE_TERMS]
        negative_terms = [token for token in tokens if token in _NEGATIVE_TERMS]
        total_terms = len(positive_terms) + len(negative_terms)
        if not total_terms:
            continue
        raw_score = (len(positive_terms) - len(negative_terms)) / max(1, total_terms)
        label = _sentiment_label(raw_score, positive_terms, negative_terms)
        confidence = min(1.0, 0.25 + total_terms / 6.0)
        signal = SentimentSignal(
            label=label,
            score=round(max(-1.0, min(1.0, raw_score)), 3),
            confidence=round(confidence, 3),
            scope="sentence",
            span_index=index,
            positive_terms=positive_terms[:4],
            negative_terms=negative_terms[:4],
            evidence=[sentence.strip()],
        )
        signals.append(signal)
        aggregate_score += signal.score * total_terms
        aggregate_weight += total_terms

    if signals:
        aggregate = aggregate_score / max(1.0, aggregate_weight)
        positive_terms = [term for signal in signals for term in signal.positive_terms]
        negative_terms = [term for signal in signals for term in signal.negative_terms]
        evidence = [signal.evidence[0] for signal in signals[:3] if signal.evidence]
        signals.append(
            SentimentSignal(
                label=_sentiment_label(aggregate, positive_terms, negative_terms),
                score=round(max(-1.0, min(1.0, aggregate)), 3),
                confidence=round(min(1.0, 0.4 + abs(aggregate) * 0.4 + len(signals) / 12.0), 3),
                scope="document",
                span_index=None,
                positive_terms=sorted(dict.fromkeys(positive_terms))[:6],
                negative_terms=sorted(dict.fromkeys(negative_terms))[:6],
                evidence=evidence,
            )
        )
        return signals

    return [
        SentimentSignal(
            label=SentimentLabel.NEUTRAL,
            score=0.0,
            confidence=0.0,
            scope="document",
            evidence=[],
        )
    ]


def build_world_profile(
    profile: SeedDocumentProfile,
    *,
    entities: Iterable[SeedEntity] | None = None,
    topics: Iterable[SeedTopic] | None = None,
    sentiment_signals: Iterable[SentimentSignal] | None = None,
) -> WorldProfile:
    extracted_entities = list(entities) if entities is not None else extract_entities(profile)
    extracted_topics = (
        list(topics) if topics is not None else extract_topics(profile, extracted_entities)
    )
    extracted_sentiments = (
        list(sentiment_signals)
        if sentiment_signals is not None
        else extract_sentiment_signals(profile)
    )
    aggregate_sentiment = _document_sentiment(extracted_sentiments)

    top_topic_labels = [topic.label for topic in extracted_topics[:3]]
    top_entity_labels = [entity.label for entity in extracted_entities[:3]]
    focus_terms = list(
        dict.fromkeys(
            [
                *profile.top_terms[:4],
                *top_topic_labels,
                *top_entity_labels,
                *profile.question_tokens[:4],
            ]
        )
    )
    summary = _build_summary(
        top_topic_labels,
        top_entity_labels,
        aggregate_sentiment,
        profile.question,
    )

    uncertainty = _clamp(
        0.25
        + (0.18 if aggregate_sentiment.label is SentimentLabel.MIXED else 0.0)
        + (0.12 if aggregate_sentiment.label is SentimentLabel.NEUTRAL else 0.0)
        + min(0.2, len(extracted_topics) * 0.02)
        + min(0.15, len(extracted_entities) * 0.015),
    )
    salience = _clamp(
        0.25
        + (extracted_topics[0].score * 0.35 if extracted_topics else 0.0)
        + (extracted_entities[0].score * 0.2 if extracted_entities else 0.0)
        + (abs(aggregate_sentiment.score) * 0.2),
    )
    complexity = _clamp(
        0.18
        + min(0.35, profile.sentence_count / 25.0)
        + min(0.25, len(extracted_topics) * 0.03)
        + min(0.2, len(extracted_entities) * 0.02),
    )

    return WorldProfile(
        document_id=profile.document_id,
        question=profile.question,
        summary=summary,
        tone=aggregate_sentiment.label,
        sentiment=aggregate_sentiment,
        sentiment_signals=extracted_sentiments,
        topics=extracted_topics,
        entities=extracted_entities,
        focus_terms=focus_terms,
        uncertainty=round(uncertainty, 3),
        salience=round(salience, 3),
        complexity=round(complexity, 3),
    )


def build_world_graph(profile: SeedDocumentProfile, world_profile: WorldProfile) -> WorldGraph:
    nodes: list[WorldNode] = []
    edges: list[WorldEdge] = []
    edge_buckets: dict[tuple[str, str, WorldEdgeKind], _EdgeBucket] = {}

    document_node_id = _stable_id("node", f"document:{profile.document_id}")
    question_node_id = _stable_id(
        "node",
        f"question:{profile.document_id}:{world_profile.question}",
    )
    sentiment_node_id = _stable_id(
        "node",
        f"sentiment:{profile.document_id}:{world_profile.sentiment.label.value}",
    )

    nodes.append(
        WorldNode(
            id=document_node_id,
            label="Seed document",
            kind=WorldNodeKind.DOCUMENT,
            weight=1.0,
            evidence=profile.sentences[:2],
            attributes={
                "document_id": profile.document_id,
                "paragraph_count": profile.paragraph_count,
                "sentence_count": profile.sentence_count,
                "token_count": profile.token_count,
            },
        )
    )
    nodes.append(
        WorldNode(
            id=question_node_id,
            label=world_profile.question or "Question",
            kind=WorldNodeKind.QUESTION,
            weight=0.9 if world_profile.question else 0.5,
            evidence=[world_profile.question] if world_profile.question else [],
            attributes={"question_tokens": profile.question_tokens[:8]},
        )
    )
    nodes.append(
        WorldNode(
            id=sentiment_node_id,
            label=world_profile.sentiment.label.value.title(),
            kind=WorldNodeKind.SENTIMENT,
            weight=abs(world_profile.sentiment.score) or 0.1,
            evidence=world_profile.sentiment.evidence[:2],
            attributes={
                "score": world_profile.sentiment.score,
                "confidence": world_profile.sentiment.confidence,
                "scope": world_profile.sentiment.scope,
            },
        )
    )

    for topic in world_profile.topics:
        topic_node_id = _stable_id("node", f"topic:{topic.id}")
        nodes.append(
            WorldNode(
                id=topic_node_id,
                label=topic.label,
                kind=WorldNodeKind.TOPIC,
                weight=topic.score,
                evidence=topic.evidence[:2],
                attributes={
                    "topic_id": topic.id,
                    "mentions": topic.mentions,
                    "keywords": topic.keywords[:6],
                },
            )
        )
        _add_edge(
            edge_buckets,
            document_node_id,
            topic_node_id,
            WorldEdgeKind.CONTAINS,
            weight=max(0.1, topic.score),
            evidence=topic.evidence[:2],
        )
        if _node_focuses_on(world_profile.question, topic.label, topic.keywords):
            _add_edge(
                edge_buckets,
                question_node_id,
                topic_node_id,
                WorldEdgeKind.FOCUSES_ON,
                weight=max(0.15, min(1.0, topic.score + 0.15)),
                evidence=[world_profile.question],
            )

    for entity in world_profile.entities:
        entity_node_id = _stable_id("node", f"entity:{entity.id}")
        nodes.append(
            WorldNode(
                id=entity_node_id,
                label=entity.label,
                kind=WorldNodeKind.ENTITY,
                weight=entity.score,
                evidence=entity.evidence[:2],
                attributes={
                    "entity_id": entity.id,
                    "entity_type": entity.entity_type,
                    "mentions": entity.mentions,
                    "aliases": entity.aliases[:6],
                },
            )
        )
        _add_edge(
            edge_buckets,
            document_node_id,
            entity_node_id,
            WorldEdgeKind.MENTIONS,
            weight=max(0.1, entity.score),
            evidence=entity.evidence[:2],
        )
        if _node_focuses_on(world_profile.question, entity.label, entity.aliases):
            _add_edge(
                edge_buckets,
                question_node_id,
                entity_node_id,
                WorldEdgeKind.FOCUSES_ON,
                weight=max(0.15, min(1.0, entity.score + 0.15)),
                evidence=[world_profile.question],
            )

    _add_edge(
        edge_buckets,
        document_node_id,
        question_node_id,
        WorldEdgeKind.FRAMES,
        weight=0.9,
        evidence=[world_profile.question] if world_profile.question else profile.sentences[:1],
    )
    _add_edge(
        edge_buckets,
        document_node_id,
        sentiment_node_id,
        WorldEdgeKind.FRAMES,
        weight=max(0.1, abs(world_profile.sentiment.score)),
        evidence=world_profile.sentiment.evidence[:2],
    )

    _add_cooccurrence_edges(
        edge_buckets,
        profile=profile,
        world_profile=world_profile,
    )

    sorted_edges = sorted(
        edge_buckets.items(),
        key=lambda item: (item[0][2].value, item[0][0], item[0][1]),
    )
    for (source, target, relation), data in sorted_edges:
        edges.append(
            WorldEdge(
                source=source,
                target=target,
                relation=relation,
                weight=round(_clamp(data["weight"]), 3),
                count=data["count"],
                evidence=list(dict.fromkeys(data["evidence"]))[:3],
            )
        )

    nodes.sort(key=lambda node: (node.kind.value, -node.weight, node.label.lower()))
    node_count = len(nodes)
    edge_count = len(edges)
    density = 0.0
    average_degree = 0.0
    if node_count > 1:
        density = min(1.0, (2.0 * edge_count) / float(node_count * (node_count - 1)))
        average_degree = round((2.0 * edge_count) / node_count, 3)

    return WorldGraph(
        nodes=nodes,
        edges=edges,
        root_id=document_node_id,
        density=round(density, 3),
        average_degree=average_degree,
    )


def build_seed_world(seed_document: str, question: str = "") -> SeedWorld:
    ingestion = ingest_seed_document(seed_document, question=question)
    entities = extract_entities(ingestion)
    topics = extract_topics(ingestion, entities=entities)
    sentiment_signals = extract_sentiment_signals(ingestion)
    profile = build_world_profile(
        ingestion,
        entities=entities,
        topics=topics,
        sentiment_signals=sentiment_signals,
    )
    graph = build_world_graph(ingestion, profile)
    return SeedWorld(ingestion=ingestion, profile=profile, graph=graph)


def _load_seed_text(seed_document: str) -> tuple[str, str | None]:
    candidate = Path(seed_document).expanduser()
    try:
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(encoding="utf-8"), str(candidate)
    except OSError:
        pass
    return seed_document, None


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    sentences = [segment.strip() for segment in _SENTENCE_RE.split(text) if segment.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        token = _clean_token(match.group(0))
        if token:
            tokens.append(token)
    return tokens


def _clean_token(token: str) -> str:
    return token.lower().strip("'\"-").strip()


def _clean_entity_label(label: str) -> str:
    cleaned = label.strip(" ,;:!?()[]{}<>\"'")
    parts = cleaned.split()
    while parts and parts[0].lower() in _ARTICLE_WORDS and len(parts) > 1:
        parts = parts[1:]
    return " ".join(parts)


def _infer_entity_type(label: str) -> str:
    tokens = label.split()
    if not tokens:
        return "concept"
    if label.isupper() and 1 < len(label) <= 8:
        return "acronym"
    lowered = tokens[-1].lower()
    if lowered in _ORG_SUFFIXES:
        return "organization"
    if lowered in _LOCATION_SUFFIXES:
        return "location"
    if len(tokens) > 1 and all(token[:1].isupper() for token in tokens):
        return "named_entity"
    return "concept"


def _entity_score(mentions: int, label: str) -> float:
    base = 0.18 + (mentions * 0.16)
    base += 0.04 * max(0, len(label.split()) - 1)
    return round(min(1.0, base), 3)


def _sentence_hits(sentences: list[str], label: str, limit: int = 2) -> list[str]:
    hits: list[str] = []
    patterns = [label]
    patterns.extend(_tokenize(label))
    for sentence in sentences:
        lower_sentence = sentence.lower()
        if any(_label_matches_text(lower_sentence, pattern) for pattern in patterns):
            hits.append(sentence.strip())
        if len(hits) >= limit:
            break
    return hits


def _label_matches_text(text: str, label: str) -> bool:
    if not label:
        return False
    pattern = rf"\b{re.escape(label.lower())}\b"
    return bool(re.search(pattern, text.lower()))


def _label_matches_any(text: str, labels: Iterable[str]) -> bool:
    return any(_label_matches_text(text, label) for label in labels if label)


def _merge_evidence(existing: list[str], new_evidence: list[str], limit: int = 2) -> list[str]:
    combined = existing.copy()
    for item in new_evidence:
        if item not in combined:
            combined.append(item)
    return combined[:limit]


def _topic_keywords(label: str) -> list[str]:
    return [token for token in label.lower().split() if token not in _STOPWORDS]


def _title_case_label(label: str) -> str:
    parts = label.replace("_", " ").split()
    return " ".join(part[:1].upper() + part[1:] if part else part for part in parts)


def _sentiment_label(
    score: float, positive_terms: list[str], negative_terms: list[str]
) -> SentimentLabel:
    if positive_terms and negative_terms:
        return SentimentLabel.MIXED
    if score > 0.2:
        return SentimentLabel.POSITIVE
    if score < -0.2:
        return SentimentLabel.NEGATIVE
    if positive_terms:
        return SentimentLabel.POSITIVE
    if negative_terms:
        return SentimentLabel.NEGATIVE
    return SentimentLabel.NEUTRAL


def _document_sentiment(signals: list[SentimentSignal]) -> SentimentSignal:
    for signal in reversed(signals):
        if signal.scope == "document":
            return signal
    return SentimentSignal(
        label=SentimentLabel.NEUTRAL, score=0.0, confidence=0.0, scope="document"
    )


def _build_summary(
    topics: list[str],
    entities: list[str],
    sentiment: SentimentSignal,
    question: str,
) -> str:
    topic_fragment = ", ".join(topics[:3]) if topics else "the seed narrative"
    entity_fragment = ", ".join(entities[:2]) if entities else ""
    if entity_fragment:
        lead = (
            f"The seed document centers on {topic_fragment} and repeatedly names {entity_fragment}"
        )
    else:
        lead = f"The seed document centers on {topic_fragment}"
    if question:
        lead = f'{lead} in response to the question "{question}"'
    tone = sentiment.label.value
    return f"{lead}. Overall tone is {tone} with a score of {sentiment.score:+.2f}."


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, round(value, 4)))


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _node_focuses_on(question: str, label: str, aliases: Iterable[str]) -> bool:
    if not question:
        return False
    question_lower = question.lower()
    if _label_matches_text(question_lower, label):
        return True
    return any(_label_matches_text(question_lower, alias) for alias in aliases)


def _add_edge(
    edge_buckets: dict[tuple[str, str, WorldEdgeKind], _EdgeBucket],
    source: str,
    target: str,
    relation: WorldEdgeKind,
    *,
    weight: float,
    evidence: Iterable[str],
) -> None:
    key = (source, target, relation)
    bucket: _EdgeBucket = edge_buckets.setdefault(key, _new_edge_bucket())
    bucket["weight"] = max(bucket["weight"], weight)
    bucket["count"] += 1
    for item in evidence:
        if item not in bucket["evidence"]:
            bucket["evidence"].append(item)


def _add_cooccurrence_edges(
    edge_buckets: dict[tuple[str, str, WorldEdgeKind], _EdgeBucket],
    *,
    profile: SeedDocumentProfile,
    world_profile: WorldProfile,
) -> None:
    topic_nodes = {
        topic.id: _stable_id("node", f"topic:{topic.id}") for topic in world_profile.topics
    }
    entity_nodes = {
        entity.id: _stable_id("node", f"entity:{entity.id}") for entity in world_profile.entities
    }

    for sentence in profile.sentences:
        sentence_lower = sentence.lower()
        matched_topics = [
            topic
            for topic in world_profile.topics
            if _label_matches_text(sentence_lower, topic.label)
            or any(_label_matches_text(sentence_lower, keyword) for keyword in topic.keywords)
        ]
        matched_entities = [
            entity
            for entity in world_profile.entities
            if _label_matches_text(sentence_lower, entity.label)
            or any(_label_matches_text(sentence_lower, alias) for alias in entity.aliases)
        ]

        for left_index, left_topic in enumerate(matched_topics):
            for right_topic in matched_topics[left_index + 1 :]:
                left_node = topic_nodes[left_topic.id]
                right_node = topic_nodes[right_topic.id]
                source, target = sorted((left_node, right_node))
                _add_edge(
                    edge_buckets,
                    source,
                    target,
                    WorldEdgeKind.CO_OCCURS_WITH,
                    weight=max(0.15, min(1.0, (left_topic.score + right_topic.score) / 2.0)),
                    evidence=[sentence],
                )

        for left_index, left_entity in enumerate(matched_entities):
            for right_entity in matched_entities[left_index + 1 :]:
                left_node = entity_nodes[left_entity.id]
                right_node = entity_nodes[right_entity.id]
                source, target = sorted((left_node, right_node))
                _add_edge(
                    edge_buckets,
                    source,
                    target,
                    WorldEdgeKind.CO_OCCURS_WITH,
                    weight=max(0.15, min(1.0, (left_entity.score + right_entity.score) / 2.0)),
                    evidence=[sentence],
                )

        for topic in matched_topics:
            for entity in matched_entities:
                topic_node = topic_nodes[topic.id]
                entity_node = entity_nodes[entity.id]
                source, target = sorted((topic_node, entity_node))
                _add_edge(
                    edge_buckets,
                    source,
                    target,
                    WorldEdgeKind.CO_OCCURS_WITH,
                    weight=max(0.15, min(1.0, (topic.score + entity.score) / 2.0)),
                    evidence=[sentence],
                )

    if world_profile.topics and world_profile.entities:
        for topic in world_profile.topics:
            for entity in world_profile.entities:
                topic_lower = topic.label.lower()
                entity_lower = entity.label.lower()
                if topic_lower in entity_lower or entity_lower in topic_lower:
                    topic_node = topic_nodes[topic.id]
                    entity_node = entity_nodes[entity.id]
                    source, target = sorted((topic_node, entity_node))
                    _add_edge(
                        edge_buckets,
                        source,
                        target,
                        WorldEdgeKind.CO_OCCURS_WITH,
                        weight=max(0.2, min(1.0, (topic.score + entity.score) / 2.0)),
                        evidence=[f"{topic.label} / {entity.label}"],
                    )
def _new_edge_bucket() -> _EdgeBucket:
    return {"weight": 0.0, "count": 0, "evidence": []}
