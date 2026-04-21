from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import random
import re
from typing import Iterable, Sequence

from lean_swarm.engine.models import AgentState


_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "another",
    "around",
    "because",
    "before",
    "between",
    "but",
    "could",
    "during",
    "each",
    "even",
    "every",
    "from",
    "have",
    "here",
    "into",
    "just",
    "more",
    "most",
    "much",
    "other",
    "over",
    "should",
    "their",
    "there",
    "these",
    "those",
    "through",
    "under",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}

_NEGATIVE_MARKERS = {
    "decline",
    "drop",
    "fear",
    "fragile",
    "risk",
    "shock",
    "uncertain",
    "uncertainty",
    "volatile",
    "weak",
}

_POSITIVE_MARKERS = {
    "growth",
    "improve",
    "gain",
    "strong",
    "support",
    "stable",
    "upbeat",
    "positive",
    "expand",
}

_ANALYTICAL_MARKERS = {
    "analysis",
    "evidence",
    "forecast",
    "indicator",
    "model",
    "policy",
    "projection",
    "trend",
}

_DOMAIN_KEYWORDS = {
    "policy": {"policy", "government", "regulation", "election", "public", "legislation", "vote"},
    "economic": {"economic", "market", "jobs", "inflation", "prices", "trade", "growth"},
    "media": {"media", "press", "coverage", "narrative", "headline", "reporting", "message"},
    "community": {"community", "local", "citizen", "voter", "residents", "public", "coalition"},
    "institutional": {"institution", "agency", "executive", "board", "committee", "court"},
    "civic": {"civic", "activist", "advocacy", "organizing", "campaign", "movement"},
    "technology": {"technology", "platform", "data", "ai", "digital", "software", "system"},
    "health": {"health", "hospital", "medical", "care", "virus", "patient", "vaccination"},
    "culture": {"culture", "identity", "values", "norms", "creative", "arts", "social"},
    "security": {"security", "defense", "risk", "threat", "border", "crime", "stability"},
}

_STANCE_VOCAB = ("pragmatic", "cautious", "optimistic", "skeptical")
_MODALITY_VOCAB = ("analyst", "organizer", "observer", "amplifier", "broker", "watchdog", "translator", "sentinel")
_NAMED_AGENTS = (
    "Aster",
    "Briar",
    "Cinder",
    "Drift",
    "Ember",
    "Flint",
    "Gale",
    "Harbor",
    "Indigo",
    "Juniper",
    "Keystone",
    "Lumen",
    "Marble",
    "North",
    "Onyx",
    "Piper",
    "Quill",
    "Reed",
    "Solace",
    "Talon",
    "Umber",
    "Vale",
    "Willow",
    "Xenon",
    "Yarrow",
    "Zephyr",
    "Atlas",
    "Beacon",
    "Cobalt",
    "Delta",
    "Echo",
    "Fable",
    "Granite",
    "Halo",
    "Ion",
    "Jade",
    "Kindle",
    "Ledger",
    "Morrow",
    "Nova",
    "Orbit",
    "Praxis",
    "Quest",
    "Ridge",
    "Signal",
    "Tempo",
    "Union",
    "Vector",
    "Warden",
    "Yonder",
)


@dataclass(frozen=True)
class PopulationSignals:
    seed_terms: tuple[str, ...]
    question_terms: tuple[str, ...]
    world_topics: tuple[str, ...]
    world_entities: tuple[str, ...]
    dominant_domains: tuple[str, ...]
    stance_mix: dict[str, float]
    modality_mix: dict[str, float]
    domain_mix: dict[str, float]


@dataclass(frozen=True)
class PopulationProfile:
    seed_fingerprint: str
    target_agents: int
    signals: PopulationSignals
    archetype_pool: tuple[str, ...]
    summary: str


@dataclass(frozen=True)
class PopulationBundle:
    agents: list[AgentState] = field(default_factory=list)
    profile: PopulationProfile | None = None


def build_population(
    seed_document: str,
    question: str,
    *,
    world_topics: Sequence[str] | None = None,
    world_entities: Sequence[str] | None = None,
    max_agents: int = 24,
    random_seed: int = 7,
) -> PopulationBundle:
    """Build a seed-conditioned population with diverse archetypes and personalized traits."""

    target_agents = max(1, min(int(max_agents), 50))
    topics = tuple(_normalize_terms(world_topics or ()))
    entities = tuple(_normalize_terms(world_entities or ()))
    signals = _derive_signals(seed_document, question, topics, entities)
    rng = random.Random(_derive_seed(seed_document, question, topics, entities, random_seed))

    archetype_pool = _build_archetype_pool(signals)
    agents = _materialize_agents(
        archetype_pool=archetype_pool,
        signals=signals,
        rng=rng,
        target_agents=target_agents,
    )
    profile = PopulationProfile(
        seed_fingerprint=_fingerprint(seed_document, question, topics, entities),
        target_agents=target_agents,
        signals=signals,
        archetype_pool=tuple(archetype_pool),
        summary=_summarize_profile(signals, target_agents),
    )
    return PopulationBundle(agents=agents, profile=profile)


def build_archetype_pool(
    seed_document: str,
    question: str,
    *,
    world_topics: Sequence[str] | None = None,
    world_entities: Sequence[str] | None = None,
) -> list[str]:
    """Return a ranked archetype pool without instantiating agents."""

    topics = tuple(_normalize_terms(world_topics or ()))
    entities = tuple(_normalize_terms(world_entities or ()))
    signals = _derive_signals(seed_document, question, topics, entities)
    return _build_archetype_pool(signals)


def extract_population_signals(
    seed_document: str,
    question: str,
    *,
    world_topics: Sequence[str] | None = None,
    world_entities: Sequence[str] | None = None,
) -> PopulationSignals:
    """Expose the derived inputs so a manager can inspect the population plan."""

    topics = tuple(_normalize_terms(world_topics or ()))
    entities = tuple(_normalize_terms(world_entities or ()))
    return _derive_signals(seed_document, question, topics, entities)


def _derive_signals(
    seed_document: str,
    question: str,
    world_topics: tuple[str, ...],
    world_entities: tuple[str, ...],
) -> PopulationSignals:
    seed_terms = _extract_terms(seed_document)
    question_terms = _extract_terms(question)
    all_terms = Counter(seed_terms + question_terms + list(world_topics) + list(world_entities))

    domain_mix = _score_domains(all_terms, world_topics, world_entities)
    dominant_domains = tuple(_top_keys(domain_mix, 3))
    stance_mix = _score_stances(all_terms, seed_document, question)
    modality_mix = _score_modalities(all_terms, world_topics, world_entities)

    return PopulationSignals(
        seed_terms=tuple(_top_terms(seed_terms, 12)),
        question_terms=tuple(_top_terms(question_terms, 8)),
        world_topics=world_topics,
        world_entities=world_entities,
        dominant_domains=dominant_domains,
        stance_mix=stance_mix,
        modality_mix=modality_mix,
        domain_mix=domain_mix,
    )


def _build_archetype_pool(signals: PopulationSignals) -> list[str]:
    scored_pool: list[tuple[float, str]] = []
    domain_order = list(signals.domain_mix.keys())
    if not domain_order:
        domain_order = sorted(_DOMAIN_KEYWORDS)

    for stance in _sorted_mix(signals.stance_mix, _STANCE_VOCAB):
        for modality in _sorted_mix(signals.modality_mix, _MODALITY_VOCAB):
            for domain_index, domain in enumerate(domain_order):
                score = signals.stance_mix.get(stance, 0.0)
                score *= signals.modality_mix.get(modality, 0.0)
                score *= signals.domain_mix.get(domain, 0.0)
                score *= 1.0 - (domain_index * 0.015)
                if domain in signals.dominant_domains:
                    score *= 1.25
                scored_pool.append((score, f"{stance}-{modality}-{domain}"))

    ranked = _dedupe_in_order([archetype for _, archetype in sorted(scored_pool, key=lambda item: item[0], reverse=True)])
    return ranked[:96]


def _materialize_agents(
    *,
    archetype_pool: Sequence[str],
    signals: PopulationSignals,
    rng: random.Random,
    target_agents: int,
) -> list[AgentState]:
    agents: list[AgentState] = []
    names = _personalized_names(signals, rng, target_agents)
    recent_domains: list[str] = []
    recent_stances: list[str] = []
    recent_modalities: list[str] = []

    for index in range(target_agents):
        archetype = _choose_archetype(
            archetype_pool,
            signals,
            rng,
            recent_domains=recent_domains,
            recent_stances=recent_stances,
            recent_modalities=recent_modalities,
        )
        stance, modality, domain = archetype.split("-", 2)
        recent_domains.append(domain)
        recent_stances.append(stance)
        recent_modalities.append(modality)
        recent_domains = recent_domains[-4:]
        recent_stances = recent_stances[-4:]
        recent_modalities = recent_modalities[-4:]

        mood, energy, attention = _jitter_traits(stance, modality, domain, signals, rng)
        agents.append(
            AgentState(
                id=f"agent-{index + 1}",
                name=names[index],
                archetype=archetype,
                mood=mood,
                energy=energy,
                attention=attention,
            )
        )

    return agents


def _choose_archetype(
    archetype_pool: Sequence[str],
    signals: PopulationSignals,
    rng: random.Random,
    *,
    recent_domains: Sequence[str],
    recent_stances: Sequence[str],
    recent_modalities: Sequence[str],
) -> str:
    if not archetype_pool:
        return "pragmatic-analyst-policy"

    scored: list[tuple[float, str]] = []
    for archetype in archetype_pool:
        stance, modality, domain = archetype.split("-", 2)
        score = signals.stance_mix.get(stance, 0.05)
        score *= signals.modality_mix.get(modality, 0.05)
        score *= signals.domain_mix.get(domain, 0.05)
        if domain in recent_domains:
            score *= 0.7
        if stance in recent_stances:
            score *= 0.8
        if modality in recent_modalities:
            score *= 0.85
        score *= 0.8 + rng.random() * 0.4
        scored.append((score, archetype))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_window = scored[: max(8, min(20, len(scored)))]
    weights = [score for score, _ in top_window]
    total = sum(weights)
    if total <= 0:
        return top_window[0][1]

    pick = rng.random() * total
    cursor = 0.0
    for score, archetype in top_window:
        cursor += score
        if cursor >= pick:
            return archetype
    return top_window[-1][1]


def _jitter_traits(
    stance: str,
    modality: str,
    domain: str,
    signals: PopulationSignals,
    rng: random.Random,
) -> tuple[float, float, float]:
    mood_base = {
        "optimistic": 0.66,
        "pragmatic": 0.56,
        "cautious": 0.45,
        "skeptical": 0.38,
    }.get(stance, 0.5)
    energy_base = {
        "organizer": 0.72,
        "amplifier": 0.68,
        "broker": 0.63,
        "analyst": 0.58,
        "watchdog": 0.55,
        "translator": 0.54,
        "observer": 0.48,
        "sentinel": 0.46,
    }.get(modality, 0.55)
    attention_base = {
        "policy": 0.66,
        "economic": 0.62,
        "media": 0.6,
        "community": 0.64,
        "institutional": 0.63,
        "civic": 0.65,
        "technology": 0.59,
        "health": 0.61,
        "culture": 0.57,
        "security": 0.67,
    }.get(domain, 0.58)

    if any(term in _NEGATIVE_MARKERS for term in signals.seed_terms + signals.question_terms):
        mood_base -= 0.05
        attention_base += 0.03
    if any(term in _POSITIVE_MARKERS for term in signals.seed_terms + signals.question_terms):
        mood_base += 0.05
    if any(term in _ANALYTICAL_MARKERS for term in signals.seed_terms + signals.question_terms):
        energy_base += 0.03
        attention_base += 0.04

    mood = _clamp(mood_base + rng.uniform(-0.12, 0.12))
    energy = _clamp(energy_base + rng.uniform(-0.14, 0.14))
    attention = _clamp(attention_base + rng.uniform(-0.15, 0.15))
    return round(mood, 3), round(energy, 3), round(attention, 3)


def _personalized_names(
    signals: PopulationSignals,
    rng: random.Random,
    target_agents: int,
) -> list[str]:
    base_names = list(_NAMED_AGENTS[:target_agents])
    topic_roots = [_compress_term(term) for term in signals.world_topics[:6] if term]
    entity_roots = [_compress_term(term) for term in signals.world_entities[:4] if term]
    palette = topic_roots + entity_roots
    if not palette:
        return base_names

    names: list[str] = []
    palette_cycle = list(dict.fromkeys(palette))
    rng.shuffle(palette_cycle)
    for index, base_name in enumerate(base_names):
        if index < len(palette_cycle):
            suffix = palette_cycle[index]
            names.append(f"{base_name} {suffix.capitalize()}")
        else:
            names.append(base_name)
    return names


def _score_domains(
    all_terms: Counter[str],
    world_topics: tuple[str, ...],
    world_entities: tuple[str, ...],
) -> dict[str, float]:
    domain_scores: dict[str, float] = {}
    terms = set(all_terms)
    topic_blob = " ".join((*world_topics, *world_entities)).lower()

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = 0.1
        score += sum(1.0 for keyword in keywords if keyword in terms or keyword in topic_blob)
        score += 0.2 * sum(1.0 for term in terms if term.startswith(domain[:3]))
        domain_scores[domain] = score

    total = sum(domain_scores.values()) or 1.0
    return {domain: score / total for domain, score in sorted(domain_scores.items(), key=lambda item: item[1], reverse=True)}


def _score_stances(all_terms: Counter[str], seed_document: str, question: str) -> dict[str, float]:
    combined = f"{seed_document} {question}".lower()
    negative_hits = sum(1 for marker in _NEGATIVE_MARKERS if marker in combined)
    positive_hits = sum(1 for marker in _POSITIVE_MARKERS if marker in combined)
    analytical_hits = sum(1 for marker in _ANALYTICAL_MARKERS if marker in combined)
    uncertainty_hits = sum(
        1 for marker in ("may", "might", "uncertain", "uncertainty", "possibly", "likely") if marker in combined
    )

    scores = {
        "pragmatic": 1.0 + analytical_hits * 0.35 + len(all_terms) * 0.015,
        "cautious": 1.0 + uncertainty_hits * 0.45 + negative_hits * 0.25,
        "optimistic": 1.0 + positive_hits * 0.45,
        "skeptical": 1.0 + negative_hits * 0.4 + uncertainty_hits * 0.2,
    }
    total = sum(scores.values()) or 1.0
    return {stance: score / total for stance, score in scores.items()}


def _score_modalities(
    all_terms: Counter[str],
    world_topics: tuple[str, ...],
    world_entities: tuple[str, ...],
) -> dict[str, float]:
    combined_terms = list(all_terms.elements()) + list(world_topics) + list(world_entities)
    combined = " ".join(combined_terms).lower()
    counts = Counter()
    counts["analyst"] += sum(1 for marker in _ANALYTICAL_MARKERS if marker in combined)
    counts["organizer"] += sum(1 for marker in ("coalition", "mobilize", "organize", "campaign", "community") if marker in combined)
    counts["observer"] += sum(1 for marker in ("observe", "watch", "monitor", "track", "signal") if marker in combined)
    counts["amplifier"] += sum(1 for marker in ("message", "narrative", "spread", "media", "broadcast") if marker in combined)
    counts["broker"] += sum(1 for marker in ("bridge", "negotiate", "mediate", "cross", "exchange") if marker in combined)
    counts["watchdog"] += sum(1 for marker in ("accountability", "oversight", "risk", "audit", "check") if marker in combined)
    counts["translator"] += sum(1 for marker in ("translate", "interpret", "frame", "context", "explain") if marker in combined)
    counts["sentinel"] += sum(1 for marker in ("security", "threat", "alert", "stability", "warning") if marker in combined)

    for modality in _MODALITY_VOCAB:
        counts[modality] += 1

    total = sum(counts.values()) or 1.0
    return {modality: counts[modality] / total for modality in _MODALITY_VOCAB}


def _sorted_mix(mapping: dict[str, float], fallback_order: Iterable[str]) -> list[str]:
    ordered = sorted(mapping.items(), key=lambda item: item[1], reverse=True)
    if not ordered:
        return list(fallback_order)
    return [key for key, _ in ordered]


def _top_terms(terms: Sequence[str], limit: int) -> list[str]:
    counts = Counter(terms)
    return [term for term, _ in counts.most_common(limit)]


def _top_keys(mapping: dict[str, float], limit: int) -> list[str]:
    return [key for key, _ in sorted(mapping.items(), key=lambda item: item[1], reverse=True)[:limit]]


def _extract_terms(text: str) -> list[str]:
    return [
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_\-']+", text.lower())
        if len(term) > 3 and term not in _STOPWORDS
    ]


def _normalize_terms(values: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        normalized.extend(_extract_terms(value))
    return normalized


def _dedupe_in_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _compress_term(term: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "", term.lower())
    if not cleaned:
        return ""
    if len(cleaned) <= 6:
        return cleaned
    return cleaned[:6]


def _summarize_profile(signals: PopulationSignals, target_agents: int) -> str:
    dominant_domains = ", ".join(signals.dominant_domains[:3]) or "general"
    topic_bits = ", ".join(signals.world_topics[:4]) or "none"
    entity_bits = ", ".join(signals.world_entities[:3]) or "none"
    return (
        f"Built {target_agents} agents around {dominant_domains} signals. "
        f"Topics: {topic_bits}. Entities: {entity_bits}."
    )


def _derive_seed(
    seed_document: str,
    question: str,
    world_topics: tuple[str, ...],
    world_entities: tuple[str, ...],
    random_seed: int,
) -> int:
    digest = hashlib.sha256()
    digest.update(seed_document.encode("utf-8"))
    digest.update(b"\0")
    digest.update(question.encode("utf-8"))
    digest.update(b"\0")
    digest.update("|".join(world_topics).encode("utf-8"))
    digest.update(b"\0")
    digest.update("|".join(world_entities).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(random_seed).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def _fingerprint(
    seed_document: str,
    question: str,
    world_topics: tuple[str, ...],
    world_entities: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(seed_document.encode("utf-8"))
    digest.update(question.encode("utf-8"))
    digest.update("|".join(world_topics).encode("utf-8"))
    digest.update("|".join(world_entities).encode("utf-8"))
    return digest.hexdigest()[:16]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
