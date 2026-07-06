from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelTier(StrEnum):
    FLAGSHIP = "flagship"
    STANDARD = "standard"
    CHEAP = "cheap"


class TaskType(StrEnum):
    WORLD_BOOTSTRAP = "world_bootstrap"
    WORLD_EXTRACTION = "world_extraction"
    PERSONA_BATCH = "persona_batch"
    AGENT_BATCH = "agent_batch"
    MEMORY_SUMMARY = "memory_summary"
    PREDICTION_SYNTHESIS = "prediction_synthesis"
    AGENT_CHAT = "agent_chat"
    REPORT_CHAT = "report_chat"
    FULL_REPORT = "full_report"


class ActivationMode(StrEnum):
    LEAN = "lean"
    NAIVE = "naive"


class AgentMemory(BaseModel):
    working: list[str] = Field(default_factory=list)
    episodic: list[str] = Field(default_factory=list)
    semantic: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    id: str
    name: str
    archetype: str
    mood: float = Field(default=0.5, ge=0.0, le=1.0)
    energy: float = Field(default=0.5, ge=0.0, le=1.0)
    attention: float = Field(default=0.5, ge=0.0, le=1.0)
    relationships: dict[str, float] = Field(default_factory=dict)
    memory: AgentMemory = Field(default_factory=AgentMemory)
    persona: str | None = None
    stance: str | None = None


class AgentAction(BaseModel):
    agent_id: str
    action_type: str
    description: str
    delta_mood: float = 0.0
    delta_energy: float = 0.0
    delta_attention: float = 0.0

    @field_validator("delta_mood", "delta_energy", "delta_attention", mode="before")
    @classmethod
    def coerce_and_clamp_delta(cls, v: Any) -> float:
        if v is None:
            return 0.0
        try:
            val = float(v)
            return max(-0.25, min(0.25, val))
        except (ValueError, TypeError):
            return 0.0


class TickRecord(BaseModel):
    tick: int
    active_agent_ids: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    mean_delta: float = 0.0
    activation_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    activation_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    activation_triggers: list[str] = Field(default_factory=list)
    relationship_churn: float = Field(default=0.0, ge=0.0, le=1.0)
    stability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    convergence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    stable: bool = False


class RetrievedSource(BaseModel):
    url: str
    title: str = ""
    chars: int = 0
    via: str = "direct"
    text: str = Field(default="", exclude=True)


class SimulationRequest(BaseModel):
    seed_document: str
    question: str
    rounds: int = Field(default=6, ge=1, le=20)
    max_agents: int = Field(default=24, ge=1, le=50)
    active_agent_fraction: float = Field(default=0.2, gt=0.0, le=1.0)
    activation_mode: ActivationMode = ActivationMode.LEAN
    convergence_threshold: int = Field(default=2, ge=1)
    group_size: int = Field(default=5, ge=1, le=10)
    random_seed: int = 7
    use_llm: bool = True
    retrieved_sources: list[RetrievedSource] = Field(default_factory=list)


class PredictionReport(BaseModel):
    question: str
    prediction: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: list[str] = Field(default_factory=list)
    key_events: list[str] = Field(default_factory=list)
    converged: bool = False
    tick_count: int = 0
    llm_calls: int = 0
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    average_active_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    activation_envelope_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    direction: str = ""


class RelationshipEdge(BaseModel):
    source: str
    target: str
    weight: float = Field(default=0.0, ge=-1.0, le=1.0)


class WorldSnapshot(BaseModel):
    agents: list[AgentState] = Field(default_factory=list)
    edges: list[RelationshipEdge] = Field(default_factory=list)
    profile: WorldProfile | None = None
    graph: WorldGraph | None = None
    population_summary: str | None = None


class SimulationResult(BaseModel):
    request: SimulationRequest
    report: PredictionReport
    ticks: list[TickRecord] = Field(default_factory=list)
    world: WorldSnapshot


class SentimentLabel(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class WorldNodeKind(StrEnum):
    DOCUMENT = "document"
    QUESTION = "question"
    ENTITY = "entity"
    TOPIC = "topic"
    SENTIMENT = "sentiment"


class WorldEdgeKind(StrEnum):
    CONTAINS = "contains"
    FOCUSES_ON = "focuses_on"
    MENTIONS = "mentions"
    CO_OCCURS_WITH = "co_occurs_with"
    FRAMES = "frames"
    CONTRASTS_WITH = "contrasts_with"
    SUPPORTS = "supports"
    OPPOSES = "opposes"
    INFLUENCES = "influences"
    PART_OF = "part_of"
    REPORTS_ON = "reports_on"
    CAUSES = "causes"
    TARGETS = "targets"
    RELATES_TO = "relates_to"


class SeedDocumentProfile(BaseModel):
    document_id: str
    source_text: str
    normalized_text: str
    question: str
    source_path: str | None = None
    paragraphs: list[str] = Field(default_factory=list)
    sentences: list[str] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)
    question_tokens: list[str] = Field(default_factory=list)
    token_count: int = 0
    unique_token_count: int = 0
    paragraph_count: int = 0
    sentence_count: int = 0
    top_terms: list[str] = Field(default_factory=list)


class SeedEntity(BaseModel):
    id: str
    label: str
    entity_type: str = "concept"
    mentions: int = 1
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    aliases: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class SeedTopic(BaseModel):
    id: str
    label: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    keywords: list[str] = Field(default_factory=list)
    mentions: int = 0
    evidence: list[str] = Field(default_factory=list)
    related_entities: list[str] = Field(default_factory=list)


class SentimentSignal(BaseModel):
    label: SentimentLabel = SentimentLabel.NEUTRAL
    score: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    scope: str = "document"
    span_index: int | None = None
    positive_terms: list[str] = Field(default_factory=list)
    negative_terms: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class WorldProfile(BaseModel):
    document_id: str
    question: str
    summary: str
    tone: SentimentLabel = SentimentLabel.NEUTRAL
    sentiment: SentimentSignal
    sentiment_signals: list[SentimentSignal] = Field(default_factory=list)
    topics: list[SeedTopic] = Field(default_factory=list)
    entities: list[SeedEntity] = Field(default_factory=list)
    focus_terms: list[str] = Field(default_factory=list)
    uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    extraction_source: str = "deterministic"


class WorldNode(BaseModel):
    id: str
    label: str
    kind: WorldNodeKind
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    attributes: dict[str, object] = Field(default_factory=dict)


class WorldEdge(RelationshipEdge):
    relation: WorldEdgeKind = WorldEdgeKind.MENTIONS
    count: int = 1
    evidence: list[str] = Field(default_factory=list)


class WorldGraph(BaseModel):
    nodes: list[WorldNode] = Field(default_factory=list)
    edges: list[WorldEdge] = Field(default_factory=list)
    root_id: str = ""
    density: float = Field(default=0.0, ge=0.0, le=1.0)
    average_degree: float = Field(default=0.0, ge=0.0)


class SeedWorld(BaseModel):
    ingestion: SeedDocumentProfile
    profile: WorldProfile
    graph: WorldGraph


class WorldBootstrapResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    summary: str = ""
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    sentiment: str = ""
    pressure_points: list[str] = Field(default_factory=list)

    @field_validator("topics", "entities", "pressure_points", mode="before")
    @classmethod
    def coerce_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if not isinstance(v, list):
            try:
                v = list(v)
            except Exception:
                return [str(v)]
        return [str(x) for x in v]


class AgentBatchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    actions: list[AgentAction] = Field(default_factory=list)


class MemorySummaryResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    summary: str = ""
    topics: list[str] = Field(default_factory=list)
    retained_signals: list[str] = Field(default_factory=list)

    @field_validator("topics", "retained_signals", mode="before")
    @classmethod
    def coerce_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if not isinstance(v, list):
            try:
                v = list(v)
            except Exception:
                return [str(v)]
        return [str(x) for x in v]


class PredictionSynthesisResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prediction: str
    confidence: float = 0.5
    rationale: list[str] = Field(default_factory=list)
    direction: str = ""
    supporting_terms: list[str] = Field(default_factory=list)
    volatility: str = ""

    @field_validator("rationale", "supporting_terms", mode="before")
    @classmethod
    def coerce_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if not isinstance(v, list):
            try:
                v = list(v)
            except Exception:
                return [str(v)]
        return [str(x) for x in v]

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: Any) -> float:
        try:
            val = float(v)
            if val > 1.0 and val <= 100.0:
                val = val / 100.0
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.5


_ALLOWED_ENTITY_TYPES = {
    "person", "organization", "location", "policy", "event", "concept", "media", "group",
}
_ALLOWED_RELATIONS = {
    "supports", "opposes", "influences", "part_of", "reports_on", "causes",
    "targets", "relates_to",
}


def _clamp01(v: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    label: str
    entity_type: str = "concept"
    salience: float = 0.5
    evidence: str = ""

    @field_validator("entity_type", mode="before")
    @classmethod
    def normalize_entity_type(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        return value if value in _ALLOWED_ENTITY_TYPES else "concept"

    @field_validator("salience", mode="before")
    @classmethod
    def clamp_salience(cls, v: Any) -> float:
        return _clamp01(v)

    @field_validator("evidence", mode="before")
    @classmethod
    def coerce_evidence(cls, v: Any) -> str:
        if isinstance(v, list):
            v = v[0] if v else ""
        return str(v or "")[:200]


class ExtractedRelation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source: str
    target: str
    relation: str = "relates_to"
    strength: float = 0.5
    evidence: str = ""

    @field_validator("relation", mode="before")
    @classmethod
    def normalize_relation(cls, v: Any) -> str:
        value = str(v or "").strip().lower()
        return value if value in _ALLOWED_RELATIONS else "relates_to"

    @field_validator("strength", mode="before")
    @classmethod
    def clamp_strength(cls, v: Any) -> float:
        return _clamp01(v)

    @field_validator("evidence", mode="before")
    @classmethod
    def coerce_evidence(cls, v: Any) -> str:
        if isinstance(v, list):
            v = v[0] if v else ""
        return str(v or "")[:200]


class ExtractedTopic(BaseModel):
    model_config = ConfigDict(extra="ignore")
    label: str
    keywords: list[str] = Field(default_factory=list)
    salience: float = 0.5

    @field_validator("keywords", mode="before")
    @classmethod
    def coerce_keywords(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(item) for item in v if item][:5]
        return []

    @field_validator("salience", mode="before")
    @classmethod
    def clamp_salience(cls, v: Any) -> float:
        return _clamp01(v)


class ExtractionSentiment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    label: SentimentLabel = SentimentLabel.NEUTRAL
    score: float = 0.0
    confidence: float = 0.0

    @field_validator("label", mode="before")
    @classmethod
    def coerce_label(cls, v: Any) -> SentimentLabel:
        try:
            return SentimentLabel(str(v).strip().lower())
        except ValueError:
            return SentimentLabel.NEUTRAL

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> float:
        try:
            return max(-1.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        return _clamp01(v, default=0.0)


def _coerce_item_list(v: Any, model: type[BaseModel]) -> list[Any]:
    """Coerce a raw list into validated models, dropping invalid items."""
    if not isinstance(v, list):
        return []
    out: list[Any] = []
    for item in v:
        if isinstance(item, str):
            item = {"label": item}
        if not isinstance(item, dict):
            continue
        try:
            out.append(model.model_validate(item))
        except Exception:
            continue
    return out


class WorldExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    summary: str = ""
    sentiment: ExtractionSentiment = Field(default_factory=ExtractionSentiment)
    topics: list[ExtractedTopic] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)

    @field_validator("sentiment", mode="before")
    @classmethod
    def coerce_sentiment(cls, v: Any) -> Any:
        return v if isinstance(v, dict) else {}

    @field_validator("topics", mode="before")
    @classmethod
    def coerce_topics(cls, v: Any) -> list[Any]:
        return _coerce_item_list(v, ExtractedTopic)

    @field_validator("entities", mode="before")
    @classmethod
    def coerce_entities(cls, v: Any) -> list[Any]:
        return _coerce_item_list(v, ExtractedEntity)

    @field_validator("relations", mode="before")
    @classmethod
    def coerce_relations(cls, v: Any) -> list[Any]:
        rels = []
        if isinstance(v, list):
            for item in v:
                if not isinstance(item, dict):
                    continue
                if not item.get("source") or not item.get("target"):
                    continue
                try:
                    rels.append(ExtractedRelation.model_validate(item))
                except Exception:
                    continue
        return rels


class GeneratedPersona(BaseModel):
    model_config = ConfigDict(extra="ignore")
    agent_id: str
    display_name: str = ""
    persona: str = ""
    stance: str = ""


class PersonaBatchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    personas: list[GeneratedPersona] = Field(default_factory=list)

    @field_validator("personas", mode="before")
    @classmethod
    def coerce_personas(cls, v: Any) -> list[Any]:
        out: list[Any] = []
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and item.get("agent_id"):
                    try:
                        out.append(GeneratedPersona.model_validate(item))
                    except Exception:
                        continue
        return out


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    reply: str = ""


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    heading: str = ""
    content: str = ""


class FullReportResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str = ""
    sections: list[ReportSection] = Field(default_factory=list)

    @field_validator("sections", mode="before")
    @classmethod
    def coerce_sections(cls, v: Any) -> list[Any]:
        out: list[Any] = []
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and (item.get("heading") or item.get("content")):
                    try:
                        out.append(ReportSection.model_validate(item))
                    except Exception:
                        continue
        return out


WorldSnapshot.model_rebuild()
SimulationResult.model_rebuild()
