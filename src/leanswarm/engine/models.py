from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ModelTier(StrEnum):
    FLAGSHIP = "flagship"
    STANDARD = "standard"
    CHEAP = "cheap"


class TaskType(StrEnum):
    WORLD_BOOTSTRAP = "world_bootstrap"
    AGENT_BATCH = "agent_batch"
    MEMORY_SUMMARY = "memory_summary"
    PREDICTION_SYNTHESIS = "prediction_synthesis"


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


class AgentAction(BaseModel):
    agent_id: str
    action_type: str
    description: str
    delta_mood: float = 0.0
    delta_energy: float = 0.0
    delta_attention: float = 0.0


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


WorldSnapshot.model_rebuild()
SimulationResult.model_rebuild()
