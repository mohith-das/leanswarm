from __future__ import annotations

import asyncio
import math
import random
from statistics import mean
from typing import Sequence

import networkx as nx

from lean_swarm.engine.config import RuntimeSettings
from lean_swarm.engine.llm import LiteLLMRouter
from lean_swarm.engine.logging import JsonlLogger
from lean_swarm.engine.memory import HierarchicalMemoryManager
from lean_swarm.engine.models import (
    ActivationMode,
    AgentAction,
    AgentState,
    PredictionReport,
    RelationshipEdge,
    SeedWorld,
    SimulationRequest,
    SimulationResult,
    TaskType,
    TickRecord,
    WorldProfile,
    WorldSnapshot,
)
from lean_swarm.engine.population import (
    build_archetype_pool as build_seeded_archetype_pool,
    build_population,
)
from lean_swarm.engine.world import build_seed_world


class LeanSwarmEngine:
    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        router: LiteLLMRouter | None = None,
    ) -> None:
        self.settings = settings or RuntimeSettings.from_env()
        self.settings.ensure_dirs()
        self.router = router or LiteLLMRouter(self.settings)
        self.memory = HierarchicalMemoryManager()
        self.tick_logger = JsonlLogger(self.settings.tick_log_path)
        self.graph = nx.Graph()

    async def smoke_test(self) -> SimulationResult:
        return await self.simulate(
            SimulationRequest(
                seed_document=(
                    "Public opinion is mixed after a week of policy coverage with no major shocks."
                ),
                question="Will sentiment improve over the next reporting cycle?",
                rounds=4,
                max_agents=12,
            )
        )

    async def simulate(self, request: SimulationRequest) -> SimulationResult:
        rng = random.Random(request.random_seed)
        self.graph = nx.Graph()
        seed_world = build_seed_world(request.seed_document, question=request.question)
        population = build_population(
            seed_world.ingestion.source_text,
            request.question,
            world_topics=[topic.label for topic in seed_world.profile.topics],
            world_entities=[entity.label for entity in seed_world.profile.entities],
            max_agents=request.max_agents,
            random_seed=request.random_seed,
        )
        agents = population.agents
        self._register_agents(agents)
        self._prime_agent_memory(agents, seed_world)
        self._seed_relationships(agents, seed_world.profile, rng)
        ticks: list[TickRecord] = []
        stable_streak = 0
        converged = False

        bootstrap = await self.router.route(
            TaskType.WORLD_BOOTSTRAP,
            {
                "question": request.question,
                "seed_document": seed_world.ingestion.source_text[:400],
                "summary": seed_world.profile.summary,
                "context": " ".join(seed_world.profile.focus_terms[:8]),
                "world_topics": [topic.label for topic in seed_world.profile.topics[:6]],
                "world_entities": [entity.label for entity in seed_world.profile.entities[:6]],
                "sentiment": seed_world.profile.sentiment.label.value,
                "agent_count": len(agents),
                "use_llm": request.use_llm,
            },
        )
        self._apply_bootstrap_hints(agents, seed_world.profile, bootstrap)

        for tick_index in range(1, request.rounds + 1):
            active_agents = self._select_active_agents(
                agents,
                request=request,
                rng=rng,
                world_profile=seed_world.profile,
                ticks=ticks,
            )
            grouped_agents = list(self._chunk(active_agents, request.group_size))
            batch_responses = await asyncio.gather(
                *[
                    self.router.route(
                        TaskType.AGENT_BATCH,
                        {
                            "question": request.question,
                            "tick": tick_index,
                            "seed_excerpt": seed_world.ingestion.source_text[:320],
                            "summary": seed_world.profile.summary,
                            "context": " ".join(seed_world.profile.focus_terms[:8]),
                            "world_topics": [topic.label for topic in seed_world.profile.topics[:6]],
                            "world_entities": [entity.label for entity in seed_world.profile.entities[:6]],
                            "sentiment": seed_world.profile.sentiment.label.value,
                            "use_llm": request.use_llm,
                            "agents": [
                                {
                                    "id": agent.id,
                                    "name": agent.name,
                                    "archetype": agent.archetype,
                                    "mood": agent.mood,
                                    "energy": agent.energy,
                                    "attention": agent.attention,
                                }
                                for agent in group
                            ],
                        },
                    )
                    for group in grouped_agents
                ]
            )

            actions = [
                AgentAction.model_validate(action)
                for response in batch_responses
                for action in response.get("actions", [])
            ]
            tick = self._apply_actions(
                tick_index,
                active_agents,
                actions,
                seed_world.profile,
                rng,
            )
            memory_events = await self._summarize_active_memories(
                active_agents,
                tick_index,
                request,
                seed_world.profile,
            )
            tick.events.extend(memory_events)
            ticks.append(tick)
            self.tick_logger.log(tick.model_dump())

            if tick.stable:
                stable_streak += 1
            else:
                stable_streak = 0

            if stable_streak >= request.convergence_threshold:
                converged = True
                break

        synthesis = await self.router.route(
            TaskType.PREDICTION_SYNTHESIS,
            {
                "question": request.question,
                "seed_document": seed_world.ingestion.source_text[:400],
                "summary": seed_world.profile.summary,
                "context": " ".join(seed_world.profile.focus_terms[:8]),
                "world_topics": [topic.label for topic in seed_world.profile.topics[:6]],
                "world_entities": [entity.label for entity in seed_world.profile.entities[:6]],
                "sentiment": seed_world.profile.sentiment.label.value,
                "ticks": [tick.model_dump() for tick in ticks],
                "agent_count": len(agents),
                "use_llm": request.use_llm,
            },
        )

        report = PredictionReport(
            question=request.question,
            prediction=str(synthesis.get("prediction", "No prediction generated.")),
            confidence=float(synthesis.get("confidence", 0.5)),
            rationale=[str(item) for item in synthesis.get("rationale", [])],
            key_events=[event for tick in ticks for event in tick.events][:5],
            converged=converged,
            tick_count=len(ticks),
            llm_calls=self.router.route_calls,
            cache_hit_rate=self.router.cache_hit_rate,
        )
        world = self._build_world_snapshot(
            agents,
            seed_world=seed_world,
            population_summary=population.profile.summary if population.profile else None,
        )
        return SimulationResult(request=request, report=report, ticks=ticks, world=world)

    def build_archetype_pool(
        self,
        seed_document: str = "",
        question: str = "",
        *,
        world_topics: Sequence[str] | None = None,
        world_entities: Sequence[str] | None = None,
    ) -> list[str]:
        return build_seeded_archetype_pool(
            seed_document,
            question,
            world_topics=world_topics,
            world_entities=world_entities,
        )

    def _register_agents(self, agents: list[AgentState]) -> None:
        for agent in agents:
            self.graph.add_node(agent.id, name=agent.name, archetype=agent.archetype)

    def _select_active_agents(
        self,
        agents: list[AgentState],
        request: SimulationRequest,
        rng: random.Random,
        world_profile: WorldProfile,
        ticks: Sequence[TickRecord],
    ) -> list[AgentState]:
        if request.activation_mode == ActivationMode.NAIVE:
            return list(agents)
        active_fraction = self._activation_trigger_fraction(
            request.active_agent_fraction,
            ticks=ticks,
            world_profile=world_profile,
        )
        target_count = max(1, math.ceil(len(agents) * active_fraction))
        ranked = sorted(
            agents,
            key=lambda agent: (
                agent.attention * 0.45
                + agent.energy * 0.18
                + self._world_relevance(agent, world_profile) * 0.24
                + self._relationship_pressure(agent) * 0.08
                + min(0.08, len(agent.memory.working) * 0.015)
                + rng.random() * 0.18
            ),
            reverse=True,
        )
        selected = ranked[: max(1, target_count - 1)]
        if len(selected) < target_count:
            remaining = [agent for agent in ranked if agent not in selected]
            if remaining:
                wildcard = max(
                    remaining,
                    key=lambda agent: self._world_relevance(agent, world_profile) + rng.random() * 0.2,
                )
                selected.append(wildcard)
        return selected[:target_count]

    def _chunk(self, agents: list[AgentState], size: int) -> list[list[AgentState]]:
        return [agents[index : index + size] for index in range(0, len(agents), size)]

    def _apply_actions(
        self,
        tick_index: int,
        active_agents: list[AgentState],
        actions: list[AgentAction],
        world_profile: WorldProfile,
        rng: random.Random,
    ) -> TickRecord:
        deltas: list[float] = []
        events: list[str] = []
        peers = {agent.id: agent for agent in active_agents}

        for action in actions:
            if action.agent_id not in peers:
                continue
            agent = peers[action.agent_id]
            before = (agent.mood, agent.energy, agent.attention)
            agent.mood = self._clamp(agent.mood + action.delta_mood)
            agent.energy = self._clamp(agent.energy + action.delta_energy)
            agent.attention = self._clamp(agent.attention + action.delta_attention)
            self.memory.record(agent, action.description)
            self.memory.rollup(agent, tick_index)
            events.append(action.description)

            for peer in peers.values():
                if peer.id == agent.id:
                    continue
                delta = self._relationship_delta(agent, peer, action, world_profile, rng)
                existing = agent.relationships.get(peer.id, 0.0)
                weight = self._clamp(existing + delta, low=-1.0)
                if abs(weight) < 0.03 and abs(existing) < 0.03:
                    continue
                agent.relationships[peer.id] = weight
                peer.relationships[agent.id] = weight
                self.graph.add_edge(agent.id, peer.id, weight=weight)

            after = (agent.mood, agent.energy, agent.attention)
            deltas.append(mean(abs(a - b) for a, b in zip(after, before, strict=True)))

        mean_delta = mean(deltas) if deltas else 0.0
        return TickRecord(
            tick=tick_index,
            active_agent_ids=[agent.id for agent in active_agents],
            events=events,
            mean_delta=round(mean_delta, 4),
            stable=mean_delta < self._stability_threshold(world_profile),
        )

    def _build_world_snapshot(
        self,
        agents: list[AgentState],
        *,
        seed_world: SeedWorld,
        population_summary: str | None,
    ) -> WorldSnapshot:
        edges = [
            RelationshipEdge(source=source, target=target, weight=round(data["weight"], 3))
            for source, target, data in self.graph.edges(data=True)
        ]
        return WorldSnapshot(
            agents=agents,
            edges=edges,
            profile=seed_world.profile,
            graph=seed_world.graph,
            population_summary=population_summary,
        )

    def _prime_agent_memory(self, agents: list[AgentState], seed_world: SeedWorld) -> None:
        shared_hints = [
            seed_world.profile.sentiment.label.value,
            *seed_world.profile.focus_terms[:4],
            *[topic.label for topic in seed_world.profile.topics[:2]],
        ]
        for agent in agents:
            self.memory.add_semantic_hint(agent, agent.archetype)
            for hint in shared_hints:
                if hint:
                    self.memory.add_semantic_hint(agent, hint)

    def _apply_bootstrap_hints(
        self,
        agents: list[AgentState],
        world_profile: WorldProfile,
        bootstrap: dict[str, object],
    ) -> None:
        hints = [str(value) for value in bootstrap.get("topics", []) if str(value).strip()]
        hints.extend(str(value) for value in bootstrap.get("entities", []) if str(value).strip())
        hints.extend(
            str(value) for value in bootstrap.get("pressure_points", []) if str(value).strip()
        )
        prioritized = hints[:4] or world_profile.focus_terms[:4]
        for agent in agents[: max(1, min(6, len(agents)))]:
            for hint in prioritized:
                self.memory.add_semantic_hint(agent, hint)

    def _seed_relationships(
        self,
        agents: list[AgentState],
        world_profile: WorldProfile,
        rng: random.Random,
    ) -> None:
        for index, agent in enumerate(agents):
            for peer in agents[index + 1 :]:
                agent_relevance = self._world_relevance(agent, world_profile)
                peer_relevance = self._world_relevance(peer, world_profile)
                shared_domain = self._parse_archetype(agent.archetype)[2] == self._parse_archetype(
                    peer.archetype
                )[2]
                shared_stance = self._parse_archetype(agent.archetype)[0] == self._parse_archetype(
                    peer.archetype
                )[0]
                base = ((agent_relevance + peer_relevance) / 2.0) * 0.25
                if shared_domain:
                    base += 0.08
                if shared_stance:
                    base += 0.04
                base += rng.uniform(-0.025, 0.025)
                weight = round(self._clamp(base, low=-1.0), 3)
                if abs(weight) < 0.075:
                    continue
                agent.relationships[peer.id] = weight
                peer.relationships[agent.id] = weight
                self.graph.add_edge(agent.id, peer.id, weight=weight)

    async def _summarize_active_memories(
        self,
        active_agents: list[AgentState],
        tick_index: int,
        request: SimulationRequest,
        world_profile: WorldProfile,
    ) -> list[str]:
        candidates = [agent for agent in active_agents if self.memory.should_summarize(agent)]
        if tick_index % 2 == 0 and not candidates:
            candidates = [
                agent
                for agent in active_agents
                if agent.memory.working and self._world_relevance(agent, world_profile) > 0.2
            ][:1]
        if not candidates:
            return []
        selected_candidates = candidates[: max(1, len(active_agents) // 2)]

        responses = await asyncio.gather(
            *[
                self.router.route(
                    TaskType.MEMORY_SUMMARY,
                    {
                        "question": request.question,
                        "agent_id": agent.id,
                        "summary": world_profile.summary,
                        "context": " ".join(world_profile.focus_terms[:8]),
                        "recent_observations": agent.memory.working[-self.memory.working_window :],
                        "episodic": agent.memory.episodic[-4:],
                        "semantic_context": self.memory.retrieve_semantic_context(
                            agent,
                            query=f"{request.question} {' '.join(world_profile.focus_terms[:4])}",
                            limit=3,
                        ),
                        "use_llm": request.use_llm,
                    },
                )
                for agent in selected_candidates
            ]
        )

        summary_events: list[str] = []
        for agent, response in zip(selected_candidates, responses, strict=True):
            topics = [str(item) for item in response.get("topics", []) if str(item).strip()]
            retained = [
                str(item) for item in response.get("retained_signals", []) if str(item).strip()
            ]
            summary = str(response.get("summary", "")).strip()
            self.memory.apply_summary(agent, summary, topics=topics + retained)
            focus = topics[:2] or retained[:2] or world_profile.focus_terms[:2]
            summary_events.append(
                f"{agent.name} condensed recent signals around {self._format_terms(focus)}."
            )
        return summary_events

    def _world_relevance(self, agent: AgentState, world_profile: WorldProfile) -> float:
        stance, modality, domain = self._parse_archetype(agent.archetype)
        focus_blob = " ".join(
            [
                world_profile.summary.lower(),
                " ".join(world_profile.focus_terms).lower(),
                " ".join(topic.label.lower() for topic in world_profile.topics[:4]),
                " ".join(entity.label.lower() for entity in world_profile.entities[:4]),
            ]
        )
        relevance = 0.08
        if domain and domain in focus_blob:
            relevance += 0.18
        if modality in {"watchdog", "sentinel"} and world_profile.uncertainty > 0.45:
            relevance += 0.06
        if modality in {"amplifier", "organizer", "broker"} and world_profile.salience > 0.45:
            relevance += 0.06
        if stance in {"cautious", "skeptical"} and world_profile.sentiment.label.value in {
            "negative",
            "mixed",
        }:
            relevance += 0.05
        if stance in {"optimistic", "pragmatic"} and world_profile.sentiment.label.value in {
            "positive",
            "neutral",
        }:
            relevance += 0.04
        if any(term[:6].lower() in agent.name.lower() for term in world_profile.focus_terms[:4]):
            relevance += 0.04
        return self._clamp(relevance)

    def _relationship_pressure(self, agent: AgentState) -> float:
        if not agent.relationships:
            return 0.0
        return min(1.0, mean(abs(weight) for weight in agent.relationships.values()))

    def _relationship_delta(
        self,
        agent: AgentState,
        peer: AgentState,
        action: AgentAction,
        world_profile: WorldProfile,
        rng: random.Random,
    ) -> float:
        stance_a, modality_a, domain_a = self._parse_archetype(agent.archetype)
        stance_b, modality_b, domain_b = self._parse_archetype(peer.archetype)
        delta = 0.0
        if domain_a == domain_b:
            delta += 0.04
        if modality_a == modality_b:
            delta += 0.015
        if stance_a == stance_b:
            delta += 0.02
        if {stance_a, stance_b} == {"optimistic", "skeptical"}:
            delta -= 0.035
        if action.action_type in {"coordinate", "sync", "connect"}:
            delta += 0.04
        elif action.action_type in {"observe", "amplify", "reassess"}:
            delta += 0.02
        elif action.action_type in {"challenge", "probe"}:
            delta -= 0.01 if domain_a != domain_b else 0.01
        delta += ((self._world_relevance(agent, world_profile) + self._world_relevance(peer, world_profile)) / 2.0 - 0.18) * 0.12
        delta += rng.uniform(-0.015, 0.015)
        return round(delta, 3)

    def _parse_archetype(self, archetype: str) -> tuple[str, str, str]:
        parts = archetype.split("-", 2)
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        if len(parts) == 2:
            return parts[0], parts[1], ""
        return archetype, "", ""

    def _stability_threshold(self, world_profile: WorldProfile) -> float:
        return round(0.02 + (world_profile.uncertainty * 0.01), 4)

    def _activation_trigger_fraction(
        self,
        base_fraction: float,
        *,
        ticks: Sequence[TickRecord],
        world_profile: WorldProfile,
    ) -> float:
        recent_volatility = mean(tick.mean_delta for tick in ticks[-2:]) if ticks else 0.0
        baseline = self._stability_threshold(world_profile)
        volatility_boost = (recent_volatility - baseline) * 3.2
        world_signal = ((world_profile.uncertainty - 0.5) * 0.16) + (
            (world_profile.salience - 0.5) * 0.14
        )
        trigger_boost = self._clamp(volatility_boost + world_signal, low=-0.12, high=0.25)
        return self._clamp(base_fraction + trigger_boost, low=0.05, high=1.0)

    def _format_terms(self, terms: Sequence[str]) -> str:
        if not terms:
            return "the current narrative"
        if len(terms) == 1:
            return str(terms[0])
        if len(terms) == 2:
            return f"{terms[0]} and {terms[1]}"
        return ", ".join(str(term) for term in terms[:-1]) + f", and {terms[-1]}"

    def _clamp(self, value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, round(value, 4)))
