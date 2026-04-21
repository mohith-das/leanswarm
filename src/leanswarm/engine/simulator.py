from __future__ import annotations

import asyncio
import math
import random
import re
from collections.abc import Sequence
from statistics import mean
from typing import Any

import networkx as nx

from leanswarm.engine.config import RuntimeSettings
from leanswarm.engine.llm import LiteLLMRouter
from leanswarm.engine.logging import JsonlLogger
from leanswarm.engine.memory import HierarchicalMemoryManager
from leanswarm.engine.models import (
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
from leanswarm.engine.population import (
    build_archetype_pool as build_seeded_archetype_pool,
)
from leanswarm.engine.population import (
    build_population,
)
from leanswarm.engine.world import build_seed_world


class LeanSwarmEngine:
    def __init__(
        self,
        settings: RuntimeSettings | None = None,
        router: LiteLLMRouter | None = None,
    ) -> None:
        self.settings = settings or RuntimeSettings.from_env()
        self.settings.ensure_dirs()
        self.router = router or LiteLLMRouter(self.settings)
        self.memory = HierarchicalMemoryManager(settings=self.settings)
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
        self.memory.reset_scope()
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
            active_agents, activation_profile = self._select_active_agents(
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
                            "world_topics": [
                                topic.label for topic in seed_world.profile.topics[:6]
                            ],
                            "world_entities": [
                                entity.label for entity in seed_world.profile.entities[:6]
                            ],
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
                activation_profile=activation_profile,
                prior_ticks=ticks,
            )
            memory_events = await self._summarize_active_memories(
                active_agents,
                tick_index,
                request,
                seed_world.profile,
            )
            tick.events.extend(memory_events)
            self._finalize_tick(tick, prior_ticks=ticks, world_profile=seed_world.profile)
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
            average_active_fraction=self._average_active_fraction(ticks),
            activation_envelope_hit_rate=self._activation_envelope_hit_rate(ticks),
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
    ) -> tuple[list[AgentState], dict[str, Any]]:
        if request.activation_mode == ActivationMode.NAIVE:
            return list(agents), {
                "fraction": 1.0,
                "pressure": 0.0,
                "triggers": ["naive"],
                "target_count": len(agents),
            }

        activation_triggers = self._activation_triggers(world_profile, ticks)
        active_fraction = self._activation_trigger_fraction(
            request.active_agent_fraction,
            ticks=ticks,
            world_profile=world_profile,
            activation_triggers=activation_triggers,
        )
        target_count = max(1, math.ceil(len(agents) * active_fraction))
        scored_agents = [
            (
                agent,
                self._activation_score(
                    agent,
                    world_profile=world_profile,
                    activation_triggers=activation_triggers,
                    rng=rng,
                ),
            )
            for agent in agents
        ]
        scored_agents.sort(key=lambda item: item[1], reverse=True)
        trigger_threshold = self._activation_threshold(world_profile, activation_triggers)
        selected = [agent for agent, score in scored_agents if score >= trigger_threshold]
        if len(selected) < target_count:
            for agent, _score in scored_agents:
                if agent in selected:
                    continue
                selected.append(agent)
                if len(selected) >= target_count:
                    break
        return selected[:target_count], {
            "fraction": round(target_count / max(1, len(agents)), 4),
            "pressure": round(
                min(
                    1.0,
                    sum(weight for _label, weight in activation_triggers)
                    / max(1, len(activation_triggers)),
                ),
                4,
            ),
            "triggers": [label for label, _weight in activation_triggers],
            "target_count": target_count,
        }

    def _chunk(self, agents: list[AgentState], size: int) -> list[list[AgentState]]:
        return [agents[index : index + size] for index in range(0, len(agents), size)]

    def _apply_actions(
        self,
        tick_index: int,
        active_agents: list[AgentState],
        actions: list[AgentAction],
        world_profile: WorldProfile,
        rng: random.Random,
        *,
        activation_profile: dict[str, Any],
        prior_ticks: Sequence[TickRecord],
    ) -> TickRecord:
        deltas: list[float] = []
        events: list[str] = []
        relationship_changes: list[float] = []
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
                relationship_changes.append(abs(weight - existing))

            after = (agent.mood, agent.energy, agent.attention)
            deltas.append(mean(abs(a - b) for a, b in zip(after, before, strict=True)))

        mean_delta = mean(deltas) if deltas else 0.0
        activation_fraction = self._coerce_float(activation_profile.get("fraction"), default=0.0)
        activation_pressure = self._coerce_float(activation_profile.get("pressure"), default=0.0)
        activation_triggers = self._coerce_str_list(activation_profile.get("triggers"))
        return TickRecord(
            tick=tick_index,
            active_agent_ids=[agent.id for agent in active_agents],
            events=events,
            mean_delta=round(mean_delta, 4),
            activation_fraction=activation_fraction,
            activation_pressure=activation_pressure,
            activation_triggers=activation_triggers,
            relationship_churn=self._relationship_churn(relationship_changes),
            stability_score=self._tick_stability_score(
                mean_delta=mean_delta,
                relationship_churn=self._relationship_churn(relationship_changes),
                activation_pressure=activation_pressure,
                event_pressure=self._event_pressure(events),
                novelty=self._event_novelty(events, prior_ticks),
                world_profile=world_profile,
            ),
            convergence_score=0.0,
            stable=False,
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
        bootstrap: dict[str, Any],
    ) -> None:
        hints = self._coerce_str_list(bootstrap.get("topics"))
        hints.extend(self._coerce_str_list(bootstrap.get("entities")))
        hints.extend(self._coerce_str_list(bootstrap.get("pressure_points")))
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
                shared_domain = (
                    self._parse_archetype(agent.archetype)[2]
                    == self._parse_archetype(peer.archetype)[2]
                )
                shared_stance = (
                    self._parse_archetype(agent.archetype)[0]
                    == self._parse_archetype(peer.archetype)[0]
                )
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

    def _activation_score(
        self,
        agent: AgentState,
        *,
        world_profile: WorldProfile,
        activation_triggers: Sequence[tuple[str, float]],
        rng: random.Random,
    ) -> float:
        stance, modality, _domain = self._parse_archetype(agent.archetype)
        memory_pressure = min(
            1.0,
            (len(agent.memory.working) / max(1, self.memory.working_window)) * 0.6
            + (len(agent.memory.episodic) / 8.0) * 0.4,
        )
        trigger_fit = self._activation_trigger_fit(
            stance=stance,
            modality=modality,
            world_profile=world_profile,
            activation_triggers=activation_triggers,
        )
        score = (
            agent.attention * 0.28
            + agent.energy * 0.12
            + self._world_relevance(agent, world_profile) * 0.28
            + self._relationship_pressure(agent) * 0.1
            + memory_pressure * 0.08
            + trigger_fit * 0.12
            + rng.uniform(-0.035, 0.035)
        )
        return self._clamp(score)

    def _activation_trigger_fit(
        self,
        *,
        stance: str,
        modality: str,
        world_profile: WorldProfile,
        activation_triggers: Sequence[tuple[str, float]],
    ) -> float:
        fit = 0.0
        sentiment = world_profile.sentiment.label.value
        for label, weight in activation_triggers:
            if label in {"watch", "contain", "scrutinize"} and modality in {
                "watchdog",
                "sentinel",
                "analyst",
            }:
                fit += weight * 0.22
            if label in {"coordinate", "synthesize", "clarify"} and modality in {
                "organizer",
                "broker",
                "translator",
            }:
                fit += weight * 0.2
            if label in {"stabilize", "recover"} and stance in {"pragmatic", "optimistic"}:
                fit += weight * 0.16
            if label in {"watch", "contain"} and stance in {"cautious", "skeptical"}:
                fit += weight * 0.15
            if label in {"pressure", "rumor"} and sentiment in {"negative", "mixed"}:
                fit += weight * 0.12
            if label in {"momentum", "clarify"} and sentiment in {"positive", "neutral"}:
                fit += weight * 0.1
        return self._clamp(fit)

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
        delta += (
            (
                self._world_relevance(agent, world_profile)
                + self._world_relevance(peer, world_profile)
            )
            / 2.0
            - 0.18
        ) * 0.12
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
        activation_triggers: Sequence[tuple[str, float]],
    ) -> float:
        recent_volatility = mean(tick.mean_delta for tick in ticks[-2:]) if ticks else 0.0
        baseline = self._stability_threshold(world_profile)
        recent_event_pressure = self._event_pressure(ticks[-1].events if ticks else [])
        trigger_strength = sum(weight for _label, weight in activation_triggers) / max(
            1, len(activation_triggers)
        )
        volatility_boost = (recent_volatility - baseline) * 2.8
        world_signal = (
            ((world_profile.uncertainty - 0.5) * 0.05)
            + ((world_profile.salience - 0.5) * 0.04)
            + ((world_profile.complexity - 0.5) * 0.03)
        )
        trigger_boost = (trigger_strength * 0.05) + (recent_event_pressure * 0.035)
        target = base_fraction + volatility_boost + world_signal + trigger_boost
        return self._clamp(target, low=0.15, high=0.25)

    def _activation_threshold(
        self,
        world_profile: WorldProfile,
        activation_triggers: Sequence[tuple[str, float]],
    ) -> float:
        trigger_strength = sum(weight for _label, weight in activation_triggers) / max(
            1, len(activation_triggers)
        )
        threshold = (
            0.44
            + ((world_profile.uncertainty - 0.5) * 0.08)
            - ((world_profile.salience - 0.5) * 0.05)
            + (trigger_strength * 0.08)
        )
        return self._clamp(threshold, low=0.34, high=0.72)

    def _activation_triggers(
        self,
        world_profile: WorldProfile,
        ticks: Sequence[TickRecord],
    ) -> list[tuple[str, float]]:
        triggers: list[tuple[str, float]] = []
        recent_volatility = mean(tick.mean_delta for tick in ticks[-2:]) if ticks else 0.0
        last_events = " ".join(ticks[-1].events).lower() if ticks else ""

        if (
            world_profile.uncertainty >= 0.58
            or recent_volatility >= self._stability_threshold(world_profile) * 1.8
        ):
            triggers.append(
                ("watch", self._clamp((world_profile.uncertainty + recent_volatility) / 2.0))
            )
        if world_profile.salience >= 0.55:
            triggers.append(("coordinate", self._clamp(world_profile.salience)))
        if world_profile.complexity >= 0.58:
            triggers.append(("synthesize", self._clamp(world_profile.complexity)))
        if world_profile.sentiment.label.value in {"negative", "mixed"}:
            triggers.append(("contain", self._clamp(0.45 + world_profile.uncertainty * 0.4)))
        if any(
            term in last_events
            for term in {"challenge", "probe", "risk", "rumor", "pressure", "delay"}
        ):
            triggers.append(("pressure", 0.78))
        if any(term in last_events for term in {"summary", "condensed", "clarify", "align"}):
            triggers.append(("clarify", 0.72))
        if any(term in last_events for term in {"coordinate", "sync", "connect"}):
            triggers.append(("momentum", 0.68))
        if not triggers:
            triggers.append(("stabilize", 0.55))
        return triggers

    def _relationship_churn(self, relationship_changes: Sequence[float]) -> float:
        if not relationship_changes:
            return 0.0
        return self._clamp(mean(relationship_changes) * 6.0)

    def _event_pressure(self, events: Sequence[str]) -> float:
        if not events:
            return 0.0
        raw = len(events) / max(1, len(events) + 2)
        text = " ".join(events).lower()
        if any(
            term in text for term in {"challenge", "probe", "risk", "rumor", "pressure", "delay"}
        ):
            raw += 0.15
        if any(term in text for term in {"summary", "condensed", "clarify", "align"}):
            raw += 0.1
        return self._clamp(raw)

    def _event_novelty(self, events: Sequence[str], prior_ticks: Sequence[TickRecord]) -> float:
        if not prior_ticks:
            return 0.75 if events else 0.0
        previous_events = " ".join(prior_ticks[-1].events)
        current_tokens = set(self._tokenize(" ".join(events)))
        previous_tokens = set(self._tokenize(previous_events))
        if not current_tokens:
            return 0.0
        union = current_tokens | previous_tokens
        if not union:
            return 0.0
        overlap = len(current_tokens & previous_tokens) / len(union)
        return self._clamp(1.0 - overlap)

    def _tick_stability_score(
        self,
        *,
        mean_delta: float,
        relationship_churn: float,
        activation_pressure: float,
        event_pressure: float,
        novelty: float,
        world_profile: WorldProfile,
    ) -> float:
        motion_component = min(
            1.0, mean_delta / max(self._stability_threshold(world_profile), 0.01)
        )
        world_component = (
            0.25 + (world_profile.uncertainty * 0.15) + (world_profile.complexity * 0.1)
        )
        stability = (
            motion_component * 0.42
            + relationship_churn * 0.2
            + activation_pressure * 0.12
            + event_pressure * 0.16
            + novelty * 0.1
            + world_component * 0.05
        )
        return self._clamp(stability)

    def _convergence_threshold(self, world_profile: WorldProfile) -> float:
        threshold = (
            0.38
            + ((0.5 - world_profile.uncertainty) * 0.08)
            + ((0.5 - world_profile.complexity) * 0.04)
        )
        return self._clamp(threshold, low=0.28, high=0.5)

    def _finalize_tick(
        self,
        tick: TickRecord,
        *,
        prior_ticks: Sequence[TickRecord],
        world_profile: WorldProfile,
    ) -> None:
        event_pressure = self._event_pressure(tick.events)
        novelty = self._event_novelty(tick.events, prior_ticks)
        convergence_score = self._tick_stability_score(
            mean_delta=tick.mean_delta,
            relationship_churn=tick.relationship_churn,
            activation_pressure=tick.activation_pressure,
            event_pressure=event_pressure,
            novelty=novelty,
            world_profile=world_profile,
        )
        tick.stability_score = round(convergence_score, 4)
        tick.convergence_score = round(1.0 - convergence_score, 4)
        tick.stable = convergence_score < self._convergence_threshold(world_profile)

    def _average_active_fraction(self, ticks: Sequence[TickRecord]) -> float:
        if not ticks:
            return 0.0
        return self._clamp(mean(tick.activation_fraction for tick in ticks))

    def _activation_envelope_hit_rate(self, ticks: Sequence[TickRecord]) -> float:
        if not ticks:
            return 0.0
        hits = sum(1 for tick in ticks if 0.15 <= tick.activation_fraction <= 0.25)
        return round(hits / len(ticks), 3)

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

    def _coerce_float(self, value: object, *, default: float) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    def _coerce_str_list(self, value: object) -> list[str]:
        if isinstance(value, str):
            normalized = value.strip()
            return [normalized] if normalized else []
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if str(item).strip()]
        return []

    def _tokenize(self, text: str) -> list[str]:
        return [
            token.lower()
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9'\-]*", text)
            if len(token) >= 4
        ]
