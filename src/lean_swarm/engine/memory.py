from __future__ import annotations

import re

from lean_swarm.engine.models import AgentState


class HierarchicalMemoryManager:
    def __init__(self, working_window: int = 5) -> None:
        self.working_window = working_window

    def record(self, agent: AgentState, event: str) -> None:
        agent.memory.working.append(event)
        agent.memory.working = agent.memory.working[-self.working_window :]

    def rollup(self, agent: AgentState, tick: int) -> None:
        if not agent.memory.working:
            return
        latest = agent.memory.working[-1]
        agent.memory.episodic.append(f"tick {tick}: {latest}")
        agent.memory.episodic = agent.memory.episodic[-8:]

    def add_semantic_hint(self, agent: AgentState, hint: str) -> None:
        if hint not in agent.memory.semantic:
            agent.memory.semantic.append(hint)
            agent.memory.semantic = agent.memory.semantic[-8:]

    def retrieve_semantic_context(self, agent: AgentState, query: str, limit: int = 3) -> list[str]:
        query_tokens = self._tokenize(query)
        candidates: list[tuple[float, int, int, str]] = []

        for index, hint in enumerate(agent.memory.semantic):
            score = self._score_relevance(hint, query, query_tokens, source_weight=1.2)
            if score > 0:
                candidates.append((score, 1, index, hint))

        for index, episode in enumerate(agent.memory.episodic):
            score = self._score_relevance(episode, query, query_tokens, source_weight=1.0)
            if score > 0:
                candidates.append((score, 0, index, episode))

        if not candidates:
            fallback = [*agent.memory.semantic[::-1], *agent.memory.episodic[::-1]]
            unique_fallback = list(dict.fromkeys(item for item in fallback if item.strip()))
            return unique_fallback[: max(1, limit)]

        ranked = sorted(candidates, key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        context = []
        seen: set[str] = set()
        for _, _, _, entry in ranked:
            normalized = entry.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            context.append(normalized)
            if len(context) >= max(1, limit):
                break
        return context

    def should_summarize(self, agent: AgentState) -> bool:
        return len(agent.memory.working) >= self.working_window

    def apply_summary(
        self,
        agent: AgentState,
        summary: str,
        *,
        topics: list[str] | None = None,
    ) -> None:
        if summary:
            agent.memory.episodic.append(summary)
            agent.memory.episodic = agent.memory.episodic[-8:]
        agent.memory.working = agent.memory.working[-2:]
        for topic in topics or []:
            if topic:
                self.add_semantic_hint(agent, topic)

    def _score_relevance(
        self,
        text: str,
        query: str,
        query_tokens: set[str],
        *,
        source_weight: float,
    ) -> float:
        normalized_text = text.strip().lower()
        if not normalized_text:
            return 0.0
        text_tokens = self._tokenize(normalized_text)
        overlap = len(query_tokens & text_tokens)
        if overlap == 0 and query_tokens:
            if not any(token in normalized_text for token in query_tokens):
                return 0.0
        score = overlap * source_weight
        if query.strip().lower() and query.strip().lower() in normalized_text:
            score += 1.0 * source_weight
        for token in query_tokens:
            if token in normalized_text:
                score += 0.1
        return round(score, 4)

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}
