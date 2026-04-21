from __future__ import annotations

import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from leanswarm.engine.config import RuntimeSettings
from leanswarm.engine.models import AgentState
from leanswarm.engine.semantic_store import SemanticStore

_WINDOW_KINDS = ("semantic", "episodic", "working")


@dataclass(slots=True)
class _FallbackCandidate:
    score: float
    kind_priority: int
    index: int
    text: str


class HierarchicalMemoryManager:
    def __init__(
        self,
        working_window: int = 5,
        episodic_window: int = 8,
        semantic_window: int = 8,
        settings: RuntimeSettings | None = None,
        semantic_store: SemanticStore | None = None,
        scope_id: str | None = None,
    ) -> None:
        self.working_window = working_window
        self.episodic_window = episodic_window
        self.semantic_window = semantic_window
        self.scope_id = scope_id or uuid4().hex
        self.settings = settings or self._build_ephemeral_settings()
        self.settings.ensure_dirs()
        self.semantic_store = semantic_store or SemanticStore.from_settings(self.settings)

    def _build_ephemeral_settings(self) -> RuntimeSettings:
        temp_root = Path(tempfile.mkdtemp(prefix="leanswarm-memory-"))
        cache_dir = temp_root / "cache"
        log_dir = temp_root / "logs"
        semantic_store_path = cache_dir / "semantic_memory.sqlite3"
        return RuntimeSettings(
            cache_dir=cache_dir,
            log_dir=log_dir,
            semantic_store_path=semantic_store_path,
            semantic_allow_remote_model=False,
        )

    def close(self) -> None:
        self.semantic_store.close()

    def reset_scope(self, scope_id: str | None = None) -> None:
        self.scope_id = scope_id or uuid4().hex

    def record(self, agent: AgentState, event: str) -> None:
        normalized_event = event.strip()
        if not normalized_event:
            return
        agent.memory.working.append(normalized_event)
        agent.memory.working = agent.memory.working[-self.working_window :]
        self.semantic_store.add_entry(
            self._scope_agent(agent.id),
            normalized_event,
            kind="working",
            source_query=normalized_event,
            deduplicate=False,
        )

    def rollup(self, agent: AgentState, tick: int) -> None:
        if not agent.memory.working:
            return
        latest = agent.memory.working[-1].strip()
        if not latest:
            return
        summary = f"tick {tick}: {latest}"
        agent.memory.episodic.append(summary)
        agent.memory.episodic = agent.memory.episodic[-self.episodic_window :]
        self.semantic_store.add_entry(
            self._scope_agent(agent.id),
            summary,
            kind="episodic",
            tick=tick,
            source_query=latest,
            deduplicate=False,
        )

    def add_semantic_hint(self, agent: AgentState, hint: str) -> None:
        normalized_hint = hint.strip()
        if not normalized_hint:
            return
        if normalized_hint not in agent.memory.semantic:
            agent.memory.semantic.append(normalized_hint)
            agent.memory.semantic = agent.memory.semantic[-self.semantic_window :]
        self.semantic_store.add_entry(
            self._scope_agent(agent.id),
            normalized_hint,
            kind="semantic",
            source_query=normalized_hint,
        )

    def retrieve_semantic_context(self, agent: AgentState, query: str, limit: int = 3) -> list[str]:
        limit = max(1, limit)
        query_tokens = self._tokenize(query)
        candidates: list[_FallbackCandidate] = []
        hits = self.semantic_store.search(
            self._scope_agent(agent.id),
            query,
            limit=max(limit * 2, 6),
            kinds=_WINDOW_KINDS,
        )
        for hit in hits:
            source_weight = {"semantic": 1.3, "episodic": 1.05, "working": 0.9}.get(
                hit.kind,
                1.0,
            )
            blended_score = max(
                float(hit.score),
                self._score_relevance(
                    hit.content, query, query_tokens, source_weight=source_weight
                ),
            )
            candidates.append(
                _FallbackCandidate(
                    score=blended_score,
                    kind_priority={"semantic": 2, "episodic": 1, "working": 0}.get(hit.kind, 0),
                    index=hit.rowid,
                    text=hit.content,
                )
            )

        for index, hint in enumerate(agent.memory.semantic):
            score = self._score_relevance(hint, query, query_tokens, source_weight=1.2)
            if score > 0:
                candidates.append(_FallbackCandidate(score, 2, index, hint))

        for index, episode in enumerate(agent.memory.episodic):
            score = self._score_relevance(episode, query, query_tokens, source_weight=1.0)
            if score > 0:
                candidates.append(_FallbackCandidate(score, 1, index, episode))

        for index, working in enumerate(agent.memory.working):
            score = self._score_relevance(working, query, query_tokens, source_weight=0.8)
            if score > 0:
                candidates.append(_FallbackCandidate(score, 0, index, working))

        if not candidates:
            fallback = [
                *agent.memory.semantic[::-1],
                *agent.memory.episodic[::-1],
                *agent.memory.working[::-1],
            ]
            return self._dedupe_strings(fallback)[:limit]

        ranked = sorted(
            candidates,
            key=lambda item: (-item.score, -item.kind_priority, -item.index, item.text),
        )
        return self._dedupe_strings(item.text for item in ranked)[:limit]

    def should_summarize(self, agent: AgentState) -> bool:
        return len(agent.memory.working) >= self.working_window

    def apply_summary(
        self,
        agent: AgentState,
        summary: str,
        *,
        topics: list[str] | None = None,
    ) -> None:
        normalized_summary = summary.strip()
        if normalized_summary:
            agent.memory.episodic.append(normalized_summary)
            agent.memory.episodic = agent.memory.episodic[-self.episodic_window :]
            self.semantic_store.add_entry(
                self._scope_agent(agent.id),
                normalized_summary,
                kind="episodic",
                source_query=normalized_summary,
            )
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
        normalized_query = query.strip().lower()
        if normalized_query and normalized_query in normalized_text:
            score += 1.0 * source_weight
        for token in query_tokens:
            if token in normalized_text:
                score += 0.1
        return round(score, 4)

    def _tokenize(self, text: str) -> set[str]:
        return {
            self._normalize_token(token)
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2
        }

    def _normalize_token(self, token: str) -> str:
        normalized = token.lower().strip()
        if len(normalized) > 4 and normalized.endswith("ies"):
            return normalized[:-3] + "y"
        if len(normalized) > 4 and normalized.endswith("s") and not normalized.endswith("ss"):
            return normalized[:-1]
        return normalized

    def _dedupe_strings(self, items: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            normalized = str(item).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _scope_agent(self, agent_id: str) -> str:
        return f"{self.scope_id}:{agent_id}"
