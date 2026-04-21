from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from leanswarm.engine.config import RuntimeSettings

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_ENTRY_TABLE = "semantic_entries"
_VECTOR_TABLE = "semantic_vectors"
_DEFAULT_FALLBACK_DIMENSION = 384


@dataclass(slots=True)
class SemanticHit:
    rowid: int
    agent_id: str
    kind: str
    content: str
    score: float
    distance: float | None = None


class _DeterministicEmbedder:
    def __init__(self, dimension: int = _DEFAULT_FALLBACK_DIMENSION) -> None:
        self.dimension = max(32, dimension)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in self._tokenize(text):
            digest = hashlib.blake2b(
                token.encode("utf-8"),
                digest_size=16,
                person=b"leanswarm",
            ).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            weight = 1.0 + min(3.0, len(token) / 6.0)
            vector[index] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0.0:
            return [value / norm for value in vector]
        return vector

    def _tokenize(self, text: str) -> list[str]:
        return [
            self._normalize_token(token)
            for token in _TOKEN_PATTERN.findall(text.lower())
            if len(token) > 2
        ]

    def _normalize_token(self, token: str) -> str:
        normalized = token.lower().strip()
        if len(normalized) > 4 and normalized.endswith("ies"):
            return normalized[:-3] + "y"
        if len(normalized) > 4 and normalized.endswith("s") and not normalized.endswith("ss"):
            return normalized[:-1]
        return normalized


class _SentenceTransformerEmbedder:
    def __init__(self, model: Any, model_name: str) -> None:
        self._model = model
        self.model_name = model_name
        self.dimension = int(model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self._model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except TypeError:
            vectors = self._model.encode(list(texts), show_progress_bar=False)
        if hasattr(vectors, "tolist"):
            raw_vectors = vectors.tolist()
        else:
            raw_vectors = list(vectors)
        if raw_vectors and isinstance(raw_vectors[0], (int, float)):
            raw_vectors = [raw_vectors]
        return [[float(value) for value in vector] for vector in raw_vectors]


class SemanticStore:
    def __init__(
        self,
        db_path: Path,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_entries_per_agent: int = 256,
        fallback_dimension: int = _DEFAULT_FALLBACK_DIMENSION,
        allow_downloads: bool = False,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.max_entries_per_agent = max(16, max_entries_per_agent)
        self.allow_downloads = allow_downloads
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._embedder: _SentenceTransformerEmbedder | _DeterministicEmbedder
        self._deterministic_embedder = _DeterministicEmbedder(fallback_dimension)
        self._vector_available = False
        self._vector_dimension: int | None = None
        self._vector_table_ready = False
        self._initialise_schema()
        self._initialise_embedder()
        self._initialise_vector_backend()

    @classmethod
    def from_settings(cls, settings: RuntimeSettings) -> SemanticStore:
        return cls(
            settings.semantic_store_path,
            model_name=settings.semantic_model_name,
            max_entries_per_agent=settings.semantic_max_entries_per_agent,
            allow_downloads=settings.semantic_allow_remote_model,
        )

    @property
    def backend_name(self) -> str:
        if isinstance(self._embedder, _SentenceTransformerEmbedder):
            return f"sentence-transformers:{self._embedder.model_name}"
        return "deterministic-hash"

    @property
    def embedding_dimension(self) -> int:
        if isinstance(self._embedder, _SentenceTransformerEmbedder):
            return self._embedder.dimension
        return self._deterministic_embedder.dimension

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add_entry(
        self,
        agent_id: str,
        content: str,
        *,
        kind: str,
        tick: int | None = None,
        metadata: dict[str, Any] | None = None,
        source_query: str | None = None,
        deduplicate: bool = True,
    ) -> int | None:
        normalized_content = self._normalize(content)
        if not normalized_content:
            return None

        metadata_json = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if deduplicate:
                existing = self._conn.execute(
                    f"""
                    SELECT rowid
                    FROM {_ENTRY_TABLE}
                    WHERE agent_id = ? AND kind = ? AND normalized_content = ?
                    ORDER BY rowid DESC
                    LIMIT 1
                    """,
                    (agent_id, kind, normalized_content),
                ).fetchone()
                if existing is not None:
                    return int(existing["rowid"])

            embedding = self._embed([normalized_content])[0]
            rowid = self._insert_entry(
                agent_id=agent_id,
                content=content.strip(),
                normalized_content=normalized_content,
                kind=kind,
                tick=tick,
                metadata_json=metadata_json,
                source_query=source_query,
                embedding=embedding,
            )
            self._prune_agent(agent_id)
            self._conn.commit()
            return rowid

    def search(
        self,
        agent_id: str,
        query: str,
        *,
        limit: int = 3,
        kinds: Sequence[str] = ("semantic", "episodic"),
    ) -> list[SemanticHit]:
        normalized_query = self._normalize(query)
        if not normalized_query:
            return self.recent(agent_id, limit=limit, kinds=kinds)

        if isinstance(self._embedder, _DeterministicEmbedder):
            return self._search_lexical(agent_id, normalized_query, limit=limit, kinds=kinds)

        allowed_kinds = tuple(dict.fromkeys(kind for kind in kinds if kind))
        query_tokens = self._tokenize(normalized_query)
        query_embedding = self._embed([normalized_query])[0]
        latest_rowid = self._latest_rowid()

        with self._lock:
            vector_hits = self._vector_search(query_embedding, limit=max(limit * 8, 16))
            entries_by_rowid: dict[int, sqlite3.Row] = {}
            candidate_rows = self._fetch_rows_by_rowid([hit.rowid for hit in vector_hits])

            for row in candidate_rows:
                if row["agent_id"] != agent_id or row["kind"] not in allowed_kinds:
                    continue
                entries_by_rowid.setdefault(int(row["rowid"]), row)

            if len(entries_by_rowid) < max(1, limit):
                recent_rows = self._fetch_recent_rows(
                    agent_id,
                    limit=max(limit * 8, 24),
                    kinds=allowed_kinds,
                )
                for row in recent_rows:
                    rowid = int(row["rowid"])
                    entries_by_rowid.setdefault(rowid, row)

            if not entries_by_rowid:
                recent = self._fetch_recent_rows(agent_id, limit=limit, kinds=allowed_kinds)
                return self._score_rows(
                    recent,
                    query,
                    query_tokens,
                    query_embedding,
                    latest_rowid=latest_rowid,
                )[:limit]

            scored = self._score_rows(
                list(entries_by_rowid.values()),
                query,
                query_tokens,
                query_embedding,
                latest_rowid=latest_rowid,
            )
            return scored[:limit]

    def recent(
        self,
        agent_id: str,
        *,
        limit: int = 3,
        kinds: Sequence[str] = ("semantic", "episodic"),
    ) -> list[SemanticHit]:
        with self._lock:
            rows = self._fetch_recent_rows(agent_id, limit=max(limit, 1) * 4, kinds=kinds)
            return self._score_recent_rows(rows)[:limit]

    def _initialise_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_ENTRY_TABLE} (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    normalized_content TEXT NOT NULL,
                    tick INTEGER,
                    created_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    source_query TEXT,
                    embedding_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{_ENTRY_TABLE}_agent_kind_rowid
                ON {_ENTRY_TABLE}(agent_id, kind, rowid DESC)
                """
            )
            self._conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{_ENTRY_TABLE}_agent_normalized
                ON {_ENTRY_TABLE}(agent_id, kind, normalized_content)
                """
            )
            self._conn.commit()

    def _initialise_embedder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            self._embedder = self._deterministic_embedder
            return

        model_path = Path(self.model_name).expanduser()
        if not self.allow_downloads and not model_path.exists():
            self._embedder = self._deterministic_embedder
            return

        load_attempts = (False,) if self.allow_downloads else (True,)

        for local_files_only in load_attempts:
            try:
                model = SentenceTransformer(
                    self.model_name,
                    local_files_only=local_files_only,
                )
                probe = model.encode(
                    ["semantic store probe"],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                if hasattr(probe, "tolist"):
                    probe_list = probe.tolist()
                else:
                    probe_list = list(probe)
                if probe_list and isinstance(probe_list[0], (int, float)):
                    probe_list = [probe_list]
                dimension = len(probe_list[0]) if probe_list else _DEFAULT_FALLBACK_DIMENSION
                self._deterministic_embedder = _DeterministicEmbedder(dimension)
                self._embedder = _SentenceTransformerEmbedder(model, self.model_name)
                return
            except Exception:
                continue

        self._embedder = self._deterministic_embedder

    def _initialise_vector_backend(self) -> None:
        if self.embedding_dimension <= 0:
            return
        try:
            import sqlite_vss
        except Exception:
            return

        try:
            self._conn.enable_load_extension(True)
            sqlite_vss.load(self._conn)
            self._conn.execute("SELECT vss_version()")
        except Exception:
            return

        self._vector_available = True
        self._vector_dimension = self.embedding_dimension
        existing = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (_VECTOR_TABLE,),
        ).fetchone()
        if existing is not None:
            sql = str(existing["sql"] or "").replace(" ", "").lower()
            expected = f"embedding({self._vector_dimension})"
            if expected not in sql:
                self._vector_available = False
                return
            self._vector_table_ready = True
            return

        try:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {_VECTOR_TABLE} USING vss0(embedding({self._vector_dimension}))"
            )
            self._vector_table_ready = True
        except Exception:
            self._vector_available = False

    def _insert_entry(
        self,
        *,
        agent_id: str,
        content: str,
        normalized_content: str,
        kind: str,
        tick: int | None,
        metadata_json: str,
        source_query: str | None,
        embedding: list[float],
    ) -> int:
        cursor = self._conn.execute(
            f"""
            INSERT INTO {_ENTRY_TABLE} (
                agent_id,
                kind,
                content,
                normalized_content,
                tick,
                created_at,
                metadata_json,
                source_query,
                embedding_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                kind,
                content,
                normalized_content,
                tick,
                time.time(),
                metadata_json,
                source_query,
                json.dumps(embedding, separators=(",", ":")),
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("semantic store insert did not produce a rowid")
        rowid = int(cursor.lastrowid)
        if self._vector_available and self._vector_table_ready:
            try:
                self._conn.execute(
                    f"INSERT INTO {_VECTOR_TABLE}(rowid, embedding) VALUES (?, ?)",
                    (rowid, json.dumps(embedding, separators=(",", ":"))),
                )
            except Exception:
                self._vector_available = False
        return rowid

    def _vector_search(self, query_embedding: list[float], *, limit: int) -> list[SemanticHit]:
        if not self._vector_available or not self._vector_table_ready:
            return []
        try:
            rows = self._conn.execute(
                f"""
                SELECT rowid, distance
                FROM {_VECTOR_TABLE}
                WHERE vss_search(embedding, ?)
                ORDER BY distance ASC
                LIMIT ?
                """,
                (json.dumps(query_embedding, separators=(",", ":")), limit),
            ).fetchall()
        except Exception:
            self._vector_available = False
            return []
        hits: list[SemanticHit] = []
        for row in rows:
            hits.append(
                SemanticHit(
                    rowid=int(row["rowid"]),
                    agent_id="",
                    kind="",
                    content="",
                    score=max(0.0, 1.0 - float(row["distance"])),
                    distance=float(row["distance"]),
                )
            )
        return hits

    def _fetch_entry_row(self, rowid: int) -> sqlite3.Row | None:
        row = self._conn.execute(
            f"SELECT * FROM {_ENTRY_TABLE} WHERE rowid = ?",
            (rowid,),
        ).fetchone()
        return row

    def _fetch_rows_by_rowid(self, rowids: Sequence[int]) -> list[sqlite3.Row]:
        unique_rowids = list(dict.fromkeys(int(rowid) for rowid in rowids))
        if not unique_rowids:
            return []
        placeholders = ",".join("?" for _ in unique_rowids)
        rows = self._conn.execute(
            f"SELECT * FROM {_ENTRY_TABLE} WHERE rowid IN ({placeholders})",
            unique_rowids,
        ).fetchall()
        return list(rows)

    def _fetch_recent_rows(
        self,
        agent_id: str,
        *,
        limit: int,
        kinds: Sequence[str],
    ) -> list[sqlite3.Row]:
        allowed_kinds = tuple(dict.fromkeys(kind for kind in kinds if kind))
        if allowed_kinds:
            placeholders = ",".join("?" for _ in allowed_kinds)
            query = f"""
                SELECT *
                FROM {_ENTRY_TABLE}
                WHERE agent_id = ? AND kind IN ({placeholders})
                ORDER BY rowid DESC
                LIMIT ?
                """
            params: tuple[Any, ...] = (agent_id, *allowed_kinds, limit)
        else:
            query = f"""
                SELECT *
                FROM {_ENTRY_TABLE}
                WHERE agent_id = ?
                ORDER BY rowid DESC
                LIMIT ?
                """
            params = (agent_id, limit)
        rows = self._conn.execute(query, params).fetchall()
        return list(rows)

    def _score_rows(
        self,
        rows: Sequence[sqlite3.Row],
        query: str,
        query_tokens: set[str],
        query_embedding: list[float],
        *,
        latest_rowid: int,
    ) -> list[SemanticHit]:
        scored: list[SemanticHit] = []
        for row in rows:
            score = self._score_entry(
                row, query, query_tokens, query_embedding, latest_rowid=latest_rowid
            )
            if score <= 0.0:
                continue
            scored.append(
                SemanticHit(
                    rowid=int(row["rowid"]),
                    agent_id=str(row["agent_id"]),
                    kind=str(row["kind"]),
                    content=str(row["content"]),
                    score=score,
                    distance=None,
                )
            )
        scored.sort(key=lambda item: (-item.score, -item.rowid, item.kind, item.content))
        return scored

    def _score_recent_rows(self, rows: Sequence[sqlite3.Row]) -> list[SemanticHit]:
        if not rows:
            return []
        newest_rowid = max(int(row["rowid"]) for row in rows)
        scored: list[SemanticHit] = []
        for row in rows:
            rowid = int(row["rowid"])
            recency_bonus = 1.0 / (1.0 + max(0, newest_rowid - rowid))
            kind_bonus = {"semantic": 0.22, "episodic": 0.16, "working": 0.1}.get(
                str(row["kind"]),
                0.0,
            )
            scored.append(
                SemanticHit(
                    rowid=rowid,
                    agent_id=str(row["agent_id"]),
                    kind=str(row["kind"]),
                    content=str(row["content"]),
                    score=round(kind_bonus + recency_bonus * 0.2, 4),
                    distance=None,
                )
            )
        scored.sort(key=lambda item: (-item.score, -item.rowid, item.kind, item.content))
        return scored

    def _search_lexical(
        self,
        agent_id: str,
        query: str,
        *,
        limit: int,
        kinds: Sequence[str],
    ) -> list[SemanticHit]:
        rows = self._fetch_recent_rows(agent_id, limit=max(limit * 8, 24), kinds=kinds)
        if not rows:
            return []

        query_tokens = self._tokenize(query)
        normalized_query = query.strip().lower()
        newest_rowid = max(int(row["rowid"]) for row in rows)
        scored: list[SemanticHit] = []
        for row in rows:
            normalized_content = self._normalize(str(row["content"]))
            content_tokens = self._tokenize(normalized_content)
            overlap = len(query_tokens & content_tokens)
            if overlap == 0 and query_tokens:
                if not any(token in normalized_content for token in query_tokens):
                    continue
            kind_bonus = {"semantic": 0.24, "episodic": 0.16, "working": 0.08}.get(
                str(row["kind"]),
                0.0,
            )
            exact_bonus = 0.0
            if normalized_query and normalized_query in normalized_content:
                exact_bonus += 0.3
            if normalized_query and normalized_content.startswith(normalized_query):
                exact_bonus += 0.08
            recency_bonus = 1.0 / (1.0 + max(0, newest_rowid - int(row["rowid"])) / 8.0)
            score = round((overlap * 0.7) + kind_bonus + exact_bonus + (recency_bonus * 0.12), 4)
            scored.append(
                SemanticHit(
                    rowid=int(row["rowid"]),
                    agent_id=str(row["agent_id"]),
                    kind=str(row["kind"]),
                    content=str(row["content"]),
                    score=score,
                    distance=None,
                )
            )
        scored.sort(key=lambda item: (-item.score, -item.rowid, item.kind, item.content))
        return scored

    def _score_entry(
        self,
        row: sqlite3.Row,
        query: str,
        query_tokens: set[str],
        query_embedding: list[float],
        *,
        latest_rowid: int,
    ) -> float:
        embedding = self._decode_embedding(str(row["embedding_json"]))
        cosine = self._cosine_similarity(query_embedding, embedding)
        if cosine <= 0.0 and not query_tokens:
            return 0.0

        content = str(row["content"])
        normalized_content = self._normalize(content)
        content_tokens = self._tokenize(normalized_content)
        overlap = len(query_tokens & content_tokens)
        kind_bonus = {"semantic": 0.18, "episodic": 0.12, "working": 0.08}.get(
            str(row["kind"]),
            0.0,
        )
        exact_bonus = 0.0
        if query and query in normalized_content:
            exact_bonus += 0.18
        if query and normalized_content.startswith(query):
            exact_bonus += 0.05
        recency_bonus = 1.0 / (1.0 + max(0, latest_rowid - int(row["rowid"])) / 12.0)
        lexical_bonus = min(0.25, overlap * 0.06)
        score = (cosine * 0.62) + lexical_bonus + kind_bonus + exact_bonus + (recency_bonus * 0.08)
        return round(score, 4)

    def _latest_rowid(self) -> int:
        row = self._conn.execute(
            f"SELECT COALESCE(MAX(rowid), 0) AS rowid FROM {_ENTRY_TABLE}"
        ).fetchone()
        return int(row["rowid"]) if row is not None else 0

    def _prune_agent(self, agent_id: str) -> None:
        count_row = self._conn.execute(
            f"SELECT COUNT(*) AS count FROM {_ENTRY_TABLE} WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()
        count = int(count_row["count"]) if count_row is not None else 0
        if count <= self.max_entries_per_agent:
            return
        excess = count - self.max_entries_per_agent
        rowids = [
            int(row["rowid"])
            for row in self._conn.execute(
                f"""
                SELECT rowid
                FROM {_ENTRY_TABLE}
                WHERE agent_id = ?
                ORDER BY rowid ASC
                LIMIT ?
                """,
                (agent_id, excess),
            ).fetchall()
        ]
        if not rowids:
            return
        placeholders = ",".join("?" for _ in rowids)
        self._conn.execute(
            f"DELETE FROM {_ENTRY_TABLE} WHERE rowid IN ({placeholders})",
            rowids,
        )
        if self._vector_available and self._vector_table_ready:
            try:
                self._conn.execute(
                    f"DELETE FROM {_VECTOR_TABLE} WHERE rowid IN ({placeholders})",
                    rowids,
                )
            except Exception:
                self._vector_available = False

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._embedder.encode(list(texts))
        except Exception:
            return self._deterministic_embedder.encode(list(texts))

    def _decode_embedding(self, payload: str) -> list[float]:
        try:
            vector = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if not isinstance(vector, list):
            return []
        return [float(value) for value in vector]

    def _cosine_similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right:
            return 0.0
        length = min(len(left), len(right))
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for index in range(length):
            left_value = float(left[index])
            right_value = float(right[index])
            dot += left_value * right_value
            left_norm += left_value * left_value
            right_norm += right_value * right_value
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / math.sqrt(left_norm * right_norm)

    def _tokenize(self, text: str) -> set[str]:
        return {
            self._normalize_token(token)
            for token in _TOKEN_PATTERN.findall(text.lower())
            if len(token) > 2
        }

    def _normalize(self, text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _normalize_token(self, token: str) -> str:
        normalized = token.lower().strip()
        if len(normalized) > 4 and normalized.endswith("ies"):
            return normalized[:-3] + "y"
        if len(normalized) > 4 and normalized.endswith("s") and not normalized.endswith("ss"):
            return normalized[:-1]
        return normalized
