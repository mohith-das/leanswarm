from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class RuntimeSettings(BaseModel):
    cache_dir: Path = Field(default=Path(".leanswarm/cache"))
    log_dir: Path = Field(default=Path(".leanswarm/logs"))
    semantic_store_path: Path = Field(default=Path(".leanswarm/cache/semantic_memory.sqlite3"))
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_allow_remote_model: bool = False
    semantic_max_entries_per_agent: int = Field(default=256, ge=16, le=4096)
    allow_embedding_downloads: bool = False
    dry_run: bool = True
    max_concurrency: int = 4
    retry_attempts: int = 3
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    flagship_model: str = "gpt-4.1"
    standard_model: str = "gpt-4.1-mini"
    cheap_model: str = "gpt-4.1-nano"

    @classmethod
    def from_env(cls) -> RuntimeSettings:
        cache_dir = Path(os.getenv("LEANSWARM_CACHE_DIR", ".leanswarm/cache"))
        semantic_store_path = Path(
            os.getenv(
                "LEANSWARM_SEMANTIC_STORE_PATH",
                str(cache_dir / "semantic_memory.sqlite3"),
            )
        )
        return cls(
            cache_dir=cache_dir,
            log_dir=Path(os.getenv("LEANSWARM_LOG_DIR", ".leanswarm/logs")),
            semantic_store_path=semantic_store_path,
            semantic_model_name=os.getenv(
                "LEANSWARM_SEMANTIC_MODEL_NAME",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            semantic_allow_remote_model=_env_flag("LEANSWARM_SEMANTIC_ALLOW_REMOTE_MODEL", False),
            semantic_max_entries_per_agent=int(
                os.getenv("LEANSWARM_SEMANTIC_MAX_ENTRIES_PER_AGENT", "256")
            ),
            allow_embedding_downloads=_env_flag(
                "LEANSWARM_ALLOW_EMBEDDING_DOWNLOADS",
                False,
            ),
            dry_run=_env_flag("LEANSWARM_DRY_RUN", True),
            max_concurrency=int(os.getenv("LEANSWARM_MAX_CONCURRENCY", "4")),
            retry_attempts=int(os.getenv("LEANSWARM_RETRY_ATTEMPTS", "3")),
            api_host=os.getenv("LEANSWARM_API_HOST", "127.0.0.1"),
            api_port=int(os.getenv("LEANSWARM_API_PORT", "8000")),
            flagship_model=os.getenv("LEANSWARM_FLAGSHIP_MODEL", "gpt-4.1"),
            standard_model=os.getenv("LEANSWARM_STANDARD_MODEL", "gpt-4.1-mini"),
            cheap_model=os.getenv("LEANSWARM_CHEAP_MODEL", "gpt-4.1-nano"),
        )

    def ensure_dirs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def llm_log_path(self) -> Path:
        return self.log_dir / "llm_calls.jsonl"

    @property
    def tick_log_path(self) -> Path:
        return self.log_dir / "ticks.jsonl"
