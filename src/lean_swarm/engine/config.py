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
    dry_run: bool = True
    max_concurrency: int = 4
    retry_attempts: int = 3
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    flagship_model: str = "gpt-4.1"
    standard_model: str = "gpt-4.1-mini"
    cheap_model: str = "gpt-4.1-nano"

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            cache_dir=Path(os.getenv("LEANSWARM_CACHE_DIR", ".leanswarm/cache")),
            log_dir=Path(os.getenv("LEANSWARM_LOG_DIR", ".leanswarm/logs")),
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

