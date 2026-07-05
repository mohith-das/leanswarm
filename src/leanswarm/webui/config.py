from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

class WebUISettings(BaseModel):
    data_dir: Path = Field(default=Path(".leanswarm/ui"))
    allow_signup: bool = True
    secure_cookies: bool = False
    max_rounds: int = 12
    max_agents: int = 48
    max_seed_chars: int = 20000
    max_concurrent_runs: int = 2
    runs_per_hour_per_ip: int = 10
    retention_seconds: int = 7200

    @classmethod
    def from_env(cls) -> WebUISettings:
        data_dir = Path(os.getenv("LEANSWARM_UI_DATA_DIR", ".leanswarm/ui"))
        return cls(
            data_dir=data_dir,
            allow_signup=_env_flag("LEANSWARM_UI_ALLOW_SIGNUP", True),
            secure_cookies=_env_flag("LEANSWARM_UI_SECURE_COOKIES", False),
            max_rounds=int(os.getenv("LEANSWARM_UI_MAX_ROUNDS", "12")),
            max_agents=int(os.getenv("LEANSWARM_UI_MAX_AGENTS", "48")),
            max_seed_chars=int(os.getenv("LEANSWARM_UI_MAX_SEED_CHARS", "20000")),
            max_concurrent_runs=int(os.getenv("LEANSWARM_UI_MAX_CONCURRENT_RUNS", "2")),
            runs_per_hour_per_ip=int(os.getenv("LEANSWARM_UI_RUNS_PER_HOUR_PER_IP", "10")),
            retention_seconds=int(os.getenv("LEANSWARM_UI_RETENTION_SECONDS", "7200")),
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "runs").mkdir(parents=True, exist_ok=True)
