from typing import Any

from pydantic import BaseModel, Field


class TierModels(BaseModel):
    flagship: str
    standard: str
    cheap: str

class StartRunRequest(BaseModel):
    seed_document: str
    question: str = Field(..., max_length=500)
    rounds: int = 4
    max_agents: int = 12
    group_size: int = 4
    active_agent_fraction: float = 0.2
    convergence_threshold: int = 2
    random_seed: int = 7
    live: bool = False
    models: TierModels
    credentials: dict[str, str] = Field(default_factory=dict)
    api_base: str | None = None
    api_key: str | None = None
    title: str | None = None
    source_urls: list[str] = Field(default_factory=list)
    use_search: bool = False
    max_sources: int = 4

    def sanitize(self) -> dict[str, Any]:
        data = self.model_dump()
        data.pop("credentials", None)
        data.pop("api_key", None)
        return data

class AuthRequest(BaseModel):
    email: str
    password: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    password: str

class PublishRequest(BaseModel):
    title: str | None = None

class EstimateRequest(BaseModel):
    rounds: int = 4
    max_agents: int = 12
    group_size: int = 4
    active_agent_fraction: float = 0.2
    models: TierModels
    seed_chars: int = 0

class DoctorRequest(BaseModel):
    models: TierModels
    credentials: dict[str, str] = Field(default_factory=dict)
    api_base: str | None = None
    api_key: str | None = None
    ping: bool = False


class ChatRequest(BaseModel):
    agent_id: str | None = None
    message: str = ""       # required for /chat (validated there); unused by /report
    history: list[dict[str, str]] = Field(default_factory=list)
    live: bool = False
    models: TierModels | None = None
    credentials: dict[str, str] = Field(default_factory=dict)
    api_base: str | None = None
    api_key: str | None = None
