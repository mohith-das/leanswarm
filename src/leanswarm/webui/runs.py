import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from leanswarm.engine.config import RuntimeSettings
from leanswarm.engine.llm import LiveCredentialsError
from leanswarm.engine.models import SimulationRequest
from leanswarm.engine.pricing import cost_usd
from leanswarm.engine.simulator import LeanSwarmEngine
from leanswarm.webui.config import WebUISettings
from leanswarm.webui.schemas import StartRunRequest


@dataclass
class RunJob:
    id: str
    status: str
    created_at: float
    finished_at: float | None
    request_sanitized: dict[str, Any]
    models: dict[str, Any]
    events: list[dict[str, Any]]
    result: dict[str, Any] | None
    error: str | None
    owner_user_id: int | None
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)

def run_cost(router: Any) -> float | None:
    total_cost = 0.0
    for model, p_tokens in router.prompt_tokens_by_model.items():
        c_tokens = router.completion_tokens_by_model.get(model, 0)
        if p_tokens == 0 and c_tokens == 0:
            continue
        c = cost_usd(model, p_tokens, c_tokens)
        if c is None:
            return None
        total_cost += c
    return round(total_cost, 4)

class RunManager:
    def __init__(self, settings: WebUISettings):
        self.settings = settings
        self.jobs: dict[str, RunJob] = {}
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_runs)

    async def start(self, req: StartRunRequest, user_id: int | None) -> str:
        run_id = uuid.uuid4().hex
        
        # Build per-run RuntimeSettings
        eng_settings = RuntimeSettings.from_env()
        eng_settings.dry_run = not req.live
        eng_settings.flagship_model = req.models.flagship
        eng_settings.standard_model = req.models.standard
        eng_settings.cheap_model = req.models.cheap
        eng_settings.credentials = req.credentials.copy()
        if req.api_base:
            eng_settings.api_base = req.api_base
        if req.api_key:
            eng_settings.api_key = req.api_key
        
        run_dir = self.settings.data_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        eng_settings.log_dir = run_dir / "logs"
        eng_settings.semantic_store_path = run_dir / "semantic.sqlite3"

        engine = LeanSwarmEngine(settings=eng_settings)
        
        sim_request = SimulationRequest(
            seed_document=req.seed_document,
            question=req.question,
            rounds=req.rounds,
            max_agents=req.max_agents,
            active_agent_fraction=req.active_agent_fraction,
            group_size=req.group_size,
            convergence_threshold=req.convergence_threshold,
            random_seed=req.random_seed,
            use_llm=req.live,
        )

        job = RunJob(
            id=run_id,
            status="running",
            created_at=time.monotonic(),
            finished_at=None,
            request_sanitized=req.sanitize(),
            models={
                "flagship": req.models.flagship,
                "standard": req.models.standard,
                "cheap": req.models.cheap,
                "live": req.live,
            },
            events=[],
            result=None,
            error=None,
            owner_user_id=user_id,
        )
        self.jobs[run_id] = job
        asyncio.create_task(self._execute(job, engine, sim_request))
        return run_id

    async def _execute(self, job: RunJob, engine: LeanSwarmEngine, sim_request: SimulationRequest) -> None:
        async with self.semaphore:
            async def push(event: dict[str, Any]) -> None:
                if event.get("type") == "tick":
                    event["cost_usd_so_far"] = run_cost(engine.router)
                async with job.cond:
                    job.events.append(event)
                    job.cond.notify_all()

            await push({"type": "status", "status": "running"})
            
            try:
                result = await engine.simulate(sim_request, on_progress=push)
                c_usd = run_cost(engine.router)
                result_dump: dict[str, Any] = result.model_dump()
                complete_event = {
                    "type": "complete",
                    "result": result_dump,
                    "prompt_tokens_total": engine.router.prompt_tokens_total,
                    "completion_tokens_total": engine.router.completion_tokens_total,
                    "cost_usd": c_usd,
                }
                async with job.cond:
                    job.events.append(complete_event)
                    job.status = "complete"
                    job.result = result_dump
                    job.finished_at = time.monotonic()
                    job.cond.notify_all()
            except Exception as exc:
                msg = str(exc)
                if isinstance(exc, LiveCredentialsError):
                    msg = str(exc)
                else:
                    msg = f"{type(exc).__name__}: {str(exc)}"
                error_event = {"type": "error", "message": msg}
                async with job.cond:
                    job.events.append(error_event)
                    job.status = "error"
                    job.error = msg
                    job.finished_at = time.monotonic()
                    job.cond.notify_all()

    async def purge_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            to_delete = []
            for run_id, job in self.jobs.items():
                if job.finished_at and (now - job.finished_at) > self.settings.retention_seconds:
                    to_delete.append(run_id)
            for run_id in to_delete:
                del self.jobs[run_id]
