from __future__ import annotations

from fastapi import FastAPI

from lean_swarm import __version__
from lean_swarm.engine.models import SimulationRequest, SimulationResult
from lean_swarm.engine.simulator import LeanSwarmEngine


def create_app() -> FastAPI:
    app = FastAPI(
        title="Lean Swarm API",
        version=__version__,
        description="HTTP surface for the Lean Swarm simulation engine.",
    )
    engine = LeanSwarmEngine()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/smoke", response_model=SimulationResult)
    async def smoke() -> SimulationResult:
        return await engine.smoke_test()

    @app.post("/simulate", response_model=SimulationResult)
    async def simulate(request: SimulationRequest) -> SimulationResult:
        return await engine.simulate(request)

    return app

