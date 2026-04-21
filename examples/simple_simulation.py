import asyncio

from lean_swarm.engine.simulator import LeanSwarmEngine
from lean_swarm.engine.models import SimulationRequest


async def main() -> None:
    engine = LeanSwarmEngine()
    result = await engine.simulate(
        SimulationRequest(
            seed_document=(
                "Regional analysts note rising uncertainty around a proposed transportation plan."
            ),
            question="Will local approval improve by the next reporting cycle?",
            rounds=5,
        )
    )
    print(result.report.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())

