from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from leanswarm.api.app import create_app
from leanswarm.engine.config import RuntimeSettings
from leanswarm.engine.models import ActivationMode, SimulationRequest
from leanswarm.engine.simulator import LeanSwarmEngine
from leanswarm.tools.benchmark import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="leanswarm", description="Lean Swarm CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("smoke", help="Run a deterministic smoke simulation")
    smoke_parser.set_defaults(handler=handle_smoke)

    simulate_parser = subparsers.add_parser(
        "simulate", help="Run a simulation from a seed document and prediction question"
    )
    simulate_parser.add_argument("--seed", required=True, help="Path to a seed document")
    simulate_parser.add_argument("--question", required=True, help="Prediction question")
    simulate_parser.add_argument("--rounds", type=int, default=6, help="Maximum tick count")
    simulate_parser.add_argument(
        "--max-agents", type=int, default=24, help="Maximum number of simulated agents"
    )
    simulate_parser.add_argument(
        "--active-agent-fraction",
        type=float,
        default=0.2,
        help="Fraction of agents active on each tick",
    )
    simulate_parser.add_argument(
        "--activation-mode",
        choices=[mode.value for mode in ActivationMode],
        default=ActivationMode.LEAN.value,
        help="Activation strategy for selecting active agents",
    )
    simulate_parser.add_argument(
        "--group-size", type=int, default=5, help="Number of active agents processed per batch"
    )
    simulate_parser.add_argument(
        "--convergence-threshold",
        type=int,
        default=2,
        help="Stable tick streak before ending simulation early",
    )
    simulate_parser.add_argument(
        "--random-seed", type=int, default=7, help="Random seed for deterministic behavior"
    )
    simulate_parser.add_argument(
        "--use-llm",
        action="store_true",
        dest="use_llm",
        default=True,
        help="Enable live LLM calls when credentials are available",
    )
    simulate_parser.add_argument(
        "--no-use-llm",
        action="store_false",
        dest="use_llm",
        help="Disable live LLM calls and force mock execution",
    )
    simulate_parser.set_defaults(handler=handle_simulate)

    api_parser = subparsers.add_parser("api", help="Run the FastAPI server")
    api_parser.add_argument("--host", default=None, help="API host override")
    api_parser.add_argument("--port", type=int, default=None, help="API port override")
    api_parser.set_defaults(handler=handle_api)

    bench_parser = subparsers.add_parser("bench", help="Run the benchmark harness")
    bench_parser.set_defaults(handler=handle_bench)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def handle_smoke(_: argparse.Namespace) -> int:
    result = asyncio.run(LeanSwarmEngine().smoke_test())
    print(result.model_dump_json(indent=2))
    return 0


def handle_simulate(args: argparse.Namespace) -> int:
    seed_path = Path(args.seed)
    request = SimulationRequest(
        seed_document=seed_path.read_text(encoding="utf-8"),
        question=args.question,
        rounds=args.rounds,
        max_agents=args.max_agents,
        active_agent_fraction=args.active_agent_fraction,
        activation_mode=args.activation_mode,
        group_size=args.group_size,
        convergence_threshold=args.convergence_threshold,
        random_seed=args.random_seed,
        use_llm=args.use_llm,
    )
    result = asyncio.run(LeanSwarmEngine().simulate(request))
    print(result.model_dump_json(indent=2))
    return 0


def handle_api(args: argparse.Namespace) -> int:
    settings = RuntimeSettings.from_env()
    host = args.host or settings.api_host
    port = args.port or settings.api_port
    uvicorn.run(create_app(), host=host, port=port)
    return 0


def handle_bench(_: argparse.Namespace) -> int:
    result = asyncio.run(run_benchmark())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
