from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from leanswarm.api.app import create_app
from leanswarm.engine.config import RuntimeSettings
from leanswarm.engine.models import ActivationMode, SimulationRequest
from leanswarm.engine.simulator import LeanSwarmEngine
from leanswarm.tools.benchmark import run_benchmark
from leanswarm.webui.app import create_webui_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="leanswarm", description="Lean Swarm CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("smoke", help="Run a deterministic smoke simulation")
    smoke_mode_group = smoke_parser.add_mutually_exclusive_group()
    smoke_mode_group.add_argument("--live", action="store_true", help="Enable live LLM mode for this run")
    smoke_mode_group.add_argument("--dry-run", action="store_true", help="Force dry-run mode for this run")
    smoke_parser.set_defaults(handler=handle_smoke)

    simulate_parser = subparsers.add_parser(
        "simulate", help="Run a simulation from a seed document and prediction question"
    )
    sim_mode_group = simulate_parser.add_mutually_exclusive_group()
    sim_mode_group.add_argument("--live", action="store_true", help="Enable live LLM mode for this run")
    sim_mode_group.add_argument("--dry-run", action="store_true", help="Force dry-run mode for this run")
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

    doctor_parser = subparsers.add_parser("doctor", help="Check live mode credentials and models")
    doctor_parser.add_argument("--ping", action="store_true", help="Make a test API call for each model")
    doctor_parser.set_defaults(handler=handle_doctor)

    bench_parser = subparsers.add_parser("bench", help="Run the benchmark harness")
    bench_parser.set_defaults(handler=handle_bench)

    ui_parser = subparsers.add_parser("ui", help="Run the Web UI server")
    ui_parser.add_argument("--host", default=None, help="API host override")
    ui_parser.add_argument("--port", type=int, default=None, help="API port override")
    ui_parser.set_defaults(handler=handle_ui)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))




def _get_settings(args: argparse.Namespace) -> RuntimeSettings:
    settings = RuntimeSettings.from_env()
    if getattr(args, "live", False):
        settings.dry_run = False
    elif getattr(args, "dry_run", False):
        settings.dry_run = True
    mode = "dry-run" if settings.dry_run else "live"
    print(
        f"leanswarm: mode={mode} flagship={settings.flagship_model} standard={settings.standard_model} cheap={settings.cheap_model}",
        file=sys.stderr,
    )
    return settings

def handle_smoke(args: argparse.Namespace) -> int:
    settings = _get_settings(args)
    result = asyncio.run(LeanSwarmEngine(settings=settings).smoke_test())
    print(result.model_dump_json(indent=2))
    return 0

def handle_simulate(args: argparse.Namespace) -> int:
    settings = _get_settings(args)
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
    result = asyncio.run(LeanSwarmEngine(settings=settings).simulate(request))
    print(result.model_dump_json(indent=2))
    return 0


def handle_doctor(args: argparse.Namespace) -> int:
    settings = RuntimeSettings.from_env()
    mode = "dry-run" if settings.dry_run else "live"
    print(f"mode: {mode}")
    print(f"flagship: {settings.flagship_model}")
    print(f"standard: {settings.standard_model}")
    print(f"cheap: {settings.cheap_model}")
    print(f"api_base: {settings.api_base}")
    print(f"LEANSWARM_API_KEY: {'set' if settings.api_key else 'unset'}")

    from leanswarm.engine.llm import LiteLLMRouter
    router = LiteLLMRouter(settings)
    models = list(dict.fromkeys([settings.flagship_model, settings.standard_model, settings.cheap_model]))
    
    exit_code = 0
    for model in models:
        ready, missing = router._live_ready(model)
        status = "ok" if ready else f"missing: {', '.join(missing)}"
        if not ready:
            exit_code = 1
        print(f"\nmodel '{model}' credentials: {status}")

        if args.ping:
            try:
                from litellm import acompletion
                kwargs = {
                    "model": model,
                    "max_tokens": 16,
                    "messages": [
                        {"role": "system", "content": 'Reply with exactly {"ok": true}'},
                        {"role": "user", "content": "ping"}
                    ]
                }
                if settings.api_base is not None:
                    kwargs["api_base"] = settings.api_base
                if settings.api_key is not None:
                    kwargs["api_key"] = settings.api_key

                import time
                start_t = time.perf_counter()
                asyncio.run(acompletion(**kwargs))
                latency = int((time.perf_counter() - start_t) * 1000)
                print(f"  ping: ok ({latency}ms)")
            except Exception as e:
                exit_code = 1
                msg_lines = str(e).split("\n", 1)
                first_line = msg_lines[0] if msg_lines else "Unknown error"
                print(f"  ping: {type(e).__name__}: {first_line}")

    return exit_code

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


def handle_ui(args: argparse.Namespace) -> int:
    from leanswarm.webui.config import WebUISettings
    settings = RuntimeSettings.from_env()
    host = args.host or settings.api_host
    port = args.port or settings.api_port
    
    webui_settings = WebUISettings.from_env()
    signup_status = "on" if webui_settings.allow_signup else "off"
    print(f"leanswarm ui: serving on http://{host}:{port} (signup={signup_status})", file=sys.stderr)
    
    uvicorn.run(create_webui_app(), host=host, port=port)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
