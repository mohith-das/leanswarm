# Contributing

Thanks for helping improve LeanSwarm. We welcome focused, production-minded contributions.

## Development setup

1. Create a Python 3.11+ virtual environment.
2. Install in editable mode with dev dependencies: `pip install -e .[dev]`
3. Copy `.env.example` to `.env` if you want live model traffic.
4. Optional: run `pre-commit install`.

## Commands

- `make lint`
- `make test`
- `make build`
- `make smoke`

## Contributor tracks

We especially invite:

- **Performance contributors**: routing efficiency, cache behavior, token/runtime reductions.
- **Researchers**: simulation validity, scoring methods, intervention design, and evaluation methods.
- **Benchmark contributors**: new benchmark datasets, scoring pipelines, and reproducible comparison suites.
- **Local model optimizers**: hardware/runtime tuning, quantization strategies, and lean-vs-naive local benchmarks.

Use the matching issue template when opening work in these areas.

## Guardrails

- Keep all direct LLM traffic inside `src/leanswarm/engine/llm.py`.
- Preserve structured Pydantic models at request/response boundaries.
- Add tests for behavior changes.
- Never commit secrets or local `.env` files.

By contributing, you agree contributions are released under the project MIT license.
