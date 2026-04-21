# Lean Swarm

Lean Swarm is a cost-focused multi-agent prediction and simulation engine designed to approximate
MiroFish-class narrative forecasting with aggressive batching, sparse activation, hybrid state, and
strict LLM routing guardrails.

## Overview

Given a seed document and a prediction question, Lean Swarm builds a simulated world of agents,
runs a bounded number of interaction ticks, and returns:

- a structured prediction report
- a post-simulation world snapshot with agent states and relationship edges

The project is structured for open-source collaboration, MIT licensing, and PyPI publishing from
day one.

## Quickstart

### Install

```bash
pip install -e .[dev]
cp .env.example .env
```

Optional with `uv`:

```bash
uv venv
uv pip install -e .[dev]
```

### Run

```bash
lean-swarm smoke
lean-swarm simulate --seed examples/seed.txt --question "Will public trust rise this quarter?"
lean-swarm api
lean-swarm bench
```

## Architecture

### Core rules

- All model traffic is routed through `engine/llm.py`.
- Every LLM route and simulation tick is logged.
- The engine uses Pydantic schemas at every boundary.
- LLM calls are retried and concurrency-limited with semaphores.

### Phase 1 engine shape

- Tiered model routing with `FLAGSHIP`, `STANDARD`, and `CHEAP` tiers.
- Batched group actions for active agents.
- Archetype pooling with 96 archetypes and up to 50 named agents.
- Hybrid numeric state for mood, energy, attention, and relationships.
- Event-driven activation with a bounded active fraction per tick.
- Hierarchical memory slices for working, episodic, and semantic references.
- Disk-backed action caching via `diskcache`.
- Early convergence detection on low-delta ticks.

See [docs/architecture.md](docs/architecture.md) for more detail.

## CLI Usage

### Smoke test

```bash
lean-swarm smoke
```

### Simulate a scenario

```bash
lean-swarm simulate \
  --seed examples/seed.txt \
  --question "Will the policy announcement improve sentiment?" \
  --activation-mode lean \
  --active-agent-fraction 0.25
```

### Run the API

```bash
lean-swarm api --host 127.0.0.1 --port 8000
```

### Run the benchmark harness

```bash
lean-swarm bench
```

## API Usage

### Start server

```bash
lean-swarm api
```

### Example request

```bash
curl -X POST http://127.0.0.1:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "seed_document": "A national survey shows mixed views on new policy proposals.",
    "question": "Will approval improve over the next month?",
    "rounds": 4
  }'
```

## Benchmarks

`lean-swarm bench` runs the same benchmark cases in both `lean` and `naive` activation modes and
returns a comparison payload with:

- top-level deltas: `cost_ratio_naive_to_lean`, `quality_delta_lean_vs_naive`,
  `runtime_ratio_naive_to_lean`
- per-mode outputs under `modes.lean` and `modes.naive` (quality proxy, runtime, cache stats, token
  and estimated cost totals)
- `plot_points`: per-case points with `mode`, `score`, `cost_usd`, `runtime_seconds`,
  `token_total`, and related fields for quality-vs-cost plotting

This lets you compare lean efficiency against naive full activation and plot quality-vs-cost points
directly from benchmark output without extra transforms.

## Limitations

- The Phase 1 engine uses deterministic mock responses unless live credentials are configured.
- Semantic memory is scaffolded but not yet backed by a production embedding and retrieval flow.
- The web client is intentionally minimal at this stage.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for V2 priorities including multi-world simulations, Monte Carlo
runs, intervention testing, replay UX, plugin architecture, large-scale agent support, and
hybrid local+cloud routing.

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), pick an issue template
that matches your contribution track, and open a focused PR.
