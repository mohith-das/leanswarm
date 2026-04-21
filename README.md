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

### Install From PyPI

```bash
pip install leanswarm
```

This installs the Python package, API, and CLI. Node.js is not required for the core package.

Optional with `uv`:

```bash
uv pip install leanswarm
```

### Configure Live Providers

Dry-run mode works without any API keys. If you want live model routing, set one of:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

### Run

```bash
leanswarm smoke
leanswarm simulate --seed examples/seed.txt --question "Will public trust rise this quarter?"
leanswarm api
leanswarm bench
```

### Import In Python

```python
from leanswarm.engine.models import SimulationRequest
from leanswarm.engine.simulator import LeanSwarmEngine
```

## Architecture

### Core rules

- All model traffic is routed through `engine/llm.py`.
- Every LLM route and simulation tick is logged.
- The engine uses Pydantic schemas at every boundary.
- LLM calls are retried and concurrency-limited with semaphores.

### Current engine shape

- Tiered model routing with `FLAGSHIP`, `STANDARD`, and `CHEAP` tiers.
- Batched group actions for active agents.
- Seed-ingestion and world-building helpers that extract topics, entities, and a world graph from the seed document.
- Seed-conditioned population construction with archetype jittering and bounded named-agent counts.
- Hybrid numeric state for mood, energy, attention, and relationships.
- Sparse activation and trigger heuristics that keep only a subset of agents active per tick.
- Hierarchical memory slices for working, episodic, and semantic references backed by SQLite with vector-search support and deterministic offline fallback.
- Disk-backed action caching via `diskcache`.
- Early convergence detection on low-delta ticks.
- A minimal Next.js viewer under `web/` for inspecting pasted simulation JSON and exploring the post-simulation world snapshot.

See [docs/architecture.md](docs/architecture.md) for more detail.

### Optional web viewer

The `web/` app is a separate, optional Next.js inspector for pasted simulation JSON. It is not
required to install or use the Python package or `leanswarm` CLI.

```bash
cd web
npm install
npm run dev
```

## CLI Usage

### Smoke test

```bash
leanswarm smoke
```

### Simulate a scenario

```bash
leanswarm simulate \
  --seed examples/seed.txt \
  --question "Will the policy announcement improve sentiment?" \
  --activation-mode lean \
  --active-agent-fraction 0.25
```

### Run the API

```bash
leanswarm api --host 127.0.0.1 --port 8000
```

### Run the benchmark harness

```bash
leanswarm bench
```

## API Usage

### Start server

```bash
leanswarm api
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

`leanswarm bench` runs the same benchmark cases in both `lean` and `naive` activation modes and
returns a comparison payload with:

- top-level deltas: `cost_ratio_naive_to_lean`, `quality_delta_lean_vs_naive`,
  `runtime_ratio_naive_to_lean`
- per-mode outputs under `modes.lean` and `modes.naive` (quality proxy, runtime, cache stats, token
  and estimated cost totals)
- `plot_points`: per-case points with `mode`, `score`, `cost_usd`, `runtime_seconds`,
  `token_total`, and related fields for quality-vs-cost plotting

This lets you compare lean efficiency against naive full activation and plot quality-vs-cost points
directly from benchmark output without extra transforms. The shipped cases are still lightweight
proxy benchmarks rather than a full public benchmark pack.

## Limitations

- Dry-run routing is still heuristic even though it is seed-sensitive and grouped by task type.
- The benchmark harness is still a compact proxy suite, not a full public-opinion evaluation set.
- The web client is intentionally minimal, focused on inspecting and exploring simulation JSON.

## Contributing

Contributions are welcome. Please open a focused PR with a clear summary of the behavior change
and the validation you ran locally.
