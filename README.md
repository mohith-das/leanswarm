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

Dry-run is the default. To use real models, set `LEANSWARM_DRY_RUN=false` or pass `--live` on the CLI. Configure ONE of the provider blocks below, then set the tier models:

```bash
# OpenAI:      OPENAI_API_KEY=...        models: gpt-4.1, gpt-4.1-mini, ...
# Anthropic:   ANTHROPIC_API_KEY=...     models: anthropic/claude-sonnet-5, ...
# DeepSeek:    DEEPSEEK_API_KEY=...      models: deepseek/deepseek-chat, deepseek/deepseek-reasoner
# Zhipu GLM:   ZHIPUAI_API_KEY=...       models: zhipuai/glm-4.5, ...
# Groq:        GROQ_API_KEY=...          models: groq/llama-3.3-70b-versatile, ...
# OpenRouter:  OPENROUTER_API_KEY=...    models: openrouter/<any>, ...
# Ollama:      (no key)                  models: ollama/llama3.1, ...
#
# Any OpenAI-compatible endpoint (MiniMax, GLM, vLLM, custom gateways):
#   LEANSWARM_API_BASE=https://api.minimax.io/v1
#   LEANSWARM_API_KEY=...
#   LEANSWARM_FLAGSHIP_MODEL=openai/MiniMax-M2
```

Verify your credentials with `leanswarm doctor` or `leanswarm doctor --ping`.
Live-mode responses are schema-validated with one automatic repair attempt. If a model cannot produce valid JSON after repair, a logged `mock_fallback` occurs.

In live mode, the engine runs a single cheap-tier LLM extraction pass over up to 6000 characters of the seed document before the simulation begins, replacing n-gram keyword topics/entities with typed entities and relations in the world profile, agent memory, and knowledge graph. This costs exactly one extra cheap-tier call per live simulation. Dry-run is unaffected — the deterministic n-gram profile is used when `LEANSWARM_DRY_RUN=true`.

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

### Web UI

```bash
pip install leanswarm
leanswarm ui
```
Open `http://127.0.0.1:8000` in your browser. The UI supports both mock and live modes, and features a dark/light theme, a Composer for running simulations, and a tabbed Gallery for exploring public and personal simulation histories. You can bring your own API keys, which are stored exclusively in your browser and used securely in memory per run.

**Hosting configuration:**
To run behind a TLS proxy, set `LEANSWARM_UI_SECURE_COOKIES=true`. To restrict registration, set `LEANSWARM_UI_ALLOW_SIGNUP=false`.

| Environment Variable | Default | Meaning |
|---|---|---|
| `LEANSWARM_UI_DATA_DIR` | `.leanswarm/ui` | Holds SQLite DB and per-run logs |
| `LEANSWARM_UI_ALLOW_SIGNUP` | `true` | Set to false to close registration |
| `LEANSWARM_UI_SECURE_COOKIES` | `false` | Set to true when hosted behind HTTPS |
| `LEANSWARM_UI_MAX_ROUNDS` | `12` | Max allowed rounds |
| `LEANSWARM_UI_MAX_AGENTS` | `48` | Max allowed agents |
| `LEANSWARM_UI_MAX_SEED_CHARS` | `20000` | Max chars for seed document |
| `LEANSWARM_UI_MAX_CONCURRENT_RUNS` | `2` | Concurrent simulation runs |
| `LEANSWARM_UI_RUNS_PER_HOUR_PER_IP` | `10` | 0 disables rate limiting |
| `LEANSWARM_UI_RETENTION_SECONDS` | `7200` | Delay before purging ephemeral jobs |

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
