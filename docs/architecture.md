# Architecture

## Current shape

Lean Swarm is organized around a narrow execution core:

- `leanswarm.engine.llm`: the only module allowed to issue model calls
- `leanswarm.engine.simulator`: world state updates, tick execution, convergence detection
- `leanswarm.engine.memory`: working, episodic, and semantic memory helpers
- `leanswarm.engine.semantic_store`: SQLite-backed semantic retrieval with optional `sqlite-vss`
- `leanswarm.api.app`: HTTP surface area
- `leanswarm.cli`: command-line entry points
- `web/`: minimal Next.js viewer for pasting simulation JSON and inspecting the resulting world

## Mandatory constraints

- Tiered model routing goes through `llm.route(task_type, payload)`.
- Agent actions are requested in batches rather than one call per agent.
- Named agents are instantiated from a larger archetype pool.
- Numeric state is updated locally; LLM work is invoked only for grouped synthesis steps.
- Only a subset of agents is active per tick.
- Memory is layered into working, episodic, and semantic slices.
- Disk caching is used to reduce repeated action generation.
- The simulation can stop early when the world stabilizes.

## Current status

The repository now has a usable Phase 2 runtime and a minimal web inspection surface:

- packaging and project metadata
- CLI and API skeletons
- baseline simulation engine and router contracts
- logging, caching, and tests
- seed-aware world ingestion and population shaping
- a JSON inspector for the post-simulation world snapshot

The remaining gaps are a broader public benchmark suite, richer convergence heuristics, and a more
fully featured replay UI.
