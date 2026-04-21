# Architecture

## Target shape

Lean Swarm is organized around a narrow execution core:

- `lean_swarm.engine.llm`: the only module allowed to issue model calls
- `lean_swarm.engine.simulator`: world state updates, tick execution, convergence detection
- `lean_swarm.engine.memory`: working, episodic, and semantic memory helpers
- `lean_swarm.api.app`: HTTP surface area
- `lean_swarm.cli`: command-line entry points

## Mandatory constraints

- Tiered model routing goes through `llm.route(task_type, payload)`.
- Agent actions are requested in batches rather than one call per agent.
- Named agents are instantiated from a larger archetype pool.
- Numeric state is updated locally; LLM work is invoked only for grouped synthesis steps.
- Only a subset of agents is active per tick.
- Memory is layered into working, episodic, and semantic slices.
- Disk caching is used to reduce repeated action generation.
- The simulation can stop early when the world stabilizes.

## Phase plan assumption

This repository started empty, so the original phase list was not recoverable from local context.
Phase 1 therefore focuses on a repo-safe foundation:

- packaging and project metadata
- CLI and API skeletons
- baseline simulation engine and router contracts
- logging, caching, and tests

