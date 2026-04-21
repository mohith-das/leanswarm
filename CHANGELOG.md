# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and the project follows Semantic Versioning.

## [Unreleased]

### Added
- Activation mode support (`lean` and `naive`) and configurable convergence threshold in simulation requests.
- Deterministic semantic memory retrieval and semantic-context wiring for memory summarization.
- Lean-vs-naive benchmark mode comparison with runtime, token, estimated-cost, cache, and plotting outputs.
- Benchmark integration tests and CLI simulation control parsing tests.
- GitHub issue templates for bug reports, feature requests, benchmark datasets, and local model optimizations.
- A pull request template aligned to LeanSwarm validation and benchmark reporting.
- A V2 roadmap document covering multi-world simulation, Monte Carlo execution, intervention testing, replay UX, plugin architecture, large-scale agent targets, and hybrid local+cloud model routing.

### Changed
- Event-driven activation now includes deterministic trigger boosts from volatility and world signals.
- `lean-swarm simulate` now exposes runtime controls (activation mode, active fraction, max agents, group size, convergence threshold, random seed, and LLM toggle flags).
- Expanded README benchmark documentation with lean-vs-naive outputs and quality-vs-cost plotting points.
- Updated contributor guidance with focused invite tracks for performance, research, benchmark, and local-model optimization contributions.

## [0.1.0] - 2026-04-21

### Added
- Phase 1 repository scaffold with `src/` layout and MIT licensing.
- Baseline CLI, FastAPI app, deterministic simulation engine skeleton, and LLM router guardrails.
- Initial packaging, CI, release workflow, examples, docs, and tests.
