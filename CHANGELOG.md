# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and the project follows Semantic Versioning.

## [Unreleased]

## [0.1.1] - 2026-04-21

### Added
- Minimal Next.js web scaffold under `web/` for pasting simulation JSON and exploring the post-simulation world.
- Repo hygiene updates to ignore plain `venv/` and common Next.js build artifacts.
- SQLite-backed semantic memory store with deterministic offline fallback and optional `sqlite-vss` acceleration.

### Changed
- README and architecture docs now describe the current Phase 2 runtime instead of the initial Phase 1 baseline.
- README now documents the minimal web inspector and the current engine shape.
- Contributing guidance is simplified to match the files present in this repository.
- Lean activation now stays within the sparse event-driven envelope while convergence scoring blends state deltas, churn, and event novelty.
- Public package imports and CLI are now `leanswarm`, with a compatibility alias for legacy `lean_swarm` imports.
- PyPI-facing README installation now uses `pip install leanswarm` and separates optional Node/Next.js viewer setup.
- GitHub Actions now use `actions/checkout@v6` and `actions/setup-python@v6` for Node 24 readiness.

## [0.1.0] - 2026-04-21

### Added
- Phase 1 repository scaffold with `src/` layout and MIT licensing.
- Baseline CLI, FastAPI app, deterministic simulation engine skeleton, and LLM router guardrails.
- Initial packaging, CI, release workflow, examples, docs, and tests.
