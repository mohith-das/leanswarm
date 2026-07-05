# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and the project follows Semantic Versioning.

## [Unreleased]

## [0.2.0] - 2026-07-05

### Added
- Provider-agnostic live routing: credentials are resolved per model via LiteLLM's
  environment validation with a static fallback map covering DeepSeek, MiniMax, Zhipu GLM,
  Gemini, Groq, Mistral, xAI, OpenRouter, and Ollama.
- `LEANSWARM_API_BASE` / `LEANSWARM_API_KEY` overrides for any OpenAI-compatible endpoint.
- Schema-aware live prompts per task type with robust JSON extraction, Pydantic validation,
  a single automatic repair attempt, and an explicit logged `mock_fallback` as last resort.
- `leanswarm doctor [--ping]` subcommand for verifying live-mode configuration and
  connectivity, plus `--live` / `--dry-run` flags on `smoke` and `simulate`.
- Live token accounting from `completion.usage` with `token_source` recorded in call logs.

### Changed
- Live mode now fails loudly with `LiveCredentialsError` when credentials are missing
  instead of silently returning mock responses.
- Cache keys include the live/mock routing decision and a version marker so mock responses
  can never be replayed into live runs.
- LLM call logs record `mode` (`mock`, `live`, `cached`, `mock_fallback`) for every route.

### Fixed
- Removed the tracked Python 3.9 `venv/` from version control and tightened `.gitignore`.

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
