# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and the project follows Semantic Versioning.

## [Unreleased]

## [0.4.2] - 2026-07-06

### Added
- "Open source — Audit on GitHub" badge in the web UI sidebar footer linking to the
  repository, so hosted visitors can verify the code they're trusting with their keys.

## [0.4.1] - 2026-07-06

### Fixed
- "Generate full report" always failed with a 422: the report endpoint reused the
  chat request schema, whose `message` field was required, but the UI sends no
  message for reports. The field is now optional (and still required for `/chat`).
- Full reports now inherit the run's own mode and models — a live run gets a live
  report instead of a silently mocked one.
- Chat/report rate limiting was a no-op due to a type-confused bucket; it now uses
  a dedicated per-IP bucket and actually enforces `LEANSWARM_UI_CHATS_PER_HOUR_PER_IP`.
- Restored mypy strict compliance in `webui/app.py` (CI was red on 0.4.0).
- `leanswarm.__version__` was stale (0.3.1) in the 0.4.0 wheel; now synced.

### Added
- Regression and safety tests for the 0.4.0 feature set: SSRF guard, HTML text
  extraction, source-gathering resilience, chat and report endpoints, persona
  gating, and calibration math.

## [0.4.0] - 2026-07-06

### Added
- **Retrieval**: simulations can now be grounded in fetched web sources. The CLI
  `simulate` subcommand gains `--url`, `--search`, and `--max-sources` flags; the
  Web UI Composer adds a Sources card with URL inputs and a Tavily/Brave-powered
  "search the web" option. All network access goes through an SSRF guard that
  refuses private/reserved addresses. Retrieved sources are assembled into the
  seed corpus with `build_corpus()`.
- **Persona depth**: live mode now generates detailed agent personas (display
  names, bios, and stances grounded in the extracted world) with a single
  cheap-tier `PERSONA_BATCH` call. Personas and stances appear in agent cards
  and feed into action prompts for more consistent agent behavior.
- **Post-run chat**: talk to any simulated agent or the Report Agent after a run
  completes. Chat works in both mock mode (deterministic canned replies) and
  live mode (real LLM responses with per-message model selection). Available
  in the run result view and gallery detail.
- **Sectioned reports**: an on-demand `FULL_REPORT` call on the flagship tier
  generates a structured multi-section report (Executive Summary, World &
  Actors, Simulation Dynamics, Prediction & Confidence, Risks) with a
  downloadable Markdown file.
- **Calibration benchmarking**: `leanswarm bench --calibration <path>` runs
  a Brier score evaluation against resolved binary outcomes (0/1 JSONL), with
  an optional single-shot baseline arm, producing per-case probability
  comparisons and a verdict (sim_better / baseline_better / tie).
- **PDF upload**: the Composer's file loader now accepts PDF files. Text
  extraction uses `pdfjs-dist` loaded dynamically — it ships as a separate
  lazy chunk in the frontend build.
- **Re-run from gallery**: any saved or published run now has an "Open in
  composer" button that prefills seed and question for easy re-execution.

### Changed
- World extraction window widened from 6k to 12k characters to accommodate the
  larger seed corpus from retrieved sources.
- AGENT_BATCH payload now includes persona and stance fields (empty strings in
  dry-run mode — no output change).
- `PredictionReport` now carries a `direction` field.
- Cost estimates account for the persona-batch call.
- `httpx>=0.27` is now an explicit runtime dependency.

## [0.3.2] - 2026-07-06

### Changed
- Composer model selection is now provider-first: the user picks a provider from a
  dropdown (only showing providers whose API keys are set), then picks or types a
  model name with contextual suggestions. The Live tab is disabled when no API keys
  are stored.
- The "leanswarm" logo is now clickable and navigates to the composer.
- The "LLM-extracted world" badge in the result view is no longer rendered inside
  the graph toggle segmented control (it appeared as a non-functional third tab).
- Advanced settings grid uses a fixed 3-column layout for consistent alignment.

## [0.3.1] - 2026-07-06

### Added
- Live mode now runs a single cheap-tier LLM extraction pass over the seed document,
  replacing n-gram topics/entities with typed entities and relations in the world
  profile, agent memory, prompts, and knowledge graph; falls back to the
  deterministic profile if extraction fails. Cost estimates account for the extra call.

### Fixed
- The web UI's per-run/browser-provided API credentials were being ignored by the
  live-mode readiness check, which only consulted real environment variables and
  litellm's environment validation. This made bring-your-own-key live runs fail
  with `LiveCredentialsError` even when a valid key was supplied in the UI. 0.3.0
  is affected; upgrade if you deployed the web UI for live-mode use.
- `GET /api/runs/{id}` no longer leaks other users' private saved runs to anyone
  who knows the run id.
- Full mypy strict-mode compliance restored across `webui/*.py`.

## [0.3.0] - 2026-07-05

### Added
- Web UI: a full single-page application built with React/Vite to compose, execute, and save simulation runs in the browser.
- Web UI API endpoints under `/api` including authentication, runs management, cost estimation, doctor ping, and gallery browsing.
- SQLite-based persistence for user accounts, sessions, and saved/published simulation runs.
- Detailed progress callback for `LeanSwarmEngine.simulate()` to push realtime events over SSE.
- `leanswarm ui` subcommand to start the web backend, which serves both the API and the built static frontend assets.
- Live cost tracking and price estimations for supported foundation models.

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
