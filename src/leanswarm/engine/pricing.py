"""Rough public list prices, USD per 1M tokens (input, output). Estimates only."""

import math
from typing import Any

from leanswarm.engine.enrichment import EXTRACTION_MAX_CHARS

PRICES: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "deepseek/deepseek-chat": (0.27, 1.10),
    "deepseek/deepseek-reasoner": (0.55, 2.19),
    "zhipuai/glm-4": (1.40, 1.40),
    "zhipuai/glm-4-air": (0.15, 0.15),
    "minimax/minimax-text-01": (0.20, 1.10),
    "gemini/gemini-2.5-flash": (0.30, 2.50),
    "gemini/gemini-2.5-pro": (1.25, 10.00),
    "groq/llama-3.3-70b-versatile": (0.59, 0.79),
    "mistral/mistral-small-latest": (0.10, 0.30),
    "anthropic/claude-sonnet-5": (3.00, 15.00),
    "anthropic/claude-haiku-4-5-20251001": (1.00, 5.00),
}


def price_for(model: str) -> tuple[float, float] | None:
    if model in PRICES:
        return PRICES[model]
    if model.startswith("openrouter/"):
        stripped = model[len("openrouter/"):]
        if stripped in PRICES:
            return PRICES[stripped]
    return None


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    prices = price_for(model)
    if prices is None:
        return None
    return (prompt_tokens / 1_000_000 * prices[0]) + (completion_tokens / 1_000_000 * prices[1])


def estimate_run(
    rounds: int,
    max_agents: int,
    active_agent_fraction: float,
    group_size: int,
    flagship_model: str,
    standard_model: str,
    cheap_model: str,
    seed_chars: int = 0,
) -> dict[str, Any]:
    """
    Estimates the LLM calls, tokens, and cost of a simulation run.
    Assumptions:
    - active = max(1, round(max_agents * active_agent_fraction))
    - batches_per_tick = ceil(active / group_size)
    - Calls: 1 world_extraction (cheap tier, LIVE ONLY — single pass over up to
      EXTRACTION_MAX_CHARS of the seed, no chunking) + 1 bootstrap (standard tier)
      + rounds * batches_per_tick agent-batch (cheap tier) + up to rounds
      memory-summary calls (standard tier) + 1 synthesis (flagship tier).
    - Early convergence can reduce tick count. min_ticks = min(2, rounds), max_ticks = rounds.
    - Tokens:
      extraction: 350 + min(seed_chars, 6000)//4 in / 450 out
      bootstrap: 450 in / 130 out
      agent batch: 550 + 60 * group_size in / 40 + 45 * group_size out
      memory summary: 400 in / 90 out
      synthesis: 500 + 180 * max_ticks in / 160 out
    """
    active = max(1, round(max_agents * active_agent_fraction))
    batches_per_tick = math.ceil(active / group_size)

    min_ticks = min(2, rounds)
    max_ticks = rounds

    capped_seed_chars = min(max(seed_chars, 0), EXTRACTION_MAX_CHARS)
    prompt_extraction = 350 + capped_seed_chars // 4
    comp_extraction = 450

    calls_min = 1 + 1 + (min_ticks * batches_per_tick) + 1
    calls_max = 1 + 1 + (max_ticks * batches_per_tick) + max_ticks + 1

    prompt_bootstrap, comp_bootstrap = 450, 130
    prompt_batch = 550 + 60 * group_size
    comp_batch = 40 + 45 * group_size
    prompt_mem, comp_mem = 400, 90
    prompt_synth, comp_synth = 500 + 180 * max_ticks, 160
    prompt_synth_min, comp_synth_min = 500 + 180 * min_ticks, 160

    # Max tokens
    prompt_tokens_est = (
        prompt_extraction +
        prompt_bootstrap +
        (max_ticks * batches_per_tick * prompt_batch) +
        (max_ticks * prompt_mem) +
        prompt_synth
    )
    completion_tokens_est = (
        comp_extraction +
        comp_bootstrap +
        (max_ticks * batches_per_tick * comp_batch) +
        (max_ticks * comp_mem) +
        comp_synth
    )


    cost_min_usd: float | None = 0.0
    cost_max_usd: float | None = 0.0
    unpriced_models = set()

    for m in [flagship_model, standard_model, cheap_model]:
        if price_for(m) is None:
            unpriced_models.add(m)

    if unpriced_models:
        cost_min_usd = None
        cost_max_usd = None
    else:
        # standard_model: bootstrap + memory summary
        c_boot = cost_usd(standard_model, prompt_bootstrap, comp_bootstrap) or 0.0
        c_mem_max = cost_usd(standard_model, prompt_mem, comp_mem) or 0.0

        # cheap_model: extraction + agent batch
        c_extraction = cost_usd(cheap_model, prompt_extraction, comp_extraction) or 0.0
        c_batch = cost_usd(cheap_model, prompt_batch, comp_batch) or 0.0

        # flagship_model: synthesis
        c_synth_max = cost_usd(flagship_model, prompt_synth, comp_synth) or 0.0
        c_synth_min = cost_usd(flagship_model, prompt_synth_min, comp_synth_min) or 0.0

        cost_min_usd = c_extraction + c_boot + (min_ticks * batches_per_tick * c_batch) + c_synth_min
        cost_max_usd = (
            c_extraction + c_boot
            + (max_ticks * batches_per_tick * c_batch)
            + (max_ticks * c_mem_max)
            + c_synth_max
        )

    return {
        "calls_min": calls_min,
        "calls_max": calls_max,
        "prompt_tokens_est": prompt_tokens_est,
        "completion_tokens_est": completion_tokens_est,
        "cost_min_usd": cost_min_usd,
        "cost_max_usd": cost_max_usd,
        "unpriced_models": list(unpriced_models),
    }
