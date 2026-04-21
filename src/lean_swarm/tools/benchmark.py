from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Any

from lean_swarm.engine.config import RuntimeSettings
from lean_swarm.engine.models import ActivationMode, ModelTier, SimulationRequest, SimulationResult
from lean_swarm.engine.simulator import LeanSwarmEngine


@dataclass(frozen=True)
class BenchmarkCase:
    seed_document: str
    question: str
    expected_polarity: int
    anchor_terms: tuple[str, ...]
    random_seed: int
    active_agent_fraction: float | None = None


DEFAULT_ACTIVE_AGENT_FRACTION = 0.15
BENCHMARK_MODES = (ActivationMode.LEAN.value, ActivationMode.NAIVE.value)
TOKEN_PRICE_PER_1K_USD = {
    ModelTier.FLAGSHIP.value: 0.01,
    ModelTier.STANDARD.value: 0.003,
    ModelTier.CHEAP.value: 0.0005,
}


CASES = [
    BenchmarkCase(
        seed_document=(
            "A weekly polling digest shows mild optimism after a steady stream "
            "of clarifying updates."
        ),
        question="Will public sentiment improve next week?",
        expected_polarity=1,
        anchor_terms=("optimism", "clarifying", "updates", "sentiment"),
        random_seed=11,
    ),
    BenchmarkCase(
        seed_document=(
            "Analysts report a fractured response after a sharp negative headline, "
            "but no systemic shift."
        ),
        question="Will approval decline by the next cycle?",
        expected_polarity=-1,
        anchor_terms=("fractured", "headline", "response", "decline"),
        random_seed=17,
    ),
    BenchmarkCase(
        seed_document=(
            "Support weakens after repeated delays and a rumor cycle spreads "
            "faster than corrections."
        ),
        question="Will confidence hold through the month?",
        expected_polarity=-1,
        anchor_terms=("delays", "rumor", "cycle", "confidence"),
        random_seed=23,
    ),
]


async def run_benchmark() -> dict[str, object]:
    mode_metrics: dict[str, dict[str, object]] = {}
    plot_points: list[dict[str, object]] = []

    for mode in BENCHMARK_MODES:
        metrics = await _run_mode_benchmark(mode)
        mode_metrics[mode] = metrics
        plot_points.extend([dict(entry) for entry in _as_list_of_dicts(metrics["case_scores"])])

    lean_metrics = mode_metrics[ActivationMode.LEAN.value]
    naive_metrics = mode_metrics[ActivationMode.NAIVE.value]
    lean_cost = _as_float(lean_metrics, "cost_usd_total")
    naive_cost = _as_float(naive_metrics, "cost_usd_total")
    lean_quality = _as_float(lean_metrics, "quality_proxy")
    naive_quality = _as_float(naive_metrics, "quality_proxy")
    lean_runtime = _as_float(lean_metrics, "runtime_seconds")
    naive_runtime = _as_float(naive_metrics, "runtime_seconds")

    return {
        "cases": len(CASES),
        "modes": mode_metrics,
        "plot_points": plot_points,
        "cost_ratio_naive_to_lean": _safe_ratio(naive_cost, lean_cost),
        "quality_delta_lean_vs_naive": round(lean_quality - naive_quality, 3),
        "runtime_ratio_naive_to_lean": _safe_ratio(naive_runtime, lean_runtime),
    }


async def _run_mode_benchmark(mode: str) -> dict[str, object]:
    settings = RuntimeSettings.from_env()
    mode_cache_dir = settings.cache_dir / "benchmark" / mode
    shutil.rmtree(mode_cache_dir, ignore_errors=True)
    settings.cache_dir = mode_cache_dir
    engine = LeanSwarmEngine(settings=settings)
    cold_scores: list[dict[str, object]] = []
    cold_runtime_seconds = 0.0
    cold_start_snapshot = _router_snapshot(engine)

    for index, case in enumerate(CASES, start=1):
        request = _request_for_case(case, mode)
        before = _router_snapshot(engine)
        started = perf_counter()
        result = await engine.simulate(request)
        elapsed = perf_counter() - started
        after = _router_snapshot(engine)
        route_calls_delta = _counter_delta(before, after, "route_calls")
        cache_hits_delta = _counter_delta(before, after, "cache_hits")
        tokens_by_tier = _token_delta_by_tier(before, after)
        token_total = sum(tokens_by_tier.values())
        cost_by_tier = _estimate_cost_usd_by_tier(tokens_by_tier)
        cost_total = round(sum(cost_by_tier.values()), 6)
        runtime_seconds = _runtime_seconds(
            elapsed=elapsed,
            route_calls=route_calls_delta,
            cache_hits=cache_hits_delta,
            tick_count=result.report.tick_count,
            dry_run=engine.settings.dry_run,
        )
        cold_runtime_seconds += runtime_seconds
        score = _score_case(case, result)
        cold_scores.append(
            {
                **score,
                "mode": mode,
                "case_id": _case_id(index),
                "token_usage_by_tier": tokens_by_tier,
                "token_total": token_total,
                "estimated_cost_usd_by_tier": cost_by_tier,
                "cost_usd": cost_total,
                "runtime_seconds": runtime_seconds,
                "cache_hit_rate": round(cache_hits_delta / max(1, route_calls_delta), 3),
            }
        )

    cold_route_calls = _counter_delta(cold_start_snapshot, _router_snapshot(engine), "route_calls")
    cold_cache_hits = _counter_delta(cold_start_snapshot, _router_snapshot(engine), "cache_hits")

    warm_start_snapshot = _router_snapshot(engine)
    warm_runtime_seconds = 0.0
    for case in CASES:
        request = _request_for_case(case, mode)
        before = _router_snapshot(engine)
        started = perf_counter()
        result = await engine.simulate(request)
        elapsed = perf_counter() - started
        after = _router_snapshot(engine)
        warm_runtime_seconds += _runtime_seconds(
            elapsed=elapsed,
            route_calls=_counter_delta(before, after, "route_calls"),
            cache_hits=_counter_delta(before, after, "cache_hits"),
            tick_count=result.report.tick_count,
            dry_run=engine.settings.dry_run,
        )

    end_snapshot = _router_snapshot(engine)
    warm_route_calls = _counter_delta(warm_start_snapshot, end_snapshot, "route_calls")
    warm_cache_hits = _counter_delta(warm_start_snapshot, end_snapshot, "cache_hits")

    quality_scores = [_dict_float(item, "score") for item in cold_scores]
    grounding_scores = [_dict_float(item, "grounding") for item in cold_scores]
    direction_scores = [_dict_float(item, "direction") for item in cold_scores]
    structure_scores = [_dict_float(item, "structure") for item in cold_scores]
    diversity_scores = [_dict_float(item, "diversity") for item in cold_scores]
    prediction_fingerprints = {str(item["prediction_fingerprint"]) for item in cold_scores}
    token_usage_by_tier = dict(engine.router.total_tokens_by_tier)
    estimated_cost_by_tier = _estimate_cost_usd_by_tier(token_usage_by_tier)

    return {
        "mode": mode,
        "cases": len(CASES),
        "activation_mode": mode,
        "active_agent_fraction": DEFAULT_ACTIVE_AGENT_FRACTION
        if mode == ActivationMode.LEAN.value
        else 1.0,
        "quality_proxy": round(mean(quality_scores), 3),
        "grounding_score": round(mean(grounding_scores), 3),
        "direction_score": round(mean(direction_scores), 3),
        "structure_score": round(mean(structure_scores), 3),
        "diversity_score": round(mean(diversity_scores), 3),
        "variation_score": round(len(prediction_fingerprints) / len(CASES), 3),
        "token_usage_by_tier": token_usage_by_tier,
        "token_total": sum(token_usage_by_tier.values()),
        "estimated_cost_usd_by_tier": estimated_cost_by_tier,
        "cost_usd_total": round(sum(estimated_cost_by_tier.values()), 6),
        "runtime_seconds": round(cold_runtime_seconds, 4),
        "warm_runtime_seconds": round(warm_runtime_seconds, 4),
        "runtime_seconds_per_case": {
            str(item["case_id"]): _dict_float(item, "runtime_seconds") for item in cold_scores
        },
        "cold_route_calls": cold_route_calls,
        "cold_cache_hits": cold_cache_hits,
        "warm_route_calls": warm_route_calls,
        "warm_cache_hits": warm_cache_hits,
        "warm_cache_hit_rate": round(warm_cache_hits / max(1, warm_route_calls), 3),
        "route_calls": engine.router.route_calls,
        "cache_hits": engine.router.cache_hits,
        "cache_hit_rate": round(engine.router.cache_hit_rate, 3),
        "case_scores": cold_scores,
    }


def _request_for_case(case: BenchmarkCase, mode: str) -> SimulationRequest:
    active_agent_fraction = case.active_agent_fraction or DEFAULT_ACTIVE_AGENT_FRACTION
    group_size = 5
    if mode == ActivationMode.NAIVE.value:
        active_agent_fraction = 1.0
        group_size = 1
    return SimulationRequest(
        seed_document=case.seed_document,
        question=case.question,
        rounds=5,
        max_agents=42,
        activation_mode=ActivationMode(mode),
        active_agent_fraction=active_agent_fraction,
        group_size=group_size,
        random_seed=case.random_seed,
    )


def _router_snapshot(engine: LeanSwarmEngine) -> dict[str, Any]:
    return {
        "route_calls": engine.router.route_calls,
        "cache_hits": engine.router.cache_hits,
        "token_usage_by_tier": dict(engine.router.total_tokens_by_tier),
    }


def _counter_delta(before: dict[str, Any], after: dict[str, Any], key: str) -> int:
    return int(after[key]) - int(before[key])


def _token_delta_by_tier(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    before_tokens = _as_int_by_tier(before["token_usage_by_tier"])
    after_tokens = _as_int_by_tier(after["token_usage_by_tier"])
    return {tier.value: after_tokens[tier.value] - before_tokens[tier.value] for tier in ModelTier}


def _estimate_cost_usd_by_tier(token_usage_by_tier: dict[str, int]) -> dict[str, float]:
    estimated: dict[str, float] = {}
    for tier in ModelTier:
        token_count = token_usage_by_tier.get(tier.value, 0)
        price = TOKEN_PRICE_PER_1K_USD[tier.value]
        estimated[tier.value] = round((token_count / 1000.0) * price, 6)
    return estimated


def _runtime_seconds(
    *,
    elapsed: float,
    route_calls: int,
    cache_hits: int,
    tick_count: int,
    dry_run: bool,
) -> float:
    if dry_run:
        uncached_calls = max(0, route_calls - cache_hits)
        return round((uncached_calls * 0.02) + (cache_hits * 0.002) + (tick_count * 0.004), 4)
    return round(elapsed, 4)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 3)


def _as_list_of_dicts(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _as_int_by_tier(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {tier.value: 0 for tier in ModelTier}
    parsed: dict[str, int] = {tier.value: 0 for tier in ModelTier}
    for tier in ModelTier:
        parsed[tier.value] = int(value.get(tier.value, 0))
    return parsed


def _as_float(payload: dict[str, object], key: str) -> float:
    return _to_float(payload.get(key, 0.0))


def _dict_float(payload: dict[str, object], key: str) -> float:
    return _to_float(payload.get(key, 0.0))


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _case_id(index: int) -> str:
    return f"case-{index}"


def _score_case(case: BenchmarkCase, result: SimulationResult) -> dict[str, object]:
    report = result.report
    combined = " ".join([report.prediction, *report.rationale, *report.key_events]).lower()
    tokens = _tokens(combined)
    anchor_hits = sum(1 for term in case.anchor_terms if term in combined)
    grounding = anchor_hits / max(1, len(case.anchor_terms))
    direction = _direction_score(report.prediction, report.rationale, case.expected_polarity)
    structure = _structure_score(report.tick_count, len(report.rationale), len(report.key_events))
    diversity = _diversity_score(tokens)
    score = round(
        (direction * 0.35) + (grounding * 0.3) + (structure * 0.2) + (diversity * 0.15),
        3,
    )
    return {
        "question": case.question,
        "score": score,
        "grounding": round(grounding, 3),
        "direction": round(direction, 3),
        "structure": round(structure, 3),
        "diversity": round(diversity, 3),
        "prediction_fingerprint": _fingerprint(
            report.prediction,
            report.rationale,
            report.key_events,
        ),
        "cache_hit_rate": round(report.cache_hit_rate, 3),
        "tick_count": report.tick_count,
        "converged": report.converged,
    }


def _direction_score(prediction: str, rationale: list[str], expected_polarity: int) -> float:
    polarity = _polarity_score(" ".join([prediction, *rationale]))
    if expected_polarity == 0:
        return 1.0 if polarity == 0 else 0.5
    if polarity == 0:
        return 0.6
    if polarity > 0 and expected_polarity > 0:
        return 1.0
    if polarity < 0 and expected_polarity < 0:
        return 1.0
    return 0.0


def _structure_score(tick_count: int, rationale_count: int, key_event_count: int) -> float:
    return min(
        1.0,
        (min(tick_count, 5) / 5) * 0.4
        + (min(rationale_count, 3) / 3) * 0.35
        + (min(key_event_count, 5) / 5) * 0.25,
    )


def _diversity_score(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    return min(1.0, unique_ratio)


def _tokens(text: str) -> list[str]:
    stopwords = {
        "a",
        "about",
        "after",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "into",
        "is",
        "it",
        "next",
        "of",
        "on",
        "or",
        "over",
        "the",
        "their",
        "this",
        "through",
        "to",
        "was",
        "will",
        "with",
        "without",
    }
    return [
        token.lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9'\-]*", text)
        if len(token) >= 4 and token.lower() not in stopwords
    ]


def _polarity_score(text: str) -> int:
    positive = {
        "improve",
        "improves",
        "improved",
        "recover",
        "rebound",
        "gain",
        "gains",
        "rise",
        "rises",
        "growth",
        "strong",
        "stronger",
        "optimism",
        "positive",
        "support",
        "stabilize",
        "stability",
        "clear",
        "clarifying",
        "lift",
        "lifted",
        "hold",
    }
    negative = {
        "decline",
        "declines",
        "declined",
        "fall",
        "falls",
        "drop",
        "drops",
        "worsen",
        "weak",
        "weaken",
        "fractured",
        "negative",
        "risk",
        "pressure",
        "backlash",
        "uncertain",
        "uncertainty",
        "volatile",
        "noise",
        "shock",
        "undercut",
        "frustration",
        "frustrated",
        "stalled",
        "stall",
        "slowing",
        "delay",
        "delays",
        "rumor",
    }
    score = 0
    for token in _tokens(text):
        if token in positive:
            score += 1
        elif token in negative:
            score -= 1
    return score


def _fingerprint(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
