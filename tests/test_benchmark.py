from __future__ import annotations

import asyncio

from lean_swarm.tools.benchmark import run_benchmark


def _run_dry_benchmark(tmp_path, monkeypatch) -> dict[str, object]:
    monkeypatch.setenv("LEANSWARM_DRY_RUN", "true")
    monkeypatch.setenv("LEANSWARM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LEANSWARM_LOG_DIR", str(tmp_path / "logs"))
    return asyncio.run(run_benchmark())


def test_benchmark_outputs_mode_metrics(tmp_path, monkeypatch) -> None:
    payload = _run_dry_benchmark(tmp_path, monkeypatch)

    required_top_level = {
        "cases",
        "modes",
        "plot_points",
        "cost_ratio_naive_to_lean",
        "quality_delta_lean_vs_naive",
        "runtime_ratio_naive_to_lean",
    }
    assert required_top_level.issubset(payload.keys())

    modes = payload["modes"]
    assert isinstance(modes, dict)
    assert {"lean", "naive"}.issubset(modes.keys())

    required_mode_keys = {
        "token_usage_by_tier",
        "token_total",
        "estimated_cost_usd_by_tier",
        "cost_usd_total",
        "runtime_seconds",
        "runtime_seconds_per_case",
        "cache_hit_rate",
        "quality_proxy",
        "cold_route_calls",
        "warm_route_calls",
        "warm_cache_hits",
        "warm_cache_hit_rate",
        "case_scores",
    }
    for mode in ("lean", "naive"):
        mode_payload = modes[mode]
        assert isinstance(mode_payload, dict)
        assert required_mode_keys.issubset(mode_payload.keys())
        assert mode_payload["activation_mode"] == mode


def test_benchmark_outputs_plot_points_and_case_fields(tmp_path, monkeypatch) -> None:
    payload = _run_dry_benchmark(tmp_path, monkeypatch)

    assert float(payload["cost_ratio_naive_to_lean"]) >= 1.0

    plot_points = payload["plot_points"]
    assert isinstance(plot_points, list)
    assert len(plot_points) >= int(payload["cases"]) * 2

    required_case_fields = {
        "mode",
        "case_id",
        "score",
        "cost_usd",
        "runtime_seconds",
        "token_total",
        "cache_hit_rate",
        "converged",
        "tick_count",
    }
    for point in plot_points:
        assert isinstance(point, dict)
        assert required_case_fields.issubset(point.keys())

    modes = payload["modes"]
    assert isinstance(modes, dict)
    for mode in ("lean", "naive"):
        case_scores = modes[mode]["case_scores"]
        assert isinstance(case_scores, list)
        for case_entry in case_scores:
            assert isinstance(case_entry, dict)
            assert required_case_fields.issubset(case_entry.keys())
