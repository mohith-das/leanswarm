import pytest
from leanswarm.engine.pricing import cost_usd, estimate_run, price_for

def test_price_for():
    assert price_for("gpt-4.1") == (2.00, 8.00)
    assert price_for("openrouter/deepseek/deepseek-chat") == (0.27, 1.10)
    assert price_for("unknown-model") is None

def test_cost_usd():
    assert cost_usd("deepseek/deepseek-chat", 1_000_000, 0) == pytest.approx(0.27)
    assert cost_usd("deepseek/deepseek-chat", 0, 1_000_000) == pytest.approx(1.10)
    assert cost_usd("unknown", 1, 1) is None

def test_estimate_run():
    est = estimate_run(
        rounds=2,
        max_agents=8,
        active_agent_fraction=0.2,
        group_size=4,
        flagship_model="deepseek/deepseek-reasoner",
        standard_model="deepseek/deepseek-chat",
        cheap_model="deepseek/deepseek-chat",
    )
    assert est["calls_min"] >= 4
    assert est["calls_max"] <= 7
    assert est["cost_min_usd"] is not None
    assert est["cost_max_usd"] is not None
    assert est["cost_min_usd"] > 0
    assert est["cost_max_usd"] > 0
    assert not est["unpriced_models"]

def test_estimate_unpriced():
    est = estimate_run(
        rounds=2,
        max_agents=8,
        active_agent_fraction=0.2,
        group_size=4,
        flagship_model="unknown1",
        standard_model="unknown2",
        cheap_model="unknown3",
    )
    assert est["cost_min_usd"] is None
    assert est["cost_max_usd"] is None
    assert "unknown1" in est["unpriced_models"]
