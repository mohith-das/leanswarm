from __future__ import annotations

import json

from leanswarm.cli import main


def test_smoke_command(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LEANSWARM_DRY_RUN", "true")
    monkeypatch.setenv("LEANSWARM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LEANSWARM_LOG_DIR", str(tmp_path / "logs"))

    exit_code = main(["smoke"])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 0
    assert payload["report"]["tick_count"] >= 1


def test_simulate_command_parses_runtime_controls(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LEANSWARM_DRY_RUN", "true")
    monkeypatch.setenv("LEANSWARM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LEANSWARM_LOG_DIR", str(tmp_path / "logs"))

    seed_path = tmp_path / "seed.txt"
    seed_path.write_text("Sentiment signals are mixed after the policy launch.", encoding="utf-8")

    exit_code = main(
        [
            "simulate",
            "--seed",
            str(seed_path),
            "--question",
            "Will sentiment improve next month?",
            "--rounds",
            "4",
            "--max-agents",
            "18",
            "--active-agent-fraction",
            "0.35",
            "--activation-mode",
            "naive",
            "--group-size",
            "3",
            "--convergence-threshold",
            "4",
            "--random-seed",
            "123",
            "--no-use-llm",
        ]
    )
    output = capsys.readouterr().out
    payload = json.loads(output)
    request = payload["request"]

    assert exit_code == 0
    assert request["seed_document"] == "Sentiment signals are mixed after the policy launch."
    assert request["question"] == "Will sentiment improve next month?"
    assert request["rounds"] == 4
    assert request["max_agents"] == 18
    assert request["active_agent_fraction"] == 0.35
    assert request["activation_mode"] == "naive"
    assert request["group_size"] == 3
    assert request["convergence_threshold"] == 4
    assert request["random_seed"] == 123
    assert request["use_llm"] is False
