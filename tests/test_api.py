from __future__ import annotations

from fastapi.testclient import TestClient

from lean_swarm.api.app import create_app


def test_simulate_endpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LEANSWARM_DRY_RUN", "true")
    monkeypatch.setenv("LEANSWARM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LEANSWARM_LOG_DIR", str(tmp_path / "logs"))

    client = TestClient(create_app())
    response = client.post(
        "/simulate",
        json={
            "seed_document": "Survey coverage is mixed after a policy rollout.",
            "question": "Will sentiment improve by next week?",
            "rounds": 3,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["report"]["question"] == "Will sentiment improve by next week?"
    assert len(payload["world"]["agents"]) > 0

