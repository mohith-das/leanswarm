import asyncio
import json
import sqlite3
import time
import pytest
from fastapi.testclient import TestClient

from leanswarm.webui.app import create_webui_app

@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("LEANSWARM_UI_DATA_DIR", str(tmp_path))
    app = create_webui_app()
    with TestClient(app) as c:
        yield c

def test_auth_flow(client):
    res = client.post("/api/auth/register", json={"email": "test@example.com", "password": "password123"})
    assert res.status_code == 200
    assert res.json() == {"email": "test@example.com"}

    res = client.post("/api/auth/register", json={"email": "test@example.com", "password": "password123"})
    assert res.status_code == 409

    res = client.post("/api/auth/register", json={"email": "test2@example.com", "password": "short"})
    assert res.status_code == 422

    res = client.get("/api/auth/me")
    assert res.json() == {"email": "test@example.com"}

    res = client.post("/api/auth/logout")
    assert res.status_code == 200

    res = client.get("/api/auth/me")
    assert res.json() == {"email": None}

    res = client.post("/api/auth/login", json={"email": "test@example.com", "password": "wrong"})
    assert res.status_code == 401

    res = client.post("/api/auth/login", json={"email": "test@example.com", "password": "password123"})
    assert res.status_code == 200
    assert res.json() == {"email": "test@example.com"}

    res = client.get("/api/auth/me")
    assert res.json() == {"email": "test@example.com"}

def wait_for_run(client, run_id):
    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                evt = json.loads(line[6:])
                if evt.get("type") in ("complete", "error"):
                    return

def test_run_lifecycle_mock(client):
    req = {
        "seed_document": "seed",
        "question": "question?",
        "rounds": 1,
        "max_agents": 2,
        "group_size": 2,
        "active_agent_fraction": 1.0,
        "convergence_threshold": 1,
        "random_seed": 7,
        "live": False,
        "models": {
            "flagship": "gpt-4.1",
            "standard": "gpt-4.1",
            "cheap": "gpt-4.1"
        }
    }
    res = client.post("/api/runs", json=req)
    assert res.status_code == 200
    run_id = res.json()["id"]

    events = []
    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                evt = json.loads(line[6:])
                events.append(evt)
                if evt.get("type") in ("complete", "error"):
                    break
        
    has_bootstrap = any(e.get("phase") == "bootstrap" for e in events)
    has_tick = any(e.get("type") == "tick" for e in events)
    has_complete = any(e.get("type") == "complete" for e in events)
    assert has_bootstrap
    assert has_tick
    assert has_complete
    
    data = client.get(f"/api/runs/{run_id}").json()
    assert data["status"] == "complete"
    assert "prediction" in data["result"]["report"]

def test_ephemeral_vs_saved(client):
    req = {
        "seed_document": "seed",
        "question": "q",
        "live": False,
        "models": {"flagship": "x", "standard": "y", "cheap": "z"}
    }
    run_id = client.post("/api/runs", json=req).json()["id"]
    wait_for_run(client, run_id)

    res = client.post(f"/api/runs/{run_id}/save", json={})
    assert res.status_code == 401

    client.post("/api/auth/register", json={"email": "save@example.com", "password": "password123"})
    run_id = client.post("/api/runs", json=req).json()["id"]
    wait_for_run(client, run_id)

    res = client.post(f"/api/runs/{run_id}/save", json={})
    assert res.status_code == 200

    runs = client.get("/api/runs").json()
    assert any(r["id"] == run_id for r in runs)

    res = client.delete(f"/api/runs/{run_id}")
    assert res.status_code == 200
    assert not any(r["id"] == run_id for r in client.get("/api/runs").json())

def test_publish_and_gallery(client):
    req = {
        "seed_document": "seed",
        "question": "q",
        "live": False,
        "models": {"flagship": "x", "standard": "y", "cheap": "z"}
    }
    run_id = client.post("/api/runs", json=req).json()["id"]
    wait_for_run(client, run_id)

    res = client.post(f"/api/runs/{run_id}/publish", json={})
    assert res.status_code == 200

    gallery = client.get("/api/gallery").json()
    assert any(g["id"] == run_id for g in gallery)

    res = client.get(f"/api/gallery/{run_id}")
    assert res.status_code == 200

def test_credentials_never_persisted(client, tmp_path):
    req = {
        "seed_document": "seed",
        "question": "q",
        "live": False,
        "models": {"flagship": "x", "standard": "y", "cheap": "z"},
        "credentials": {"DEEPSEEK_API_KEY": "sk-SECRET-MARKER"}
    }
    run_id = client.post("/api/runs", json=req).json()["id"]
    wait_for_run(client, run_id)

    client.post(f"/api/runs/{run_id}/publish", json={})

    import os
    marker = b"sk-SECRET-MARKER"
    for root, _, files in os.walk(str(tmp_path)):
        for f in files:
            path = os.path.join(root, f)
            with open(path, "rb") as fd:
                content = fd.read()
                assert marker not in content, f"Secret found in {path}"

def test_limits(client):
    req = {
        "seed_document": "seed",
        "question": "q",
        "rounds": 100,
        "live": False,
        "models": {"flagship": "x", "standard": "y", "cheap": "z"}
    }
    res = client.post("/api/runs", json=req)
    assert res.status_code == 422

    req["rounds"] = 4
    req["seed_document"] = "a" * 30000
    res = client.post("/api/runs", json=req)
    assert res.status_code == 422
    req = {
        "seed_document": "seed",
        "question": "q",
        "rounds": 100,
        "live": False,
        "models": {"flagship": "x", "standard": "y", "cheap": "z"}
    }
    res = client.post("/api/runs", json=req)
    assert res.status_code == 422

    req["rounds"] = 4
    req["seed_document"] = "a" * 30000
    res = client.post("/api/runs", json=req)
    assert res.status_code == 422

def test_get_run_privacy(client):
    req = {
        "seed_document": "seed",
        "question": "q",
        "live": False,
        "models": {"flagship": "x", "standard": "y", "cheap": "z"}
    }
    
    # User A registers and saves a run privately
    client.post("/api/auth/register", json={"email": "usera@example.com", "password": "password123"})
    run_id = client.post("/api/runs", json=req).json()["id"]
    wait_for_run(client, run_id)
    res = client.post(f"/api/runs/{run_id}/save", json={})
    assert res.status_code == 200

    # User A can get the run
    res = client.get(f"/api/runs/{run_id}")
    assert res.status_code == 200

    # Logout and try to get run anonymously -> 404
    client.post("/api/auth/logout")
    res = client.get(f"/api/runs/{run_id}")
    assert res.status_code == 404

    # User B registers and tries to get User A's private run -> 404
    client.post("/api/auth/register", json={"email": "userb@example.com", "password": "password123"})
    res = client.get(f"/api/runs/{run_id}")
    assert res.status_code == 404

    # User A logs back in, publishes it
    client.post("/api/auth/logout")
    client.post("/api/auth/login", json={"email": "usera@example.com", "password": "password123"})
    res = client.post(f"/api/runs/{run_id}/publish", json={})
    assert res.status_code == 200

    # Now anonymous can get it
    client.post("/api/auth/logout")
    res = client.get(f"/api/runs/{run_id}")
    assert res.status_code == 200
