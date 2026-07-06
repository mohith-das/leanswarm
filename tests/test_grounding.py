import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from leanswarm.engine import retrieval
from leanswarm.engine.models import RetrievedSource
from leanswarm.engine.retrieval import (
    UnsafeURLError,
    assert_public_http_url,
    build_corpus,
    gather_sources,
    html_to_text,
)
from leanswarm.tools.benchmark import brier_score, probability_from_report
from leanswarm.webui.app import create_webui_app


def test_ssrf_guard(monkeypatch):
    for bad in (
        "ftp://example.com/a",
        "file:///etc/passwd",
        "http://localhost/x",
        "http://127.0.0.1:8000/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
    ):
        with pytest.raises(UnsafeURLError):
            assert_public_http_url(bad)

    def fake_getaddrinfo_private(host, port):
        return [(2, 1, 6, "", ("10.0.0.5", 0))]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo_private)
    with pytest.raises(UnsafeURLError):
        assert_public_http_url("http://internal.corp/secret")

    def fake_getaddrinfo_public(host, port):
        return [(2, 1, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo_public)
    assert_public_http_url("https://example.com/article")


def test_html_to_text():
    html = (
        "<html><head><title>T</title><style>.x{}</style></head><body>"
        "<nav>Menu Menu</nav><p>Real paragraph text.</p>"
        "<script>var hidden = 1;</script><div>More content"
    )
    text = html_to_text(html)
    assert "Real paragraph text." in text
    assert "More content" in text
    assert "hidden" not in text
    assert "Menu" not in text


def test_gather_sources_resilient(monkeypatch):
    async def fake_fetch(client, url):
        if "good" in url:
            return RetrievedSource(url=url, title="Good", chars=10, via="direct", text="x" * 5000)
        raise RuntimeError("boom")

    monkeypatch.setattr(retrieval, "fetch_url", fake_fetch)

    sources, errors = asyncio.run(
        gather_sources(
            "q?",
            ["https://good.example/a", "https://bad.example/b"],
            {},
            use_search=False,
        )
    )
    assert len(sources) == 1
    assert len(errors) == 1
    assert "bad.example" in errors[0]

    corpus = build_corpus("seed text", sources)
    assert corpus.startswith("seed text")
    assert "--- SOURCE:" in corpus


def test_search_requires_key():
    async def run():
        import httpx

        async with httpx.AsyncClient() as client:
            return await retrieval.search_web(client, "q", {}, 3)

    assert asyncio.run(run()) == []


def test_calibration_math():
    assert probability_from_report("positive", 0.8) == 0.8
    assert probability_from_report("negative", 0.8) == pytest.approx(0.2)
    assert probability_from_report("", 0.9) == 0.5
    assert probability_from_report("positive", 1.7) == 1.0
    assert brier_score([(1.0, 1), (0.0, 0)]) == 0.0
    assert brier_score([(0.0, 1)]) == 1.0
    assert brier_score([]) == 0.0


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("LEANSWARM_UI_DATA_DIR", str(tmp_path))
    app = create_webui_app()
    with TestClient(app) as c:
        yield c


def _complete_mock_run(client) -> str:
    res = client.post(
        "/api/runs",
        json={
            "seed_document": "A public debate over a new policy splits supporters and skeptics.",
            "question": "Will support grow?",
            "rounds": 1,
            "max_agents": 4,
            "group_size": 4,
            "live": False,
            "models": {"flagship": "x", "standard": "x", "cheap": "x"},
            "credentials": {},
        },
    )
    assert res.status_code == 200
    run_id = res.json()["id"]
    for _ in range(200):
        status = client.get(f"/api/runs/{run_id}").json()
        if status["status"] == "complete":
            return run_id
        time.sleep(0.05)
    raise AssertionError("mock run did not complete")


def test_report_endpoint_mock(client):
    """Regression: /report must work WITHOUT a message field (0.4.0 returned 422)."""
    run_id = _complete_mock_run(client)
    res = client.post(f"/api/runs/{run_id}/report", json={"live": False})
    assert res.status_code == 200
    body = res.json()
    assert len(body["sections"]) >= 3
    assert all(s["heading"] for s in body["sections"])


def test_chat_endpoint_mock(client):
    run_id = _complete_mock_run(client)
    status = client.get(f"/api/runs/{run_id}").json()
    agent_id = status["result"]["world"]["agents"][0]["id"]

    res = client.post(
        f"/api/runs/{run_id}/chat",
        json={"agent_id": agent_id, "message": "What do you think?", "live": False},
    )
    assert res.status_code == 200
    first_reply = res.json()["reply"]
    assert first_reply

    # Deterministic in mock mode.
    res = client.post(
        f"/api/runs/{run_id}/chat",
        json={"agent_id": agent_id, "message": "What do you think?", "live": False},
    )
    assert res.json()["reply"] == first_reply

    # Report agent (no agent_id).
    res = client.post(
        f"/api/runs/{run_id}/chat", json={"message": "Summarize.", "live": False}
    )
    assert res.status_code == 200
    assert res.json()["reply"]

    # Empty message rejected; unknown agent 404; oversized run of urls rejected.
    res = client.post(f"/api/runs/{run_id}/chat", json={"message": "   ", "live": False})
    assert res.status_code == 422
    res = client.post(
        f"/api/runs/{run_id}/chat",
        json={"agent_id": "agent-does-not-exist", "message": "hi", "live": False},
    )
    assert res.status_code == 404


def test_chat_rate_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("LEANSWARM_UI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LEANSWARM_UI_CHATS_PER_HOUR_PER_IP", "2")
    app = create_webui_app()
    with TestClient(app) as c:
        run_id = _complete_mock_run(c)
        for _ in range(2):
            res = c.post(f"/api/runs/{run_id}/chat", json={"message": "hi", "live": False})
            assert res.status_code == 200
        res = c.post(f"/api/runs/{run_id}/chat", json={"message": "hi", "live": False})
        assert res.status_code == 429


def test_source_urls_limit(client):
    res = client.post(
        "/api/runs",
        json={
            "seed_document": "doc",
            "question": "q?",
            "live": False,
            "models": {"flagship": "x", "standard": "x", "cheap": "x"},
            "source_urls": [f"https://example.com/{i}" for i in range(7)],
        },
    )
    assert res.status_code == 422
