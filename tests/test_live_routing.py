import json

import pytest

from leanswarm.engine.config import RuntimeSettings
from leanswarm.engine.llm import LiteLLMRouter, LiveCredentialsError, extract_json_object
from leanswarm.engine.models import AgentAction, TaskType
from leanswarm.engine.simulator import LeanSwarmEngine


def test_extract_json_object():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('```json\n{"b": 2}\n```') == {"b": 2}
    assert extract_json_object('```\n{"c": 3}\n```') == {"c": 3}
    assert extract_json_object('Here is the json: {"d": 4} and some text') == {"d": 4}
    assert extract_json_object('Trailing commentary {"e": 5} this is ignored') == {"e": 5}
    assert extract_json_object('Garbage string without braces') is None
    assert extract_json_object('[1, 2, 3]') is None


def test_credential_gate(monkeypatch):
    router = LiteLLMRouter(RuntimeSettings(dry_run=False))
    # mock litellm.validate_environment to avoid actual network/env checks deep inside litellm
    import litellm
    def mock_validate_raise(model):
        raise Exception("force static map")
    monkeypatch.setattr(litellm, "validate_environment", mock_validate_raise)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    ready, missing = router._live_ready("deepseek/deepseek-chat")
    assert ready is True
    assert missing == []

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    ready, missing = router._live_ready("deepseek/deepseek-chat")
    assert ready is False
    assert "DEEPSEEK_API_KEY" in missing

    monkeypatch.setenv("MINIMAX_API_KEY", "x")
    ready, missing = router._live_ready("minimax/minimax-m2")
    assert ready is True

    monkeypatch.setenv("ZHIPUAI_API_KEY", "x")
    ready, missing = router._live_ready("zhipuai/glm-4")
    assert ready is True
    
    router.settings.api_key = "universal_key"
    ready, missing = router._live_ready("any_model")
    assert ready is True


@pytest.mark.asyncio
async def test_live_credentials_error(monkeypatch):
    router = LiteLLMRouter(RuntimeSettings(dry_run=False))
    import litellm
    def mock_validate_raise(model):
        raise Exception("force static map")
    monkeypatch.setattr(litellm, "validate_environment", mock_validate_raise)
    
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    
    with pytest.raises(LiveCredentialsError) as exc:
        await router.route(TaskType.WORLD_BOOTSTRAP, {"use_llm": True})
    assert "OPENAI_API_KEY" in str(exc.value)

    # With use_llm=False, should return mock and not raise
    response = await router.route(TaskType.WORLD_BOOTSTRAP, {"use_llm": False})
    assert "summary" in response


@pytest.mark.asyncio
async def test_live_validation_and_repair(monkeypatch, tmp_path):
    settings = RuntimeSettings(dry_run=False, cache_dir=tmp_path / "cache", log_dir=tmp_path / "logs", api_key="test")
    router = LiteLLMRouter(settings)

    call_count = 0
    responses_to_give = []

    async def mock_acompletion(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        content = responses_to_give.pop(0)
        
        class Usage:
            prompt_tokens = 10
            completion_tokens = 5
            
        class Message:
            def __init__(self, content):
                self.content = content

        class Choice:
            def __init__(self, msg):
                self.message = Message(msg)

        class Completion:
            def __init__(self, choices):
                self.choices = choices
                self.usage = Usage()
                
        return Completion([Choice(content)])

    monkeypatch.setattr("leanswarm.engine.llm.acompletion", mock_acompletion)

    # a) valid AGENT_BATCH JSON
    responses_to_give = [json.dumps({"actions": [{"agent_id": "a", "action_type": "observe", "description": "d", "delta_mood": 0.1, "delta_energy": 0.1, "delta_attention": 0.1}]})]
    response = await router.route(TaskType.AGENT_BATCH, {"use_llm": True})
    assert call_count == 1
    assert len(response["actions"]) == 1
    
    # Check log mode
    log_lines = settings.llm_log_path.read_text().splitlines()
    last_log = json.loads(log_lines[-1])
    assert last_log["mode"] == "live"
    assert last_log["token_source"] == "usage"
    
    # b) invalid JSON first, valid on repair
    responses_to_give = ["Not JSON", json.dumps({"actions": []})]
    # Wait, the prompt says "The `actions` array must contain exactly one object per agent in the input `agents` array, in the same order." but we are passing an empty payload "agents": [] so [] is valid!
    # Wait, the response is cached! Let's change the payload.
    response = await router.route(TaskType.AGENT_BATCH, {"use_llm": True, "agents": []})
    assert call_count == 3  # 1 from previous test, +2 now
    
    # c) invalid twice -> mock fallback
    responses_to_give = ["Not JSON", "Still not JSON"]
    response = await router.route(TaskType.AGENT_BATCH, {"use_llm": True, "payload_diff": "c"})
    assert call_count == 5
    assert "actions" in response
    log_lines = settings.llm_log_path.read_text().splitlines()
    last_log = json.loads(log_lines[-1])
    assert last_log["mode"] == "mock_fallback"


@pytest.mark.asyncio
async def test_cache_separation(tmp_path):
    settings = RuntimeSettings(dry_run=True, cache_dir=tmp_path / "cache", log_dir=tmp_path / "logs")
    router = LiteLLMRouter(settings)
    
    # Dry run caches mock
    await router.route(TaskType.WORLD_BOOTSTRAP, {"question": "test"})
    assert router.route_calls == 1

    # Same payload dry run returns from cache
    await router.route(TaskType.WORLD_BOOTSTRAP, {"question": "test"})
    assert router.cache_hits == 1
    
    # Live run (using use_llm=True, dry_run=False, and api_key to pass gate)
    settings.dry_run = False
    settings.api_key = "dummy"
    # To prevent actual network call, we can just check if it misses the cache and tries to call
    # But wait, it will try to call litellm. acompletion might raise error or we mock it.
    try:
        await router.route(TaskType.WORLD_BOOTSTRAP, {"question": "test"})
    except Exception:
        pass
    
    # Cache hit should still be 1, because live=True generated a different key
    assert router.cache_hits == 1


def test_agent_action_tolerance():
    action = AgentAction.model_validate({
        "agent_id": "a", 
        "action_type": "observe",
        "description": "d", 
        "delta_mood": "0.9",
        "delta_energy": "-0.3",
        "delta_attention": "invalid"
    })
    assert action.delta_mood == 0.25
    assert action.delta_energy == -0.25
    assert action.delta_attention == 0.0


@pytest.mark.asyncio
async def test_end_to_end_dry_run():
    engine = LeanSwarmEngine(RuntimeSettings(dry_run=True))
    result = await engine.smoke_test()
    assert result.report.prediction
