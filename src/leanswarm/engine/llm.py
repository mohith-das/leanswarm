from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import Mapping
from typing import Any

from diskcache import Cache
from pydantic import BaseModel

from leanswarm.engine.config import RuntimeSettings
from leanswarm.engine.logging import JsonlLogger
from leanswarm.engine.models import (
    AgentBatchResponse,
    MemorySummaryResponse,
    ModelTier,
    PredictionSynthesisResponse,
    TaskType,
    WorldBootstrapResponse,
    WorldExtractionResponse,
)
from leanswarm.engine.prompts import system_prompt

try:
    from litellm import acompletion
except ImportError:  # pragma: no cover
    acompletion = None


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    stripped = text
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        else:
            stripped = stripped[3:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        stripped = text[start : end + 1]
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


_RESPONSE_MODELS: dict[TaskType, type[BaseModel]] = {
    TaskType.WORLD_BOOTSTRAP: WorldBootstrapResponse,
    TaskType.WORLD_EXTRACTION: WorldExtractionResponse,
    TaskType.AGENT_BATCH: AgentBatchResponse,
    TaskType.MEMORY_SUMMARY: MemorySummaryResponse,
    TaskType.PREDICTION_SYNTHESIS: PredictionSynthesisResponse,
}


_PROVIDER_ENV_KEYS = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "minimax": ["MINIMAX_API_KEY"],
    "zhipuai": ["ZHIPUAI_API_KEY"],
    "zai": ["ZHIPUAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "ollama": [],
}

class LiveCredentialsError(Exception):
    def __init__(self, model: str, missing_keys: list[str]) -> None:
        msg = f"Live mode requested for model '{model}' but missing env keys: {', '.join(missing_keys)}. Set the key, set LEANSWARM_API_KEY/LEANSWARM_API_BASE, or run with LEANSWARM_DRY_RUN=true."
        super().__init__(msg)


class LiteLLMRouter:
    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        self.settings = settings or RuntimeSettings.from_env()
        self.settings.ensure_dirs()
        self.cache = Cache(str(self.settings.cache_dir / "llm"))
        self.logger = JsonlLogger(self.settings.llm_log_path)
        self.semaphore = asyncio.Semaphore(self.settings.max_concurrency)
        self.route_calls = 0
        self.cache_hits = 0
        self.prompt_tokens_by_tier = {tier.value: 0 for tier in ModelTier}
        self.completion_tokens_by_tier = {tier.value: 0 for tier in ModelTier}
        self.prompt_tokens_total: int = 0
        self.completion_tokens_total: int = 0
        self.prompt_tokens_by_model: dict[str, int] = {}
        self.completion_tokens_by_model: dict[str, int] = {}

    @property
    def cache_hit_rate(self) -> float:
        if self.route_calls == 0:
            return 0.0
        return self.cache_hits / self.route_calls

    @property
    def total_tokens_by_tier(self) -> dict[str, int]:
        return {
            tier: self.prompt_tokens_by_tier[tier] + self.completion_tokens_by_tier[tier]
            for tier in self.prompt_tokens_by_tier
        }

    async def route(self, task_type: TaskType | str, payload: Mapping[str, Any]) -> dict[str, Any]:
        resolved_task = TaskType(task_type)
        tier = self._tier_for_task(resolved_task)
        model = self._model_for_tier(tier)

        live = self._decide_live(payload, model)
        cache_key = self._cache_key(resolved_task, payload, model, live)

        self.route_calls += 1
        cached_entry = self.cache.get(cache_key)
        if cached_entry is not None:
            self.cache_hits += 1
            if isinstance(cached_entry, dict) and "response" in cached_entry and "mode" in cached_entry:
                cached_response = cached_entry["response"]
                cached_mode = cached_entry["mode"]
            else:
                cached_response = cached_entry
                cached_mode = "unknown"

            self.logger.log(
                {
                    "task_type": resolved_task.value,
                    "tier": tier.value,
                    "model": model,
                    "cached": True,
                    "mode": "cached",
                    "cached_mode": cached_mode,
                    "token_source": "none",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            )
            return dict(cached_response)

        async with self.semaphore:
            response, mode, prompt_usage, completion_usage = await self._call_with_retry(
                resolved_task, payload, model, live
            )

        if prompt_usage is not None and completion_usage is not None:
            prompt_tokens = prompt_usage
            completion_tokens = completion_usage
            token_source = "usage"
        else:
            prompt_tokens = self._estimate_tokens({"task_type": resolved_task.value, "payload": payload})
            completion_tokens = self._estimate_tokens(response)
            token_source = "estimated" if mode in ("live", "mock", "mock_fallback") else "none"

        self.prompt_tokens_by_tier[tier.value] += prompt_tokens
        self.completion_tokens_by_tier[tier.value] += completion_tokens
        self.prompt_tokens_total += prompt_tokens
        self.completion_tokens_total += completion_tokens
        
        if model not in self.prompt_tokens_by_model:
            self.prompt_tokens_by_model[model] = 0
            self.completion_tokens_by_model[model] = 0
        self.prompt_tokens_by_model[model] += prompt_tokens
        self.completion_tokens_by_model[model] += completion_tokens

        self.cache.set(cache_key, {"response": response, "mode": mode})
        self.logger.log(
            {
                "task_type": resolved_task.value,
                "tier": tier.value,
                "model": model,
                "cached": False,
                "mode": mode,
                "token_source": token_source,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        )
        return response

    async def _call_with_retry(
        self, task_type: TaskType, payload: Mapping[str, Any], model: str, live: bool
    ) -> tuple[dict[str, Any], str, int | None, int | None]:
        if not live:
            return self._mock_response(task_type, payload), "mock", None, None

        for attempt in range(1, self.settings.retry_attempts + 1):
            try:
                return await self._live_response(task_type, payload, model)
            except Exception as exc:  # pragma: no cover
                if attempt == self.settings.retry_attempts:
                    raise exc
                await asyncio.sleep(0.2 * attempt)
        return self._mock_response(task_type, payload), "mock", None, None

    async def _live_response(
        self, task_type: TaskType, payload: Mapping[str, Any], model: str
    ) -> tuple[dict[str, Any], str, int | None, int | None]:
        if acompletion is None:
            return self._mock_response(task_type, payload), "mock_fallback", None, None

        user_content = json.dumps(
            {"task_type": task_type.value, "payload": payload},
            sort_keys=True,
            default=str,
        )
        sys_prompt = system_prompt(task_type)

        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.settings.api_base is not None:
            kwargs["api_base"] = self.settings.api_base
        
        resolved_key = self._resolve_api_key(model)
        if resolved_key is not None:
            kwargs["api_key"] = resolved_key

        try:
            completion = await acompletion(**kwargs)
        except Exception as e:
            if "response_format" in str(e).lower():
                kwargs.pop("response_format")
                completion = await acompletion(**kwargs)
            else:
                raise

        prompt_tokens = None
        completion_tokens = None
        if hasattr(completion, "usage") and completion.usage:
            usage = completion.usage
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)

        message = completion.choices[0].message.content or ""
        parsed = extract_json_object(message)

        response_model = _RESPONSE_MODELS[task_type]
        validation_error = ""

        if parsed is None:
            validation_error = "response was not a JSON object"
        else:
            try:
                model_instance = response_model.model_validate(parsed)
                return model_instance.model_dump(), "live", prompt_tokens, completion_tokens
            except Exception as e:
                validation_error = str(e)

        repair_messages = kwargs["messages"].copy()
        err_msg = validation_error[:500]
        raw_msg = message[:500]
        repair_messages.append(
            {
                "role": "user",
                "content": f"Your previous response failed validation: {err_msg}. Previous response: {raw_msg}. Return ONLY the corrected JSON object matching the schema.",
            }
        )
        kwargs["messages"] = repair_messages
        completion = await acompletion(**kwargs)

        prompt_tokens = None
        completion_tokens = None
        if hasattr(completion, "usage") and completion.usage:
            usage = completion.usage
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)

        message = completion.choices[0].message.content or ""
        parsed = extract_json_object(message)

        if parsed is None:
            return self._mock_response(task_type, payload), "mock_fallback", prompt_tokens, completion_tokens

        try:
            model_instance = response_model.model_validate(parsed)
            return model_instance.model_dump(), "live", prompt_tokens, completion_tokens
        except Exception:
            return self._mock_response(task_type, payload), "mock_fallback", prompt_tokens, completion_tokens

    def _tier_for_task(self, task_type: TaskType) -> ModelTier:
        if task_type is TaskType.PREDICTION_SYNTHESIS:
            return ModelTier.FLAGSHIP
        if task_type in (TaskType.AGENT_BATCH, TaskType.WORLD_EXTRACTION):
            return ModelTier.CHEAP
        return ModelTier.STANDARD

    def _model_for_tier(self, tier: ModelTier) -> str:
        if tier is ModelTier.FLAGSHIP:
            return self.settings.flagship_model
        if tier is ModelTier.STANDARD:
            return self.settings.standard_model
        return self.settings.cheap_model

    def _decide_live(self, payload: Mapping[str, Any], model: str) -> bool:
        use_llm = bool(payload.get("use_llm", True))
        if self.settings.dry_run or not use_llm:
            return False
        
        ready, missing = self._live_ready(model)
        if not ready:
            raise LiveCredentialsError(model, missing)
        return True

    def _cache_key(
        self, task_type: TaskType, payload: Mapping[str, Any], model: str, live: bool
    ) -> str:
        serialized = json.dumps(
            {"v": 2, "task_type": task_type.value, "payload": payload, "model": model, "live": live},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _live_ready(self, model: str) -> tuple[bool, list[str]]:
        """Return (ready, missing_keys) for live calls against `model`."""
        if self.settings.api_key:
            return True, []

        # Per-run credentials (e.g. browser-provided keys in the web UI) are not
        # visible to litellm's env-based validation, so check them first.
        prefix = model.split("/", 1)[0] if "/" in model else "openai"
        required = _PROVIDER_ENV_KEYS.get(prefix)
        if required is not None and required and all(
            self.settings.credentials.get(k) for k in required
        ):
            return True, []

        try:
            from litellm import validate_environment
            res = validate_environment(model=model)
            return res.get("keys_in_environment", False), res.get("missing_keys", [])
        except Exception:
            pass

        if prefix not in _PROVIDER_ENV_KEYS:
            return True, []

        missing = [
            k for k in _PROVIDER_ENV_KEYS[prefix] 
            if not (self.settings.credentials.get(k) or os.getenv(k))
        ]
        return len(missing) == 0, missing

    def _resolve_api_key(self, model: str) -> str | None:
        """Explicit key for this model: generic override first, then per-provider."""
        if self.settings.api_key:
            return self.settings.api_key
        prefix = model.split("/", 1)[0] if "/" in model else "openai"
        for env_name in _PROVIDER_ENV_KEYS.get(prefix, []):
            value = self.settings.credentials.get(env_name)
            if value:
                return value
        return None

    def _estimate_tokens(self, payload: Any) -> int:
        serialized = json.dumps(payload, sort_keys=True, default=str)
        token_like_units = re.findall(r"[A-Za-z0-9_'\-]+", serialized)
        return max(1, int(len(token_like_units) * 1.15))

    def _mock_response(self, task_type: TaskType, payload: Mapping[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", "the outcome"))
        seed_text = self._payload_text(
            payload, ("seed_document", "seed_excerpt", "summary", "context")
        )
        question_terms = self._extract_terms(question, limit=5)
        seed_terms = self._extract_terms(seed_text, limit=8)

        if task_type is TaskType.WORLD_EXTRACTION:
            return {
                "summary": "",
                "sentiment": {"label": "neutral", "score": 0.0, "confidence": 0.0},
                "topics": [],
                "entities": [],
                "relations": [],
            }

        if task_type is TaskType.WORLD_BOOTSTRAP:
            return self._mock_world_bootstrap(
                question, seed_text, seed_terms, question_terms, payload
            )

        if task_type is TaskType.AGENT_BATCH:
            return self._mock_agent_batch(question, seed_text, seed_terms, question_terms, payload)

        if task_type is TaskType.MEMORY_SUMMARY:
            return self._mock_memory_summary(
                question, seed_text, seed_terms, question_terms, payload
            )

        if task_type is TaskType.PREDICTION_SYNTHESIS:
            return self._mock_prediction_synthesis(
                question, seed_text, seed_terms, question_terms, payload
            )

        return {
            "summary": f"World bootstrap completed for {question}.",
            "topics": question_terms[:3],
        }

    def _mock_world_bootstrap(
        self,
        question: str,
        seed_text: str,
        seed_terms: list[str],
        question_terms: list[str],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        signature = self._signature(question, seed_text, payload.get("agent_count", 0), "world")
        tone_options = ["optimistic", "cautious", "fractured", "measured", "volatile"]
        pressure_options = ["attention", "trust", "narrative control", "coordination", "legitimacy"]
        tone = tone_options[signature % len(tone_options)]
        pressure = pressure_options[(signature // 11) % len(pressure_options)]
        entities = self._pick_unique_terms(seed_terms, question_terms, signature, limit=4)
        topics = self._pick_unique_terms(question_terms, seed_terms, signature >> 3, limit=4)
        return {
            "summary": (
                f"Bootstrapped a {tone} world around {self._format_terms(topics)} "
                f"with pressure on {pressure}."
            ),
            "topics": topics,
            "entities": entities,
            "sentiment": tone,
            "pressure_points": [
                pressure,
                self._format_terms(entities[:2]) if entities else pressure,
            ],
            "agent_count": int(payload.get("agent_count", 0) or 0),
        }

    def _mock_agent_batch(
        self,
        question: str,
        seed_text: str,
        seed_terms: list[str],
        question_terms: list[str],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        agents = list(payload.get("agents", []))
        tick = int(payload.get("tick", 0) or 0)
        actions: list[dict[str, Any]] = []

        for index, agent in enumerate(agents):
            agent_id = str(agent.get("id", f"agent-{index + 1}"))
            name = str(agent.get("name", agent_id))
            archetype = str(agent.get("archetype", "generalist"))
            signature = self._signature(question, seed_text, archetype, agent_id, tick, index)
            action_type = self._action_type_for(archetype, signature)
            focus_terms = self._pick_unique_terms(seed_terms, question_terms, signature, limit=2)
            target_terms = focus_terms or question_terms or ["outcome"]
            focus_text = self._format_terms(target_terms)
            stance = self._stance_for_archetype(archetype)
            deltas = self._deltas_for_action(action_type, stance, signature)
            description = self._describe_agent_action(
                name, archetype, action_type, focus_text, question, signature
            )
            actions.append(
                {
                    "agent_id": agent_id,
                    "action_type": action_type,
                    "description": description,
                    "delta_mood": deltas[0],
                    "delta_energy": deltas[1],
                    "delta_attention": deltas[2],
                }
            )
        return {"actions": actions}

    def _mock_memory_summary(
        self,
        question: str,
        seed_text: str,
        seed_terms: list[str],
        question_terms: list[str],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        signature = self._signature(question, seed_text, payload.get("agent_id", ""), "memory")
        focus_terms = self._pick_unique_terms(seed_terms, question_terms, signature, limit=3)
        recent_notes = self._extract_terms(
            self._payload_text(payload, ("recent_observations", "working", "episodic", "notes")),
            limit=4,
        )
        retained = self._pick_unique_terms(focus_terms, recent_notes, signature >> 5, limit=4)
        return {
            "summary": (
                f"Compressed recent observations into a short memory about "
                f"{self._format_terms(retained or focus_terms or question_terms)}."
            ),
            "topics": retained or focus_terms,
            "retained_signals": retained,
        }

    def _mock_prediction_synthesis(
        self,
        question: str,
        seed_text: str,
        seed_terms: list[str],
        question_terms: list[str],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        ticks = list(payload.get("ticks", []))
        signature = self._signature(
            question, seed_text, ticks, payload.get("agent_count", 0), "prediction"
        )
        seed_bias = self._polarity_score(seed_text)
        question_bias = self._polarity_score(question)
        tick_bias = self._tick_bias(ticks)
        combined_bias = (seed_bias * 4) + question_bias + tick_bias
        direction = "positive" if combined_bias > 0 else "negative"
        confidence = self._confidence_from_ticks(ticks, combined_bias)
        dominant_terms = self._pick_unique_terms(question_terms, seed_terms, signature, limit=3)
        support_terms = self._pick_unique_terms(seed_terms, question_terms, signature >> 4, limit=3)
        volatility = self._volatility_from_ticks(ticks)
        tick_count = len(ticks)
        stable_ticks = sum(1 for tick in ticks if bool(tick.get("stable")))

        if direction == "positive":
            prediction = (
                f"The simulation leans positive on '{question}', with {self._format_terms(dominant_terms)} "
                f"gaining traction despite some volatility."
            )
        else:
            prediction = (
                f"The simulation leans negative on '{question}', with {self._format_terms(dominant_terms)} "
                f"facing drag from volatility and competing signals."
            )

        rationale = [
            f"The seed frames the issue around {self._format_terms(support_terms or seed_terms[:3])}.",
            f"Observed activity covered {tick_count} ticks and {stable_ticks} stable ticks, indicating {volatility} volatility.",
            f"Active agent discussion concentrated on {self._format_terms(dominant_terms or question_terms[:3])}.",
        ]
        if combined_bias == 0:
            rationale.append(
                "Seed and question signals were balanced, so the forecast remains cautious."
            )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "rationale": rationale,
            "direction": direction,
            "supporting_terms": support_terms,
            "volatility": volatility,
        }

    def _payload_text(self, payload: Mapping[str, Any], keys: tuple[str, ...]) -> str:
        parts = [str(payload.get(key, "")) for key in keys]
        return " ".join(part for part in parts if part).strip()

    def _extract_terms(self, text: str, limit: int) -> list[str]:
        stopwords = {
            "a",
            "about",
            "after",
            "all",
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
            "to",
            "was",
            "will",
            "with",
            "without",
        }
        tokens = [token.lower() for token in re.findall(r"[a-zA-Z][a-zA-Z0-9'\-]*", text)]
        terms: list[str] = []
        for token in tokens:
            if len(token) < 4 or token in stopwords:
                continue
            if token not in terms:
                terms.append(token)
            if len(terms) >= limit:
                break
        return terms

    def _signature(self, *parts: Any) -> int:
        serialized = json.dumps(parts, sort_keys=True, default=str)
        return int(hashlib.sha256(serialized.encode("utf-8")).hexdigest(), 16)

    def _pick_unique_terms(self, *term_groups: Any, limit: int) -> list[str]:
        signature = 0
        groups: list[list[str]] = []
        for group in term_groups:
            if isinstance(group, int):
                signature ^= group
                continue
            if isinstance(group, (list, tuple)):
                groups.append([str(term) for term in group])
                continue
            if isinstance(group, str) and group:
                groups.append([group])
        selected: list[str] = []
        for index, group in enumerate(groups):
            if not group:
                continue
            offset = (signature >> (index * 3)) % len(group) if signature else 0
            ordered = group[offset:] + group[:offset]
            for term in ordered:
                if term not in selected:
                    selected.append(term)
                if len(selected) >= limit:
                    return selected
        if signature and selected:
            rotation = signature % len(selected)
            selected = selected[rotation:] + selected[:rotation]
        return selected

    def _format_terms(self, terms: list[str] | tuple[str, ...] | Any) -> str:
        if not terms:
            return "the outcome"
        if isinstance(terms, (list, tuple)):
            if len(terms) == 1:
                return str(terms[0])
            if len(terms) == 2:
                return f"{terms[0]} and {terms[1]}"
            return ", ".join(str(term) for term in terms[:-1]) + f", and {terms[-1]}"
        return str(terms)

    def _action_type_for(self, archetype: str, signature: int) -> str:
        archetype_lower = archetype.lower()
        if "skept" in archetype_lower or "observer" in archetype_lower:
            options = ["observe", "challenge", "probe"]
        elif "organ" in archetype_lower or "ampl" in archetype_lower:
            options = ["coordinate", "amplify", "sync"]
        elif "analyst" in archetype_lower:
            options = ["probe", "observe", "reassess"]
        else:
            options = ["observe", "connect", "rebalance"]
        return options[signature % len(options)]

    def _stance_for_archetype(self, archetype: str) -> str:
        archetype_lower = archetype.lower()
        if "optim" in archetype_lower:
            return "uplift"
        if "skept" in archetype_lower:
            return "pressure"
        if "caut" in archetype_lower:
            return "guarded"
        if "prag" in archetype_lower:
            return "balanced"
        return "adaptive"

    def _deltas_for_action(
        self, action_type: str, stance: str, signature: int
    ) -> tuple[float, float, float]:
        mood_base = {
            "uplift": 0.05,
            "pressure": -0.03,
            "guarded": -0.01,
            "balanced": 0.01,
            "adaptive": 0.0,
        }.get(stance, 0.0)
        energy_base = {
            "coordinate": -0.03,
            "amplify": -0.01,
            "sync": -0.02,
            "observe": -0.01,
            "challenge": -0.02,
            "probe": -0.015,
            "reassess": -0.02,
            "connect": -0.015,
            "rebalance": -0.01,
        }.get(action_type, -0.01)
        attention_base = {
            "coordinate": 0.08,
            "amplify": 0.07,
            "sync": 0.06,
            "observe": 0.05,
            "challenge": 0.08,
            "probe": 0.09,
            "reassess": 0.06,
            "connect": 0.05,
            "rebalance": 0.04,
        }.get(action_type, 0.05)
        jitter = ((signature % 7) - 3) * 0.006
        mood = round(mood_base + jitter, 3)
        energy = round(energy_base - jitter / 2, 3)
        attention = round(attention_base + abs(jitter), 3)
        return mood, energy, attention

    def _describe_agent_action(
        self,
        name: str,
        archetype: str,
        action_type: str,
        focus_text: str,
        question: str,
        signature: int,
    ) -> str:
        variants = {
            "observe": "tracks",
            "challenge": "pushes back on",
            "probe": "tests",
            "coordinate": "aligns the group around",
            "amplify": "amplifies",
            "sync": "synchronizes around",
            "reassess": "rechecks",
            "connect": "connects",
            "rebalance": "rebalances",
        }
        connector = variants.get(action_type, "updates around")
        question_focus = self._format_terms(self._extract_terms(question, limit=3)[:2])
        suffixes = [
            "to sharpen the forecast.",
            "before the next tick.",
            "to keep the narrative grounded.",
            "while the group reconsiders its stance.",
        ]
        suffix = suffixes[signature % len(suffixes)]
        return f"{name} ({archetype}) {connector} {focus_text} against {question_focus} {suffix}"

    def _polarity_score(self, text: str) -> int:
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
            "slip",
        }
        score = 0
        for token in self._extract_terms(text, limit=32):
            if token in positive:
                score += 1
            elif token in negative:
                score -= 1
        return score

    def _tick_bias(self, ticks: list[Mapping[str, Any]]) -> int:
        if not ticks:
            return 0
        bias = 0
        for tick in ticks:
            mean_delta = float(tick.get("mean_delta", 0.0) or 0.0)
            if mean_delta > 0.04:
                bias += 1
            elif mean_delta < 0.02:
                bias -= 1
            if tick.get("stable"):
                bias += 1
        return bias

    def _volatility_from_ticks(self, ticks: list[Mapping[str, Any]]) -> str:
        if not ticks:
            return "low"
        mean_delta = sum(float(tick.get("mean_delta", 0.0) or 0.0) for tick in ticks) / len(ticks)
        stable_count = sum(1 for tick in ticks if tick.get("stable"))
        if mean_delta >= 0.05:
            return "high"
        if stable_count >= max(1, len(ticks) // 2):
            return "low"
        return "moderate"

    def _confidence_from_ticks(self, ticks: list[Mapping[str, Any]], combined_bias: int) -> float:
        if not ticks:
            return 0.55
        mean_delta = sum(float(tick.get("mean_delta", 0.0) or 0.0) for tick in ticks) / len(ticks)
        stable_count = sum(1 for tick in ticks if tick.get("stable"))
        confidence = 0.5 + min(0.2, stable_count * 0.04) - min(0.12, mean_delta * 1.4)
        confidence += min(0.08, abs(combined_bias) * 0.015)
        return round(max(0.1, min(0.95, confidence)), 3)
