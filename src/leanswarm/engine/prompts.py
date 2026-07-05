from leanswarm.engine.models import TaskType


def system_prompt(task_type: TaskType) -> str:
    common_rules = "Respond with a single JSON object and nothing else. No markdown fences, no commentary, no keys other than those specified. All strings in English."

    if task_type == TaskType.WORLD_BOOTSTRAP:
        return f"""You are a world-building simulation engine.
{common_rules}

Return exactly this JSON schema:
{{
  "summary": "1-2 sentences summarizing the world state",
  "topics": ["topic1", "topic2", "topic3"],
  "entities": ["entity1", "entity2", "entity3"],
  "sentiment": "positive|negative|mixed|neutral",
  "pressure_points": ["pressure1", "pressure2"]
}}

Ground everything in the provided seed excerpt. Topics should be 3-5 short strings. Entities should be 3-5 short strings. Sentiment must be one of the literal strings positive, negative, mixed, or neutral. Pressure points should be 2-3 short strings."""

    elif task_type == TaskType.AGENT_BATCH:
        return f"""You are an agent simulation engine processing a batch of agents.
{common_rules}

Return exactly this JSON schema:
{{
  "actions": [
    {{
      "agent_id": "exact_string_from_input",
      "action_type": "observe|challenge|probe|coordinate|amplify|sync|reassess|connect|rebalance",
      "description": "One sentence, third person, mentions the agent by name.",
      "delta_mood": 0.0,
      "delta_energy": 0.0,
      "delta_attention": 0.0
    }}
  ]
}}

The `actions` array must contain exactly one object per agent in the input `agents` array, in the exact same order.
The `agent_id` must be copied verbatim from the input agent's `id`.
The `action_type` must be one of the literal strings listed above.
The `delta_*` fields must be floats between -0.15 and 0.15 reflecting how the action shifts that agent's state given the world summary and question."""

    elif task_type == TaskType.MEMORY_SUMMARY:
        return f"""You are a cognitive simulation engine summarizing an agent's memory.
{common_rules}

Return exactly this JSON schema:
{{
  "summary": "≤ 2 sentences compressing recent_observations and episodic memory",
  "topics": ["topic1", "topic2"],
  "retained_signals": ["signal1", "signal2"]
}}

Topics should be 2-4 strings. Retained signals should be 2-4 strings."""

    elif task_type == TaskType.PREDICTION_SYNTHESIS:
        return f"""You are a predictive synthesis engine.
{common_rules}

Return exactly this JSON schema:
{{
  "prediction": "2-3 sentences directly answering the question",
  "confidence": 0.5,
  "rationale": ["Sentence 1.", "Sentence 2."],
  "direction": "positive|negative",
  "supporting_terms": ["term1", "term2"],
  "volatility": "low|moderate|high"
}}

Confidence must be a float between 0.0 and 1.0. Rationale should be an array of 2-4 sentences citing tick activity and seed evidence. Direction must be positive or negative. Supporting terms should be 2-4 strings. Volatility must be low, moderate, or high."""

    return common_rules
