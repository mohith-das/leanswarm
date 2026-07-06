export const PROVIDER_ENV_KEYS: Record<string, string[]> = {
  openai: ["OPENAI_API_KEY"],
  anthropic: ["ANTHROPIC_API_KEY"],
  deepseek: ["DEEPSEEK_API_KEY"],
  minimax: ["MINIMAX_API_KEY"],
  zhipuai: ["ZHIPUAI_API_KEY"],
  zai: ["ZHIPUAI_API_KEY"],
  gemini: ["GEMINI_API_KEY"],
  groq: ["GROQ_API_KEY"],
  mistral: ["MISTRAL_API_KEY"],
  xai: ["XAI_API_KEY"],
  openrouter: ["OPENROUTER_API_KEY"],
  ollama: [],
};

export const PROVIDER_LABELS: Record<string, string> = {
  openai: "OpenAI",
  anthropic: "Anthropic",
  deepseek: "DeepSeek",
  minimax: "MiniMax",
  zhipuai: "Zhipu GLM",
  gemini: "Gemini",
  groq: "Groq",
  mistral: "Mistral",
  xai: "xAI",
  openrouter: "OpenRouter",
  ollama: "Ollama (local)",
  custom: "Custom endpoint",
};

export const PROVIDER_MODELS: Record<string, string[]> = {
  openai: ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o"],
  anthropic: ["anthropic/claude-sonnet-5", "anthropic/claude-haiku-4-5-20251001"],
  deepseek: ["deepseek/deepseek-chat", "deepseek/deepseek-reasoner"],
  minimax: ["minimax/minimax-text-01"],
  zhipuai: ["zhipuai/glm-4", "zhipuai/glm-4-air"],
  gemini: ["gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro"],
  groq: ["groq/llama-3.3-70b-versatile"],
  mistral: ["mistral/mistral-small-latest"],
  xai: [],
  openrouter: [],
  ollama: ["ollama/llama3.1", "ollama/qwen2.5"],
  custom: [],
};

export function getAvailableProviders(): string[] {
  let stored: Record<string, string> = {};
  try {
    stored = JSON.parse(localStorage.getItem("leanswarm.keys.v1") || "{}");
  } catch {
    stored = {};
  }
  const providers: string[] = [];
  for (const [prefix, keys] of Object.entries(PROVIDER_ENV_KEYS)) {
    if (keys.length === 0) {
      providers.push(prefix);
    } else if (keys.every((k) => stored[k])) {
      providers.push(prefix);
    }
  }
  if (stored.LEANSWARM_API_KEY) {
    providers.push("custom");
  }
  return providers;
}

export function getRequiredKeys(models: string[]): string[] {
  const keys = new Set<string>();
  for (const m of models) {
    const prefix = m.includes("/") ? m.split("/")[0] : "openai";
    const req = PROVIDER_ENV_KEYS[prefix] || [];
    req.forEach(k => keys.add(k));
  }
  return Array.from(keys);
}

export function collectCredentials(models: string[]): Record<string, string> {
  const req = getRequiredKeys(models);
  const stored = JSON.parse(localStorage.getItem("leanswarm.keys.v1") || "{}");
  const result: Record<string, string> = {};
  for (const k of req) {
    if (stored[k]) result[k] = stored[k];
  }
  return result;
}

export function getOverrides(): { api_base?: string; api_key?: string } {
  let stored: Record<string, string> = {};
  try {
    stored = JSON.parse(localStorage.getItem("leanswarm.keys.v1") || "{}");
  } catch {
    stored = {};
  }
  const out: { api_base?: string; api_key?: string } = {};
  if (stored.LEANSWARM_API_BASE) out.api_base = stored.LEANSWARM_API_BASE;
  if (stored.LEANSWARM_API_KEY) out.api_key = stored.LEANSWARM_API_KEY;
  return out;
}
