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
