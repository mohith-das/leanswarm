import { useState } from 'react';

const PROVIDERS = [
  { label: 'OpenAI', key: 'OPENAI_API_KEY' },
  { label: 'Anthropic', key: 'ANTHROPIC_API_KEY' },
  { label: 'DeepSeek', key: 'DEEPSEEK_API_KEY' },
  { label: 'MiniMax', key: 'MINIMAX_API_KEY' },
  { label: 'Zhipu GLM', key: 'ZHIPUAI_API_KEY' },
  { label: 'Gemini', key: 'GEMINI_API_KEY' },
  { label: 'Groq', key: 'GROQ_API_KEY' },
  { label: 'Mistral', key: 'MISTRAL_API_KEY' },
  { label: 'xAI', key: 'XAI_API_KEY' },
  { label: 'OpenRouter', key: 'OPENROUTER_API_KEY' },
];

const STORAGE_KEY = 'leanswarm.keys.v1';

function loadKeys(): Record<string, string> {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); } catch { return {}; }
}

export default function KeysModal({ onClose }: { onClose: () => void }) {
  const [values, setValues] = useState<Record<string, string>>(loadKeys());

  function setValue(key: string, value: string) {
    const next = { ...values, [key]: value };
    setValues(next);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  }

  function clearAll() {
    setValues({});
    localStorage.removeItem(STORAGE_KEY);
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal card" onClick={(e) => e.stopPropagation()}>
        <h3>API keys</h3>
        <p className="muted">
          Keys are stored only in this browser. They are sent to this server only when you
          start a live run, used in memory for that run, and never written to disk.
        </p>
        {PROVIDERS.map((p) => (
          <div key={p.key} className="key-row">
            <label>{p.label}</label>
            <input
              type="password"
              value={values[p.key] || ''}
              onChange={(e) => setValue(p.key, e.target.value)}
              placeholder={p.key}
            />
          </div>
        ))}
        <div className="key-row">
          <label>Custom base URL</label>
          <input type="text" value={values.LEANSWARM_API_BASE || ''} onChange={(e) => setValue('LEANSWARM_API_BASE', e.target.value)} placeholder="https://…/v1" />
        </div>
        <div className="key-row">
          <label>Custom API key</label>
          <input type="password" value={values.LEANSWARM_API_KEY || ''} onChange={(e) => setValue('LEANSWARM_API_KEY', e.target.value)} placeholder="any OpenAI-compatible endpoint" />
        </div>
        <div className="modal-actions">
          <button className="btn" onClick={clearAll}>Clear all</button>
          <button className="btn btn-accent" onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  );
}
