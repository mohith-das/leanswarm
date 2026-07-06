import { useState } from 'react';
import { api } from '../api';
import { collectCredentials, getOverrides } from '../keys';

interface Turn { role: 'user' | 'assistant'; content: string }

export default function ChatPanel({ runId, agents }: { runId: string; agents: Array<{ id: string; name: string }> }) {
  const [target, setTarget] = useState<string>('');
  const [live, setLive] = useState(false);
  const [model, setModel] = useState('deepseek/deepseek-chat');
  const [history, setHistory] = useState<Turn[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function send() {
    const message = input.trim();
    if (!message || busy) return;
    setBusy(true);
    setError(null);
    const nextHistory: Turn[] = [...history, { role: 'user', content: message }];
    setHistory(nextHistory);
    setInput('');
    try {
      const body: Record<string, unknown> = {
        agent_id: target || null,
        message,
        history: history.slice(-8),
        live,
      };
      if (live) {
        body.models = { flagship: model, standard: model, cheap: model };
        body.credentials = collectCredentials([model]);
        Object.assign(body, getOverrides());
      }
      const res = await api.chat(runId, body);
      setHistory([...nextHistory, { role: 'assistant', content: res.reply }]);
    } catch (err: any) {
      setError(err.message || 'Chat failed');
      setHistory(history);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card chat-panel">
      <div className="row-between">
        <h3>Talk to the world</h3>
        <div className="row chat-controls">
          <select value={target} onChange={(e) => { setTarget(e.target.value); setHistory([]); }}>
            <option value="">Report Agent</option>
            {agents.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
          </select>
          <label className="row">
            <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} />
            Live
          </label>
          {live && <input list="chat-model-presets" value={model} onChange={(e) => setModel(e.target.value)} placeholder="model" />}
        </div>
      </div>
      <datalist id="chat-model-presets">
        <option value="deepseek/deepseek-chat" />
        <option value="deepseek/deepseek-reasoner" />
        <option value="gpt-4.1" />
        <option value="gpt-4.1-mini" />
        <option value="groq/llama-3.3-70b-versatile" />
        <option value="anthropic/claude-sonnet-5" />
      </datalist>
      <div className="chat-history">
        {history.length === 0 && (
          <p className="muted">
            Ask {target ? 'this agent' : 'the Report Agent'} about the simulation.
            {!live && ' (Mock mode: deterministic canned replies — flip Live for real answers.)'}
          </p>
        )}
        {history.map((t, i) => (
          <div key={i} className={`chat-turn chat-${t.role}`}>{t.content}</div>
        ))}
        {busy && <div className="chat-turn chat-assistant muted">…</div>}
      </div>
      {error && <p className="error-text">{error}</p>}
      <div className="row chat-input-row">
        <input
          className="w-full"
          value={input}
          maxLength={1000}
          placeholder="Type a question…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') send(); }}
        />
        <button className="btn btn-accent" onClick={send} disabled={busy || !input.trim()}>Send</button>
      </div>
    </div>
  );
}
