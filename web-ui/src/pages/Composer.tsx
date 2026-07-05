import { useState, type ChangeEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import { getRequiredKeys, collectCredentials, getOverrides } from '../keys';
import CostEstimate from '../components/CostEstimate';

const MODEL_PRESETS = [
  'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
  'deepseek/deepseek-chat', 'deepseek/deepseek-reasoner',
  'zhipuai/glm-4-air', 'minimax/minimax-text-01',
  'gemini/gemini-2.5-flash', 'gemini/gemini-2.5-pro',
  'groq/llama-3.3-70b-versatile', 'mistral/mistral-small-latest',
  'anthropic/claude-sonnet-5',
];

export default function Composer() {
  const navigate = useNavigate();
  const [title, setTitle] = useState('');
  const [seedDocument, setSeedDocument] = useState('');
  const [question, setQuestion] = useState('');
  const [live, setLive] = useState(false);
  const [sameModel, setSameModel] = useState(true);
  const [flagship, setFlagship] = useState('gpt-4.1');
  const [standard, setStandard] = useState('gpt-4.1-mini');
  const [cheap, setCheap] = useState('gpt-4.1-nano');
  const [rounds, setRounds] = useState(4);
  const [maxAgents, setMaxAgents] = useState(12);
  const [groupSize, setGroupSize] = useState(4);
  const [activeAgentFraction, setActiveAgentFraction] = useState(0.2);
  const [convergenceThreshold, setConvergenceThreshold] = useState(2);
  const [randomSeed, setRandomSeed] = useState(7);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [doctorResult, setDoctorResult] = useState<any[] | null>(null);
  const [doctorLoading, setDoctorLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const models = sameModel
    ? { flagship, standard: flagship, cheap: flagship }
    : { flagship, standard, cheap };
  const modelList = Object.values(models);
  const requiredKeys = live ? getRequiredKeys(modelList) : [];
  const storedKeys: Record<string, string> = JSON.parse(localStorage.getItem('leanswarm.keys.v1') || '{}');
  const missingKeys = requiredKeys.filter((k) => !storedKeys[k]);
  const canRun = !live || missingKeys.length === 0;

  function handleFile(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setSeedDocument(String(reader.result || ''));
    reader.readAsText(file);
  }

  async function runDoctor() {
    setDoctorLoading(true);
    try {
      const result = await api.doctor({
        models,
        credentials: collectCredentials(modelList),
        ...getOverrides(),
        ping: true,
      });
      setDoctorResult(result);
    } catch (err: any) {
      setDoctorResult([{ model: 'error', ready: false, missing: [err.message] }]);
    } finally {
      setDoctorLoading(false);
    }
  }

  async function handleRun() {
    setError(null);
    if (!seedDocument.trim() || !question.trim()) {
      setError('Seed document and question are required.');
      return;
    }
    setStarting(true);
    try {
      const { id } = await api.startRun({
        seed_document: seedDocument,
        question,
        rounds,
        max_agents: maxAgents,
        group_size: groupSize,
        active_agent_fraction: activeAgentFraction,
        convergence_threshold: convergenceThreshold,
        random_seed: randomSeed,
        live,
        models,
        credentials: live ? collectCredentials(modelList) : {},
        ...(live ? getOverrides() : {}),
        title: title || null,
      });
      navigate(`/run/${id}`);
    } catch (err: any) {
      setError(err.message || 'Failed to start run.');
    } finally {
      setStarting(false);
    }
  }

  return (
    <div className="composer">
      <div className="card">
        <label>Title (optional)</label>
        <input className="w-full" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Untitled simulation" />
      </div>

      <div className="card">
        <label>Seed document</label>
        <textarea
          className="w-full seed-textarea"
          rows={10}
          value={seedDocument}
          onChange={(e) => setSeedDocument(e.target.value)}
          placeholder="Paste the source text your agents will reason about…"
          maxLength={20000}
        />
        <div className="row-between">
          <span className="muted">{seedDocument.length} / 20000 characters</span>
          <label className="btn">
            Load .txt/.md file
            <input type="file" accept=".txt,.md" onChange={handleFile} style={{ display: 'none' }} />
          </label>
        </div>
      </div>

      <div className="card">
        <label>Question</label>
        <input className="w-full" value={question} onChange={(e) => setQuestion(e.target.value.slice(0, 500))} placeholder="Will public trust rise this quarter?" />
      </div>

      <div className="card">
        <label>Mode</label>
        <div className="segmented">
          <button className={!live ? 'active' : ''} onClick={() => setLive(false)}>Mock (free)</button>
          <button className={live ? 'active' : ''} onClick={() => setLive(true)}>Live (uses your API keys)</button>
        </div>
        {!live && <p className="muted">Deterministic mock engine — no API calls, no keys needed.</p>}
      </div>

      {live && (
        <div className="card">
          <label className="row">
            <input type="checkbox" checked={sameModel} onChange={(e) => setSameModel(e.target.checked)} />
            Use the same model for all tiers
          </label>
          <div className="model-grid">
            <div>
              <label>Flagship (synthesis)</label>
              <input list="model-presets" value={flagship} onChange={(e) => setFlagship(e.target.value)} />
            </div>
            {!sameModel && (
              <>
                <div>
                  <label>Standard (world bootstrap, memory)</label>
                  <input list="model-presets" value={standard} onChange={(e) => setStandard(e.target.value)} />
                </div>
                <div>
                  <label>Cheap (agent batches)</label>
                  <input list="model-presets" value={cheap} onChange={(e) => setCheap(e.target.value)} />
                </div>
              </>
            )}
          </div>
          <datalist id="model-presets">
            {MODEL_PRESETS.map((m) => <option key={m} value={m} />)}
          </datalist>

          <div className="key-chips">
            {requiredKeys.length === 0 && <span className="muted">No provider selected yet.</span>}
            {requiredKeys.map((k) => (
              <span key={k} className={storedKeys[k] ? 'chip chip-ok' : 'chip chip-missing'}>
                {storedKeys[k] ? '✓' : '✗'} {k}
              </span>
            ))}
          </div>
          <button className="btn" onClick={runDoctor} disabled={doctorLoading || requiredKeys.length === 0}>
            {doctorLoading ? 'Checking…' : 'Run doctor check'}
          </button>
          {doctorResult && (
            <ul className="doctor-results">
              {doctorResult.map((r, i) => (
                <li key={i}>
                  {r.model}: {r.ready ? '✓ ready' : `✗ missing ${r.missing?.join(', ')}`}
                  {r.ping_ms != null && ` — ping ${r.ping_ms}ms`}
                  {r.ping_error && ` — ${r.ping_error}`}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="card">
        <button className="disclosure" onClick={() => setAdvancedOpen(!advancedOpen)}>
          {advancedOpen ? '▾' : '▸'} Advanced
        </button>
        {advancedOpen && (
          <div className="advanced-grid">
            <div><label>Rounds</label><input type="number" min={1} max={12} value={rounds} onChange={(e) => setRounds(+e.target.value)} /></div>
            <div><label>Max agents</label><input type="number" min={2} max={48} value={maxAgents} onChange={(e) => setMaxAgents(+e.target.value)} /></div>
            <div><label>Group size</label><input type="number" min={1} max={16} value={groupSize} onChange={(e) => setGroupSize(+e.target.value)} /></div>
            <div><label>Active agent fraction</label><input type="number" step={0.05} min={0.05} max={1} value={activeAgentFraction} onChange={(e) => setActiveAgentFraction(+e.target.value)} /></div>
            <div><label>Convergence threshold</label><input type="number" min={1} max={10} value={convergenceThreshold} onChange={(e) => setConvergenceThreshold(+e.target.value)} /></div>
            <div><label>Random seed</label><input type="number" value={randomSeed} onChange={(e) => setRandomSeed(+e.target.value)} /></div>
          </div>
        )}
      </div>

      <CostEstimate live={live} rounds={rounds} maxAgents={maxAgents} groupSize={groupSize} activeAgentFraction={activeAgentFraction} models={models} />

      {error && <div className="card error-card">{error}</div>}

      <button className="btn btn-accent w-full btn-run" disabled={!canRun || starting} onClick={handleRun}>
        {starting ? 'Starting…' : 'Run simulation'}
      </button>
      {!canRun && <p className="muted">Add the missing API key(s) above before running live.</p>}
    </div>
  );
}
