import { useState, type ChangeEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import { collectCredentials, getOverrides, getAvailableProviders, PROVIDER_MODELS, PROVIDER_LABELS } from '../keys';
import CostEstimate from '../components/CostEstimate';

function ModelDatalist({ id, provider }: { id: string; provider: string }) {
  const suggestions = PROVIDER_MODELS[provider] || [];
  return (
    <datalist id={id}>
      {suggestions.map((m) => <option key={m} value={m} />)}
    </datalist>
  );
}

export default function Composer() {
  const navigate = useNavigate();
  const [title, setTitle] = useState('');
  const [seedDocument, setSeedDocument] = useState('');
  const [question, setQuestion] = useState('');
  const [live, setLive] = useState(false);
  const [sameModel, setSameModel] = useState(true);
  const [provider, setProvider] = useState('');
  const [model, setModel] = useState('');
  const [flagshipProvider, setFlagshipProvider] = useState('');
  const [standardProvider, setStandardProvider] = useState('');
  const [cheapProvider, setCheapProvider] = useState('');
  const [flagship, setFlagship] = useState('');
  const [standard, setStandard] = useState('');
  const [cheap, setCheap] = useState('');
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

  const availableProviders = getAvailableProviders();
  const hasKeys = availableProviders.length > 0;

  const effProvider = availableProviders.includes(provider) ? provider : (availableProviders[0] || '');
  const effFlagshipProvider = availableProviders.includes(flagshipProvider) ? flagshipProvider : (availableProviders[0] || '');
  const effStandardProvider = availableProviders.includes(standardProvider) ? standardProvider : (availableProviders[0] || '');
  const effCheapProvider = availableProviders.includes(cheapProvider) ? cheapProvider : (availableProviders[0] || '');

  const effModel = model || (PROVIDER_MODELS[effProvider]?.[0] || '');
  const effFlagship = flagship || (PROVIDER_MODELS[effFlagshipProvider]?.[0] || '');
  const effStandard = standard || (PROVIDER_MODELS[effStandardProvider]?.[0] || '');
  const effCheap = cheap || (PROVIDER_MODELS[effCheapProvider]?.[0] || '');

  const models = sameModel
    ? { flagship: effModel, standard: effModel, cheap: effModel }
    : { flagship: effFlagship, standard: effStandard, cheap: effCheap };
  const modelList = Object.values(models);
  const canRun = !live || (hasKeys && modelList.every((m) => m.length > 0));

  function handleProviderChange(newProvider: string, setProv: (v: string) => void, setMod: (v: string) => void) {
    setProv(newProvider);
    const suggestions = PROVIDER_MODELS[newProvider] || [];
    setMod(suggestions[0] || '');
  }

  function handleSameModelChange(checked: boolean) {
    if (!checked && sameModel) {
      setFlagshipProvider(effProvider);
      setStandardProvider(effProvider);
      setCheapProvider(effProvider);
      setFlagship(effModel);
      setStandard(effModel);
      setCheap(effModel);
    }
    setSameModel(checked);
  }

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
          <button
            className={live ? 'active' : ''}
            onClick={() => hasKeys && setLive(true)}
            disabled={!hasKeys}
          >
            Live (uses your API keys)
          </button>
        </div>
        {!live && <p className="muted">Deterministic mock engine — no API calls, no keys needed.</p>}
        {live && !hasKeys && (
          <p className="muted no-keys-message">No API keys set. Open the API keys panel (sidebar) to add keys for live mode.</p>
        )}
      </div>

      {live && hasKeys && (
        <div className="card">
          <label className="row">
            <input type="checkbox" checked={sameModel} onChange={(e) => handleSameModelChange(e.target.checked)} />
            Use the same model for all tiers
          </label>

          {sameModel ? (
            <div className="model-grid two-col">
              <div>
                <label>Provider</label>
                <select className="provider-select" value={effProvider} onChange={(e) => handleProviderChange(e.target.value, setProvider, setModel)}>
                  {availableProviders.map((p) => <option key={p} value={p}>{PROVIDER_LABELS[p] || p}</option>)}
                </select>
              </div>
              <div>
                <label>Model</label>
                <input list="model-presets-same" value={effModel} onChange={(e) => setModel(e.target.value)} placeholder="model name" />
                <ModelDatalist id="model-presets-same" provider={effProvider} />
              </div>
            </div>
          ) : (
            <div className="tier-grid">
              <div className="tier-block">
                <div className="tier-label">Flagship (synthesis)</div>
                <label>Provider</label>
                <select className="provider-select" value={effFlagshipProvider} onChange={(e) => handleProviderChange(e.target.value, setFlagshipProvider, setFlagship)}>
                  {availableProviders.map((p) => <option key={p} value={p}>{PROVIDER_LABELS[p] || p}</option>)}
                </select>
                <label>Model</label>
                <input list="model-presets-flagship" value={effFlagship} onChange={(e) => setFlagship(e.target.value)} placeholder="model name" />
                <ModelDatalist id="model-presets-flagship" provider={effFlagshipProvider} />
              </div>
              <div className="tier-block">
                <div className="tier-label">Standard (world bootstrap, memory)</div>
                <label>Provider</label>
                <select className="provider-select" value={effStandardProvider} onChange={(e) => handleProviderChange(e.target.value, setStandardProvider, setStandard)}>
                  {availableProviders.map((p) => <option key={p} value={p}>{PROVIDER_LABELS[p] || p}</option>)}
                </select>
                <label>Model</label>
                <input list="model-presets-standard" value={effStandard} onChange={(e) => setStandard(e.target.value)} placeholder="model name" />
                <ModelDatalist id="model-presets-standard" provider={effStandardProvider} />
              </div>
              <div className="tier-block">
                <div className="tier-label">Cheap (agent batches)</div>
                <label>Provider</label>
                <select className="provider-select" value={effCheapProvider} onChange={(e) => handleProviderChange(e.target.value, setCheapProvider, setCheap)}>
                  {availableProviders.map((p) => <option key={p} value={p}>{PROVIDER_LABELS[p] || p}</option>)}
                </select>
                <label>Model</label>
                <input list="model-presets-cheap" value={effCheap} onChange={(e) => setCheap(e.target.value)} placeholder="model name" />
                <ModelDatalist id="model-presets-cheap" provider={effCheapProvider} />
              </div>
            </div>
          )}

          <button className="btn" onClick={runDoctor} disabled={doctorLoading}>
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

      <CostEstimate live={live} rounds={rounds} maxAgents={maxAgents} groupSize={groupSize} activeAgentFraction={activeAgentFraction} models={models} seedChars={seedDocument.length} />

      {error && <div className="card error-card">{error}</div>}

      <button className="btn btn-accent w-full btn-run" disabled={!canRun || starting} onClick={handleRun}>
        {starting ? 'Starting…' : 'Run simulation'}
      </button>
      {!canRun && live && <p className="muted">Add API keys and select a model before running live.</p>}
    </div>
  );
}
