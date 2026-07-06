import { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { api } from '../api';
import ForceGraph from '../components/ForceGraph';
import ChatPanel from '../components/ChatPanel';

type Phase = 'sources' | 'bootstrap' | 'simulation' | 'synthesis';
type PhaseStatus = 'waiting' | 'running' | 'done';

const KIND_COLORS: Record<string, string> = {
  document: '#78716c', entity: '#0f766e', topic: '#2563eb',
  question: '#b45309', sentiment: '#be185d',
};

export default function RunPage({ readOnly = false }: { readOnly?: boolean }) {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const isGallery = readOnly;

  const [phases, setPhases] = useState<Record<Phase, PhaseStatus>>({ sources: 'waiting', bootstrap: 'waiting', simulation: 'waiting', synthesis: 'waiting' });
  const [currentTick, setCurrentTick] = useState(0);
  const [totalRounds, setTotalRounds] = useState(0);
  const [ticks, setTicks] = useState<any[]>([]);
  const [tokensTotal, setTokensTotal] = useState({ prompt: 0, completion: 0 });
  const [llmCalls, setLlmCalls] = useState(0);
  const [costSoFar, setCostSoFar] = useState<number | null>(null);
  const [status, setStatus] = useState<'running' | 'complete' | 'error'>('running');
  const [result, setResult] = useState<any | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [graphView, setGraphView] = useState<'agents' | 'knowledge'>('agents');
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const [reportContent, setReportContent] = useState<{ title: string; sections: Array<{ heading: string; content: string }> } | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [sourceCount, setSourceCount] = useState(0);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;

    const fetcher = isGallery ? api.galleryItem : api.getRun;
    fetcher(id).then((data) => {
      if (cancelled) return;
      if (data.status === 'complete') {
        setStatus('complete');
        setResult(data.result);
        return;
      }
      if (data.status === 'error') {
        setStatus('error');
        setErrorMsg(data.error || 'Run failed.');
        return;
      }
      // Still running: open the SSE stream and replay from the start.
      const es = new EventSource(`/api/runs/${id}/events`);
      eventSourceRef.current = es;
      es.onmessage = (evt) => {
        const parsed = JSON.parse(evt.data);
        handleEvent(parsed);
      };
      es.onerror = () => {
        // Reconnect is handled by the browser automatically for transient drops;
        // if the run already finished the server will have closed the stream cleanly.
      };
    }).catch((err) => {
      setStatus('error');
      setErrorMsg(err.message || 'Run not found.');
    });

    return () => {
      cancelled = true;
      eventSourceRef.current?.close();
    };
  }, [id, isGallery]);

  function handleEvent(evt: any) {
    switch (evt.type) {
      case 'phase':
        setPhases((prev) => ({ ...prev, [evt.phase]: evt.status === 'done' ? 'done' : 'running' }));
        if (evt.phase === 'simulation' && evt.tick) {
          setCurrentTick(evt.tick);
          setTotalRounds(evt.rounds);
        }
        if (evt.phase === 'sources' && evt.status === 'done') {
          setSourceCount(evt.count ?? 0);
        }
        break;
      case 'tick':
        setTicks((prev) => [...prev, evt.record]);
        setTokensTotal({ prompt: evt.prompt_tokens_total, completion: evt.completion_tokens_total });
        setLlmCalls(evt.llm_calls);
        setCostSoFar(evt.cost_usd_so_far);
        break;
      case 'complete':
        setStatus('complete');
        setResult(evt.result);
        setTokensTotal({ prompt: evt.prompt_tokens_total, completion: evt.completion_tokens_total });
        setCostSoFar(evt.cost_usd);
        eventSourceRef.current?.close();
        break;
      case 'error':
        setStatus('error');
        setErrorMsg(evt.message);
        eventSourceRef.current?.close();
        break;
    }
  }

  async function handleSave() {
    if (!id) return;
    try {
      await api.saveRun(id);
      setSaveMsg('Saved to your history.');
    } catch (err: any) {
      setSaveMsg(err.status === 401 ? 'Sign in to save runs.' : err.message);
    }
  }

  async function handlePublish() {
    if (!id) return;
    if (!window.confirm('This makes the full result, including your seed text, publicly visible in the gallery. Publish?')) return;
    try {
      await api.publishRun(id);
      setSaveMsg('Published — visible in the gallery.');
    } catch (err: any) {
      setSaveMsg(err.message);
    }
  }

  function handleDownload() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `leanswarm-run-${id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function handleReport() {
    if (!id) return;
    setReportLoading(true);
    try {
      const res = await api.report(id, { live: false });
      setReportContent(res);
    } catch (_err) {
      setSaveMsg('Could not generate report.');
    } finally {
      setReportLoading(false);
    }
  }

  function openInComposer() {
    const reqData = result?.request ?? {};
    sessionStorage.setItem('leanswarm.rerun.v1', JSON.stringify(reqData));
    navigate('/');
  }

  if (status === 'error') {
    return (
      <div className="card error-card">
        <h2>Run failed</h2>
        <p>{errorMsg}</p>
        {errorMsg?.includes('expired') && <p className="muted">Results are kept for 2 hours unless saved or published.</p>}
      </div>
    );
  }

  if (status === 'running') {
    return (
      <div className="run-live">
        <div className="phase-timeline">
          {(['sources', 'bootstrap', 'simulation', 'synthesis'] as Phase[]).map((p) => {
            if (p === 'sources' && phases.sources === 'waiting') return null;
            return (
            <div key={p} className={`phase-chip phase-${phases[p]}`}>
              {p === 'simulation' && phases[p] === 'running' ? `Simulation (tick ${currentTick}/${totalRounds})` : p === 'sources' && phases[p] === 'done' ? `Sources (${sourceCount})` : p[0].toUpperCase() + p.slice(1)}
              {phases[p] === 'done' && ' ✓'}
            </div>
          )})}
        </div>
        <div className="cost-ticker card">
          {llmCalls} calls · {tokensTotal.prompt + tokensTotal.completion} tokens
          {costSoFar != null && ` · $${costSoFar.toFixed(4)}`}
        </div>
        <div className="tick-feed">
          {ticks.map((t) => (
            <div key={t.tick} className="card tick-card">
              <strong>Tick {t.tick}</strong>
              <ul>{t.events.map((e: string, i: number) => <li key={i}>{e}</li>)}</ul>
              <div className="stat-row muted">
                mean_delta {t.mean_delta.toFixed(3)} · activation {(t.activation_fraction * 100).toFixed(0)}% · convergence {t.convergence_score.toFixed(2)}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // status === 'complete'
  if (!result) return <div className="card">Loading result…</div>;
  const { report, world, ticks: resultTicks } = result;

  const agentNodes = world.agents.map((a: any) => ({
    id: a.id, label: a.name, color: '#0f766e',
    radius: 6 + Object.keys(a.relationships).length,
  }));
  const agentEdges = world.edges.map((e: any) => ({ source: e.source, target: e.target, weight: e.weight }));

  const kgNodes = (world.graph?.nodes || []).map((n: any) => ({
    id: n.id, label: n.label, color: KIND_COLORS[n.kind] || '#6b7280', radius: 4 + n.weight * 8,
  }));
  const kgEdges = (world.graph?.edges || []).map((e: any) => ({ source: e.source, target: e.target, weight: e.weight }));

  return (
    <>
    <div className="run-result">
      {result.request?.retrieved_sources?.length > 0 && (
        <div className="card">
          <h3>Sources</h3>
          {result.request.retrieved_sources.map((s: any, i: number) => (
            <div key={i} className="source-entry">
              <a href={s.url} target="_blank" rel="noopener noreferrer">{s.title || s.url}</a>
              <span className="muted"> · {s.chars} chars · <span className="chip chip-ok">{s.via}</span></span>
            </div>
          ))}
        </div>
      )}
      <div className="card prediction-card">
        <h2>{report.prediction}</h2>
        <div className="confidence-meter">
          <div className="confidence-fill" style={{ width: `${report.confidence * 100}%` }} />
        </div>
        <span className="muted">{Math.round(report.confidence * 100)}% confidence{report.converged ? ' · converged' : ''}</span>
        <ul>{report.rationale.map((r: string, i: number) => <li key={i}>{r}</li>)}</ul>
      </div>

      <div className="stat-row card">
        <span>{report.tick_count} ticks</span>
        <span>{report.llm_calls} LLM calls</span>
        <span>{(report.cache_hit_rate * 100).toFixed(0)}% cache hits</span>
        {costSoFar != null && <span>${costSoFar.toFixed(4)}</span>}
      </div>

      <div className="agents-grid">
        {world.agents.map((a: any) => (
          <div key={a.id} className="card agent-card">
            <strong>{a.name}</strong>
            <div className="muted">{a.archetype}</div>
            {a.persona && <div className="muted" style={{ fontSize: '0.85rem', marginTop: '0.25rem' }}>{a.persona}</div>}
            {a.stance && <div className="muted" style={{ fontSize: '0.85rem', fontStyle: 'italic', marginTop: '0.15rem' }}>{a.stance}</div>}
            <div className="bar-row"><span>mood</span><div className="bar"><div style={{ width: `${a.mood * 100}%` }} /></div></div>
            <div className="bar-row"><span>energy</span><div className="bar"><div style={{ width: `${a.energy * 100}%` }} /></div></div>
            <div className="bar-row"><span>attention</span><div className="bar"><div style={{ width: `${a.attention * 100}%` }} /></div></div>
            {a.memory.working.slice(0, 3).map((m: string, i: number) => <div key={i} className="muted memory-line">{m}</div>)}
          </div>
        ))}
      </div>

      <div className="card">
        {result.world?.profile?.extraction_source === 'llm' && (
          <span className="extraction-badge">LLM-extracted world</span>
        )}
        <div className="segmented">
          <button className={graphView === 'agents' ? 'active' : ''} onClick={() => setGraphView('agents')}>Agent network</button>
          <button className={graphView === 'knowledge' ? 'active' : ''} onClick={() => setGraphView('knowledge')}>Knowledge graph</button>
        </div>
        {graphView === 'agents' ? (
          <ForceGraph nodes={agentNodes} edges={agentEdges} />
        ) : (
          <ForceGraph nodes={kgNodes} edges={kgEdges} legend={Object.entries(KIND_COLORS).map(([label, color]) => ({ label, color }))} />
        )}
      </div>

      <details className="card">
        <summary>Ticks ({resultTicks.length})</summary>
        {resultTicks.map((t: any) => (
          <div key={t.tick} className="tick-detail">
            <strong>Tick {t.tick}</strong>
            <ul>{t.events.map((e: string, i: number) => <li key={i}>{e}</li>)}</ul>
          </div>
        ))}
      </details>

      {!isGallery && (
        <div className="action-bar">
          <button className="btn" onClick={handleDownload}>Download JSON</button>
          <button className="btn" onClick={handleSave}>Save</button>
          <button className="btn btn-accent" onClick={handlePublish}>Publish to gallery</button>
          <button className="btn" onClick={handleReport} disabled={reportLoading}>{reportLoading ? 'Generating…' : 'Generate full report'}</button>
          <button className="btn" onClick={openInComposer}>Open in composer</button>
          {saveMsg && <span className="muted">{saveMsg}</span>}
        </div>
      )}
      {isGallery && (
        <div className="action-bar">
          <button className="btn" onClick={handleDownload}>Download JSON</button>
          <button className="btn" onClick={handleReport} disabled={reportLoading}>{reportLoading ? 'Generating…' : 'Generate full report'}</button>
          <button className="btn" onClick={openInComposer}>Open in composer</button>
        </div>
      )}
      {reportContent && (
        <div className="card report-display">
          <h2>{reportContent.title}</h2>
          {reportContent.sections.map((s, i) => (
            <div key={i}>
              <h3>{s.heading}</h3>
              <p>{s.content}</p>
            </div>
          ))}
          <button className="btn" onClick={() => {
            const md = `# ${reportContent.title}\n\n${reportContent.sections.map((s: { heading: string; content: string }) => `## ${s.heading}\n\n${s.content}`).join('\n\n')}`;
            const blob = new Blob([md], { type: 'text/markdown' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = 'report.md'; a.click(); URL.revokeObjectURL(url);
          }}>Download .md</button>
        </div>
      )}
    </div>
    <ChatPanel runId={id || ''} agents={(result?.world?.agents || []).map((a: any) => ({ id: a.id, name: a.name }))} />
    </>
  );
}
