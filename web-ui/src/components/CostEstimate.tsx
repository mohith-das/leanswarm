import { useEffect, useState } from 'react';
import { api } from '../api';

interface Props {
  live: boolean;
  rounds: number;
  maxAgents: number;
  groupSize: number;
  activeAgentFraction: number;
  models: { flagship: string; standard: string; cheap: string };
}

export default function CostEstimate({ live, rounds, maxAgents, groupSize, activeAgentFraction, models }: Props) {
  const [estimate, setEstimate] = useState<Awaited<ReturnType<typeof api.estimate>> | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!live) { setEstimate(null); return; }
    setLoading(true);
    const handle = setTimeout(() => {
      api.estimate({ rounds, max_agents: maxAgents, group_size: groupSize, active_agent_fraction: activeAgentFraction, models })
        .then(setEstimate)
        .catch(() => setEstimate(null))
        .finally(() => setLoading(false));
    }, 400);
    return () => clearTimeout(handle);
  }, [live, rounds, maxAgents, groupSize, activeAgentFraction, models.flagship, models.standard, models.cheap]);

  if (!live) {
    return <div className="card cost-estimate">Mock mode — free, deterministic, no API calls.</div>;
  }
  if (loading || !estimate) {
    return <div className="card cost-estimate">Estimating cost…</div>;
  }
  const { calls_min, calls_max, cost_min_usd, cost_max_usd, unpriced_models } = estimate;
  return (
    <div className="card cost-estimate">
      <div>~{calls_min}–{calls_max} LLM calls</div>
      {cost_min_usd != null && cost_max_usd != null ? (
        <div className="cost-figure">${cost_min_usd.toFixed(4)} – ${cost_max_usd.toFixed(4)}</div>
      ) : (
        <div className="muted">$ unavailable for {unpriced_models.join(', ')} (tokens only)</div>
      )}
    </div>
  );
}
