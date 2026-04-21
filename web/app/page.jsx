"use client";

import { startTransition, useDeferredValue, useEffect, useState } from "react";
import {
  EXAMPLE_PAYLOAD_TEXT,
  createSimulationSummary,
  hashString,
  normalizeSimulationPayload,
  parseSimulationJson,
  positionForNode,
} from "../lib/simulation.js";

function formatPercent(value) {
  if (!Number.isFinite(value)) {
    return "n/a";
  }

  return `${Math.round(value * 100)}%`;
}

function formatNumber(value, digits = 2) {
  if (!Number.isFinite(value)) {
    return "n/a";
  }

  return value.toFixed(digits);
}

function valueToText(value) {
  if (value === null || value === undefined) {
    return "n/a";
  }

  if (typeof value === "string") {
    return value;
  }

  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : formatNumber(value);
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  if (Array.isArray(value)) {
    return value.map(valueToText).join(", ");
  }

  return JSON.stringify(value);
}

function renderReportEntries(report) {
  const entries = Object.entries(report ?? {}).filter(([, value]) => value !== undefined);

  if (entries.length === 0) {
    return <p className="muted">No prediction report fields detected.</p>;
  }

  return (
    <dl className="stack">
      {entries.map(([key, value]) => (
        <div className="key-value" key={key}>
          <dt>{key.replaceAll("_", " ")}</dt>
          <dd>{valueToText(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function AgentCard({ agent, active, onClick }) {
  const mood = Number.isFinite(agent.mood) ? formatPercent(agent.mood) : "n/a";
  const energy = Number.isFinite(agent.energy) ? formatPercent(agent.energy) : "n/a";
  const attention = Number.isFinite(agent.attention) ? formatPercent(agent.attention) : "n/a";

  return (
    <button
      type="button"
      className={`agent-card ${active ? "active" : ""}`}
      onClick={onClick}
    >
      <div className="agent-card__top">
        <strong>{agent.name}</strong>
        <span>{agent.archetype || "agent"}</span>
      </div>
      <div className="pill-row">
        <span className="pill">Mood {mood}</span>
        <span className="pill">Energy {energy}</span>
        <span className="pill">Attention {attention}</span>
      </div>
    </button>
  );
}

export default function Page() {
  const [payloadText, setPayloadText] = useState(EXAMPLE_PAYLOAD_TEXT);
  const [selectedAgentId, setSelectedAgentId] = useState(null);
  const [search, setSearch] = useState("");
  const deferredSearch = useDeferredValue(search.trim().toLowerCase());

  const parsed = parseSimulationJson(payloadText);
  const normalized = parsed.ok ? normalizeSimulationPayload(parsed.value) : null;
  const summary = normalized ? createSimulationSummary(normalized) : null;

  const visibleAgents = normalized
    ? normalized.agents.filter((agent) => {
        if (!deferredSearch) {
          return true;
        }

        const haystack = [
          agent.id,
          agent.name,
          agent.archetype,
          agent.role,
          ...(agent.tags ?? []),
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();

        return haystack.includes(deferredSearch);
      })
    : [];

  const selectedAgent =
    visibleAgents.find((agent) => agent.id === selectedAgentId) ?? visibleAgents[0] ?? null;

  useEffect(() => {
    if (!normalized || visibleAgents.length === 0) {
      setSelectedAgentId(null);
      return;
    }

    if (!selectedAgentId || !visibleAgents.some((agent) => agent.id === selectedAgentId)) {
      setSelectedAgentId(visibleAgents[0].id);
    }
  }, [normalized, selectedAgentId, visibleAgents]);

  const nodePositions = new Map(
    visibleAgents.map((agent, index) => [
      agent.id,
      positionForNode(agent.id ?? hashString(agent.name), index, visibleAgents.length),
    ]),
  );

  const visibleEdges = normalized
    ? normalized.edges.filter(
        (edge) => nodePositions.has(edge.source) && nodePositions.has(edge.target),
      )
    : [];

  return (
    <main className="shell">
      <section className="hero card">
        <div className="hero__copy">
          <p className="eyebrow">Lean Swarm</p>
          <h1>Post-simulation world inspector</h1>
          <p className="lede">
            Paste the JSON returned by `leanswarm simulate` or the API, then inspect the
            normalized world snapshot, report, and live agent network.
          </p>
        </div>
        <div className="hero__stats">
          <div className="stat">
            <span>Agents</span>
            <strong>{summary?.agentCount ?? 0}</strong>
          </div>
          <div className="stat">
            <span>Edges</span>
            <strong>{summary?.edgeCount ?? 0}</strong>
          </div>
          <div className="stat">
            <span>Ticks</span>
            <strong>{summary?.tickCount ?? 0}</strong>
          </div>
        </div>
      </section>

      <section className="card editor">
        <div className="card__head">
          <div>
            <h2>Simulation payload</h2>
            <p className="muted">
              Accepts `result`, `world_snapshot`, and nested payload wrappers. You can paste raw CLI
              output or API JSON.
            </p>
          </div>
          <div className="actions">
            <button
              type="button"
              className="button"
              onClick={() => {
                startTransition(() => {
                  setPayloadText(EXAMPLE_PAYLOAD_TEXT);
                });
                setSearch("");
                setSelectedAgentId(null);
              }}
            >
              Load example
            </button>
            <button
              type="button"
              className="button button--ghost"
              onClick={() => {
                setPayloadText("");
                setSearch("");
                setSelectedAgentId(null);
              }}
            >
              Clear
            </button>
          </div>
        </div>
        <textarea
          aria-label="Simulation payload"
          className="editor__input"
          spellCheck="false"
          value={payloadText}
          onChange={(event) => setPayloadText(event.target.value)}
        />
        {!parsed.ok ? (
          <div className="error">
            <strong>JSON parse error</strong>
            <p>{parsed.error}</p>
          </div>
        ) : (
          <div className="success">
            <strong>Payload parsed</strong>
            <p>{summary?.title || "Simulation payload recognized."}</p>
          </div>
        )}
      </section>

      {normalized ? (
        <section className="workspace">
          <article className="card world-card">
            <div className="card__head">
              <div>
                <h2>World graph</h2>
                <p className="muted">
                  Click a node to inspect an agent. Edge thickness reflects relationship weight.
                </p>
              </div>
            </div>

            <div className="graph">
              <svg className="graph__edges" viewBox="0 0 100 100" preserveAspectRatio="none">
                {visibleEdges.map((edge, index) => {
                  const source = nodePositions.get(edge.source);
                  const target = nodePositions.get(edge.target);

                  if (!source || !target) {
                    return null;
                  }

                  const selected =
                    selectedAgent && (edge.source === selectedAgent.id || edge.target === selectedAgent.id);

                  return (
                    <line
                      key={`${edge.source}-${edge.target}-${index}`}
                      x1={source.x}
                      y1={source.y}
                      x2={target.x}
                      y2={target.y}
                      className={selected ? "edge edge--active" : "edge"}
                      strokeWidth={Math.max(0.2, edge.weight * 0.25)}
                    />
                  );
                })}
              </svg>

              {visibleAgents.map((agent, index) => {
                const position = nodePositions.get(agent.id);
                if (!position) {
                  return null;
                }

                const active = selectedAgent?.id === agent.id;
                const score = Math.round((agent.energy ?? 0) * 100);

                return (
                  <button
                    key={agent.id}
                    type="button"
                    className={`graph-node ${active ? "graph-node--active" : ""}`}
                    style={{
                      left: `${position.x}%`,
                      top: `${position.y}%`,
                      transform: "translate(-50%, -50%)",
                      animationDelay: `${index * 40}ms`,
                    }}
                    onClick={() => setSelectedAgentId(agent.id)}
                  >
                    <span className="graph-node__label">{agent.name}</span>
                    <span className="graph-node__meta">
                      {agent.archetype || "agent"} · {score}% energy
                    </span>
                  </button>
                );
              })}
            </div>
          </article>

          <aside className="card sidebar">
            <div className="card__head">
              <div>
                <h2>Agents</h2>
                <p className="muted">Filter the current post-simulation population.</p>
              </div>
            </div>

            <label className="field">
              <span>Search</span>
              <input
                type="search"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="name, archetype, role, tag"
              />
            </label>

            <div className="list">
              {visibleAgents.length === 0 ? (
                <p className="muted">No agents matched the current filter.</p>
              ) : (
                visibleAgents.map((agent) => (
                  <AgentCard
                    key={agent.id}
                    agent={agent}
                    active={selectedAgent?.id === agent.id}
                    onClick={() => setSelectedAgentId(agent.id)}
                  />
                ))
              )}
            </div>
          </aside>

          <aside className="card details">
            <div className="card__head">
              <div>
                <h2>World details</h2>
                <p className="muted">Prediction report and focused agent state.</p>
              </div>
            </div>

            <div className="detail-block">
              <h3>Report</h3>
              {renderReportEntries(normalized.report)}
            </div>

            <div className="detail-block">
              <h3>Focused agent</h3>
              {selectedAgent ? (
                <dl className="stack">
                  <div className="key-value">
                    <dt>Name</dt>
                    <dd>{selectedAgent.name}</dd>
                  </div>
                  <div className="key-value">
                    <dt>Archetype</dt>
                    <dd>{selectedAgent.archetype || "n/a"}</dd>
                  </div>
                  <div className="key-value">
                    <dt>Role</dt>
                    <dd>{selectedAgent.role || "n/a"}</dd>
                  </div>
                  <div className="key-value">
                    <dt>Mood</dt>
                    <dd>{formatPercent(selectedAgent.mood)}</dd>
                  </div>
                  <div className="key-value">
                    <dt>Energy</dt>
                    <dd>{formatPercent(selectedAgent.energy)}</dd>
                  </div>
                  <div className="key-value">
                    <dt>Attention</dt>
                    <dd>{formatPercent(selectedAgent.attention)}</dd>
                  </div>
                  <div className="key-value">
                    <dt>Notes</dt>
                    <dd>{selectedAgent.summary || selectedAgent.notes || "n/a"}</dd>
                  </div>
                </dl>
              ) : (
                <p className="muted">No agent selected.</p>
              )}
            </div>

            <div className="detail-block">
              <h3>Recent ticks</h3>
              {normalized.ticks.length === 0 ? (
                <p className="muted">No tick records were detected.</p>
              ) : (
                <div className="tick-list">
                  {normalized.ticks.slice(-6).map((tick) => (
                    <div className="tick" key={tick.id}>
                      <strong>Tick {tick.label}</strong>
                      <span>{tick.summary}</span>
                      <span>
                        mean delta {formatNumber(tick.meanDelta)} ·{" "}
                        {tick.stable ? "stable" : "active"}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="detail-block">
              <h3>Snapshot summary</h3>
              <dl className="stack">
                <div className="key-value">
                  <dt>Topics</dt>
                  <dd>
                    {normalized.topics.length > 0
                      ? normalized.topics.map((topic) => topic.label).join(", ")
                      : "n/a"}
                  </dd>
                </div>
                <div className="key-value">
                  <dt>World hash</dt>
                  <dd>{summary?.worldHash ?? "n/a"}</dd>
                </div>
                <div className="key-value">
                  <dt>Converged</dt>
                  <dd>{normalized.report.converged ? "yes" : "no"}</dd>
                </div>
              </dl>
            </div>
          </aside>
        </section>
      ) : (
        <section className="card empty-state">
          <h2>Paste a simulation payload to begin.</h2>
          <p className="muted">
            The viewer accepts the current Lean Swarm response shape and normalizes common wrappers
            so you can inspect the world without rewriting the JSON.
          </p>
        </section>
      )}
    </main>
  );
}
