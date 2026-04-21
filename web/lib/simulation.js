const EXAMPLE_PAYLOAD = {
  seed_document:
    "A national policy announcement, mixed press coverage, and rising social chatter are shifting public opinion.",
  question: "Will public trust improve over the next month?",
  prediction_report: {
    answer: "Likely modestly higher",
    confidence: 0.68,
    rationale: [
      "The seed document signals steady institutional support.",
      "Narrative pressure remains mixed but not strongly negative.",
    ],
    risks: ["A new controversy could reverse the trend."],
    converged: true,
  },
  world_snapshot: {
    world_name: "Public trust",
    summary: "A moderately optimistic world with a few skeptical nodes.",
    topics: [
      { id: "topic_public_trust", label: "public trust" },
      { id: "topic_coverage", label: "media coverage" },
      { id: "topic_policy", label: "policy response" },
    ],
    agents: [
      {
        id: "agent_amara",
        name: "Amara",
        archetype: "Institutionalist",
        role: "anchor",
        mood: 0.74,
        energy: 0.61,
        attention: 0.82,
        tags: ["optimistic", "networked"],
        summary: "Treats the announcement as a stabilizing event.",
      },
      {
        id: "agent_jon",
        name: "Jon",
        archetype: "Skeptic",
        role: "watcher",
        mood: 0.42,
        energy: 0.55,
        attention: 0.91,
        tags: ["critical", "media"],
        summary: "Waits for evidence before changing stance.",
      },
      {
        id: "agent_ren",
        name: "Ren",
        archetype: "Coalition Builder",
        role: "bridge",
        mood: 0.64,
        energy: 0.71,
        attention: 0.76,
        tags: ["mediator", "connective"],
        summary: "Tries to align opposing sides around a compromise.",
      },
      {
        id: "agent_lina",
        name: "Lina",
        archetype: "Crisis Monitor",
        role: "sentinel",
        mood: 0.51,
        energy: 0.68,
        attention: 0.88,
        tags: ["risk", "fast-reacting"],
        summary: "Focuses on downside risks and second-order effects.",
      },
    ],
    relationships: [
      { source: "agent_amara", target: "agent_ren", weight: 0.82, label: "support" },
      { source: "agent_ren", target: "agent_jon", weight: 0.44, label: "debate" },
      { source: "agent_lina", target: "agent_jon", weight: 0.57, label: "watch" },
      { source: "agent_amara", target: "agent_lina", weight: 0.36, label: "pressure" },
    ],
    ticks: [
      { tick: 1, mean_delta: 0.11, stable: false, summary: "Agents reacted to the announcement." },
      { tick: 2, mean_delta: 0.06, stable: false, summary: "Some opinions hardened." },
      { tick: 3, mean_delta: 0.02, stable: true, summary: "World began to settle." },
    ],
  },
};

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function objectValue(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : null;
}

function unwrapRoot(value) {
  const root = objectValue(value) ?? {};
  return objectValue(root.result) ?? objectValue(root.simulation_result) ?? root;
}

function pickFirstObject(source, keys) {
  for (const key of keys) {
    const candidate = objectValue(source?.[key]);
    if (candidate) {
      return candidate;
    }
  }

  return null;
}

function pickFirstArray(source, keys) {
  for (const key of keys) {
    const candidate = source?.[key];
    if (Array.isArray(candidate)) {
      return candidate;
    }
  }

  return [];
}

function asNumber(value, fallback = NaN) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function asText(value, fallback = "") {
  if (typeof value === "string") {
    return value.trim();
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  return fallback;
}

function uniqueBy(items, keyFn) {
  const seen = new Set();
  const output = [];

  for (const item of items) {
    const key = keyFn(item);
    if (!key || seen.has(key)) {
      continue;
    }

    seen.add(key);
    output.push(item);
  }

  return output;
}

export const EXAMPLE_PAYLOAD_TEXT = JSON.stringify(EXAMPLE_PAYLOAD, null, 2);

export function parseSimulationJson(text) {
  if (!text.trim()) {
    return {
      ok: false,
      error: "Paste a simulation JSON payload to inspect it.",
    };
  }

  try {
    return { ok: true, value: JSON.parse(text) };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : "Unknown parse failure.",
    };
  }
}

function normalizeAgent(entry, index) {
  const source = objectValue(entry) ?? {};
  const id =
    asText(source.id, "") ||
    asText(source.agent_id, "") ||
    asText(source.name, "") ||
    `agent_${index + 1}`;

  const state = objectValue(source.state) ?? objectValue(source.metrics) ?? {};
  const archetype =
    asText(source.archetype, "") || asText(source.persona, "") || asText(source.cluster, "");

  const energy = asNumber(source.energy ?? state.energy, NaN);
  const mood = asNumber(source.mood ?? state.mood, NaN);
  const attention = asNumber(source.attention ?? state.attention, NaN);

  return {
    id,
    name: asText(source.name, id),
    archetype,
    role: asText(source.role, ""),
    mood,
    energy,
    attention,
    tags: safeArray(source.tags).map((tag) => asText(tag, "")).filter(Boolean),
    summary:
      asText(source.summary, "") ||
      asText(source.note, "") ||
      asText(state.summary, "") ||
      asText(state.status, ""),
    notes: asText(source.notes, ""),
    raw: source,
  };
}

function normalizeTopic(entry, index) {
  const source = objectValue(entry) ?? {};
  const label =
    asText(source.label, "") ||
    asText(source.topic, "") ||
    asText(source.name, "") ||
    `topic-${index + 1}`;

  return {
    id: asText(source.id, "") || label,
    label,
    weight: asNumber(source.weight ?? source.score, NaN),
  };
}

function normalizeEdge(entry, index) {
  const source = objectValue(entry) ?? {};
  const from =
    asText(source.source, "") ||
    asText(source.from, "") ||
    asText(source.a, "") ||
    asText(source.origin, "");
  const to =
    asText(source.target, "") ||
    asText(source.to, "") ||
    asText(source.b, "") ||
    asText(source.destination, "");

  return {
    id: asText(source.id, "") || `${from}-${to}-${index}`,
    source: from,
    target: to,
    weight: asNumber(source.weight ?? source.strength ?? source.score, 0.25),
    label: asText(source.label, "") || asText(source.type, ""),
  };
}

function normalizeTick(entry, index) {
  const source = objectValue(entry) ?? {};
  const id = asText(source.id, "") || `tick-${index + 1}`;
  const label = asText(source.tick, "") || asText(source.index, "") || String(index + 1);

  return {
    id,
    label,
    meanDelta: asNumber(source.mean_delta ?? source.meanDelta ?? source.delta, NaN),
    stable: Boolean(source.stable ?? source.settled),
    converged: Boolean(source.converged),
    summary: asText(source.summary, "") || asText(source.notes, ""),
  };
}

export function normalizeSimulationPayload(raw) {
  const root = unwrapRoot(raw);
  const worldSnapshot =
    pickFirstObject(root, ["world_snapshot", "world", "snapshot", "post_simulation_world"]) ?? {};
  const report =
    pickFirstObject(root, ["prediction_report", "report", "summary", "prediction"]) ?? {};
  const rootAgents = pickFirstArray(root, ["agents", "agent_states"]);
  const snapshotAgents = pickFirstArray(worldSnapshot, ["agents", "agent_states"]);
  const graphNodes = pickFirstArray(worldSnapshot.world_graph, ["nodes"]);
  const agents = uniqueBy(
    [...snapshotAgents, ...rootAgents, ...graphNodes].map(normalizeAgent),
    (agent) => agent.id,
  );

  const graphEdges = pickFirstArray(worldSnapshot.world_graph, ["edges"]);
  const snapshotEdges = pickFirstArray(worldSnapshot, ["relationships", "edges"]);
  const edges = [...snapshotEdges, ...graphEdges].map(normalizeEdge).filter((edge) => edge.source && edge.target);

  const tickEntries = [
    ...pickFirstArray(root, ["ticks", "tick_log", "tickLog"]),
    ...pickFirstArray(worldSnapshot, ["ticks", "tick_log", "tickLog"]),
  ];
  const ticks = tickEntries.map(normalizeTick);

  const topics = uniqueBy(
    [
      ...pickFirstArray(worldSnapshot, ["topics"]),
      ...pickFirstArray(root, ["topics"]),
    ].map(normalizeTopic),
    (topic) => topic.id,
  );

  return {
    raw: root,
    report,
    world: worldSnapshot,
    agents,
    edges,
    ticks,
    topics,
  };
}

export function createSimulationSummary(snapshot) {
  const root = objectValue(snapshot?.raw) ?? {};
  const world = objectValue(snapshot?.world) ?? {};
  const report = objectValue(snapshot?.report) ?? {};
  const agents = Array.isArray(snapshot?.agents) ? snapshot.agents : [];
  const title =
    asText(report.answer, "") ||
    asText(world.summary, "") ||
    asText(root.question, "") ||
    asText(world.world_name, "") ||
    "Simulation";

  const worldHash = hashString(
    JSON.stringify({
      question: asText(root.question, ""),
      seed: asText(root.seed_document, ""),
      agentCount: snapshot?.agents?.length ?? 0,
      edgeCount: snapshot?.edges?.length ?? 0,
      report: title,
    }),
  ).toString(16);

  const activeAgents = agents.filter((agent) =>
    Number.isFinite(agent.attention)
      ? agent.attention >= 0.6
      : Number.isFinite(agent.energy)
        ? agent.energy >= 0.6
        : false,
  ).length;

  return {
    title,
    worldHash,
    agentCount: agents.length,
    edgeCount: snapshot?.edges?.length ?? 0,
    tickCount: snapshot?.ticks?.length ?? 0,
    topicCount: snapshot?.topics?.length ?? 0,
    activeAgents,
  };
}

export function hashString(value) {
  const text = String(value ?? "");
  let hash = 0x811c9dc5;

  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }

  return hash >>> 0;
}

export function positionForNode(id, index, total) {
  const hash = hashString(id);
  const totalCount = Math.max(total, 1);
  const ring = 28 + ((hash >> 3) % 14);
  const angle = ((index + (hash % totalCount) / Math.max(totalCount, 1)) / totalCount) * Math.PI * 2;
  const x = 50 + Math.cos(angle) * ring;
  const y = 50 + Math.sin(angle) * ring;

  return {
    x: Math.max(12, Math.min(88, x)),
    y: Math.max(12, Math.min(88, y)),
  };
}
