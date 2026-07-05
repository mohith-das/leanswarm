import { useEffect, useRef, useState } from 'react';
import { forceSimulation, forceLink, forceManyBody, forceCenter, forceCollide } from 'd3-force';

export interface GraphNode {
  id: string;
  label: string;
  color: string;
  radius: number;
}
export interface GraphEdge {
  source: string;
  target: string;
  weight: number; // 0..1, drives opacity
}

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  height?: number;
  legend?: Array<{ label: string; color: string }>;
}

interface SimNode extends GraphNode {
  x: number; y: number;
}

export default function ForceGraph({ nodes, edges, height = 480, legend }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const [positioned, setPositioned] = useState<SimNode[]>([]);

  useEffect(() => {
    if (containerRef.current) setWidth(containerRef.current.clientWidth || 800);
  }, []);

  useEffect(() => {
    if (nodes.length === 0) return;
    const simNodes: SimNode[] = nodes.map((n) => ({ ...n, x: width / 2, y: height / 2 }));
    const simEdges = edges.map((e) => ({ ...e }));
    const sim = forceSimulation(simNodes as any)
      .force('link', forceLink(simEdges as any).id((d: any) => d.id).distance(70))
      .force('charge', forceManyBody().strength(-140))
      .force('center', forceCenter(width / 2, height / 2))
      .force('collide', forceCollide().radius((d: any) => d.radius + 4))
      .stop();
    for (let i = 0; i < 300; i++) sim.tick();
    setPositioned(simNodes);
  }, [nodes, edges, width, height]);

  const nodeById = new Map(positioned.map((n) => [n.id, n]));

  return (
    <div ref={containerRef} style={{ position: 'relative' }}>
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
        <g stroke="var(--border)" strokeWidth={1}>
          {edges.map((e, i) => {
            const s = nodeById.get(e.source);
            const t = nodeById.get(e.target);
            if (!s || !t) return null;
            return (
              <line
                key={i}
                x1={s.x} y1={s.y} x2={t.x} y2={t.y}
                strokeOpacity={Math.max(0.1, Math.min(1, e.weight))}
              />
            );
          })}
        </g>
        <g>
          {positioned.map((n) => (
            <g key={n.id} transform={`translate(${n.x},${n.y})`}>
              <circle r={n.radius} fill={n.color} />
              <text x={n.radius + 4} y={4} fontSize={11} fill="var(--text)">{n.label}</text>
            </g>
          ))}
        </g>
      </svg>
      {legend && (
        <div style={{ position: 'absolute', bottom: 8, left: 8, background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, padding: '0.5rem 0.75rem', fontSize: 12 }}>
          {legend.map((l) => (
            <div key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
              <span style={{ width: 10, height: 10, borderRadius: 5, background: l.color, display: 'inline-block' }} />
              {l.label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
