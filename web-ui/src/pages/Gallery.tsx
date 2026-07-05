import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';

export default function Gallery() {
  const [items, setItems] = useState<any[]>([]);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [loading, setLoading] = useState(false);
  const LIMIT = 20;

  async function load(nextOffset: number) {
    setLoading(true);
    try {
      const data = await api.gallery(LIMIT, nextOffset);
      setItems((prev) => (nextOffset === 0 ? data : [...prev, ...data]));
      setHasMore(data.length === LIMIT);
      setOffset(nextOffset + data.length);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(0); }, []);

  if (items.length === 0 && !loading) {
    return <div className="card">No public simulations yet. Be the first to publish one.</div>;
  }

  return (
    <div>
      <div className="gallery-grid">
        {items.map((item) => (
          <Link key={item.id} to={`/gallery/${item.id}`} className="card gallery-card">
            <h3>{item.title || item.question}</h3>
            <p>{item.prediction}</p>
            <div className="confidence-meter small">
              <div className="confidence-fill" style={{ width: `${item.confidence * 100}%` }} />
            </div>
            <div className="muted gallery-meta">
              {Object.values(item.models_json || {}).filter((v) => typeof v === 'string').join(' · ')}
              {item.cost_usd != null && ` · $${item.cost_usd.toFixed(4)}`}
              {' · '}{new Date(item.created_at).toLocaleDateString()}
            </div>
          </Link>
        ))}
      </div>
      {hasMore && (
        <button className="btn" disabled={loading} onClick={() => load(offset)}>
          {loading ? 'Loading…' : 'Load more'}
        </button>
      )}
    </div>
  );
}
