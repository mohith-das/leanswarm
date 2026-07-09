import { useEffect, useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';

export default function Gallery({ me }: { me?: boolean }) {
  const [tab, setTab] = useState<'public' | 'private'>('public');
  const [items, setItems] = useState<any[]>([]);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');
  const LIMIT = 20;

  async function load(nextOffset: number, currentTab: 'public' | 'private') {
    setLoading(true);
    try {
      if (currentTab === 'public') {
        const data = await api.gallery(LIMIT, nextOffset);
        setItems((prev) => (nextOffset === 0 ? data : [...prev, ...data]));
        setHasMore(data.length === LIMIT);
        setOffset(nextOffset + data.length);
      } else {
        const data = await api.myRuns();
        setItems(data);
        setHasMore(false);
        setOffset(0);
      }
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { 
    setItems([]);
    load(0, tab); 
  }, [tab]);

  useEffect(() => {
    if (!me && tab === 'private') setTab('public');
  }, [me, tab]);

  const filteredItems = useMemo(() => {
    if (!search) return items;
    const lowerSearch = search.toLowerCase();
    return items.filter(item => 
      (item.title || '').toLowerCase().includes(lowerSearch) || 
      (item.question || '').toLowerCase().includes(lowerSearch) ||
      (item.prediction || '').toLowerCase().includes(lowerSearch)
    );
  }, [items, search]);

  return (
    <div>
      {me && (
        <div className="segmented" style={{ maxWidth: '300px', margin: '0 auto 1.5rem auto' }}>
          <button className={tab === 'public' ? 'active' : ''} onClick={() => setTab('public')}>Public Gallery</button>
          <button className={tab === 'private' ? 'active' : ''} onClick={() => setTab('private')}>Personal Gallery</button>
        </div>
      )}

      <input 
        type="text" 
        placeholder="Search simulations..." 
        value={search} 
        onChange={(e) => setSearch(e.target.value)} 
        style={{ marginBottom: '1.5rem' }}
      />

      {items.length === 0 && !loading && (
        <div className="card">
          {tab === 'public' ? 'No public simulations yet. Be the first to publish one.' : 'No personal simulations yet.'}
        </div>
      )}

      <div className="gallery-grid">
        {filteredItems.map((item) => (
          <Link key={item.id} to={tab === 'private' ? `/run/${item.id}` : `/gallery/${item.id}`} className="card gallery-card">
            <h3>{item.title || item.question}</h3>
            <p>{item.prediction}</p>
            {item.confidence != null && (
              <div className="confidence-meter small">
                <div className="confidence-fill" style={{ width: `${item.confidence * 100}%` }} />
              </div>
            )}
            <div className="muted gallery-meta">
              {Object.values(item.models_json || {}).filter((v) => typeof v === 'string').join(' · ')}
              {item.cost_usd != null && ` · $${item.cost_usd.toFixed(4)}`}
              {' · '}{new Date(item.created_at).toLocaleDateString()}
            </div>
          </Link>
        ))}
      </div>
      {hasMore && tab === 'public' && !search && (
        <button className="btn" disabled={loading} onClick={() => load(offset, tab)}>
          {loading ? 'Loading…' : 'Load more'}
        </button>
      )}
    </div>
  );
}
