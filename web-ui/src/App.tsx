import { useCallback, useEffect, useState } from 'react';
import { BrowserRouter, Link, Route, Routes, useLocation } from 'react-router-dom';
import Composer from './pages/Composer';
import RunPage from './pages/RunPage';
import Gallery from './pages/Gallery';
import AuthPage from './pages/AuthPage';
import KeysModal from './components/KeysModal';
import { api } from './api';

function Shell() {
  const location = useLocation();
  const [me, setMe] = useState<string | null>(null);
  const [myRuns, setMyRuns] = useState<any[]>([]);
  const [keysOpen, setKeysOpen] = useState(false);

  const refreshMe = useCallback(() => {
    api.me().then((r) => setMe(r.email)).catch(() => setMe(null));
  }, []);

  useEffect(() => { refreshMe(); }, [refreshMe]);

  // Refetch the run list on login state change and on navigation, so a freshly
  // saved run appears in the sidebar without a full page reload.
  useEffect(() => {
    if (!me) { setMyRuns([]); return; }
    api.myRuns().then(setMyRuns).catch(() => setMyRuns([]));
  }, [me, location.pathname]);

  async function handleLogout() {
    await api.logout();
    setMe(null);
  }

  function toggleTheme() {
    const root = document.documentElement;
    const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem('leanswarm.theme', next);
  }

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <h1 className="logo">leanswarm</h1>
        <Link to="/" className="btn btn-accent w-full">+ New simulation</Link>
        {me && (
          <div className="sidebar-section">
            <div className="sidebar-heading">My runs</div>
            {myRuns.length === 0 && <div className="muted">No saved runs yet.</div>}
            {myRuns.map((r) => (
              <Link key={r.id} to={`/run/${r.id}`} className="sidebar-run-link">
                <span className="run-title">{r.title || r.question}</span>
                <span className="muted">{new Date(r.created_at).toLocaleDateString()}</span>
              </Link>
            ))}
          </div>
        )}
        <nav className="sidebar-section">
          <Link to="/gallery">Gallery</Link>
        </nav>
        <div className="sidebar-footer">
          <button className="btn theme-toggle" onClick={toggleTheme}>Toggle theme</button>
          <button className="btn" onClick={() => setKeysOpen(true)}>API keys</button>
          {me ? (
            <div className="auth-area">
              <span className="muted">{me}</span>
              <button className="btn" onClick={handleLogout}>Logout</button>
            </div>
          ) : (
            <Link to="/login">Sign in</Link>
          )}
        </div>
      </aside>
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Composer />} />
          <Route path="/run/:id" element={<RunPage />} />
          <Route path="/gallery" element={<Gallery />} />
          <Route path="/gallery/:id" element={<RunPage readOnly />} />
          <Route path="/login" element={<AuthPage type="login" onAuthChange={refreshMe} />} />
          <Route path="/register" element={<AuthPage type="register" onAuthChange={refreshMe} />} />
        </Routes>
      </main>
      {keysOpen && <KeysModal onClose={() => setKeysOpen(false)} />}
    </div>
  );
}

export default function App() {
  useEffect(() => {
    const saved = localStorage.getItem('leanswarm.theme');
    if (saved) document.documentElement.setAttribute('data-theme', saved);
  }, []);
  return (
    <BrowserRouter>
      <Shell />
    </BrowserRouter>
  );
}
