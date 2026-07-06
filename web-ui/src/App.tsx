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
        <Link to="/" className="logo-link">
          <h1 className="logo">leanswarm</h1>
        </Link>
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
          <a
            className="github-badge"
            href="https://github.com/mohith-das/leanswarm"
            target="_blank"
            rel="noopener noreferrer"
            title="Audit the source code on GitHub"
          >
            <svg viewBox="0 0 16 16" width="22" height="22" fill="currentColor" aria-hidden="true">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
            </svg>
            <span className="github-badge-text">
              <strong>Open source</strong>
              <span>Audit on GitHub</span>
            </span>
          </a>
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
