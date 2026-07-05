import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import Composer from './pages/Composer';
import RunPage from './pages/RunPage';
import Gallery from './pages/Gallery';
import AuthPage from './pages/AuthPage';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <aside className="sidebar">
          <h1 className="logo">leanswarm</h1>
          <Link to="/" className="btn btn-accent w-full">+ New simulation</Link>
          <nav>
            <Link to="/gallery">Gallery</Link>
            <Link to="/login">Login / Register</Link>
          </nav>
        </aside>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Composer />} />
            <Route path="/run/:id" element={<RunPage />} />
            <Route path="/gallery" element={<Gallery />} />
            <Route path="/gallery/:id" element={<RunPage />} />
            <Route path="/login" element={<AuthPage type="login" />} />
            <Route path="/register" element={<AuthPage type="register" />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
