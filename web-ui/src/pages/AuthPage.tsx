import { useState, type FormEvent } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { api } from '../api';

export default function AuthPage({ type, onAuthChange }: { type: 'login' | 'register'; onAuthChange?: () => void }) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    if (type === 'register' && password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }
    setSubmitting(true);
    try {
      if (type === 'register') {
        await api.register(email, password);
      } else {
        await api.login(email, password);
      }
      if (onAuthChange) onAuthChange();
      navigate(searchParams.get('next') || '/');
    } catch (err: any) {
      setError(err.message || 'Something went wrong.');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-card card">
      <h2>{type === 'login' ? 'Sign in' : 'Create an account'}</h2>
      <form onSubmit={handleSubmit}>
        <label>Email</label>
        <input className="w-full" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        <label>Password</label>
        <input className="w-full" type="password" value={password} onChange={(e) => setPassword(e.target.value)} minLength={8} required />
        {type === 'register' && (
          <>
            <label>Confirm password</label>
            <input className="w-full" type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} required />
          </>
        )}
        {error && <p className="error-text">{error}</p>}
        <button className="btn btn-accent w-full" type="submit" disabled={submitting}>
          {submitting ? 'Please wait…' : type === 'login' ? 'Sign in' : 'Sign up'}
        </button>
      </form>
      {type === 'login' ? (
        <p className="muted">Don't have an account? <Link to="/register">Sign up</Link></p>
      ) : (
        <p className="muted">Already have an account? <Link to="/login">Sign in</Link></p>
      )}
    </div>
  );
}
