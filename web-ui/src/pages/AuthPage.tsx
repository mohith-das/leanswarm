import { useState, type FormEvent } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { api } from '../api';

export default function AuthPage({ type, onAuthChange }: { type: 'login' | 'register' | 'forgot' | 'reset'; onAuthChange?: () => void }) {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [sent, setSent] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    if ((type === 'register' || type === 'reset') && password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }
    setSubmitting(true);
    try {
      if (type === 'register') {
        await api.register(email, password);
      } else if (type === 'login') {
        await api.login(email, password);
      } else if (type === 'forgot') {
        await api.forgotPassword(email);
        setSent(true);
        return;
      } else if (type === 'reset') {
        const token = searchParams.get('token') || '';
        if (!token) {
          setError('Missing reset token.');
          return;
        }
        await api.resetPassword(token, password);
        navigate('/login?reset=1');
        return;
      }
      if (onAuthChange) onAuthChange();
      navigate(searchParams.get('next') || '/');
    } catch (err: any) {
      setError(err.message || 'Something went wrong.');
    } finally {
      setSubmitting(false);
    }
  }

  if (type === 'forgot' && sent) {
    return (
      <div className="auth-card card">
        <h2>Check your email</h2>
        <p>If that email address is registered, a password reset link has been sent.</p>
        <Link to="/login" className="btn btn-accent w-full">Back to sign in</Link>
      </div>
    );
  }

  const heading = type === 'login' ? 'Sign in'
    : type === 'register' ? 'Create an account'
    : type === 'forgot' ? 'Reset your password'
    : 'Set a new password';

  return (
    <div className="auth-card card">
      <h2>{heading}</h2>
      <form onSubmit={handleSubmit}>
        {(type === 'login' || type === 'register' || type === 'forgot') && (
          <>
            <label>Email</label>
            <input className="w-full" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
          </>
        )}
        {(type === 'login' || type === 'register' || type === 'reset') && (
          <>
            <label>Password</label>
            <input className="w-full" type="password" value={password} onChange={(e) => setPassword(e.target.value)} minLength={8} required />
          </>
        )}
        {(type === 'register' || type === 'reset') && (
          <>
            <label>Confirm password</label>
            <input className="w-full" type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} required />
          </>
        )}
        {error && <p className="error-text">{error}</p>}
        {type === 'login' && searchParams.get('reset') === '1' && (
          <p className="muted">Password has been reset. Sign in with your new password.</p>
        )}
        <button className="btn btn-accent w-full" type="submit" disabled={submitting}>
          {submitting ? 'Please wait…'
            : type === 'login' ? 'Sign in'
            : type === 'register' ? 'Sign up'
            : type === 'forgot' ? 'Send reset link'
            : 'Reset password'}
        </button>
      </form>
      {type === 'login' ? (
        <>
          <p className="muted">Don't have an account? <Link to="/register">Sign up</Link></p>
          <p className="muted"><Link to="/forgot-password">Forgot your password?</Link></p>
        </>
      ) : type === 'register' ? (
        <p className="muted">Already have an account? <Link to="/login">Sign in</Link></p>
      ) : (
        <p className="muted"><Link to="/login">Back to sign in</Link></p>
      )}
    </div>
  );
}
