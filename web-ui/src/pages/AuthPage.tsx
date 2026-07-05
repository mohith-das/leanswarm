export default function AuthPage({ type }: { type: 'login' | 'register' }) {
  return <div className="card">Auth: {type}</div>;
}
