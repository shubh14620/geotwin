'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowRight, Lock, Radar } from 'lucide-react';

export default function LoginPage() {
  const [email, setEmail] = useState('admin@geotwin.local');
  const [password, setPassword] = useState('geotwin123');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError('');

    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });

    const payload = (await response.json().catch(() => ({}))) as { error?: string; hint?: string };
    if (!response.ok) {
      setError(payload.hint || payload.error || 'Login failed');
      setLoading(false);
      return;
    }

    window.location.href = '/dashboard';
  }

  return (
    <main className="auth-shell">
      <section className="auth-card glass-card">
        <div>
          <span className="eyebrow"><Radar size={14} /> Secure access</span>
          <h1>Sign in to GeoTwin Monitor</h1>
          <p className="muted">Use the bundled demo credentials now, then replace them with your production auth provider later.</p>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label>
            <span>Email</span>
            <input value={email} onChange={(event) => setEmail(event.target.value)} placeholder="admin@geotwin.local" />
          </label>
          <label>
            <span>Password</span>
            <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="••••••••" />
          </label>
          {error ? <div className="form-error">{error}</div> : null}
          <button disabled={loading} className="primary-btn full-btn" type="submit">
            <Lock size={16} /> {loading ? 'Signing in...' : 'Enter dashboard'}
          </button>
        </form>

        <div className="auth-note">
          <strong>Demo access</strong>
          <p>admin@geotwin.local / geotwin123</p>
        </div>

        <Link href="/" className="secondary-btn inline-btn">
          Back to home <ArrowRight size={16} />
        </Link>
      </section>
    </main>
  );
}
