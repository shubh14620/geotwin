import Link from 'next/link';
import { ArrowRight, Database, Lock, Radar, Satellite, Waves, CloudRain, Cpu, Activity } from 'lucide-react';

const features = [
  {
    title: 'Live weather ingestion',
    text: 'The dashboard now pulls real forecast and precipitation signals from Open-Meteo instead of using only static demo numbers.',
    icon: CloudRain
  },
  {
    title: 'Protected monitoring access',
    text: 'A working session-based login flow is included so the monitoring console behaves like a real platform.',
    icon: Lock
  },
  {
    title: 'Database-ready architecture',
    text: 'A production PostgreSQL schema and environment template are included for persisting alerts, users, and monitoring runs.',
    icon: Database
  },
  {
    title: 'Earth Engine backend scaffold',
    text: 'The project now contains a dedicated Earth Engine integration layer and Python bootstrap for service-account-based satellite processing.',
    icon: Satellite
  },
  {
    title: 'Python flood pipeline',
    text: 'A standalone Python pipeline is included for flood-risk scoring so you can run analysis outside the frontend runtime.',
    icon: Cpu
  },
  {
    title: 'Operational UX',
    text: 'The monitoring page now looks and behaves like a real control room with alerts, forecasts, maps, and system health status.',
    icon: Activity
  }
];

const stack = [
  'Next.js App Router frontend',
  'Live Open-Meteo weather ingestion',
  'Cookie-based session auth',
  'PostgreSQL schema for production persistence',
  'Google Earth Engine backend scaffold',
  'Python flood analysis pipeline'
];

export default function HomePage() {
  return (
    <main className="landing-shell">
      <section className="topbar">
        <div className="brand">
          <div className="brand-dot" />
          <span>GeoTwin Monitor</span>
        </div>
        <nav className="topnav">
          <Link href="/login">Login</Link>
          <Link href="/dashboard">Dashboard</Link>
        </nav>
      </section>

      <section className="hero hero-landing">
        <div className="hero-copy glass-card">
          <span className="eyebrow"><Radar size={14} /> Full-stack monitoring upgrade</span>
          <h1>GeoTwin is now structured like a real flood monitoring platform</h1>
          <p>
            I upgraded the uploaded demo into a more production-style website: live weather feeds, protected access, API routes,
            database schema, Earth Engine scaffolding, and a standalone Python flood analysis pipeline are now included in the source code.
          </p>
          <div className="hero-actions">
            <Link href="/login" className="primary-btn">
              Login to console <ArrowRight size={18} />
            </Link>
            <Link href="/dashboard" className="secondary-btn">
              Open monitoring UI
            </Link>
          </div>
          <div className="pill-row">
            <span><Waves size={14} /> Flood monitoring</span>
            <span><CloudRain size={14} /> Live precipitation</span>
            <span><Satellite size={14} /> Earth Engine ready</span>
          </div>
        </div>

        <div className="hero-panel glass-card">
          <div className="panel-badge">What changed</div>
          <ul className="panel-list">
            {stack.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
          <div className="status-highlight">
            <strong>Demo credentials</strong>
            <p>admin@geotwin.local / geotwin123</p>
          </div>
        </div>
      </section>

      <section className="feature-grid">
        {features.map((feature) => {
          const Icon = feature.icon;
          return (
            <article key={feature.title} className="glass-card feature-card">
              <div className="feature-icon"><Icon size={20} /></div>
              <h3>{feature.title}</h3>
              <p>{feature.text}</p>
            </article>
          );
        })}
      </section>

      <section className="glass-card architecture-card">
        <div className="section-topline"><h3>Production path</h3></div>
        <div className="architecture-grid">
          <article>
            <h4>Frontend</h4>
            <p>Next.js dashboard, protected routes, live alerts, forecast charts, and GIS map components.</p>
          </article>
          <article>
            <h4>Data services</h4>
            <p>Weather ingestion through Open-Meteo API, plus route handlers that normalize and score flood risk.</p>
          </article>
          <article>
            <h4>Persistence</h4>
            <p>Environment template and PostgreSQL schema for users, monitoring runs, and alert history.</p>
          </article>
          <article>
            <h4>Advanced analysis</h4>
            <p>Earth Engine and Python pipeline scaffolds are included so you can attach satellite processing next.</p>
          </article>
        </div>
      </section>
    </main>
  );
}
