'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import {
  Activity,
  AlertTriangle,
  CloudRain,
  Cpu,
  Database,
  Lock,
  LogOut,
  RefreshCcw,
  Satellite,
  ShieldCheck,
  Thermometer,
  Waves,
  Wind
} from 'lucide-react';
import { AREA_OPTIONS, type StudyArea } from '@/lib/areas';
import type { LiveMonitorPayload } from '@/lib/server/open-meteo';
import { MetricCard } from '@/components/metric-card';
import { HeatmapCard } from '@/components/heatmap-card';
import { MapPanel } from '@/components/map-panel';
import { RainForecastChart, RiskTrendChart, SensorLevelChart, SoilMoistureChart, VegetationPieChart } from '@/components/charts';

type SessionPayload = {
  authenticated: boolean;
  user: {
    email: string;
    role: 'admin' | 'analyst';
    name: string;
  };
};

type AnalysisPayload = {
  pipeline: string;
  generatedAt: string;
  analysis: Record<string, string | number>;
};

export default function DashboardPage() {
  const [selectedArea, setSelectedArea] = useState<StudyArea>('ganga');
  const [session, setSession] = useState<SessionPayload['user'] | null>(null);
  const [data, setData] = useState<LiveMonitorPayload | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [runningAnalysis, setRunningAnalysis] = useState(false);
  const [error, setError] = useState('');

  async function fetchSession() {
    const response = await fetch('/api/auth/session', { cache: 'no-store' });
    if (!response.ok) {
      window.location.href = '/login';
      return false;
    }
    const payload = (await response.json()) as SessionPayload;
    setSession(payload.user);
    return true;
  }

  async function fetchDashboard(area: StudyArea) {
    setLoading(true);
    setError('');
    const response = await fetch(`/api/live-monitor?area=${area}`, { cache: 'no-store' });
    if (!response.ok) {
      if (response.status === 401) {
        window.location.href = '/login';
        return;
      }
      setError('Unable to load monitoring feed.');
      setLoading(false);
      return;
    }
    const payload = (await response.json()) as { payload: LiveMonitorPayload };
    setData(payload.payload);
    setLoading(false);
  }

  async function runAnalysis() {
    setRunningAnalysis(true);
    const response = await fetch('/api/flood-analysis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ area: selectedArea })
    });
    const payload = (await response.json()) as AnalysisPayload;
    setAnalysis(payload);
    setRunningAnalysis(false);
  }

  async function logout() {
    await fetch('/api/auth/logout', { method: 'POST' });
    window.location.href = '/login';
  }

  useEffect(() => {
    fetchSession().then((ok) => {
      if (ok) {
        fetchDashboard(selectedArea);
      }
    });
  }, []);

  useEffect(() => {
    if (session) {
      fetchDashboard(selectedArea);
    }
  }, [selectedArea]);

  const sensorLevels = useMemo(
    () => data?.sensors.map((sensor) => ({ label: sensor.name.replace(' Station ', '-'), waterLevel: sensor.waterLevel })) ?? [],
    [data]
  );

  if (loading && !data) {
    return <main className="loading-shell">Loading monitoring console...</main>;
  }

  if (!data) {
    return <main className="loading-shell">{error || 'No monitoring data available.'}</main>;
  }

  return (
    <main className="dashboard-shell upgraded-shell">
      <aside className="sidebar glass-card">
        <div>
          <span className="eyebrow">GeoTwin Control Room</span>
          <h2>{data.areaLabel}</h2>
          <p className="muted">{data.region}</p>
        </div>

        <div className="control-group">
          <label>Study area</label>
          <select value={selectedArea} onChange={(event) => setSelectedArea(event.target.value as StudyArea)}>
            {AREA_OPTIONS.map((area) => (
              <option key={area.key} value={area.key}>{area.label}</option>
            ))}
          </select>
        </div>

        <div className="control-stack">
          <button className="primary-btn full-btn" onClick={() => fetchDashboard(selectedArea)}>
            <RefreshCcw size={16} /> Refresh live feed
          </button>
          <button className="secondary-btn full-btn" onClick={runAnalysis} disabled={runningAnalysis}>
            <Cpu size={16} /> {runningAnalysis ? 'Running analysis...' : 'Run flood pipeline'}
          </button>
        </div>

        <div className="glass-subcard">
          <strong>Signed in as</strong>
          <p>{session?.name}</p>
          <small>{session?.email}</small>
        </div>

        <div className="stack-badges">
          <span><Lock size={14} /> Auth enabled</span>
          <span><Database size={14} /> {data.integrations.databaseProvider}</span>
          <span><Satellite size={14} /> {data.integrations.earthEngineConfigured ? 'EE ready' : 'EE pending'}</span>
        </div>

        <button className="secondary-btn full-btn" onClick={logout}>
          <LogOut size={16} /> Logout
        </button>
      </aside>

      <section className="dashboard-content">
        <header className="glass-card dashboard-hero">
          <div>
            <span className="eyebrow">{data.source === 'open-meteo' ? 'Live weather feed' : 'Fallback mode'}</span>
            <h1>{data.summary.warningLevel} flood watch</h1>
            <p>
              Snapshot generated at <strong>{new Date(data.generatedAt).toLocaleString('en-IN')}</strong>. Recommendation: {data.summary.recommendation}
            </p>
          </div>
          <div className="hero-tags">
            <span><CloudRain size={14} /> {data.summary.next24hRainMm} mm / 24h</span>
            <span><Waves size={14} /> Risk {data.summary.floodRiskIndex}</span>
            <span><Wind size={14} /> {data.current.windSpeedKmh} km/h</span>
          </div>
        </header>

        <section className="metric-grid">
          <MetricCard label="Flood risk index" value={String(data.summary.floodRiskIndex)} subtext={data.summary.warningLevel} tone="red" />
          <MetricCard label="Rain next 24h" value={`${data.summary.next24hRainMm} mm`} subtext={`Peak ${data.summary.peakHourlyRainMm} mm/h`} tone="cyan" />
          <MetricCard label="Surface moisture" value={String(data.summary.averageSurfaceMoisture)} subtext="Top soil saturation" tone="green" />
          <MetricCard label="Current temperature" value={`${data.current.temperatureC}°C`} subtext={`Humidity ${data.current.humidity}%`} tone="amber" />
          <MetricCard label="Cloud cover" value={`${data.current.cloudCover}%`} subtext={`Wind ${data.current.windSpeedKmh} km/h`} tone="violet" />
        </section>

        <section className="insight-strip glass-card">
          <span><ShieldCheck size={16} /> Session-protected monitoring console</span>
          <span><Activity size={16} /> {data.source === 'open-meteo' ? 'Real weather-driven dashboard' : 'Automatic fallback data source'}</span>
          <span><AlertTriangle size={16} /> {data.alerts.length} active alert item(s)</span>
        </section>

        <section className="two-column-grid">
          <MapPanel center={data.center} bbox={data.bbox} sensors={data.sensors} />
          <div className="stack-grid">
            <HeatmapCard title="Flood intensity grid" values={data.heatmaps.floodGrid} mode="flood" />
            <HeatmapCard title="Vegetation classification" values={data.heatmaps.classificationGrid} mode="class" />
          </div>
        </section>

        <section className="chart-grid">
          <RainForecastChart data={data.forecast} />
          <RiskTrendChart data={data.forecast} />
          <SoilMoistureChart data={data.forecast} />
          <SensorLevelChart data={sensorLevels} />
          <VegetationPieChart data={data.classBreakdown} />
        </section>

        <section className="split-grid">
          <section className="glass-card panel-card">
            <div className="section-topline"><h3>Active alerts</h3></div>
            <div className="alert-list">
              {data.alerts.map((alert) => (
                <article key={alert.id} className={`alert-item severity-${alert.severity.toLowerCase()}`}>
                  <div>
                    <strong>{alert.title}</strong>
                    <p>{alert.message}</p>
                  </div>
                  <small>{new Date(alert.time).toLocaleString('en-IN')}</small>
                </article>
              ))}
            </div>
          </section>

          <section className="glass-card panel-card">
            <div className="section-topline"><h3>Platform readiness</h3></div>
            <div className="readiness-grid">
              <article><strong>Live API</strong><span>{data.integrations.liveWeatherApi ? 'Connected' : 'Missing'}</span></article>
              <article><strong>Auth</strong><span>{data.integrations.authConfigured ? 'Configured' : 'Demo mode'}</span></article>
              <article><strong>Database</strong><span>{data.integrations.databaseConfigured ? 'Configured' : 'Schema included'}</span></article>
              <article><strong>Earth Engine</strong><span>{data.integrations.earthEngineConfigured ? 'Configured' : 'Needs credentials'}</span></article>
              <article><strong>Python pipeline</strong><span>{data.integrations.pythonPipelineAvailable ? 'Source included' : 'Missing'}</span></article>
            </div>
          </section>
        </section>

        {analysis ? (
          <section className="glass-card panel-card analysis-card">
            <div className="section-topline"><h3>Latest flood analysis</h3></div>
            <div className="analysis-grid">
              <article><span>Pipeline</span><strong>{analysis.pipeline}</strong></article>
              {Object.entries(analysis.analysis).map(([key, value]) => (
                <article key={key}>
                  <span>{key.replaceAll('_', ' ')}</span>
                  <strong>{String(value)}</strong>
                </article>
              ))}
            </div>
          </section>
        ) : null}

        <section className="glass-card panel-card delivery-card">
          <div className="section-topline"><h3>What is already real vs what needs credentials</h3></div>
          <p>
            Live weather monitoring is already wired into the dashboard. Database persistence, Earth Engine analysis, and external auth providers are scaffolded in the source code and can be activated by filling the environment variables and deploying the matching backend services.
          </p>
          <div className="pill-row">
            <span><Thermometer size={14} /> Current weather</span>
            <span><CloudRain size={14} /> Hourly precipitation</span>
            <span><Satellite size={14} /> Satellite backend scaffold</span>
            <span><Cpu size={14} /> Python pipeline</span>
          </div>
          <Link href="/login" className="secondary-btn inline-btn">Manage access</Link>
        </section>
      </section>
    </main>
  );
}
