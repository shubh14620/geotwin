'use client';

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';

export function RainForecastChart({ data }: { data: { label: string; precipitationMm: number; probability: number }[] }) {
  return (
    <section className="glass-card panel-card chart-card">
      <div className="section-topline"><h3>Rainfall forecast</h3></div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data}>
          <CartesianGrid stroke="#22304a" strokeDasharray="4 4" />
          <XAxis dataKey="label" stroke="#8aa0bf" />
          <YAxis stroke="#8aa0bf" />
          <Tooltip contentStyle={{ background: '#08111f', border: '1px solid #1d3352' }} />
          <Bar dataKey="precipitationMm" fill="#38bdf8" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </section>
  );
}

export function RiskTrendChart({ data }: { data: { label: string; floodRisk: number }[] }) {
  return (
    <section className="glass-card panel-card chart-card">
      <div className="section-topline"><h3>Flood risk trend</h3></div>
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.5} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#22304a" strokeDasharray="4 4" />
          <XAxis dataKey="label" stroke="#8aa0bf" />
          <YAxis stroke="#8aa0bf" domain={[0, 100]} />
          <Tooltip contentStyle={{ background: '#08111f', border: '1px solid #1d3352' }} />
          <Area type="monotone" dataKey="floodRisk" stroke="#ef4444" fill="url(#riskFill)" strokeWidth={2.5} />
        </AreaChart>
      </ResponsiveContainer>
    </section>
  );
}

export function SoilMoistureChart({ data }: { data: { label: string; soilSurface: number; soilRoot: number }[] }) {
  return (
    <section className="glass-card panel-card chart-card">
      <div className="section-topline"><h3>Soil moisture trend</h3></div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data}>
          <CartesianGrid stroke="#22304a" strokeDasharray="4 4" />
          <XAxis dataKey="label" stroke="#8aa0bf" />
          <YAxis stroke="#8aa0bf" domain={[0, 1]} />
          <Tooltip contentStyle={{ background: '#08111f', border: '1px solid #1d3352' }} />
          <Line type="monotone" dataKey="soilSurface" stroke="#22c55e" strokeWidth={3} dot={{ r: 4, fill: '#22c55e' }} />
          <Line type="monotone" dataKey="soilRoot" stroke="#8b5cf6" strokeWidth={3} dot={{ r: 4, fill: '#8b5cf6' }} />
        </LineChart>
      </ResponsiveContainer>
    </section>
  );
}

export function SensorLevelChart({ data }: { data: { label: string; waterLevel: number }[] }) {
  return (
    <section className="glass-card panel-card chart-card">
      <div className="section-topline"><h3>Station water levels</h3></div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data}>
          <CartesianGrid stroke="#22304a" strokeDasharray="4 4" />
          <XAxis dataKey="label" stroke="#8aa0bf" interval={0} angle={-12} textAnchor="end" height={60} />
          <YAxis stroke="#8aa0bf" />
          <Tooltip contentStyle={{ background: '#08111f', border: '1px solid #1d3352' }} />
          <Bar dataKey="waterLevel" radius={[8, 8, 0, 0]}>
            {data.map((entry) => (
              <Cell key={entry.label} fill={entry.waterLevel > 5 ? '#ef4444' : entry.waterLevel > 3.2 ? '#f59e0b' : '#22c55e'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </section>
  );
}

export function VegetationPieChart({ data }: { data: { name: string; value: number; color: string }[] }) {
  return (
    <section className="glass-card panel-card chart-card">
      <div className="section-topline"><h3>Vegetation classification</h3></div>
      <ResponsiveContainer width="100%" height={280}>
        <PieChart>
          <Pie data={data} dataKey="value" innerRadius={55} outerRadius={92} paddingAngle={4}>
            {data.map((entry) => (
              <Cell key={entry.name} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip contentStyle={{ background: '#08111f', border: '1px solid #1d3352' }} />
        </PieChart>
      </ResponsiveContainer>
      <div className="chart-legend">
        {data.map((entry) => (
          <span key={entry.name}><i style={{ background: entry.color }} /> {entry.name}</span>
        ))}
      </div>
    </section>
  );
}
