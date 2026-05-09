'use client';

type HeatmapCardProps = {
  title: string;
  values: number[][];
  mode: 'flood' | 'ndvi' | 'class';
};

function getCellColor(value: number, mode: HeatmapCardProps['mode']) {
  if (mode === 'flood') {
    if (value < -18) return '#0ea5e9';
    if (value < -16) return '#38bdf8';
    if (value < -14) return '#64748b';
    if (value < -12) return '#f59e0b';
    return '#ef4444';
  }
  if (mode === 'class') {
    return ['#ef4444', '#f59e0b', '#22c55e'][value] ?? '#94a3b8';
  }
  const scaled = Math.max(0, Math.min(1, (value + 0.1) / 0.98));
  const hue = 10 + scaled * 120;
  return `hsl(${hue}, 76%, ${38 + scaled * 12}%)`;
}

export function HeatmapCard({ title, values, mode }: HeatmapCardProps) {
  return (
    <section className="glass-card panel-card">
      <div className="section-topline">
        <h3>{title}</h3>
      </div>
      <div className="heatmap-grid" style={{ gridTemplateColumns: `repeat(${values[0].length}, 1fr)` }}>
        {values.flatMap((row, rowIndex) =>
          row.map((value, colIndex) => (
            <span
              key={`${rowIndex}-${colIndex}`}
              className="heatmap-cell"
              style={{ background: getCellColor(value, mode) }}
              title={String(Number(value.toFixed ? value.toFixed(2) : value).toString())}
            />
          ))
        )}
      </div>
    </section>
  );
}
