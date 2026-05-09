export function MetricCard({ label, value, subtext, tone = 'cyan' }: { label: string; value: string; subtext: string; tone?: 'cyan' | 'green' | 'amber' | 'red' | 'violet'; }) {
  return (
    <article className={`metric-card metric-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{subtext}</small>
    </article>
  );
}
