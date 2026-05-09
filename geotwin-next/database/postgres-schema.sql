create table if not exists users (
  id uuid primary key default gen_random_uuid(),
  email text not null unique,
  full_name text not null,
  password_hash text not null,
  role text not null default 'analyst',
  created_at timestamptz not null default now()
);

create table if not exists monitoring_runs (
  id uuid primary key default gen_random_uuid(),
  study_area text not null,
  source text not null,
  warning_level text not null,
  flood_risk_index numeric(5,2) not null,
  next_24h_rain_mm numeric(8,2) not null,
  payload jsonb not null,
  created_by uuid references users(id) on delete set null,
  created_at timestamptz not null default now()
);

create table if not exists alerts (
  id uuid primary key default gen_random_uuid(),
  study_area text not null,
  severity text not null,
  title text not null,
  message text not null,
  status text not null default 'open',
  payload jsonb,
  created_at timestamptz not null default now()
);

create index if not exists monitoring_runs_created_at_idx on monitoring_runs(created_at desc);
create index if not exists alerts_created_at_idx on alerts(created_at desc);
create index if not exists alerts_status_idx on alerts(status);
