# GeoTwin Monitor

A production-style upgrade of the original GeoTwin demo.

## What is now included
- Live weather ingestion using Open-Meteo forecast data
- Protected dashboard with session-based login
- Real monitoring UX: alerts, map, sensor list, forecast charts, flood risk scoring
- API routes for live monitoring, health checks, auth, and flood analysis
- PostgreSQL schema for users, alerts, and monitoring runs
- Earth Engine backend scaffold
- Python flood analysis pipeline

## Demo login
- **Email:** `admin@geotwin.local`
- **Password:** `geotwin123`

## Local run
```bash
npm install
npm run dev
```

Open `http://localhost:3000` and sign in.

## Environment variables
Copy `.env.example` to `.env.local` and fill what you need.

### Already works without extra credentials
- Live Open-Meteo weather ingestion
- Cookie-based demo login
- Monitoring UI and flood risk scoring

### Needs production credentials
- PostgreSQL / Supabase persistence
- Google Earth Engine service account
- External Python service deployment

## Useful paths
- `app/api/live-monitor/route.ts` — live weather + derived flood-risk API
- `app/api/flood-analysis/route.ts` — optional Python pipeline execution route
- `database/postgres-schema.sql` — production schema
- `python/flood_pipeline.py` — standalone Python analysis script
- `python/earth_engine_backend.py` — Earth Engine bootstrap

## Python pipeline quick test
```bash
python3 python/flood_pipeline.py <<'EOF'
{
  "area": "Ganga Floodplain",
  "precipitation_mm": [1.2, 4.4, 8.1, 3.6],
  "precipitation_probability": [25, 42, 78, 60],
  "soil_surface": [0.21, 0.28, 0.35, 0.37],
  "soil_root": [0.31, 0.33, 0.36, 0.39]
}
EOF
```

## Earth Engine note
The source now contains the backend scaffold, but production use still requires a registered Google Cloud project, Earth Engine API access, and service-account credentials.
