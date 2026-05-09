# GeoTwin Monitor

GeoTwin Monitor is a production-style flood monitoring dashboard built with Next.js. It combines a modern monitoring UI, live weather ingestion from Open-Meteo, session-based authentication, API routes for monitoring and analysis, a PostgreSQL-ready schema, and a standalone Python flood-risk pipeline.

The project is designed to work immediately in local demo mode while also giving you a clean path toward production integrations such as database persistence, Google Earth Engine, and external Python services.

---

## Overview

This repository upgrades the original GeoTwin demo into a more realistic geospatial monitoring platform. Out of the box, it provides:

- A landing page and protected login flow
- A live monitoring dashboard with map, charts, alerts, and readiness indicators
- Weather-driven flood-risk scoring using Open-Meteo data
- Session-based authentication using signed cookies
- Database schema for users, monitoring runs, and alerts
- Optional Python-based flood analysis execution
- Google Earth Engine integration scaffolding for future satellite workflows

---

## Key Features

### 1. Live weather-driven monitoring
The dashboard pulls forecast and precipitation-related signals from Open-Meteo and converts them into monitoring insights such as flood risk index, warning level, rainfall outlook, soil moisture summaries, and operational alerts.

### 2. Protected dashboard access
The application includes a working authentication flow using signed session cookies. A bundled demo admin account is included so the app can be tested immediately without external auth setup.

### 3. Multi-panel control room UI
The dashboard includes:

- Metric cards for risk, rainfall, soil moisture, temperature, and cloud cover
- Forecast and trend charts
- A GIS map panel with sensor markers
- Heatmap-style flood and vegetation grids
- Alert and platform-readiness panels

### 4. Python flood analysis pipeline
A standalone Python script accepts JSON input and returns a structured flood-risk assessment. The Next.js backend can optionally invoke this script locally.

### 5. Production-ready extension points
The repository already includes:

- Environment variable scaffolding
- PostgreSQL schema
- Earth Engine configuration helpers
- Health and system-status endpoints

---

## Tech Stack

### Frontend
- Next.js 15
- React 19
- TypeScript
- Leaflet + React Leaflet
- Recharts
- Lucide React icons

### Backend / APIs
- Next.js Route Handlers
- Open-Meteo weather API
- Cookie-based session auth
- Node.js runtime for server routes

### Data / Infrastructure
- PostgreSQL schema (`database/postgres-schema.sql`)
- Optional Supabase-compatible environment support
- Google Earth Engine scaffold

### Python
- Python flood-risk pipeline
- Optional FastAPI / Earth Engine dependencies in `python/requirements.txt`

---

## Application Flow

```text
User -> Login -> Session Cookie -> Protected Dashboard
                         |
                         v
                /api/live-monitor
                         |
                         v
         Open-Meteo forecast + local risk computation
                         |
                         v
      Metrics + alerts + charts + map + heatmap visualizations
                         |
                         v
              /api/flood-analysis (optional)
                         |
                         v
     TypeScript fallback OR local Python flood pipeline run
```

---

## Project Structure

```text
geotwin-next/
├── app/
│   ├── api/
│   │   ├── auth/
│   │   │   ├── login/route.ts
│   │   │   ├── logout/route.ts
│   │   │   └── session/route.ts
│   │   ├── flood-analysis/route.ts
│   │   ├── health/route.ts
│   │   ├── live-monitor/route.ts
│   │   └── system-status/route.ts
│   ├── dashboard/page.tsx
│   ├── login/page.tsx
│   ├── page.tsx
│   └── globals.css
├── components/
│   ├── charts.tsx
│   ├── heatmap-card.tsx
│   ├── map-panel.tsx
│   ├── map-panel-inner.tsx
│   └── metric-card.tsx
├── database/
│   └── postgres-schema.sql
├── lib/
│   ├── areas.ts
│   ├── geotwin-data.ts
│   └── server/
│       ├── auth.ts
│       ├── earth-engine.ts
│       ├── open-meteo.ts
│       └── system-status.ts
├── public/
├── python/
│   ├── earth_engine_backend.py
│   ├── flood_pipeline.py
│   └── requirements.txt
├── .env.example
├── next.config.ts
├── package.json
└── tsconfig.json
```

---

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Node.js (current LTS recommended)
- npm
- Python 3 (optional, only if you want to run the local analysis pipeline)

### 1. Install dependencies

```bash
npm install
```

### 2. Create your local environment file

```bash
cp .env.example .env.local
```

### 3. Start the development server

```bash
npm run dev
```

Open:

```text
http://localhost:3000
```

---

## Demo Login

Use the bundled demo account for local testing:

- **Email:** `admin@geotwin.local`
- **Password:** `geotwin123`

These values can be changed through environment variables.

---

## Environment Variables

Copy `.env.example` to `.env.local` and configure what you need.

| Variable | Required | Purpose |
|---|---:|---|
| `SESSION_SECRET` | Recommended | Secret used to sign session cookies |
| `ADMIN_EMAIL` | Optional | Demo admin login email |
| `ADMIN_PASSWORD` | Optional | Demo admin login password |
| `ADMIN_NAME` | Optional | Demo admin display name |
| `DATABASE_URL` | Optional | PostgreSQL or compatible database connection string |
| `NEXT_PUBLIC_SUPABASE_URL` | Optional | Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Optional | Supabase anonymous client key |
| `SUPABASE_SERVICE_ROLE_KEY` | Optional | Supabase server/service key |
| `EE_PROJECT_ID` | Optional | Google Cloud / Earth Engine project ID |
| `EE_SERVICE_ACCOUNT_EMAIL` | Optional | Earth Engine service account email |
| `EE_PRIVATE_KEY` | Optional | Multiline private key for service-account auth |
| `GOOGLE_APPLICATION_CREDENTIALS` | Optional | Path to a credentials file |
| `OPEN_METEO_BASE_URL` | Optional | Base URL for Open-Meteo API |
| `PYTHON_PIPELINE_URL` | Optional | External Python pipeline endpoint |

### Optional local Python execution flag
The flood analysis route also supports a local execution mode when this variable is enabled:

```env
RUN_LOCAL_PYTHON_PIPELINE=true
```

If enabled, the backend attempts to run `python/flood_pipeline.py` locally.

---

## Available Scripts

```bash
npm run dev
npm run build
npm run start
npm run lint
```

---

## Supported Study Areas

The dashboard includes predefined monitoring areas:

- `ganga` — Ganga Floodplain
- `brahmaputra` — Brahmaputra Basin
- `mumbai` — Mumbai Coastal Region
- `kerala` — Kerala Backwaters
- `punjab` — Punjab Cropland

These are defined in `lib/areas.ts` and can be extended with additional regions.

---

## API Endpoints

### Authentication

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/login` | Logs in using demo credentials and sets session cookie |
| `POST` | `/api/auth/logout` | Clears the session cookie |
| `GET` | `/api/auth/session` | Returns current authenticated session |

### Monitoring / Status

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/live-monitor?area=ganga` | Returns live monitoring payload for a study area |
| `POST` | `/api/flood-analysis` | Runs TypeScript fallback or local Python flood analysis |
| `GET` | `/api/health` | Health check plus integration and Earth Engine readiness |
| `GET` | `/api/system-status` | Integration status summary |

---

## Python Pipeline

The Python pipeline accepts JSON input and outputs a structured flood-risk assessment.

### Quick test

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

### Example output

```json
{
  "area": "Ganga Floodplain",
  "next_24h_rain_mm": 17.3,
  "peak_hourly_rain_mm": 8.1,
  "average_surface_moisture": 0.303,
  "average_root_moisture": 0.348,
  "flood_risk_index": 67.0,
  "warning_level": "High",
  "recommendation": "Notify district response teams and review evacuation readiness."
}
```

---

## Database Schema

The project includes a PostgreSQL schema for:

- `users`
- `monitoring_runs`
- `alerts`

Apply the SQL in `database/postgres-schema.sql` to initialize your monitoring database.

---

## Google Earth Engine Integration

The repository already includes an Earth Engine configuration helper and a Python backend scaffold.

To enable production Earth Engine workflows, provide:

- `EE_PROJECT_ID`
- `EE_SERVICE_ACCOUNT_EMAIL`
- `EE_PRIVATE_KEY` or `GOOGLE_APPLICATION_CREDENTIALS`

This project does **not** ship with active Earth Engine credentials. It only includes the integration scaffold.

---

## What Works Immediately vs What Needs Setup

### Works immediately
- Landing page and dashboard UI
- Demo login flow
- Session cookie handling
- Live Open-Meteo ingestion
- Risk scoring and charts
- Local route handlers
- Python source code included in the repository

### Requires additional setup
- Persistent database storage
- Real user management / production auth
- Earth Engine credentials
- Hosted Python microservice integration
- Deployment hardening and observability

---

## Production Upgrade Ideas

If you want to take this beyond demo mode, recommended next steps are:

1. Replace demo auth with Supabase Auth, Auth.js, or an enterprise identity provider.
2. Persist monitoring runs and alerts to PostgreSQL.
3. Move flood analysis to a dedicated Python API service.
4. Connect Earth Engine outputs to the dashboard.
5. Add audit logs, RBAC, and alert notification channels.
6. Add tests, CI/CD, and environment-specific deployment configs.

---

## Troubleshooting

### Login does not work
- Make sure your demo credentials match the values in `.env.local`
- If you changed `SESSION_SECRET`, clear browser cookies and retry

### Dashboard shows fallback-style behavior
- Check network access to the Open-Meteo API
- Verify `OPEN_METEO_BASE_URL` has not been changed incorrectly

### Python analysis is not running locally
- Ensure Python 3 is installed
- Set `RUN_LOCAL_PYTHON_PIPELINE=true`
- Make sure `python/flood_pipeline.py` exists and can run with `python3`

### Earth Engine shows as not configured
- This is expected until valid Earth Engine credentials are added

---

## Notes

- Authentication is demo-oriented by default and intended for development use.
- The repository is structured to be easy to extend into a fuller flood intelligence platform.
- The dashboard can still operate meaningfully even before database or Earth Engine credentials are configured.

---

## Acknowledgements

- Weather data flow is designed around the Open-Meteo API
- Mapping is powered by Leaflet / React Leaflet
- Charts are powered by Recharts

---

## License

No license file was included in the provided source package. Add a license before publishing or distributing the project publicly.
