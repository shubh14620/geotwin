import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { NextRequest, NextResponse } from 'next/server';
import { parseSessionCookie, AUTH_COOKIE_NAME } from '@/lib/server/auth';
import { getLiveMonitorPayload } from '@/lib/server/open-meteo';
import { AREA_DEFINITIONS, type StudyArea } from '@/lib/areas';

export const runtime = 'nodejs';

export async function POST(request: NextRequest) {
  const session = parseSessionCookie(request.cookies.get(AUTH_COOKIE_NAME)?.value ?? null);
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const body = (await request.json().catch(() => ({}))) as { area?: StudyArea };
  const area = body.area && body.area in AREA_DEFINITIONS ? body.area : 'ganga';
  const live = await getLiveMonitorPayload(area);

  const input = {
    area: live.areaLabel,
    precipitation_mm: live.forecast.map((item) => item.precipitationMm),
    precipitation_probability: live.forecast.map((item) => item.probability),
    soil_surface: live.forecast.map((item) => item.soilSurface),
    soil_root: live.forecast.map((item) => item.soilRoot)
  };

  const pipelinePath = join(process.cwd(), 'python', 'flood_pipeline.py');
  let pipeline = 'typescript-fallback';
  let analysis: Record<string, unknown> = {
    area: live.areaLabel,
    next_24h_rain_mm: live.summary.next24hRainMm,
    peak_hourly_rain_mm: live.summary.peakHourlyRainMm,
    average_surface_moisture: live.summary.averageSurfaceMoisture,
    average_root_moisture: live.summary.averageRootMoisture,
    flood_risk_index: live.summary.floodRiskIndex,
    warning_level: live.summary.warningLevel,
    recommendation: live.summary.recommendation
  };

  if (process.env.RUN_LOCAL_PYTHON_PIPELINE === 'true' && existsSync(pipelinePath)) {
    const python = spawnSync('python3', [pipelinePath], {
      input: JSON.stringify(input),
      encoding: 'utf8',
      timeout: 10000
    });

    if (python.status === 0 && python.stdout) {
      try {
        analysis = JSON.parse(python.stdout) as Record<string, unknown>;
        pipeline = 'python-local';
      } catch {
        pipeline = 'typescript-fallback';
      }
    }
  }

  return NextResponse.json({
    ok: true,
    pipeline,
    generatedAt: new Date().toISOString(),
    analysis,
    requestedBy: session.email
  });
}
