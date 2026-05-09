import { NextResponse } from 'next/server';
import { getIntegrationStatus } from '@/lib/server/system-status';
import { getEarthEngineSetupSummary } from '@/lib/server/earth-engine';

export const runtime = 'nodejs';

export async function GET() {
  return NextResponse.json({
    status: 'ok',
    generatedAt: new Date().toISOString(),
    integrations: getIntegrationStatus(),
    earthEngine: getEarthEngineSetupSummary()
  });
}
