import { NextResponse } from 'next/server';
import { getIntegrationStatus } from '@/lib/server/system-status';

export const runtime = 'nodejs';

export async function GET() {
  return NextResponse.json({
    generatedAt: new Date().toISOString(),
    integrations: getIntegrationStatus()
  });
}
