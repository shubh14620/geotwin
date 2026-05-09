import { NextRequest, NextResponse } from 'next/server';
import { getLiveMonitorPayload } from '@/lib/server/open-meteo';
import { parseSessionCookie, AUTH_COOKIE_NAME } from '@/lib/server/auth';
import { AREA_DEFINITIONS, type StudyArea } from '@/lib/areas';

export const runtime = 'nodejs';

export async function GET(request: NextRequest) {
  const session = parseSessionCookie(request.cookies.get(AUTH_COOKIE_NAME)?.value ?? null);
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const area = (request.nextUrl.searchParams.get('area') as StudyArea | null) ?? 'ganga';
  const studyArea = area in AREA_DEFINITIONS ? area : 'ganga';
  const payload = await getLiveMonitorPayload(studyArea);

  return NextResponse.json({
    user: session,
    payload
  });
}
