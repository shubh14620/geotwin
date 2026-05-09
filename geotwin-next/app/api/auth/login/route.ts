import { NextRequest, NextResponse } from 'next/server';
import { AUTH_COOKIE_NAME, createSessionCookie, getDemoCredentials } from '@/lib/server/auth';

export const runtime = 'nodejs';

export async function POST(request: NextRequest) {
  const { email, password } = (await request.json()) as { email?: string; password?: string };
  const demo = getDemoCredentials();

  if (!email || !password || email.toLowerCase() !== demo.email.toLowerCase() || password !== demo.password) {
    return NextResponse.json(
      {
        error: 'Invalid credentials',
        hint: `Use ${demo.email} / ${demo.password}`
      },
      { status: 401 }
    );
  }

  const response = NextResponse.json({ ok: true, user: demo.user });
  response.cookies.set({
    name: AUTH_COOKIE_NAME,
    value: createSessionCookie(demo.user),
    httpOnly: true,
    sameSite: 'lax',
    secure: process.env.NODE_ENV === 'production',
    maxAge: 60 * 60 * 12,
    path: '/'
  });
  return response;
}
