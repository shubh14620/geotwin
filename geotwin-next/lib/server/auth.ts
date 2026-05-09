import { createHmac, timingSafeEqual } from 'node:crypto';

const COOKIE_NAME = 'geotwin_session';

export type SessionUser = {
  email: string;
  role: 'admin' | 'analyst';
  name: string;
};

function getSecret() {
  return process.env.SESSION_SECRET || 'geotwin-dev-secret-change-me';
}

function sign(data: string) {
  return createHmac('sha256', getSecret()).update(data).digest('hex');
}

function toBase64Url(input: string) {
  return Buffer.from(input, 'utf8').toString('base64url');
}

function fromBase64Url(input: string) {
  return Buffer.from(input, 'base64url').toString('utf8');
}

export function createSessionCookie(user: SessionUser) {
  const payload = JSON.stringify({ ...user, exp: Date.now() + 1000 * 60 * 60 * 12 });
  const encoded = toBase64Url(payload);
  const signature = sign(encoded);
  return `${encoded}.${signature}`;
}

export function parseSessionCookie(rawCookie?: string | null): SessionUser | null {
  if (!rawCookie) return null;
  const [encoded, providedSignature] = rawCookie.split('.');
  if (!encoded || !providedSignature) return null;

  const expected = sign(encoded);
  if (providedSignature.length !== expected.length) return null;
  const valid = timingSafeEqual(Buffer.from(providedSignature), Buffer.from(expected));
  if (!valid) return null;

  const parsed = JSON.parse(fromBase64Url(encoded)) as SessionUser & { exp: number };
  if (!parsed.exp || parsed.exp < Date.now()) return null;

  return {
    email: parsed.email,
    role: parsed.role,
    name: parsed.name
  };
}

export function getDemoCredentials() {
  return {
    email: process.env.ADMIN_EMAIL || 'admin@geotwin.local',
    password: process.env.ADMIN_PASSWORD || 'geotwin123',
    user: {
      email: process.env.ADMIN_EMAIL || 'admin@geotwin.local',
      role: 'admin' as const,
      name: process.env.ADMIN_NAME || 'GeoTwin Administrator'
    }
  };
}

export const AUTH_COOKIE_NAME = COOKIE_NAME;
