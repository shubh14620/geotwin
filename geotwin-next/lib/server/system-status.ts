import { existsSync } from 'node:fs';
import { join } from 'node:path';

export type IntegrationStatus = {
  liveWeatherApi: boolean;
  authConfigured: boolean;
  databaseConfigured: boolean;
  earthEngineConfigured: boolean;
  pythonPipelineAvailable: boolean;
  deploymentTarget: 'local' | 'vercel' | 'server';
  databaseProvider: 'supabase' | 'postgres' | 'mysql' | 'sqlite' | 'not-configured';
};

function detectDatabaseProvider(): IntegrationStatus['databaseProvider'] {
  const dbUrl = process.env.DATABASE_URL ?? '';
  if (process.env.NEXT_PUBLIC_SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY) return 'supabase';
  if (dbUrl.startsWith('postgres')) return 'postgres';
  if (dbUrl.startsWith('mysql')) return 'mysql';
  if (dbUrl.startsWith('file:') || dbUrl.endsWith('.db') || dbUrl.endsWith('.sqlite')) return 'sqlite';
  return 'not-configured';
}

export function getIntegrationStatus(): IntegrationStatus {
  const earthEngineConfigured = Boolean(
    process.env.EE_SERVICE_ACCOUNT_EMAIL &&
    (process.env.EE_PRIVATE_KEY || process.env.GOOGLE_APPLICATION_CREDENTIALS) &&
    process.env.EE_PROJECT_ID
  );

  return {
    liveWeatherApi: true,
    authConfigured: Boolean(process.env.SESSION_SECRET || (process.env.NEXT_PUBLIC_SUPABASE_URL && process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY)),
    databaseConfigured: Boolean(process.env.DATABASE_URL || (process.env.NEXT_PUBLIC_SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY)),
    earthEngineConfigured,
    pythonPipelineAvailable: existsSync(join(process.cwd(), 'python', 'flood_pipeline.py')),
    deploymentTarget: process.env.VERCEL ? 'vercel' : process.env.NODE_ENV === 'development' ? 'local' : 'server',
    databaseProvider: detectDatabaseProvider()
  };
}
