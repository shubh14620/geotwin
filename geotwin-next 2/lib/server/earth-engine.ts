export type EarthEngineConfig = {
  projectId: string;
  serviceAccountEmail: string;
  privateKeyPresent: boolean;
  credentialFilePresent: boolean;
  ready: boolean;
};

export function getEarthEngineConfig(): EarthEngineConfig {
  const projectId = process.env.EE_PROJECT_ID || '';
  const serviceAccountEmail = process.env.EE_SERVICE_ACCOUNT_EMAIL || '';
  const privateKeyPresent = Boolean(process.env.EE_PRIVATE_KEY);
  const credentialFilePresent = Boolean(process.env.GOOGLE_APPLICATION_CREDENTIALS);

  return {
    projectId,
    serviceAccountEmail,
    privateKeyPresent,
    credentialFilePresent,
    ready: Boolean(projectId && serviceAccountEmail && (privateKeyPresent || credentialFilePresent))
  };
}

export function getEarthEngineSetupSummary() {
  const config = getEarthEngineConfig();
  return {
    ...config,
    note: config.ready
      ? 'Earth Engine credentials detected. Backend service can initialize the satellite pipeline.'
      : 'Add EE_PROJECT_ID, EE_SERVICE_ACCOUNT_EMAIL and EE_PRIVATE_KEY (or GOOGLE_APPLICATION_CREDENTIALS) to enable production Earth Engine analysis.'
  };
}
