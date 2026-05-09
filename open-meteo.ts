import { getAreaDefinition, type StudyArea } from '@/lib/areas';
import { getIntegrationStatus } from '@/lib/server/system-status';

export type AlertLevel = 'Low' | 'Moderate' | 'High' | 'Severe';

export type LiveMonitorPayload = {
  studyArea: StudyArea;
  areaLabel: string;
  region: string;
  center: [number, number];
  bbox: [[number, number], [number, number]];
  source: 'open-meteo' | 'fallback';
  generatedAt: string;
  current: {
    temperatureC: number;
    humidity: number;
    windSpeedKmh: number;
    precipitationMm: number;
    cloudCover: number;
    soilSurface: number;
    soilRoot: number;
    localTime: string;
  };
  summary: {
    next24hRainMm: number;
    peakHourlyRainMm: number;
    averageSurfaceMoisture: number;
    averageRootMoisture: number;
    floodRiskIndex: number;
    warningLevel: AlertLevel;
    recommendation: string;
  };
  forecast: Array<{
    label: string;
    precipitationMm: number;
    probability: number;
    soilSurface: number;
    soilRoot: number;
    floodRisk: number;
  }>;
  sensors: Array<{
    id: string;
    name: string;
    lat: number;
    lng: number;
    waterLevel: number;
    battery: number;
    risk: 'Low' | 'Moderate' | 'High';
  }>;
  alerts: Array<{
    id: string;
    severity: AlertLevel;
    title: string;
    message: string;
    time: string;
  }>;
  heatmaps: {
    floodGrid: number[][];
    classificationGrid: number[][];
  };
  classBreakdown: { name: string; value: number; color: string }[];
  integrations: ReturnType<typeof getIntegrationStatus>;
};

type OpenMeteoResponse = {
  current?: Record<string, number | string>;
  current_units?: Record<string, string>;
  hourly?: Record<string, Array<number | string>>;
};

function seededValue(seed: string) {
  let hash = 2166136261;
  for (let i = 0; i < seed.length; i += 1) {
    hash ^= seed.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return () => {
    hash += 0x6d2b79f5;
    let t = hash;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function computeRisk(precipitationMm: number, probability: number, soilSurface: number, soilRoot: number, windSpeed = 0) {
  return clamp(
    precipitationMm * 4.2 + probability * 0.22 + soilSurface * 120 + soilRoot * 95 + windSpeed * 0.35,
    0,
    100
  );
}

function getWarningLevel(index: number): AlertLevel {
  if (index >= 75) return 'Severe';
  if (index >= 55) return 'High';
  if (index >= 35) return 'Moderate';
  return 'Low';
}

function getRecommendation(level: AlertLevel) {
  switch (level) {
    case 'Severe':
      return 'Activate district flood response workflow and publish immediate field alerts.';
    case 'High':
      return 'Increase station polling, inspect pumps and embankments, and prepare evacuation notices.';
    case 'Moderate':
      return 'Watch rainfall accumulation closely and notify local monitoring staff.';
    default:
      return 'Continue routine observation and archive the current monitoring snapshot.';
  }
}

function buildHeatmaps(seed: string, baseRisk: number, surfaceMoisture: number) {
  const random = seededValue(seed);
  const size = 18;
  const floodGrid: number[][] = [];
  const classificationGrid: number[][] = [];
  const counts = [0, 0, 0];

  for (let row = 0; row < size; row += 1) {
    const floodRow: number[] = [];
    const classRow: number[] = [];

    for (let col = 0; col < size; col += 1) {
      const wave = Math.sin(row * 0.42) + Math.cos(col * 0.36) + Math.sin((row + col) * 0.18);
      const riverBias = Math.abs(col - (size * 0.25 + row * 0.35 + Math.sin(row * 0.4) * 1.5)) < 2 ? -5.5 : 0;
      const vv = -11 - baseRisk * 0.08 - surfaceMoisture * 9 + wave * 0.9 + riverBias + (random() - 0.5) * 1.6;
      floodRow.push(Number(vv.toFixed(2)));

      const vegetation = clamp(0.24 + surfaceMoisture * 0.9 + Math.sin(col * 0.34) * 0.18 - Math.cos(row * 0.22) * 0.09 + (random() - 0.5) * 0.15, 0, 0.95);
      const classValue = vegetation >= 0.56 ? 2 : vegetation >= 0.31 ? 1 : 0;
      classRow.push(classValue);
      counts[classValue] += 1;
    }

    floodGrid.push(floodRow);
    classificationGrid.push(classRow);
  }

  return {
    floodGrid,
    classificationGrid,
    classBreakdown: [
      { name: 'Low', value: counts[0], color: '#ef4444' },
      { name: 'Moderate', value: counts[1], color: '#f59e0b' },
      { name: 'Healthy', value: counts[2], color: '#22c55e' }
    ]
  };
}

function buildSensors(area: ReturnType<typeof getAreaDefinition>, riskIndex: number, peakRain: number) {
  const random = seededValue(`${area.key}-${Math.round(riskIndex * 10)}`);
  return Array.from({ length: 6 }, (_, index) => {
    const lat = area.bbox[0][0] + (area.bbox[1][0] - area.bbox[0][0]) * (0.18 + index * 0.11 + random() * 0.05);
    const lng = area.bbox[0][1] + (area.bbox[1][1] - area.bbox[0][1]) * (0.19 + index * 0.11 + random() * 0.06);
    const waterLevel = Number((1.6 + riskIndex * 0.04 + peakRain * 0.07 + random() * 1.2).toFixed(2));
    const battery = Math.round(52 + random() * 45);
    const risk: 'Low' | 'Moderate' | 'High' = waterLevel > 5 ? 'High' : waterLevel > 3.2 ? 'Moderate' : 'Low';
    return {
      id: `${area.stationPrefix}-${index + 1}`,
      name: `${area.stationPrefix} Station ${index + 1}`,
      lat: Number(lat.toFixed(5)),
      lng: Number(lng.toFixed(5)),
      waterLevel,
      battery,
      risk
    };
  });
}

function buildAlerts(areaLabel: string, generatedAt: string, warningLevel: AlertLevel, next24hRainMm: number, peakHourlyRainMm: number, sensors: LiveMonitorPayload['sensors']) {
  const alerts: LiveMonitorPayload['alerts'] = [];
  if (warningLevel !== 'Low') {
    alerts.push({
      id: `rain-${generatedAt}`,
      severity: warningLevel,
      title: `${warningLevel} rainfall watch`,
      message: `${areaLabel} is showing ${next24hRainMm.toFixed(1)} mm of projected rain over the next 24h with a peak hourly burst of ${peakHourlyRainMm.toFixed(1)} mm.`,
      time: generatedAt
    });
  }

  sensors.filter((sensor) => sensor.risk !== 'Low').slice(0, 2).forEach((sensor) => {
    alerts.push({
      id: `sensor-${sensor.id}`,
      severity: sensor.risk === 'High' ? 'High' : 'Moderate',
      title: `${sensor.name} elevated water level`,
      message: `${sensor.name} is reporting ${sensor.waterLevel} m water level with ${sensor.battery}% battery remaining.`,
      time: generatedAt
    });
  });

  if (alerts.length === 0) {
    alerts.push({
      id: 'routine-watch',
      severity: 'Low',
      title: 'Routine monitoring state',
      message: `${areaLabel} remains in low-alert mode. Continue automated observation and archive current conditions.`,
      time: generatedAt
    });
  }

  return alerts;
}

function fallbackPayload(areaKey: StudyArea): LiveMonitorPayload {
  const area = getAreaDefinition(areaKey);
  const generatedAt = new Date().toISOString();
  const forecast = Array.from({ length: 12 }, (_, index) => {
    const precipitationMm = Number((Math.max(0, Math.sin(index / 2) * 5 + 3)).toFixed(1));
    const probability = Math.round(clamp(25 + index * 5, 10, 92));
    const soilSurface = Number((0.24 + index * 0.01).toFixed(2));
    const soilRoot = Number((0.31 + index * 0.008).toFixed(2));
    return {
      label: `${index}:00`,
      precipitationMm,
      probability,
      soilSurface,
      soilRoot,
      floodRisk: Number(computeRisk(precipitationMm, probability, soilSurface, soilRoot).toFixed(1))
    };
  });

  const next24hRainMm = forecast.reduce((sum, item) => sum + item.precipitationMm, 0);
  const peakHourlyRainMm = Math.max(...forecast.map((item) => item.precipitationMm));
  const averageSurfaceMoisture = forecast.reduce((sum, item) => sum + item.soilSurface, 0) / forecast.length;
  const averageRootMoisture = forecast.reduce((sum, item) => sum + item.soilRoot, 0) / forecast.length;
  const floodRiskIndex = computeRisk(next24hRainMm / 3, 62, averageSurfaceMoisture, averageRootMoisture, 18);
  const warningLevel = getWarningLevel(floodRiskIndex);
  const sensors = buildSensors(area, floodRiskIndex, peakHourlyRainMm);
  const heatmaps = buildHeatmaps(`${area.key}-fallback`, floodRiskIndex, averageSurfaceMoisture);

  return {
    studyArea: area.key,
    areaLabel: area.label,
    region: area.region,
    center: area.center,
    bbox: area.bbox,
    source: 'fallback',
    generatedAt,
    current: {
      temperatureC: 29.4,
      humidity: 74,
      windSpeedKmh: 18.4,
      precipitationMm: 1.1,
      cloudCover: 66,
      soilSurface: Number(averageSurfaceMoisture.toFixed(2)),
      soilRoot: Number(averageRootMoisture.toFixed(2)),
      localTime: generatedAt
    },
    summary: {
      next24hRainMm: Number(next24hRainMm.toFixed(1)),
      peakHourlyRainMm: Number(peakHourlyRainMm.toFixed(1)),
      averageSurfaceMoisture: Number(averageSurfaceMoisture.toFixed(2)),
      averageRootMoisture: Number(averageRootMoisture.toFixed(2)),
      floodRiskIndex: Number(floodRiskIndex.toFixed(1)),
      warningLevel,
      recommendation: getRecommendation(warningLevel)
    },
    forecast,
    sensors,
    alerts: buildAlerts(area.label, generatedAt, warningLevel, next24hRainMm, peakHourlyRainMm, sensors),
    heatmaps,
    classBreakdown: heatmaps.classBreakdown,
    integrations: getIntegrationStatus()
  };
}

export async function getLiveMonitorPayload(areaKey: StudyArea): Promise<LiveMonitorPayload> {
  const area = getAreaDefinition(areaKey);
  const params = new URLSearchParams({
    latitude: String(area.center[0]),
    longitude: String(area.center[1]),
    timezone: 'auto',
    forecast_days: '3',
    current: [
      'temperature_2m',
      'relative_humidity_2m',
      'wind_speed_10m',
      'precipitation',
      'cloud_cover'
    ].join(','),
    hourly: [
      'precipitation',
      'precipitation_probability',
      'soil_moisture_0_to_1cm',
      'soil_moisture_9_to_27cm',
      'cloud_cover',
      'wind_speed_10m'
    ].join(',')
  });

  try {
    const response = await fetch(`https://api.open-meteo.com/v1/forecast?${params.toString()}`, {
      headers: { Accept: 'application/json' },
      next: { revalidate: 900 }
    });

    if (!response.ok) {
      throw new Error(`Open-Meteo error ${response.status}`);
    }

    const payload = (await response.json()) as OpenMeteoResponse;
    const hourlyTimes = (payload.hourly?.time as string[] | undefined) ?? [];
    const precipitation = (payload.hourly?.precipitation as number[] | undefined) ?? [];
    const probability = (payload.hourly?.precipitation_probability as number[] | undefined) ?? [];
    const soilSurface = (payload.hourly?.soil_moisture_0_to_1cm as number[] | undefined) ?? [];
    const soilRoot = (payload.hourly?.soil_moisture_9_to_27cm as number[] | undefined) ?? [];
    const cloudCover = (payload.hourly?.cloud_cover as number[] | undefined) ?? [];
    const windSpeed = (payload.hourly?.wind_speed_10m as number[] | undefined) ?? [];

    const forecast = hourlyTimes.slice(0, 12).map((time, index) => {
      const rain = Number(precipitation[index] ?? 0);
      const pop = Number(probability[index] ?? 0);
      const surface = Number(soilSurface[index] ?? 0.25);
      const root = Number(soilRoot[index] ?? 0.3);
      const wind = Number(windSpeed[index] ?? 12);
      return {
        label: new Date(String(time)).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' }),
        precipitationMm: Number(rain.toFixed(1)),
        probability: Number(pop.toFixed(0)),
        soilSurface: Number(surface.toFixed(2)),
        soilRoot: Number(root.toFixed(2)),
        floodRisk: Number(computeRisk(rain, pop, surface, root, wind).toFixed(1))
      };
    });

    const precipitation24 = precipitation.slice(0, 24).map((value) => Number(value ?? 0));
    const surface24 = soilSurface.slice(0, 24).map((value) => Number(value ?? 0.25));
    const root24 = soilRoot.slice(0, 24).map((value) => Number(value ?? 0.3));
    const next24hRainMm = precipitation24.reduce((sum, value) => sum + value, 0);
    const peakHourlyRainMm = precipitation24.length ? Math.max(...precipitation24) : 0;
    const averageSurfaceMoisture = surface24.reduce((sum, value) => sum + value, 0) / Math.max(surface24.length, 1);
    const averageRootMoisture = root24.reduce((sum, value) => sum + value, 0) / Math.max(root24.length, 1);
    const floodRiskIndex = computeRisk(next24hRainMm / 3, probability[0] ? Number(probability[0]) : 42, averageSurfaceMoisture, averageRootMoisture, Number(payload.current?.wind_speed_10m ?? 0));
    const warningLevel = getWarningLevel(floodRiskIndex);
    const sensors = buildSensors(area, floodRiskIndex, peakHourlyRainMm);
    const generatedAt = new Date().toISOString();
    const heatmaps = buildHeatmaps(`${area.key}-${generatedAt.slice(0, 13)}`, floodRiskIndex, averageSurfaceMoisture);

    return {
      studyArea: area.key,
      areaLabel: area.label,
      region: area.region,
      center: area.center,
      bbox: area.bbox,
      source: 'open-meteo',
      generatedAt,
      current: {
        temperatureC: Number(Number(payload.current?.temperature_2m ?? 0).toFixed(1)),
        humidity: Number(Number(payload.current?.relative_humidity_2m ?? 0).toFixed(0)),
        windSpeedKmh: Number(Number(payload.current?.wind_speed_10m ?? 0).toFixed(1)),
        precipitationMm: Number(Number(payload.current?.precipitation ?? 0).toFixed(1)),
        cloudCover: Number(Number(payload.current?.cloud_cover ?? cloudCover[0] ?? 0).toFixed(0)),
        soilSurface: Number(averageSurfaceMoisture.toFixed(2)),
        soilRoot: Number(averageRootMoisture.toFixed(2)),
        localTime: String(payload.current?.time ?? generatedAt)
      },
      summary: {
        next24hRainMm: Number(next24hRainMm.toFixed(1)),
        peakHourlyRainMm: Number(peakHourlyRainMm.toFixed(1)),
        averageSurfaceMoisture: Number(averageSurfaceMoisture.toFixed(2)),
        averageRootMoisture: Number(averageRootMoisture.toFixed(2)),
        floodRiskIndex: Number(floodRiskIndex.toFixed(1)),
        warningLevel,
        recommendation: getRecommendation(warningLevel)
      },
      forecast,
      sensors,
      alerts: buildAlerts(area.label, generatedAt, warningLevel, next24hRainMm, peakHourlyRainMm, sensors),
      heatmaps,
      classBreakdown: heatmaps.classBreakdown,
      integrations: getIntegrationStatus()
    };
  } catch {
    return fallbackPayload(areaKey);
  }
}
