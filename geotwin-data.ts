export type StudyArea = 'ganga' | 'brahmaputra' | 'mumbai' | 'kerala' | 'punjab';
export type Basemap = 'dark' | 'satellite' | 'osm' | 'topo';

export type DashboardControls = {
  thresholdDb: number;
  ndviLow: number;
  ndviHigh: number;
  studyArea: StudyArea;
  seed: number;
};

export type SensorPoint = {
  id: string;
  name: string;
  lat: number;
  lng: number;
  waterLevel: number;
  battery: number;
  risk: 'Low' | 'Moderate' | 'High';
};

export type DashboardData = {
  metrics: {
    floodedPct: number;
    floodPixels: number;
    avgBackscatter: number;
    avgNdvi: number;
    healthyPct: number;
    riskIndex: number;
  };
  floodGrid: number[][];
  ndviGrid: number[][];
  classificationGrid: number[][];
  floodHistogram: { bucket: string; count: number }[];
  floodSeries: { label: string; floodedPct: number }[];
  ndviSeries: { label: string; avgNdvi: number }[];
  classBreakdown: { name: string; value: number; color: string }[];
  sensors: SensorPoint[];
  bbox: [[number, number], [number, number]];
  center: [number, number];
  studyAreaLabel: string;
};

const studyAreas: Record<StudyArea, { label: string; center: [number, number]; bbox: [[number, number], [number, number]] }> = {
  ganga: {
    label: 'Ganga Floodplain, UP/Bihar',
    center: [25.95, 81.55],
    bbox: [[25.0, 80.0], [27.0, 83.0]]
  },
  brahmaputra: {
    label: 'Brahmaputra, Assam',
    center: [26.55, 91.5],
    bbox: [[25.5, 90.0], [27.5, 93.0]]
  },
  mumbai: {
    label: 'Mumbai Coastal Region',
    center: [19.15, 72.95],
    bbox: [[18.8, 72.5], [19.5, 73.5]]
  },
  kerala: {
    label: 'Kerala Backwaters',
    center: [9.9, 76.5],
    bbox: [[8.0, 75.5], [12.0, 77.5]]
  },
  punjab: {
    label: 'Punjab Cropland',
    center: [30.5, 75.5],
    bbox: [[29.0, 74.0], [32.0, 77.0]]
  }
};

function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function smoothGrid(grid: number[][], iterations = 1) {
  let result = grid.map((row) => [...row]);
  const rows = result.length;
  const cols = result[0].length;

  for (let k = 0; k < iterations; k += 1) {
    const next = result.map((row) => [...row]);
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        let sum = 0;
        let count = 0;
        for (let dr = -1; dr <= 1; dr += 1) {
          for (let dc = -1; dc <= 1; dc += 1) {
            const rr = r + dr;
            const cc = c + dc;
            if (rr >= 0 && rr < rows && cc >= 0 && cc < cols) {
              sum += result[rr][cc];
              count += 1;
            }
          }
        }
        next[r][c] = sum / count;
      }
    }
    result = next;
  }

  return result;
}

function generateBaseSurface(size: number, random: () => number) {
  const grid = Array.from({ length: size }, (_, r) =>
    Array.from({ length: size }, (_, c) => {
      const x = (c / size) * Math.PI * 2;
      const y = (r / size) * Math.PI * 2;
      const wave = Math.sin(x * 0.9) * Math.cos(y * 0.6) + Math.sin(x * 0.35 + 0.5) * Math.sin(y * 0.45 + 1.1) * 0.45;
      return wave + (random() - 0.5) * 0.4;
    })
  );
  return smoothGrid(grid, 3);
}

function normalizeGrid(grid: number[][]) {
  let min = Infinity;
  let max = -Infinity;
  grid.forEach((row) => row.forEach((v) => {
    min = Math.min(min, v);
    max = Math.max(max, v);
  }));
  return grid.map((row) => row.map((v) => (v - min) / (max - min || 1)));
}

function buildSeries(random: () => number, base: number, spread: number, labels: string[], floor = 0, ceil = 100) {
  return labels.map((label, index) => {
    const curve = Math.sin(index * 0.75) * spread * 0.4;
    return {
      label,
      value: clamp(base + curve + (random() - 0.5) * spread, floor, ceil)
    };
  });
}

export function generateDashboardData(controls: DashboardControls): DashboardData {
  const random = mulberry32(controls.seed + Math.round((controls.thresholdDb + 30) * 100) + Math.round(controls.ndviHigh * 1000));
  const size = 42;
  const terrain = normalizeGrid(generateBaseSurface(size, random));
  const floodGrid = Array.from({ length: size }, (_, r) =>
    Array.from({ length: size }, (_, c) => {
      const diagonalRiver = Math.abs(c - (size * 0.24 + r * 0.42 + Math.sin(r * 0.2) * 3)) < 2.8;
      const floodplain = terrain[r][c] < 0.35;
      const vv = -8 - terrain[r][c] * 10 - (diagonalRiver ? 6 : 0) - (floodplain ? 4 : 0) + (random() - 0.5) * 2.5;
      return vv;
    })
  );

  const ndviGrid = Array.from({ length: size }, (_, r) =>
    Array.from({ length: size }, (_, c) => {
      const field = 0.5 + Math.sin((c / size) * Math.PI * 3) * 0.2 + Math.cos((r / size) * Math.PI * 2.5) * 0.16;
      const waterInfluence = Math.abs(c - (size * 0.18 + r * 0.18 + Math.sin(r * 0.15) * 1.5)) < 1.9 ? -0.4 : 0;
      const ndvi = field + waterInfluence - terrain[r][c] * 0.12 + (random() - 0.5) * 0.08;
      return clamp(ndvi, -0.1, 0.88);
    })
  );

  const classificationGrid = ndviGrid.map((row) =>
    row.map((value) => (value >= controls.ndviHigh ? 2 : value >= controls.ndviLow ? 1 : 0))
  );

  const floodMask = floodGrid.map((row) => row.map((value) => (value < controls.thresholdDb ? 1 : 0)));
  const floodPixels = floodMask.flat().reduce<number>((sum, value) => sum + Number(value), 0);
  const totalPixels = size * size;
  const floodedPct = (floodPixels / totalPixels) * 100;
  const avgBackscatter = floodGrid.flat().reduce<number>((sum, value) => sum + value, 0) / totalPixels;
  const avgNdvi = ndviGrid.flat().reduce<number>((sum, value) => sum + value, 0) / totalPixels;
  const classCounts = classificationGrid.flat().reduce(
    (acc, current) => {
      acc[current] += 1;
      return acc;
    },
    [0, 0, 0]
  );
  const healthyPct = (classCounts[2] / totalPixels) * 100;
  const riskIndex = clamp(floodedPct * 0.62 + (100 - healthyPct) * 0.38, 0, 100);

  const histogramBuckets = [
    { min: -24, max: -20 },
    { min: -20, max: -18 },
    { min: -18, max: -16 },
    { min: -16, max: -14 },
    { min: -14, max: -12 },
    { min: -12, max: -10 },
    { min: -10, max: -8 }
  ];

  const floodHistogram = histogramBuckets.map((bucket) => ({
    bucket: `${bucket.min} to ${bucket.max} dB`,
    count: floodGrid.flat().filter((value) => value >= bucket.min && value < bucket.max).length
  }));

  const labels = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov'];
  const floodSeries = buildSeries(random, floodedPct, 9, labels, 3, 74).map((item) => ({ label: item.label, floodedPct: Number(item.value.toFixed(1)) }));
  const ndviSeries = buildSeries(random, avgNdvi, 0.12, labels, 0.08, 0.86).map((item) => ({ label: item.label, avgNdvi: Number(item.value.toFixed(2)) }));

  const area = studyAreas[controls.studyArea];
  const sensors: SensorPoint[] = Array.from({ length: 6 }, (_, idx) => {
    const lat = area.bbox[0][0] + (area.bbox[1][0] - area.bbox[0][0]) * (0.18 + idx * 0.12 + random() * 0.08);
    const lng = area.bbox[0][1] + (area.bbox[1][1] - area.bbox[0][1]) * (0.22 + idx * 0.1 + random() * 0.08);
    const waterLevel = Number((1.2 + floodedPct / 35 + random() * 2.4).toFixed(2));
    const battery = Math.round(38 + random() * 60);
    const risk = waterLevel > 4 ? 'High' : waterLevel > 2.8 ? 'Moderate' : 'Low';
    return {
      id: `sensor-${idx + 1}`,
      name: `Station-${idx + 1}`,
      lat,
      lng,
      waterLevel,
      battery,
      risk
    };
  });

  return {
    metrics: {
      floodedPct: Number(floodedPct.toFixed(1)),
      floodPixels,
      avgBackscatter: Number(avgBackscatter.toFixed(2)),
      avgNdvi: Number(avgNdvi.toFixed(2)),
      healthyPct: Number(healthyPct.toFixed(1)),
      riskIndex: Number(riskIndex.toFixed(1))
    },
    floodGrid,
    ndviGrid,
    classificationGrid,
    floodHistogram,
    floodSeries,
    ndviSeries,
    classBreakdown: [
      { name: 'Low', value: classCounts[0], color: '#ef4444' },
      { name: 'Moderate', value: classCounts[1], color: '#f59e0b' },
      { name: 'Healthy', value: classCounts[2], color: '#22c55e' }
    ],
    sensors,
    bbox: area.bbox,
    center: area.center,
    studyAreaLabel: area.label
  };
}
