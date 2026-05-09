export type StudyArea = 'ganga' | 'brahmaputra' | 'mumbai' | 'kerala' | 'punjab';

export type AreaDefinition = {
  key: StudyArea;
  label: string;
  region: string;
  center: [number, number];
  bbox: [[number, number], [number, number]];
  stationPrefix: string;
};

export const AREA_DEFINITIONS: Record<StudyArea, AreaDefinition> = {
  ganga: {
    key: 'ganga',
    label: 'Ganga Floodplain',
    region: 'Uttar Pradesh / Bihar, India',
    center: [25.95, 81.55],
    bbox: [[25.0, 80.0], [27.0, 83.0]],
    stationPrefix: 'GNG'
  },
  brahmaputra: {
    key: 'brahmaputra',
    label: 'Brahmaputra Basin',
    region: 'Assam, India',
    center: [26.55, 91.5],
    bbox: [[25.5, 90.0], [27.5, 93.0]],
    stationPrefix: 'BRM'
  },
  mumbai: {
    key: 'mumbai',
    label: 'Mumbai Coastal Region',
    region: 'Maharashtra, India',
    center: [19.15, 72.95],
    bbox: [[18.8, 72.5], [19.5, 73.5]],
    stationPrefix: 'MBY'
  },
  kerala: {
    key: 'kerala',
    label: 'Kerala Backwaters',
    region: 'Kerala, India',
    center: [9.9, 76.5],
    bbox: [[8.0, 75.5], [12.0, 77.5]],
    stationPrefix: 'KRL'
  },
  punjab: {
    key: 'punjab',
    label: 'Punjab Cropland',
    region: 'Punjab, India',
    center: [30.5, 75.5],
    bbox: [[29.0, 74.0], [32.0, 77.0]],
    stationPrefix: 'PNJ'
  }
};

export const AREA_OPTIONS = Object.values(AREA_DEFINITIONS);

export function getAreaDefinition(area: StudyArea): AreaDefinition {
  return AREA_DEFINITIONS[area] ?? AREA_DEFINITIONS.ganga;
}
