'use client';

import 'leaflet/dist/leaflet.css';
import dynamic from 'next/dynamic';

const LeafletMap = dynamic(() => import('./map-panel-inner').then((mod) => mod.MapPanelInner), { ssr: false });

export function MapPanel(props: {
  center: [number, number];
  bbox: [[number, number], [number, number]];
  sensors: { id: string; name: string; lat: number; lng: number; waterLevel: number; battery: number; risk: 'Low' | 'Moderate' | 'High' }[];
}) {
  return <LeafletMap {...props} />;
}
