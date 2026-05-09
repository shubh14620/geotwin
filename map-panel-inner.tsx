'use client';

import { CircleMarker, MapContainer, Popup, Rectangle, TileLayer } from 'react-leaflet';

export function MapPanelInner({ center, bbox, sensors }: {
  center: [number, number];
  bbox: [[number, number], [number, number]];
  sensors: { id: string; name: string; lat: number; lng: number; waterLevel: number; battery: number; risk: 'Low' | 'Moderate' | 'High' }[];
}) {
  return (
    <section className="glass-card panel-card map-card">
      <div className="section-topline"><h3>Live monitoring map</h3></div>
      <div className="leaflet-wrap">
        <MapContainer center={center} zoom={8} scrollWheelZoom className="leaflet-canvas">
          <TileLayer
            attribution="&copy; OpenStreetMap contributors"
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <Rectangle bounds={bbox} pathOptions={{ color: '#38bdf8', weight: 2, dashArray: '6 6' }} />
          {sensors.map((sensor) => (
            <CircleMarker
              key={sensor.id}
              center={[sensor.lat, sensor.lng]}
              radius={10}
              pathOptions={{
                color: sensor.risk === 'High' ? '#ef4444' : sensor.risk === 'Moderate' ? '#f59e0b' : '#22c55e',
                fillOpacity: 0.82
              }}
            >
              <Popup>
                <strong>{sensor.name}</strong>
                <br /> Water level: {sensor.waterLevel} m
                <br /> Battery: {sensor.battery}%
                <br /> Risk: {sensor.risk}
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
    </section>
  );
}
