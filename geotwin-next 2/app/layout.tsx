import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'GeoTwin Monitor',
  description: 'Production-style geospatial flood monitoring dashboard with live weather ingestion, auth, database schema, and Earth Engine / Python backend scaffolding.'
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
