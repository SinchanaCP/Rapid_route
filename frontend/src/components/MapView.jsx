import React from 'react'
import { MapContainer, TileLayer, Polyline, Marker, Popup, useMap } from 'react-leaflet'
import AmbulanceMarker from './AmbulanceMarker.jsx'

const BENGALURU_CENTER = [12.9716, 77.5946]

function FitBounds({ coordsList }) {
  const map = useMap()
  React.useEffect(() => {
    const all = coordsList.flat().filter(Boolean)
    if (all.length > 1) {
      map.fitBounds(all, { padding: [40, 40] })
    }
  }, [coordsList, map])
  return null
}

export default function MapView({ routeData, ambulancePos }) {
  const standardCoords = routeData?.standard_route?.coords || []
  const optimizedCoords = routeData?.optimized_route?.coords || []

  return (
    <MapContainer center={BENGALURU_CENTER} zoom={12} className="h-full w-full">
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {standardCoords.length > 1 && (
  <Polyline
    positions={standardCoords}
    pathOptions={{
      color: '#2563eb',
      weight: 10,
      opacity: 0.7
    }}
  />
)}

{optimizedCoords.length > 1 && (
  <Polyline
    positions={optimizedCoords}
    pathOptions={{
      color: '#22c55e',
      weight: 6,
      opacity: 1
    }}
  />
)}
      {routeData?.origin && (
        <Marker position={[routeData.origin.lat, routeData.origin.lng]}>
          <Popup>Origin</Popup>
        </Marker>
      )}
      {routeData?.destination && (
        <Marker position={[routeData.destination.lat, routeData.destination.lng]}>
          <Popup>Destination</Popup>
        </Marker>
      )}

      {ambulancePos && <AmbulanceMarker position={ambulancePos} />}

      {(standardCoords.length > 1 || optimizedCoords.length > 1) && (
        <FitBounds coordsList={[standardCoords, optimizedCoords]} />
      )}

      <div className="leaflet-bottom leaflet-right m-3 bg-slate-900/90 text-xs text-slate-200 rounded px-3 py-2 space-y-1 pointer-events-none">
        <div className="flex items-center gap-2"><span className="w-4 h-0.5 bg-slate-500" style={{ borderTop: '2px dashed #2563eb' }} /> Standard route</div>
        <div className="flex items-center gap-2"><span className="w-4 h-1 bg-green-500 inline-block" /> RapidRoute (predictive)</div>
      </div>
    </MapContainer>
  )
}
