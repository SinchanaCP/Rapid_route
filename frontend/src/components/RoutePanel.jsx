import React from 'react'

const CONGESTION_COLORS = {
  LOW: 'text-green-400',
  MODERATE: 'text-yellow-400',
  HIGH: 'text-orange-400',
  SEVERE: 'text-red-400',
}

export default function RoutePanel({
  origin, destination, onOriginChange, onDestinationChange,
  onUseCurrentLocation, onFindRoute, loading, routeData,
}) {
  return (
    <div className="space-y-3">
      <div>
        <label className="block text-xs text-slate-400 mb-1">Origin</label>
        <div className="flex gap-2">
          <input
            className="flex-1 bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm"
            placeholder="e.g. Indiranagar, Bengaluru"
            value={origin}
            onChange={(e) => onOriginChange(e.target.value)}
          />
          <button
            onClick={onUseCurrentLocation}
            className="text-xs px-2 py-2 bg-slate-800 border border-slate-700 rounded hover:bg-slate-700"
            title="Use current GPS location"
          >
            📍
          </button>
        </div>
      </div>

      <div>
        <label className="block text-xs text-slate-400 mb-1">Destination</label>
        <input
          className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm"
          placeholder="e.g. Whitefield, Bengaluru"
          value={destination}
          onChange={(e) => onDestinationChange(e.target.value)}
        />
      </div>

      <button
        onClick={onFindRoute}
        disabled={loading}
        className="w-full bg-green-600 hover:bg-green-500 disabled:opacity-50 rounded py-2 text-sm font-medium"
      >
        {loading ? 'Finding route…' : 'Find Route'}
      </button>

      {routeData && (
        <div className="mt-4 border-t border-slate-800 pt-3 space-y-3 text-sm">
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-800 rounded p-3">
              <div className="text-slate-400 text-xs mb-1">Standard Route</div>
              <div>{(routeData.standard_route.distance_m / 1000).toFixed(2)} km</div>
              <div>{routeData.standard_route.eta_minutes} min ETA</div>
            </div>
            <div className="bg-slate-800 rounded p-3 border border-green-700">
              <div className="text-green-400 text-xs mb-1">RapidRoute</div>
              <div>{(routeData.optimized_route.distance_m / 1000).toFixed(2)} km</div>
              <div>{routeData.optimized_route.eta_minutes} min predicted ETA</div>
            </div>
          </div>

          <div className="bg-slate-800 rounded p-3">
            <div className="text-xs text-slate-400 mb-1">Result</div>
            <div>{routeData.message}</div>
          </div>

          <div className="bg-slate-800 rounded p-3 flex items-center justify-between">
            <span className="text-xs text-slate-400">Predicted Route Congestion</span>
            <span className={`font-semibold ${CONGESTION_COLORS[routeData.congestion_level] || ''}`}>
              {routeData.congestion_level || 'Unavailable — no monitored roads matched'}
            </span>
          </div>
          {routeData.predicted_route_avg_speed_kmh != null && (
            <div className="text-xs text-slate-400 text-right">
              Predicted speed on monitored segments: {routeData.predicted_route_avg_speed_kmh} km/h
            </div>
          )}
        </div>
      )}
    </div>
  )
}
