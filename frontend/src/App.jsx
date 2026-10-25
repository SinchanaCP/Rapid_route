import React, { useState, useCallback } from 'react'
import axios from 'axios'
import MapView from './components/MapView.jsx'
import RoutePanel from './components/RoutePanel.jsx'
import PredictionChart from './components/PredictionChart.jsx'
import StatsPanel from './components/StatsPanel.jsx'

const api = axios.create({ baseURL: '/api' })

export default function App() {
  const [origin, setOrigin] = useState('')
  const [destination, setDestination] = useState('')
  const [originSource, setOriginSource] = useState('manual')
  const [gpsCoords, setGpsCoords] = useState(null)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [routeData, setRouteData] = useState(null) // full /api/route response

  const [ambulancePos, setAmbulancePos] = useState(null)

  const useCurrentLocation = () => {
    if (!navigator.geolocation) {
      setError('Geolocation is not supported by this browser. Enter an origin manually.')
      return
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setGpsCoords({ lat: pos.coords.latitude, lng: pos.coords.longitude })
        setOriginSource('gps')
        setOrigin(`${pos.coords.latitude.toFixed(5)}, ${pos.coords.longitude.toFixed(5)}`)
      },
      () => {
        setError('GPS permission denied. Please enter an origin address manually.')
        setOriginSource('manual')
      }
    )
  }

  const handleOriginChange = (val) => {
    setOrigin(val)
    setOriginSource('manual')
    setGpsCoords(null)
  }

  const findRoute = useCallback(async () => {
    if (!destination || (!origin && !gpsCoords)) {
      setError('Please provide both an origin and a destination.')
      return
    }
    setLoading(true)
    setError(null)
    setRouteData(null)
    try {
      const payload = {
        origin: gpsCoords ? { lat: gpsCoords.lat, lng: gpsCoords.lng } : origin,
        destination,
        origin_source: originSource,
      }
      const res = await api.post('/route', payload)
      setRouteData(res.data)
      if (res.data?.optimized_route?.coords?.length) {
        const [lat, lng] = res.data.optimized_route.coords[0]
        setAmbulancePos({ lat, lng })
      }
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || 'Failed to compute route.')
    } finally {
      setLoading(false)
    }
  }, [origin, destination, originSource, gpsCoords])

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-950 text-slate-100">
      <header className="px-6 py-4 border-b border-slate-800 bg-slate-900 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight">RapidRoute</h1>
          <p className="text-xs text-slate-400">
            AI-Powered Predictive Ambulance Routing &middot; Bengaluru &middot;
          </p>
        </div>
      </header>

      {error && (
        <div className="bg-red-950 border-b border-red-800 text-red-200 text-sm px-6 py-2">
          {error}
        </div>
      )}

      <div className="flex-1 flex overflow-hidden">
        <aside className="w-96 border-r border-slate-800 bg-slate-900 overflow-y-auto p-4 space-y-4">
          <RoutePanel
            origin={origin}
            destination={destination}
            onOriginChange={handleOriginChange}
            onDestinationChange={setDestination}
            onUseCurrentLocation={useCurrentLocation}
            onFindRoute={findRoute}
            loading={loading}
            routeData={routeData}
          />
          {routeData?.predictions_used && (
            <PredictionChart predictions={routeData.predictions_used} />
          )}
          <StatsPanel />
        </aside>

        <main className="flex-1 relative">
          <MapView routeData={routeData} ambulancePos={ambulancePos} />
        </main>
      </div>
    </div>
  )
}
