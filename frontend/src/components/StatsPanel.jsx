import React, { useEffect, useState, useCallback } from 'react'
import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export default function StatsPanel() {
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    try {
      const res = await api.get('/stats')
      setStats(res.data)
      setError(null)
    } catch (err) {
      setError('Could not load statistics.')
    }
  }, [])

  useEffect(() => {
    load()
    const id = setInterval(load, 15000)
    return () => clearInterval(id)
  }, [load])

  if (error) return <div className="text-xs text-red-400">{error}</div>
  if (!stats) return <div className="text-xs text-slate-500">Loading statistics…</div>

  return (
    <div className="bg-slate-800 rounded p-3 text-sm">
      <div className="text-xs text-slate-400 mb-2">Dispatch Statistics </div>
      <div className="grid grid-cols-2 gap-2 text-center">
        <Stat label="Total dispatches" value={stats.total_dispatches} />
        <Stat label="Total reroutes" value={stats.total_reroutes} />
        <Stat label="Avg time saved" value={stats.avg_time_saved_min != null ? `${stats.avg_time_saved_min} min` : '—'} />
        <Stat label="Total time saved" value={stats.total_time_saved_min != null ? `${stats.total_time_saved_min} min` : '—'} />
      </div>
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div className="bg-slate-900 rounded p-2">
      <div className="text-lg font-semibold">{value}</div>
      <div className="text-[10px] text-slate-500">{label}</div>
    </div>
  )
}
