import React from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts'

const CONGESTION_COLOR = {
  LOW: '#22c55e',
  MODERATE: '#eab308',
  HIGH: '#f97316',
  SEVERE: '#ef4444',
}

export default function PredictionChart({ predictions }) {
  const data = Object.entries(predictions).map(([road, p]) => ({
    road: road.length > 14 ? road.slice(0, 13) + '…' : road,
    speed: p.predicted_avg_speed_kmh,
    level: p.congestion_level,
  }))

  return (
    <div className="bg-slate-800 rounded p-3">
      <div className="text-xs text-slate-400 mb-2">
        Predicted Average Speed by Road (LSTM output, km/h)
      </div>
      <div style={{ width: '100%', height: 220 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="road" tick={{ fill: '#94a3b8', fontSize: 10 }} angle={-35} textAnchor="end" interval={0} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
            <Tooltip
              contentStyle={{ background: '#52506dc3', border: '1px solid #334155', fontSize: 12 }}
              formatter={(value, name, props) => [`${value} km/h (${props.payload.level})`, 'Predicted speed']}
            />
            <Bar dataKey="speed">
              {data.map((d, i) => (
                <Cell key={i} fill={CONGESTION_COLOR[d.level] || '#38bdf8'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="flex gap-3 text-[10px] mt-1 text-slate-400">
        {Object.entries(CONGESTION_COLOR).map(([lvl, color]) => (
          <span key={lvl} className="flex items-center gap-1">
            <span className="w-2 h-2 inline-block rounded-sm" style={{ background: color }} /> {lvl}
          </span>
        ))}
      </div>
      <div className="text-[10px] text-slate-500 mt-2">
        Source: trained LSTM prediction from historical data.
      </div>
    </div>
  )
}
