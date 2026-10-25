import React from 'react'

export default function AlertBanner({ decision, onDismiss }) {
  if (!decision) return null
  return (
    <div className="bg-amber-950 border-b border-amber-700 text-amber-100 px-6 py-3 text-sm flex items-center justify-between">
      <div>
        <div className="font-semibold">⚠ Traffic conditions changed. RapidRoute found a faster alternative. Route updated.</div>
        <div className="text-xs text-amber-300 mt-1">
          Previous ETA: {decision.current_route_eta_minutes} min &middot; New ETA: {decision.candidate_route_eta_minutes} min
          {decision.improvement_minutes != null && (
            <> &middot; Estimated improvement: {decision.improvement_minutes} min</>
          )}
        </div>
        <div className="text-xs text-amber-400 mt-0.5">{decision.reason}</div>
      </div>
      <button onClick={onDismiss} className="text-amber-300 hover:text-amber-100 text-lg leading-none px-2">×</button>
    </div>
  )
}
