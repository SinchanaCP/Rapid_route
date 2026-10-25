import React from 'react'
import { Marker, Popup } from 'react-leaflet'
import L from 'leaflet'

const icon = L.divIcon({
  html: '<div style="font-size:22px;line-height:22px;">🚑</div>',
  className: '',
  iconSize: [22, 22],
  iconAnchor: [11, 11],
})

export default function AmbulanceMarker({ position }) {
  if (!position) return null
  return (
    <Marker position={[position.lat, position.lng]} icon={icon}>
      <Popup>Ambulance (simulated position)</Popup>
    </Marker>
  )
}
