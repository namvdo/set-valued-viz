import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import HenonPeriodicViz from './HenonPeriodicViz.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <HenonPeriodicViz />
  </StrictMode>,
)
