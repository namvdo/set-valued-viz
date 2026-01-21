import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import HenonMapVisualization from './Viz'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <HenonMapVisualization />
  </StrictMode>,
)
