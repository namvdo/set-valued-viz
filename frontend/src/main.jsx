import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import SetValuedViz from './SetValuedViz'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <SetValuedViz />
  </StrictMode>,
)

