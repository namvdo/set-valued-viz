import React from 'react';
import { Collapsible } from '../ui/Collapsible';
import { Toggle } from '../ui/Toggle';

export const VisualizationPanel = ({ manifoldState, setManifoldState }) => {
  return (
    <Collapsible title="Visualization" defaultOpen={true}>
      <div className="view-toggle">
        <button className="view-btn active">Phase portrait</button>
        <button className="view-btn disabled" disabled>Poincaré section</button>
        <button className="view-btn disabled" disabled>Time series</button>
      </div>
      
      <Toggle 
        label="Trajectory trail" 
        checked={manifoldState.showTrail} 
        onChange={v => setManifoldState(prev => ({ ...prev, showTrail: v }))} 
      />
      <Toggle 
        label="Show orbit markers" 
        checked={manifoldState.showOrbits} 
        onChange={v => setManifoldState(prev => ({ ...prev, showOrbits: v }))} 
      />
      
      <div className="small-label" style={{ marginTop: '10px' }}>Integration method</div>
      <div style={{ display: 'flex', gap: '4px' }}>
        <button className="per-btn on" style={{ flex: 1, padding: '6px' }}>Euler</button>
        <button className="per-btn" style={{ flex: 1, padding: '6px', opacity: .5, cursor: 'default' }} disabled>RK4</button>
      </div>
    </Collapsible>
  );
};
