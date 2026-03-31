import React from 'react';
import { Collapsible } from '../ui/Collapsible';

export const StartingPoint = ({ type, startPoint, updateStartPoint }) => {
  return (
    <Collapsible title="Starting point" defaultOpen={true}>
      <div className="start-grid">
        <div className="start-field">
          <label>x₀</label>
          <input
            value={startPoint.x.toFixed(4)}
            onChange={e => updateStartPoint('x', parseFloat(e.target.value) || 0)}
          />
        </div>
        {type !== 'continuous' && (
          <div className="start-field">
            <label>y₀</label>
            <input
              value={startPoint.y.toFixed(4)}
              onChange={e => updateStartPoint('y', parseFloat(e.target.value) || 0)}
            />
          </div>
        )}
      </div>
      {type !== 'continuous' && (
        <div id="normal-inputs" className="start-grid" style={{ marginTop: '6px' }}>
          <div className="start-field">
            <label>n_x (normal)</label>
            <input
              value={startPoint.nx.toFixed(4)}
              onChange={e => updateStartPoint('nx', parseFloat(e.target.value) || 0)}
            />
          </div>
          <div className="start-field">
            <label>n_y</label>
            <input
              value={startPoint.ny.toFixed(4)}
              onChange={e => updateStartPoint('ny', parseFloat(e.target.value) || 0)}
            />
          </div>
        </div>
      )}
    </Collapsible>
  );
};
