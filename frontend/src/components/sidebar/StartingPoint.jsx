import React from 'react';
import { Collapsible } from '../ui/Collapsible';

export const StartingPoint = ({ type, startPoint, updateStartPoint }) => {
  const isContinuous = type === 'continuous';

  return (
    <Collapsible title="Starting point" defaultOpen={true}>
      <div className="start-grid">
        <div className="start-field">
          <label>{isContinuous ? 'x₀ (position)' : 'x₀'}</label>
          <input
            aria-label="x0-position"
            value={startPoint.x.toFixed(4)}
            onChange={e => updateStartPoint('x', parseFloat(e.target.value) || 0)}
          />
        </div>
        <div className="start-field">
          <label>{isContinuous ? 'v₀ (velocity)' : 'y₀'}</label>
          <input
            aria-label={isContinuous ? 'v0-velocity' : 'y0-position'}
            value={startPoint.y.toFixed(4)}
            onChange={e => updateStartPoint('y', parseFloat(e.target.value) || 0)}
          />
        </div>
      </div>
      {isContinuous && (
        <div className="start-hint">
          x is position, y is velocity.
        </div>
      )}
      {!isContinuous && (
        <div id="normal-inputs" className="start-grid" style={{ marginTop: '6px' }}>
          <div className="start-field">
            <label>n_x (normal)</label>
            <input
              aria-label="nx-normal"
              value={startPoint.nx.toFixed(4)}
              onChange={e => updateStartPoint('nx', parseFloat(e.target.value) || 0)}
            />
          </div>
          <div className="start-field">
            <label>n_y</label>
            <input
              aria-label="ny-normal"
              value={startPoint.ny.toFixed(4)}
              onChange={e => updateStartPoint('ny', parseFloat(e.target.value) || 0)}
            />
          </div>
        </div>
      )}
    </Collapsible>
  );
};
