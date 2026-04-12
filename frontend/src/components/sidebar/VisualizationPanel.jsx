import React from 'react';
import { Collapsible } from '../ui/Collapsible';
import { Toggle } from '../ui/Toggle';

export const VisualizationPanel = ({ manifoldState, setManifoldState, viewRange, setViewRange, rangeLimit, resetViewRange }) => {
  const limitLabel = rangeLimit ?? 10;
  return (
    <Collapsible title="Axis range" defaultOpen={true}>
      <div className="axis-range-grid">
        <div className="axis-range-row">
          <span className="axis-range-label">x</span>
          <input
            className="axis-range-input"
            type="number"
            value={viewRange.xMin}
            onChange={(e) => setViewRange({ xMin: parseFloat(e.target.value) })}
          />
          <input
            className="axis-range-input"
            type="number"
            value={viewRange.xMax}
            onChange={(e) => setViewRange({ xMax: parseFloat(e.target.value) })}
          />
        </div>
        <div className="axis-range-row">
          <span className="axis-range-label">y</span>
          <input
            className="axis-range-input"
            type="number"
            value={viewRange.yMin}
            onChange={(e) => setViewRange({ yMin: parseFloat(e.target.value) })}
          />
          <input
            className="axis-range-input"
            type="number"
            value={viewRange.yMax}
            onChange={(e) => setViewRange({ yMax: parseFloat(e.target.value) })}
          />
        </div>
      </div>
      <div className="axis-range-actions">
        <button className="axis-range-reset" onClick={resetViewRange}>Reset</button>
        <div className="axis-range-hint">Clamp ±{limitLabel}</div>
      </div>
    </Collapsible>
  );
};
