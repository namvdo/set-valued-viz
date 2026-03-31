import React from 'react';

export const EquationDisplay = ({ systemId, customEquations, setCustomEquations, equationError, disabled }) => {
  if (systemId === 'custom') {
    return (
      <div className="eq-display">
        <div className="eq-display-label">Equation</div>
        <div className="eq-lines">
          <div className="eq-custom-inputs">
            <div className="eq-custom-row">
              <span className="eq-custom-label">x′ =</span>
              <input
                className="eq-custom-input"
                value={customEquations.xEq}
                onChange={(e) => setCustomEquations(prev => ({ ...prev, xEq: e.target.value }))}
                disabled={disabled}
              />
            </div>
            <div className="eq-custom-row">
              <span className="eq-custom-label">y′ =</span>
              <input
                className="eq-custom-input"
                value={customEquations.yEq}
                onChange={(e) => setCustomEquations(prev => ({ ...prev, yEq: e.target.value }))}
                disabled={disabled}
              />
            </div>
          </div>
          <div className="eq-hint">Variables: x, y, a, b. Functions: sin, cos, exp, sqrt, abs. Power: ^.</div>
          {equationError && (
            <div style={{ fontSize: '11px', color: '#e74c3c', marginTop: '4px', padding: '4px 6px', backgroundColor: '#2a1a1a', borderRadius: '3px' }}>
              {equationError}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Predefined equations
  const htmlMap = {
    'henon': (
      <>
        <div className="eq-line"><span className="prime">x′</span> = 1 − <span className="sym">a</span>x² + y</div>
        <div className="eq-line"><span className="prime">y′</span> = <span className="sym">b</span>x</div>
      </>
    ),
    'duffing': (
      <>
        <div className="eq-line"><span className="prime">x′</span> = y</div>
        <div className="eq-line"><span className="prime">y′</span> = −<span className="sym">b</span>x + <span className="sym">a</span>y − y³</div>
      </>
    ),
    'duffing_ode': (
      <>
        <div className="eq-line">ẍ + <span className="sym">δ</span>ẋ − x + x³ = 0</div>
        <div className="eq-line" style={{ marginTop: '4px', fontSize: '10px', color: 'var(--text-3)' }}> discretised via RK4 (step h)</div>
      </>
    )
  };

  return (
    <div className="eq-display">
      <div className="eq-display-label">Equation</div>
      <div className="eq-lines">
        {htmlMap[systemId]}
      </div>
    </div>
  );
};
