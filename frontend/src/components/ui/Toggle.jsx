import React from 'react';

export const Toggle = ({ label, checked, onChange, colorLine, disabled }) => {
  return (
    <div className="t-row" onClick={() => !disabled && onChange(!checked)}>
      <div className="t-label">
        {colorLine && <div className="t-swatch-line" style={{ background: colorLine }}></div>}
        {label}
      </div>
      <div className={`t-switch ${checked ? 'on' : ''} ${disabled ? 'disabled' : ''}`}>
        <div className="t-track"></div>
        <div className="t-thumb"></div>
      </div>
    </div>
  );
};
