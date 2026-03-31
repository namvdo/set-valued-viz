import React from 'react';

export const Slider = ({ label, hint, min, max, step, value, onChange, disabled }) => {
  return (
    <div className="p-row">
      <div className="p-head">
        <span className="p-name">
          <em>{label}</em>
          {hint && <small>{hint}</small>}
        </span>
        <input 
          className={`p-val ${disabled ? 'disabled' : ''}`}
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value) || min)}
          disabled={disabled}
        />
      </div>
      <input 
        type="range" 
        className={`p-track ${disabled ? 'disabled' : ''}`}
        min={min} 
        max={max} 
        step={step} 
        value={value} 
        onChange={(e) => onChange(parseFloat(e.target.value))}
        disabled={disabled}
      />
    </div>
  );
};
