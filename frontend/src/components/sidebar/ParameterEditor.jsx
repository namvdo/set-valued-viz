import React from 'react';

const nextParamName = (params) => {
  let idx = 1;
  const existing = new Set(params.map(p => (p.name || '').trim()));
  while (existing.has(`p${idx}`)) idx += 1;
  return `p${idx}`;
};

export const ParameterEditor = ({ params, setParams, errors, disabled }) => {
  const updateParam = (index, key, value) => {
    setParams(prev => prev.map((p, i) => (i === index ? { ...p, [key]: value } : p)));
  };

  const removeParam = (index) => {
    setParams(prev => prev.filter((_, i) => i !== index));
  };

  const addParam = () => {
    setParams(prev => ([
      ...prev,
      { name: nextParamName(prev), value: 0 }
    ]));
  };

  return (
    <div className="param-editor">
      {params.length === 0 && (
        <div className="param-empty">No parameters yet. Add one below.</div>
      )}
      {params.map((param, idx) => (
        <div key={`${param.name}-${idx}`} className={`param-row ${errors?.[idx] ? 'has-error' : ''}`}>
          <input
            className="param-name"
            value={param.name}
            onChange={(e) => updateParam(idx, 'name', e.target.value)}
            placeholder="name"
            disabled={disabled}
          />
          <input
            className="param-value"
            type="number"
            value={Number.isFinite(param.value) ? param.value : ''}
            onChange={(e) => updateParam(idx, 'value', parseFloat(e.target.value))}
            placeholder="0"
            disabled={disabled}
          />
          <button className="param-remove" onClick={() => removeParam(idx)} disabled={disabled}>
            ×
          </button>
          {errors?.[idx] && <div className="param-error">{errors[idx]}</div>}
        </div>
      ))}
      <button className="param-add" onClick={addParam} disabled={disabled}>Add parameter</button>
    </div>
  );
};
