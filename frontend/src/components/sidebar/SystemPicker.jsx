import React from 'react';

export const SystemPicker = ({ type, setType, systemId, setSystemId, systems }) => {
  return (
    <>
      <div className="type-toggle-wrap">
        <div className="type-toggle-label">System type</div>
        <div className="type-toggle">
          <button 
            className={`type-btn ${type === 'discrete' ? 'active' : ''}`} 
            onClick={() => setType('discrete')}
          >
            Discrete
            <span className="type-sub">maps &amp; iterations</span>
          </button>
          <button 
            className={`type-btn ${type === 'continuous' ? 'active' : ''}`} 
            onClick={() => setType('continuous')}
          >
            Continuous
            <span className="type-sub">ODEs &amp; flows</span>
          </button>
        </div>
      </div>
      <div className="system-pick-wrap">
        <div className="sys-pick-label">System</div>
        <div className="sys-options">
          {systems[type].map(s => (
            <div 
              key={s.id}
              className={`sys-opt ${s.id === systemId ? 'active' : ''}`} 
              onClick={() => setSystemId(s.id)}
            >
              <span className="sys-opt-name">{s.name}</span>
            </div>
          ))}
        </div>
      </div>
    </>
  );
};
