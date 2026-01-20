import React from 'react';

export default function ParameterControls({
  sigma,
  rho,
  beta,
  dt,
  x0,
  y0,
  z0,
  nSteps,
  onParametersChange,
  onSimulate,
  onReset,
  lyapunovExponent = null,
}) {
  const handleChange = (param, value) => {
    onParametersChange({ [param]: parseFloat(value) });
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Lorenz System Parameters</h2>

      <div style={styles.section}>
        <h3 style={styles.sectionTitle}>Dynamical System</h3>
        
        <div style={styles.parameter}>
          <label style={styles.label}>
            σ (sigma) - Prandtl Number
            <span style={styles.value}>{sigma.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min="0"
            max="20"
            step="0.1"
            value={sigma}
            onChange={(e) => handleChange('sigma', e.target.value)}
            style={styles.slider}
          />
          <div style={styles.hint}>Controls rate of convection (typical: 10)</div>
        </div>

        <div style={styles.parameter}>
          <label style={styles.label}>
            ρ (rho) - Rayleigh Number
            <span style={styles.value}>{rho.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min="0"
            max="40"
            step="0.1"
            value={rho}
            onChange={(e) => handleChange('rho', e.target.value)}
            style={styles.slider}
          />
          <div style={styles.hint}>
            Controls temperature variation (typical: 28)
            {rho <= 1 && <span style={styles.warning}> ⚠ ρ ≤ 1: converges to origin</span>}
            {rho > 1 && rho < 24.74 && <span style={styles.info}> ℹ stable fixed points</span>}
            {rho >= 24.74 && <span style={styles.success}> ✓ chaotic regime</span>}
          </div>
        </div>

        <div style={styles.parameter}>
          <label style={styles.label}>
            β (beta) - Geometric Factor
            <span style={styles.value}>{beta.toFixed(4)}</span>
          </label>
          <input
            type="range"
            min="0"
            max="5"
            step="0.01"
            value={beta}
            onChange={(e) => handleChange('beta', e.target.value)}
            style={styles.slider}
          />
          <div style={styles.hint}>Aspect ratio factor (typical: 8/3 ≈ 2.667)</div>
        </div>
      </div>

      <div style={styles.section}>
        <h3 style={styles.sectionTitle}>Integration Settings</h3>
        
        <div style={styles.parameter}>
          <label style={styles.label}>
            dt - Time Step
            <span style={styles.value}>{dt.toFixed(4)}</span>
          </label>
          <input
            type="range"
            min="0.001"
            max="0.05"
            step="0.001"
            value={dt}
            onChange={(e) => handleChange('dt', e.target.value)}
            style={styles.slider}
          />
          <div style={styles.hint}>
            RK4 integration step (typical: 0.01)
            {dt > 0.02 && <span style={styles.warning}> ⚠ May lose accuracy</span>}
          </div>
        </div>

        <div style={styles.parameter}>
          <label style={styles.label}>
            Steps
            <span style={styles.value}>{nSteps}</span>
          </label>
          <input
            type="range"
            min="1000"
            max="50000"
            step="1000"
            value={nSteps}
            onChange={(e) => handleChange('nSteps', parseInt(e.target.value))}
            style={styles.slider}
          />
          <div style={styles.hint}>Number of integration steps</div>
        </div>
      </div>

      <div style={styles.section}>
        <h3 style={styles.sectionTitle}>Initial Conditions</h3>
        
        <div style={styles.parameter}>
          <label style={styles.label}>
            x₀ <span style={styles.value}>{x0.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min="-20"
            max="20"
            step="0.1"
            value={x0}
            onChange={(e) => handleChange('x0', e.target.value)}
            style={styles.slider}
          />
        </div>

        <div style={styles.parameter}>
          <label style={styles.label}>
            y₀ <span style={styles.value}>{y0.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min="-20"
            max="20"
            step="0.1"
            value={y0}
            onChange={(e) => handleChange('y0', e.target.value)}
            style={styles.slider}
          />
        </div>

        <div style={styles.parameter}>
          <label style={styles.label}>
            z₀ <span style={styles.value}>{z0.toFixed(2)}</span>
          </label>
          <input
            type="range"
            min="-20"
            max="20"
            step="0.1"
            value={z0}
            onChange={(e) => handleChange('z0', e.target.value)}
            style={styles.slider}
          />
        </div>
      </div>

      {lyapunovExponent !== null && (
        <div style={styles.lyapunov}>
          <h3 style={styles.sectionTitle}>System Analysis</h3>
          <div style={styles.metric}>
            <span>Largest Lyapunov Exponent (λ₁):</span>
            <span style={{
              ...styles.metricValue,
              color: lyapunovExponent > 0 ? '#4ade80' : '#ef4444'
            }}>
              {lyapunovExponent.toFixed(4)}
            </span>
          </div>
          <div style={styles.hint}>
            {lyapunovExponent > 0 && '✓ Chaotic (exponential divergence)'}
            {lyapunovExponent === 0 && 'ℹ Neutral (periodic)'}
            {lyapunovExponent < 0 && '✗ Stable (converges)'}
          </div>
        </div>
      )}

      <div style={styles.buttons}>
        <button onClick={onSimulate} style={styles.buttonPrimary}>
          Run Simulation
        </button>
        <button onClick={onReset} style={styles.buttonSecondary}>
          Reset
        </button>
      </div>
    </div>
  );
}

const styles = {
  container: {
    width: '350px',
    height: '100vh',
    overflowY: 'auto',
    background: '#1e1e1e',
    color: '#fff',
    padding: '20px',
    fontFamily: 'monospace',
    borderRight: '1px solid #333',
  },
  title: {
    fontSize: '18px',
    marginBottom: '20px',
    borderBottom: '2px solid #4ade80',
    paddingBottom: '10px',
  },
  section: {
    marginBottom: '30px',
  },
  sectionTitle: {
    fontSize: '14px',
    color: '#4ade80',
    marginBottom: '15px',
  },
  parameter: {
    marginBottom: '20px',
  },
  label: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '8px',
    fontSize: '12px',
  },
  value: {
    color: '#60a5fa',
    fontWeight: 'bold',
  },
  slider: {
    width: '100%',
    cursor: 'pointer',
  },
  hint: {
    fontSize: '10px',
    color: '#888',
    marginTop: '4px',
  },
  warning: {
    color: '#fbbf24',
  },
  info: {
    color: '#60a5fa',
  },
  success: {
    color: '#4ade80',
  },
  lyapunov: {
    marginBottom: '20px',
    padding: '15px',
    background: '#2a2a2a',
    borderRadius: '8px',
  },
  metric: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '12px',
    marginBottom: '8px',
  },
  metricValue: {
    fontWeight: 'bold',
    fontSize: '14px',
  },
  buttons: {
    display: 'flex',
    gap: '10px',
  },
  buttonPrimary: {
    flex: 1,
    padding: '12px',
    background: '#4ade80',
    color: '#000',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 'bold',
    fontFamily: 'monospace',
  },
  buttonSecondary: {
    flex: 1,
    padding: '12px',
    background: '#374151',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 'bold',
    fontFamily: 'monospace',
  },
};