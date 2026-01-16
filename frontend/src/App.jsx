import React, { useState, useEffect, useCallback, useRef } from 'react';
import LorenzVisualizer from './LorenzVisualizer.jsx';
import ParameterControls from './ParameterControls.jsx';


function App() {
  const [params, setParams] = useState({
    sigma: 10.0,
    rho: 28.0,
    beta: 8.0 / 3.0,
    dt: 0.01,
    x0: 1.0,
    y0: 1.0,
    z0: 1.0,
    nSteps: 10000,
  });

  const [trajectories, setTrajectories] = useState([]);
  const [fixedPoints, setFixedPoints] = useState([]);
  const [lyapunovExponent, setLyapunovExponent] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [wasmModule, setWasmModule] = useState(null);

  const animationRef = useRef(null);
  const fullDataRef = useRef([]);

  useEffect(() => {
    async function loadWasm() {
      try {
        const wasm = await import('../pkg/lorenz_backend.js');
        await wasm.default();
        wasm.init_panic_hook();
        setWasmModule(wasm);
      } catch (error) {
        console.error('Failed to load WASM module:', error);
      }
    }
    loadWasm();
    return () => cancelAnimationFrame(animationRef.current);
  }, []);

  const handleParametersChange = useCallback((updates) => {
    setParams(prev => ({ ...prev, ...updates }));
  }, []);

  const animateTrajectory = useCallback((fullTrajectories) => {
    cancelAnimationFrame(animationRef.current);
    
    let currentStep = 0;
    const totalSteps = params.nSteps;
    const pointsPerFrame = Math.max(1, Math.ceil(totalSteps / 300)); 

    const step = () => {
      currentStep += pointsPerFrame;
      
      if (currentStep <= totalSteps) {
        const visibleData = fullTrajectories.map(traj => traj.slice(0, currentStep));
        setTrajectories(visibleData);
        animationRef.current = requestAnimationFrame(step);
      } else {
        setTrajectories(fullTrajectories);
        setIsLoading(false);
      }
    };

    animationRef.current = requestAnimationFrame(step);
  }, [params.nSteps]);

  const runSimulation = useCallback(async () => {
    if (!wasmModule) return;

    setIsLoading(true);
    setTrajectories([]);

    try {
      const sim = new wasmModule.LorenzSimulation(
        params.sigma,
        params.rho,
        params.beta,
        params.dt
      );

      const trajectoryJson = sim.compute_trajectory(
        params.x0, params.y0, params.z0,
        params.nSteps, 1
      );
      const trajectory = JSON.parse(trajectoryJson);
      
      const fixedPointsJson = sim.get_fixed_points();
      setFixedPoints(JSON.parse(fixedPointsJson));
      setLyapunovExponent(sim.compute_lyapunov_exponent(params.x0, params.y0, params.z0));

      animateTrajectory([trajectory]);
      
    } catch (error) {
      console.error('Simulation error:', error);
      setIsLoading(false);
    }
  }, [wasmModule, params, animateTrajectory]);

  const runButterflyEffect = useCallback(async () => {
    if (!wasmModule) return;

    setIsLoading(true);
    setTrajectories([]);

    try {
      const sim = new wasmModule.LorenzSimulation(
        params.sigma, params.rho, params.beta, params.dt
      );

      const trajectoriesJson = sim.compute_butterfly_effect(
        params.x0, params.y0, params.z0,
        0.001, 5, params.nSteps, 1
      );
      const allTrajectories = JSON.parse(trajectoriesJson);

      animateTrajectory(allTrajectories);

    } catch (error) {
      console.error('Butterfly effect error:', error);
      setIsLoading(false);
    }
  }, [wasmModule, params, animateTrajectory]);

  const reset = useCallback(() => {
    cancelAnimationFrame(animationRef.current);
    setTrajectories([]);
    setFixedPoints([]);
    setLyapunovExponent(null);
  }, []);

  return (
    <div style={{ display: 'flex', width: '100%', height: '100vh' }}>
      <ParameterControls
        {...params}
        onParametersChange={handleParametersChange}
        onSimulate={runSimulation}
        onReset={reset}
        lyapunovExponent={lyapunovExponent}
      />

      <div style={{ flex: 1, position: 'relative' }}>
        <LorenzVisualizer
          trajectories={trajectories}
          fixedPoints={fixedPoints}
          showAxes={true}
          showFixedPoints={true}
          animated={false} 
        />

        {isLoading && trajectories.length === 0 && (
          <div style={styles.loading}>
            <div style={styles.spinner} />
            <div style={styles.loadingText}>Computing...</div>
          </div>
        )}

        <div style={styles.info}>
          <h3 style={styles.infoTitle}>Lorenz Attractor</h3>
          <div style={styles.infoText}>
            <div>σ = {params.sigma.toFixed(2)}</div>
            <div>ρ = {params.rho.toFixed(2)}</div>
            <div>β = {params.beta.toFixed(4)}</div>
          </div>
          <div style={styles.infoText}>
            <div>Points: {trajectories[0]?.length || 0} / {params.nSteps}</div>
          </div>
        </div>

        <button
          onClick={runButterflyEffect}
          disabled={isLoading && trajectories.length === 0}
          style={styles.butterflyButton}
        >
          Butterfly Effect
        </button>
      </div>
    </div>
  );
}


const styles = {
  loading: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    textAlign: 'center',
    color: '#fff',
    zIndex: 100,
  },
  spinner: {
    width: '50px',
    height: '50px',
    border: '4px solid #333',
    borderTop: '4px solid #4ade80',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
    margin: '0 auto 15px',
  },
  loadingText: {
    fontFamily: 'monospace',
    fontSize: '14px',
  },
  info: {
    position: 'absolute',
    top: '20px',
    right: '20px',
    background: 'rgba(0, 0, 0, 0.8)',
    color: '#fff',
    padding: '15px',
    borderRadius: '8px',
    fontFamily: 'monospace',
    fontSize: '12px',
    maxWidth: '200px',
  },
  infoTitle: {
    fontSize: '14px',
    marginBottom: '10px',
    color: '#4ade80',
  },
  infoText: {
    marginBottom: '10px',
    lineHeight: '1.5',
  },
  controls: {
    fontSize: '10px',
    color: '#888',
    marginTop: '15px',
    display: 'flex',
    flexDirection: 'column',
    gap: '5px',
  },
  butterflyButton: {
    position: 'absolute',
    bottom: '20px',
    right: '20px',
    padding: '12px 20px',
    background: '#8b5cf6',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: 'bold',
    fontFamily: 'monospace',
  },
};

const styleSheet = document.createElement('style');
styleSheet.textContent = `
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  
  kbd {
    background: #333;
    padding: 2px 6px;
    border-radius: 3px;
    margin-right: 8px;
    font-size: 10px;
  }
`;
document.head.appendChild(styleSheet);

export default App;