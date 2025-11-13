import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import Algorithm1Viz from './Algorithm1Viz';

// WebAssembly module will be loaded dynamically
let wasmModule = null;

function App() {
  // WebAssembly state
  const [wasmLoaded, setWasmLoaded] = useState(false);
  const [loadError, setLoadError] = useState(null);

  // Hénon map instances (will be WASM objects)
  const henonMapRef = useRef(null);
  const setValuedSimulationRef = useRef(null);

  const [algorithm1VisualizationMode, setAlgorithm1VisualizationMode] = useState('all');
  const [currentStepData, setCurrentStepData] = useState(null);

  // Parameters
  const [params, setParams] = useState({
    a: 1.4,
    b: 0.3,
    x0: 0.6,
    y0: 0.3,
    epsilonX: 0.05,
    epsilonY: 0.05,
    iterations: 100,
    skipTransient: 0,
    algorithm1Epsilon: 0.1,
    nBoundaryPoints: 8,
    convergenceThreshold: 1e-6,
    maxAlgorithm1Iterations: 100
  });
  const [isRunning, setIsRunning] = useState(false);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [trajectoryData, setTrajectoryData] = useState({
    deterministic: [],
    noisy: [],
    currentIndex: 0,
  });
  const [showNoiseCircles, setShowNoiseCircles] = useState(true);

  // Algorithm 1 simulation state
  const [algorithm1Data, setAlgorithm1Data] = useState({
    boundaryHistory: [],
    normalVectors: [],
    currentIteration: 0,
    isConverged: false,
    convergenceHistory: [],
    centroid: [0, 0],
    area: 0
  });
  const [isAlgorithm1Running, setIsAlgorithm1Running] = useState(false);
  const [showNormals, setShowNormals] = useState(false);

  // Incremental iteration state
  const [currentState, setCurrentState] = useState({ x: 0, y: 0 });
  const currentStateRef = useRef({ x: 0, y: 0 });

  // Performance metrics
  const [performanceStats, setPerformanceStats] = useState({
    iterationsPerSecond: 0,
    lastRenderTime: 0
  });

  // Refs for canvases
  const deterministicCanvasRef = useRef(null);
  const noisyCanvasRef = useRef(null);
  const algorithm1CanvasRef = useRef(null);

  // Load WebAssembly module
  useEffect(() => {
    const loadWasm = async () => {
      try {
        console.log('Loading WebAssembly module...');

        const wasm = await import('../pkg/set_valued_viz.js');
        await wasm.default();

        wasmModule = wasm;

        wasm.Utils.init();
        console.log(`WASM Module version: ${wasm.Utils.version()}`);

        const testDuration = wasm.Utils.performance_test(10000);
        const iterPerSec = Math.round(10000 / (testDuration / 1000));
        setPerformanceStats(prev => ({ ...prev, iterationsPerSecond: iterPerSec }));

        henonMapRef.current = wasm.HenonMap.withParameters(params.a, params.b);

        const systemParams = new wasm.SystemParameters(params.a, params.b);

        setValuedSimulationRef.current = new wasm.SetValuedSimulation(
          systemParams,
          params.algorithm1Epsilon,
          params.x0,
          params.y0,
          params.nBoundaryPoints
        );

        setWasmLoaded(true);
        console.log('WebAssembly module loaded successfully');

        setCurrentState({ x: params.x0, y: params.y0 });
        currentStateRef.current = { x: params.x0, y: params.y0 };
      } catch (error) {
        console.error('Failed to load WebAssembly module:', error);
        setLoadError(error.message);
      }
    };
    loadWasm();
  }, []);

  // Cleanup WASM instances on unmount
  useEffect(() => {
    return () => {
      try {
        if (henonMapRef.current) {
          henonMapRef.current.free();
          henonMapRef.current = null;
        }
        if (setValuedSimulationRef.current) {
          setValuedSimulationRef.current.free();
          setValuedSimulationRef.current = null;
        }
      } catch (error) {
        console.warn('Error during cleanup:', error);
      }
    };
  }, []);

  // Generate trajectories
  const generateTrajectories = useCallback(() => {
    if (!wasmModule) {
      console.log('WASM not ready:', { wasmModule: !!wasmModule });
      return;
    }

    setIsRunning(true);
    const startTime = performance.now();

    try {
      console.log('Generating trajectories with params:', params);

      if (!isFinite(params.x0) || !isFinite(params.y0) || !isFinite(params.a) || !isFinite(params.b)) {
        throw new Error('Invalid parameters: contains non-finite values');
      }

      if (params.iterations < 1 || params.iterations > 50000) {
        throw new Error('Invalid iteration count: must be between 1 and 50000');
      }

      console.log('Creating fresh Henon map...');

      if (henonMapRef.current) {
        try {
          henonMapRef.current.free();
          henonMapRef.current = null;
        } catch (e) {
          console.warn('Error freeing Henon map:', e);
        }
      }

      henonMapRef.current = wasmModule.HenonMap.withParameters(params.a, params.b);

      console.log('Fresh Henon map created successfully');

      console.log('Generating deterministic trajectory...');
      const detTrajectory = henonMapRef.current.generate_trajectory(
        params.x0, params.y0, params.iterations, params.skipTransient
      );
      console.log('Deterministic trajectory length:', detTrajectory.length);

      if (!detTrajectory || detTrajectory.length === 0) {
        throw new Error('Failed to generate deterministic trajectory');
      }

      const detArray = Array.from(detTrajectory);

      const noisyArray = [];
      for (let i = 0; i < detArray.length; i += 2) {
        const x = detArray[i];
        const y = detArray[i + 1];

        const angle = Math.random() * 2 * Math.PI;
        const radius = Math.random() * params.epsilonX;
        const noiseX = radius * Math.cos(angle);
        const noiseY = radius * Math.sin(angle);

        noisyArray.push(x + noiseX);
        noisyArray.push(y + noiseY);
      }

      console.log('Sample deterministic points:', detArray.slice(0, 10));
      console.log('Sample noisy points:', noisyArray.slice(0, 10));

      setTrajectoryData({
        deterministic: detArray,
        noisy: noisyArray
      });

      const endTime = performance.now();
      const renderTime = endTime - startTime;
      setPerformanceStats(prev => ({
        ...prev,
        lastRenderTime: renderTime,
        iterationsPerSecond: Math.round(params.iterations / (renderTime / 1000))
      }));

      setCurrentIteration(params.iterations);
      console.log(`Generated ${params.iterations} iterations in ${renderTime.toFixed(2)}ms`);
    } catch (error) {
      console.error('Error generating trajectories:', error);
    } finally {
      setIsRunning(false);
    }
  }, [params, wasmModule]);

  // Algorithm 1 simulation functions
  const initializeAlgorithm1Simulation = useCallback(() => {
    if (!wasmModule) {
      console.log('WASM not ready for Algorithm 1 simulation');
      return;
    }

    try {
      console.log('Initializing Algorithm 1 simulation...');
      console.log('Params:', { a: params.a, b: params.b, epsilon: params.algorithm1Epsilon, x0: params.x0, y0: params.y0, n: params.nBoundaryPoints });

      if (setValuedSimulationRef.current) {
        try {
          setValuedSimulationRef.current.free();
        } catch (e) {
          console.warn('Error freeing old simulation:', e);
        }
        setValuedSimulationRef.current = null;
      }

      const systemParams = new wasmModule.SystemParameters(params.a, params.b);
      console.log('SystemParameters created');

      setValuedSimulationRef.current = new wasmModule.SetValuedSimulation(
        systemParams,
        params.algorithm1Epsilon,
        params.x0,
        params.y0,
        params.nBoundaryPoints
      );
      console.log('SetValuedSimulation created');

      const initialBoundary = setValuedSimulationRef.current.getBoundaryPositions();
      console.log('Initial boundary retrieved:', initialBoundary.length / 2, 'points');

      const initialStateJson = setValuedSimulationRef.current.get_initial_state_details();
      const initialState = JSON.parse(initialStateJson);
      console.log('Initial state details:', initialState);

      setCurrentStepData(initialState);

      setAlgorithm1Data({
        boundaryHistory: [Array.from(initialBoundary)],
        normalVectors: [],
        currentIteration: 0,
        isConverged: false,
        convergenceHistory: [0],
        centroid: [0, 0],
        area: 0
      });

      console.log('Algorithm 1 simulation initialized successfully');

    } catch (error) {
      console.error('Error initializing Algorithm 1 simulation:', error);
      console.error('Error stack:', error.stack);
      setValuedSimulationRef.current = null;
    }
  }, [params.a, params.b, params.algorithm1Epsilon, params.x0, params.y0, params.nBoundaryPoints, wasmModule]);

  const runAlgorithm1Simulation = useCallback(async () => {
    if (!wasmModule || !setValuedSimulationRef.current) {
      console.log('WASM or simulation not ready');
      return;
    }

    setIsAlgorithm1Running(true);
    const startTime = performance.now();

    try {
      console.log('Running Algorithm 1 simulation...');

      const history = await setValuedSimulationRef.current.track_boundary_evolution(
        params.maxAlgorithm1Iterations
      );

      const iterations = [];
      let currentIteration = [];

      for (let i = 0; i < history.length; i++) {
        if (isNaN(history[i])) {
          if (currentIteration.length > 0) {
            iterations.push(Array.from(currentIteration));
            currentIteration = [];
          }
        } else {
          currentIteration.push(history[i]);
        }
      }

      if (currentIteration.length > 0) {
        iterations.push(Array.from(currentIteration));
      }

      const finalIteration = setValuedSimulationRef.current.getIterationCount();

      setAlgorithm1Data({
        boundaryHistory: iterations,
        normalVectors: [],
        currentIteration: finalIteration,
        isConverged: finalIteration < params.maxAlgorithm1Iterations,
        convergenceHistory: iterations.map((_, idx) => idx),
        centroid: [0, 0],
        area: 0
      });

      const endTime = performance.now();
      console.log(`Algorithm 1 simulation completed in ${(endTime - startTime).toFixed(2)}ms`);
      console.log(`Converged after ${finalIteration} iterations`);

    } catch (error) {
      console.error('Error running Algorithm 1 simulation:', error);
    } finally {
      setIsAlgorithm1Running(false);
    }
  }, [params, wasmModule]);

  const stepAlgorithm1SimulationDetailed = useCallback(() => {
    if (!wasmModule || !setValuedSimulationRef.current) {
      console.log('WASM or simulation not ready for stepping');
      return;
    }

    try {
      console.log(`Stepping Algorithm 1 with details: iteration ${algorithm1Data.currentIteration}`);

      const currentBoundary = setValuedSimulationRef.current.getBoundaryPositions();

      const detailsJson = setValuedSimulationRef.current.iterate_boundary_with_details();

      if (!detailsJson) {
        console.error('No details returned from iteration');
        return;
      }

      const details = JSON.parse(detailsJson);

      if (details.diverged) {
        console.warn("System diverged during iteration. Stopping simulation.");

        setAlgorithm1Data(prev => ({
          ...prev,
          isDiverged: true,
          currentIteration: details.iteration,
          divergenceMessage: `Boundary diverged: ${details.points_lost} points lost, ${details.points_remaining} remaining`
        }));

        return;
      }

      setCurrentStepData(details);

      const iteration = details.iteration || (algorithm1Data.currentIteration + 1);

      setAlgorithm1Data(prev => ({
        boundaryHistory: [...prev.boundaryHistory, Array.from(details.projected_points || [])],
        normalVectors: details.normals || [],
        currentIteration: iteration,
        isConverged: details.max_movement < 1e-6,
        convergenceHistory: [...prev.convergenceHistory, iteration],
        centroid: [0, 0],
        area: 0
      }));

      console.log(`Algorithm 1 step ${iteration} completed`);
      console.log(`   Mapped points: ${details.mapped_points.length / 2}`);
      console.log(`   Projected points: ${details.projected_points.length / 2}`);
      console.log(`   Max movement: ${details.max_movement}`);

    } catch (error) {
      console.error('Error stepping Algorithm 1 simulation:', error);
      console.error('Error details:', error.message, error.stack);
    }
  }, [wasmModule, algorithm1Data.currentIteration]);

  const resetAlgorithm1Simulation = useCallback(() => {
    setCurrentStepData(null);
    initializeAlgorithm1Simulation();
  }, [initializeAlgorithm1Simulation]);

  const handleParamChange = (field, value) => {
    const numValue = parseFloat(value);

    if (!isFinite(numValue)) {
      console.warn(`Invalid value for ${field}:`, value);
      return;
    }

    if (field === 'iterations' && (numValue < 1 || numValue > 50000)) {
      console.warn('Iterations must be between 1 and 50000');
      return;
    }

    if ((field === 'epsilonX' || field === 'epsilonY' || field === 'algorithm1Epsilon') && numValue < 0) {
      console.warn('Epsilon values must be non-negative');
      return;
    }

    setParams(prev => ({
      ...prev,
      [field]: numValue
    }));
  };

  useEffect(() => {
    if (wasmLoaded && wasmModule) {
      initializeAlgorithm1Simulation();
    }
  }, [wasmLoaded, wasmModule, initializeAlgorithm1Simulation]);

  if (loadError) {
    return (
      <div className="app">
        <div className="loading-overlay">
          <div className="loading-content">
            <h2>Error Loading WebAssembly</h2>
            <div className="error-message">
              {loadError}
            </div>
            <button className="button button-primary" onClick={() => window.location.reload()}>
              Reload Page
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!wasmLoaded) {
    return (
      <div className="app">
        <div className="loading-overlay">
          <div className="loading-content">
            <div className="spinner"></div>
            <h2>Loading WebAssembly Module</h2>
            <p>Initializing Hénon Map simulation...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Hénon Map with Algorithm 1 Visualization</h1>
        <p>Real-time visualization of deterministic and set-valued dynamical systems</p>
      </header>

      <div className="container">
        <aside className="sidebar">
          <div className="controls-section">
            <h3>System Parameters</h3>
            <div className="form-group">
              <label>
                Parameter a: {params.a.toFixed(3)}
                <input
                  type="range"
                  min="0"
                  max="1.6"
                  step="0.01"
                  value={params.a}
                  onChange={(e) => handleParamChange('a', e.target.value)}
                />
              </label>
            </div>
            <div className="form-group">
              <label>
                Parameter b: {params.b.toFixed(3)}
                <input
                  type="range"
                  min="0.1"
                  max="0.55"
                  step="0.01"
                  value={params.b}
                  onChange={(e) => handleParamChange('b', e.target.value)}
                />
              </label>
            </div>
            <div className="form-group">
              <label>
                Initial x₀: {params.x0.toFixed(3)}
                <input
                  type="range"
                  min="-0.5"
                  max="0.5"
                  step="0.01"
                  value={params.x0}
                  onChange={(e) => handleParamChange('x0', e.target.value)}
                />
              </label>
            </div>
            <div className="form-group">
              <label>
                Initial y₀: {params.y0.toFixed(3)}
                <input
                  type="range"
                  min="-0.5"
                  max="0.5"
                  step="0.01"
                  value={params.y0}
                  onChange={(e) => handleParamChange('y0', e.target.value)}
                />
              </label>
            </div>
          </div>

          <div className="controls-section">
            <h3>Algorithm 1 Parameters</h3>
            <div className="form-group">
              <label>
                Boundary ε: {params.algorithm1Epsilon.toFixed(3)}
                <input
                  type="range"
                  min="0.02"
                  max="0.3"
                  step="0.01"
                  value={params.algorithm1Epsilon}
                  onChange={(e) => handleParamChange('algorithm1Epsilon', e.target.value)}
                />
              </label>
            </div>
            <div className="form-group">
              <label>
                Boundary Points: {params.nBoundaryPoints}
                <input
                  type="range"
                  min="6"
                  max="1000"
                  step="1"
                  value={params.nBoundaryPoints}
                  onChange={(e) => handleParamChange('nBoundaryPoints', e.target.value)}
                />
              </label>
            </div>
            <div className="form-group">
              <label>
                Max Iterations: {params.maxAlgorithm1Iterations}
                <input
                  type="range"
                  min="10"
                  max="200"
                  step="10"
                  value={params.maxAlgorithm1Iterations}
                  onChange={(e) => handleParamChange('maxAlgorithm1Iterations', e.target.value)}
                />
              </label>
            </div>
          </div>

          <div className="controls-section">
            <h3>Visualization Mode</h3>
            <div className="form-group">
              <label>
                Display Mode:
                <select
                  value={algorithm1VisualizationMode}
                  onChange={(e) => setAlgorithm1VisualizationMode(e.target.value)}
                >
                  <option value="all">All Steps (Default)</option>
                  <option value="mapped">Step 1: Mapped Points f(zₖ)</option>
                  <option value="circles">Step 2: Noise Circles</option>
                  <option value="projected">Step 3: Envelope Boundary</option>
                </select>
              </label>
            </div>
          </div>

          <div className="controls-section">
            <h3>Algorithm 1 Controls</h3>
            <div className="main-controls">
              <button
                className="button button-primary"
                onClick={stepAlgorithm1SimulationDetailed}
                disabled={!wasmLoaded || isAlgorithm1Running || algorithm1Data.isDiverged}
              >
                Step Forward
              </button>
              <button
                className="button button-secondary"
                onClick={runAlgorithm1Simulation}
                disabled={!wasmLoaded || isAlgorithm1Running}
              >
                {isAlgorithm1Running ? 'Running...' : 'Run Full'}
              </button>
              <button
                className="button button-secondary"
                onClick={resetAlgorithm1Simulation}
              >
                Reset
              </button>
            </div>
          </div>
        </aside>

        <main className="visualization-area">
          <div className="canvas-container" style={{ gridColumn: '1 / -1' }}>
            <h3>Algorithm 1: Boundary Evolution</h3>
            <Algorithm1Viz
              boundaryHistory={algorithm1Data.boundaryHistory}
              currentIteration={algorithm1Data.currentIteration}
              epsilon={params.algorithm1Epsilon}
              nBoundaryPoints={params.nBoundaryPoints}
              isConverged={algorithm1Data.isConverged}
              stepData={currentStepData}
              visualizationMode={algorithm1VisualizationMode}
              showDetailedViz={currentStepData !== null}
              isDiverged={algorithm1Data.isDiverged}
            />
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;