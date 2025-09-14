


import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

// WebAssembly module will be loaded dynamically
let wasmModule = null;

function App() {
  // WebAssembly state
  const [wasmLoaded, setWasmLoaded] = useState(false);
  const [loadError, setLoadError] = useState(null);

  // Hénon map instances (will be WASM objects)
  const henonMapRef = useRef(null);
  const setValuedMapRef = useRef(null);

  // Parameters
  const [params, setParams] = useState({
    a: 1.4, 
    b: 0.3, 
    x0: 0.1, 
    y0: 0.1,
    epsilonX: 0.005, 
    epsilonY: 0.005,
    iterations: 1000,
    skipTransient: 500
  });

  // Visualization state
  const [isRunning, setIsRunning] = useState(false);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [trajectoryData, setTrajectoryData] = useState({
    deterministic: [],
    noisy: []
  });
  const [autoGenerate, setAutoGenerate] = useState(false);

  // Performance metrics
  const [performanceStats, setPerformanceStats] = useState({ 
    iterationsPerSecond: 0,
    lastRenderTime: 0 
  });

  // Refs for canvases
  const deterministicCanvasRef = useRef(null);
  const noisyCanvasRef = useRef(null);
  const animationRef = useRef(null);

  // Load WebAssembly module
  useEffect(() => {
    const loadWasm = async () => {
      try {
        console.log('Loading WebAssembly module...');
        
        // Import the generated WASM module
        const wasm = await import('./pkg/set_valued_viz.js');
        await wasm.default(); // Initialize the WASM module
        
        wasmModule = wasm;
        
        // Initialize utility
        wasm.Utils.init();
        console.log(`WASM Module version: ${wasm.Utils.version()}`);
        
        // Performance test
        const testDuration = wasm.Utils.performance_test(10000);
        const iterPerSec = Math.round(10000 / (testDuration / 1000));
        setPerformanceStats(prev => ({ ...prev, iterationsPerSecond: iterPerSec }));
        
        // Create initial Hénon map instances using with_parameters
        henonMapRef.current = wasm.HenonMap.with_parameters(params.a, params.b);
        setValuedMapRef.current = wasm.SetValuedHenonMap.with_parameters(params.a, params.b, params.epsilonX, params.epsilonY);
        
        setWasmLoaded(true);
        console.log('✅ WebAssembly module loaded successfully!');
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
        if (setValuedMapRef.current) {
          setValuedMapRef.current.free();
          setValuedMapRef.current = null;
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
      
      // Validate parameters before generation
      if (!isFinite(params.x0) || !isFinite(params.y0) || !isFinite(params.a) || !isFinite(params.b)) {
        throw new Error('Invalid parameters: contains non-finite values');
      }
      
      if (params.iterations < 1 || params.iterations > 50000) {
        throw new Error('Invalid iteration count: must be between 1 and 50000');
      }
      
      // Ensure fresh WASM objects before each generation to reset any internal state
      console.log('Creating fresh WASM objects for clean state...');
      
      // Free existing objects if they exist
      try {
        if (henonMapRef.current && typeof henonMapRef.current.free === 'function') {
          henonMapRef.current.free();
        }
        if (setValuedMapRef.current && typeof setValuedMapRef.current.free === 'function') {
          setValuedMapRef.current.free();
        }
      } catch (freeError) {
        console.warn('Error freeing existing WASM objects:', freeError);
      }
      
      // Create fresh WASM objects with current parameters
      henonMapRef.current = wasmModule.HenonMap.with_parameters(params.a, params.b);
      setValuedMapRef.current = wasmModule.SetValuedHenonMap.with_parameters(
        params.a, params.b, params.epsilonX, params.epsilonY
      );
      
      console.log('Fresh WASM objects created successfully');
      
      // Generate deterministic trajectory
      console.log('Generating deterministic trajectory...');
      const detTrajectory = henonMapRef.current.generate_trajectory(
        params.x0, params.y0, params.iterations, params.skipTransient
      );
      console.log('Deterministic trajectory length:', detTrajectory.length);
      
      // Validate deterministic trajectory
      if (!detTrajectory || detTrajectory.length === 0) {
        throw new Error('Failed to generate deterministic trajectory');
      }
      
      // Generate noisy trajectory  
      console.log('Generating noisy trajectory...');
      const noisyTrajectory = setValuedMapRef.current.generate_trajectory(
        params.x0, params.y0, params.iterations, params.skipTransient
      );
      console.log('Noisy trajectory length:', noisyTrajectory.length);
      
      // Validate noisy trajectory
      if (!noisyTrajectory || noisyTrajectory.length === 0) {
        throw new Error('Failed to generate noisy trajectory');
      }

      const detArray = Array.from(detTrajectory);
      const noisyArray = Array.from(noisyTrajectory);
      
      console.log('Sample deterministic points:', detArray.slice(0, 10));
      console.log('Sample noisy points:', noisyArray.slice(0, 10));
      
      // Check for invalid values in noisy data
      const invalidNoisyPoints = noisyArray.filter((val, idx) => !isFinite(val));
      if (invalidNoisyPoints.length > 0) {
        console.error('Found invalid values in noisy trajectory:', invalidNoisyPoints.length, 'out of', noisyArray.length);
        console.error('First few invalid values:', invalidNoisyPoints.slice(0, 10));
      }
      
      // Check for invalid values in deterministic data
      const invalidDetPoints = detArray.filter((val, idx) => !isFinite(val));
      if (invalidDetPoints.length > 0) {
        console.error('Found invalid values in deterministic trajectory:', invalidDetPoints.length, 'out of', detArray.length);
      }

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
      console.log(`✅ Generated ${params.iterations} iterations in ${renderTime.toFixed(2)}ms`);
    } catch (error) {
      console.error('❌ Error generating trajectories:', error);
    } finally {
      setIsRunning(false);
    }
  }, [params, wasmModule]);

  // Draw trajectory on canvas
  const drawTrajectory = useCallback((canvas, trajectory, color = '#00ff00', title = '') => {
    if (!canvas || !trajectory || trajectory.length === 0) {
      console.log(`Cannot draw ${title}:`, { canvas: !!canvas, trajectory: !!trajectory, length: trajectory?.length });
      return;
    }

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    console.log(`Drawing ${title} with ${trajectory.length / 2} points`);

    // Find bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < trajectory.length; i += 2) {
      const x = trajectory[i];
      const y = trajectory[i + 1];
      if (isFinite(x) && isFinite(y)) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }

    console.log(`${title} bounds:`, { minX, maxX, minY, maxY });

    // Check for valid bounds
    if (!isFinite(minX) || !isFinite(maxX) || !isFinite(minY) || !isFinite(maxY)) {
      console.error(`Invalid bounds for ${title}:`, { minX, maxX, minY, maxY });
      // Draw error message
      ctx.fillStyle = '#ffffff';
      ctx.font = '16px Arial';
      ctx.fillText(`Error: Invalid data for ${title}`, 10, 30);
      return;
    }

    // Add padding
    const padding = 0.1;
    const rangeX = maxX - minX;
    const rangeY = maxY - minY;
    
    // Prevent division by zero
    if (rangeX === 0 || rangeY === 0) {
      console.error(`Zero range for ${title}:`, { rangeX, rangeY });
      ctx.fillStyle = '#ffffff';
      ctx.font = '16px Arial';
      ctx.fillText(`Error: Zero range for ${title}`, 10, 30);
      return;
    }
    
    minX -= rangeX * padding;
    maxX += rangeX * padding;
    minY -= rangeY * padding;
    maxY += rangeY * padding;

    // Draw trajectory
    ctx.strokeStyle = color;
    ctx.lineWidth = 0.5;
    ctx.globalAlpha = 0.7;

    for (let i = 0; i < trajectory.length - 2; i += 2) {
      const x1 = ((trajectory[i] - minX) / (maxX - minX)) * width;
      const y1 = height - ((trajectory[i + 1] - minY) / (maxY - minY)) * height;
      const x2 = ((trajectory[i + 2] - minX) / (maxX - minX)) * width;
      const y2 = height - ((trajectory[i + 3] - minY) / (maxY - minY)) * height;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // Draw points
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.8;
    for (let i = 0; i < trajectory.length; i += 2) {
      const x = ((trajectory[i] - minX) / (maxX - minX)) * width;
      const y = height - ((trajectory[i + 1] - minY) / (maxY - minY)) * height;
      
      ctx.beginPath();
      ctx.arc(x, y, 0.5, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Draw title
    if (title) {
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#ffffff';
      ctx.font = '16px Arial';
      ctx.fillText(title, 10, 25);
    }

    // Draw iteration count
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px Arial';
    ctx.fillText(`Iterations: ${trajectory.length / 2}`, 10, height - 10);
  }, []);

  // Update visualizations when trajectory data changes
  useEffect(() => {
    if (trajectoryData.deterministic.length > 0) {
      drawTrajectory(deterministicCanvasRef.current, trajectoryData.deterministic, '#00ff00', 'Deterministic');
    }
    if (trajectoryData.noisy.length > 0) {
      drawTrajectory(noisyCanvasRef.current, trajectoryData.noisy, '#ff6b6b', 'With Bounded Noise');
    }
  }, [trajectoryData, drawTrajectory]);

  // Handle parameter changes with validation
  const handleParamChange = (field, value) => {
    const numValue = parseFloat(value);
    
    // Validate input
    if (!isFinite(numValue)) {
      console.warn(`Invalid value for ${field}:`, value);
      return;
    }
    
    // Additional field-specific validation
    if (field === 'iterations' && (numValue < 1 || numValue > 50000)) {
      console.warn('Iterations must be between 1 and 50000');
      return;
    }
    
    if ((field === 'epsilonX' || field === 'epsilonY') && numValue < 0) {
      console.warn('Epsilon values must be non-negative');
      return;
    }
    
    setParams(prev => ({
      ...prev,
      [field]: numValue
    }));
  };

  // Update WASM instances when parameters change
  useEffect(() => {
    if (!wasmModule || !wasmLoaded) return;

    try {
      // Safely free existing instances before creating new ones
      if (henonMapRef.current) {
        try {
          henonMapRef.current.free();
        } catch (e) {
          console.warn('Error freeing HenonMap:', e);
        }
        henonMapRef.current = null;
      }

      if (setValuedMapRef.current) {
        try {
          setValuedMapRef.current.free();
        } catch (e) {
          console.warn('Error freeing SetValuedHenonMap:', e);
        }
        setValuedMapRef.current = null;
      }

      // Create new instances with updated parameters
      henonMapRef.current = wasmModule.HenonMap.with_parameters(params.a, params.b);
      setValuedMapRef.current = wasmModule.SetValuedHenonMap.with_parameters(
        params.a, params.b, params.epsilonX, params.epsilonY
      );
    } catch (error) {
      console.error('Error updating parameters:', error);
      // Reset to null if creation fails
      henonMapRef.current = null;
      setValuedMapRef.current = null;
    }
  }, [params.a, params.b, params.epsilonX, params.epsilonY, wasmModule, wasmLoaded]);

  // Auto-generate when parameters change (only if enabled)
  useEffect(() => {
    if (autoGenerate && wasmLoaded && henonMapRef.current && setValuedMapRef.current) {
      generateTrajectories();
    }
  }, [autoGenerate, wasmLoaded, generateTrajectories]);

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
        <h1>Hénon Map Visualization</h1>
        <p>Real-time comparison of deterministic vs. set-valued dynamical systems</p>
      </header>

      <div className="main-content">
        <aside className="controls-panel">
          <div className="controls-section">
            <h3>System Parameters</h3>
            <div className="form-row">
              <div className="form-group">
                <label>Parameter a</label>
                <input
                  type="number"
                  step="0.01"
                  value={params.a}
                  onChange={(e) => handleParamChange('a', e.target.value)}
                />
              </div>
              <div className="form-group">
                <label>Parameter b</label>
                <input
                  type="number"
                  step="0.01"
                  value={params.b}
                  onChange={(e) => handleParamChange('b', e.target.value)}
                />
              </div>
            </div>
          </div>

          <div className="controls-section">
            <h3>Initial Conditions</h3>
            <div className="form-row">
              <div className="form-group">
                <label>Initial x₀</label>
                <input
                  type="number"
                  step="0.01"
                  value={params.x0}
                  onChange={(e) => handleParamChange('x0', e.target.value)}
                />
              </div>
              <div className="form-group">
                <label>Initial y₀</label>
                <input
                  type="number"
                  step="0.01"
                  value={params.y0}
                  onChange={(e) => handleParamChange('y0', e.target.value)}
                />
              </div>
            </div>
          </div>

          <div className="controls-section">
            <h3>Noise Parameters</h3>
            <div className="form-row">
              <div className="form-group">
                <label>Epsilon X (εₓ)</label>
                <input
                  type="number"
                  step="0.001"
                  min="0"
                  value={params.epsilonX}
                  onChange={(e) => handleParamChange('epsilonX', e.target.value)}
                />
              </div>
              <div className="form-group">
                <label>Epsilon Y (εᵧ)</label>
                <input
                  type="number"
                  step="0.001"
                  min="0"
                  value={params.epsilonY}
                  onChange={(e) => handleParamChange('epsilonY', e.target.value)}
                />
              </div>
            </div>
          </div>

          <div className="controls-section">
            <h3>Simulation Settings</h3>
            <div className="form-group">
              <label>Iterations</label>
              <input
                type="number"
                min="100"
                max="50000"
                step="100"
                value={params.iterations}
                onChange={(e) => handleParamChange('iterations', e.target.value)}
              />
            </div>
            <div className="form-group">
              <label>Skip Transient</label>
              <input
                type="number"
                min="0"
                max="1000"
                step="10"
                value={params.skipTransient}
                onChange={(e) => handleParamChange('skipTransient', e.target.value)}
              />
            </div>
          </div>

          <div className="controls-section">
            <h3>Generation Controls</h3>
            <button 
              className="button button-primary"
              onClick={generateTrajectories}
              disabled={!wasmLoaded || isRunning}
              style={{ width: '100%', marginBottom: '12px' }}
            >
              {isRunning ? 'Generating...' : 'Generate Trajectories'}
            </button>
            
            <div className="form-group">
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <input
                  type="checkbox"
                  checked={autoGenerate}
                  onChange={(e) => setAutoGenerate(e.target.checked)}
                />
                Auto-generate on parameter change
              </label>
            </div>
          </div>
        </aside>

        <main className="visualization-area">
          <div className="visualization-grid">
            <div className="canvas-container">
              <h3>Deterministic Hénon Map</h3>
              <canvas
                ref={deterministicCanvasRef}
                width={600}
                height={400}
                style={{ width: '100%', height: 'auto' }}
              />
            </div>

            <div className="canvas-container">
              <h3>Set-Valued (Noisy) Hénon Map</h3>
              <canvas
                ref={noisyCanvasRef}
                width={600}
                height={400}
                style={{ width: '100%', height: 'auto' }}
              />
            </div>
          </div>

          <div className="stats-panel">
            <div className="stats-grid">
              <div className="stat-item">
                <div className="stat-value">{currentIteration.toLocaleString()}</div>
                <div className="stat-label">Current Iterations</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{performanceStats.iterationsPerSecond.toLocaleString()}</div>
                <div className="stat-label">Iterations/sec</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{performanceStats.lastRenderTime.toFixed(1)}ms</div>
                <div className="stat-label">Render Time</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{(trajectoryData.deterministic.length / 2).toLocaleString()}</div>
                <div className="stat-label">Det. Points</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{(trajectoryData.noisy.length / 2).toLocaleString()}</div>
                <div className="stat-label">Noisy Points</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">
                  {(() => {
                    try {
                      return wasmModule?.Utils?.version() || 'N/A';
                    } catch (error) {
                      console.error('Error getting WASM version:', error);
                      return 'Error';
                    }
                  })()}
                </div>
                <div className="stat-label">WASM Version</div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;