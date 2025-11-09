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

  const [algorithm1VisualizationMode, setAlgorithm1VisualizationMode] = useState('all'); // 'mapped', 'circles', 'projected', 'all'
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
    // Algorithm 1 parameters
    algorithm1Epsilon: 0.1,
    nBoundaryPoints: 8,
    convergenceThreshold: 1e-6,
    maxAlgorithm1Iterations: 100
  });
  // Visualization state
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
  const [showAlgorithm1Visualization, setShowAlgorithm1Visualization] = useState(true);
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

        // Import the generated WASM module
        const wasm = await import('../../pkg/set_valued_viz.js'); // changed from '../pkg/set_valued_viz.js'
        await wasm.default(); // Initialize the WASM module

        wasmModule = wasm;

        // Initialize utility
        wasm.Utils.init();
        console.log(`WASM Module version: ${wasm.Utils.version()}`);

        // Performance test
        const testDuration = wasm.Utils.performance_test(10000);
        const iterPerSec = Math.round(10000 / (testDuration / 1000));
        setPerformanceStats(prev => ({ ...prev, iterationsPerSecond: iterPerSec }));

        // Create initial Hénon map instance using with_parameters (your current API)
        henonMapRef.current = wasm.HenonMap.withParameters(params.a, params.b);

        const systemParams = new wasm.SystemParameters(params.a, params.b);

        // Create initial Algorithm 1 simulation instance
        setValuedSimulationRef.current = new wasm.SetValuedSimulation(
          systemParams,
          params.algorithm1Epsilon,
          params.x0, // initial_center_x
          params.y0, // initial_center_y
          params.nBoundaryPoints
        );

        setWasmLoaded(true);
        console.log('✅ WebAssembly module loaded successfully!');

        // Initialize incremental mode
        setCurrentState({ x: params.x0, y: params.y0 });
        currentStateRef.current = { x: params.x0, y: params.y0 };

        // Don't free systemParams - it's stored in SetValuedSimulation
        // systemParams.free();
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

      // Validate parameters
      if (!isFinite(params.x0) || !isFinite(params.y0) || !isFinite(params.a) || !isFinite(params.b)) {
        throw new Error('Invalid parameters: contains non-finite values');
      }

      if (params.iterations < 1 || params.iterations > 50000) {
        throw new Error('Invalid iteration count: must be between 1 and 50000');
      }

      // Create fresh Henon map
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

      // Generate deterministic trajectory
      console.log('Generating deterministic trajectory...');
      const detTrajectory = henonMapRef.current.generate_trajectory(
        params.x0, params.y0, params.iterations, params.skipTransient
      );
      console.log('Deterministic trajectory length:', detTrajectory.length);

      if (!detTrajectory || detTrajectory.length === 0) {
        throw new Error('Failed to generate deterministic trajectory');
      }

      // For noisy trajectory, we'll use the deterministic one with added noise in visualization
      // since SetValuedHenonMap doesn't exist in your current WASM
      const detArray = Array.from(detTrajectory);

      // Simulate noisy trajectory by adding random noise
      const noisyArray = [];
      for (let i = 0; i < detArray.length; i += 2) {
        const x = detArray[i];
        const y = detArray[i + 1];

        // Add circular noise
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
      console.log(`✅ Generated ${params.iterations} iterations in ${renderTime.toFixed(2)}ms`);
    } catch (error) {
      console.error('❌ Error generating trajectories:', error);
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

      // Free existing simulation if it exists
      if (setValuedSimulationRef.current) {
        try {
          setValuedSimulationRef.current.free();
        } catch (e) {
          console.warn('Error freeing old simulation:', e);
        }
        setValuedSimulationRef.current = null;
      }

      // Create system parameters (don't store in ref, will be freed)
      const systemParams = new wasmModule.SystemParameters(params.a, params.b);
      console.log('✓ SystemParameters created');

      // Create simulation
      setValuedSimulationRef.current = new wasmModule.SetValuedSimulation(
        systemParams,
        params.algorithm1Epsilon,
        params.x0,
        params.y0,
        params.nBoundaryPoints
      );
      console.log('✓ SetValuedSimulation created');

      // Note: Don't free systemParams here - WASM binding needs it
      // The SetValuedSimulation stores the params internally
      // systemParams.free(); // This causes "memory access out of bounds"
      console.log('✓ SystemParameters passed to simulation');

      // Get initial boundary
      const initialBoundary = setValuedSimulationRef.current.getBoundaryPositions();
      console.log('✓ Initial boundary retrieved:', initialBoundary.length / 2, 'points');

      // Get initial state details for visualization
      const initialStateJson = setValuedSimulationRef.current.get_initial_state_details();
      const initialState = JSON.parse(initialStateJson);
      console.log('✓ Initial state details:', initialState);

      // Set step data to show initial state
      setCurrentStepData(initialState);

      // Set initial algorithm data
      setAlgorithm1Data({
        boundaryHistory: [Array.from(initialBoundary)],
        normalVectors: [],
        currentIteration: 0,
        isConverged: false,
        convergenceHistory: [0],
        centroid: [0, 0],
        area: 0
      });

      console.log('✅ Algorithm 1 simulation initialized successfully');

    } catch (error) {
      console.error('❌ Error initializing Algorithm 1 simulation:', error);
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

      // Parse the history (separated by NaN values)
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

      // Add the last iteration if it exists
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
      console.log(`✅ Algorithm 1 simulation completed in ${(endTime - startTime).toFixed(2)}ms`);
      console.log(`Converged after ${finalIteration} iterations`);

    } catch (error) {
      console.error('❌ Error running Algorithm 1 simulation:', error);
    } finally {
      setIsAlgorithm1Running(false);
    }
  }, [params, wasmModule]);


  // Step through Algorithm 1 with detailed visualization
  const stepAlgorithm1SimulationDetailed = useCallback(() => {
    if (!wasmModule || !setValuedSimulationRef.current) {
      console.log('WASM or simulation not ready for stepping');
      return;
    }

    try {
      console.log(`📍 Stepping Algorithm 1 with details: iteration ${algorithm1Data.currentIteration}`);

      // Get current boundary BEFORE iteration
      const currentBoundary = setValuedSimulationRef.current.getBoundaryPositions();

      // Perform iteration and get detailed data
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

      // Update step data for visualization
      setCurrentStepData(details);

      // Get iteration count
      const iteration = details.iteration || (algorithm1Data.currentIteration + 1);

      // Update algorithm data
      setAlgorithm1Data(prev => ({
        boundaryHistory: [...prev.boundaryHistory, Array.from(details.projected_points || [])],
        normalVectors: details.normals || [],
        currentIteration: iteration,
        isConverged: details.max_movement < 1e-6,
        convergenceHistory: [...prev.convergenceHistory, iteration],
        centroid: [0, 0],
        area: 0
      }));

      console.log(`✅ Algorithm 1 step ${iteration} completed`);
      console.log(`   Mapped points: ${details.mapped_points.length / 2}`);
      console.log(`   Projected points: ${details.projected_points.length / 2}`);
      console.log(`   Max movement: ${details.max_movement}`);

    } catch (error) {
      console.error('❌ Error stepping Algorithm 1 simulation:', error);
      console.error('Error details:', error.message, error.stack);
    }
  }, [wasmModule, algorithm1Data.currentIteration]);

  // broken 
  const stepAlgorithm1Simulation = useCallback(() => {
    if (!wasmModule || !setValuedSimulationRef.current) {
      console.log('WASM or simulation not ready for stepping');
      return;
    }

    try {
      console.log(`📍 Stepping Algorithm 1: iteration ${algorithm1Data.currentIteration}`);

      // Perform one iteration
      setValuedSimulationRef.current.iterate_boundary();

      // Get new boundary
      const newBoundary = setValuedSimulationRef.current.getBoundaryPositions();
      const iteration = setValuedSimulationRef.current.getIterationCount();

      // Update state
      setAlgorithm1Data(prev => ({
        boundaryHistory: [...prev.boundaryHistory, Array.from(newBoundary)],
        normalVectors: prev.normalVectors,
        currentIteration: iteration,
        isConverged: false,
        convergenceHistory: [...prev.convergenceHistory, iteration],
        centroid: [0, 0],
        area: 0
      }));

      console.log(`✅ Algorithm 1 step ${iteration} completed with ${newBoundary.length / 2} boundary points`);
    } catch (error) {
      console.error('❌ Error stepping Algorithm 1 simulation:', error);
    }
  }, [wasmModule, algorithm1Data.currentIteration]);

  const resetAlgorithm1Simulation = useCallback(() => {
    setCurrentStepData(null);
    initializeAlgorithm1Simulation();
  }, [initializeAlgorithm1Simulation]);

  // Draw trajectory on canvas
  const drawTrajectory = useCallback((canvas, trajectory, color = '#00ff00', title = '', showNoise = false) => {
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

    // Check for valid bounds
    if (!isFinite(minX) || !isFinite(maxX) || !isFinite(minY) || !isFinite(maxY)) {
      console.error(`Invalid bounds for ${title}:`, { minX, maxX, minY, maxY });
      ctx.fillStyle = '#ffffff';
      ctx.font = '16px Arial';
      ctx.fillText(`Error: Invalid data for ${title}`, 10, 30);
      return;
    }

    // Add padding
    const padding = 0.1;
    const rangeX = maxX - minX;
    const rangeY = maxY - minY;

    if (rangeX === 0 || rangeY === 0 || rangeX < 1e-10 || rangeY < 1e-10) {
      const defaultRange = 0.5;
      minX = minX - defaultRange;
      maxX = maxX + defaultRange;
      minY = minY - defaultRange;
      maxY = maxY + defaultRange;
    } else {
      minX -= rangeX * padding;
      maxX += rangeX * padding;
      minY -= rangeY * padding;
      maxY += rangeY * padding;
    }

    // Coordinate transformation
    const scaleX = (x) => ((x - minX) / (maxX - minX)) * width;
    const scaleY = (y) => height - ((y - minY) / (maxY - minY)) * height;

    // Draw noise circles if requested
    if (showNoise && showNoiseCircles) {
      const epsilonRadius = (params.epsilonX / (maxX - minX)) * width;

      ctx.strokeStyle = '#ffaa00';
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.3;

      for (let i = 0; i < trajectory.length; i += 2) {
        const x = scaleX(trajectory[i]);
        const y = scaleY(trajectory[i + 1]);

        ctx.beginPath();
        ctx.arc(x, y, epsilonRadius, 0, 2 * Math.PI);
        ctx.stroke();
      }

      ctx.globalAlpha = 1.0;
    }

    // Draw points
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.8;
    for (let i = 0; i < trajectory.length; i += 2) {
      const x = scaleX(trajectory[i]);
      const y = scaleY(trajectory[i + 1]);

      ctx.beginPath();
      ctx.arc(x, y, 0.5, 0, 2 * Math.PI);
      ctx.fill();
    }

    ctx.globalAlpha = 1.0;

    // Draw title
    if (title) {
      ctx.fillStyle = '#ffffff';
      ctx.font = '16px Arial';
      ctx.fillText(title, 10, 25);
    }

    // Draw iteration count
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px Arial';
    ctx.fillText(`Iterations: ${trajectory.length / 2}`, 10, height - 10);
  }, [params.epsilonX, showNoiseCircles]);



  const drawAlgorithm1BoundaryDetailed = useCallback((canvas, stepData, visualizationMode) => {
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    if (!stepData) {
      ctx.fillStyle = '#ffffff';
      ctx.font = '16px Arial';
      ctx.fillText('Click "Step" to start visualization', width / 2 - 100, height / 2);
      return;
    }

    const { mapped_points, projected_points, normals, epsilon, iteration } = stepData;

    if (!mapped_points || mapped_points.length === 0) return;

    // Find bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

    for (let i = 0; i < projected_points.length; i += 2) {
      const x = projected_points[i];
      const y = projected_points[i + 1];
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }

    // Add padding
    const padding = 0.2;
    const rangeX = maxX - minX;
    const rangeY = maxY - minY;
    minX -= rangeX * padding;
    maxX += rangeX * padding;
    minY -= rangeY * padding;
    maxY += rangeY * padding;

    // Coordinate transformation
    const scaleX = (x) => ((x - minX) / (maxX - minX)) * width;
    const scaleY = (y) => height - ((y - minY) / (maxY - minY)) * height;
    const scaleRadius = (r) => (r / (maxX - minX)) * width;

    // Step 1: Draw noise circles around mapped points (if mode includes circles)
    if (visualizationMode === 'circles' || visualizationMode === 'all') {
      ctx.strokeStyle = '#ffaa00';
      ctx.fillStyle = 'rgba(255, 170, 0, 0.1)';
      ctx.lineWidth = 1;

      const radiusPixels = scaleRadius(epsilon);

      for (let i = 0; i < mapped_points.length; i += 2) {
        const cx = scaleX(mapped_points[i]);
        const cy = scaleY(mapped_points[i + 1]);

        ctx.beginPath();
        ctx.arc(cx, cy, radiusPixels, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      }
    }

    // Step 2: Draw mapped points f(zk)
    if (visualizationMode === 'mapped' || visualizationMode === 'all') {
      ctx.fillStyle = '#00ffff';
      for (let i = 0; i < mapped_points.length; i += 2) {
        const x = scaleX(mapped_points[i]);
        const y = scaleY(mapped_points[i + 1]);

        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }

    // Step 3: Draw normal vectors
    if (visualizationMode === 'all' && normals.length > 0) {
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 1;

      const normalLength = scaleRadius(epsilon * 1.5);

      for (let i = 0; i < mapped_points.length; i += 2) {
        const x = scaleX(mapped_points[i]);
        const y = scaleY(mapped_points[i + 1]);
        const nx = normals[i];
        const ny = normals[i + 1];

        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + nx * normalLength, y - ny * normalLength);
        ctx.stroke();
      }
    }

    // Step 4: Draw projected boundary points
if (visualizationMode === 'projected' || visualizationMode === 'all') {
  // Connect with lines to form CLOSED curve
  ctx.strokeStyle = '#ff00ff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  
  for (let i = 0; i < projected_points.length; i += 2) {
    const x = scaleX(projected_points[i]);
    const y = scaleY(projected_points[i + 1]);
    
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  
  ctx.closePath();
  ctx.stroke();
  
  // Optional: Fill the interior with semi-transparent color
  ctx.fillStyle = 'rgba(255, 0, 255, 0.1)';
  ctx.fill();
  
  // Draw boundary points on top
  ctx.fillStyle = '#ff00ff';
  for (let i = 0; i < projected_points.length; i += 2) {
    const x = scaleX(projected_points[i]);
    const y = scaleY(projected_points[i + 1]);
    
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, 2 * Math.PI);
    ctx.fill();
  }
}

    // Draw labels
    ctx.fillStyle = '#ffffff';
    ctx.font = '14px Arial';
    ctx.fillText(`Iteration: ${iteration}`, 10, 20);
    ctx.fillText(`Boundary Points: ${projected_points.length / 2}`, 10, 40);
    ctx.fillText(`ε = ${epsilon.toFixed(3)}`, 10, 60);

    // Legend
    const legendY = height - 80;
    if (visualizationMode === 'all') {
      ctx.fillStyle = '#00ffff';
      ctx.fillRect(10, legendY, 15, 15);
      ctx.fillStyle = '#ffffff';
      ctx.fillText('Mapped points f(zₖ)', 30, legendY + 12);

      ctx.fillStyle = '#ffaa00';
      ctx.fillRect(10, legendY + 20, 15, 15);
      ctx.fillStyle = '#ffffff';
      ctx.fillText('Noise circles Bε(f(zₖ))', 30, legendY + 32);

      ctx.fillStyle = '#ff00ff';
      ctx.fillRect(10, legendY + 40, 15, 15);
      ctx.fillStyle = '#ffffff';
      ctx.fillText('Projected boundary', 30, legendY + 52);
    }

  }, []);



  // Draw Algorithm 1 boundary evolution
  const drawAlgorithm1Boundary = useCallback((canvas, boundaryHistory, currentIteration) => {
    if (!canvas || !boundaryHistory || boundaryHistory.length === 0) {
      console.log('Cannot draw Algorithm 1 boundary:', { canvas: !!canvas, history: boundaryHistory?.length });
      return;
    }

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    console.log(`Drawing Algorithm 1 boundary evolution with ${boundaryHistory.length} iterations`);

    // Find global bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

    for (const iteration of boundaryHistory) {
      for (let i = 0; i < iteration.length; i += 2) {
        const x = iteration[i];
        const y = iteration[i + 1];
        if (isFinite(x) && isFinite(y)) {
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
        }
      }
    }

    // Add padding
    const padding = 0.15;
    const rangeX = maxX - minX;
    const rangeY = maxY - minY;

    if (rangeX === 0 || rangeY === 0 || rangeX < 1e-10 || rangeY < 1e-10) {
      const defaultRange = 0.5;
      minX = minX - defaultRange;
      maxX = maxX + defaultRange;
      minY = minY - defaultRange;
      maxY = maxY + defaultRange;
    } else {
      minX -= rangeX * padding;
      maxX += rangeX * padding;
      minY -= rangeY * padding;
      maxY += rangeY * padding;
    }

    // Coordinate transformation
    const scaleX = (x) => ((x - minX) / (maxX - minX)) * width;
    const scaleY = (y) => height - ((y - minY) / (maxY - minY)) * height;

    // Draw all boundary iterations with color gradient
    const maxIterations = boundaryHistory.length;

    for (let iter = 0; iter < maxIterations; iter++) {
      const boundary = boundaryHistory[iter];
      if (!boundary || boundary.length === 0) continue;

      // Color gradient: blue → cyan → green → yellow → red
      const intensity = iter / Math.max(1, maxIterations - 1);
      const r = Math.floor(255 * Math.pow(intensity, 0.7));
      const g = Math.floor(255 * (1 - Math.abs(2 * intensity - 1)));
      const b = Math.floor(255 * Math.pow(1 - intensity, 0.7));
      const alpha = 0.2 + 0.6 * intensity;

      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = alpha;

      // Draw boundary polygon
      ctx.beginPath();
      for (let i = 0; i < boundary.length; i += 2) {
        const x = scaleX(boundary[i]);
        const y = scaleY(boundary[i + 1]);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.closePath();
      ctx.stroke();

      // Draw boundary points
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      for (let i = 0; i < boundary.length; i += 2) {
        const x = scaleX(boundary[i]);
        const y = scaleY(boundary[i + 1]);
        ctx.beginPath();
        ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
        ctx.fill();
      }
    }

    // Highlight final iteration
    if (boundaryHistory.length > 0) {
      const finalBoundary = boundaryHistory[boundaryHistory.length - 1];

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3;
      ctx.globalAlpha = 1.0;

      ctx.beginPath();
      for (let i = 0; i < finalBoundary.length; i += 2) {
        const x = scaleX(finalBoundary[i]);
        const y = scaleY(finalBoundary[i + 1]);
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.closePath();
      ctx.stroke();

      // Draw final boundary points
      ctx.fillStyle = '#ffffff';
      for (let i = 0; i < finalBoundary.length; i += 2) {
        const x = scaleX(finalBoundary[i]);
        const y = scaleY(finalBoundary[i + 1]);
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    }

    ctx.globalAlpha = 1.0;

    // Draw title and info
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 16px Arial';
    ctx.fillText('Algorithm 1: Boundary Evolution', 10, 25);

    ctx.font = '12px Arial';
    ctx.fillText(`Iteration: ${currentIteration}/${maxIterations - 1}`, 10, height - 45);
    ctx.fillText(`Boundary Points: ${params.nBoundaryPoints}`, 10, height - 30);
    ctx.fillText(`ε = ${params.algorithm1Epsilon.toFixed(3)}`, 10, height - 15);

    if (algorithm1Data.isConverged) {
      ctx.fillStyle = '#00ff00';
      ctx.font = 'bold 14px Arial';
      ctx.fillText('✓ CONVERGED', width - 120, height - 15);
    }
  }, [params, algorithm1Data]);

  // Update visualizations when trajectory data changes
  useEffect(() => {
    if (trajectoryData.deterministic.length > 0) {
      drawTrajectory(deterministicCanvasRef.current, trajectoryData.deterministic, '#00ff00', 'Deterministic', false);
    }
    if (trajectoryData.noisy.length > 0) {
      drawTrajectory(noisyCanvasRef.current, trajectoryData.noisy, '#ff6b6b', 'With Bounded Noise', true);
    }
  }, [trajectoryData, drawTrajectory]);

  // Update Algorithm 1 visualization when data changes
  useEffect(() => {
    if (algorithm1Data.boundaryHistory.length > 0 && showAlgorithm1Visualization) {
      drawAlgorithm1Boundary(
        algorithm1CanvasRef.current,
        algorithm1Data.boundaryHistory,
        algorithm1Data.currentIteration
      );
    }
  }, [algorithm1Data, drawAlgorithm1Boundary, showAlgorithm1Visualization]);

  // Handle parameter changes
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

  // Initialize Algorithm 1 simulation when WASM loads
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
        <aside className="sidebar" width="200px">
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
            <h3>Trajectory Generation</h3>
            <div className="form-group">
              <label>
                Iterations: {params.iterations}
                <input
                  type="range"
                  min="10"
                  max="1000"
                  step="10"
                  value={params.iterations}
                  onChange={(e) => handleParamChange('iterations', e.target.value)}
                />
              </label>
            </div>
            <button
              className="button button-primary"
              onClick={generateTrajectories}
              disabled={!wasmLoaded || isRunning}
              style={{ width: '100%', marginBottom: '12px' }}
            >
              {isRunning ? 'Generating...' : '🎯 Generate Trajectories'}
            </button>

            <div className="form-group">
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <input
                  type="checkbox"
                  checked={showNoiseCircles}
                  onChange={(e) => setShowNoiseCircles(e.target.checked)}
                />
                Show noise circles
              </label>
            </div>
          </div>

          <div className="controls-section">
            <h3>Algorithm 1 Controls</h3>

            <button
              className="button button-primary"
              onClick={stepAlgorithm1SimulationDetailed}
              disabled={!wasmLoaded || isAlgorithm1Running || algorithm1Data.isDiverged}
              style={{ width: '100%', marginBottom: '8px' }}
            >
              ⏭ Step Forward (Detailed)
            </button>

            <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
              <button
                className="button button-secondary"
                onClick={runAlgorithm1Simulation}
                disabled={!wasmLoaded || isAlgorithm1Running}
                style={{ flex: '1' }}
              >
                {isAlgorithm1Running ? 'Running...' : '🚀 Run Full'}
              </button>
              <button
                className="button button-secondary"
                onClick={resetAlgorithm1Simulation}
                style={{ flex: '1' }}
              >
                🔄 Reset
              </button>
            </div>

            <div className="form-group">
              <label>Visualization Mode:</label>
              <select
                value={algorithm1VisualizationMode}
                onChange={(e) => setAlgorithm1VisualizationMode(e.target.value)}
                style={{ width: '100%', marginTop: '4px' }}
              >
                <option value="all">All Steps (Default)</option>
                <option value="mapped">Step 1: Mapped Points f(zₖ)</option>
                <option value="circles">Step 2: Noise Circles</option>
                <option value="projected">Step 3: Envelope Boundary</option>
              </select>
            </div>

            <div className="form-group">
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <input
                  type="checkbox"
                  checked={showAlgorithm1Visualization}
                  onChange={(e) => setShowAlgorithm1Visualization(e.target.checked)}
                />
                Show 3D visualization
              </label>
            </div>

            <div style={{ fontSize: '11px', color: '#333', background: '#f5f5f5', padding: '8px', borderRadius: '4px' }}>
              <div><strong>Algorithm 1 Status:</strong></div>
              <div>Iteration: {algorithm1Data.currentIteration}</div>
              <div>History: {algorithm1Data.boundaryHistory.length} steps</div>
              <div>Status: {algorithm1Data.isConverged ? '✅ Converged' : '⏳ Running'}</div>
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
              <h3>With Bounded Noise</h3>
              <canvas
                ref={noisyCanvasRef}
                width={600}
                height={400}
                style={{ width: '100%', height: 'auto' }}
              />
            </div>

            {showAlgorithm1Visualization && (
              <div className="canvas-container" style={{ gridColumn: '1 / -1' }}>
                <h3>Algorithm 1: Boundary Evolution (3D)</h3>
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
            )}

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
                <div className="stat-value">{algorithm1Data.currentIteration}</div>
                <div className="stat-label">Algorithm 1 Iterations</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{algorithm1Data.boundaryHistory.length}</div>
                <div className="stat-label">Boundary History</div>
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