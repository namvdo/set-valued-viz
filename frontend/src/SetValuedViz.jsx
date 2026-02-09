import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import * as THREE from 'three';

const GRID_CONFIG = {
    xRange: [-2, 2],
    yRange: [-1.5, 1.5],
    gridDivisions: 8,
    axisColor: 0x888888,
    gridColor: 0x333333
};

const henonMap = (x, y, a, b) => ({
    x: 1.0 - a * x * x + y,
    y: b * x
});

const duffingMap = (x, y, a, b) => ({
    x: y,
    y: -b * x + a * y - y * y * y
});

const createCoordinateSystem = (scene) => {
    const { xRange, yRange, gridDivisions, axisColor, gridColor } = GRID_CONFIG;
    const [xMin, xMax] = xRange;
    const [yMin, yMax] = yRange;
    const xStep = (xMax - xMin) / gridDivisions;
    const yStep = (yMax - yMin) / gridDivisions;

    const gridGroup = new THREE.Group();
    gridGroup.name = 'coordinate-system';

    for (let i = 0; i <= gridDivisions; i++) {
        const x = xMin + i * xStep;
        const isAxis = Math.abs(x) < 0.01;
        const points = [
            new THREE.Vector3(x, yMin, -0.01),
            new THREE.Vector3(x, yMax, -0.01)
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: isAxis ? axisColor : gridColor,
            transparent: true,
            opacity: isAxis ? 1.0 : 0.4
        });
        const line = new THREE.Line(geometry, material);
        line.userData.isGrid = true;
        gridGroup.add(line);
    }

    for (let i = 0; i <= gridDivisions; i++) {
        const y = yMin + i * yStep;
        const isAxis = Math.abs(y) < 0.01;
        const points = [
            new THREE.Vector3(xMin, y, -0.01),
            new THREE.Vector3(xMax, y, -0.01)
        ];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: isAxis ? axisColor : gridColor,
            transparent: true,
            opacity: isAxis ? 1.0 : 0.4
        });
        const line = new THREE.Line(geometry, material);
        line.userData.isGrid = true;
        gridGroup.add(line);
    }

    const createTextSprite = (text, position, fontSize = 0.15) => {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 128;
        canvas.height = 64;
        context.fillStyle = 'transparent';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.font = 'Bold 32px Arial';
        context.fillStyle = '#aaaaaa';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText(text, canvas.width / 2, canvas.height / 2);

        const texture = new THREE.CanvasTexture(canvas);
        texture.minFilter = THREE.LinearFilter;
        const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
        const sprite = new THREE.Sprite(material);
        sprite.position.copy(position);
        sprite.scale.set(fontSize * 2, fontSize, 1);
        sprite.userData.isGrid = true;
        return sprite;
    };

    for (let i = 0; i <= gridDivisions; i++) {
        const x = xMin + i * xStep;
        if (Math.abs(x) > 0.01) {
            gridGroup.add(createTextSprite(x.toFixed(1), new THREE.Vector3(x, yMin - 0.15, 0), 0.12));
        }
    }
    for (let i = 0; i <= gridDivisions; i++) {
        const y = yMin + i * yStep;
        if (Math.abs(y) > 0.01) {
            gridGroup.add(createTextSprite(y.toFixed(1), new THREE.Vector3(xMin - 0.2, y, 0), 0.12));
        }
    }

    gridGroup.add(createTextSprite('x', new THREE.Vector3(xMax + 0.2, 0, 0), 0.18));
    gridGroup.add(createTextSprite('y', new THREE.Vector3(0, yMax + 0.15, 0), 0.18));
    gridGroup.add(createTextSprite('0', new THREE.Vector3(-0.12, -0.12, 0), 0.1));

    scene.add(gridGroup);
    return gridGroup;
};

const ORBIT_COLORS = {
    period1: { stable: '#27ae60', unstable: '#e74c3c', saddle: '#eedf32' },
    period2: { stable: '#27ae60', unstable: '#e74c3c', saddle: '#eedf32' },
    period3: { stable: '#27ae60', unstable: '#e74c3c', saddle: '#eedf32' },
    period4: { stable: '#27ae60', unstable: '#e74c3c', saddle: '#eedf32' },
    period5: { stable: '#27ae60', unstable: '#e74c3c', saddle: '#eedf32' },
    period6plus: { stable: '#27ae60', unstable: '#e74c3c', saddle: '#eedf32' },
    trajectory: '#ff00ff',  // Bright magenta for high visibility
    manifold: '#1e90ff',  // Blue for unstable manifold
    stableManifold: '#ffa500', // Orange for stable manifold
    attractor: '#27ae60',
    repeller: '#e74c3c',
    saddlePoint: '#eedf32',
    periodicBlue: '#3498db'
};

const SetValuedViz = () => {
    const canvasRef = useRef(null);
    const rendererRef = useRef(null);
    const sceneRef = useRef(null);
    const cameraRef = useRef(null);
    const animationFrameRef = useRef(null);
    const batchAnimationRef = useRef(null);
    const manifoldDebounceRef = useRef(null);

    const [mode, setMode] = useState('manifold'); // 'periodic' or 'manifold'
    const [dynamicSystem, setDynamicSystem] = useState('henon'); // 'henon' or 'duffing'
    const [wasmModule, setWasmModule] = useState(null);

    const [params, setParams] = useState({
        a: 0.4,
        b: 0.3,
        epsilon: 0.0625,
        startX: 0.1,
        startY: 0.1,
        maxIterations: 1000,
        maxPeriod: 5
    });

    const [periodicState, setPeriodicState] = useState({
        orbits: [],
        trajectoryPoints: [],
        currentPoint: { x: 0.1, y: 0.1 },
        iteration: 0,
        isRunning: false,
        isReady: false,
        showOrbits: false,
        showTrail: true,
        showOrbitLines: false,
        highlightedOrbitId: null,
        hasStarted: false
    });

    const [manifoldState, setManifoldState] = useState({
        manifolds: [],
        stableManifolds: [],
        fixedPoints: [],
        intersections: [],
        isComputing: false,
        isReady: false,
        showOrbits: true,
        showOrbitLines: false,
        showUnstableManifold: true,
        showStableManifold: false,
        intersectionThreshold: 0.05,
        highlightedOrbitId: null,
        // Trajectory tracking state
        currentPoint: { x: 0.1, y: 0.1, nx: 1.0, ny: 0.0 }, // 4D point for boundary map
        trajectoryPoints: [],
        iteration: 0,
        isRunning: false,
        hasStarted: false,
        showTrail: true,
        startPoint: { x: 0.1, y: 0.1, nx: 1.0, ny: 0.0 }
    });

    const [filters, setFilters] = useState({
        period1: true, period2: true, period3: true,
        period4: true, period5: true, period6plus: false
    });

    // Ulam method state
    const [ulamState, setUlamState] = useState({
        gridBoxes: [],
        invariantMeasure: null,
        leftEigenvector: null, // backward invariant measure
        transitions: null, // array of {index, probability}
        selectedBoxIndex: null,
        currentBoxIndex: -1,
        isComputing: false,
        subdivisions: 20,
        pointsPerBox: 64, // 8x8 grid per box
        epsilon: 0.05, // epsilon ball radius for set-valued transitions
        showUlamOverlay: false,
        showTransitions: true,
        showCurrentBox: true, // highlight box containing current trajectory point
        needsRecompute: false // flag for auto-recompute
    });

    // Parameter animation state for manifold mode
    const [animationState, setAnimationState] = useState({
        isAnimating: false,
        parameter: 'a', // 'a', 'b', or 'epsilon'
        rangeValue: 0.1, // the range amount (e.g., 0.1 means go from current to current+0.1 or current-0.1)
        direction: 1, // +1 for positive direction, -1 for negative direction
        steps: 10, // number of steps to divide the range
        currentStep: 0, // current step in the animation
        baseValue: null, // the original value when animation started
        targetValue: null // the target value at the end of animation
    });

    const animationIntervalRef = useRef(null);

    // Video recording state
    const [recordingState, setRecordingState] = useState({
        isRecording: false,
        isEncoding: false,
        frameCount: 0,
        recordingEnabled: false, // toggle for recording with animation
        encodingProgress: 0,
        error: null
    });

    const recordedFramesRef = useRef([]);
    const encoderWorkerRef = useRef(null);

    const ulamComputerRef = useRef(null);
    const ulamDebounceRef = useRef(null);

    const [tooltip, setTooltip] = useState({
        visible: false,
        x: 0,
        y: 0,
        data: null
    });

    const raycasterRef = useRef(new THREE.Raycaster());
    const mouseRef = useRef(new THREE.Vector2());

    useEffect(() => {
        if (!canvasRef.current) return;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a0a);
        sceneRef.current = scene;

        const [xMin, xMax] = GRID_CONFIG.xRange;
        const [yMin, yMax] = GRID_CONFIG.yRange;
        const gridHeight = yMax - yMin;
        const padding = 0.5;
        const aspect = (window.innerWidth * 0.75) / window.innerHeight;
        const frustumHeight = gridHeight + padding * 2;
        const frustumWidth = frustumHeight * aspect;

        const camera = new THREE.OrthographicCamera(
            -frustumWidth / 2, frustumWidth / 2,
            frustumHeight / 2, -frustumHeight / 2, 0.1, 1000
        );
        camera.position.z = 5;
        cameraRef.current = camera;

        const renderer = new THREE.WebGLRenderer({
            canvas: canvasRef.current,
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true  // Required for screenshots
        });
        renderer.setSize(window.innerWidth * 0.75, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        rendererRef.current = renderer;

        createCoordinateSystem(scene);

        const handleResize = () => {
            const aspect = (window.innerWidth * 0.75) / window.innerHeight;
            const newFrustumWidth = frustumHeight * aspect;
            camera.left = -newFrustumWidth / 2;
            camera.right = newFrustumWidth / 2;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth * 0.75, window.innerHeight);
        };
        window.addEventListener('resize', handleResize);

        const animate = () => {
            animationFrameRef.current = requestAnimationFrame(animate);
            renderer.render(scene, camera);
        };
        animate();

        return () => {
            window.removeEventListener('resize', handleResize);
            if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
            if (batchAnimationRef.current) cancelAnimationFrame(batchAnimationRef.current);
            renderer.dispose();
        };
    }, []);

    useEffect(() => {
        const loadWasm = async () => {
            try {
                const wasm = await import('../pkg/henon_periodic_orbits.js');
                await wasm.default();
                setWasmModule(wasm);
                console.log('WASM module loaded successfully');
            } catch (err) {
                console.error('Failed to load WASM module:', err);
            }
        };
        loadWasm();
    }, []);

    useEffect(() => {
        if (dynamicSystem === 'duffing') {
            setParams(prev => ({ ...prev, a: 2.75, b: 0.2 }));
        } else {
            setParams(prev => ({ ...prev, a: 0.4, b: 0.3 }));
        }
    }, [dynamicSystem]);

    const computeJacobian = useCallback((x, y) => {
        const a = params.a;
        const b = params.b;
        const j11 = -2 * a * x;
        const j12 = 1;
        const j21 = b;
        const j22 = 0;
        const trace = j11 + j22;
        const det = j11 * j22 - j12 * j21;
        return { j11, j12, j21, j22, trace, det };
    }, [params.a, params.b]);

    const handleMouseMove = useCallback((event) => {
        if (!canvasRef.current || !sceneRef.current || !cameraRef.current) return;

        const rect = canvasRef.current.getBoundingClientRect();
        mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);

        if (ulamState.showUlamOverlay && ulamState.gridBoxes.length > 0) {
            const ulamMesh = sceneRef.current.getObjectByName('ulam-grid');
            if (ulamMesh) {
                const intersects = raycasterRef.current.intersectObject(ulamMesh);
                if (intersects.length > 0 && intersects[0].instanceId !== undefined) {
                    const boxIndex = intersects[0].instanceId;
                    const box = ulamState.gridBoxes[boxIndex];
                    const measure = ulamState.invariantMeasure ? ulamState.invariantMeasure[boxIndex] : 0;
                    const maxMeasure = ulamState.invariantMeasure ? Math.max(...ulamState.invariantMeasure) : 1;

                    let numTransitions = 0;
                    let topTransitions = [];
                    if (ulamComputerRef.current) {
                        try {
                            const trans = ulamComputerRef.current.get_transitions(boxIndex);
                            if (trans && trans.length > 0) {
                                numTransitions = trans.length;
                                topTransitions = trans
                                    .sort((a, b) => (b.probability || 0) - (a.probability || 0))
                                    .slice(0, 3);
                            }
                        } catch (e) {
                            console.error("Error getting transitions:", e);
                        }
                    }

                    setTooltip({
                        visible: true,
                        x: event.clientX,
                        y: event.clientY,
                        data: {
                            type: 'Ulam Box',
                            boxIndex: boxIndex,
                            pos: { x: box.center[0], y: box.center[1] },
                            measure: measure,
                            measurePercent: maxMeasure > 0 ? (measure / maxMeasure * 100) : 0,
                            numTransitions: numTransitions,
                            topTransitions: topTransitions,
                            isCurrentBox: boxIndex === ulamState.currentBoxIndex
                        }
                    });
                    return;
                }
            }
        }

        // Check for periodic points / fixed points
        const meshes = [];
        sceneRef.current.traverse((obj) => {
            if (obj.isMesh && (obj.userData.type === 'orbit' || obj.userData.type === 'fixedPoint')) {
                meshes.push(obj);
            }
        });

        const intersects = raycasterRef.current.intersectObjects(meshes, false);

        if (intersects.length > 0) {
            const hit = intersects[0].object;
            const data = hit.userData;
            const jac = computeJacobian(data.pos.x, data.pos.y);

            // Set highlighted orbit when hovering (only if orbit lines are shown in respective mode)
            if (data.type === 'orbit' && data.orbitId) {
                if (mode === 'periodic' && periodicState.showOrbitLines) {
                    setPeriodicState(prev => prev.highlightedOrbitId !== data.orbitId
                        ? { ...prev, highlightedOrbitId: data.orbitId }
                        : prev);
                } else if (mode === 'manifold' && manifoldState.showOrbitLines) {
                    setManifoldState(prev => prev.highlightedOrbitId !== data.orbitId
                        ? { ...prev, highlightedOrbitId: data.orbitId }
                        : prev);
                }
            }

            setTooltip({
                visible: true,
                x: event.clientX,
                y: event.clientY,
                data: {
                    type: data.type === 'fixedPoint' ? 'Fixed Point' : 'Periodic Point',
                    period: data.period,
                    stability: data.stability,
                    pos: data.pos,
                    eigenvalues: data.eigenvalues,
                    jacobian: jac,
                    orbitSize: data.orbitPoints?.length || 1
                }
            });
        } else {
            setTooltip(prev => prev.visible ? { ...prev, visible: false } : prev);
            // Clear highlighted orbit when not hovering (in both modes)
            if (mode === 'periodic' && periodicState.highlightedOrbitId !== null) {
                setPeriodicState(prev => prev.highlightedOrbitId !== null
                    ? { ...prev, highlightedOrbitId: null }
                    : prev);
            }
            if (mode === 'manifold' && manifoldState.highlightedOrbitId !== null) {
                setManifoldState(prev => prev.highlightedOrbitId !== null
                    ? { ...prev, highlightedOrbitId: null }
                    : prev);
            }
        }
    }, [computeJacobian, ulamState.showUlamOverlay, ulamState.gridBoxes, ulamState.invariantMeasure, ulamState.currentBoxIndex, mode, periodicState.showOrbitLines, periodicState.highlightedOrbitId, manifoldState.showOrbitLines, manifoldState.highlightedOrbitId]);



    useEffect(() => {
        if (!canvasRef.current || !sceneRef.current || !cameraRef.current) return;
        const rect = canvasRef.current.getBoundingClientRect();
    }, [handleMouseMove]);


    // Handle Ulam interaction
    const handleUlamClick = useCallback((index) => {
        if (!ulamComputerRef.current) return;

        const transitions = ulamComputerRef.current.get_transitions(index);
        setUlamState(prev => ({
            ...prev,
            selectedBoxIndex: index,
            transitions: transitions // array of {index, probability}
        }));
    }, []);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const handleClick = (event) => {
            if (!ulamState.showUlamOverlay || !ulamState.gridBoxes.length) return;

            const rect = canvas.getBoundingClientRect();
            const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycasterRef.current.setFromCamera(new THREE.Vector2(x, y), cameraRef.current);

            const scene = sceneRef.current;
            const ulamMesh = scene.getObjectByName('ulam-grid');

            if (ulamMesh) {
                const intersects = raycasterRef.current.intersectObject(ulamMesh);
                if (intersects.length > 0) {
                    const instanceId = intersects[0].instanceId;
                    if (instanceId !== undefined) {
                        handleUlamClick(instanceId);
                    }
                } else {
                    setUlamState(prev => ({ ...prev, selectedBoxIndex: null, transitions: null }));
                }
            }
        };

        canvas.addEventListener('mousemove', handleMouseMove);
        canvas.addEventListener('click', handleClick);

        return () => {
            canvas.removeEventListener('mousemove', handleMouseMove);
            canvas.removeEventListener('click', handleClick);
        };
    }, [handleMouseMove, ulamState.showUlamOverlay, ulamState.gridBoxes.length, handleUlamClick]);

    // Compute periodic orbits when params change and WASM is ready
    useEffect(() => {
        if (!wasmModule) return;

        let cancelled = false;
        setPeriodicState(prev => ({ ...prev, isReady: false, orbits: [], trajectoryPoints: [], iteration: 0, hasStarted: false, showOrbits: false }));

        const initSystem = () => {
            try {
                if (cancelled) return;

                // Use appropriate system based on dynamicSystem selection
                let SystemClass;
                let system;

                if (dynamicSystem === 'duffing') {
                    SystemClass = wasmModule.DuffingSystemWasm;
                    system = new SystemClass(params.a, params.b, params.maxPeriod);
                } else {
                    SystemClass = wasmModule.BoundaryHenonSystemWasm;
                    system = new SystemClass(params.a, params.b, params.epsilon, params.maxPeriod);
                }
                if (cancelled) { system.free(); return; }

                const orbits = system.getPeriodicOrbits();
                system.free();

                setPeriodicState(prev => ({
                    ...prev, orbits, isReady: true,
                    currentPoint: { x: params.startX, y: params.startY }
                }));
            } catch (err) {
                console.error('Failed to compute periodic orbits:', err);
                setPeriodicState(prev => ({ ...prev, isReady: true, orbits: [] }));
            }
        };
        initSystem();
        return () => { cancelled = true; };
    }, [wasmModule, dynamicSystem, params.a, params.b, params.epsilon, params.maxPeriod, params.startX, params.startY]);

    useEffect(() => {
        if (mode !== 'manifold') return;

        if (manifoldDebounceRef.current) {
            clearTimeout(manifoldDebounceRef.current);
        }

        setManifoldState(prev => ({ ...prev, isComputing: true }));

        manifoldDebounceRef.current = setTimeout(() => {
            if (!wasmModule) return;
            try {
                if (dynamicSystem === 'duffing') {
                    // Use Duffing manifold computation
                    console.log('Computing Duffing manifold');
                    const result = wasmModule.compute_duffing_manifold_simple(
                        params.a,
                        params.b,
                        params.epsilon
                    );

                    setManifoldState(prev => ({
                        ...prev,
                        manifolds: result.manifolds || [],
                        fixedPoints: result.fixed_points || [],
                        isComputing: false,
                        isReady: true
                    }));
                } else {
                    // Use Henon manifold computation
                    // Use periodic orbits from the periodic state if available
                    // Otherwise fall back to the simple computation
                    if (periodicState.orbits && periodicState.orbits.length > 0) {
                        // Check if stable manifold display is enabled
                        if (manifoldState.showStableManifold) {
                            console.log('Computing stable AND unstable manifolds:', periodicState.orbits.length, 'orbits');
                            const result = wasmModule.compute_stable_and_unstable_manifolds(
                                params.a,
                                params.b,
                                params.epsilon,
                                periodicState.orbits,
                                manifoldState.intersectionThreshold
                            );

                            setManifoldState(prev => ({
                                ...prev,
                                manifolds: result.unstable_manifolds || [],
                                stableManifolds: result.stable_manifolds || [],
                                fixedPoints: result.fixed_points || [],
                                intersections: result.intersections || [],
                                isComputing: false,
                                isReady: true
                            }));
                        } else {
                            console.log('Using periodic orbits for manifold:', periodicState.orbits.length, 'orbits');
                            const result = wasmModule.compute_manifold_from_orbits(
                                params.a,
                                params.b,
                                params.epsilon,
                                periodicState.orbits
                            );

                            setManifoldState(prev => ({
                                ...prev,
                                manifolds: result.manifolds || [],
                                stableManifolds: [],
                                fixedPoints: result.fixed_points || [],
                                intersections: [],
                                isComputing: false,
                                isReady: true
                            }));
                        }
                    } else {
                        console.log('No periodic orbits available, using simple computation');
                        const result = wasmModule.compute_manifold_simple(params.a, params.b, params.epsilon);

                        setManifoldState(prev => ({
                            ...prev,
                            manifolds: result.manifolds || [],
                            stableManifolds: [],
                            fixedPoints: result.fixed_points || [],
                            intersections: [],
                            isComputing: false,
                            isReady: true
                        }));
                    }
                }
            } catch (err) {
                console.error('Manifold computation error:', err);
                setManifoldState(prev => ({ ...prev, isComputing: false }));
            }
        }, 500);

        return () => {
            if (manifoldDebounceRef.current) {
                clearTimeout(manifoldDebounceRef.current);
            }
        };
    }, [mode, dynamicSystem, params.a, params.b, params.epsilon, periodicState.orbits, wasmModule, manifoldState.showStableManifold, manifoldState.intersectionThreshold]);

    // Sequential animation - advance to next step only when manifold computation completes
    useEffect(() => {
        if (!animationState.isAnimating || mode !== 'manifold') {
            return;
        }

        // Only advance when manifold is ready (not computing)
        if (manifoldState.isComputing) {
            return;
        }

        const { parameter, rangeValue, direction, steps, currentStep, baseValue } = animationState;

        // Check if we've completed all steps
        if (currentStep >= steps) {
            // Animation complete
            setAnimationState(prev => ({
                ...prev,
                isAnimating: false
            }));
            return;
        }

        // Calculate the next value
        const stepSize = rangeValue / steps;
        const nextStep = currentStep + 1;
        const nextValue = baseValue + (direction * stepSize * nextStep);

        // Update the parameter and advance step
        setParams(p => ({ ...p, [parameter]: parseFloat(nextValue.toFixed(4)) }));
        setAnimationState(prev => ({
            ...prev,
            currentStep: nextStep
        }));

    }, [animationState.isAnimating, animationState.currentStep, manifoldState.isComputing, mode]);

    const startAnimation = useCallback(async () => {
        const baseVal = params[animationState.parameter];
        const targetVal = baseVal + (animationState.direction * animationState.rangeValue);

        // Capture initial frame if recording
        if (recordingState.recordingEnabled && canvasRef.current) {
            try {
                const canvas = canvasRef.current;
                const width = 1280;
                const height = 720;
                const offscreen = new OffscreenCanvas(width, height);
                const ctx = offscreen.getContext('2d');
                ctx.drawImage(canvas, 0, 0, width, height);

                // Draw parameter overlay
                const overlayText = `a = ${params.a.toFixed(4)}  b = ${params.b.toFixed(4)}  ε = ${params.epsilon.toFixed(4)}`;
                ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                ctx.fillRect(10, height - 40, 400, 30);
                ctx.font = 'bold 16px monospace';
                ctx.fillStyle = '#4CAF50';
                ctx.fillText(overlayText, 20, height - 18);

                const bitmap = await createImageBitmap(offscreen);
                recordedFramesRef.current.push(bitmap);
                setRecordingState(prev => ({ ...prev, frameCount: 1 }));
                console.log('Initial frame captured');
            } catch (err) {
                console.error('Initial frame capture error:', err);
            }
        }

        setAnimationState(prev => ({
            ...prev,
            isAnimating: true,
            baseValue: baseVal,
            targetValue: targetVal,
            currentStep: 0
        }));
    }, [params, animationState.parameter, animationState.direction, animationState.rangeValue, recordingState.recordingEnabled]);

    const stopAnimation = useCallback(() => {
        setAnimationState(prev => ({
            ...prev,
            isAnimating: false,
            currentStep: 0
        }));
    }, []);

    const captureFrame = useCallback(async () => {
        if (!canvasRef.current || !recordingState.recordingEnabled) return null;

        const canvas = canvasRef.current;
        const width = 1280;
        const height = 720;

        const offscreen = new OffscreenCanvas(width, height);
        const ctx = offscreen.getContext('2d');

        ctx.drawImage(canvas, 0, 0, width, height);

        const overlayText = `a = ${params.a.toFixed(4)}  b = ${params.b.toFixed(4)}  ε = ${params.epsilon.toFixed(4)}`;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(10, height - 40, 400, 30);
        ctx.font = 'bold 16px monospace';
        ctx.fillStyle = '#4CAF50';
        ctx.fillText(overlayText, 20, height - 18);

        const bitmap = await createImageBitmap(offscreen);
        return bitmap;
    }, [params.a, params.b, params.epsilon, recordingState.recordingEnabled]);

    // Initialize video encoder worker
    const initEncoderWorker = useCallback(() => {
        if (encoderWorkerRef.current) {
            encoderWorkerRef.current.terminate();
        }

        const worker = new Worker(
            new URL('./videoEncoder.worker.js', import.meta.url),
            { type: 'classic' }
        );

        worker.onmessage = (e) => {
            const { type, blob, frameCount, error } = e.data;

            switch (type) {
                case 'ready':
                    console.log('Encoder ready');
                    break;
                case 'progress':
                    setRecordingState(prev => ({ ...prev, frameCount: frameCount }));
                    break;
                case 'complete':
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    const filename = generateFilename();
                    a.href = url;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);

                    setRecordingState(prev => ({
                        ...prev,
                        isEncoding: false,
                        isRecording: false,
                        recordingEnabled: false
                    }));
                    recordedFramesRef.current = [];
                    break;
                case 'error':
                    console.error('Encoder error:', error);
                    setRecordingState(prev => ({ ...prev, error, isEncoding: false }));
                    break;
            }
        };

        encoderWorkerRef.current = worker;
        return worker;
    }, []);

    const generateFilename = useCallback(() => {
        const aStr = params.a.toFixed(3).replace('.', 'p');
        const bStr = params.b.toFixed(3).replace('.', 'p');
        const epsStr = params.epsilon.toFixed(4).replace('.', 'p');
        const paramName = animationState.parameter;
        const startStr = (animationState.baseValue || 0).toFixed(3).replace('.', 'p').replace('-', 'm');
        const endStr = (animationState.targetValue || 0).toFixed(3).replace('.', 'p').replace('-', 'm');

        return `henon_${paramName}_a${aStr}_b${bStr}_eps${epsStr}_${startStr}_to_${endStr}.mp4`;
    }, [params.a, params.b, params.epsilon, animationState.parameter, animationState.baseValue, animationState.targetValue]);

    const savePNG = useCallback(async () => {
        if (!canvasRef.current || !rendererRef.current || !sceneRef.current || !cameraRef.current) return;

        // Force a render to ensure canvas has the current frame
        rendererRef.current.render(sceneRef.current, cameraRef.current);

        const canvas = canvasRef.current;
        const width = 1920;
        const height = 1080;

        const offscreen = new OffscreenCanvas(width, height);
        const ctx = offscreen.getContext('2d');

        ctx.drawImage(canvas, 0, 0, width, height);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(10, height - 80, 500, 70);

        ctx.font = 'bold 18px monospace';
        ctx.fillStyle = '#4CAF50';

        if (mode === 'periodic') {
            ctx.fillText(`Periodic Orbits | Iteration: ${periodicState.iteration}`, 20, height - 55);
            ctx.font = '14px monospace';
            ctx.fillStyle = '#aaa';
            ctx.fillText(`Current: (${periodicState.currentPoint.x.toFixed(6)}, ${periodicState.currentPoint.y.toFixed(6)})`, 20, height - 32);
        } else if (mode === 'manifold') {
            ctx.fillText(`Manifold Approximation`, 20, height - 55);
            ctx.font = '14px monospace';
            ctx.fillStyle = '#aaa';
            const unstablePts = manifoldState.manifolds.reduce((sum, m) => sum + (m.points_positive?.length || 0) + (m.points_negative?.length || 0), 0);
            const stablePts = manifoldState.stableManifolds.reduce((sum, m) => sum + (m.points_positive?.length || 0) + (m.points_negative?.length || 0), 0);
            const stableInfo = manifoldState.showStableManifold ? `, ${stablePts} stable pts` : '';
            ctx.fillText(`${manifoldState.fixedPoints.length} fixed pts, ${unstablePts} unstable pts${stableInfo}`, 20, height - 32);
        }

        ctx.font = 'bold 14px monospace';
        ctx.fillStyle = '#4CAF50';
        ctx.fillText(`a = ${params.a.toFixed(4)}   b = ${params.b.toFixed(4)}   ε = ${params.epsilon.toFixed(4)}`, 20, height - 12);

        const blob = await offscreen.convertToBlob({ type: 'image/png', quality: 1.0 });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');

        const aStr = params.a.toFixed(3).replace('.', 'p');
        const bStr = params.b.toFixed(3).replace('.', 'p');
        const epsStr = params.epsilon.toFixed(4).replace('.', 'p');
        const modeStr = mode === 'periodic' ? 'periodic' : 'manifold';
        const iterStr = mode === 'periodic' ? `_iter${periodicState.iteration}` : '';

        a.href = url;
        a.download = `henon_${modeStr}_a${aStr}_b${bStr}_eps${epsStr}${iterStr}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, [mode, params, periodicState.iteration, periodicState.currentPoint, manifoldState.manifolds, manifoldState.fixedPoints, manifoldState.stableManifolds, manifoldState.showUnstableManifold, manifoldState.showStableManifold]);

    const startEncoding = useCallback(async () => {
        if (recordedFramesRef.current.length === 0) {
            console.warn('No frames to encode');
            return;
        }

        setRecordingState(prev => ({ ...prev, isEncoding: true, encodingProgress: 0 }));

        const worker = initEncoderWorker();

        worker.postMessage({
            type: 'init',
            data: {
                width: 1280,
                height: 720,
                fps: 2,
                filename: generateFilename()
            }
        });

        await new Promise(resolve => setTimeout(resolve, 100));

        const fps = 2;
        const frameDuration = 1000000 / fps; // microseconds (500ms per frame)

        for (let i = 0; i < recordedFramesRef.current.length; i++) {
            const frame = recordedFramesRef.current[i];
            worker.postMessage({
                type: 'frame',
                data: {
                    imageData: frame,
                    timestamp: i * frameDuration,
                    duration: frameDuration
                }
            });

            if (i % 5 === 0) {
                await new Promise(resolve => setTimeout(resolve, 10));
            }
        }

        worker.postMessage({ type: 'finish' });
    }, [initEncoderWorker, generateFilename]);

    // Simple frame capture: capture whenever manifold finishes computing while recording
    const wasComputingRef = useRef(false);

    useEffect(() => {
        const wasComputing = wasComputingRef.current;
        const isComputing = manifoldState.isComputing;
        wasComputingRef.current = isComputing;

        // Only care about recording during animation
        if (!recordingState.recordingEnabled || !animationState.isAnimating) {
            return;
        }

        // Capture when manifold computation finishes (true -> false transition)
        if (wasComputing && !isComputing) {
            console.log(`[Recording] Manifold finished, capturing frame for step ${animationState.currentStep}...`);

            // Use requestAnimationFrame to ensure render is complete
            requestAnimationFrame(async () => {
                try {
                    const frame = await captureFrame();
                    if (frame) {
                        recordedFramesRef.current.push(frame);
                        setRecordingState(prev => ({ ...prev, frameCount: recordedFramesRef.current.length }));
                        console.log(`[Recording] Frame ${recordedFramesRef.current.length} captured`);
                    } else {
                        console.log('[Recording] captureFrame returned null');
                    }
                } catch (err) {
                    console.error('[Recording] Frame capture error:', err);
                }
            });
        }
    }, [manifoldState.isComputing, animationState.isAnimating, recordingState.recordingEnabled, animationState.currentStep, captureFrame]);

    // Start encoding when animation finishes (if recording was enabled and frames were captured)
    useEffect(() => {
        // Trigger encoding when: animation stops, recording was enabled, we have frames, and not already encoding
        if (!animationState.isAnimating && recordingState.recordingEnabled && recordedFramesRef.current.length > 0 && !recordingState.isEncoding) {
            console.log(`[Recording] Animation finished with ${recordedFramesRef.current.length} frames, starting encoding...`);
            startEncoding();
        }
    }, [animationState.isAnimating, recordingState.recordingEnabled, recordingState.isEncoding, startEncoding]);

    // Toggle recording mode
    const toggleRecording = useCallback(() => {
        if (recordingState.recordingEnabled) {
            // Disable recording
            setRecordingState(prev => ({ ...prev, recordingEnabled: false }));
            recordedFramesRef.current = [];
        } else {
            // Enable recording
            setRecordingState(prev => ({ ...prev, recordingEnabled: true, frameCount: 0, error: null }));
            recordedFramesRef.current = [];
        }
    }, [recordingState.recordingEnabled]);

    useEffect(() => {
        if (!sceneRef.current) return;
        const scene = sceneRef.current;

        const toRemove = [];
        scene.traverse(child => {
            if (child.userData.type === 'trajectory' || child.userData.type === 'orbit' || child.userData.type === 'orbitLine' || child.userData.type === 'manifold' || child.userData.type === 'fixedPoint') {
                toRemove.push(child);
            }
        });
        toRemove.forEach(obj => {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
            scene.remove(obj);
        });

        if (mode === 'periodic') {
            // Render periodic orbits
            if (periodicState.showOrbits && periodicState.orbits.length > 0) {
                const visibleOrbits = periodicState.orbits.filter(o => isOrbitVisible(o));
                const HIGHLIGHT_COLOR = '#ff00ff'; // Magenta for highlighted orbits - distinct from yellow/green/red

                visibleOrbits.forEach((orbit, orbitIdx) => {
                    const orbitId = `orbit-${orbit.period}-${orbitIdx}`;
                    const isHighlighted = periodicState.highlightedOrbitId === orbitId;
                    // Use highlight color (gold) when hovered, otherwise use stability-based color
                    const pointColor = isHighlighted ? HIGHLIGHT_COLOR : getOrbitColor(orbit);

                    orbit.points.forEach((pt, ptIdx) => {
                        const geom = new THREE.SphereGeometry(isHighlighted ? 0.06 : 0.03, 12, 12);
                        const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(pointColor) });
                        const sphere = new THREE.Mesh(geom, mat);
                        sphere.position.set(pt[0], pt[1], isHighlighted ? 0.15 : 0.1); // Bring highlighted points forward
                        // Store full orbit info for tooltip and hover matching
                        sphere.userData = {
                            type: 'orbit',
                            orbitId: orbitId,
                            period: orbit.period,
                            stability: orbit.stability,
                            pointIndex: ptIdx,
                            pos: { x: pt[0], y: pt[1] },
                            orbitPoints: orbit.points,
                            eigenvalues: orbit.eigenvalues || null
                        };
                        scene.add(sphere);
                    });

                    // Only render orbit lines when showOrbitLines is enabled
                    if (periodicState.showOrbitLines && orbit.points.length > 1) {
                        const lineGeom = new THREE.BufferGeometry();
                        const positions = new Float32Array(orbit.points.length * 3);
                        orbit.points.forEach((pt, i) => {
                            positions[i * 3] = pt[0];
                            positions[i * 3 + 1] = pt[1];
                            positions[i * 3 + 2] = isHighlighted ? 0.15 : 0.1;
                        });
                        lineGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                        // Use highlight color for lines when hovered
                        const lineColor = isHighlighted ? HIGHLIGHT_COLOR : getOrbitColor(orbit);
                        const opacity = isHighlighted ? 1.0 : 0.4;
                        const lineMat = new THREE.LineBasicMaterial({
                            color: new THREE.Color(lineColor),
                            opacity: opacity,
                            transparent: true,
                            linewidth: isHighlighted ? 3 : 1
                        });
                        const line = new THREE.LineLoop(lineGeom, lineMat);
                        line.userData = { type: 'orbitLine', orbitId: orbitId };
                        scene.add(line);
                    }
                });
            }

            if (periodicState.showTrail && periodicState.trajectoryPoints.length > 0) {
                periodicState.trajectoryPoints.forEach((point, idx) => {
                    const normalizedIdx = idx / periodicState.trajectoryPoints.length;
                    const size = 0.012 * (0.3 + 0.7 * normalizedIdx);
                    const geom = new THREE.SphereGeometry(size, 6, 6);
                    const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(ORBIT_COLORS.trajectory), opacity: 0.2 + 0.6 * normalizedIdx, transparent: true });
                    const sphere = new THREE.Mesh(geom, mat);
                    sphere.position.set(point.x, point.y, 0.05);
                    sphere.userData.type = 'trajectory';
                    scene.add(sphere);
                });
            }

            // Render current point
            if (periodicState.hasStarted && periodicState.currentPoint) {
                const geom = new THREE.SphereGeometry(0.04, 16, 16);
                const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color('#ffffff') });
                const sphere = new THREE.Mesh(geom, mat);
                sphere.position.set(periodicState.currentPoint.x, periodicState.currentPoint.y, 0.2);
                sphere.userData.type = 'trajectory';
                scene.add(sphere);
            }
        } else {
            // Render unstable manifolds (blue) - only when showUnstableManifold is enabled
            if (manifoldState.showUnstableManifold && manifoldState.manifolds.length > 0) {
                manifoldState.manifolds.forEach(m => {
                    [m.plus, m.minus].forEach(traj => {
                        if (traj && traj.points && traj.points.length > 1) {
                            const lineGeom = new THREE.BufferGeometry();
                            const positions = new Float32Array(traj.points.length * 3);
                            traj.points.forEach(([x, y], i) => {
                                positions[i * 3] = x;
                                positions[i * 3 + 1] = y;
                                positions[i * 3 + 2] = 0.1;
                            });
                            lineGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                            const lineMat = new THREE.LineBasicMaterial({ color: new THREE.Color(ORBIT_COLORS.manifold), linewidth: 2 });
                            const line = new THREE.Line(lineGeom, lineMat);
                            line.userData.type = 'manifold';
                            scene.add(line);
                        }
                    });
                });
            }

            // Render stable manifolds (orange) - only when showStableManifold is enabled
            if (manifoldState.showStableManifold && manifoldState.stableManifolds.length > 0) {
                manifoldState.stableManifolds.forEach(m => {
                    [m.plus, m.minus].forEach(traj => {
                        if (traj && traj.points && traj.points.length > 1) {
                            const lineGeom = new THREE.BufferGeometry();
                            const positions = new Float32Array(traj.points.length * 3);
                            traj.points.forEach(([x, y], i) => {
                                positions[i * 3] = x;
                                positions[i * 3 + 1] = y;
                                positions[i * 3 + 2] = 0.08; // Slightly behind unstable manifolds
                            });
                            lineGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                            const lineMat = new THREE.LineBasicMaterial({
                                color: new THREE.Color(ORBIT_COLORS.stableManifold),
                                linewidth: 2
                            });
                            const line = new THREE.Line(lineGeom, lineMat);
                            line.userData.type = 'manifold';
                            scene.add(line);
                        }
                    });
                });
            }

            manifoldState.fixedPoints.forEach(fp => {
                const stabLower = (fp.stability || '').toLowerCase();
                const isAttractor = stabLower === 'attractor' || stabLower === 'stable';
                const isRepeller = stabLower === 'repeller' || stabLower === 'unstable';
                const isSaddle = stabLower === 'saddle';
                const color = isAttractor ? ORBIT_COLORS.attractor :
                    (isRepeller || isSaddle) ? ORBIT_COLORS.saddlePoint : ORBIT_COLORS.periodicBlue;
                const radius = isAttractor ? 0.03 : 0.025;
                const geom = new THREE.SphereGeometry(radius, 12, 12);
                const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(color) });
                const sphere = new THREE.Mesh(geom, mat);
                sphere.position.set(fp.x, fp.y, 0.2);
                sphere.userData = {
                    type: 'fixedPoint',
                    period: 1,
                    stability: fp.stability,
                    pos: { x: fp.x, y: fp.y },
                    eigenvalues: fp.eigenvalues || null
                };
                scene.add(sphere);
            });

            // render the trajectory trail

            if (manifoldState.showTrail && manifoldState.trajectoryPoints.length > 0) {
                manifoldState.trajectoryPoints.forEach((point, idx) => {
                    const normalizedIdx = idx / manifoldState.trajectoryPoints.length;
                    const size = 0.012 * (0.3 + 0.7 * normalizedIdx);
                    const geom = new THREE.SphereGeometry(size, 6, 6);
                    const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(ORBIT_COLORS.trajectory), opacity: 0.2 + 0.6 * normalizedIdx, transparent: true });
                    const sphere = new THREE.Mesh(geom, mat);
                    sphere.position.set(point.x, point.y, 0.05);
                    sphere.userData.type = 'trajectory';
                    scene.add(sphere);
                });
            }




            if (manifoldState.hasStarted && manifoldState.currentPoint) {
                const geom = new THREE.SphereGeometry(0.04, 16, 16);
                const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color('#ffffff') });
                const sphere = new THREE.Mesh(geom, mat);
                sphere.position.set(manifoldState.currentPoint.x, manifoldState.currentPoint.y, 0.2);
                sphere.userData.type = 'trajectory';
                scene.add(sphere);
            }

            // Also render periodic orbits in manifold mode if enabled
            if (manifoldState.showOrbits && periodicState.orbits.length > 0) {
                const visibleOrbits = periodicState.orbits.filter(o => isOrbitVisible(o));
                const HIGHLIGHT_COLOR = '#ff00ff';

                visibleOrbits.forEach((orbit, orbitIdx) => {
                    const orbitId = `orbit-${orbit.period}-${orbitIdx}`;
                    const isHighlighted = manifoldState.highlightedOrbitId === orbitId;
                    const pointColor = isHighlighted ? HIGHLIGHT_COLOR : getOrbitColor(orbit);

                    // Render orbit points with smaller size than periodic mode
                    orbit.points.forEach((pt, ptIdx) => {
                        const geom = new THREE.SphereGeometry(isHighlighted ? 0.04 : 0.02, 10, 10);
                        const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(pointColor) });
                        const sphere = new THREE.Mesh(geom, mat);
                        sphere.position.set(pt[0], pt[1], isHighlighted ? 0.15 : 0.05);
                        sphere.userData = {
                            type: 'orbit',
                            orbitId: orbitId,
                            period: orbit.period,
                            stability: orbit.stability,
                            pointIndex: ptIdx,
                            pos: { x: pt[0], y: pt[1] },
                            orbitPoints: orbit.points,
                            eigenvalues: orbit.eigenvalues || null
                        };
                        scene.add(sphere);
                    });

                    // Render orbit lines if enabled
                    if (manifoldState.showOrbitLines && orbit.points.length > 1) {
                        const lineGeom = new THREE.BufferGeometry();
                        const positions = new Float32Array(orbit.points.length * 3);
                        orbit.points.forEach((pt, i) => {
                            positions[i * 3] = pt[0];
                            positions[i * 3 + 1] = pt[1];
                            positions[i * 3 + 2] = isHighlighted ? 0.15 : 0.05;
                        });
                        lineGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                        const lineColor = isHighlighted ? HIGHLIGHT_COLOR : getOrbitColor(orbit);
                        const opacity = isHighlighted ? 1.0 : 0.4;
                        const lineMat = new THREE.LineBasicMaterial({
                            color: new THREE.Color(lineColor),
                            opacity: opacity,
                            transparent: true,
                            linewidth: isHighlighted ? 3 : 1
                        });
                        const line = new THREE.LineLoop(lineGeom, lineMat);
                        line.userData = { type: 'orbitLine', orbitId: orbitId };
                        scene.add(line);
                    }
                });
            }
        }
    }, [mode, periodicState, manifoldState, filters]);

    const getOrbitColor = (orbit) => {
        const { stability } = orbit;
        // Match reference image: green=stable, red=saddle/unstable
        if (stability.toLowerCase() === 'stable') return ORBIT_COLORS.attractor;
        if (stability.toLowerCase() === 'saddle') return ORBIT_COLORS.saddlePoint;
        if (stability.toLowerCase() === 'unstable') return ORBIT_COLORS.repeller;
        return ORBIT_COLORS.periodicBlue;
    };

    const isOrbitVisible = (orbit) => {
        if (orbit.period === 1) return filters.period1;
        if (orbit.period === 2) return filters.period2;
        if (orbit.period === 3) return filters.period3;
        if (orbit.period === 4) return filters.period4;
        if (orbit.period === 5) return filters.period5;
        return filters.period6plus;
    };

    const stepForward = useCallback(() => {
        if (!periodicState.isReady || periodicState.isRunning) return;
        const { x, y } = periodicState.currentPoint;
        if (!isFinite(x) || !isFinite(y) || Math.abs(x) > 10 || Math.abs(y) > 10) {
            setPeriodicState(prev => ({ ...prev, currentPoint: { x: params.startX, y: params.startY }, trajectoryPoints: [], iteration: 0, hasStarted: false }));
            return;
        }
        // Use appropriate map based on dynamicSystem
        const mapFn = dynamicSystem === 'duffing' ? duffingMap : henonMap;
        const nextPoint = mapFn(x, y, params.a, params.b);
        setPeriodicState(prev => ({ ...prev, currentPoint: nextPoint, trajectoryPoints: [...prev.trajectoryPoints, { x, y }], iteration: prev.iteration + 1, hasStarted: true }));
    }, [periodicState.isReady, periodicState.isRunning, periodicState.currentPoint, params, dynamicSystem]);

    const runToConvergence = useCallback(() => {
        if (!periodicState.isReady || periodicState.isRunning) return;
        setPeriodicState(prev => ({ ...prev, isRunning: true }));

        // Use appropriate map based on dynamicSystem
        const mapFn = dynamicSystem === 'duffing' ? duffingMap : henonMap;

        let currentX = periodicState.currentPoint.x;
        let currentY = periodicState.currentPoint.y;
        let iteration = periodicState.iteration;
        const newPoints = [...periodicState.trajectoryPoints];
        const batchSize = 5;

        const animateStep = () => {
            for (let i = 0; i < batchSize && iteration < params.maxIterations; i++) {
                if (!isFinite(currentX) || !isFinite(currentY) || Math.abs(currentX) > 10 || Math.abs(currentY) > 10) {
                    setPeriodicState(prev => ({ ...prev, isRunning: false, showOrbits: true, hasStarted: true, trajectoryPoints: newPoints, currentPoint: { x: currentX, y: currentY }, iteration }));
                    return;
                }
                newPoints.push({ x: currentX, y: currentY });
                const next = mapFn(currentX, currentY, params.a, params.b);
                currentX = next.x;
                currentY = next.y;
                iteration++;
            }
            setPeriodicState(prev => ({ ...prev, currentPoint: { x: currentX, y: currentY }, trajectoryPoints: [...newPoints], iteration, hasStarted: true }));
            if (iteration < params.maxIterations) {
                batchAnimationRef.current = requestAnimationFrame(animateStep);
            } else {
                setPeriodicState(prev => ({ ...prev, isRunning: false, showOrbits: true }));
            }
        };
        batchAnimationRef.current = requestAnimationFrame(animateStep);
    }, [periodicState, params, dynamicSystem]);

    const reset = useCallback(() => {
        if (batchAnimationRef.current) cancelAnimationFrame(batchAnimationRef.current);
        setPeriodicState(prev => ({ ...prev, currentPoint: { x: params.startX, y: params.startY }, trajectoryPoints: [], iteration: 0, isRunning: false, hasStarted: false, showOrbits: false }));
    }, [params.startX, params.startY]);

    // Manifold mode trajectory controls
    const stepForwardManifold = useCallback(() => {
        if (!manifoldState.isReady || manifoldState.isRunning || !wasmModule) return;
        const { x, y, nx, ny } = manifoldState.currentPoint;

        // Check bounds
        if (!isFinite(x) || !isFinite(y) || Math.abs(x) > 10 || Math.abs(y) > 10) {
            setManifoldState(prev => ({
                ...prev,
                currentPoint: { ...prev.startPoint },
                trajectoryPoints: [],
                iteration: 0,
                hasStarted: false
            }));
            return;
        }

        // Use boundary_map from WASM
        const { boundary_map } = wasmModule;
        if (!boundary_map) {
            console.error('boundary_map not found in WASM module');
            return;
        }

        const nextPoint = boundary_map(x, y, nx, ny, params.a, params.b, params.epsilon);

        setManifoldState(prev => ({
            ...prev,
            currentPoint: { x: nextPoint.x, y: nextPoint.y, nx: nextPoint.nx, ny: nextPoint.ny },
            trajectoryPoints: [...prev.trajectoryPoints, { x, y, nx, ny }],
            iteration: prev.iteration + 1,
            hasStarted: true
        }));
    }, [manifoldState.isReady, manifoldState.isRunning, manifoldState.currentPoint, wasmModule, params]);

    const runToConvergenceManifold = useCallback(() => {
        if (!manifoldState.isReady || manifoldState.isRunning || !wasmModule) return;
        setManifoldState(prev => ({ ...prev, isRunning: true }));

        const { boundary_map } = wasmModule;
        if (!boundary_map) {
            console.error('boundary_map not found in WASM module');
            setManifoldState(prev => ({ ...prev, isRunning: false }));
            return;
        }

        let currentX = manifoldState.currentPoint.x;
        let currentY = manifoldState.currentPoint.y;
        let currentNx = manifoldState.currentPoint.nx;
        let currentNy = manifoldState.currentPoint.ny;
        let iteration = manifoldState.iteration;
        const newPoints = [...manifoldState.trajectoryPoints];
        const batchSize = 5;

        const animateStep = () => {
            for (let i = 0; i < batchSize && iteration < params.maxIterations; i++) {
                if (!isFinite(currentX) || !isFinite(currentY) || Math.abs(currentX) > 10 || Math.abs(currentY) > 10) {
                    setManifoldState(prev => ({
                        ...prev,
                        isRunning: false,
                        hasStarted: true,
                        trajectoryPoints: newPoints,
                        currentPoint: { x: currentX, y: currentY, nx: currentNx, ny: currentNy },
                        iteration
                    }));
                    return;
                }
                newPoints.push({ x: currentX, y: currentY, nx: currentNx, ny: currentNy });
                const next = boundary_map(currentX, currentY, currentNx, currentNy, params.a, params.b, params.epsilon);
                currentX = next.x;
                currentY = next.y;
                currentNx = next.nx;
                currentNy = next.ny;
                iteration++;
            }
            setManifoldState(prev => ({
                ...prev,
                currentPoint: { x: currentX, y: currentY, nx: currentNx, ny: currentNy },
                trajectoryPoints: [...newPoints],
                iteration,
                hasStarted: true
            }));
            if (iteration < params.maxIterations) {
                batchAnimationRef.current = requestAnimationFrame(animateStep);
            } else {
                setManifoldState(prev => ({ ...prev, isRunning: false }));
            }
        };
        batchAnimationRef.current = requestAnimationFrame(animateStep);
    }, [manifoldState, params, wasmModule]);

    const resetManifold = useCallback(() => {
        if (batchAnimationRef.current) cancelAnimationFrame(batchAnimationRef.current);
        setManifoldState(prev => ({
            ...prev,
            currentPoint: { ...prev.startPoint },
            trajectoryPoints: [],
            iteration: 0,
            isRunning: false,
            hasStarted: false
        }));
    }, []);

    useEffect(() => {
        if (mode === 'manifold') {
            resetManifold();
        }
    }, [params.a, params.b, params.epsilon, mode, resetManifold]);



    const totalManifoldPoints = useMemo(() => {
        let count = 0;
        manifoldState.manifolds.forEach(m => {
            if (m.plus?.points) count += m.plus.points.length;
            if (m.minus?.points) count += m.minus.points.length;
        });
        return count;
    }, [manifoldState.manifolds]);

    const computeUlam = useCallback(async () => {
        if (!wasmModule) return;
        setUlamState(prev => ({ ...prev, isComputing: true, needsRecompute: false }));
        try {
            const { UlamComputer } = wasmModule;
            if (!UlamComputer) {
                console.error('UlamComputer export is missing from WASM module!');
                throw new Error('UlamComputer definition missing');
            }

            const computer = new UlamComputer(
                params.a,
                params.b,
                ulamState.subdivisions,
                ulamState.pointsPerBox,
                ulamState.epsilon
            );

            ulamComputerRef.current = computer;
            const boxes = computer.get_grid_boxes();
            const invariantMeasure = computer.get_invariant_measure();
            const leftEigenvector = computer.get_left_eigenvector();

            // Get current box if we have a trajectory
            let currentBoxIndex = -1;
            if (periodicState.hasStarted && periodicState.currentPoint) {
                currentBoxIndex = computer.get_box_index(
                    periodicState.currentPoint.x,
                    periodicState.currentPoint.y
                );
            }

            setUlamState(prev => ({
                ...prev,
                isComputing: false,
                gridBoxes: boxes,
                invariantMeasure: invariantMeasure,
                leftEigenvector: leftEigenvector,
                currentBoxIndex: currentBoxIndex,
                selectedBoxIndex: null,
                transitions: null
            }));

        } catch (err) {
            console.error("Ulam computation failed:", err);
            setUlamState(prev => ({ ...prev, isComputing: false }));
        }
    }, [wasmModule, params.a, params.b, ulamState.subdivisions, ulamState.pointsPerBox, ulamState.epsilon, periodicState.hasStarted, periodicState.currentPoint]);

    useEffect(() => {
        if (mode === 'manifold') {
            setUlamState(prev => ({ ...prev, epsilon: params.epsilon }));
        }
    }, [mode, params.epsilon]);

    useEffect(() => {
        if (!ulamState.showUlamOverlay || !wasmModule) return;

        if (ulamDebounceRef.current) {
            clearTimeout(ulamDebounceRef.current);
        }


        ulamDebounceRef.current = setTimeout(() => {
            computeUlam();
        }, 200);

        return () => {
            if (ulamDebounceRef.current) {
                clearTimeout(ulamDebounceRef.current);
            }
        };
    }, [
        ulamState.showUlamOverlay,
        wasmModule,
        params.a,
        params.b,
        ulamState.epsilon,
        ulamState.subdivisions,
        ulamState.pointsPerBox,
        computeUlam
    ]);

    useEffect(() => {
        if (!ulamComputerRef.current || !ulamState.showUlamOverlay) return;

        if (periodicState.hasStarted && periodicState.currentPoint) {
            const boxIdx = ulamComputerRef.current.get_box_index(
                periodicState.currentPoint.x,
                periodicState.currentPoint.y
            );

            if (boxIdx !== ulamState.currentBoxIndex) {
                const transitions = boxIdx >= 0 ? ulamComputerRef.current.get_transitions(boxIdx) : null;
                setUlamState(prev => ({
                    ...prev,
                    currentBoxIndex: boxIdx,
                    transitions: prev.showCurrentBox ? transitions : prev.transitions,
                    selectedBoxIndex: prev.showCurrentBox ? boxIdx : prev.selectedBoxIndex
                }));
            }
        }
    }, [periodicState.currentPoint, periodicState.hasStarted, ulamState.showUlamOverlay, ulamState.currentBoxIndex]);


    useEffect(() => {
        const scene = sceneRef.current;
        if (!scene) return;

        const cleanup = () => {
            const oldMesh = scene.getObjectByName('ulam-grid');
            if (oldMesh) {
                oldMesh.geometry.dispose();
                oldMesh.material.dispose();
                scene.remove(oldMesh);
            }
        };

        if (!ulamState.showUlamOverlay || !ulamState.gridBoxes.length) {
            cleanup();
            return;
        }

        cleanup();

        const boxes = ulamState.gridBoxes;
        const count = boxes.length;
        const geometry = new THREE.PlaneGeometry(1, 1);
        const material = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.5, // Base opacity, vertex colors will modulate
            side: THREE.DoubleSide,
            depthWrite: false // Good for overlays
        });

        const mesh = new THREE.InstancedMesh(geometry, material, count);
        mesh.name = 'ulam-grid';
        mesh.userData.type = 'ulamGrid';

        const dummy = new THREE.Object3D();
        const color = new THREE.Color();

        // Build map of transitions for fast lookup if selected
        const transitionMap = new Map();
        if (ulamState.selectedBoxIndex !== null && ulamState.transitions) {
            ulamState.transitions.forEach(t => {
                transitionMap.set(t.index, t.probability);
            });
        }

        let maxMeasure = 0;
        if (ulamState.invariantMeasure) {
            maxMeasure = Math.max(...ulamState.invariantMeasure);
        }

        boxes.forEach((box, i) => {
            const cx = box.center[0];
            const cy = box.center[1];
            const rx = box.radius[0];
            const ry = box.radius[1];

            dummy.position.set(cx, cy, -0.05); // Slightly behind orbits
            dummy.scale.set(rx * 2 * 0.95, ry * 2 * 0.95, 1);
            dummy.updateMatrix();
            mesh.setMatrixAt(i, dummy.matrix);

            // Coloring logic - Priority: Current Box > Selected Box > Transitions > Invariant Measure
            const isCurrentBox = ulamState.showCurrentBox && i === ulamState.currentBoxIndex;
            const isSelectedBox = ulamState.selectedBoxIndex !== null && i === ulamState.selectedBoxIndex;

            if (isCurrentBox && !isSelectedBox) {
                // Current trajectory position box - bright magenta
                color.setHex(0xff00ff);
                mesh.setColorAt(i, color);
            } else if (ulamState.selectedBoxIndex !== null) {
                // Interaction mode: Show transitions from selected box
                if (i === ulamState.selectedBoxIndex) {
                    color.setHex(0x00ffff); // Cyan for source
                    mesh.setColorAt(i, color);
                } else if (transitionMap.has(i)) {
                    const prob = transitionMap.get(i);
                    // Heatmap: Blue (low) -> Red (high)
                    color.setHSL(0.7 - prob * 0.7, 1.0, 0.5);
                    mesh.setColorAt(i, color);
                } else {
                    // Dim unrelated boxes
                    color.setHex(0x222222);
                    mesh.setColorAt(i, color);
                }
            } else if (ulamState.invariantMeasure && ulamState.invariantMeasure.length === count) {
                // Invariant Measure mode
                const measure = ulamState.invariantMeasure[i];
                if (measure > 0) {
                    // Log scale often looks better for measures
                    const intensity = measure / maxMeasure;
                    // Viridis-like or simple Heatmap: Dark Blue -> Green -> Yellow
                    // HSL: 0.66 (Blue) -> 0.16 (Yellow)
                    const h = 0.66 - (intensity * 0.5);
                    color.setHSL(h, 1.0, 0.5);
                    mesh.setColorAt(i, color);
                } else {
                    color.setHex(0x111111);
                    mesh.setColorAt(i, color);
                }
            } else {
                color.setHex(0x333333);
                mesh.setColorAt(i, color);
            }
        });

        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

        scene.add(mesh);

        return cleanup;
    }, [ulamState.showUlamOverlay, ulamState.gridBoxes, ulamState.selectedBoxIndex, ulamState.transitions, ulamState.invariantMeasure, ulamState.currentBoxIndex, ulamState.showCurrentBox]);

    useEffect(() => {
        if (!sceneRef.current) return;
        const scene = sceneRef.current;

        const toRemove = [];
        scene.traverse(child => {
            if (mode !== 'periodic' && (child.userData.type === 'orbit' || child.userData.type === 'trajectory' || child.userData.type === 'orbitLine')) {
                toRemove.push(child);
            }
            if (mode !== 'manifold' && (child.userData.type === 'manifold' || child.userData.type === 'fixedPoint')) {
                toRemove.push(child);
            }
            if (mode !== 'ulam' && child.userData.type === 'ulamGrid') {
                // Do NOT remove ulamGrid based on mode. Remove only if !showUlamOverlay (handled in ulam render effect)
                // So we remove this check.
            }
        });

        toRemove.forEach(obj => {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
            scene.remove(obj);
        });
    }, [mode]);



    return (
        <div style={styles.container}>
            <div style={styles.sidebar}>
                {/* Mode Toggle */}
                <div style={styles.section}>
                    <h3 style={styles.sectionTitle}>Mode</h3>
                    <div style={styles.toggleContainer}>
                        <button
                            onClick={() => setMode('periodic')}
                            style={{ ...styles.toggleButton, ...(mode === 'periodic' ? styles.toggleActive : {}) }}
                        >
                            Periodic Orbits
                        </button>
                        <button
                            onClick={() => setMode('manifold')}
                            style={{ ...styles.toggleButton, ...(mode === 'manifold' ? styles.toggleActive : {}) }}
                        >
                            Manifold Approximation
                        </button>
                    </div>
                </div>

                {/* Dynamic System Selector */}
                <div style={styles.section}>
                    <h3 style={styles.sectionTitle}>Dynamic System</h3>
                    <select
                        value={dynamicSystem}
                        onChange={(e) => setDynamicSystem(e.target.value)}
                        style={{
                            width: '100%',
                            padding: '8px 12px',
                            backgroundColor: '#2d2d2d',
                            color: '#fff',
                            border: '1px solid #444',
                            borderRadius: '6px',
                            fontSize: '14px',
                            cursor: 'pointer'
                        }}
                        disabled={periodicState.isRunning || animationState.isAnimating}
                    >
                        <option value="henon">Hénon Map</option>
                        <option value="duffing">Duffing Map</option>
                    </select>
                    <div style={{ fontSize: '11px', color: '#666', marginTop: '4px' }}>
                        {dynamicSystem === 'henon'
                            ? 'x\' = 1 - ax² + y, y\' = bx'
                            : 'x\' = y, y\' = -bx + ay - y³'}
                    </div>
                </div>

                {/* System Parameters */}
                <div style={styles.section}>
                    <h3 style={styles.sectionTitle}>System Parameters</h3>
                    <label style={styles.label}>
                        <div style={styles.paramRow}>
                            <span>a =</span>
                            <input type="number" step="0.01" value={params.a}
                                onChange={(e) => setParams({ ...params, a: parseFloat(e.target.value) || 0.1 })}
                                style={styles.numberInput} disabled={periodicState.isRunning} />
                        </div>
                        <input type="range" min="0.1" max={dynamicSystem === 'duffing' ? 3.0 : 2.0} step="0.01" value={params.a}
                            onChange={(e) => setParams({ ...params, a: parseFloat(e.target.value) })}
                            style={styles.slider} disabled={periodicState.isRunning} />
                    </label>
                    <label style={styles.label}>
                        <div style={styles.paramRow}>
                            <span>b =</span>
                            <input type="number" step="0.01" value={params.b}
                                onChange={(e) => setParams({ ...params, b: parseFloat(e.target.value) || 0.1 })}
                                style={styles.numberInput} disabled={periodicState.isRunning} />
                        </div>
                        <input type="range" min="0.1" max="0.5" step="0.01" value={params.b}
                            onChange={(e) => setParams({ ...params, b: parseFloat(e.target.value) })}
                            style={styles.slider} disabled={periodicState.isRunning} />
                    </label>
                    {mode === 'periodic' && (
                        <>
                            <label style={styles.label}>
                                <div style={styles.paramRow}>
                                    <span>Max Period =</span>
                                    <input type="number" step="1" min="1" max="20" value={params.maxPeriod}
                                        onChange={(e) => setParams({ ...params, maxPeriod: parseInt(e.target.value) || 2 })}
                                        style={styles.numberInput} disabled={periodicState.isRunning} />
                                </div>
                                <input type="range" min="2" max="10" step="1" value={params.maxPeriod}
                                    onChange={(e) => setParams({ ...params, maxPeriod: parseInt(e.target.value) })}
                                    style={styles.slider} disabled={periodicState.isRunning} />
                            </label>
                            <label style={styles.label}>
                                <div style={styles.paramRow}>
                                    <span>Max Iterations =</span>
                                    <input type="number" step="100" min="100" max="10000" value={params.maxIterations}
                                        onChange={(e) => setParams({ ...params, maxIterations: parseInt(e.target.value) || 100 })}
                                        style={styles.numberInput} disabled={periodicState.isRunning} />
                                </div>
                                <input type="range" min="100" max="5000" step="100" value={params.maxIterations}
                                    onChange={(e) => setParams({ ...params, maxIterations: parseInt(e.target.value) })}
                                    style={styles.slider} disabled={periodicState.isRunning} />
                            </label>
                        </>
                    )}
                    {mode === 'manifold' && (
                        <label style={styles.label}>
                            <div style={styles.paramRow}>
                                <span>epsilon =</span>
                                <input type="number" step="0.001" value={params.epsilon}
                                    onChange={(e) => setParams({ ...params, epsilon: parseFloat(e.target.value) || 0.01 })}
                                    style={styles.numberInput} />
                            </div>
                            <input type="range" min="0.001" max="0.2" step="0.001" value={params.epsilon}
                                onChange={(e) => setParams({ ...params, epsilon: parseFloat(e.target.value) })}
                                style={styles.slider} />
                        </label>
                    )}

                    {mode === 'manifold' && dynamicSystem === 'henon' && (
                        <div style={{ marginTop: '12px', borderTop: '1px solid #333', paddingTop: '12px' }}>
                            <h4 style={{ fontSize: '12px', fontWeight: '600', marginBottom: '10px', color: '#888' }}>
                                Manifold Display
                            </h4>

                            {/* Unstable Manifold Toggle */}
                            <label style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '10px',
                                marginBottom: '10px',
                                padding: '8px 10px',
                                backgroundColor: manifoldState.showUnstableManifold ? 'rgba(30, 144, 255, 0.15)' : 'transparent',
                                borderRadius: '6px',
                                border: `1px solid ${manifoldState.showUnstableManifold ? ORBIT_COLORS.manifold : '#444'}`,
                                cursor: 'pointer',
                                transition: 'all 0.2s ease'
                            }}>
                                <input
                                    type="checkbox"
                                    checked={manifoldState.showUnstableManifold}
                                    onChange={(e) => setManifoldState(prev => ({ ...prev, showUnstableManifold: e.target.checked }))}
                                    style={{ width: '16px', height: '16px', accentColor: ORBIT_COLORS.manifold }}
                                />
                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                    <div style={{
                                        width: '12px',
                                        height: '3px',
                                        backgroundColor: ORBIT_COLORS.manifold,
                                        borderRadius: '2px'
                                    }} />
                                    <span style={{ color: '#ccc', fontSize: '13px' }}>Unstable Manifold</span>
                                </div>
                            </label>

                            <label style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '10px',
                                marginBottom: '10px',
                                padding: '8px 10px',
                                backgroundColor: manifoldState.showStableManifold ? 'rgba(255, 165, 0, 0.15)' : 'transparent',
                                borderRadius: '6px',
                                border: `1px solid ${manifoldState.showStableManifold ? ORBIT_COLORS.stableManifold : '#444'}`,
                                cursor: 'pointer',
                                transition: 'all 0.2s ease'
                            }}>
                                <input
                                    type="checkbox"
                                    checked={manifoldState.showStableManifold}
                                    onChange={(e) => setManifoldState(prev => ({ ...prev, showStableManifold: e.target.checked }))}
                                    style={{ width: '16px', height: '16px', accentColor: ORBIT_COLORS.stableManifold }}
                                />
                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                    <div style={{
                                        width: '12px',
                                        height: '3px',
                                        backgroundColor: ORBIT_COLORS.stableManifold,
                                        borderRadius: '2px'
                                    }} />
                                    <span style={{ color: '#ccc', fontSize: '13px' }}>Stable Manifold</span>
                                </div>
                            </label>

                            {manifoldState.showStableManifold && (
                                <div style={{
                                    marginTop: '8px',
                                    padding: '10px',
                                    backgroundColor: 'rgba(0,0,0,0.2)',
                                    borderRadius: '6px',
                                    border: '1px solid #333'
                                }}>
                                    <div style={{ fontSize: '11px', fontWeight: '600', color: '#888', marginBottom: '8px' }}>
                                        Intersection Detection
                                    </div>
                                    <label style={styles.label}>
                                        <div style={styles.paramRow}>
                                            <span>Threshold ε</span>
                                            <input type="number" step="0.01" value={manifoldState.intersectionThreshold}
                                                onChange={(e) => setManifoldState(prev => ({
                                                    ...prev,
                                                    intersectionThreshold: parseFloat(e.target.value) || 0.01
                                                }))}
                                                style={styles.numberInput} />
                                        </div>
                                        <input type="range" min="0.001" max="0.2" step="0.001"
                                            value={manifoldState.intersectionThreshold}
                                            onChange={(e) => setManifoldState(prev => ({
                                                ...prev,
                                                intersectionThreshold: parseFloat(e.target.value)
                                            }))}
                                            style={styles.slider} />
                                    </label>
                                    <div style={{ fontSize: '11px', marginTop: '6px' }}>
                                        {(() => {
                                            const heteroClinic = manifoldState.intersections.filter(i => i.has_intersection);
                                            if (heteroClinic.length > 0) {
                                                const minDist = Math.min(...heteroClinic.map(i => i.min_distance));
                                                return (
                                                    <div style={{
                                                        color: '#e74c3c',
                                                        fontWeight: 'bold',
                                                    }}>
                                                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                                            <span style={{ fontSize: '14px' }}>⚠</span>
                                                            <span>Heteroclinic connection!</span>
                                                        </div>
                                                        <div style={{ fontSize: '10px', color: '#e74c3c', opacity: 0.8, marginTop: '4px' }}>
                                                            {heteroClinic.length} connection{heteroClinic.length > 1 ? 's' : ''} found (min d = {minDist.toFixed(4)})
                                                        </div>
                                                    </div>
                                                );
                                            } else if (manifoldState.intersections.length > 0) {
                                                return <div style={{ color: '#27ae60' }}>✓ No heteroclinic connections ({manifoldState.intersections.length} pairs checked)</div>;
                                            } else {
                                                return <div style={{ color: '#888' }}>Need ≥2 saddles for detection</div>;
                                            }
                                        })()}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Save PNG Button - available in both modes */}
                <button
                    onClick={savePNG}
                    style={{
                        ...styles.button,
                        width: '100%',
                        marginTop: '12px',
                        backgroundColor: '#2d4a2d',
                        borderColor: '#3a6a3a',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '8px'
                    }}
                >
                    Save as PNG
                </button>

                {/* Parameter Animation Section (manifold mode only) */}
                {mode === 'manifold' && (
                    <div style={{ marginTop: '16px', borderTop: '1px solid #333', paddingTop: '16px' }}>
                        <h4 style={{ fontSize: '12px', fontWeight: '600', marginBottom: '12px', color: '#888' }}>Parameter Animation</h4>

                        <label style={styles.label}>
                            <div style={styles.paramRow}>
                                <span>Animate:</span>
                                <select
                                    value={animationState.parameter}
                                    onChange={(e) => setAnimationState(prev => ({ ...prev, parameter: e.target.value }))}
                                    style={{ ...styles.numberInput, width: '100px' }}
                                    disabled={animationState.isAnimating}
                                >
                                    <option value="a">a</option>
                                    <option value="b">b</option>
                                    <option value="epsilon">epsilon</option>
                                </select>
                            </div>
                        </label>

                        <label style={styles.label}>
                            <div style={styles.paramRow}>
                                <span>Direction:</span>
                                <div style={{ display: 'flex', gap: '4px' }}>
                                    <button
                                        onClick={() => setAnimationState(prev => ({ ...prev, direction: -1 }))}
                                        style={{
                                            ...styles.button,
                                            width: '40px',
                                            padding: '4px',
                                            marginBottom: 0,
                                            backgroundColor: animationState.direction === -1 ? '#3d5afe' : '#2d2d2d'
                                        }}
                                        disabled={animationState.isAnimating}
                                    >−</button>
                                    <button
                                        onClick={() => setAnimationState(prev => ({ ...prev, direction: 1 }))}
                                        style={{
                                            ...styles.button,
                                            width: '40px',
                                            padding: '4px',
                                            marginBottom: 0,
                                            backgroundColor: animationState.direction === 1 ? '#3d5afe' : '#2d2d2d'
                                        }}
                                        disabled={animationState.isAnimating}
                                    >+</button>
                                </div>
                            </div>
                        </label>

                        <label style={styles.label}>
                            <div style={styles.paramRow}>
                                <span>Range:</span>
                                <input
                                    type="number"
                                    step="0.05"
                                    min="0.01"
                                    max="1.0"
                                    value={animationState.rangeValue}
                                    onChange={(e) => setAnimationState(prev => ({ ...prev, rangeValue: parseFloat(e.target.value) || 0.1 }))}
                                    style={styles.numberInput}
                                    disabled={animationState.isAnimating}
                                />
                            </div>
                            <input
                                type="range"
                                min="0.01"
                                max="0.5"
                                step="0.01"
                                value={animationState.rangeValue}
                                onChange={(e) => setAnimationState(prev => ({ ...prev, rangeValue: parseFloat(e.target.value) }))}
                                style={styles.slider}
                                disabled={animationState.isAnimating}
                            />
                        </label>

                        <label style={styles.label}>
                            <div style={styles.paramRow}>
                                <span>Steps:</span>
                                <input
                                    type="number"
                                    step="1"
                                    min="5"
                                    max="30"
                                    value={animationState.steps}
                                    onChange={(e) => setAnimationState(prev => ({ ...prev, steps: parseInt(e.target.value) || 10 }))}
                                    style={styles.numberInput}
                                    disabled={animationState.isAnimating}
                                />
                            </div>
                            <input
                                type="range"
                                min="5"
                                max="30"
                                step="1"
                                value={animationState.steps}
                                onChange={(e) => setAnimationState(prev => ({ ...prev, steps: parseInt(e.target.value) }))}
                                style={styles.slider}
                                disabled={animationState.isAnimating}
                            />
                        </label>

                        {/* Play and Record buttons */}
                        <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
                            <button
                                onClick={animationState.isAnimating ? stopAnimation : startAnimation}
                                disabled={(manifoldState.isComputing && !animationState.isAnimating) || recordingState.isEncoding}
                                style={{
                                    ...styles.button,
                                    flex: 1,
                                    marginBottom: 0,
                                    backgroundColor: animationState.isAnimating ? '#8b0000' : '#1a4a1a',
                                    borderColor: animationState.isAnimating ? '#b22222' : '#2a6a2a'
                                }}
                            >
                                {animationState.isAnimating ? '⏹ Stop' : '▶ Play'}
                                {recordingState.recordingEnabled && !animationState.isAnimating && ' & Rec'}
                            </button>
                            <button
                                onClick={toggleRecording}
                                disabled={animationState.isAnimating || recordingState.isEncoding}
                                style={{
                                    ...styles.button,
                                    width: '50px',
                                    marginBottom: 0,
                                    backgroundColor: recordingState.recordingEnabled ? '#b22222' : '#2d2d2d',
                                    borderColor: recordingState.recordingEnabled ? '#ff4444' : '#444'
                                }}
                                title={recordingState.recordingEnabled ? 'Recording enabled - will record next animation' : 'Enable recording'}
                            >
                                {recordingState.recordingEnabled ? '🔴' : '⏺'}
                            </button>
                        </div>

                        {/* Recording status */}
                        {recordingState.recordingEnabled && !animationState.isAnimating && !recordingState.isEncoding && (
                            <div style={{ marginTop: '8px', padding: '6px', background: '#1a0a0a', borderRadius: '4px', border: '1px solid #b22222' }}>
                                <div style={{ fontSize: '11px', color: '#ff6666' }}>
                                    🔴 Recording armed - Press Play to start
                                </div>
                            </div>
                        )}

                        {/* Encoding status */}
                        {recordingState.isEncoding && (
                            <div style={{ marginTop: '8px', padding: '8px', background: '#0a0a1a', borderRadius: '4px', border: '1px solid #3d5afe' }}>
                                <div style={{ fontSize: '11px', color: '#7986cb', marginBottom: '4px' }}>
                                    🔄 Encoding video... ({recordingState.frameCount} frames)
                                </div>
                                <div style={{
                                    height: '4px',
                                    backgroundColor: '#1a1a2e',
                                    borderRadius: '2px',
                                    overflow: 'hidden'
                                }}>
                                    <div style={{
                                        width: '100%',
                                        height: '100%',
                                        backgroundColor: '#3d5afe',
                                        animation: 'pulse 1s infinite'
                                    }} />
                                </div>
                            </div>
                        )}

                        {/* Error message */}
                        {recordingState.error && (
                            <div style={{ marginTop: '8px', padding: '6px', background: '#1a0a0a', borderRadius: '4px', fontSize: '10px', color: '#ff6666' }}>
                                Error: {recordingState.error}
                            </div>
                        )}

                        {animationState.isAnimating && (
                            <div style={{ marginTop: '8px', padding: '8px', background: '#0f0f0f', borderRadius: '4px' }}>
                                <div style={{ fontSize: '11px', color: '#888', marginBottom: '4px' }}>
                                    {animationState.parameter}: {animationState.baseValue?.toFixed(3)} → {animationState.targetValue?.toFixed(3)}
                                </div>
                                <div style={{ fontSize: '11px', color: '#888' }}>
                                    Current: <span style={{ color: '#4CAF50', fontWeight: 'bold' }}>
                                        {params[animationState.parameter].toFixed(4)}
                                    </span>
                                    {recordingState.recordingEnabled && (
                                        <span style={{ color: '#ff6666', marginLeft: '8px' }}>
                                            📹 {recordingState.frameCount} frames
                                        </span>
                                    )}
                                </div>
                                <div style={{ fontSize: '10px', color: '#555', marginTop: '4px' }}>
                                    Step {animationState.currentStep} / {animationState.steps}
                                    {manifoldState.isComputing && <span style={{ color: '#ff9800', marginLeft: '8px' }}>Computing...</span>}
                                </div>
                                {/* Progress bar */}
                                <div style={{
                                    marginTop: '6px',
                                    height: '4px',
                                    backgroundColor: '#333',
                                    borderRadius: '2px',
                                    overflow: 'hidden'
                                }}>
                                    <div style={{
                                        width: `${(animationState.currentStep / animationState.steps) * 100}%`,
                                        height: '100%',
                                        backgroundColor: recordingState.recordingEnabled ? '#ff6666' : '#4CAF50',
                                        transition: 'width 0.3s ease'
                                    }} />
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Ulam Settings Section */}
                <div style={{ marginTop: '16px', borderTop: '1px solid #333', paddingTop: '16px' }}>
                    <label style={styles.checkboxLabel}>
                        <input type="checkbox" checked={ulamState.showUlamOverlay}
                            onChange={(e) => setUlamState({ ...ulamState, showUlamOverlay: e.target.checked })} />
                        Show Ulam Grid
                    </label>

                    {ulamState.showUlamOverlay && (
                        <>
                            <label style={styles.checkboxLabel}>
                                <input type="checkbox" checked={ulamState.showTransitions}
                                    onChange={(e) => setUlamState({ ...ulamState, showTransitions: e.target.checked })} />
                                Show Transitions
                            </label>
                            {mode === 'periodic' && (
                                <label style={styles.checkboxLabel}>
                                    <input type="checkbox" checked={ulamState.showCurrentBox}
                                        onChange={(e) => setUlamState({ ...ulamState, showCurrentBox: e.target.checked })} />
                                    Track Current Position
                                </label>
                            )}



                            <label style={styles.label}>
                                <div style={styles.paramRow}>
                                    <span>Grid =</span>
                                    <input type="number" step="1" min="10" max="80" value={ulamState.subdivisions}
                                        onChange={(e) => setUlamState({ ...ulamState, subdivisions: parseInt(e.target.value) || 10 })}
                                        style={styles.numberInput} disabled={ulamState.isComputing} />
                                </div>
                                <input type="range" min="10" max="60" step="5" value={ulamState.subdivisions}
                                    onChange={(e) => setUlamState({ ...ulamState, subdivisions: parseInt(e.target.value) })}
                                    style={styles.slider} disabled={ulamState.isComputing} />
                            </label>
                            <label style={styles.label}>
                                <div style={styles.paramRow}>
                                    <span>Samples =</span>
                                    <input type="number" step="16" min="16" max="256" value={ulamState.pointsPerBox}
                                        onChange={(e) => setUlamState({ ...ulamState, pointsPerBox: parseInt(e.target.value) || 64 })}
                                        style={styles.numberInput} disabled={ulamState.isComputing} />
                                </div>
                                <input type="range" min="16" max="256" step="16" value={ulamState.pointsPerBox}
                                    onChange={(e) => setUlamState({ ...ulamState, pointsPerBox: parseInt(e.target.value) })}
                                    style={styles.slider} disabled={ulamState.isComputing} />
                                <span style={{ fontSize: '10px', color: '#666' }}>Points per box (√n × √n grid)</span>
                            </label>
                            <div style={{ marginTop: '8px', marginBottom: '8px', fontSize: '12px', color: ulamState.isComputing ? '#ff9800' : '#4CAF50', display: 'flex', alignItems: 'center', gap: '6px' }}>
                                {ulamState.isComputing ? (
                                    <>
                                        <span className="spinner" style={{ width: '10px', height: '10px', border: '2px solid #ff9800', borderTop: '2px solid transparent', borderRadius: '50%', display: 'inline-block', animation: 'spin 1s linear infinite' }}></span>
                                        Computing Ulam Grid...
                                    </>
                                ) : (
                                    <span>✓ Grid up to date</span>
                                )}
                            </div>
                            <style>{`
                                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                            `}</style>

                            {ulamState.gridBoxes.length > 0 && (
                                <div style={{ marginTop: '12px', padding: '8px', background: '#0f0f0f', borderRadius: '4px' }}>
                                    <div style={{ fontSize: '11px', fontWeight: '600', color: '#888', marginBottom: '6px' }}>
                                        Invariant Measure
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                        <span style={{ fontSize: '10px', color: '#666' }}>Low</span>
                                        <div style={{
                                            flex: 1,
                                            height: '12px',
                                            borderRadius: '2px',
                                            background: 'linear-gradient(to right, hsl(238, 100%, 50%), hsl(180, 100%, 50%), hsl(100, 100%, 50%), hsl(60, 100%, 50%))'
                                        }} />
                                        <span style={{ fontSize: '10px', color: '#666' }}>High</span>
                                    </div>
                                    <div style={{ fontSize: '9px', color: '#555', marginTop: '4px' }}>
                                        Probability of trajectory visiting each region
                                    </div>
                                </div>
                            )}

                            {ulamState.currentBoxIndex >= 0 && (
                                <div style={{ fontSize: '11px', color: '#888', marginTop: '4px' }}>
                                    Current box: {ulamState.currentBoxIndex}
                                </div>
                            )}
                        </>
                    )}
                </div>

                {/* Mode-specific controls */}
                {mode === 'periodic' && (
                    <>
                        <div style={styles.section}>
                            <h3 style={styles.sectionTitle}>Controls</h3>
                            <button onClick={stepForward} disabled={!periodicState.isReady || periodicState.isRunning} style={styles.button}>
                                Step Forward
                            </button>
                            <button onClick={runToConvergence} disabled={!periodicState.isReady || periodicState.isRunning}
                                style={{ ...styles.button, ...styles.runButton }}>
                                {periodicState.isRunning ? 'Running...' : 'Run to Max Iterations'}
                            </button>
                            <button onClick={reset} disabled={periodicState.isRunning} style={{ ...styles.button, ...styles.resetButton }}>
                                Reset
                            </button>
                        </div>

                        <div style={styles.section}>
                            <h3 style={styles.sectionTitle}>Periodic Orbits</h3>
                            {[1, 2, 3, 4, 5].map(p => (
                                <label key={p} style={styles.checkboxLabel}>
                                    <input type="checkbox" checked={filters[`period${p}`]}
                                        onChange={(e) => setFilters({ ...filters, [`period${p}`]: e.target.checked })} />
                                    <span style={{ ...styles.colorBox, backgroundColor: ORBIT_COLORS[`period${p}`].stable }} />
                                    Period-{p} ({periodicState.orbits.filter(o => o.period === p).length})
                                </label>
                            ))}
                            <label style={styles.checkboxLabel}>
                                <input type="checkbox" checked={periodicState.showOrbits}
                                    onChange={(e) => setPeriodicState({ ...periodicState, showOrbits: e.target.checked })} />
                                Show orbit markers
                            </label>
                            <label style={styles.checkboxLabel}>
                                <input type="checkbox" checked={periodicState.showTrail}
                                    onChange={(e) => setPeriodicState({ ...periodicState, showTrail: e.target.checked })} />
                                Show trajectory trail
                            </label>
                            <label style={styles.checkboxLabel}>
                                <input type="checkbox" checked={periodicState.showOrbitLines}
                                    onChange={(e) => setPeriodicState({ ...periodicState, showOrbitLines: e.target.checked })} />
                                Show orbit lines
                            </label>
                        </div>
                    </>
                )}

                {mode === 'manifold' && (
                    <>
                        <div style={styles.section}>
                            <h3 style={styles.sectionTitle}>Controls</h3>
                            <button onClick={stepForwardManifold} disabled={!manifoldState.isReady || manifoldState.isRunning} style={styles.button}>
                                Step Forward
                            </button>
                            <button onClick={runToConvergenceManifold} disabled={!manifoldState.isReady || manifoldState.isRunning}
                                style={{ ...styles.button, ...styles.runButton }}>
                                {manifoldState.isRunning ? 'Running...' : 'Run to Max Iterations'}
                            </button>
                            <button onClick={resetManifold} disabled={manifoldState.isRunning} style={{ ...styles.button, ...styles.resetButton }}>
                                Reset
                            </button>
                        </div>

                        <div style={styles.section}>
                            <h3 style={styles.sectionTitle}>Periodic Orbits</h3>
                            <label style={styles.checkboxLabel}>
                                <input type="checkbox" checked={manifoldState.showOrbits}
                                    onChange={(e) => setManifoldState({ ...manifoldState, showOrbits: e.target.checked })} />
                                Show orbit markers
                            </label>
                            <label style={styles.checkboxLabel}>
                                <input type="checkbox" checked={manifoldState.showOrbitLines}
                                    onChange={(e) => setManifoldState({ ...manifoldState, showOrbitLines: e.target.checked })} />
                                Show orbit lines
                            </label>
                            <label style={styles.checkboxLabel}>
                                <input type="checkbox" checked={manifoldState.showTrail}
                                    onChange={(e) => setManifoldState({ ...manifoldState, showTrail: e.target.checked })} />
                                Show trajectory trail
                            </label>
                        </div>
                        {manifoldState.fixedPoints.length > 0 && (
                            <div style={styles.section}>
                                <h3 style={styles.sectionTitle}>Fixed Points ({manifoldState.fixedPoints.length})</h3>
                                {manifoldState.fixedPoints.map((fp, i) => (
                                    <div key={i} style={styles.fixedPointItem}>
                                        <span style={{ fontWeight: 'bold', color: fp.stability === 'Attractor' ? '#00ff00' : fp.stability === 'Repeller' ? '#ff4444' : '#ffaa00' }}>
                                            {fp.stability}
                                        </span>
                                        <span> ({fp.x.toFixed(3)}, {fp.y.toFixed(3)})</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </>
                )}

                {/* Info panel */}
                <div style={styles.info}>
                    <div style={styles.infoHeader}>
                        {dynamicSystem === 'henon'
                            ? "Hénon Map: x' = 1 - ax² + y, y' = bx"
                            : "Duffing Map: x' = y, y' = -bx + ay - y³"}
                    </div>
                    {mode === 'periodic' ? (
                        <>
                            <div>Status: {periodicState.isReady ? (periodicState.isRunning ? 'Running...' : 'Ready') : 'Loading...'}</div>
                            <div>Iteration: {periodicState.iteration} / {params.maxIterations}</div>
                            {periodicState.hasStarted && <div>Position: ({periodicState.currentPoint.x.toFixed(4)}, {periodicState.currentPoint.y.toFixed(4)})</div>}
                            <div>Orbits found: {periodicState.orbits.length}</div>
                        </>
                    ) : (
                        <>
                            <div>Status: {manifoldState.isComputing ? 'Computing...' : (manifoldState.isRunning ? 'Running...' : 'Ready')}</div>
                            {manifoldState.hasStarted && (
                                <>
                                    <div>Iteration: {manifoldState.iteration} / {params.maxIterations}</div>
                                    <div>Position: ({manifoldState.currentPoint.x.toFixed(4)}, {manifoldState.currentPoint.y.toFixed(4)})</div>
                                </>
                            )}
                            <div>Manifolds: {manifoldState.manifolds.length}</div>
                            <div>Total Points: {totalManifoldPoints.toLocaleString()}</div>
                        </>
                    )}
                </div>
            </div>

            <div style={styles.viewport}>
                <canvas ref={canvasRef} style={styles.canvas} />

                {/* Tooltip visualization */}
                {tooltip.visible && tooltip.data && (
                    <div style={{
                        position: 'absolute',
                        left: tooltip.x + 15,
                        top: tooltip.y + 15,
                        backgroundColor: 'rgba(20, 20, 20, 0.95)',
                        border: '1px solid #444',
                        borderRadius: '4px',
                        padding: '12px',
                        zIndex: 1000,
                        pointerEvents: 'none',
                        minWidth: '180px',
                        maxWidth: '280px',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
                        fontSize: '12px',
                        fontFamily: 'monospace'
                    }}>
                        <div style={{ fontWeight: 'bold', color: '#fff', marginBottom: '8px', borderBottom: '1px solid #444', paddingBottom: '4px' }}>
                            {tooltip.data.type}
                            {tooltip.data.isCurrentBox && <span style={{ color: '#ff00ff', marginLeft: '8px' }}>● Current</span>}
                        </div>

                        {/* Ulam Box Tooltip */}
                        {tooltip.data.type === 'Ulam Box' ? (
                            <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '6px 12px' }}>
                                <span style={{ color: '#888' }}>Box #:</span>
                                <span style={{ color: '#00ccff' }}>{tooltip.data.boxIndex}</span>

                                <span style={{ color: '#888' }}>Center:</span>
                                <span style={{ color: '#ddd' }}>
                                    ({tooltip.data.pos.x.toFixed(2)}, {tooltip.data.pos.y.toFixed(2)})
                                </span>

                                <span style={{ color: '#888' }}>Measure:</span>
                                <span style={{ color: '#4CAF50' }}>
                                    {(tooltip.data.measure * 100).toFixed(2)}%
                                    <span style={{ color: '#666', marginLeft: '4px' }}>
                                        ({tooltip.data.measurePercent.toFixed(0)}% of max)
                                    </span>
                                </span>

                                <span style={{ color: '#888' }}>Transitions:</span>
                                <span style={{ color: '#ddd' }}>{tooltip.data.numTransitions} boxes</span>

                                {tooltip.data.topTransitions && tooltip.data.topTransitions.length > 0 && (
                                    <>
                                        <span style={{ color: '#888' }}>Top targets:</span>
                                        <div style={{ fontSize: '10px' }}>
                                            {tooltip.data.topTransitions.map((t, i) => (
                                                <div key={i} style={{ color: '#ff9800' }}>
                                                    → Box {t.index}: {(t.probability * 100).toFixed(1)}%
                                                </div>
                                            ))}
                                        </div>
                                    </>
                                )}
                            </div>
                        ) : (
                            /* Periodic/Fixed Point Tooltip */
                            <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '8px 16px' }}>
                                <span style={{ color: '#888' }}>Position:</span>
                                <span style={{ color: '#00ccff' }}>
                                    ({tooltip.data.pos.x.toFixed(4)}, {tooltip.data.pos.y.toFixed(4)})
                                </span>

                                <span style={{ color: '#888' }}>Stability:</span>
                                <span style={{
                                    color: (tooltip.data.stability?.toLowerCase() === 'attractor' || tooltip.data.stability?.toLowerCase() === 'stable') ? '#27ae60' :
                                        (tooltip.data.stability?.toLowerCase() === 'repeller' || tooltip.data.stability?.toLowerCase() === 'unstable') ? '#e74c3c' : '#e74c3c'
                                }}>
                                    {tooltip.data.stability || 'Unknown'}
                                </span>

                                <span style={{ color: '#888' }}>Period:</span>
                                <span style={{ color: '#ddd' }}>{tooltip.data.period}</span>

                                {tooltip.data.jacobian && (
                                    <>
                                        <span style={{ color: '#888' }}>Jacobian:</span>
                                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px', background: '#333', padding: '4px', borderRadius: '2px' }}>
                                            <span>{tooltip.data.jacobian.j11?.toFixed(3)}</span>
                                            <span>{tooltip.data.jacobian.j12?.toFixed(3)}</span>
                                            <span>{tooltip.data.jacobian.j21?.toFixed(3)}</span>
                                            <span>{tooltip.data.jacobian.j22?.toFixed(3)}</span>
                                        </div>

                                        <span style={{ color: '#888' }}>Det/Trace:</span>
                                        <span style={{ color: '#ddd' }}>
                                            D={tooltip.data.jacobian.det?.toFixed(3)}, Tr={tooltip.data.jacobian.trace?.toFixed(3)}
                                        </span>
                                    </>
                                )}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div >
    );
}

const styles = {
    container: { display: 'flex', height: '100vh', width: '100vw', backgroundColor: '#0a0a0a', fontFamily: 'system-ui, -apple-system, sans-serif', color: '#e0e0e0', overflow: 'hidden' },
    sidebar: { width: '320px', minWidth: '320px', padding: '20px', backgroundColor: '#1a1a1a', borderRight: '1px solid #333', overflowY: 'auto' },
    section: { marginBottom: '24px' },
    sectionTitle: { fontSize: '14px', fontWeight: '600', marginBottom: '12px', color: '#fff', textTransform: 'uppercase', letterSpacing: '0.5px' },
    toggleContainer: { display: 'flex', gap: '4px' },
    toggleButton: { flex: 1, padding: '10px', backgroundColor: '#2d2d2d', color: '#888', border: '1px solid #444', borderRadius: '4px', cursor: 'pointer', fontSize: '12px', fontWeight: '500', transition: 'all 0.2s' },
    toggleActive: { backgroundColor: '#3d5afe', color: '#fff', borderColor: '#3d5afe' },
    label: { display: 'flex', flexDirection: 'column', marginBottom: '12px', fontSize: '13px', color: '#b0b0b0' },
    paramRow: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' },
    numberInput: { width: '80px', padding: '4px 8px', backgroundColor: '#2d2d2d', color: '#fff', border: '1px solid #444', borderRadius: '4px', fontSize: '13px', textAlign: 'right' },
    slider: { marginTop: '6px', width: '100%' },
    checkboxLabel: { display: 'flex', alignItems: 'center', marginBottom: '10px', fontSize: '13px', cursor: 'pointer', gap: '8px' },
    colorBox: { width: '14px', height: '14px', borderRadius: '2px', display: 'inline-block' },
    button: { width: '100%', padding: '10px', marginBottom: '8px', backgroundColor: '#2d2d2d', color: '#fff', border: '1px solid #444', borderRadius: '4px', cursor: 'pointer', fontSize: '13px', fontWeight: '500', transition: 'all 0.2s' },
    runButton: { backgroundColor: '#1a4a1a', borderColor: '#2a6a2a' },
    resetButton: { backgroundColor: '#1a1a1a', borderColor: '#555' },
    fixedPointItem: { marginBottom: '6px', fontSize: '12px', padding: '6px', background: '#0f0f0f', borderRadius: '4px' },
    info: { marginTop: '16px', padding: '12px', backgroundColor: '#0f0f0f', borderRadius: '4px', fontSize: '11px', lineHeight: '1.6', color: '#888' },
    infoHeader: { fontSize: '12px', fontWeight: '600', color: '#4CAF50', marginBottom: '8px' },
    viewport: { flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' },
    canvas: { display: 'block' }
};

export default SetValuedViz;