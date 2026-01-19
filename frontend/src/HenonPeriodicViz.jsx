import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';

const ORBIT_COLORS = {
    period1: { stable: '#e74c3c', unstable: '#c0392b', saddle: '#e67e22' },
    period2: { stable: '#27ae60', unstable: '#229954', saddle: '#16a085' },
    period3: { stable: '#3498db', unstable: '#2980b9', saddle: '#2c3e50' },
    period4plus: { stable: '#f39c12', unstable: '#d68910', saddle: '#ca6f1e' },
    regular: '#e91e63'
};

const HenonPeriodicViz = () => {
    const canvasRef = useRef(null);
    const rendererRef = useRef(null);
    const sceneRef = useRef(null);
    const cameraRef = useRef(null);
    const systemRef = useRef(null);
    const isProcessingRef = useRef(false);
    const animationFrameRef = useRef(null);

    const [params, setParams] = useState({
        a: 1.4,
        b: 0.3,
        epsilon: 0.05,
        centerX: 0.0,
        centerY: 0.0,
        numPoints: 128,
        maxIterations: 30,
        maxPeriod: 2
    });

    const [state, setState] = useState({
        orbits: [],
        boundary: [],
        iteration: 0,
        totalIterations: 0,
        isRunning: false,
        isReady: false,
        showOrbits: true
    });

    const [filters, setFilters] = useState({
        period1: true,
        period2: true,
        period3: true,
        period4plus: false
    });

    useEffect(() => {
        if (!canvasRef.current) return;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a0a);
        sceneRef.current = scene;

        const aspect = window.innerWidth / window.innerHeight;
        const frustumSize = 3;
        const camera = new THREE.OrthographicCamera(
            -frustumSize * aspect / 2,
            frustumSize * aspect / 2,
            frustumSize / 2,
            -frustumSize / 2,
            0.1,
            1000
        );
        camera.position.z = 5;
        cameraRef.current = camera;

        const renderer = new THREE.WebGLRenderer({
            canvas: canvasRef.current,
            antialias: true,
            alpha: true
        });
        renderer.setSize(window.innerWidth * 0.75, window.innerHeight); 
        renderer.setPixelRatio(window.devicePixelRatio);
        rendererRef.current = renderer;

        const axesHelper = new THREE.AxesHelper(2);
        scene.add(axesHelper);

        const gridHelper = new THREE.GridHelper(4, 20, 0x444444, 0x222222);
        gridHelper.rotation.x = Math.PI / 2;
        scene.add(gridHelper);

        const handleResize = () => {
            const aspect = window.innerWidth / window.innerHeight;
            camera.left = -frustumSize * aspect / 2;
            camera.right = frustumSize * aspect / 2;
            camera.top = frustumSize / 2;
            camera.bottom = -frustumSize / 2;
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
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
            renderer.dispose();
        };
    }, []);

    useEffect(() => {
        let cancelled = false;

        const initSystem = async () => {
            try {
                const wasm = await import('../pkg/henon_periodic_orbits.js');
                await wasm.default();

                if (cancelled) return;

                const system = new wasm.HenonSystemWasm(
                    params.a,
                    params.b,
                    params.epsilon,
                    params.maxPeriod
                );

                const orbits = system.getPeriodicOrbits();
                systemRef.current = system;

                console.log(`Initialized system with ${orbits.length} periodic orbits`);

                setState(prev => ({ 
                    ...prev, 
                    orbits, 
                    isReady: true, 
                    boundary: [], 
                    iteration: 0, 
                    totalIterations: 0 
                }));
            } catch (err) {
                console.error('Failed to initialize WASM:', err);
            }
        };

        initSystem();

        return () => {
            cancelled = true;
            if (systemRef.current) {
                try {
                    systemRef.current.free();
                } catch (e) {
                    console.warn('Cleanup error:', e);
                }
                systemRef.current = null;
            }
        };
    }, [params.a, params.b, params.epsilon, params.maxPeriod]);

    useEffect(() => {
        if (!sceneRef.current) return;

        const scene = sceneRef.current;

        const objectsToRemove = [];
        scene.children.forEach(child => {
            if (child.userData.type === 'boundary' || child.userData.type === 'orbit') {
                objectsToRemove.push(child);
            }
        });
        objectsToRemove.forEach(obj => {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
            scene.remove(obj);
        });

        if (state.boundary.length > 0) {
            console.log(`Rendering ${state.boundary.length} boundary points`);

            const boundaryGeom = new THREE.BufferGeometry();
            const positions = new Float32Array(state.boundary.length * 3);
            const colors = new Float32Array(state.boundary.length * 3);

            state.boundary.forEach((point, i) => {
                positions[i * 3] = point.x;
                positions[i * 3 + 1] = point.y;
                positions[i * 3 + 2] = 0;

                let color;
                if (point.classification === 'periodic' && point.period) {
                    const orbit = state.orbits.find(o => o.period === point.period);
                    if (orbit) {
                        color = new THREE.Color(getOrbitColor(orbit));
                    } else {
                        color = new THREE.Color(getOrbitColor({ 
                            period: point.period, 
                            stability: point.stability || 'stable' 
                        }));
                    }
                } else {
                    color = new THREE.Color(ORBIT_COLORS.regular);
                }

                colors[i * 3] = color.r;
                colors[i * 3 + 1] = color.g;
                colors[i * 3 + 2] = color.b;
            });

            boundaryGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            boundaryGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

            const boundaryMat = new THREE.PointsMaterial({
                size: 0.03,
                vertexColors: true,
                sizeAttenuation: false
            });

            const boundaryPoints = new THREE.Points(boundaryGeom, boundaryMat);
            boundaryPoints.userData.type = 'boundary';
            scene.add(boundaryPoints);

            const lineGeom = new THREE.BufferGeometry();
            lineGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            
            const lineMat = new THREE.LineBasicMaterial({
                color: ORBIT_COLORS.regular,
                opacity: 0.5,
                transparent: true
            });

            const boundaryLine = new THREE.LineLoop(lineGeom, lineMat);
            boundaryLine.userData.type = 'boundary';
            scene.add(boundaryLine);
        }

        if (state.showOrbits && state.orbits.length > 0) {
            console.log(`Rendering ${state.orbits.length} periodic orbits`);

            state.orbits
                .filter(isOrbitVisible)
                .forEach(orbit => {
                    const orbitGeom = new THREE.BufferGeometry();
                    const positions = new Float32Array(orbit.points.length * 3);

                    orbit.points.forEach((pt, i) => {
                        positions[i * 3] = pt[0];
                        positions[i * 3 + 1] = pt[1];
                        positions[i * 3 + 2] = 0.1;
                    });

                    orbitGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                    const color = getOrbitColor(orbit);
                    const orbitMat = new THREE.PointsMaterial({
                        size: 0.06,
                        color: new THREE.Color(color),
                        sizeAttenuation: false
                    });

                    const orbitPoints = new THREE.Points(orbitGeom, orbitMat);
                    orbitPoints.userData.type = 'orbit';
                    scene.add(orbitPoints);

                    const lineGeom = new THREE.BufferGeometry();
                    lineGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));

                    const lineMat = new THREE.LineBasicMaterial({
                        color: new THREE.Color(color),
                        opacity: 0.6,
                        transparent: true
                    });

                    const orbitLine = new THREE.LineLoop(lineGeom, lineMat);
                    orbitLine.userData.type = 'orbit';
                    scene.add(orbitLine);
                });
        }
    }, [state.boundary, state.orbits, state.showOrbits, filters]);

    const stepForward = useCallback(() => {
        const system = systemRef.current;
        if (!system || !state.isReady || isProcessingRef.current) return;

        isProcessingRef.current = true;

        try {
            if (state.totalIterations === 0) {
                console.log("Initializing boundary tracking...");
                system.trackBoundary(
                    params.centerX,
                    params.centerY,
                    params.numPoints,
                    params.maxIterations,
                    1e-4
                );
                system.reset();
                const totalIter = system.getTotalIterations();
                const boundary = system.getCurrentBoundary();
                
                console.log(`Initialized with ${totalIter} total iterations`);
                console.log(`Initial boundary has ${boundary ? boundary.length : 0} points`);

                setState(prev => ({
                    ...prev,
                    boundary: boundary || [],
                    iteration: 0,
                    totalIterations: totalIter
                }));
            } else {
                const canStep = system.step();
                if (canStep) {
                    const boundary = system.getCurrentBoundary();
                    const currentIter = system.getCurrentIteration();
                    
                    console.log(`Step ${currentIter}: ${boundary ? boundary.length : 0} points`);

                    setState(prev => ({
                        ...prev,
                        boundary: boundary || [],
                        iteration: currentIter
                    }));
                } else {
                    console.log("Cannot step forward - reached end of iterations");
                }
            }
        } catch (err) {
            console.error('Step failed:', err);
        } finally {
            isProcessingRef.current = false;
        }
    }, [state.isReady, state.totalIterations, params]);

    const runFull = useCallback(() => {
        const system = systemRef.current;
        if (!system || !state.isReady || isProcessingRef.current) return;

        isProcessingRef.current = true;
        setState(prev => ({ ...prev, isRunning: true }));

        setTimeout(() => {
            try {
                console.log("Running full simulation...");
                system.trackBoundary(
                    params.centerX,
                    params.centerY,
                    params.numPoints,
                    params.maxIterations,
                    1e-4
                );

                const totalIter = system.getTotalIterations();
                console.log(`Total iterations computed: ${totalIter}`);

                for (let i = 0; i < totalIter - 1; i++) {
                    system.step();
                }

                const boundary = system.getCurrentBoundary();
                const currentIter = system.getCurrentIteration();

                console.log(`Final state: iteration ${currentIter}, ${boundary ? boundary.length : 0} points`);

                setState(prev => ({
                    ...prev,
                    boundary: boundary || [],
                    iteration: currentIter,
                    totalIterations: totalIter,
                    isRunning: false
                }));
            } catch (err) {
                console.error('Run failed:', err);
                setState(prev => ({ ...prev, isRunning: false }));
            } finally {
                isProcessingRef.current = false;
            }
        }, 10);
    }, [state.isReady, params]);

    const reset = useCallback(() => {
        const system = systemRef.current;
        if (!system || isProcessingRef.current) return;

        try {
            system.reset();
            const boundary = system.getCurrentBoundary();
            setState(prev => ({
                ...prev,
                boundary: boundary || [],
                iteration: 0
            }));
            console.log("Reset to initial state");
        } catch (err) {
            console.error('Reset failed:', err);
        }
    }, []);

    const getOrbitColor = (orbit) => {
        const { period, stability } = orbit;

        let colorSet;
        if (period === 1) colorSet = ORBIT_COLORS.period1;
        else if (period === 2) colorSet = ORBIT_COLORS.period2;
        else if (period === 3) colorSet = ORBIT_COLORS.period3;
        else colorSet = ORBIT_COLORS.period4plus;

        return colorSet[stability.toLowerCase()] || colorSet.stable;
    };

    const isOrbitVisible = (orbit) => {
        if (orbit.period === 1) return filters.period1;
        if (orbit.period === 2) return filters.period2;
        if (orbit.period === 3) return filters.period3;
        if (orbit.period >= 4) return filters.period4plus;
        return false;
    };

    return (
        <div style={styles.container}>
            <div style={styles.sidebar}>
                <div style={styles.section}>
                    <h3 style={styles.sectionTitle}>System Parameters</h3>

                    <label style={styles.label}>
                        a = {params.a.toFixed(2)}
                        <input
                            type="range"
                            min="0.5"
                            max="2.0"
                            step="0.01"
                            value={params.a}
                            onChange={(e) => setParams({...params, a: parseFloat(e.target.value)})}
                            style={styles.slider}
                        />
                    </label>

                    <label style={styles.label}>
                        b = {params.b.toFixed(2)}
                        <input
                            type="range"
                            min="0.1"
                            max="0.5"
                            step="0.01"
                            value={params.b}
                            onChange={(e) => setParams({...params, b: parseFloat(e.target.value)})}
                            style={styles.slider}
                        />
                    </label>

                    <label style={styles.label}>
                        ε = {params.epsilon.toFixed(3)}
                        <input
                            type="range"
                            min="0.01"
                            max="0.2"
                            step="0.005"
                            value={params.epsilon}
                            onChange={(e) => setParams({...params, epsilon: parseFloat(e.target.value)})}
                            style={styles.slider}
                        />
                    </label>
                </div>

                <div style={styles.section}>
                    <h3 style={styles.sectionTitle}>Periodic Orbits</h3>

                    <label style={styles.checkboxLabel}>
                        <input
                            type="checkbox"
                            checked={filters.period1}
                            onChange={(e) => setFilters({...filters, period1: e.target.checked})}
                        />
                        <span style={{...styles.colorBox, backgroundColor: ORBIT_COLORS.period1.stable}} />
                        Period-1 ({state.orbits.filter(o => o.period === 1).length})
                    </label>

                    <label style={styles.checkboxLabel}>
                        <input
                            type="checkbox"
                            checked={filters.period2}
                            onChange={(e) => setFilters({...filters, period2: e.target.checked})}
                        />
                        <span style={{...styles.colorBox, backgroundColor: ORBIT_COLORS.period2.stable}} />
                        Period-2 ({state.orbits.filter(o => o.period === 2).length})
                    </label>

                    <label style={styles.checkboxLabel}>
                        <input
                            type="checkbox"
                            checked={filters.period3}
                            onChange={(e) => setFilters({...filters, period3: e.target.checked})}
                        />
                        <span style={{...styles.colorBox, backgroundColor: ORBIT_COLORS.period3.stable}} />
                        Period-3 ({state.orbits.filter(o => o.period === 3).length})
                    </label>

                    <label style={styles.checkboxLabel}>
                        <input
                            type="checkbox"
                            checked={state.showOrbits}
                            onChange={(e) => setState({...state, showOrbits: e.target.checked})}
                        />
                        Show orbit markers
                    </label>
                </div>

                <div style={styles.section}>
                    <h3 style={styles.sectionTitle}>Controls</h3>

                    <button
                        onClick={stepForward}
                        disabled={!state.isReady || state.isRunning}
                        style={styles.button}
                    >
                        Step Forward
                    </button>

                    <button
                        onClick={runFull}
                        disabled={!state.isReady || state.isRunning}
                        style={styles.button}
                    >
                        {state.isRunning ? 'Computing...' : 'Run to Convergence'}
                    </button>

                    <button
                        onClick={reset}
                        disabled={!state.isReady || state.isRunning}
                        style={{...styles.button, ...styles.resetButton}}
                    >
                        Reset
                    </button>
                </div>

                <div style={styles.info}>
                    <div>Status: {state.isReady ? 'Ready' : 'Loading...'}</div>
                    <div>Iteration: {state.iteration} / {state.totalIterations}</div>
                    <div>Boundary points: {state.boundary.length}</div>
                    <div>Orbits found: {state.orbits.length}</div>
                </div>
            </div>

            <div style={styles.viewport}>
                <canvas
                    ref={canvasRef}
                    style={styles.canvas}
                />
            </div>
        </div>
    );
};

const styles = {
    container: {
        display: 'flex',
        height: '100vh',
        width: '100vw',
        backgroundColor: '#0a0a0a',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        color: '#e0e0e0',
        overflow: 'hidden'
    },
    sidebar: {
        width: '320px',
        minWidth: '320px',
        padding: '20px',
        backgroundColor: '#1a1a1a',
        borderRight: '1px solid #333',
        overflowY: 'auto'
    },
    section: {
        marginBottom: '32px'
    },
    sectionTitle: {
        fontSize: '14px',
        fontWeight: '600',
        marginBottom: '16px',
        color: '#fff',
        textTransform: 'uppercase',
        letterSpacing: '0.5px'
    },
    label: {
        display: 'flex',
        flexDirection: 'column',
        marginBottom: '16px',
        fontSize: '13px',
        color: '#b0b0b0'
    },
    slider: {
        marginTop: '8px',
        width: '100%'
    },
    checkboxLabel: {
        display: 'flex',
        alignItems: 'center',
        marginBottom: '12px',
        fontSize: '13px',
        cursor: 'pointer',
        gap: '8px'
    },
    colorBox: {
        width: '16px',
        height: '16px',
        borderRadius: '2px',
        display: 'inline-block'
    },
    button: {
        width: '100%',
        padding: '12px',
        marginBottom: '8px',
        backgroundColor: '#2d2d2d',
        color: '#fff',
        border: '1px solid #444',
        borderRadius: '4px',
        cursor: 'pointer',
        fontSize: '13px',
        fontWeight: '500',
        transition: 'all 0.2s'
    },
    resetButton: {
        backgroundColor: '#1a1a1a',
        borderColor: '#555'
    },
    info: {
        marginTop: '24px',
        padding: '16px',
        backgroundColor: '#0f0f0f',
        borderRadius: '4px',
        fontSize: '12px',
        lineHeight: '1.8',
        color: '#888'
    },
    viewport: {
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
        overflow: 'hidden'
    },
    canvas: {
        maxWidth: '100%',
        maxHeight: '100%',
        borderRadius: '4px',
        display: 'block'
    }
};

export default HenonPeriodicViz;