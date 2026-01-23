import React, { useState, useEffect, useRef, useCallback } from 'react';
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

        const material = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            depthTest: false
        });
        const sprite = new THREE.Sprite(material);
        sprite.position.copy(position);
        sprite.scale.set(fontSize * 2, fontSize, 1);
        sprite.userData.isGrid = true;

        return sprite;
    };

    for (let i = 0; i <= gridDivisions; i++) {
        const x = xMin + i * xStep;
        if (Math.abs(x) > 0.01) {
            const label = createTextSprite(
                x.toFixed(1),
                new THREE.Vector3(x, yMin - 0.15, 0),
                0.12
            );
            gridGroup.add(label);
        }
    }

    for (let i = 0; i <= gridDivisions; i++) {
        const y = yMin + i * yStep;
        if (Math.abs(y) > 0.01) {
            const label = createTextSprite(
                y.toFixed(1),
                new THREE.Vector3(xMin - 0.2, y, 0),
                0.12
            );
            gridGroup.add(label);
        }
    }

    const xLabel = createTextSprite('x', new THREE.Vector3(xMax + 0.2, 0, 0), 0.18);
    const yLabel = createTextSprite('y', new THREE.Vector3(0, yMax + 0.15, 0), 0.18);
    gridGroup.add(xLabel);
    gridGroup.add(yLabel);

    const originLabel = createTextSprite('0', new THREE.Vector3(-0.12, -0.12, 0), 0.1);
    gridGroup.add(originLabel);

    scene.add(gridGroup);
    return gridGroup;
};

const ORBIT_COLORS = {
    period1: { stable: '#e74c3c', unstable: '#c0392b', saddle: '#e67e22' },
    period2: { stable: '#27ae60', unstable: '#229954', saddle: '#16a085' },
    period3: { stable: '#3498db', unstable: '#2980b9', saddle: '#2c3e50' },
    period4: { stable: '#9b59b6', unstable: '#8e44ad', saddle: '#71368a' },
    period5: { stable: '#f39c12', unstable: '#d68910', saddle: '#ca6f1e' },
    period6plus: { stable: '#1abc9c', unstable: '#16a085', saddle: '#138d75' },
    trajectory: '#00ffff'
};

const HenonPeriodicViz = () => {
    const canvasRef = useRef(null);
    const rendererRef = useRef(null);
    const sceneRef = useRef(null);
    const cameraRef = useRef(null);
    const animationFrameRef = useRef(null);
    const batchAnimationRef = useRef(null);

    const [params, setParams] = useState({
        a: 1.4,
        b: 0.3,
        startX: 0.1,
        startY: 0.1,
        maxIterations: 1000,
        maxPeriod: 5
    });

    const [state, setState] = useState({
        orbits: [],
        trajectoryPoints: [],
        currentPoint: { x: 0.1, y: 0.1 },
        iteration: 0,
        isRunning: false,
        isReady: false,
        showOrbits: false,
        showTrail: true,
        hasStarted: false
    });

    const [filters, setFilters] = useState({
        period1: true,
        period2: true,
        period3: true,
        period4: true,
        period5: true,
        period6plus: false
    });

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
            -frustumWidth / 2,
            frustumWidth / 2,
            frustumHeight / 2,
            -frustumHeight / 2,
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

        createCoordinateSystem(scene);

        const handleResize = () => {
            const aspect = (window.innerWidth * 0.75) / window.innerHeight;
            const newFrustumWidth = frustumHeight * aspect;
            camera.left = -newFrustumWidth / 2;
            camera.right = newFrustumWidth / 2;
            camera.top = frustumHeight / 2;
            camera.bottom = -frustumHeight / 2;
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
            if (batchAnimationRef.current) {
                cancelAnimationFrame(batchAnimationRef.current);
            }
            renderer.dispose();
        };
    }, []);

    useEffect(() => {
        let cancelled = false;

        setState(prev => ({
            ...prev,
            isReady: false,
            orbits: [],
            trajectoryPoints: [],
            currentPoint: { x: params.startX, y: params.startY },
            iteration: 0,
            hasStarted: false,
            showOrbits: false
        }));

        const initSystem = async () => {
            try {
                const wasm = await import('../pkg/henon_periodic_orbits.js');
                await wasm.default();

                if (cancelled) return;

                console.log(`Computing periodic orbits with max_period=${params.maxPeriod}`);
                const system = new wasm.HenonSystemWasm(
                    params.a,
                    params.b,
                    params.maxPeriod
                );

                if (cancelled) {
                    system.free();
                    return;
                }

                const orbits = system.getPeriodicOrbits();
                console.log(`Found ${orbits.length} periodic orbits`);

                system.free();

                setState(prev => ({
                    ...prev,
                    orbits,
                    isReady: true,
                    currentPoint: { x: params.startX, y: params.startY },
                    trajectoryPoints: [],
                    iteration: 0,
                    hasStarted: false,
                    showOrbits: false
                }));
            } catch (err) {
                console.error('Failed to initialize WASM:', err);
                setState(prev => ({ ...prev, isReady: true, orbits: [] }));
            }
        };

        initSystem();

        return () => {
            cancelled = true;
        };
    }, [params.a, params.b, params.maxPeriod, params.startX, params.startY]);

    useEffect(() => {
        if (!sceneRef.current) return;

        const scene = sceneRef.current;

        const objectsToRemove = [];
        scene.traverse(child => {
            if (child.userData.type === 'trajectory' || child.userData.type === 'orbit') {
                objectsToRemove.push(child);
            }
        });
        objectsToRemove.forEach(obj => {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
            scene.remove(obj);
        });

        if (state.showOrbits && state.orbits.length > 0) {
            const visibleOrbits = state.orbits.filter(isOrbitVisible);

            visibleOrbits.forEach(orbit => {
                orbit.points.forEach((pt) => {
                    const geom = new THREE.SphereGeometry(0.03, 12, 12);
                    const color = getOrbitColor(orbit);
                    const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(color) });
                    const sphere = new THREE.Mesh(geom, mat);
                    sphere.position.set(pt[0], pt[1], 0.1);
                    sphere.userData.type = 'orbit';
                    scene.add(sphere);
                });

                if (orbit.points.length > 1) {
                    const lineGeom = new THREE.BufferGeometry();
                    const positions = new Float32Array(orbit.points.length * 3);
                    orbit.points.forEach((pt, i) => {
                        positions[i * 3] = pt[0];
                        positions[i * 3 + 1] = pt[1];
                        positions[i * 3 + 2] = 0.1;
                    });
                    lineGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    const lineMat = new THREE.LineBasicMaterial({
                        color: new THREE.Color(getOrbitColor(orbit)),
                        opacity: 0.5,
                        transparent: true
                    });
                    const line = new THREE.LineLoop(lineGeom, lineMat);
                    line.userData.type = 'orbit';
                    scene.add(line);
                }
            });
        }

        if (state.showTrail && state.trajectoryPoints.length > 0) {
            const points = state.trajectoryPoints;

            points.forEach((point, idx) => {
                const normalizedIdx = idx / points.length;
                const baseSize = 0.012;
                const size = baseSize * (0.3 + 0.7 * normalizedIdx);

                const geom = new THREE.SphereGeometry(size, 6, 6);
                const mat = new THREE.MeshBasicMaterial({
                    color: new THREE.Color(ORBIT_COLORS.trajectory),
                    opacity: 0.2 + 0.6 * normalizedIdx,
                    transparent: true
                });
                const sphere = new THREE.Mesh(geom, mat);
                sphere.position.set(point.x, point.y, 0.05);
                sphere.userData.type = 'trajectory';
                scene.add(sphere);
            });
        }

        if (state.hasStarted && state.currentPoint) {
            const geom = new THREE.SphereGeometry(0.04, 16, 16);
            const mat = new THREE.MeshBasicMaterial({
                color: new THREE.Color('#ffffff')
            });
            const sphere = new THREE.Mesh(geom, mat);
            sphere.position.set(state.currentPoint.x, state.currentPoint.y, 0.2);
            sphere.userData.type = 'trajectory';
            scene.add(sphere);
        }
    }, [state.currentPoint, state.trajectoryPoints, state.orbits, state.showOrbits, state.showTrail, state.iteration, state.hasStarted, filters]);

    const stepForward = useCallback(() => {
        if (!state.isReady || state.isRunning) return;

        const { x, y } = state.currentPoint;

        if (!isFinite(x) || !isFinite(y) || Math.abs(x) > 10 || Math.abs(y) > 10) {
            console.warn('Point diverged, resetting');
            setState(prev => ({
                ...prev,
                currentPoint: { x: params.startX, y: params.startY },
                trajectoryPoints: [],
                iteration: 0,
                hasStarted: false
            }));
            return;
        }

        const nextPoint = henonMap(x, y, params.a, params.b);

        setState(prev => ({
            ...prev,
            currentPoint: nextPoint,
            trajectoryPoints: [...prev.trajectoryPoints, { x, y }],
            iteration: prev.iteration + 1,
            hasStarted: true
        }));
    }, [state.isReady, state.isRunning, state.currentPoint, params.a, params.b, params.startX, params.startY]);

    const runToConvergence = useCallback(() => {
        if (!state.isReady || state.isRunning) return;

        setState(prev => ({ ...prev, isRunning: true }));

        let currentX = state.currentPoint.x;
        let currentY = state.currentPoint.y;
        let iteration = state.iteration;
        const newPoints = [...state.trajectoryPoints];
        const batchSize = 5; 

        const animateStep = () => {
            for (let i = 0; i < batchSize && iteration < params.maxIterations; i++) {
                if (!isFinite(currentX) || !isFinite(currentY) ||
                    Math.abs(currentX) > 10 || Math.abs(currentY) > 10) {
                    console.warn('Point diverged at iteration', iteration);
                    setState(prev => ({
                        ...prev,
                        isRunning: false,
                        showOrbits: true,
                        hasStarted: true,
                        trajectoryPoints: newPoints,
                        currentPoint: { x: currentX, y: currentY },
                        iteration
                    }));
                    return;
                }

                newPoints.push({ x: currentX, y: currentY });
                const next = henonMap(currentX, currentY, params.a, params.b);
                currentX = next.x;
                currentY = next.y;
                iteration++;
            }

            setState(prev => ({
                ...prev,
                currentPoint: { x: currentX, y: currentY },
                trajectoryPoints: [...newPoints],
                iteration,
                hasStarted: true
            }));

            if (iteration < params.maxIterations) {
                batchAnimationRef.current = requestAnimationFrame(animateStep);
            } else {
                setState(prev => ({
                    ...prev,
                    isRunning: false,
                    showOrbits: true
                }));
            }
        };

        batchAnimationRef.current = requestAnimationFrame(animateStep);
    }, [state.isReady, state.isRunning, state.currentPoint, state.iteration, state.trajectoryPoints, params.a, params.b, params.maxIterations]);

    const reset = useCallback(() => {
        if (batchAnimationRef.current) {
            cancelAnimationFrame(batchAnimationRef.current);
        }

        setState(prev => ({
            ...prev,
            currentPoint: { x: params.startX, y: params.startY },
            trajectoryPoints: [],
            iteration: 0,
            isRunning: false,
            hasStarted: false,
            showOrbits: false
        }));
    }, [params.startX, params.startY]);

    const getOrbitColor = (orbit) => {
        const { period, stability } = orbit;

        let colorSet;
        if (period === 1) colorSet = ORBIT_COLORS.period1;
        else if (period === 2) colorSet = ORBIT_COLORS.period2;
        else if (period === 3) colorSet = ORBIT_COLORS.period3;
        else if (period === 4) colorSet = ORBIT_COLORS.period4;
        else if (period === 5) colorSet = ORBIT_COLORS.period5;
        else colorSet = ORBIT_COLORS.period6plus;

        return colorSet[stability.toLowerCase()] || colorSet.stable;
    };

    const isOrbitVisible = (orbit) => {
        if (orbit.period === 1) return filters.period1;
        if (orbit.period === 2) return filters.period2;
        if (orbit.period === 3) return filters.period3;
        if (orbit.period === 4) return filters.period4;
        if (orbit.period === 5) return filters.period5;
        if (orbit.period >= 6) return filters.period6plus;
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
                            disabled={state.isRunning}
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
                            disabled={state.isRunning}
                        />
                    </label>

                    <label style={styles.label}>
                        Max Period = {params.maxPeriod}
                        <input
                            type="range"
                            min="2"
                            max="10"
                            step="1"
                            value={params.maxPeriod}
                            onChange={(e) => setParams({...params, maxPeriod: parseInt(e.target.value)})}
                            style={styles.slider}
                            disabled={state.isRunning}
                        />
                    </label>

                    <label style={styles.label}>
                        Max Iterations = {params.maxIterations}
                        <input
                            type="range"
                            min="100"
                            max="5000"
                            step="100"
                            value={params.maxIterations}
                            onChange={(e) => setParams({...params, maxIterations: parseInt(e.target.value)})}
                            style={styles.slider}
                            disabled={state.isRunning}
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
                            checked={filters.period4}
                            onChange={(e) => setFilters({...filters, period4: e.target.checked})}
                        />
                        <span style={{...styles.colorBox, backgroundColor: ORBIT_COLORS.period4.stable}} />
                        Period-4 ({state.orbits.filter(o => o.period === 4).length})
                    </label>

                    <label style={styles.checkboxLabel}>
                        <input
                            type="checkbox"
                            checked={filters.period5}
                            onChange={(e) => setFilters({...filters, period5: e.target.checked})}
                        />
                        <span style={{...styles.colorBox, backgroundColor: ORBIT_COLORS.period5.stable}} />
                        Period-5 ({state.orbits.filter(o => o.period === 5).length})
                    </label>

                    <label style={styles.checkboxLabel}>
                        <input
                            type="checkbox"
                            checked={state.showOrbits}
                            onChange={(e) => setState({...state, showOrbits: e.target.checked})}
                        />
                        Show orbit markers
                    </label>

                    <label style={styles.checkboxLabel}>
                        <input
                            type="checkbox"
                            checked={state.showTrail}
                            onChange={(e) => setState({...state, showTrail: e.target.checked})}
                        />
                        Show trajectory trail
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
                        onClick={runToConvergence}
                        disabled={!state.isReady || state.isRunning}
                        style={{...styles.button, ...styles.runButton}}
                    >
                        {state.isRunning ? 'Running...' : 'Run to Max Iterations'}
                    </button>

                    <button
                        onClick={reset}
                        disabled={state.isRunning}
                        style={{...styles.button, ...styles.resetButton}}
                    >
                        Reset
                    </button>
                </div>

                <div style={styles.info}>
                    <div style={styles.infoHeader}>Henon Map: x' = 1 - ax^2 + y, y' = bx</div>
                    <div>Status: {state.isReady ? (state.isRunning ? 'Running...' : 'Ready') : 'Loading...'}</div>
                    <div>Iteration: {state.iteration} / {params.maxIterations}</div>
                    {state.hasStarted && (
                        <div>Position: ({state.currentPoint.x.toFixed(4)}, {state.currentPoint.y.toFixed(4)})</div>
                    )}
                    <div>Orbits found: {state.orbits.length}</div>
                    <div>Trail points: {state.trajectoryPoints.length}</div>
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
    runButton: {
        backgroundColor: '#1a4a1a',
        borderColor: '#2a6a2a'
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
    infoHeader: {
        fontSize: '13px',
        fontWeight: '600',
        color: '#4CAF50',
        marginBottom: '8px'
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
