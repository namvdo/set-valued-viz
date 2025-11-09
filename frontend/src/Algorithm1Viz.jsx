import React, { useRef, useEffect, useMemo } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Line } from '@react-three/drei';
import * as THREE from 'three';

function AutoCamera({ bounds, enabled, resetTrigger = 0 }) {
  const { camera, size } = useThree();
  
  useEffect(() => {
    if (!enabled || !bounds) return;
    
    const { minX, maxX, minY, maxY } = bounds;
    const rangeX = Math.max(maxX - minX, 0.1);
    const rangeY = Math.max(maxY - minY, 0.1);
    
    // Calculate zoom to fit
    const zoomX = size.width / rangeX;
    const zoomY = size.height / rangeY;
    const zoom = Math.min(zoomX, zoomY) * 0.7;
    
    camera.zoom = zoom;
    camera.updateProjectionMatrix();
  }, [bounds, camera, size, resetTrigger]);
  
  return null;
}

function DetailedStepVisualization({ stepData, visualizationMode, epsilon }) {
  if (!stepData) return null;

  const { mapped_points, projected_points } = stepData;

  const mappedPoints = useMemo(() => {
    if (!mapped_points?.length) return [];
    const points = [];
    for (let i = 0; i < mapped_points.length; i += 2) {
      points.push(new THREE.Vector3(mapped_points[i], mapped_points[i + 1], 0));
    }
    return points;
  }, [mapped_points]);

  const projectedPoints = useMemo(() => {
    if (!projected_points?.length) return [];
    const points = [];
    for (let i = 0; i < projected_points.length; i += 2) {
      points.push(new THREE.Vector3(projected_points[i], projected_points[i + 1], 0));
    }
    return points;
  }, [projected_points]);

  const projectedBoundary = useMemo(() => {
    if (!projectedPoints.length) return [];
    return [...projectedPoints, projectedPoints[0]];
  }, [projectedPoints]);

  const showMapped = visualizationMode === 'mapped' || visualizationMode === 'all';
  const showCircles = visualizationMode === 'circles' || visualizationMode === 'all';
  const showProjected = visualizationMode === 'projected' || visualizationMode === 'all';
  const showNormals = visualizationMode === 'all';

  return (
    <>
      {/* Mapped points */}
      {showMapped && mappedPoints.map((point, idx) => (
        <mesh key={`m-${idx}`} position={point}>
          <sphereGeometry args={[0.008, 16, 16]} />
          <meshBasicMaterial color="#00ffff" />
        </mesh>
      ))}

      {/* Noise circles */}
      {showCircles && mappedPoints.map((point, idx) => (
        <group key={`c-${idx}`}>
          <mesh position={point}>
            <circleGeometry args={[epsilon, 64]} />
            <meshBasicMaterial color="#ffaa00" transparent opacity={0.15} side={THREE.DoubleSide} />
          </mesh>
          <mesh position={point}>
            <ringGeometry args={[epsilon * 0.98, epsilon, 64]} />
            <meshBasicMaterial color="#ffaa00" transparent opacity={0.6} side={THREE.DoubleSide} />
          </mesh>
        </group>
      ))}

      {/* Projected boundary */}
      {showProjected && projectedBoundary.length > 0 && (
        <>
          <Line points={projectedBoundary} color="#ff00ff" lineWidth={3} />
          {projectedPoints.map((point, idx) => (
            <mesh key={`p-${idx}`} position={point}>
              <sphereGeometry args={[0.010, 16, 16]} />
              <meshBasicMaterial color="#ff00ff" />
            </mesh>
          ))}
        </>
      )}

      {/* Normal vectors */}
      {showNormals && mappedPoints.map((mPoint, idx) => {
        if (idx >= projectedPoints.length) return null;
        return (
          <Line key={`n-${idx}`} points={[mPoint, projectedPoints[idx]]} color="#00ff00" lineWidth={1.5} />
        );
      })}
    </>
  );
}

function BoundaryEvolution({ boundaryHistory, epsilon }) {
  const geometries = useMemo(() => {
    if (!boundaryHistory?.length) return [];

    return boundaryHistory.map((boundary, iter) => {
      if (!boundary?.length) return null;

      const points = [];
      for (let i = 0; i < boundary.length; i += 2) {
        points.push(new THREE.Vector3(boundary[i], boundary[i + 1], 0));
      }
      if (points.length > 0) points.push(points[0].clone());

      const intensity = iter / Math.max(1, boundaryHistory.length - 1);
      const color = new THREE.Color(
        Math.pow(intensity, 0.7),
        1 - Math.abs(2 * intensity - 1),
        Math.pow(1 - intensity, 0.7)
      );
      const alpha = 0.2 + 0.6 * intensity;
      const isFinal = iter === boundaryHistory.length - 1;

      return { points, color, alpha, isFinal };
    }).filter(Boolean);
  }, [boundaryHistory]);

  return (
    <>
      {geometries.map((geo, idx) => (
        <group key={idx}>
          <Line points={geo.points} color={geo.color} lineWidth={geo.isFinal ? 3 : 1.5} transparent opacity={geo.alpha} />
          {geo.points.slice(0, -1).map((point, pidx) => (
            <mesh key={pidx} position={point}>
              <sphereGeometry args={[geo.isFinal ? 0.015 : 0.008, 8, 8]} />
              <meshBasicMaterial color={geo.isFinal ? '#ffffff' : geo.color} transparent opacity={geo.alpha} />
            </mesh>
          ))}
        </group>
      ))}
    </>
  );
}

function Scene({ boundaryHistory, stepData, visualizationMode, epsilon, showDetailedViz }) {
  const bounds = useMemo(() => {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

    const processPoints = (points) => {
      for (let i = 0; i < points.length; i += 2) {
        const x = points[i], y = points[i + 1];
        if (isFinite(x) && isFinite(y)) {
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
        }
      }
    };

    if (showDetailedViz && stepData) {
      const { mapped_points, projected_points } = stepData;
      if (mapped_points) processPoints(mapped_points);
      if (projected_points) processPoints(projected_points);
      // Add padding for circles
      const pad = epsilon * 1.5;
      minX -= pad; maxX += pad; minY -= pad; maxY += pad;
    } else if (boundaryHistory?.length) {
      boundaryHistory.forEach(processPoints);
    }

    if (!isFinite(minX)) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };

    const padding = 0.2;
    const rangeX = maxX - minX;
    const rangeY = maxY - minY;

    return {
      minX: minX - rangeX * padding,
      maxX: maxX + rangeX * padding,
      minY: minY - rangeY * padding,
      maxY: maxY + rangeY * padding
    };
  }, [boundaryHistory, stepData, showDetailedViz, epsilon]);

  return (
    <>
      <color attach="background" args={['#000000']} />
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />

      <AutoCamera bounds={bounds} enabled={true}/>

      {showDetailedViz && stepData ? (
        <DetailedStepVisualization stepData={stepData} visualizationMode={visualizationMode} epsilon={epsilon} />
      ) : (
        <BoundaryEvolution boundaryHistory={boundaryHistory} epsilon={epsilon} />
      )}

      <OrbitControls 
        enableRotate={false} 
        enablePan={true} 
        enableZoom={true}
        mouseButtons={{
          LEFT: THREE.MOUSE.PAN,
          MIDDLE: THREE.MOUSE.DOLLY,
          RIGHT: THREE.MOUSE.PAN
        }}
      />
    </>
  );
}

export default function Algorithm1Viz({
  boundaryHistory = [],
  currentIteration = 0,
  epsilon = 0.1,
  nBoundaryPoints = 8,
  isConverged = false,
  stepData = null,
  visualizationMode = 'all',
  isDiverged = false,
  showDetailedViz = false
}) {
  return (
    <div style={{ width: '100%', height: '100%', minHeight: '500px', position: 'relative' }}>
      <Canvas
        orthographic
        camera={{ position: [0, 0, 5], zoom: 100 }}
        gl={{ preserveDrawingBuffer: true, antialias: true, alpha: false, powerPreference: "high-performance" }}
      >
        <Scene
          boundaryHistory={boundaryHistory}
          stepData={stepData}
          visualizationMode={visualizationMode}
          epsilon={epsilon}
          showDetailedViz={showDetailedViz}
        />
      </Canvas>

      {/* Divergence overlay - shows on top of everything */}
      {isDiverged && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          background: 'rgba(211, 47, 47, 0.95)',
          color: 'white',
          padding: '24px 32px',
          borderRadius: '12px',
          border: '3px solid #ff5252',
          fontFamily: 'monospace',
          fontSize: '16px',
          textAlign: 'center',
          boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
          zIndex: 1000,
          maxWidth: '400px'
        }}>
          <div style={{ fontSize: '32px', marginBottom: '12px' }}>⚠️</div>
          <div style={{ fontWeight: 'bold', fontSize: '18px', marginBottom: '12px' }}>
            SYSTEM DIVERGED
          </div>
          {
            <div style={{ fontSize: '13px', opacity: 0.9, marginBottom: '16px', lineHeight: '1.4' }}>
              {"System diverged: all boundary points moved outside the defined domain."}
            </div>
          }
          <div style={{ fontSize: '12px', opacity: 0.8, borderTop: '1px solid rgba(255,255,255,0.3)', paddingTop: '12px' }}>
            Try: smaller ε, different initial conditions, or fewer iterations
          </div>
        </div>
      )}



      {/* Info overlay */}
      <div style={{
        position: 'absolute', top: '10px', left: '10px', color: 'white',
        background: 'rgba(0,0,0,0.7)', padding: '12px', borderRadius: '6px',
        fontFamily: 'monospace', fontSize: '14px', pointerEvents: 'none'
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>Algorithm 1: Boundary Evolution</div>
        <div>Iteration: {currentIteration}</div>
        <div>Boundary Points: {nBoundaryPoints}</div>
        <div>ε = {epsilon.toFixed(3)}</div>
        {isConverged && <div style={{ color: '#00ff00', marginTop: '4px' }}>✓ CONVERGED</div>}
      </div>

      {/* Legend */}
      {showDetailedViz && stepData && (
        <div style={{
          position: 'absolute', bottom: '10px', left: '10px', color: 'white',
          background: 'rgba(0,0,0,0.7)', padding: '10px', borderRadius: '6px',
          fontFamily: 'monospace', fontSize: '12px', pointerEvents: 'none'
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '6px' }}>Legend:</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <div style={{ width: '12px', height: '12px', background: '#00ffff', borderRadius: '50%' }}></div>
            <span>Mapped points f(z<sub>k</sub>)</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <div style={{ width: '12px', height: '12px', background: '#ffaa00', border: '1px solid #ffaa00' }}></div>
            <span>Noise circles B<sub>ε</sub>(f(z<sub>k</sub>))</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: '12px', height: '12px', background: '#ff00ff', borderRadius: '50%' }}></div>
            <span>Projected boundary</span>
          </div>
        </div>
      )}

      {/* Controls */}
      <div style={{
        position: 'absolute', top: '10px', right: '10px', color: 'white',
        background: 'rgba(0,0,0,0.7)', padding: '10px', borderRadius: '6px',
        fontFamily: 'monospace', fontSize: '11px', pointerEvents: 'none', maxWidth: '200px'
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>Controls:</div>
        <div>• Scroll: Zoom</div>
      </div>
    </div>
  );
}