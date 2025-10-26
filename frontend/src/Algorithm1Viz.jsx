import React, { useRef, useEffect, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, Text } from '@react-three/drei';
import * as THREE from 'three';

function DetailedStepVisualization({ stepData, visualizationMode, epsilon }) {
  if (!stepData) return null;

  const { mapped_points, projected_points, normals } = stepData;

  // Convert mapped points to 3D
  const mappedPoints = useMemo(() => {
    if (!mapped_points || mapped_points.length === 0) return [];
    const points = [];
    for (let i = 0; i < mapped_points.length; i += 2) {
      points.push(new THREE.Vector3(mapped_points[i], mapped_points[i + 1], 0));
    }
    return points;
  }, [mapped_points]);

  // Convert projected points to 3D
  const projectedPointsArray = useMemo(() => {
    if (!projected_points || projected_points.length === 0) return [];
    const points = [];
    for (let i = 0; i < projected_points.length; i += 2) {
      points.push(new THREE.Vector3(projected_points[i], projected_points[i + 1], 0));
    }
    return points;
  }, [projected_points]);

  // Create projected boundary line (close the loop)
  const projectedBoundaryLine = useMemo(() => {
    if (projectedPointsArray.length === 0) return [];
    return [...projectedPointsArray, projectedPointsArray[0]];
  }, [projectedPointsArray]);

  const showMapped = visualizationMode === 'mapped' || visualizationMode === 'all';
  const showCircles = visualizationMode === 'circles' || visualizationMode === 'all';
  const showProjected = visualizationMode === 'projected' || visualizationMode === 'all';
  const showNormals = visualizationMode === 'all' && normals && normals.length > 0;

  return (
    <>
      {/* Step 1: Show mapped points f(zk) */}
      {showMapped && mappedPoints.map((point, idx) => (
        <mesh key={`mapped-${idx}`} position={point}>
          <sphereGeometry args={[0.02, 16, 16]} />
          <meshBasicMaterial color="#00ffff" />
        </mesh>
      ))}

      {/* Step 2: Show noise circles around each mapped point */}
      {showCircles && mappedPoints.map((point, idx) => (
        <group key={`circle-${idx}`}>
          {/* Filled circle */}
          <mesh position={point} rotation={[0, 0, 0]}>
            <circleGeometry args={[epsilon, 64]} />
            <meshBasicMaterial
              color="#ffaa00"
              transparent
              opacity={0.15}
              side={THREE.DoubleSide}
            />
          </mesh>
          {/* Circle outline */}
          <mesh position={point} rotation={[0, 0, 0]}>
            <ringGeometry args={[epsilon * 0.98, epsilon, 64]} />
            <meshBasicMaterial
              color="#ffaa00"
              transparent
              opacity={0.6}
              side={THREE.DoubleSide}
            />
          </mesh>
        </group>
      ))}

      {/* Step 3: Show the union/envelope (projected boundary) */}
      {showProjected && projectedBoundaryLine.length > 0 && (
        <>
          {/* Boundary line */}
          <Line
            points={projectedBoundaryLine}
            color="#ff00ff"
            lineWidth={3}
          />
          {/* Boundary points */}
          {projectedPointsArray.map((point, idx) => (
            <mesh key={`projected-${idx}`} position={point}>
              <sphereGeometry args={[0.025, 16, 16]} />
              <meshBasicMaterial color="#ff00ff" />
            </mesh>
          ))}
        </>
      )}

      {/* Optional: Show normal vectors */}
      {showNormals && mappedPoints.map((point, idx) => {
        if (idx * 2 + 1 >= normals.length) return null;
        const nx = normals[idx * 2];
        const ny = normals[idx * 2 + 1];
        const endPoint = new THREE.Vector3(
          point.x + nx * epsilon * 1.5,
          point.y + ny * epsilon * 1.5,
          0
        );
        return (
          <Line
            key={`normal-${idx}`}
            points={[point, endPoint]}
            color="#00ff00"
            lineWidth={1}
          />
        );
      })}
    </>
  );
}

function BoundaryEvolution({ boundaryHistory, currentIteration, epsilon }) {
  const meshRefs = useRef([]);

  // Create geometry for each iteration
  const boundaryGeometries = useMemo(() => {
    if (!boundaryHistory || boundaryHistory.length === 0) return [];

    return boundaryHistory.map((boundary, iter) => {
      if (!boundary || boundary.length === 0) return null;

      // Convert flat array to 3D points (z = 0 for 2D visualization)
      const points = [];
      for (let i = 0; i < boundary.length; i += 2) {
        points.push(new THREE.Vector3(boundary[i], boundary[i + 1], 0));
      }
      // Close the loop
      if (points.length > 0) {
        points.push(points[0].clone());
      }

      // Color gradient: blue → cyan → green → yellow → red
      const maxIterations = boundaryHistory.length;
      const intensity = iter / Math.max(1, maxIterations - 1);
      const r = Math.pow(intensity, 0.7);
      const g = 1 - Math.abs(2 * intensity - 1);
      const b = Math.pow(1 - intensity, 0.7);
      const alpha = 0.2 + 0.6 * intensity;

      return {
        points,
        color: new THREE.Color(r, g, b),
        alpha,
        isFinal: iter === boundaryHistory.length - 1
      };
    }).filter(Boolean);
  }, [boundaryHistory]);

  return (
    <>
      {boundaryGeometries.map((geo, idx) => (
        <group key={idx}>
          {/* Boundary line */}
          <Line
            points={geo.points}
            color={geo.color}
            lineWidth={geo.isFinal ? 3 : 1.5}
            transparent
            opacity={geo.alpha}
          />

          {/* Boundary points */}
          {geo.points.slice(0, -1).map((point, pidx) => (
            <mesh key={pidx} position={point}>
              <sphereGeometry args={[geo.isFinal ? 0.015 : 0.008, 8, 8]} />
              <meshBasicMaterial
                color={geo.isFinal ? '#ffffff' : geo.color}
                transparent
                opacity={geo.alpha}
              />
            </mesh>
          ))}
        </group>
      ))}
    </>
  );
}

function Scene({ boundaryHistory, currentIteration, stepData, visualizationMode, epsilon, showDetailedViz }) {
  // Calculate bounds for camera positioning
  const bounds = useMemo(() => {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

    if (showDetailedViz && stepData) {
      // Use stepData for bounds
      const { mapped_points, projected_points } = stepData;
      const allPoints = [...(mapped_points || []), ...(projected_points || [])];

      for (let i = 0; i < allPoints.length; i += 2) {
        const x = allPoints[i];
        const y = allPoints[i + 1];
        if (isFinite(x) && isFinite(y)) {
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
        }
      }

      // Add epsilon padding for circles
      minX -= epsilon * 1.5;
      maxX += epsilon * 1.5;
      minY -= epsilon * 1.5;
      maxY += epsilon * 1.5;
    } else if (boundaryHistory && boundaryHistory.length > 0) {
      // Use boundary history for bounds
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
    }

    if (!isFinite(minX)) {
      return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    }

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

  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerY = (bounds.minY + bounds.maxY) / 2;

  return (
    <>
      <color attach="background" args={['#000000']} />
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />

      {/* Main visualization */}
      <group position={[-centerX, -centerY, 0]}>
        {showDetailedViz && stepData ? (
          <DetailedStepVisualization
            stepData={stepData}
            visualizationMode={visualizationMode}
            epsilon={epsilon}
          />
        ) : (
          <BoundaryEvolution
            boundaryHistory={boundaryHistory}
            currentIteration={currentIteration}
            epsilon={epsilon}
          />
        )}
      </group>

      <OrbitControls
        enableRotate={true}
        enablePan={true}
        enableZoom={true}
        target={[0, 0, 0]}
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
  showDetailedViz = false
}) {
  return (
    <div style={{ width: '100%', height: '100%', minHeight: '500px', position: 'relative' }}>
      <Canvas
        camera={{ position: [0, 0, 5], fov: 50 }}
        style={{ background: '#000' }}
      >
        <Scene
          boundaryHistory={boundaryHistory}
          currentIteration={currentIteration}
          stepData={stepData}
          visualizationMode={visualizationMode}
          epsilon={epsilon}
          showDetailedViz={showDetailedViz}
        />
      </Canvas>

      {/* Info overlay */}
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        color: 'white',
        fontFamily: 'Arial',
        fontSize: '14px',
        pointerEvents: 'none',
        background: 'rgba(0,0,0,0.7)',
        padding: '12px',
        borderRadius: '6px'
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
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          color: 'white',
          fontFamily: 'Arial',
          fontSize: '12px',
          pointerEvents: 'none',
          background: 'rgba(0,0,0,0.7)',
          padding: '10px',
          borderRadius: '6px'
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
            <span>Projected boundary (envelope)</span>
          </div>
        </div>
      )}

      {/* Instructions */}
      {showDetailedViz && stepData && (
        <div style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          color: 'white',
          fontFamily: 'Arial',
          fontSize: '11px',
          pointerEvents: 'none',
          background: 'rgba(0,0,0,0.7)',
          padding: '10px',
          borderRadius: '6px',
          maxWidth: '200px'
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>Controls:</div>
          <div>• Left-click + drag: Rotate</div>
          <div>• Right-click + drag: Pan</div>
          <div>• Scroll: Zoom</div>
          <div style={{ marginTop: '8px', fontSize: '10px', opacity: 0.8 }}>
            Use "Step Forward (Detailed)" to iterate
          </div>
        </div>
      )}
    </div>
  );
}
