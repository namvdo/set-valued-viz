import React, { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, Text } from '@react-three/drei';
import * as THREE from 'three';

function TrajectoryLine({ points, color = null }) {
  const lineRef = useRef();

  const positions = useMemo(() => {
    return points.map(p => new THREE.Vector3(p.x, p.y, p.z));
  }, [points]);

  const colors = useMemo(() => {
    if (color) {
      return positions.map(() => color);
    }
    
    return positions.map((_, i) => {
      const t = i / (positions.length - 1); 
      const hue = (1 - t) * 0.67; 
      return new THREE.Color().setHSL(hue, 1.0, 0.5);
    });
  }, [positions, color]);

  return (
    <Line
      ref={lineRef}
      points={positions}
      color="white"
      vertexColors={colors}
      lineWidth={2}
    />
  );
}

function FixedPoints({ fixedPoints }) {
  if (!fixedPoints || fixedPoints.length === 0) return null;

  return (
    <>
      {fixedPoints.map((point, i) => (
        <mesh key={i} position={[point.x, point.y, point.z]}>
          <sphereGeometry args={[0.5, 16, 16]} />
          <meshStandardMaterial 
            color={i === 0 ? 'yellow' : (i === 1 ? 'lime' : 'magenta')}
            emissive={i === 0 ? 'yellow' : (i === 1 ? 'lime' : 'magenta')}
            emissiveIntensity={0.5}
          />
        </mesh>
      ))}
    </>
  );
}

function Axes({ size = 50 }) {
  return (
    <group>
      <Line
        points={[[-size, 0, 0], [size, 0, 0]]}
        color="red"
        lineWidth={1}
      />
      <Text
        position={[size + 2, 0, 0]}
        fontSize={2}
        color="red"
      >
        X
      </Text>

      <Line
        points={[[0, -size, 0], [0, size, 0]]}
        color="green"
        lineWidth={1}
      />
      <Text
        position={[0, size + 2, 0]}
        fontSize={2}
        color="green"
      >
        Y
      </Text>

      <Line
        points={[[0, 0, -size], [0, 0, size]]}
        color="blue"
        lineWidth={1}
      />
      <Text
        position={[0, 0, size + 2]}
        fontSize={2}
        color="blue"
      >
        Z
      </Text>
    </group>
  );
}

function AnimatedTrajectory({ points, speed = 1 }) {
  const [currentIndex, setCurrentIndex] = React.useState(0);

  useFrame(() => {
    if (currentIndex < points.length - 1) {
      setCurrentIndex(prev => Math.min(prev + speed, points.length - 1));
    }
  });

  const visiblePoints = useMemo(() => {
    return points.slice(0, Math.floor(currentIndex) + 1);
  }, [points, currentIndex]);

  if (visiblePoints.length < 2) return null;

  return <TrajectoryLine points={visiblePoints} />;
}

export default function LorenzVisualizer({ 
  trajectories = [],
  fixedPoints = [],
  showAxes = true,
  showFixedPoints = true,
  animated = false,
  animationSpeed = 1,
}) {
  return (
    <div style={{ width: '80vw', height: '100vh', background: '#000' }}>
      <Canvas
        camera={{ position: [50, 50, 50], fov: 60 }}
        gl={{ antialias: true }}
      >
        <ambientLight intensity={0.5} />
        <pointLight position={[50, 50, 50]} intensity={1} />
        <pointLight position={[-50, -50, -50]} intensity={0.5} />

        {trajectories.map((trajectory, i) => (
          animated ? (
            <AnimatedTrajectory
              key={i}
              points={trajectory}
              speed={animationSpeed}
            />
          ) : (
            <TrajectoryLine key={i} points={trajectory} />
          )
        ))}

        {showFixedPoints && <FixedPoints fixedPoints={fixedPoints} />}

        {showAxes && <Axes />}

        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          rotateSpeed={0.5}
          zoomSpeed={0.5}
        />
      </Canvas>
    </div>
  );
}