// src/components/HenonMapVisualization.jsx
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { CoordinateTransform } from './coordinateTransform';
import CoordinateSystem from './Grid';

const HenonMapVisualization = ({ 
  simulationData, 
  parameters 
}) => {
  const mathBounds = {
    xRange: [-2, 2],
    yRange: [-0.5, 0.5]
  };

  const viewportSize = {
    width: 8,
    height: 6
  };

  const coordTransform = new CoordinateTransform(mathBounds, viewportSize);

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <Canvas>
        <PerspectiveCamera
          makeDefault
          position={[0, 0, 10]}
          fov={50}
        />

        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />

        <CoordinateSystem
          xRange={mathBounds.xRange}
          yRange={mathBounds.yRange}
          gridDivisions={10}
        />

        <OrbitControls
          enableRotate={false}
          enablePan={true}
          enableZoom={true}
          minZoom={0.5}
          maxZoom={5}
        />
      </Canvas>
    </div>
  );
};

export default HenonMapVisualization;