// src/visualization/CoordinateSystem.jsx
import { useMemo } from 'react';
import { Line, Text } from '@react-three/drei';
import * as THREE from 'three';

const CoordinateSystem = ({ 
  xRange = [-2, 2], 
  yRange = [-0.5, 0.5],
  gridDivisions = 10,
  axisColor = "#666666",
  gridColor = "#333333",
  labelColor = "#ffffff",
  fontSize = 0.08
}) => {
  const [xMin, xMax] = xRange;
  const [yMin, yMax] = yRange;

  const gridLines = useMemo(() => {
    const lines = [];
    const xStep = (xMax - xMin) / gridDivisions;
    const yStep = (yMax - yMin) / gridDivisions;

    for (let i = 0; i <= gridDivisions; i++) {
      const x = xMin + i * xStep;
      lines.push({
        points: [[x, yMin, 0], [x, yMax, 0]],
        isAxis: Math.abs(x) < 0.01
      });
    }

    for (let i = 0; i <= gridDivisions; i++) {
      const y = yMin + i * yStep;
      lines.push({
        points: [[xMin, y, 0], [xMax, y, 0]],
        isAxis: Math.abs(y) < 0.01
      });
    }

    return lines;
  }, [xMin, xMax, yMin, yMax, gridDivisions]);

  const labels = useMemo(() => {
    const xStep = (xMax - xMin) / gridDivisions;
    const yStep = (yMax - yMin) / gridDivisions;
    const labelList = [];

    for (let i = 0; i <= gridDivisions; i++) {
      const x = xMin + i * xStep;
      if (Math.abs(x) > 0.01) {
        labelList.push({
          position: [x, yMin - fontSize * 2, 0],
          text: x.toFixed(1)
        });
      }
    }
    for (let i = 0; i <= gridDivisions; i++) {
      const y = yMin + i * yStep;
      if (Math.abs(y) > 0.01) { 
        labelList.push({
          position: [xMin - fontSize * 2, y, 0],
          text: y.toFixed(2)
        });
      }
    }

    return labelList;
  }, [xMin, xMax, yMin, yMax, gridDivisions, fontSize]);

  return (
    <group name="coordinate-system">
      {gridLines.map((line, idx) => (
        <Line
          key={`grid-${idx}`}
          points={line.points}
          color={line.isAxis ? axisColor : gridColor}
          lineWidth={line.isAxis ? 2 : 1}
          transparent
          opacity={line.isAxis ? 1 : 0.3}
        />
      ))}

      {labels.map((label, idx) => (
        <Text
          key={`label-${idx}`}
          position={label.position}
          fontSize={fontSize}
          color={labelColor}
          anchorX="center"
          anchorY="middle"
        >
          {label.text}
        </Text>
      ))}

      <Text
        position={[xMax + fontSize * 3, 0, 0]}
        fontSize={fontSize * 1.5}
        color={labelColor}
        anchorX="left"
      >
        x
      </Text>
      <Text
        position={[0, yMax + fontSize * 3, 0]}
        fontSize={fontSize * 1.5}
        color={labelColor}
        anchorY="bottom"
      >
        y
      </Text>
    </group>
  );
};

export default CoordinateSystem;