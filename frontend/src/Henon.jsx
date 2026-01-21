import React, { useMemo } from "react"
import { Canvas } from "@react-three/fiber"
import * as THREE from "three"

function HenonGrid({
  a = 1.4,
  b = 0.3,
  iterations = 500000,
  gridSize = 400,
  extent = 1.5
}) {
  const texture = useMemo(() => {
    const counts = new Float32Array(gridSize * gridSize)
    let x = 0
    let y = 0

    for (let i = 0; i < iterations; i++) {
      const xn = 1 - a * x * x + y
      const yn = b * x
      x = xn
      y = yn

      const gx = Math.floor(((x + extent) / (2 * extent)) * gridSize)
      const gy = Math.floor(((y + extent) / (2 * extent)) * gridSize)

      if (gx >= 0 && gx < gridSize && gy >= 0 && gy < gridSize) {
        counts[gy * gridSize + gx]++
      }
    }

    let max = 0
    for (let i = 0; i < counts.length; i++) {
      if (counts[i] > max) max = counts[i]
    }

    const data = new Uint8Array(gridSize * gridSize * 4)
    for (let i = 0; i < counts.length; i++) {
      const v = Math.log1p(counts[i]) / Math.log1p(max)
      const c = Math.floor(v * 255)
      const j = i * 4
      data[j] = c
      data[j + 1] = c
      data[j + 2] = c
      data[j + 3] = 255
    }

    const tex = new THREE.DataTexture(
      data,
      gridSize,
      gridSize,
      THREE.RGBAFormat,
      THREE.UnsignedByteType
    )

    tex.needsUpdate = true
    tex.minFilter = THREE.NearestFilter
    tex.magFilter = THREE.NearestFilter
    tex.generateMipmaps = false

    return tex
  }, [a, b, iterations, gridSize, extent])

  return (
    <mesh>
      <planeGeometry args={[2, 2]} />
      <meshBasicMaterial map={texture} />
    </mesh>
  )
}

function Axes() {
  const helper = useMemo(() => {
    const h = new THREE.AxesHelper(5.1)
    h.position.set(1, 1, 1)
    return h
  }, [])

  return <primitive object={helper} />
}

export default function App() {
  return (
    <Canvas
      orthographic
      camera={{ zoom: 160, position: [0, 0, 10] }}
    >
      <color attach="background" args={["white"]} />
      <Axes />
      <HenonGrid />
    </Canvas>
  )
}
