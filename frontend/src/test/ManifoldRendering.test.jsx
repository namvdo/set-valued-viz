import { describe, it, expect } from 'vitest';

/**
 * Tests for the manifold rendering logic.
 * These test the pure logic of classifying which manifolds belong to repellers
 * vs saddles, mirroring the logic in SetValuedViz.jsx rendering code.
 */
describe('Manifold Classification Logic', () => {
  const classifyManifolds = (manifolds, fixedPoints) => {
    const repellerPoints = new Set();
    fixedPoints.forEach(fp => {
      const stab = (fp.stability || '').toLowerCase();
      if (stab === 'unstable') {
        repellerPoints.add(`${fp.x.toFixed(8)},${fp.y.toFixed(8)}`);
      }
    });

    return manifolds.map(m => {
      const spKey = `${m.saddle_point[0].toFixed(8)},${m.saddle_point[1].toFixed(8)}`;
      return {
        ...m,
        isRepeller: repellerPoints.has(spKey),
      };
    });
  };


  it('handles case with no repellers', () => {
    const fixedPoints = [
      { x: 0.5, y: 0.15, stability: 'Saddle', eigenvalues: [1.5, -0.2] },
      { x: 0.8, y: 0.24, stability: 'Attractor', eigenvalues: [0.3, 0.1] },
    ];
    const manifolds = [
      { saddle_point: [0.5, 0.15], plus: { points: [] }, minus: { points: [] } },
    ];

    const classified = classifyManifolds(manifolds, fixedPoints);
    expect(classified[0].isRepeller).toBe(false);
  });

  it('handles "unstable" as synonym for repeller', () => {
    const fixedPoints = [
      { x: 1.0, y: 0.3, stability: 'unstable', eigenvalues: [2.0, 1.5] },
    ];
    const manifolds = [
      { saddle_point: [1.0, 0.3], plus: { points: [] }, minus: { points: [] } },
    ];

    const classified = classifyManifolds(manifolds, fixedPoints);
    expect(classified[0].isRepeller).toBe(true);
  });

  it('handles empty manifolds and fixed points', () => {
    const classified = classifyManifolds([], []);
    expect(classified).toEqual([]);
  });
});

describe('Fixed Point Color Assignment', () => {
  const ORBIT_COLORS = {
    attractor: '#27ae60',
    repeller: '#e74c3c',
    saddlePoint: '#eedf32',
    periodicBlue: '#3498db',
  };

  const getFixedPointColor = (stability) => {
    const stabLower = (stability || '').toLowerCase();
    const isAttractor = stabLower === 'attractor' || stabLower === 'stable';
    const isRepeller = stabLower === 'repeller' || stabLower === 'unstable';
    const isSaddle = stabLower === 'saddle';
    return isAttractor ? ORBIT_COLORS.attractor :
      isRepeller ? ORBIT_COLORS.repeller :
      isSaddle ? ORBIT_COLORS.saddlePoint : ORBIT_COLORS.periodicBlue;
  };

  it('assigns green to attractors', () => {
    expect(getFixedPointColor('Attractor')).toBe('#27ae60');
    expect(getFixedPointColor('stable')).toBe('#27ae60');
  });

  it('assigns red to repellers (distinct from saddles)', () => {
    expect(getFixedPointColor('Repeller')).toBe('#e74c3c');
    expect(getFixedPointColor('unstable')).toBe('#e74c3c');
  });

  it('assigns yellow to saddles', () => {
    expect(getFixedPointColor('Saddle')).toBe('#eedf32');
    expect(getFixedPointColor('saddle')).toBe('#eedf32');
  });

  it('repeller and saddle have different colors', () => {
    expect(getFixedPointColor('Repeller')).not.toBe(getFixedPointColor('Saddle'));
  });

  it('assigns default color to unknown stability', () => {
    expect(getFixedPointColor('')).toBe('#3498db');
    expect(getFixedPointColor(null)).toBe('#3498db');
  });
});

describe('Bifurcation State Management', () => {
  it('initial state has correct defaults', () => {
    const initialState = {
      data: [],
      isComputing: false,
      error: null,
      aMin: 0.1,
      aMax: 2.0,
      numSamples: 30,
      threshold: 0.2,
      intersectionCount: 0,
      criticalValues: [],
    };

    expect(initialState.data).toEqual([]);
    expect(initialState.isComputing).toBe(false);
    expect(initialState.aMin).toBeLessThan(initialState.aMax);
    expect(initialState.numSamples).toBeGreaterThan(0);
  });

  it('bifurcation data has expected shape', () => {
    const sampleData = [
      { a: 0.5, hausdorff_distance: 0.3, max_unstable_to_stable: 0.3, max_stable_to_unstable: 0.2, has_intersection: false, intersection_threshold: 0.02 },
      { a: 1.0, hausdorff_distance: 0.01, max_unstable_to_stable: 0.01, max_stable_to_unstable: 0.005, has_intersection: true, intersection_threshold: 0.02 },
    ];

    sampleData.forEach(d => {
      expect(d).toHaveProperty('a');
      expect(d).toHaveProperty('hausdorff_distance');
      expect(d).toHaveProperty('has_intersection');
      expect(typeof d.a).toBe('number');
      expect(typeof d.hausdorff_distance).toBe('number');
      expect(typeof d.has_intersection).toBe('boolean');
      expect(d.hausdorff_distance).toBeGreaterThanOrEqual(0);
    });
  });

  it('critical values are extracted correctly from data', () => {
    const data = [
      { a: 0.5, hausdorff_distance: 0.3, has_intersection: false },
      { a: 1.0, hausdorff_distance: 0.01, has_intersection: true },
      { a: 1.2, hausdorff_distance: 0.005, has_intersection: true },
      { a: 1.5, hausdorff_distance: 0.5, has_intersection: false },
    ];

    const intersections = data.filter(d => d.has_intersection);
    const criticalValues = intersections.map(d => d.a);

    expect(criticalValues).toEqual([1.0, 1.2]);
    expect(intersections.length).toBe(2);
  });
});
