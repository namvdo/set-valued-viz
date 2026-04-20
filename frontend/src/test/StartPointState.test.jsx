import { describe, expect, it } from 'vitest';
import { applyStartPointUpdate } from '../utils/startPointState';

describe('applyStartPointUpdate', () => {
  it('resets trajectory and uses new start point', () => {
    const prev = {
      startPoint: { x: 0, y: 0, nx: 1, ny: 0 },
      currentPoint: { x: 1, y: 1, nx: 1, ny: 0 },
      trajectoryPoints: [{ x: 1, y: 1, nx: 1, ny: 0 }],
      iteration: 5,
      hasStarted: true,
      isRunning: true
    };
    const next = applyStartPointUpdate(prev, { x: 0.1, y: 0.2, nx: 1, ny: 0 });

    expect(next.startPoint).toEqual({ x: 0.1, y: 0.2, nx: 1, ny: 0 });
    expect(next.currentPoint).toEqual({ x: 0.1, y: 0.2, nx: 1, ny: 0 });
    expect(next.trajectoryPoints).toEqual([]);
    expect(next.iteration).toBe(0);
    expect(next.hasStarted).toBe(false);
    expect(next.isRunning).toBe(false);
  });
});
