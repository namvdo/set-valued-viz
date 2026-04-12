import { describe, expect, it } from 'vitest';
import { normalizeViewRange, RANGE_LIMIT } from './viewRange';

describe('normalizeViewRange', () => {
  it('orders and clamps values', () => {
    const result = normalizeViewRange({ xMin: 12, xMax: -5, yMin: -20, yMax: 3 });
    expect(result.xMin).toBe(-5);
    expect(result.xMax).toBe(RANGE_LIMIT);
    expect(result.yMin).toBe(-RANGE_LIMIT);
    expect(result.yMax).toBe(3);
  });

  it('expands zero-width ranges', () => {
    const result = normalizeViewRange({ xMin: 1, xMax: 1, yMin: 2, yMax: 2 });
    expect(result.xMax).toBeGreaterThan(result.xMin);
    expect(result.yMax).toBeGreaterThan(result.yMin);
  });
});
