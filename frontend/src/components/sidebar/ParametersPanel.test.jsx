import React from 'react';
import { render, screen } from '@testing-library/react';
import { vi } from 'vitest';
import { ParametersPanel } from './ParametersPanel';

const baseProps = {
  systemId: 'henon',
  params: {
    a: 1.4,
    b: 0.3,
    epsilon: 0.01,
    maxPeriod: 5,
    maxIterations: 1000,
  },
  setParams: vi.fn(),
  disabled: false,
  systems: {
    discrete: [{ id: 'henon', presets: [] }],
    continuous: []
  },
  applyPreset: vi.fn(),
  customParams: [],
  setCustomParams: vi.fn(),
  paramErrors: []
};

describe('ParametersPanel', () => {
  it('uses [-10, 10] bounds for a and b controls', () => {
    render(<ParametersPanel {...baseProps} />);
    const spinboxes = screen.getAllByRole('spinbutton');

    expect(spinboxes[0]).toHaveAttribute('min', '-10');
    expect(spinboxes[0]).toHaveAttribute('max', '10');
    expect(spinboxes[1]).toHaveAttribute('min', '-10');
    expect(spinboxes[1]).toHaveAttribute('max', '10');
  });

  it('does not render recompute button (moved to shared sidebar action)', () => {
    render(<ParametersPanel {...baseProps} />);
    expect(screen.queryByRole('button', { name: 'Apply & Recompute' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Recompute' })).toBeNull();
  });
});
