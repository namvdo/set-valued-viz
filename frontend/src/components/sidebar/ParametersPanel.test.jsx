import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
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
  paramErrors: [],
  hasPendingInputChanges: false,
  applyInputsAndRecompute: vi.fn(),
};

describe('ParametersPanel apply action', () => {
  it('uses [-10, 10] bounds for a and b controls', () => {
    render(<ParametersPanel {...baseProps} />);
    const spinboxes = screen.getAllByRole('spinbutton');

    expect(spinboxes[0]).toHaveAttribute('min', '-10');
    expect(spinboxes[0]).toHaveAttribute('max', '10');
    expect(spinboxes[1]).toHaveAttribute('min', '-10');
    expect(spinboxes[1]).toHaveAttribute('max', '10');
  });

  it('shows Apply & Recompute label when there are pending changes', () => {
    render(<ParametersPanel {...baseProps} hasPendingInputChanges={true} />);
    expect(screen.getByRole('button', { name: 'Apply & Recompute' })).toBeInTheDocument();
  });

  it('shows Recompute label when there are no pending changes', () => {
    render(<ParametersPanel {...baseProps} hasPendingInputChanges={false} />);
    expect(screen.getByRole('button', { name: 'Recompute' })).toBeInTheDocument();
  });

  it('triggers apply callback on click', () => {
    const onApply = vi.fn();
    render(<ParametersPanel {...baseProps} hasPendingInputChanges={true} applyInputsAndRecompute={onApply} />);

    fireEvent.click(screen.getByRole('button', { name: 'Apply & Recompute' }));
    expect(onApply).toHaveBeenCalledTimes(1);
  });
});
